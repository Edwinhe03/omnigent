"""Client-side helpers for launching runners through the connect daemon.

These are the CLI-side counterpart to the host-runner protocol: the CLI
(``run`` / ``claude`` / ``codex``) asks the Omnigent server to launch
a runner on this machine's daemon via
``POST /v1/hosts/{host_id}/runners``; the server forwards a launch frame to
the daemon, which spawns the runner subprocess and binds it to the session.
The daemon owns the runner lifecycle — the CLI only connects.

Harness-agnostic on purpose: the same launch path serves headless
``run`` agents and the ``claude``/``codex`` terminal wrappers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import click
import httpx

from omnigent.harnesses.claude_native.bridge import url_component
from omnigent.process_logging import display_log_path, process_log_dir

# Steady-state poll cadence while waiting for a daemon-spawned runner to
# connect its tunnel or for a resource to appear.
DAEMON_POLL_INTERVAL_S = 0.5

# Readiness usually lands on the first probe or two, so open tight and ease
# off to the steady cadence for the long tail.
DAEMON_POLL_INITIAL_INTERVAL_S = 0.1
DAEMON_POLL_BACKOFF_FACTOR = 1.5


class AtomicHostCreateFallback(Exception):
    """Signal that atomic create was rejected before a session was created."""


@dataclass(frozen=True)
class DaemonSessionCreateResult:
    """Result of a bundled session create used by native daemon launches.

    :param session_id: Newly-created Omnigent session id.
    :param runner_id: Runner bound by inline host launch, if reported.
    :param runner_launch_status: ``"launched"``, ``"failed"``, or
        ``"indeterminate"`` for an atomic create; ``None`` for a legacy
        hostless create.
    :param runner_launch_error: Inline launch failure, if any.
    """

    session_id: str
    runner_id: str | None = None
    runner_launch_status: Literal["launched", "failed", "indeterminate"] | None = None
    runner_launch_error: str | None = None


def daemon_session_id(result: str | DaemonSessionCreateResult) -> str:
    """Return the session id from legacy or atomic bundled-create results."""
    return result if isinstance(result, str) else result.session_id


def response_error_code(resp: httpx.Response) -> str | None:
    """Extract ``error.code`` from an Omnigent error response."""
    error = _json_body(resp).get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


def atomic_host_create_fallback_reason(resp: httpx.Response) -> str | None:
    """Return why an atomic create can safely retry as a legacy create.

    ``host_launch_contract=result_v1`` is schema-validated before bundle
    persistence. An older server therefore rejects the unknown field before
    creating a session. On a supporting server, external-host workspace
    validation also runs before persistence, so its explicit wrong-replica and
    host-offline errors are safe to retry. Other conflicts stay loud because
    they do not prove whether a session row already exists.

    :param resp: Failed multipart session-create response.
    :returns: Fallback reason, or ``None`` when retrying could duplicate a
        session and the original error must remain loud.
    """
    message = error_text(resp)
    lower_message = message.lower()
    if (
        resp.status_code in {400, 422}
        and response_error_code(resp) == "invalid_input"
        and "invalid session metadata:" in lower_message
        and "host_launch_contract" in lower_message
        and (
            "extra inputs are not permitted" in lower_message or "extra_forbidden" in lower_message
        )
    ):
        return "server does not support atomic host-launch results"
    code = response_error_code(resp)
    if (
        resp.status_code == 400
        and code == "wrong_replica"
        and lower_message.startswith("host ")
        and " is on another replica" in lower_message
    ):
        return message
    if (
        resp.status_code == 400
        and code == "invalid_input"
        and lower_message.startswith("host ")
        and " is offline" in lower_message
    ):
        return message
    return None


def daemon_session_create_result(
    resp: httpx.Response,
    *,
    harness_name: str,
    atomic: bool,
) -> DaemonSessionCreateResult:
    """Validate and decode a native bundled session-create response.

    :param resp: Successful multipart ``POST /v1/sessions`` response.
    :param harness_name: Human-readable harness name for errors.
    :param atomic: Whether the request required ``result_v1`` launch metadata.
    :returns: Parsed create result.
    :raises click.ClickException: On a malformed response.
    """
    body = _json_body(resp)
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise click.ClickException(
            f"{harness_name} session creation response did not include session_id."
        )
    if not atomic:
        return DaemonSessionCreateResult(session_id=session_id)

    runner_id = body.get("runner_id")
    runner_id = runner_id if isinstance(runner_id, str) and runner_id else None
    launch_status = body.get("runner_launch_status")
    launch_error = body.get("runner_launch_error")
    launch_error = launch_error if isinstance(launch_error, str) and launch_error else None
    if launch_status == "launched" and runner_id is not None:
        return DaemonSessionCreateResult(
            session_id=session_id,
            runner_id=runner_id,
            runner_launch_status="launched",
        )
    if launch_status == "failed":
        return DaemonSessionCreateResult(
            session_id=session_id,
            runner_id=runner_id,
            runner_launch_status="failed",
            runner_launch_error=launch_error,
        )
    if launch_status == "indeterminate":
        return DaemonSessionCreateResult(
            session_id=session_id,
            runner_id=runner_id,
            runner_launch_status="indeterminate",
            runner_launch_error=launch_error,
        )
    return DaemonSessionCreateResult(
        session_id=session_id,
        runner_id=runner_id,
        runner_launch_status="indeterminate",
        runner_launch_error=(
            launch_error
            or "server created the session but did not report a complete atomic launch result"
        ),
    )


def daemon_poll_intervals() -> Iterator[float]:
    """
    Yield successive sleeps between daemon readiness probes.

    Starts at :data:`DAEMON_POLL_INITIAL_INTERVAL_S` and grows
    geometrically to :data:`DAEMON_POLL_INTERVAL_S`, then holds there.
    Infinite: callers stop on their own deadline.

    :returns: Iterator of sleep durations in seconds, e.g.
        ``0.1, 0.15, 0.225, ... 0.5, 0.5``.
    """
    interval = DAEMON_POLL_INITIAL_INTERVAL_S
    while True:
        yield interval
        interval = min(interval * DAEMON_POLL_BACKOFF_FACTOR, DAEMON_POLL_INTERVAL_S)


def _json_body(resp: httpx.Response) -> dict[str, object]:
    """Decode a host/runner status response body, tolerating non-JSON.

    The host + runner status endpoints (``GET /v1/hosts/{id}``,
    ``GET /v1/runners/{id}/status``) are expected to answer JSON. But a
    server reached over ``--server`` that does not mount the host router
    (e.g. an API-only deployment, or a misconfigured server) lets these
    paths fall through to the SPA HTML5-history fallback, which answers
    ``200 text/html`` with ``index.html``. Calling ``resp.json()`` on that
    raised an opaque ``json.JSONDecodeError`` that crashed the REPL before
    it ever became ready. Treat any non-dict / non-JSON 200 body as "no
    status yet" so the caller keeps polling and ultimately fails with the
    actionable timeout message instead.

    :param resp: A ``200`` host/runner status response.
    :returns: The decoded JSON object, or an empty dict when the body is
        not a JSON object (e.g. the SPA HTML fallback).
    """
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def open_daemon_client(
    base_url: str,
    headers: dict[str, str],
    host_id: str | None,
    *,
    auth: httpx.Auth | None = None,
    timeout: httpx.Timeout | float | None = None,
) -> httpx.AsyncClient:
    """Open an httpx client for the host-runner protocol, pinned to *host_id*.

    Every request a caller makes on this client — the launch, runner-status
    polls, and the session / terminal calls (including a resume-time reattach
    check) — is scoped to one host, whose control and runner tunnels register
    on a single server replica. Baking the host_id routing header into the
    client's headers at construction pins all of them to that replica, ahead
    of the first request regardless of the caller's call order. The builder
    (:func:`~omnigent.cli_auth.databricks_request_headers`) emits the header
    only on a host-sharded mount, so an unsharded server is unaffected.

    :param base_url: Omnigent server base URL, e.g. the workspace API mount.
    :param headers: Base HTTP headers (auth bearer, workspace routing); not
        mutated — a fresh dict carries the merged routing headers.
    :param host_id: The host to pin to, e.g. ``"host_abc123"``; ``None`` (a
        hostless / local session) leaves routing to the default fallback.
    :param auth: Optional per-request ``httpx.Auth`` (e.g. token refresh).
    :param timeout: Optional httpx timeout for the client.
    :returns: An ``httpx.AsyncClient`` whose requests all name *host_id*.
    """
    from omnigent_client._http import is_loopback_url

    from omnigent.cli_auth import databricks_request_headers

    pinned = {**headers, **databricks_request_headers(base_url, host_id=host_id)}
    # A proxy cannot reach a loopback server, so local targets bypass it.
    return httpx.AsyncClient(
        base_url=base_url,
        headers=pinned,
        auth=auth,
        timeout=timeout,
        trust_env=not is_loopback_url(base_url),
    )


async def wait_for_host_online(
    client: httpx.AsyncClient,
    host_id: str,
    *,
    timeout_s: float,
) -> None:
    """
    Poll the host status endpoint until the daemon is registered.

    ``_ensure_host_daemon`` spawns the connect daemon as a subprocess that
    connects to the server asynchronously, so the runner-launch endpoint
    would 409 ("host offline") until that WebSocket is up. This waits for it.

    Transient transport errors (connection refused while a local server
    is still binding its socket, a dropped keepalive, etc.) are treated
    as "not online yet" and polled through; only the deadline fails the
    wait, with the last transport error included in the message.

    :param client: HTTP client pointed at the Omnigent server.
    :param host_id: This machine's host id, e.g. ``"host_abc123"``.
    :param timeout_s: Max seconds to wait, e.g. ``30.0``.
    :returns: None once the host reports ``status == "online"``.
    :raises click.ClickException: If the host is not online in time.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    intervals = daemon_poll_intervals()
    last_error: httpx.TransportError | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            resp = await client.get(f"/v1/hosts/{url_component(host_id)}")
        except httpx.TransportError as exc:
            last_error = exc
        else:
            if resp.status_code == 200 and _json_body(resp).get("status") == "online":
                return
        await asyncio.sleep(next(intervals))
    message = (
        f"The connect daemon for host {host_id!r} did not come online within {timeout_s:.0f}s."
    )
    if last_error is not None:
        message += f" Last connection error: {last_error!r}."
    message += f" Check that this machine can reach {client.base_url}."
    raise click.ClickException(message)


async def runner_is_online(client: httpx.AsyncClient, runner_id: str) -> bool:
    """
    Return whether a runner currently has an open tunnel to the server.

    :param client: HTTP client pointed at the Omnigent server.
    :param runner_id: Runner id, e.g. ``"runner_abc123"``.
    :returns: ``True`` when the status endpoint reports ``online``.
    """
    resp = await client.get(f"/v1/runners/{url_component(runner_id)}/status")
    return resp.status_code == 200 and bool(_json_body(resp).get("online"))


async def wait_for_runner_online(
    client: httpx.AsyncClient,
    runner_id: str,
    *,
    timeout_s: float,
) -> None:
    """
    Poll until a daemon-spawned runner has connected its tunnel.

    Fails fast when the status endpoint reports the runner process
    died (the host daemon watches its spawned runners and reports
    ``host.runner_exited`` with the exit code and log tail) — a dead
    process can never connect, so waiting out the full timeout would
    only hide the cause.

    Transient transport errors are treated as "not online yet" and
    polled through (same rationale as :func:`wait_for_host_online`);
    only the deadline fails the wait, with the last transport error
    included in the message.

    :param client: HTTP client pointed at the Omnigent server.
    :param runner_id: Runner id the host was asked to spawn, e.g.
        ``"runner_abc123"``.
    :param timeout_s: Max seconds to wait, e.g. ``60.0``.
    :returns: None once the runner is online.
    :raises click.ClickException: If the runner process died, or if it
        does not connect in time.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    intervals = daemon_poll_intervals()
    last_error: httpx.TransportError | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            resp = await client.get(f"/v1/runners/{url_component(runner_id)}/status")
        except httpx.TransportError as exc:
            last_error = exc
        else:
            if resp.status_code == 200:
                body = _json_body(resp)
                if body.get("online"):
                    return
                exit_error = body.get("error")
                if isinstance(exit_error, str) and exit_error:
                    # The runner process is dead — it can never come
                    # online. Surface the daemon-composed cause (exit
                    # code + log tail) instead of polling to a timeout.
                    raise click.ClickException(
                        f"Runner {runner_id!r} failed to start: {exit_error}"
                    )
        await asyncio.sleep(next(intervals))
    message = f"Runner {runner_id!r} did not connect within {timeout_s:.0f}s."
    if last_error is not None:
        message += f" Last connection error: {last_error!r}."
    message += f" Check the runner logs under {display_log_path(process_log_dir('runner'))}/."
    raise click.ClickException(message)


async def launch_or_reuse_daemon_runner(
    client: httpx.AsyncClient,
    *,
    host_id: str,
    session_id: str,
    workspace: str,
    fresh: bool = False,
) -> str:
    """
    Ensure the session is bound to a daemon-spawned runner; return its id.

    Reuses the session's currently-bound runner when it is still online
    (resume into a live session). Otherwise clears any stale binding and
    asks the server to launch a fresh runner on the host via
    ``POST /v1/hosts/{host_id}/runners`` (which atomically binds it).

    :param client: HTTP client pointed at the Omnigent server.
    :param host_id: This machine's host id, e.g. ``"host_abc123"``.
    :param session_id: Session to bind, e.g. ``"conv_abc123"``.
    :param workspace: Absolute host path for the runner cwd, e.g.
        ``"/Users/me/proj"``.
    :param fresh: When ``True``, skip the ``GET /v1/sessions/{id}``
        runner-binding check and go straight to launching a new runner.
        Safe to set when the session was just created in this same startup
        sequence — a brand-new session can't have a runner bound yet, so
        the read would always return empty and only add latency (~2-3s).
    :returns: The bound runner id, e.g. ``"runner_abc123"``.
    :raises click.ClickException: If the launch request fails.
    """
    if fresh:
        existing = None
    else:
        snap = await client.get(f"/v1/sessions/{url_component(session_id)}")
        existing = _json_body(snap).get("runner_id") if snap.status_code == 200 else None
    if isinstance(existing, str) and existing:
        if await runner_is_online(client, existing):
            return existing
        # Stale binding (offline runner): clear it so the launch
        # endpoint's atomic ``UPDATE ... WHERE runner_id IS NULL`` can
        # bind the freshly-spawned runner. "" is the clear sentinel.
        await client.patch(
            f"/v1/sessions/{url_component(session_id)}",
            json={"runner_id": ""},
        )
    # The host tunnel can be briefly absent from the server's in-memory
    # registry while it (re)connects — e.g. just after `omnigent host`
    # restarts, after a server restart/redeploy, or under a flapping tunnel.
    # During that window the launch 409s "host is offline" even though the
    # host is online per the cross-replica DB, and the whole session start
    # fails. Retry transient 409s across the reconnect window so
    # high-latency / reconnecting setups start reliably. Bounded, so a
    # genuinely-offline host still fails reasonably fast.
    _RETRY_DELAYS_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0)  # ~16.5s budget
    for attempt in range(len(_RETRY_DELAYS_S) + 1):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS_S[attempt - 1])
        resp = await client.post(
            f"/v1/hosts/{url_component(host_id)}/runners",
            json={"session_id": session_id, "workspace": workspace},
            timeout=60.0,
        )
        if resp.status_code < 400:
            break
        transient = resp.status_code == 409 and "offline" in error_text(resp).lower()
        if not (transient and attempt < len(_RETRY_DELAYS_S)):
            raise click.ClickException(
                f"Failed to launch a runner on host {host_id!r} "
                f"({resp.status_code}): {error_text(resp)}"
            )
    runner_id = resp.json().get("runner_id")
    if not isinstance(runner_id, str) or not runner_id:
        raise click.ClickException("Host launch response did not include a runner_id.")
    return runner_id


def error_text(resp: httpx.Response) -> str:
    """
    Extract a concise server error message from an HTTP response.

    :param resp: HTTP response returned by AP.
    :returns: Human-readable error text.
    """
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:400]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail
    return json.dumps(body)[:400]
