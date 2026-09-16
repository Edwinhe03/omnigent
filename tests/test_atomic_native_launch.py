"""Focused coverage for warm atomic native CLI create-and-launch."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import click
import httpx
import pytest

from omnigent.harnesses.claude_native import main as claude_native
from omnigent.harnesses.codex_native import main as codex_native
from omnigent.host.daemon_launch import (
    AtomicHostCreateFallback,
    DaemonSessionCreateResult,
    atomic_host_create_fallback_reason,
)


@contextlib.asynccontextmanager
async def _client_context() -> AsyncIterator[object]:
    yield object()


def _multipart_metadata(request: httpx.Request) -> dict[str, object]:
    body = request.content.decode("utf-8")
    marker = 'name="metadata"'
    part = body.index(marker)
    start = body.index("\r\n\r\n", part) + 4
    end = body.index("\r\n", start)
    payload = json.loads(body[start:end])
    assert isinstance(payload, dict)
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_session,harness_name",
    [
        (claude_native._create_claude_session, "Claude"),
        (codex_native._create_codex_session, "Codex"),
    ],
)
async def test_atomic_create_preserves_bundle_metadata_and_launch_args(
    create_session: Callable[..., Awaitable[str | DaemonSessionCreateResult]],
    harness_name: str,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["metadata"] = _multipart_metadata(request)
        captured["body"] = request.content
        return httpx.Response(
            201,
            json={
                "session_id": "conv_new",
                "runner_id": "runner_new",
                "runner_launch_status": "launched",
                "runner_launch_error": None,
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.com"
    ) as client:
        result = await create_session(
            client,
            b"exact-bundle-bytes",
            bridge_id=None,
            terminal_launch_args=["--flag", "value"],
            host_id="host_local",
            workspace="/repo",
            require_host_launch_result=True,
        )

    assert result == DaemonSessionCreateResult(
        session_id="conv_new",
        runner_id="runner_new",
        runner_launch_status="launched",
    )
    assert captured["metadata"]["host_id"] == "host_local"
    assert captured["metadata"]["workspace"] == "/repo"
    assert captured["metadata"]["host_launch_contract"] == "result_v1"
    assert captured["metadata"]["terminal_launch_args"] == ["--flag", "value"]
    assert b"exact-bundle-bytes" in captured["body"]
    assert harness_name in {"Claude", "Codex"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_session,harness_name",
    [
        (claude_native._create_claude_session, "Claude"),
        (codex_native._create_codex_session, "Codex"),
    ],
)
async def test_atomic_create_timeout_is_not_retried(
    create_session: Callable[..., Awaitable[str | DaemonSessionCreateResult]],
    harness_name: str,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("response lost", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.com"
    ) as client:
        with pytest.raises(click.ClickException, match="may have created"):
            await create_session(
                client,
                b"bundle",
                bridge_id=None,
                host_id="host_local",
                workspace="/repo",
                require_host_launch_result=True,
            )

    assert attempts == 1
    assert harness_name in {"Claude", "Codex"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_session",
    [claude_native._create_claude_session, codex_native._create_codex_session],
)
async def test_atomic_create_non_host_conflict_is_not_retried(
    create_session: Callable[..., Awaitable[str | DaemonSessionCreateResult]],
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            409,
            json={
                "error": {
                    "code": "conflict",
                    "message": "session title is already reserved",
                }
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.com"
    ) as client:
        with pytest.raises(click.ClickException, match="title is already reserved"):
            await create_session(
                client,
                b"bundle",
                bridge_id=None,
                host_id="host_local",
                workspace="/repo",
                require_host_launch_result=True,
            )

    assert attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_session",
    [claude_native._create_claude_session, codex_native._create_codex_session],
)
async def test_old_server_rejects_contract_before_creating_then_legacy_create_succeeds(
    create_session: Callable[..., Awaitable[str | DaemonSessionCreateResult]],
) -> None:
    created_sessions = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal created_sessions
        metadata = _multipart_metadata(request)
        if "host_launch_contract" in metadata:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "invalid_input",
                        "message": (
                            "invalid session metadata: host_launch_contract: "
                            "Extra inputs are not permitted"
                        ),
                    }
                },
            )
        created_sessions += 1
        return httpx.Response(201, json={"session_id": "conv_legacy"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.com"
    ) as client:
        with pytest.raises(AtomicHostCreateFallback, match="does not support"):
            await create_session(
                client,
                b"bundle",
                bridge_id=None,
                host_id="host_local",
                workspace="/repo",
                require_host_launch_result=True,
            )
        session_id = await create_session(client, b"bundle", bridge_id=None)

    assert session_id == "conv_legacy"
    assert created_sessions == 1


@pytest.mark.parametrize(
    ("status_code", "code", "message", "safe"),
    [
        (
            400,
            "invalid_input",
            ("invalid session metadata: host_launch_contract: Extra inputs are not permitted"),
            True,
        ),
        (400, "invalid_input", "host 'laptop' is offline; reconnect the host", True),
        (400, "wrong_replica", "host 'laptop' is on another replica; retry", True),
        (409, "wrong_replica", "host 'laptop' is on another replica; retry", False),
        (400, "wrong_replica", "unrelated wrong-replica conflict", False),
        (500, "invalid_input", "host 'laptop' is offline; reconnect the host", False),
        (
            400,
            "invalid_input",
            "host_launch_contract requires a connected host",
            False,
        ),
        (409, "conflict", "session already has a runner bound", False),
        (409, "conflict", "unrelated resource conflict", False),
    ],
)
def test_atomic_fallback_only_accepts_proven_precreate_host_errors(
    status_code: int,
    code: str,
    message: str,
    safe: bool,
) -> None:
    request = httpx.Request("POST", "https://example.com/v1/sessions")
    response = httpx.Response(
        status_code,
        request=request,
        json={"error": {"code": code, "message": message}},
    )

    reason = atomic_host_create_fallback_reason(response)

    assert (reason is not None) is safe


@pytest.mark.asyncio
async def test_warm_claude_uses_inline_runner_without_explicit_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_calls = 0

    async def create_session(*args: object, **kwargs: object) -> DaemonSessionCreateResult:
        nonlocal create_calls
        create_calls += 1
        assert args[1] == b"bundle"
        assert kwargs["host_id"] == "host_local"
        assert kwargs["workspace"] == "/repo"
        assert kwargs["terminal_launch_args"] == ["--print", "hi"]
        assert kwargs["require_host_launch_result"] is True
        return DaemonSessionCreateResult(
            session_id="conv_new",
            runner_id="runner_inline",
            runner_launch_status="launched",
        )

    async def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected fallback: {args!r} {kwargs!r}")

    async def runner_online(client: object, runner_id: str) -> None:
        del client
        assert runner_id == "runner_inline"

    async def terminal_ready(client: object, session_id: str, *, timeout_s: float) -> str:
        del client, timeout_s
        assert session_id == "conv_new"
        return claude_native.claude_terminal_resource_id()

    async def read_tmux(client: object, session_id: str) -> claude_native._ClaudeTerminalTmux:
        del client, session_id
        return claude_native._ClaudeTerminalTmux(socket=tmp_path / "tmux.sock", target="c:1")

    monkeypatch.setattr(claude_native, "open_daemon_client", lambda *a, **k: _client_context())
    monkeypatch.setattr(claude_native, "_create_claude_session", create_session)
    monkeypatch.setattr(claude_native, "wait_for_host_online", unexpected)
    monkeypatch.setattr(claude_native, "launch_or_reuse_daemon_runner", unexpected)
    monkeypatch.setattr(claude_native, "_wait_for_runner_online_with_startup_event", runner_online)
    monkeypatch.setattr(claude_native, "_wait_for_claude_terminal_ready", terminal_ready)
    monkeypatch.setattr(claude_native, "_read_claude_terminal_tmux", read_tmux)

    prepared = await claude_native._prepare_claude_terminal_via_daemon(
        base_url="https://example.com",
        headers={},
        session_id=None,
        session_bundle=b"bundle",
        claude_args=("--print", "hi"),
        host_id="host_local",
        workspace="/repo",
        host_already_connected=True,
    )

    assert create_calls == 1
    assert prepared.session_id == "conv_new"


@pytest.mark.asyncio
async def test_warm_claude_failed_inline_launch_recovers_without_new_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_calls = 0
    launch_fresh: list[bool] = []
    launch_calls = 0

    async def create_session(*args: object, **kwargs: object) -> DaemonSessionCreateResult:
        nonlocal create_calls
        del args, kwargs
        create_calls += 1
        return DaemonSessionCreateResult(
            session_id="conv_new",
            runner_id="runner_bound",
            runner_launch_status="failed",
            runner_launch_error="host disconnected",
        )

    async def host_online(client: object, host_id: str, *, timeout_s: float) -> None:
        del client, host_id, timeout_s

    async def launch_runner(
        client: object,
        *,
        host_id: str,
        session_id: str,
        workspace: str,
        fresh: bool = False,
    ) -> str:
        nonlocal launch_calls
        del client, host_id, session_id, workspace
        launch_calls += 1
        launch_fresh.append(fresh)
        return "runner_bound"

    async def runner_online(client: object, runner_id: str) -> None:
        del client, runner_id

    async def terminal_ready(client: object, session_id: str, *, timeout_s: float) -> str:
        del client, session_id, timeout_s
        return claude_native.claude_terminal_resource_id()

    async def read_tmux(client: object, session_id: str) -> claude_native._ClaudeTerminalTmux:
        del client, session_id
        return claude_native._ClaudeTerminalTmux(socket=tmp_path / "tmux.sock", target="c:1")

    monkeypatch.setattr(claude_native, "open_daemon_client", lambda *a, **k: _client_context())
    monkeypatch.setattr(claude_native, "_create_claude_session", create_session)
    monkeypatch.setattr(claude_native, "wait_for_host_online", host_online)
    monkeypatch.setattr(claude_native, "launch_or_reuse_daemon_runner", launch_runner)
    monkeypatch.setattr(claude_native, "_wait_for_runner_online_with_startup_event", runner_online)
    monkeypatch.setattr(claude_native, "_wait_for_claude_terminal_ready", terminal_ready)
    monkeypatch.setattr(claude_native, "_read_claude_terminal_tmux", read_tmux)

    await claude_native._prepare_claude_terminal_via_daemon(
        base_url="https://example.com",
        headers={},
        session_id=None,
        session_bundle=b"bundle",
        claude_args=(),
        host_id="host_local",
        workspace="/repo",
        host_already_connected=True,
    )

    assert create_calls == 1
    assert launch_calls == 1
    assert launch_fresh == [False]


@pytest.mark.asyncio
async def test_stale_connected_marker_falls_back_before_create_and_keeps_legacy_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_calls = 0
    created_sessions = 0
    order: list[str] = []
    launch_fresh: list[bool] = []

    async def create_session(*args: object, **kwargs: object) -> str | DaemonSessionCreateResult:
        nonlocal create_calls, created_sessions
        del args
        create_calls += 1
        if kwargs.get("require_host_launch_result"):
            raise AtomicHostCreateFallback("host is offline")
        order.append("create:start")
        await asyncio.sleep(0)
        order.append("create:end")
        created_sessions += 1
        return "conv_new"

    async def host_online(client: object, host_id: str, *, timeout_s: float) -> None:
        del client, host_id, timeout_s
        order.append("host:start")
        await asyncio.sleep(0)
        order.append("host:end")

    async def launch_runner(
        client: object,
        *,
        host_id: str,
        session_id: str,
        workspace: str,
        fresh: bool = False,
    ) -> str:
        del client, host_id, session_id, workspace
        launch_fresh.append(fresh)
        return "runner_new"

    async def runner_online(client: object, runner_id: str, *, timeout_s: float) -> None:
        del client, runner_id, timeout_s

    async def bind(client: object, session_id: str, runner_id: str) -> None:
        del client, session_id, runner_id

    async def ensure(client: object, session_id: str) -> None:
        del client, session_id

    async def terminal_ready(
        client: object, session_id: str, *, timeout_s: float
    ) -> codex_native.LaunchedCodexTerminal:
        del client, session_id, timeout_s
        return codex_native.LaunchedCodexTerminal(
            terminal_id="terminal_codex_main", tmux_socket=None, tmux_target=None
        )

    monkeypatch.setattr(codex_native, "open_daemon_client", lambda *a, **k: _client_context())
    monkeypatch.setattr(codex_native, "_create_codex_session", create_session)
    monkeypatch.setattr(codex_native, "wait_for_host_online", host_online)
    monkeypatch.setattr(codex_native, "launch_or_reuse_daemon_runner", launch_runner)
    monkeypatch.setattr(codex_native, "wait_for_runner_online", runner_online)
    monkeypatch.setattr(codex_native, "_bind_session_runner", bind)
    monkeypatch.setattr(codex_native, "_ensure_codex_terminal_on_runner", ensure)
    monkeypatch.setattr(codex_native, "_wait_for_codex_terminal_ready", terminal_ready)

    await codex_native._prepare_codex_terminal_via_daemon(
        base_url="https://example.com",
        headers={},
        session_id=None,
        session_bundle=b"bundle",
        codex_args=(),
        model=None,
        host_id="host_local",
        workspace="/repo",
        host_already_connected=True,
    )

    assert create_calls == 2
    assert created_sessions == 1
    assert order.index("host:start") < order.index("create:end")
    assert launch_fresh == [True]


@pytest.mark.asyncio
async def test_indeterminate_inline_launch_does_not_start_duplicate_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_calls = 0

    async def create_session(*args: object, **kwargs: object) -> DaemonSessionCreateResult:
        nonlocal create_calls
        del args, kwargs
        create_calls += 1
        return DaemonSessionCreateResult(
            session_id="conv_created",
            runner_id="runner_maybe_started",
            runner_launch_status="indeterminate",
            runner_launch_error="host launch timed out",
        )

    async def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected duplicate runner fallback: {args!r} {kwargs!r}")

    monkeypatch.setattr(codex_native, "open_daemon_client", lambda *a, **k: _client_context())
    monkeypatch.setattr(codex_native, "_create_codex_session", create_session)
    monkeypatch.setattr(codex_native, "wait_for_host_online", unexpected)
    monkeypatch.setattr(codex_native, "launch_or_reuse_daemon_runner", unexpected)

    with pytest.raises(click.ClickException, match=r"conv_created.*outcome is unknown"):
        await codex_native._prepare_codex_terminal_via_daemon(
            base_url="https://example.com",
            headers={},
            session_id=None,
            session_bundle=b"bundle",
            codex_args=(),
            model=None,
            host_id="host_local",
            workspace="/repo",
            host_already_connected=True,
        )

    assert create_calls == 1


@pytest.mark.asyncio
async def test_warm_codex_refusal_surfaces_without_duplicate_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_calls = 0

    async def create_session(*args: object, **kwargs: object) -> DaemonSessionCreateResult:
        nonlocal create_calls
        del args, kwargs
        create_calls += 1
        return DaemonSessionCreateResult(
            session_id="conv_new",
            runner_id="runner_refused",
            runner_launch_status="failed",
            runner_launch_error="codex not configured",
        )

    async def host_online(client: object, host_id: str, *, timeout_s: float) -> None:
        del client, host_id, timeout_s

    async def refuse(*args: object, **kwargs: object) -> str:
        raise click.ClickException("host failed to launch runner: codex not configured")

    monkeypatch.setattr(codex_native, "open_daemon_client", lambda *a, **k: _client_context())
    monkeypatch.setattr(codex_native, "_create_codex_session", create_session)
    monkeypatch.setattr(codex_native, "wait_for_host_online", host_online)
    monkeypatch.setattr(codex_native, "launch_or_reuse_daemon_runner", refuse)

    with pytest.raises(click.ClickException, match="codex not configured"):
        await codex_native._prepare_codex_terminal_via_daemon(
            base_url="https://example.com",
            headers={},
            session_id=None,
            session_bundle=b"bundle",
            codex_args=(),
            model=None,
            host_id="host_local",
            workspace="/repo",
            host_already_connected=True,
        )

    assert create_calls == 1
