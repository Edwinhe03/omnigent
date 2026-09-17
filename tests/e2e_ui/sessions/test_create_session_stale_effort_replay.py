"""E2E: creating a session must not replay a prior in-session effort pick.

Reported journey: the user changes the reasoning-effort picker to Medium inside
an existing claude-native session, then opens the create composer, leaves
effort at Default, types an initial prompt, and creates a new session. Two
things can go wrong:

1. (deterministic) The Medium pick is kept as a cross-session preference and
   silently PATCHed onto the freshly-created session when it binds, so the new
   session comes up on Medium even though the composer showed Default.
2. (order-dependent) The follow-up config command could race the seed prompt and
   leave the new session empty.

This drives the literal create-with-prompt journey against the live SPA:

* A real ``claude`` pick in session B arms any cross-session effort state
  (reload so it is stable — the mock CLI reports its own default, so the reload
  pins the picked state deterministically instead of racing that report).
* The create composer's host list, agent catalog, and create POST are stubbed
  (the directly-tunneled harness registers no host, and a real create would
  launch a runner), so the REAL ``setPendingInitialPrompt`` + navigate handoff
  lands on a pre-seeded effort-capable session A. ``/events`` is intercepted so
  no real turn runs and we can see exactly where the seed prompt is POSTed. The
  bind-time effort PATCH is let through to the real server, so A's persisted
  ``reasoning_effort`` is the authoritative leak signal. This mirrors
  ``test_initial_prompt_session_switch.py``.

Assertions:

* Facet 2 (seed prompt): the seed prompt must reach the new session's /events.
* Facet 1 (stale effort): the new session's ``reasoning_effort`` must stay null.
  Red while the bug lives — binding A replays B's Medium onto it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import subprocess
import threading
from collections.abc import Coroutine, Iterator
from typing import Any

import httpx
import pytest
from playwright.async_api import Page, Route, async_playwright

from tests.e2e_ui.conftest import (
    _create_native_claude_session,
    _ensure_runner_online,
    _server_state,
    _temp_omnigent_mock_config,
)

_PROMPT = "sentinel-stale-effort-seed-prompt read the README please"
_SESSIONS_RE = re.compile(r"/v1/sessions(\?.*)?$")
_EVENTS_RE = re.compile(r"/v1/sessions/([^/]+)/events$")

_GEAR = '[data-testid="composer-config-gear"]'
_EFFORT_ROW = '[data-testid="composer-agent-effort-select"]'
_EFFORT_MEDIUM = '[data-testid="composer-agent-effort-medium"]'

_UI_TIMEOUT_MS = 45_000
# The bind-time PATCH is fire-and-forget; give it a generous quiet window so a
# "stayed null" read is a real negative, not an early one.
_SETTLE_S = 6.0


@pytest.fixture
def claude_session_pair(
    live_server: str,
    mock_llm_server_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, str, str]]:
    """Two runner-bound claude-native sessions on the live server.

    :returns: ``(base_url, session_a, session_b)`` — A is the create target,
        B is where the prior effort pick happens.
    """
    respawned = _ensure_runner_online(live_server, tmp_path_factory)
    runner_id = str(_server_state["runner_id"])
    use_mock = not os.environ.get("LLM_API_KEY")
    ctx: Any = (
        _temp_omnigent_mock_config(mock_llm_server_url, "claude")
        if use_mock
        else contextlib.nullcontext()
    )
    with ctx:
        session_a = _create_native_claude_session(live_server, runner_id)
        session_b = _create_native_claude_session(live_server, runner_id)
        try:
            yield (live_server, session_a, session_b)
        finally:
            for session_id in (session_a, session_b):
                httpx.delete(f"{live_server}/v1/sessions/{session_id}", timeout=10.0)
            if respawned is not None:
                respawned.terminate()
                try:
                    respawned.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    respawned.kill()
                    respawned.wait(timeout=5)


def _run_in_fresh_loop(coro: Coroutine[Any, Any, None]) -> None:
    """Run *coro* in a dedicated thread's event loop and re-raise its failure.

    The e2e_ui suite runs sync pytest-playwright tests in the same session, so
    pytest-asyncio can't start a loop on the main thread once one has run.
    """
    captured: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            asyncio.run(coro)
        except BaseException as exc:
            captured["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if "error" in captured:
        raise captured["error"]


async def _wait_until(predicate, *, timeout_s: float = 15.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


def _server_effort(base_url: str, session_id: str) -> str | None:
    resp = httpx.get(f"{base_url}/v1/sessions/{session_id}", timeout=10.0)
    resp.raise_for_status()
    return resp.json().get("reasoning_effort")


async def _pick_effort_medium_in(page: Page, base_url: str, session_id: str) -> None:
    await page.goto(f"{base_url}/c/{session_id}")
    await page.locator(_GEAR).wait_for(state="visible", timeout=_UI_TIMEOUT_MS)
    await page.locator(_GEAR).click()
    await page.locator(_EFFORT_ROW).wait_for(state="visible", timeout=10_000)
    await page.locator(_EFFORT_ROW).click()
    await page.locator(_EFFORT_MEDIUM).wait_for(state="visible", timeout=10_000)
    await page.locator(_EFFORT_MEDIUM).click()


def test_creating_a_session_does_not_replay_a_prior_effort_pick(
    claude_session_pair: tuple[str, str, str],
) -> None:
    """The exact reported journey: pick Medium in B, then create with a prompt.

    Red while the bug lives: the created session's ``reasoning_effort`` flips
    from ``null`` to ``"medium"`` (session B's picker value) with no effort
    picked in the create composer.
    """
    base_url, session_a, session_b = claude_session_pair
    _run_in_fresh_loop(_drive(base_url, session_a, session_b))


async def _drive(base_url: str, session_a: str, session_b: str) -> None:
    assert _server_effort(base_url, session_a) is None
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        try:
            event_posts: list[tuple[str, str]] = []

            async def handle_events(route: Route) -> None:
                match = _EVENTS_RE.search(route.request.url)
                assert match is not None
                text = route.request.post_data_json["data"]["content"][0]["text"]
                event_posts.append((match.group(1), text))
                await route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"queued": True, "item_id": "ci_e2e"}),
                )

            async def handle_hosts(route: Route) -> None:
                await route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "hosts": [
                                {
                                    "host_id": "host_e2e",
                                    "name": "e2e-host",
                                    "owner": "e2e",
                                    "status": "online",
                                }
                            ]
                        }
                    ),
                )

            async def handle_agents(route: Route) -> None:
                await route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "data": [
                                {
                                    "id": "ag_e2e",
                                    "name": "hello_world",
                                    "display_name": "Hello World",
                                    "description": None,
                                    "harness": None,
                                }
                            ]
                        }
                    ),
                )

            async def handle_sessions(route: Route) -> None:
                # Fake ONLY the composer's create POST; everything else is real.
                if route.request.method == "POST":
                    await route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({"id": session_a}),
                    )
                else:
                    await route.continue_()

            await page.route("**/v1/sessions/*/events", handle_events)
            await page.route("**/v1/hosts", handle_hosts)
            await page.route("**/v1/agents", handle_agents)
            await page.route(_SESSIONS_RE, handle_sessions)

            # A real pick in B, then a reload so the picked state is stable at
            # the create handoff.
            await _pick_effort_medium_in(page, base_url, session_b)
            assert await _wait_until(
                lambda: _server_effort(base_url, session_b) == "medium",
                timeout_s=20.0,
            ), "the Medium pick never persisted to session B"
            await page.goto(f"{base_url}/c/{session_b}")
            await page.locator(_GEAR).wait_for(state="visible", timeout=_UI_TIMEOUT_MS)
            await page.evaluate(
                """() => localStorage.setItem(
                    "omnigent:recent-workspaces",
                    JSON.stringify({ host_e2e: ["/tmp"] }),
                )"""
            )

            # Create a new session from the composer with a prompt, effort left
            # at Default. The stubbed create lands the real handoff on A.
            await page.get_by_test_id("new-chat-button").click()
            await page.get_by_test_id("new-chat-landing-input").wait_for(
                state="visible", timeout=15_000
            )
            await page.get_by_test_id("new-chat-landing-input").fill(_PROMPT)
            await page.get_by_test_id("new-chat-landing-submit").click()
            await page.wait_for_url(re.compile(rf"/c/{re.escape(session_a)}"))

            # Facet 2 (seed prompt): the prompt must reach the new session.
            seed_delivered = await _wait_until(
                lambda: any(text == _PROMPT for _, text in event_posts),
                timeout_s=12.0,
            )
            prompt_targets = [sid for sid, text in event_posts if text == _PROMPT]
            assert seed_delivered and prompt_targets == [session_a], (
                f"the initial prompt did not reach the new session {session_a}; "
                f"/events POST targets for the prompt were {prompt_targets}"
            )

            # Facet 1 (stale effort): no effort was picked in the create
            # composer, so the new session must stay on its own (null) effort.
            await asyncio.sleep(_SETTLE_S)
            effort_a = _server_effort(base_url, session_a)
            assert effort_a is None, (
                f"creating the session replayed session B's Medium pick onto it: "
                f"reasoning_effort became {effort_a!r} with Default selected in the "
                f"create composer"
            )
        finally:
            await browser.close()
