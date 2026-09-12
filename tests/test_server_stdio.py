"""End-to-end stdio test for the ``python -m aegisub_mcp`` server entry point.

This does not touch the server in-process: it spawns
``sys.executable -m aegisub_mcp`` exactly the way a real MCP client (e.g. Hermes)
would, connects over the stdio transport with the official ``mcp`` SDK client and
walks a full session:

1. ``initialize``      — the handshake, and the server identifies itself
2. ``tools/list``      — every public ``ass_*`` tool of all six tool modules is
                         published with a non-empty description and an object
                         input schema
3. ``tools/call``      — ``ass_open`` + ``ass_list_lines`` against the real
                         fixture ``tests/fixtures/real/basic.ass``, asserting on
                         the returned JSON payload; plus one failing call to
                         pin the documented ``{"error": "<message>"}`` contract
                         for :class:`aegisub_mcp.tools.base.ToolError`

The test skips only when the ``mcp`` SDK is genuinely not importable.  The
subprocess's stderr is inherited, so a bootstrap failure is visible in the
pytest output; stdout is never read as text — it is the JSON-RPC channel.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

pytest.importorskip("mcp", reason="the `mcp` SDK is required for the stdio handshake test")

from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "real" / "basic.ass"

#: Tool modules that must be exposed, mirroring aegisub_mcp.server.TOOL_MODULES.
TOOL_MODULES: tuple[str, ...] = (
    "aegisub_mcp.tools.lines",
    "aegisub_mcp.tools.styles",
    "aegisub_mcp.tools.timing",
    "aegisub_mcp.tools.karaoke_tools",
    "aegisub_mcp.tools.tags_tools",
    "aegisub_mcp.tools.drawing_tools",
)

#: A few names that must survive any refactor of the registration wiring.
ANCHOR_TOOLS: tuple[str, ...] = (
    "ass_open",
    "ass_list_lines",
    "ass_get_line",
    "ass_list_styles",
    "ass_get_style",
    "ass_qc",
    "ass_shift_times",
    "ass_karaoke_get",
    "ass_parse_text",
    "ass_get_drawing",
)

#: The fixture is small and fixed: 1 dialogue + 1 comment.
FIXTURE_LINES = 2
FIXTURE_DIALOGUE = 1
FIXTURE_COMMENTS = 1

HANDSHAKE_TIMEOUT_SECONDS = 120.0
REPORT_PREFIX = "MCP STDIO REPORT: "


# --------------------------------------------------------------------- helpers


def _expected_tool_names() -> set[str]:
    """Every public ``ass_*`` callable the six tool modules define right now."""
    sys.path.insert(0, str(SRC_DIR))  # pyproject's pythonpath=[src] does this too
    names: set[str] = set()
    for dotted in TOOL_MODULES:
        module = importlib.import_module(dotted)
        for name, obj in vars(module).items():
            if name.startswith("ass_") and callable(obj) and not isinstance(obj, type):
                names.add(name)
    return names


def _child_env(out_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    parts = [str(SRC_DIR)] + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # Tools that export/save must not litter the checkout during tests.
    env["AEGISUB_MCP_OUT"] = str(out_dir)
    return env


def _server_params(out_dir: Path, command: Sequence[str] | None = None) -> StdioServerParameters:
    argv = list(command) if command else [sys.executable, "-m", "aegisub_mcp"]
    return StdioServerParameters(
        command=argv[0],
        args=argv[1:],
        env=_child_env(out_dir),
        cwd=str(REPO_ROOT),
    )


def _payload(result: Any) -> dict[str, Any]:
    """The JSON object a tool returned, from structured content or text content."""
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and structured:
        return structured
    for block in getattr(result, "content", None) or ():
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise AssertionError(f"tool result carried no JSON object payload: {result!r}")


def _assert_tool_ok(result: Any, tool: str) -> dict[str, Any]:
    """Fail with the server-side message when a call came back as an error."""
    if getattr(result, "is_error", False):
        text = "\n".join(
            getattr(block, "text", "") or "" for block in (getattr(result, "content", None) or ())
        )
        pytest.fail(f"{tool} returned an MCP error: {text.strip() or '(no message)'}")
    payload = _payload(result)
    if "error" in payload and isinstance(payload["error"], str):
        pytest.fail(f"{tool} returned a tool error: {payload['error']}")
    return payload


async def _session_report(out_dir: Path, command: Sequence[str] | None = None) -> dict[str, Any]:
    """Open one real MCP session and return everything the assertions need."""
    params = _server_params(out_dir, command)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()

            listing = await session.list_tools()
            tools = list(listing.tools)
            names = sorted(tool.name for tool in tools)

            opened = _assert_tool_ok(
                await session.call_tool("ass_open", {"path": str(FIXTURE)}), "ass_open"
            )
            doc_id = opened.get("doc_id")
            assert doc_id, f"ass_open returned no doc_id: {opened!r}"

            lines = _assert_tool_ok(
                await session.call_tool(
                    "ass_list_lines", {"doc_id": doc_id, "limit": None}
                ),
                "ass_list_lines",
            )

            # The tool layer's documented user-error contract, over the wire:
            # base.ToolError must arrive as a *successful* result whose JSON is
            # {"error": "<message>"} — not as a masked `Error executing tool ...`.
            missing = await session.call_tool(
                "ass_open", {"path": str(FIXTURE.with_name("definitely_missing.ass"))}
            )
            missing_error: dict[str, Any] = {
                "is_error": bool(getattr(missing, "is_error", False)),
                "payload": None,
                "text": "\n".join(
                    getattr(block, "text", "") or ""
                    for block in (getattr(missing, "content", None) or ())
                ),
            }
            if not missing_error["is_error"]:
                try:
                    missing_error["payload"] = _payload(missing)
                except AssertionError:
                    pass

    server_info = getattr(init, "server_info", None)
    return {
        "protocol_version": getattr(init, "protocol_version", None),
        "server_name": getattr(server_info, "name", None),
        "server_version": getattr(server_info, "version", None),
        "tool_count": len(names),
        "tool_names": names,
        "undescribed_tools": sorted(
            tool.name for tool in tools if not (getattr(tool, "description", None) or "").strip()
        ),
        "opened": opened,
        "lines": lines,
        "missing_file": missing_error,
    }


def _run(coro: Any) -> Any:
    import asyncio

    async def guarded() -> Any:
        return await asyncio.wait_for(coro, timeout=HANDSHAKE_TIMEOUT_SECONDS)

    try:
        return asyncio.run(guarded())
    except asyncio.TimeoutError:  # pragma: no cover - only on a hung server
        pytest.fail(
            f"the aegisub-mcp stdio handshake did not finish within "
            f"{HANDSHAKE_TIMEOUT_SECONDS:.0f}s (command: {sys.executable} -m aegisub_mcp)"
        )


# ----------------------------------------------------------------------- tests


def test_stdio_server_handshake_lists_and_calls_tools(tmp_path: Path) -> None:
    """initialize -> tools/list -> tools/call over a real stdio subprocess."""
    assert FIXTURE.is_file(), f"missing real fixture: {FIXTURE}"

    # Guard against measuring an *installed* copy instead of this checkout.
    import aegisub_mcp

    assert Path(aegisub_mcp.__file__).resolve().is_relative_to(REPO_ROOT), (
        f"the test imported aegisub_mcp from {aegisub_mcp.__file__}, not from {SRC_DIR}"
    )

    expected = _expected_tool_names()
    assert len(expected) > 100, f"suspiciously few public tools discovered: {len(expected)}"

    report = _run(_session_report(tmp_path))
    print(REPORT_PREFIX + json.dumps(
        {
            "server_name": report["server_name"],
            "server_version": report["server_version"],
            "protocol_version": report["protocol_version"],
            "tool_count": report["tool_count"],
            "tool_names": report["tool_names"],
            "ass_open": {k: report["opened"].get(k) for k in ("doc_id", "lines", "dialogue", "comments", "script_type", "play_res_x", "play_res_y")},
            "ass_list_lines": {
                k: report["lines"].get(k) for k in ("doc_id", "total", "returned", "indices")
            },
            "missing_file": report["missing_file"],
        },
        sort_keys=True,
    ))

    received = set(report["tool_names"])

    # (1) every tool the tool modules define must be published over MCP
    missing = sorted(expected - received)
    assert not missing, f"tools defined but not exposed over MCP: {missing}"
    unexpected = sorted(received - expected)
    assert not unexpected, f"tools exposed over MCP that no module defines: {unexpected}"
    assert report["tool_count"] == len(received) == len(expected)
    assert report["tool_count"] > 100, f"non-trivial count expected, got {report['tool_count']}"

    # (2) the anchor names, and no tool without a description
    anchors = [name for name in ANCHOR_TOOLS if name not in received]
    assert not anchors, f"anchor tools missing from tools/list: {anchors}"
    assert not report["undescribed_tools"], (
        f"tools published without a description: {report['undescribed_tools']}"
    )

    # (3) the handshake identifies this server
    assert report["server_name"] == "aegisub-mcp", report["server_name"]
    assert report["server_version"], "server reported no version"

    # (4) real payload from the real fixture
    opened = report["opened"]
    assert opened.get("path", "").endswith("basic.ass"), opened
    assert opened.get("lines") == FIXTURE_LINES, opened
    assert opened.get("dialogue") == FIXTURE_DIALOGUE, opened
    assert opened.get("comments") == FIXTURE_COMMENTS, opened
    assert opened.get("script_type") == "v4.00+", opened
    assert (opened.get("play_res_x"), opened.get("play_res_y")) == (1280, 720), opened

    lines = report["lines"]
    assert lines.get("total") == FIXTURE_LINES, lines
    assert lines.get("returned") == FIXTURE_LINES, lines
    assert lines.get("indices") == [0, 1], lines
    rows = lines.get("lines")
    assert isinstance(rows, list) and len(rows) == FIXTURE_LINES, lines
    dialogue, comment = rows[0], rows[1]
    assert dialogue["kind"] == "Dialogue", dialogue
    assert dialogue["start_ms"] == 1000 and dialogue["end_ms"] == 4000, dialogue
    assert dialogue["style"] == "Default", dialogue
    assert dialogue["text"] == r"Hello {\i1}world{\i0}", dialogue
    assert dialogue["plain_text"] == "Hello world", dialogue
    assert comment["kind"] == "Comment", comment
    assert comment["text"] == "Preserve me", comment

    # (5) a real user error travels as {"error": ...}, not as a masked crash
    missing = report["missing_file"]
    assert missing["is_error"] is False, (
        "a ToolError surfaced as an MCP error instead of the documented "
        "{'error': ...} payload: " + missing["text"].strip()[:200]
    )
    payload = missing["payload"]
    assert isinstance(payload, dict), f"no JSON payload for the failed call: {missing!r}"
    message = payload.get("error")
    assert isinstance(message, str) and "file not found" in message, payload
    assert "definitely_missing.ass" in message, payload


def test_console_script_target_serves_stdio(tmp_path: Path) -> None:
    """The ``[project.scripts]`` target must itself be a working stdio server.

    ``pyproject.toml`` declares ``aegisub-mcp = "aegisub_mcp.server:main"``; this
    spawns exactly that callable (without needing the project installed) so the
    declared entry point is verified, not just ``python -m aegisub_mcp``.
    """
    launcher = "import aegisub_mcp.server as s; raise SystemExit(s.main())"
    report = _run(_session_report(tmp_path, [sys.executable, "-c", launcher]))
    names = set(report["tool_names"])
    assert report["tool_count"] > 100, report["tool_count"]
    assert {"ass_open", "ass_list_lines"} <= names, report["tool_count"]


def test_stdio_server_keeps_stdout_free_of_diagnostics(tmp_path: Path) -> None:
    """Serving with stdin closed must leave stdout empty and log to stderr."""
    proc = subprocess.run(
        [sys.executable, "-m", "aegisub_mcp"],
        cwd=str(REPO_ROOT),
        env=_child_env(tmp_path),
        input=b"",  # immediate EOF: the server handshakes nothing and exits
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=HANDSHAKE_TIMEOUT_SECONDS,
    )
    assert proc.returncode == 0, (
        f"server exited {proc.returncode}\nstderr:\n{proc.stderr.decode(errors='replace')}"
    )
    assert proc.stdout == b"", (
        f"the server wrote {len(proc.stdout)} byte(s) to stdout outside the MCP protocol: "
        f"{proc.stdout[:400]!r}"
    )
    stderr = proc.stderr.decode(errors="replace")
    assert re.search(r"registered \d+ tool\(s\) from 6 module\(s\)", stderr), (
        f"registration summary missing from stderr:\n{stderr}"
    )
