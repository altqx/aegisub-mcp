"""End-to-end streamable-HTTP test for ``python -m aegisub_mcp.http_server``.

The stdio entry point can only ever negotiate the ``initialize`` handshake revisions
(``... 2025-11-25``); ``2026-07-28`` is the stateless per-request envelope and is
served over HTTP only.  This test spawns the real HTTP entry point as a subprocess
and drives it over the wire with the standard library — no SDK client, so nothing
in the SDK's client stack can hide a protocol-level failure:

1. ``server/discover``  — the 2026-07-28 handshake replacement; the server must
   advertise ``2026-07-28`` and must *not* issue an ``Mcp-Session-Id``
2. ``tools/call``       — a real tool call on the real fixture with no ``initialize``
   sent at all, plus a second, independent POST that sees the first one's document
   (the per-request envelope has no session to carry it, so process state must)
3. rejections           — header/body disagreement, a missing ``_meta`` envelope,
   an unsupported modern version, and a rebinding ``Host`` header
4. legacy compat        — the same endpoint still negotiates ``2025-11-25`` with the
   ``initialize`` handshake and a session, so nothing regressed for stdio-era clients
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest

pytest.importorskip("mcp", reason="the `mcp` SDK is required for the streamable-HTTP test")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "real" / "basic.ass"

MCP_PATH = "/mcp"
MODERN_VERSION = "2026-07-28"
LEGACY_VERSION = "2025-11-25"

PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
CLIENT_CAPABILITIES_META_KEY = "io.modelcontextprotocol/clientCapabilities"
CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"

START_TIMEOUT_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 30.0
REPORT_PREFIX = "MCP HTTP REPORT: "

#: The fixture is small and fixed: 1 dialogue + 1 comment.
FIXTURE_LINES = 2


# --------------------------------------------------------------------- helpers


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _child_env(out_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR)] + ([existing] if existing else []))
    env["AEGISUB_MCP_OUT"] = str(out_dir)
    return env


def _envelope(version: str = MODERN_VERSION) -> dict[str, Any]:
    """The 2026-07-28 per-request `_meta` envelope every modern request must carry."""
    return {
        PROTOCOL_VERSION_META_KEY: version,
        CLIENT_CAPABILITIES_META_KEY: {},
        CLIENT_INFO_META_KEY: {"name": "pytest-http-probe", "version": "0"},
    }


def _post(
    port: int,
    body: dict[str, Any],
    headers: Sequence[tuple[str, str]] = (),
    *,
    host_header: str | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> tuple[int, dict[str, str], str]:
    """One raw POST to the endpoint; returns (status, lowercase headers, body text)."""
    payload = json.dumps(body).encode("utf-8")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.putrequest("POST", MCP_PATH, skip_host=True)
        conn.putheader("Host", host_header or f"127.0.0.1:{port}")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Accept", "application/json, text/event-stream")
        for name, value in headers:
            conn.putheader(name, value)
        conn.putheader("Content-Length", str(len(payload)))
        conn.endheaders(payload)
        response = conn.getresponse()
        text = response.read().decode("utf-8", "replace")
        return response.status, {k.lower(): v for k, v in response.getheaders()}, text
    finally:
        conn.close()


def _json_payload(text: str) -> dict[str, Any]:
    """The JSON-RPC message in a response, whether it arrived plain or as SSE."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    for line in stripped.splitlines():
        if not line.startswith("data:"):
            continue
        candidate = line[len("data:") :].strip()
        if candidate.startswith("{"):
            return json.loads(candidate)
    raise AssertionError(f"no JSON-RPC message in response body: {text[:400]!r}")


def _modern(
    port: int,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    name: str | None = None,
    version: str = MODERN_VERSION,
    request_id: int = 1,
    headers: Sequence[tuple[str, str]] = (),
    host_header: str | None = None,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    """Send one modern per-request-envelope call and decode its JSON-RPC message."""
    body: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {**(params or {}), "_meta": _envelope(version)},
    }
    envelope_headers: list[tuple[str, str]] = [
        ("MCP-Protocol-Version", version),
        ("Mcp-Method", method),
    ]
    if name is not None:
        envelope_headers.append(("Mcp-Name", name))
    status, response_headers, text = _post(
        port, body, [*envelope_headers, *headers], host_header=host_header
    )
    return status, response_headers, _json_payload(text)


def _tool_payload(message: dict[str, Any], tool: str) -> dict[str, Any]:
    """The JSON object a tool returned, from structured content or text content."""
    result = message.get("result")
    assert isinstance(result, dict), f"{tool} produced no result: {message!r}"
    assert result.get("isError") is False, f"{tool} returned an MCP error: {result!r}"
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and structured:
        return structured
    for block in result.get("content") or ():
        text = block.get("text") if isinstance(block, dict) else None
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise AssertionError(f"{tool} carried no JSON object payload: {result!r}")


class _Server:
    """A running ``python -m aegisub_mcp.http_server`` subprocess."""

    def __init__(self, port: int, proc: subprocess.Popen[bytes], stdout: Path, stderr: Path) -> None:
        self.port = port
        self.proc = proc
        self.stdout_path = stdout
        self.stderr_path = stderr

    @property
    def stderr(self) -> str:
        return self.stderr_path.read_text(errors="replace") if self.stderr_path.exists() else ""

    @property
    def stdout(self) -> str:
        return self.stdout_path.read_text(errors="replace") if self.stdout_path.exists() else ""


@pytest.fixture(scope="module")
def http_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Server]:
    """The HTTP entry point, started with no flags at all — exactly what a user runs."""
    assert FIXTURE.is_file(), f"missing real fixture: {FIXTURE}"
    port = _free_port()
    tmp = tmp_path_factory.mktemp("http-server")
    stdout_path, stderr_path = tmp / "stdout.log", tmp / "stderr.log"

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            [sys.executable, "-m", "aegisub_mcp.http_server", "--port", str(port)],
            cwd=str(REPO_ROOT),
            env=_child_env(tmp),
            stdout=stdout,
            stderr=stderr,
        )

    server = _Server(port, proc, stdout_path, stderr_path)
    try:
        # Readiness = the server answers the modern handshake replacement, not just
        # "the socket is open": uvicorn accepts connections before the app is up.
        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        last_error = "no attempt made"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(
                    f"the HTTP server exited with {proc.returncode} during start-up\n"
                    f"stderr:\n{server.stderr[-2000:]}"
                )
            try:
                status, _, message = _modern(port, "server/discover")
            except OSError as exc:  # connection refused while uvicorn binds
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.25)
                continue
            except AssertionError as exc:
                last_error = str(exc)
                time.sleep(0.25)
                continue
            if status == 200 and "result" in message:
                print(
                    REPORT_PREFIX
                    + json.dumps(
                        {
                            "entry_point": f"{sys.executable} -m aegisub_mcp.http_server --port {port}",
                            "startup_seconds": round(START_TIMEOUT_SECONDS - (deadline - time.monotonic()), 2),
                            "supported_versions": message["result"].get("supportedVersions"),
                        },
                        sort_keys=True,
                    )
                )
                break
            last_error = f"HTTP {status}: {json.dumps(message)[:200]}"
            time.sleep(0.25)
        else:
            pytest.fail(
                f"the HTTP server never answered within {START_TIMEOUT_SECONDS:.0f}s "
                f"(last: {last_error})\nstderr:\n{server.stderr[-2000:]}"
            )
        yield server
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - unresponsive shutdown
            proc.kill()
            proc.wait(timeout=15)


# ----------------------------------------------------------------------- tests


def test_modern_discover_advertises_2026_07_28_without_a_session(http_server: _Server) -> None:
    """The 2026-07-28 handshake replacement: ``server/discover``, no session issued."""
    status, headers, message = _modern(http_server.port, "server/discover", request_id=1)

    assert status == 200, f"HTTP {status}: {message!r}"
    assert "error" not in message, message
    result = message["result"]
    assert MODERN_VERSION in result["supportedVersions"], result
    assert result["resultType"] == "complete", result
    server_info = result["_meta"]["io.modelcontextprotocol/serverInfo"]
    assert server_info["name"] == "aegisub-mcp", server_info
    assert server_info["version"], server_info
    assert {"tools", "resources", "prompts"} <= set(result["capabilities"]), result["capabilities"]
    assert (result.get("instructions") or "").strip(), "discover advertised no instructions"

    # The whole point of the modern envelope: this POST was self-contained.
    assert "mcp-session-id" not in headers, (
        "a 2026-07-28 request must not create a session: " + str(headers)
    )
    print(
        REPORT_PREFIX
        + json.dumps(
            {
                "discover": {
                    "supported_versions": result["supportedVersions"],
                    "capabilities": sorted(result["capabilities"]),
                    "server_info": server_info,
                    "response_headers": sorted(headers),
                }
            },
            sort_keys=True,
        )
    )


def test_modern_tool_call_needs_no_handshake_and_shares_process_state(http_server: _Server) -> None:
    """``tools/call`` over the per-request envelope: no ``initialize``, no session id.

    The second and third POSTs prove the *state* half of "stateless": documents
    opened by one request are visible to the next, because they live in the
    process-level workspace rather than in a session the client must carry.
    """
    opened = _modern(
        http_server.port,
        "tools/call",
        {"name": "ass_open", "arguments": {"path": str(FIXTURE), "doc_id": "http-e2e"}},
        name="ass_open",
        request_id=2,
    )
    status, headers, message = opened
    assert status == 200, f"HTTP {status}: {message!r}"
    payload = _tool_payload(message, "ass_open")
    assert payload["doc_id"] == "http-e2e", payload
    assert str(payload["path"]).endswith("basic.ass"), payload
    assert payload["lines"] == FIXTURE_LINES, payload
    assert payload["script_type"] == "v4.00+", payload
    assert "mcp-session-id" not in headers, headers

    # A brand-new document created in another request, then edited and read back.
    _, _, created = _modern(
        http_server.port,
        "tools/call",
        {
            "name": "ass_new_document",
            "arguments": {"play_res_x": 1280, "play_res_y": 720, "doc_id": "http-fresh"},
        },
        name="ass_new_document",
        request_id=3,
    )
    assert _tool_payload(created, "ass_new_document")["doc_id"] == "http-fresh", created

    _, _, added = _modern(
        http_server.port,
        "tools/call",
        {
            "name": "ass_add_line",
            "arguments": {
                "start_ms": 1000,
                "end_ms": 2500,
                "text": "added over HTTP",
                "doc_id": "http-fresh",
            },
        },
        name="ass_add_line",
        request_id=4,
    )
    assert _tool_payload(added, "ass_add_line")["index"] == 0, added

    _, _, read_back = _modern(
        http_server.port,
        "tools/call",
        {"name": "ass_list_lines", "arguments": {"doc_id": "http-fresh", "limit": None}},
        name="ass_list_lines",
        request_id=5,
    )
    lines = _tool_payload(read_back, "ass_list_lines")
    assert lines["total"] == 1 and lines["returned"] == 1, lines
    assert lines["lines"][0]["text"] == "added over HTTP", lines["lines"][0]

    # ...and the fixture document opened three requests ago is still open too.
    _, _, fixture_doc = _modern(
        http_server.port,
        "tools/call",
        {"name": "ass_list_lines", "arguments": {"doc_id": "http-e2e", "limit": None}},
        name="ass_list_lines",
        request_id=6,
    )
    fixture_lines = _tool_payload(fixture_doc, "ass_list_lines")
    assert fixture_lines["total"] == FIXTURE_LINES, fixture_lines
    assert fixture_lines["lines"][0]["text"] == r"Hello {\i1}world{\i0}", fixture_lines["lines"][0]

    print(
        REPORT_PREFIX
        + json.dumps(
            {
                "modern_tool_call": {
                    "handshake_sent": False,
                    "session_header": None,
                    "opened": {
                        k: payload[k]
                        for k in ("doc_id", "lines", "dialogue", "comments", "script_type")
                    },
                    "state_after_two_more_requests": {
                        "http-fresh": {"total": lines["total"], "text": lines["lines"][0]["text"]},
                        "http-e2e": {"total": fixture_lines["total"]},
                    },
                }
            },
            sort_keys=True,
        )
    )


def test_header_body_disagreement_is_rejected(http_server: _Server) -> None:
    """``Mcp-Method``/``Mcp-Name`` must mirror the body; disagreement is an error."""
    # (a) the method header contradicts the body's method
    status, _, message = _modern(
        http_server.port,
        "tools/list",
        {},
        request_id=7,
        headers=[],  # real headers are built by _modern from `method`; override below
    )
    assert status == 200, "control: a self-consistent tools/list must succeed"

    body = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {
            "name": "ass_list_lines",
            "arguments": {"doc_id": "http-e2e", "limit": 1},
            "_meta": _envelope(),
        },
    }
    status, _, text = _post(
        http_server.port,
        body,
        [
            ("MCP-Protocol-Version", MODERN_VERSION),
            ("Mcp-Method", "tools/list"),  # lies about the body
            ("Mcp-Name", "ass_list_lines"),
        ],
    )
    message = _json_payload(text)
    assert status == 400, f"expected HTTP 400, got {status}: {message!r}"
    assert message["error"]["code"] == -32020, message
    assert "mcp-method" in message["error"]["message"].lower(), message

    # (b) the name header contradicts the named body param
    status, _, text = _post(
        http_server.port,
        body,
        [
            ("MCP-Protocol-Version", MODERN_VERSION),
            ("Mcp-Method", "tools/call"),
            ("Mcp-Name", "ass_get_line"),  # not the tool the body calls
        ],
    )
    message = _json_payload(text)
    assert status == 400, f"expected HTTP 400, got {status}: {message!r}"
    assert message["error"]["code"] == -32020, message
    assert "mcp-name" in message["error"]["message"].lower(), message

    print(REPORT_PREFIX + json.dumps({"header_mismatch": "both arms rejected with -32020"}))


def test_missing_envelope_meta_is_rejected(http_server: _Server) -> None:
    """A modern request without the ``_meta`` envelope cannot be routed."""
    status, _, text = _post(
        http_server.port,
        {"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}},
        [("MCP-Protocol-Version", MODERN_VERSION), ("Mcp-Method", "tools/list")],
    )
    message = _json_payload(text)
    assert status == 400, f"expected HTTP 400, got {status}: {message!r}"
    assert message["error"]["code"] == -32602, message
    assert PROTOCOL_VERSION_META_KEY in message["error"]["message"], message
    print(REPORT_PREFIX + json.dumps({"missing_envelope": message["error"]["message"]}))


def test_unsupported_modern_version_names_the_supported_list(http_server: _Server) -> None:
    """A future revision is refused with the list this server actually serves."""
    status, _, message = _modern(
        http_server.port,
        "tools/list",
        {},
        version="2027-01-01",
        request_id=11,
    )
    assert status == 400, f"expected HTTP 400, got {status}: {message!r}"
    error = message["error"]
    assert error["code"] == -32022, message
    assert error["data"]["requested"] == "2027-01-01", error
    assert MODERN_VERSION in error["data"]["supported"], error
    print(REPORT_PREFIX + json.dumps({"unsupported_version": error["data"]}))


def test_rebinding_host_header_is_refused(http_server: _Server) -> None:
    """The loopback bind keeps DNS-rebinding protection: a foreign Host is refused."""
    status, _, text = _post(
        http_server.port,
        {"jsonrpc": "2.0", "id": 12, "method": "server/discover", "params": {"_meta": _envelope()}},
        [("MCP-Protocol-Version", MODERN_VERSION), ("Mcp-Method", "server/discover")],
        host_header="evil.example.com",
    )
    assert status == 421, f"expected HTTP 421, got {status}: {text[:200]!r}"
    assert "Invalid Host header" in text, text[:200]

    # control: the same request with the real Host is served
    status, _, message = _modern(http_server.port, "server/discover", request_id=13)
    assert status == 200 and "result" in message, message
    print(REPORT_PREFIX + json.dumps({"rebinding": {"foreign_host": 421, "loopback_host": 200}}))


def test_legacy_handshake_still_negotiated_on_the_same_endpoint(http_server: _Server) -> None:
    """stdio-era clients are unaffected: ``initialize`` still lands on 2025-11-25."""
    body = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "initialize",
        "params": {
            "protocolVersion": LEGACY_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "pytest-legacy-probe", "version": "0"},
        },
    }
    status, headers, text = _post(http_server.port, body)  # no MCP-Protocol-Version
    assert status == 200, f"HTTP {status}: {text[:300]!r}"
    message = _json_payload(text)
    assert "error" not in message, message
    result = message["result"]
    assert result["protocolVersion"] == LEGACY_VERSION, result
    assert result["serverInfo"]["name"] == "aegisub-mcp", result
    assert "mcp-session-id" in headers, (
        "the legacy handshake path must still open a session: " + str(headers)
    )

    # A modern-version *request* through the handshake path is counter-offered, not
    # errored — that is the SDK's negotiation, and worth pinning so the HTTP endpoint
    # is never mistaken for an upgrade of the stdio ceiling.
    counter = {**body, "id": 21}
    counter["params"] = {**body["params"], "protocolVersion": MODERN_VERSION}
    _, _, counter_text = _post(http_server.port, counter)
    counter_result = _json_payload(counter_text)["result"]
    assert counter_result["protocolVersion"] == LEGACY_VERSION, counter_result
    print(
        REPORT_PREFIX
        + json.dumps(
            {
                "legacy_handshake": {
                    "negotiated": result["protocolVersion"],
                    "session_opened": True,
                    "modern_version_offered_to_handshake": counter_result["protocolVersion"],
                }
            },
            sort_keys=True,
        )
    )


def test_http_entry_point_is_declared_as_a_console_script() -> None:
    """``pyproject.toml`` must publish the HTTP entry point, not just the stdio one."""
    tomllib = pytest.importorskip("tomllib", reason="tomllib is stdlib from Python 3.11")
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = config["project"]["scripts"]
    assert scripts["aegisub-mcp"] == "aegisub_mcp.server:main", scripts
    assert scripts["aegisub-mcp-http"] == "aegisub_mcp.http_server:main", scripts

    import aegisub_mcp.http_server as http_server

    assert Path(http_server.__file__).resolve().is_relative_to(REPO_ROOT), http_server.__file__
    assert callable(http_server.main) and callable(http_server.build_http_app)
    assert http_server.MODERN_PROTOCOL_VERSION == MODERN_VERSION, http_server.MODERN_PROTOCOL_VERSION


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["--allow-origin", "http://localhost:*"], "--allow-host"),
        (["--path", "mcp"], "must start with '/'"),
    ],
)
def test_cli_guardrails_exit_2(argv: list[str], expected: str) -> None:
    """Misconfiguration is refused up front, not served as a confusing 421/404."""
    proc = subprocess.run(
        [sys.executable, "-m", "aegisub_mcp.http_server", *argv],
        cwd=str(REPO_ROOT),
        env=_child_env(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=START_TIMEOUT_SECONDS,
    )
    stderr = proc.stderr.decode(errors="replace")
    assert proc.returncode == 2, f"exit {proc.returncode} for {argv}; stderr:\n{stderr}"
    assert expected in stderr, f"{expected!r} missing from stderr for {argv}:\n{stderr}"


def test_off_loopback_bind_warns_about_missing_rebinding_protection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``--host 0.0.0.0`` must not silently drop the protection a loopback bind has."""
    import aegisub_mcp.http_server as http_server

    with caplog.at_level(logging.WARNING, logger="aegisub_mcp.http_server"):
        http_server.warn_off_loopback_bind("0.0.0.0", None)
    assert "DNS-rebinding" in caplog.text, caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="aegisub_mcp.http_server"):
        http_server.warn_off_loopback_bind("127.0.0.1", None)  # SDK protects this one
        http_server.warn_off_loopback_bind("0.0.0.0", ["example.com"])  # caller chose
    assert caplog.text == "", caplog.text
    print(
        REPORT_PREFIX
        + json.dumps(
            {"off_loopback_warning": "warned without --allow-host, silent with it"}
        )
    )
