"""Streamable-HTTP entry point for aegisub-mcp — the transport that serves protocol 2026-07-28.

The stdio entry point (:mod:`aegisub_mcp.server`) is stuck below this revision by
design: a stdio server is always on the ``initialize`` handshake path, whose last
rung is ``2025-11-25``, so asking it for ``2026-07-28`` gets a counter-offer rather
than the modern protocol.  ``2026-07-28`` is a *stateless, per-request* protocol —
no handshake, no session, one self-contained JSON-RPC POST in and one response
out — and it is only reachable over HTTP.  This module serves that HTTP transport.

Serving one endpoint, two eras
------------------------------
``server.streamable_http_app()`` wires the SDK's era routing for us.  Whether a
request is modern or legacy is decided by its ``MCP-Protocol-Version`` header:
values that are not one of the handshake revisions (``2024-11-05`` … ``2025-11-25``)
go to the modern single-exchange handler, and everything else keeps the classic
session behaviour.  So a single ``POST /mcp`` endpoint answers both:

* **modern** — ``params._meta`` carries
  ``io.modelcontextprotocol/protocolVersion`` (``2026-07-28``) and
  ``io.modelcontextprotocol/clientCapabilities``, mirrored into the
  ``MCP-Protocol-Version`` / ``Mcp-Method`` / ``Mcp-Name`` headers.  The handshake
  method here is ``server/discover``, and every request is independent.
* **legacy** — clients that never send ``MCP-Protocol-Version`` (or send a
  handshake revision) are served exactly as before, ``initialize`` included.

Because the request envelope is per-request, documents live in the process-level
:data:`aegisub_mcp.tools.base.workspace` singleton, not in a session: a document
opened by one POST is still open for the next one, with no ``Mcp-Session-Id``
required (and none issued on the modern path).

Running it
----------
Inside the project checkout (the package is not installed into the venv)::

    PYTHONPATH=src .venv/bin/python -m aegisub_mcp.http_server --port 8000

or, once the project is installed, through the console script::

    aegisub-mcp-http --host 127.0.0.1 --port 8000

Serving is loopback-only by default.  Binding elsewhere is allowed, but the
DNS-rebinding protection the SDK enables for loopback binds is *not* automatic
off-loopback: pass ``--allow-host`` (and optionally ``--allow-origin``) when the
endpoint is reachable by anything other than this machine.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Sequence

from .server import SERVER_NAME, build_server

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_PATH",
    "MODERN_PROTOCOL_VERSION",
    "build_http_app",
    "parse_args",
    "main",
]

#: Interface the HTTP transport binds to when nothing else is requested.
DEFAULT_HOST = "127.0.0.1"

#: Port the HTTP transport binds to when nothing else is requested.
DEFAULT_PORT = 8000

#: Path the streamable-HTTP endpoint is mounted at.
DEFAULT_PATH = "/mcp"

#: The stateless per-request-envelope revision; see the module docstring.
MODERN_PROTOCOL_VERSION = "2026-07-28"

#: Binds for which the SDK auto-enables DNS-rebinding protection.
LOOPBACK_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- arguments


def _env_port() -> int:
    raw = os.environ.get("AEGISUB_MCP_HTTP_PORT")
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{SERVER_NAME}-http: AEGISUB_MCP_HTTP_PORT must be an integer, got {raw!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the HTTP entry point's command line (``--help`` for the full list)."""
    parser = argparse.ArgumentParser(
        prog=f"{SERVER_NAME}-http",
        description=(
            "Serve aegisub-mcp over streamable HTTP. One endpoint answers both "
            f"protocol eras: the modern stateless envelope ({MODERN_PROTOCOL_VERSION}) and "
            "the legacy initialize-handshake revisions (2024-11-05 ... 2025-11-25)."
        ),
        epilog=(
            "Environment: AEGISUB_MCP_HTTP_HOST, AEGISUB_MCP_HTTP_PORT set the defaults "
            "for --host and --port."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("AEGISUB_MCP_HTTP_HOST", DEFAULT_HOST),
        help=f"interface to bind (default: {DEFAULT_HOST}; env AEGISUB_MCP_HTTP_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_port(),
        help=f"port to bind (default: {DEFAULT_PORT}; env AEGISUB_MCP_HTTP_PORT)",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"path the endpoint is served at (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--json-response",
        action="store_true",
        help=(
            "answer legacy-handshake requests with one application/json body instead of "
            "an SSE stream (the modern envelope is one JSON-RPC response either way)"
        ),
    )
    parser.add_argument(
        "--stateless",
        dest="stateless_http",
        action="store_true",
        help=(
            "give legacy-handshake clients a fresh transport per request instead of a "
            "session (Mcp-Session-Id); the modern envelope is always per-request"
        ),
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        dest="allowed_hosts",
        metavar="PATTERN",
        help=(
            "Host header value to accept, e.g. '127.0.0.1:*' (repeatable). Passing this "
            "enables DNS-rebinding protection for any bind, not just loopback"
        ),
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        dest="allowed_origins",
        metavar="PATTERN",
        help=(
            "Origin header value to accept, e.g. 'http://localhost:*' (repeatable). "
            "Requires --allow-host; clients that send no Origin header are unaffected"
        ),
    )
    args = parser.parse_args(argv)
    if not args.path.startswith("/"):
        parser.error(f"--path must start with '/', got {args.path!r}")
    if not args.allowed_hosts and args.allowed_origins:
        parser.error(
            "--allow-origin needs --allow-host as well: with protection enabled and no "
            "allowed host, every request is rejected with 'Invalid Host header' (HTTP 421)"
        )
    return args


def warn_off_loopback_bind(host: str, allowed_hosts: Sequence[str] | None) -> None:
    """Warn when the endpoint is reachable off-machine with no rebinding protection.

    The SDK protects loopback binds by itself, but an off-loopback bind with no
    named allowed hosts accepts *any* ``Host`` header -- a silent security change
    that a caller who typed ``--host 0.0.0.0`` deserves to see in the log.
    """
    if allowed_hosts or host in LOOPBACK_HOSTS:
        return
    log.warning(
        "binding %s: the SDK's DNS-rebinding protection is off for off-loopback binds, "
        "so every Host header is accepted; pass --allow-host when this endpoint is "
        "reachable by anything but this machine",
        host,
    )


def _transport_security(allowed_hosts: Sequence[str] | None, allowed_origins: Sequence[str] | None) -> Any:
    """Build the SDK's transport-security settings, or ``None`` for its defaults.

    ``None`` means "let the SDK decide": protection is auto-enabled when the bind is
    loopback and left off otherwise.  Naming hosts or origins explicitly always turns
    protection on.
    """
    if allowed_hosts is None and allowed_origins is None:
        return None
    from mcp.server.transport_security import TransportSecuritySettings

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(allowed_hosts or []),
        allowed_origins=list(allowed_origins or []),
    )


# ------------------------------------------------------------------- serving


def build_http_app(
    *,
    host: str = DEFAULT_HOST,
    path: str = DEFAULT_PATH,
    json_response: bool = False,
    stateless_http: bool = False,
    allowed_hosts: Sequence[str] | None = None,
    allowed_origins: Sequence[str] | None = None,
) -> Any:
    """Return the streamable-HTTP ASGI app (a Starlette app) for this server.

    Exposed separately from :func:`main` so the app can be mounted in-process — under
    an existing ASGI server, or in a test — instead of being bound to a port.  The
    app's lifespan must run (``uvicorn`` does that); it starts the session manager
    that both the modern and the legacy paths dispatch through.
    """
    server = build_server()
    return server.streamable_http_app(
        streamable_http_path=path,
        json_response=json_response,
        stateless_http=stateless_http,
        host=host,
        transport_security=_transport_security(allowed_hosts, allowed_origins),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run aegisub-mcp on the streamable-HTTP transport (console script entry point)."""
    args = parse_args(argv)

    try:
        server = build_server()
    except Exception as exc:  # noqa: BLE001 - report, never traceback at startup
        log.error("could not start %s: %s", SERVER_NAME, exc)
        return 1

    log.info(
        "%s serving MCP over streamable HTTP on http://%s:%d%s "
        "(modern %s envelope + legacy handshake revisions)",
        SERVER_NAME,
        args.host,
        args.port,
        args.path,
        MODERN_PROTOCOL_VERSION,
    )

    warn_off_loopback_bind(args.host, args.allowed_hosts)

    try:
        server.run(
            "streamable-http",
            host=args.host,
            port=args.port,
            streamable_http_path=args.path,
            json_response=args.json_response,
            stateless_http=args.stateless_http,
            transport_security=_transport_security(args.allowed_hosts, args.allowed_origins),
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        return 130
    except Exception:  # noqa: BLE001 - report, never traceback out of main
        log.exception("%s serving on streamable HTTP failed", SERVER_NAME)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - `python -m aegisub_mcp.http_server`
    sys.exit(main())
