"""``python -m aegisub_mcp`` — start the aegisub-mcp server on the stdio transport.

The package is deliberately thin: :mod:`aegisub_mcp.server` owns the server
construction and tool registration, this module only makes the package
runnable as ``python -m aegisub_mcp`` (what a Hermes ``mcp_servers`` entry
invokes).  stdout is reserved for MCP framing; all diagnostics go to stderr.
"""

from __future__ import annotations

import sys

from .server import main

if __name__ == "__main__":
    sys.exit(main())
