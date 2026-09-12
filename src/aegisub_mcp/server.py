"""MCP server entry point for aegisub-mcp.

The tool layer lives in :mod:`aegisub_mcp.tools` and is split per feature area
(``lines``, ``styles``, ``timing``, ``karaoke_tools``, ``tags_tools``,
``drawing_tools``).  Every one of those modules exposes a ``register(mcp, ws=None)``
helper and defines its public tools as module-level ``ass_*`` callables.  This
module is the missing glue: it imports all six modules, hands them a real MCP
server instance and starts the process on the stdio transport.

Running it
----------
Inside the project checkout (the package is not installed into the venv)::

    PYTHONPATH=src .venv/bin/python -m aegisub_mcp

or, once the project is installed (``pip install -e .``), through the console
script declared in ``pyproject.toml``::

    aegisub-mcp

Tool results
------------
Every tool answers with a JSON object.  The tool layer signals *expected* user
errors by raising :class:`aegisub_mcp.tools.base.ToolError`; that contract says
the server turns it into ``{"error": "<message>"}``, which :func:`wrap_tool` does
on the way in (the SDK would otherwise mask the message behind a generic
``Error executing tool <name>``).  Unexpected exceptions become the same payload,
with their traceback logged to stderr.

Protocol hygiene
----------------
In stdio mode **stdout carries MCP JSON-RPC framing and nothing else**.  All
diagnostics from this module (and from the SDK's logging) go to stderr, so a
client such as Hermes can parse the stream unmodified.  The module never calls
``print``; :func:`_harden_logging` additionally moves any pre-existing root
handler that is bound to stdout over to stderr.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import sys
from types import ModuleType
from typing import Any, Callable, Sequence

try:  # the tool layer's user-error class; the server maps it to {"error": ...}
    from .tools.base import ToolError
except Exception:  # pragma: no cover - a broken tool layer must not stop import
    ToolError = RuntimeError  # type: ignore[assignment,misc]

__all__ = [
    "SERVER_NAME",
    "SERVER_VERSION",
    "SERVER_INSTRUCTIONS",
    "TOOL_MODULES",
    "TOOL_PREFIX",
    "ToolRegistrar",
    "public_tool_functions",
    "register_all_tools",
    "wrap_tool",
    "build_server",
    "main",
]

SERVER_NAME = "aegisub-mcp"

SERVER_VERSION_FALLBACK = "0.1.0"

SERVER_INSTRUCTIONS = """\
Aegisub / ASS subtitle toolkit. Work happens on *open documents* held by the
server: call `ass_open` (path) or `ass_new_document` first, then use the other
tools against the returned `doc_id` (or the current document when `doc_id` is
omitted). Line indices are 0-based and include comments.

Feature areas: line editing and document I/O (`ass_*_line`, `ass_open`,
`ass_save`), styles and script info (`ass_*_style`, `ass_get_script_info`),
timing/QC (`ass_shift_times`, `ass_snap_to_frames`, `ass_qc`), karaoke
(`ass_karaoke_*`), override tags/typesetting (`ass_parse_text`, `ass_set_tag`,
`ass_add_typesetting`) and vector drawings/clipping/fonts (`ass_*_drawing`,
`ass_*_clip`, `ass_fonts_*`).

Every tool returns a JSON object. User errors come back as
`{"error": "<message>"}` in a successful call result.
"""

#: The tool modules that make up the server, in registration order.
TOOL_MODULES: tuple[str, ...] = (
    "aegisub_mcp.tools.lines",
    "aegisub_mcp.tools.styles",
    "aegisub_mcp.tools.timing",
    "aegisub_mcp.tools.karaoke_tools",
    "aegisub_mcp.tools.tags_tools",
    "aegisub_mcp.tools.drawing_tools",
)

#: Public tools in this project are module-level callables with this prefix.
TOOL_PREFIX = "ass_"

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- logging


def _harden_logging() -> None:
    """Make sure this process never writes diagnostics to stdout.

    Installing a root ``StreamHandler`` (default target: stderr) when none is
    configured, and repointing any handler that is bound to stdout, keeps the
    stdio transport's stdout stream free of anything but MCP framing.
    """
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler(sys.stderr))
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        if getattr(handler, "stream", None) is sys.stdout:
            handler.stream = sys.stderr  # type: ignore[attr-defined]


# ------------------------------------------------------------------- discovery


def public_tool_functions(module: ModuleType) -> dict[str, Any]:
    """Return the public ``ass_*`` callables *module* defines, in name order.

    Only callables whose name carries :data:`TOOL_PREFIX` are considered, which
    is the naming contract documented in :mod:`aegisub_mcp.tools.base`
    (``ass_*`` tools; helpers are private and start with ``_``).
    """
    found: dict[str, Any] = {}
    for name, obj in vars(module).items():
        if not name.startswith(TOOL_PREFIX):
            continue
        if not callable(obj) or isinstance(obj, type):
            continue
        found[name] = obj
    return dict(sorted(found.items()))


def _load_module(dotted: str) -> ModuleType | None:
    """Import *dotted*, returning ``None`` (with a stderr traceback) on failure."""
    try:
        return importlib.import_module(dotted)
    except Exception:  # noqa: BLE001 - a broken module must not hide the others
        log.exception("could not import tool module %s — its tools will be missing", dotted)
        return None


# ------------------------------------------------------- tool-result contract


def _tool_error_payload(exc: BaseException) -> dict[str, str]:
    """The documented user-error payload: ``{"error": "<message>"}``."""
    return {"error": str(exc).strip() or exc.__class__.__name__}


def _tool_description(fn: Any) -> str | None:
    """Full (dedented) docstring of *fn*, or ``None`` when it has none."""
    try:
        doc = inspect.getdoc(fn)
    except Exception:  # noqa: BLE001 - a weird callable must not break registration
        return None
    return doc or None


def wrap_tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return *fn* wrapped so a call always answers with a JSON object.

    :mod:`aegisub_mcp.tools.base` states the contract: *every tool returns a
    JSON-serialisable dict; user errors raise* :class:`ToolError` *and the server
    turns that into* ``{"error": "..."}``.  Without this wrapper the SDK treats a
    :class:`ToolError` as an unanticipated crash, so the model would only ever
    see ``Error executing tool <name>`` with the real message stranded on the
    server's stderr.

    The wrapper:

    * converts the tool layer's ``ToolError`` into ``{"error": "<message>"}``
      (logged at INFO, no traceback — it is an expected user error);
    * converts any other exception into the same payload but logs its traceback
      at ERROR, because a crash must still be debuggable;
    * re-raises MCP protocol errors and the SDK's own ``ToolError`` untouched so
      the SDK's handling of them stays intact.

    ``functools.wraps`` keeps the original signature visible: the SDK builds each
    tool's input schema with ``inspect.signature(fn, eval_str=True)``, which
    follows ``__wrapped__`` down to the genuine ``ass_*`` function, so the schema
    is identical to the unwrapped one.
    """
    from mcp.server.mcpserver.exceptions import ToolError as SDKToolError
    from mcp.shared.exceptions import MCPError

    passthrough = (SDKToolError, MCPError)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except passthrough:
                raise
            except ToolError as exc:
                log.info("%s: %s", getattr(fn, "__name__", fn), exc)
                return _tool_error_payload(exc)
            except Exception as exc:  # noqa: BLE001
                log.exception("unhandled error in tool %s", getattr(fn, "__name__", fn))
                return _tool_error_payload(exc)

        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except passthrough:
            raise
        except ToolError as exc:
            log.info("%s: %s", getattr(fn, "__name__", fn), exc)
            return _tool_error_payload(exc)
        except Exception as exc:  # noqa: BLE001
            log.exception("unhandled error in tool %s", getattr(fn, "__name__", fn))
            return _tool_error_payload(exc)

    return wrapper


class ToolRegistrar:
    """Stands in for the MCP server and wraps every tool on its way in.

    The feature modules publish tools through a FastMCP-style ``mcp.tool()``
    decorator (see each module's ``register()``) and mostly ignore the decorator's
    return value, so the result contract cannot be applied after the fact — it has
    to happen here, between the module and the server.  The adapter mirrors the
    server's surface that the modules use (``tool``, ``add_tool``, ``name``).
    """

    def __init__(self, server: Any) -> None:
        self.server = server
        self.name = getattr(server, "name", SERVER_NAME)
        #: tool name -> the real ``ass_*`` function now serving it
        self.wrapped: dict[str, Any] = {}

    def tool(self, *, name: str | None = None, **options: Any) -> Callable[[Any], Any]:
        """FastMCP-compatible decorator: register *fn* and hand it straight back."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.add_tool(fn, name=name, **options)
            return fn

        return decorator

    def add_tool(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        title: str | None = None,
        annotations: Any = None,
        icons: Any = None,
        meta: Any = None,
        structured_output: bool | None = None,
        **extra: Any,
    ) -> Any:
        """Register *fn* under *name* (default: its own name), wrapped."""
        tool_name = name or getattr(fn, "__name__", None)
        if not tool_name:
            raise TypeError(f"cannot register an unnamed tool: {fn!r}")
        if extra:
            log.warning("ignoring unsupported tool option(s) for %s: %s", tool_name, ", ".join(sorted(extra)))
        options: dict[str, Any] = {
            "name": tool_name,
            "description": _tool_description(fn) if description is None else description,
        }
        for key, value in (
            ("title", title),
            ("annotations", annotations),
            ("icons", icons),
            ("meta", meta),
            ("structured_output", structured_output),
        ):
            if value is not None:
                options[key] = value
        self.wrapped[tool_name] = fn
        return self.server.add_tool(wrap_tool(fn), **options)


# ---------------------------------------------------------------- registration


def _register_via_module(registrar: ToolRegistrar, module: ModuleType) -> list[str] | None:
    """Call the module's own ``register()`` helper through *registrar*.

    Returns the names it reports, or ``None`` when the module has no usable
    ``register`` helper / it raised (in which case the caller falls back to
    registering the public tools directly).
    """
    register = getattr(module, "register", None)
    if not callable(register):
        return None
    attempts: tuple[tuple[Any, ...], ...] = ((registrar, None), (registrar,))
    for args in attempts:  # `ws` is optional in most modules
        try:
            names = register(*args)
        except TypeError:
            continue
        except Exception:  # noqa: BLE001
            log.exception("%s.register() failed part-way", module.__name__)
            return []
        if names:
            return sorted(str(name) for name in names)
        return []
    log.warning("%s.register() is not callable as register(mcp[, ws])", module.__name__)
    return None


def _register_directly(registrar: ToolRegistrar, module: ModuleType, names: Sequence[str]) -> list[str]:
    """Register the remaining public tools straight from the module namespace."""
    registered: list[str] = []
    for name in names:
        fn = getattr(module, name, None)
        if fn is None:
            continue
        try:
            registrar.add_tool(fn, name=name)
        except Exception:  # noqa: BLE001
            log.exception("could not register tool %s from %s", name, module.__name__)
            continue
        registered.append(name)
    return sorted(registered)


def register_all_tools(server: Any, module_names: Sequence[str] = TOOL_MODULES) -> dict[str, list[str]]:
    """Register every public tool of every module with *server*.

    Each module's own ``register()`` helper is used first; any public tool it
    left out is then registered directly, so the server exposes the full
    ``ass_*`` surface even if a module's helper is incomplete.  Every tool passes
    through a :class:`ToolRegistrar`, which applies the ``{"error": ...}`` result
    contract (see :func:`wrap_tool`).

    Returns a mapping of module name to the tool names registered for it, and
    logs a summary plus every discrepancy to stderr.
    """
    registrar = server if isinstance(server, ToolRegistrar) else ToolRegistrar(server)
    report: dict[str, list[str]] = {}
    for dotted in module_names:
        module = _load_module(dotted)
        if module is None:
            report[dotted] = []
            continue

        expected = public_tool_functions(module)
        reported = _register_via_module(registrar, module)

        if reported is None:
            names = _register_directly(registrar, module, list(expected))
        else:
            missing = [name for name in expected if name not in set(reported)]
            unknown = [name for name in reported if name not in expected]
            if missing:
                log.warning(
                    "%s.register() did not report %d public tool(s): %s"
                    " — registering them directly",
                    dotted,
                    len(missing),
                    ", ".join(missing),
                )
                names = sorted(set(reported) | set(_register_directly(registrar, module, missing)))
            else:
                names = reported
            if unknown:
                log.warning(
                    "%s.register() reported %d name(s) that are not public tools: %s",
                    dotted,
                    len(unknown),
                    ", ".join(unknown),
                )

        imported = sorted(name for name in expected if getattr(expected[name], "__module__", dotted) != dotted)
        if imported:
            log.info("%s also matches imported callables: %s", dotted, ", ".join(imported))

        report[dotted] = names
        log.info("%s: registered %d tool(s)", dotted, len(names))

    total = sum(len(names) for names in report.values())
    log.info(
        "registered %d tool(s) from %d module(s) with server %r",
        total,
        len(report),
        registrar.name,
    )
    return report


# --------------------------------------------------------------------- server


def _server_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("aegisub-mcp")
        except PackageNotFoundError:
            return SERVER_VERSION_FALLBACK
    except Exception:  # noqa: BLE001 - metadata must never stop the server
        return SERVER_VERSION_FALLBACK


#: version of this installation (see :func:`_server_version`); listed in ``__all__``
SERVER_VERSION = _server_version()


def build_server(
    *,
    name: str = SERVER_NAME,
    version: str | None = None,
    instructions: str = SERVER_INSTRUCTIONS,
    log_level: str | None = None,
) -> Any:
    """Build a ready-to-run :class:`mcp.server.mcpserver.MCPServer`.

    Raises :class:`ImportError` when the MCP SDK is not installed; a tool module
    that cannot be imported is reported on stderr and simply contributes no
    tools, so a single broken module cannot take the whole server down.
    """
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the `mcp` Python SDK is required to run the aegisub-mcp server "
            "(pip install 'mcp>=2.2'); see https://py.sdk.modelcontextprotocol.io/"
        ) from exc

    _harden_logging()

    options: dict[str, Any] = {
        "name": name,
        "version": version or _server_version(),
        "instructions": instructions,
    }
    if log_level is not None:
        options["log_level"] = log_level
    server = MCPServer(**options)
    register_all_tools(server)
    return server


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aegisub-mcp server on stdio (entry point of ``python -m aegisub_mcp``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        sys.stderr.write(
            f"{SERVER_NAME}: unexpected argument(s): {' '.join(args)}; "
            "this server takes no arguments and speaks MCP on stdin/stdout\n"
        )
        return 2

    _harden_logging()
    try:
        server = build_server()
    except Exception as exc:  # noqa: BLE001 - report, never traceback into stdout
        log.error("could not start %s: %s", SERVER_NAME, exc)
        return 1

    try:
        server.run("stdio")
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        return 130
    except Exception:  # noqa: BLE001
        log.exception("%s serving on stdio failed", SERVER_NAME)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - `python -m aegisub_mcp.server`
    raise SystemExit(main())
