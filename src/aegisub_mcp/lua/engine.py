"""Automation 4 Lua runner for aegisub-mcp.

Runs *real* Aegisub automation scripts headlessly on a byte-faithful document,
with Automation 4 semantics:

* LuaJIT 2.1 (same VM family Aegisub ships), Lua 5.1 language level,
  byte-transparent strings (Lua sees UTF-8 bytes, exactly like in Aegisub).
* ``subs`` exposes **every** line of the file, in file order, tagged with a
  ``class`` of ``"info"`` / ``"style"`` / ``"dialogue"`` / ``"unknown"``.
* ``subs[i]`` returns a **copy** of the line; writing it back requires
  ``subs[i] = line`` (this is how Aegisub behaves and how scripts are written).
* Structural editing through ``subs[0] = line``, ``subs[-i] = line``,
  ``subs[i] = nil``, ``subs.append``, ``subs.insert``, ``subs.delete``,
  ``subs.deleterange`` and ``subs.n`` / ``#subs``.
* Macros, filters (with priority ordering), ``include``/``require`` against the
  automation include directories (karaskel, utils, cleantags, unicode ship
  vendored), ``aegisub.progress``, ``aegisub.debug``, ``aegisub.set_undo_point``,
  ``aegisub.cancel``, ``aegisub.dialog``, ``aegisub.text_extents``,
  ``aegisub.keyframes``, ``aegisub.project_properties``, ``aegisub.re``,
  ``aegisub.util`` and the ``unicode`` module.

Anything an Aegisub script uses that is *not* implemented fails loudly with a
descriptive error (never a silent ``nil``) and is recorded in the result.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re as _pyre
import tempfile
import time
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import typing

import lupa.luajit21 as lupa

from ..asscore import assutil as U
from ..asscore import fontmetrics as FM
from ..asscore.document import (
    KIND_EVENTS,
    KIND_INFO,
    KIND_STYLES,
    AssDocument,
    DataEntry,
    Entry,
    EventEntry,
    RawEntry,
    Section,
    StyleEntry,
    canonical_style_key,
)

__all__ = [
    "LuaEngine",
    "MacroResult",
    "LuaError",
    "MacroNotFound",
    "DialogUnavailable",
    "RegisteredItem",
    "to_lua_str",
    "from_lua_str",
    "include_dirs_default",
    "run_script_cli",
]

PKG_DIR = Path(__file__).resolve().parent
PRELUDE_PATH = PKG_DIR / "prelude.lua"
VENDORED_INCLUDE = PKG_DIR / "include"

CANCEL_MARKER = "__AEGISUB_MCP_CANCEL__"

# Automation 4 line classes ---------------------------------------------------

CLASS_INFO = "info"
CLASS_STYLE = "style"
CLASS_DIALOGUE = "dialogue"
CLASS_UNKNOWN = "unknown"


class LuaError(RuntimeError):
    """A Lua script raised an error (message carries the Lua traceback)."""


class MacroNotFound(LuaError):
    """The requested macro/filter is not registered by the loaded script."""


class DialogUnavailable(LuaError):
    """A dialog was requested but the engine runs headless and has no answers."""


# ---------------------------------------------------------------------------
# string conversion (byte transparency)
# ---------------------------------------------------------------------------


def to_lua_str(text: str) -> str:
    """UTF-8 bytes of *text* reinterpreted as latin-1 for a Lua string."""
    return text.encode("utf-8", "surrogateescape").decode("latin-1")


def from_lua_str(text: str) -> str:
    """Lua string (UTF-8 bytes as latin-1) decoded back to a Python string."""
    return text.encode("latin-1", "surrogateescape").decode("utf-8", "surrogateescape")


def _table_items(value: Any) -> Iterable[tuple[Any, Any]]:
    try:
        items = value.items()
    except Exception:
        return []
    try:
        return list(items)
    except Exception:
        return []


def lua_to_py(value: Any) -> Any:
    """Convert a Lua value (or table) into plain Python data."""
    if isinstance(value, str):
        return from_lua_str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "surrogateescape")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if lupa.lua_type(value) == "table":
        items = _table_items(value)
        if not items:
            return {}
        numeric = [(k, v) for k, v in items if isinstance(k, (int, float)) and float(k).is_integer()]
        if len(numeric) == len(items):
            keys = sorted(int(k) for k, _ in numeric)
            if keys == list(range(1, len(keys) + 1)):
                return [lua_to_py(value[i]) for i in keys]
        return {lua_to_py(k) if isinstance(k, str) else k: lua_to_py(v) for k, v in items}
    return value


def py_to_lua(value: Any) -> Any:
    """Convert plain Python data into Lua-friendly values (str -> bytes)."""
    if isinstance(value, str):
        return to_lua_str(value)
    if isinstance(value, tuple):
        return tuple(py_to_lua(item) for item in value)
    if isinstance(value, dict):
        return {py_to_lua(k): py_to_lua(v) for k, v in value.items()}
    if isinstance(value, (list, set, frozenset)):
        return [py_to_lua(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return value


def include_dirs_default() -> list[Path]:
    """Automation include directories, vendored one first."""
    dirs = [VENDORED_INCLUDE]
    config_home = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    for candidate in (
        config_home / "aegisub" / "automation" / "include",
        Path.home() / ".aegisub" / "automation" / "include",
        data_home / "aegisub" / "automation" / "include",
    ):
        if candidate not in dirs:
            dirs.append(candidate)
    return dirs


# ---------------------------------------------------------------------------
# field tables
# ---------------------------------------------------------------------------

# A4 dialogue fields -> (ASS column, kind)
DIALOGUE_SPEC: dict[str, tuple[str, str]] = {
    "layer": ("Layer", "int"),
    "start_time": ("Start", "time"),
    "end_time": ("End", "time"),
    "style": ("Style", "str"),
    "actor": ("Name", "str"),
    "margin_l": ("MarginL", "int"),
    "margin_r": ("MarginR", "int"),
    "margin_v": ("MarginV", "int"),
    "margin_t": ("MarginV", "int"),
    "margin_b": ("MarginV", "int"),
    "effect": ("Effect", "str"),
    "text": ("Text", "raw"),
    "comment": ("Comment", "bool"),
}

# A4 style fields -> (ASS column, kind)
STYLE_SPEC: dict[str, tuple[str, str]] = {
    "name": ("Name", "str"),
    "fontname": ("Fontname", "str"),
    "fontsize": ("Fontsize", "num"),
    "color1": ("PrimaryColour", "str"),
    "color2": ("SecondaryColour", "str"),
    "color3": ("OutlineColour", "str"),
    "color4": ("BackColour", "str"),
    "bold": ("Bold", "bool"),
    "italic": ("Italic", "bool"),
    "underline": ("Underline", "bool"),
    "strikeout": ("StrikeOut", "bool"),
    "scale_x": ("ScaleX", "num"),
    "scale_y": ("ScaleY", "num"),
    "spacing": ("Spacing", "num"),
    "angle": ("Angle", "num"),
    "borderstyle": ("BorderStyle", "int"),
    "outline": ("Outline", "num"),
    "shadow": ("Shadow", "num"),
    "align": ("Alignment", "int"),
    "margin_l": ("MarginL", "int"),
    "margin_r": ("MarginR", "int"),
    "margin_t": ("MarginV", "int"),
    "margin_b": ("MarginV", "int"),
    "margin_v": ("MarginV", "int"),
    "encoding": ("Encoding", "int"),
}

_SCRIPT_HEADER = "[Script Info]"
_STYLE_HEADERS = ("[V4+ Styles]", "[V4 Styles]", "[V4++ Styles]")
_EVENT_HEADER = "[Events]"

V4P_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
    "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
    "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
V4_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, "
    "Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding"
)
EVENT_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


def _num_str(value: float) -> str:
    if isinstance(value, bool):
        return "-1" if value else "0"
    number = float(value)
    if math.isfinite(number) and number == int(number):
        return str(int(number))
    return f"{number:g}"


def _int_of(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return -1 if value else 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip() or default)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default


def _float_of(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return -1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _time_value(value: Any, default: int = 0) -> int:
    """Milliseconds for an A4 ``start_time``/``end_time`` field.

    A4 semantics: the field is a number of milliseconds.  Documents are read
    back as their literal ASS timestamp text, so accept both forms and fall
    back to ``default`` for junk (malformed lines must not explode a macro).
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    if value is None:
        return default
    try:
        return U.parse_time(str(value).strip())
    except Exception:
        return _int_of(value, default)


def _bool_of(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip()
    if not text:
        return False
    if text.lower() in {"false", "nil", "no"}:
        return False
    return _int_of(text, 0) != 0


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class RegisteredItem:
    """A macro or filter registered by a loaded script."""

    name: str
    description: str
    kind: str  # "macro" | "filter"
    priority: int
    order: int
    fn: Any = None
    script: str | None = None
    # Automation 4 optional registration arguments: the macro validation
    # function (4th argument of register_macro) and the filter's options window
    # provider (5th argument of register_filter, GUI-only).
    is_valid: Any = None
    options_provider: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "priority": self.priority,
            "script": self.script,
        }


@dataclass
class MacroResult:
    """Outcome of one macro/filter run."""

    name: str
    kind: str
    ok: bool
    error: str | None = None
    output: list[dict[str, Any]] = _dc_field(default_factory=list)
    progress_title: str | None = None
    progress_task: str | None = None
    progress: float | None = None
    undo_points: list[str] = _dc_field(default_factory=list)
    dialogs: list[dict[str, Any]] = _dc_field(default_factory=list)
    # Dialogs the script opened that had no configured answer.  Non-empty means
    # the script was interrupted by an unattended dialog, so ``ok`` is False.
    dialog_failures: list[str] = _dc_field(default_factory=list)
    unsupported: list[str] = _dc_field(default_factory=list)
    elapsed_ms: float = 0.0
    lines_before: int = 0
    lines_after: int = 0
    changed_lines: list[int] = _dc_field(default_factory=list)
    structure_changed: bool = False
    selection: list[int] = _dc_field(default_factory=list)
    # File index of the active line after the run (0 = none).  A macro may
    # return a new active line index; Aegisub moves the GUI selection to it.
    active_line: int = 0
    cancelled: bool = False
    dry_run: bool = False

    @property
    def log(self) -> str:
        return "".join(item["message"] for item in self.output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "ok": self.ok,
            "error": self.error,
            "log": self.log,
            "messages": self.output,
            "progress_title": self.progress_title,
            "progress_task": self.progress_task,
            "progress": self.progress,
            "undo_points": self.undo_points,
            "dialogs": self.dialogs,
            "dialog_failures": self.dialog_failures,
            "unsupported": self.unsupported,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "lines_before": self.lines_before,
            "lines_after": self.lines_after,
            "changed_lines": self.changed_lines,
            "structure_changed": self.structure_changed,
            "selection": self.selection,
            "active_line": self.active_line,
            "cancelled": self.cancelled,
            "dry_run": self.dry_run,
        }


# ---------------------------------------------------------------------------
# subs handle
# ---------------------------------------------------------------------------


class SubsHandle:
    """Python side of the Lua ``subtitles`` object (1-based, A4 semantics)."""

    def __init__(self, engine: LuaEngine):
        self.engine = engine
        self.doc = engine.doc

    # -- helpers -----------------------------------------------------------
    @property
    def _lines(self) -> list[tuple[Section | None, Entry]]:
        return self.doc.all_lines()

    def _entry_class(self, section: Section | None, entry: Entry) -> str:
        return self.engine.line_class(section, entry)

    # -- reading -----------------------------------------------------------
    def count(self) -> int:
        return len(self._lines)

    def get(self, index: int) -> Any:
        index = _int_of(index, 1)
        lines = self._lines
        position = index - 1
        if position < 0 or position >= len(lines):
            return None
        section, entry = lines[position]
        table = self.engine.line_table(section, entry)
        return self.engine.lua_value(table)

    def class_of(self, index: int) -> str | None:
        lines = self._lines
        position = _int_of(index, 1) - 1
        if position < 0 or position >= len(lines):
            return None
        section, entry = lines[position]
        return self._entry_class(section, entry)

    # -- writing -----------------------------------------------------------
    def set(self, index: int, table: Any) -> None:
        payload = lua_to_py(table)
        if not isinstance(payload, dict):
            raise LuaError(f"subs[{index}] = value: expected a line table, got {type(payload).__name__}")
        position = _int_of(index, 1) - 1
        lines = self._lines
        if position < 0 or position >= len(lines):
            return
        section, entry = lines[position]
        self.engine.apply_line_table(section, entry, payload, position=position)

    def append(self, table: Any) -> None:
        payload = lua_to_py(table)
        if not isinstance(payload, dict):
            raise LuaError("subs.append(line): expected a line table")
        self.engine.create_line(payload, position=None)

    def append_many(self, tables: Any) -> None:
        for table in lua_to_py(tables) or []:
            self.append(table)

    def insert(self, index: int, table: Any) -> None:
        payload = lua_to_py(table)
        if not isinstance(payload, dict):
            raise LuaError("subs.insert(i, line): expected a line table")
        self.engine.create_line(payload, position=max(_int_of(index, 1), 1))

    def insert_many(self, index: int, tables: Any) -> None:
        base = max(_int_of(index, 1), 1)
        for offset, table in enumerate(lua_to_py(tables) or []):
            self.insert(base + offset, table)

    def delete(self, index: int, count: int = 1) -> None:
        start = _int_of(index, 1)
        amount = max(_int_of(count, 1), 1)
        self.doc.remove_lines(range(start - 1, start - 1 + amount))

    def delete_range(self, first: int, last: int) -> None:
        start = max(_int_of(first, 1), 1)
        end = _int_of(last, start)
        if end < start:
            start, end = end, start
        self.doc.remove_lines(range(start - 1, end))


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class LuaEngine:
    """Loads Aegisub automation scripts and runs their macros/filters."""

    def __init__(
        self,
        doc: AssDocument,
        *,
        selection: Sequence[int] | None = None,
        config: dict[str, Any] | None = None,
        video: str | None = None,
        audio: str | None = None,
        video_size: Sequence[int] | None = None,
        keyframes: Sequence[int] | None = None,
        fps: float | None = None,
        project_path: str | None = None,
        timecodes: Sequence[float] | None = None,
        automation_scripts: Sequence[str] | None = None,
        include_dirs: Sequence[str | os.PathLike[str]] | None = None,
        dialog_mode: str = "raise",
        dialog_answers: dict[str, Any] | None = None,
        interactive: bool = False,
        log_limit: int = 5000,
    ):
        self.doc = doc
        self.selection = [int(i) for i in (selection or [])]
        self.config = dict(config or {})
        self.video = video
        self.audio = audio
        self.video_size = tuple(int(v) for v in video_size) if video_size else None
        self.keyframes = [int(k) for k in keyframes] if keyframes is not None else None
        self.fps = float(fps) if fps is not None else None
        self.project_path = project_path
        self.timecodes = list(timecodes) if timecodes else None
        self.automation_scripts = list(automation_scripts or [])
        self.include_dirs = [Path(p) for p in (include_dirs or include_dirs_default())]
        self.dialog_mode = dialog_mode if interactive else ("raise" if dialog_mode == "interactive" else dialog_mode)
        self.dialog_answers = dict(dialog_answers or {})
        self.log_limit = log_limit

        self.macros: list[RegisteredItem] = []
        self.filters: list[RegisteredItem] = []
        self.scripts: list[str] = []
        self.unsupported: list[str] = []
        self.output: list[dict[str, Any]] = []
        self.undo_points: list[str] = []
        self.dialogs: list[dict[str, Any]] = []
        # Dialogs the script opened but that had no configured answer.  They are
        # *not* raised across the Lua boundary: a Python exception escaping into
        # lupa inside xpcall surfaces as a useless "error in error handling", so
        # the run continues as if the user closed the dialog and the messages are
        # folded into the final result by :meth:`_finish`.
        self.dialog_failures: list[str] = []
        self.progress_title: str | None = None
        self.progress_task: str | None = None
        self.progress: float | None = None
        self._cancelled = False
        self._include_cache: dict[str, Any] = {}
        self._handle = SubsHandle(self)
        self._registration_order = 0

        self.lua = lupa.LuaRuntime(unpack_returned_tuples=True, encoding="latin-1")
        self._install_host()
        self._load_prelude()
        self._install_state()

    # -- Lua ↔ Python plumbing --------------------------------------------
    def _wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def inner(*args: Any) -> Any:
            return self._lua_result(fn(*[lua_to_py(arg) for arg in args]))

        return inner

    def _lua_result(self, value: Any) -> Any:
        """Convert an ``aegisub.*`` return value into something Lua can use.

        Tuples stay tuples: ``lupa``'s ``unpack_returned_tuples`` turns them
        into *multiple* Lua return values (``aegisub.text_extents``,
        ``aegisub.video_size``, ...).  Dicts and lists become real Lua tables --
        handing Lua a raw Python container makes lupa expose it as *userdata*,
        which silently breaks ``#t``, ``t[i]``, ``ipairs``/``pairs`` and every
        macro that walks the value (this is what killed
        ``aegisub.parse_karaoke_data`` inside karaskel).
        """
        if isinstance(value, tuple):
            return tuple(self._lua_result(item) for item in value)
        if isinstance(value, (dict, list)):
            return self.lua_value(value)
        if isinstance(value, str):
            return to_lua_str(value)
        return value

    def lua_value(self, value: Any) -> Any:
        """Build a *real* Lua table/value from Python data."""
        if isinstance(value, dict):
            table = self.lua.table()
            for key, item in value.items():
                table[py_to_lua(key)] = self.lua_value(item)
            return table
        if isinstance(value, (list, tuple)):
            table = self.lua.table()
            for index, item in enumerate(value, start=1):
                table[index] = self.lua_value(item)
            return table
        if isinstance(value, str):
            return to_lua_str(value)
        return value

    def _install_host(self) -> None:
        host = self.lua.table()
        for name, fn in {
            "__host_log": self._h_log,
            "__host_progress": self._h_progress,
            "__host_is_cancelled": self._h_is_cancelled,
            "__host_cancel": self._h_cancel,
            "__host_undo_point": self._h_undo_point,
            "__host_include": self._h_include,
            "__host_require": self._h_require,
            "__host_re_find": self._h_re_find,
            "__host_re_match": self._h_re_match,
            "__host_re_gsub": self._h_re_gsub,
            "__host_re_split": self._h_re_split,
            "__host_unicode_charwidth": self._h_unicode_charwidth,
            "__host_unicode_len": self._h_unicode_len,
            "__host_unicode_sub": self._h_unicode_sub,
            "__host_unicode_char": self._h_unicode_char,
            "__host_unicode_codepoint": self._h_unicode_codepoint,
            "__host_unicode_upper": self._h_unicode_upper,
            "__host_unicode_lower": self._h_unicode_lower,
            "__host_unicode_reverse": self._h_unicode_reverse,
            "__host_hsv_to_rgb": self._h_hsv_to_rgb,
            "__host_hsl_to_rgb": self._h_hsl_to_rgb,
            "__host_missing_api": self._h_missing_api,
        }.items():
            host[name] = self._wrap(fn)
        self.lua.globals()["__host"] = host

    def _load_prelude(self) -> None:
        source = PRELUDE_PATH.read_text(encoding="utf-8")
        # NOTE: lupa's execute() returns the chunk's *return values*. A Lua chunk
        # that ends on a statement yields None, so the chunk result must never be
        # treated as "the prelude failed to define anything" — and must never be
        # written back over a global the prelude already defined (doing that was
        # what wiped `__build_subs`). The prelude ends with `return __build_subs`;
        # the global lookup is the authoritative fallback.
        result = self.lua.execute(to_lua_str(source))
        g = self.lua.globals()
        builder = result if callable(result) else g["__build_subs"]
        if not callable(builder):
            raise LuaError(
                "aegisub-mcp prelude did not define __build_subs "
                "(the subs/`subtitles` object would be unusable)"
            )
        g["__build_subs"] = builder
        if not callable(g["__new_subs"]):
            g["__new_subs"] = lambda handle: builder(handle)

    def _install_state(self) -> None:
        g = self.lua.globals()
        api = g.aegisub
        api.register_macro = self._wrap(self._api_register_macro)
        api.register_filter = self._wrap(self._api_register_filter)
        api.dialog = self.lua.table()
        api.dialog.display = self._wrap(self._api_dialog_display)
        api.dialog.open = self._wrap(self._api_dialog_open)
        api.dialog.save = self._wrap(self._api_dialog_save)
        api.text_extents = self._wrap(self._api_text_extents)
        api.video_size = self._wrap(self._api_video_size)
        api.project_properties = self._wrap(self._api_project_properties)
        api.frame_from_ms = self._wrap(self._api_frame_from_ms)
        api.ms_from_frame = self._wrap(self._api_ms_from_frame)
        api.gettext = self._wrap(lambda text: text)
        api.decode_path = self._wrap(self._api_decode_path)
        api.encode_path = self._wrap(self._api_encode_path)
        api.parse_karaoke_data = self._wrap(self._api_parse_karaoke_data)
        api.get_configuration = self._wrap(self._api_get_configuration)
        api.keyframes = self.lua_value(list(self.keyframes)) if self.keyframes is not None else False
        api.lua_automation_version = 4
        g.subs = None
        g.sel = None

    def _build_subs(self) -> Any:
        builder = self.lua.globals()["__new_subs"]
        return builder(self._handle)

    def _mark_unsupported(self, name: str) -> None:
        if name not in self.unsupported:
            self.unsupported.append(name)

    # -- host callbacks ----------------------------------------------------
    def _h_log(self, level: Any, message: Any) -> None:
        entry = {"level": _int_of(level, 0), "message": str(message)}
        if len(self.output) < self.log_limit:
            self.output.append(entry)
        elif self.output and self.output[-1].get("truncated") is not True:
            self.output.append({"level": 0, "message": "... log truncated ...", "truncated": True})

    def _h_progress(self, kind: Any, value: Any) -> None:
        name = str(kind)
        if name == "title":
            self.progress_title = str(value)
        elif name == "task":
            self.progress_task = str(value)
        elif name == "set":
            self.progress = _float_of(value, 0.0)

    def _h_is_cancelled(self) -> bool:
        return bool(self._cancelled)

    def _h_cancel(self) -> None:
        self._cancelled = True
        raise LuaError(CANCEL_MARKER)

    def _h_undo_point(self, description: Any) -> None:
        self.undo_points.append(str(description))

    def _h_include(self, path: Any) -> bool:
        target = self.resolve_include(str(path))
        if target is None:
            raise LuaError(f"include: cannot find {path!r} in {[str(p) for p in self.include_dirs]}")
        key = str(target)
        if key in self._include_cache:
            return True
        self._include_cache[key] = True
        self._run_chunk(target.read_text(encoding="utf-8", errors="surrogateescape"), str(target))
        return True

    def _h_require(self, name: Any) -> Any:
        module = str(name)
        if module.startswith("aegisub."):
            raise LuaError(f"require: module {module!r} is not provided by aegisub-mcp")
        target = self.resolve_include(module.replace(".", "/") + ".lua") or self.resolve_include(module + ".lua")
        if target is None:
            raise LuaError(f"require: cannot find module {module!r}")
        result = self._run_chunk(target.read_text(encoding="utf-8", errors="surrogateescape"), str(target))
        return result

    def _h_re_find(self, text: Any, pattern: Any) -> list[str]:
        try:
            rx = _pyre.compile(str(pattern))
        except _pyre.error as exc:
            raise LuaError(f"aegisub.re.find: invalid pattern {pattern!r}: {exc}") from None
        return [str(match.group(0)) for match in rx.finditer(str(text))]

    def _h_re_match(self, text: Any, pattern: Any) -> list[list[str]]:
        try:
            rx = _pyre.compile(str(pattern))
        except _pyre.error as exc:
            raise LuaError(f"aegisub.re.match: invalid pattern {pattern!r}: {exc}") from None
        out: list[list[str]] = []
        for match in rx.finditer(str(text)):
            groups = [match.group(0)]
            groups.extend("" if g is None else str(g) for g in match.groups())
            out.append(groups)
        return out

    def _h_re_gsub(self, text: Any, pattern: Any, replacement: Any) -> tuple[str, int]:
        try:
            rx = _pyre.compile(str(pattern))
        except _pyre.error as exc:
            raise LuaError(f"aegisub.re.gsub: invalid pattern {pattern!r}: {exc}") from None
        result, count = rx.subn(str(replacement), str(text))
        return result, count

    def _h_re_split(self, text: Any, pattern: Any) -> list[str]:
        try:
            rx = _pyre.compile(str(pattern))
        except _pyre.error as exc:
            raise LuaError(f"aegisub.re.split: invalid pattern {pattern!r}: {exc}") from None
        return [str(part) for part in rx.split(str(text))]

    # unicode module: A4 counts *bytes* for indices, *characters* for len/sub.
    def _h_unicode_charwidth(self, text: Any, index: Any) -> int:
        raw = str(text).encode("utf-8", "surrogateescape")
        position = _int_of(index, 1) - 1
        if position < 0 or position >= len(raw):
            # Aegisub: `b = s\byte i or 1 ... if not b then 1` -- an
            # out-of-range index (karaskel's empty placeholder syllable at
            # kara[0] hits this) returns 1 rather than raising.
            return 1
        lead = raw[position]
        if lead < 0x80:
            return 1
        if lead >> 5 == 0b110:
            return 2
        if lead >> 4 == 0b1110:
            return 3
        if lead >> 3 == 0b11110:
            return 4
        return 1

    def _h_unicode_len(self, text: Any) -> int:
        return len(str(text))

    def _h_unicode_sub(self, text: Any, start: Any, end: Any = None) -> str:
        chars = list(str(text))
        total = len(chars)
        first = _int_of(start, 1)
        if first < 0:
            first = total + first + 1
        first = max(first, 1)
        if end is None:
            last = total
        else:
            last = _int_of(end, total)
            if last < 0:
                last = total + last + 1
        last = min(last, total)
        if last < first:
            return ""
        return "".join(chars[first - 1 : last])

    def _h_unicode_char(self, *codepoints: Any) -> str:
        out = []
        for value in codepoints:
            try:
                out.append(chr(int(_int_of(value, 0))))
            except ValueError:
                out.append("\ufffd")
        return "".join(out)

    def _h_unicode_codepoint(self, text: Any, index: Any) -> int:
        chars = list(str(text))
        position = _int_of(index, 1)
        if position < 1 or position > len(chars):
            raise LuaError(f"unicode.codepoint: index {index} out of range (string has {len(chars)} characters)")
        return ord(chars[position - 1])

    def _h_unicode_upper(self, text: Any) -> str:
        return str(text).upper()

    def _h_unicode_lower(self, text: Any) -> str:
        return str(text).lower()

    def _h_unicode_reverse(self, text: Any) -> str:
        return "".join(reversed(list(str(text))))

    def _h_hsv_to_rgb(self, hue: Any, saturation: Any, value: Any) -> tuple[float, float, float]:
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(_float_of(hue, 0.0), _float_of(saturation, 0.0), _float_of(value, 0.0))
        return (r * 255.0, g * 255.0, b * 255.0)

    def _h_hsl_to_rgb(self, hue: Any, saturation: Any, lightness: Any) -> tuple[float, float, float]:
        import colorsys

        r, g, b = colorsys.hls_to_rgb(_float_of(hue, 0.0), _float_of(lightness, 0.0), _float_of(saturation, 0.0))
        return (r * 255.0, g * 255.0, b * 255.0)

    def _h_missing_api(self, name: Any) -> None:
        api = str(name)
        self._mark_unsupported(api)
        raise LuaError(
            f"aegisub.{api} is not implemented by aegisub-mcp. "
            f"The script needs an Aegisub API that has no equivalent here; "
            f"supported surface: macros/filters, progress, debug, undo points, dialogs, "
            f"text_extents, video_size, project_properties, frame/ms conversion, "
            f"gettext, (de)encode_path, parse_karaoke_data, keyframes, aegisub.util, aegisub.re, unicode."
        )

    # -- aegisub.* API -----------------------------------------------------
    def _api_register_macro(self, name: Any, description: Any, fn: Any, is_valid: Any = None) -> None:
        # Automation 4: ``register_macro(name, description, processing_function,
        # validation_function)`` (upstream automation/v4-docs/basic-function-
        # interface.txt).  The optional fourth argument is the "Macro Validation
        # Function": Aegisub calls it to decide whether the macro can act on the
        # current subtitles (it greys the menu entry out when it returns false).
        # Headless there is no menu to grey out, so the validator is honoured at
        # run time instead: a false result refuses the run with a clear error
        # (see run_macro / run_filters).
        self._registration_order += 1
        self.macros.append(
            RegisteredItem(
                name=str(name),
                description=str(description),
                kind="macro",
                priority=0,
                order=self._registration_order,
                fn=fn,
                is_valid=is_valid,
            )
        )

    def _api_register_filter(
        self, name: Any, description: Any, priority: Any, fn: Any, options_provider: Any = None
    ) -> None:
        # Automation 4: ``register_filter(name, description, priority,
        # processing_function, options_window_provider)`` (upstream
        # automation/v4-docs/basic-function-interface.txt).  The fifth argument
        # opens the filter's option window in the GUI; headless it is recorded
        # and reported as unsupported at run time rather than silently dropped.
        self._registration_order += 1
        self.filters.append(
            RegisteredItem(
                name=str(name),
                description=str(description),
                kind="filter",
                priority=_int_of(priority, 0),
                order=self._registration_order,
                fn=fn,
                options_provider=options_provider,
            )
        )

    def _api_dialog_display(self, dialog: Any, buttons: Any = None, button_ids: Any = None) -> Any:
        # Aegisub passes the *array* of controls as the first argument: LuaDialog
        # walks argument 1 with ``lua_for_each`` (src/auto4_lua_dialog.cpp).  A
        # ``{controls = {...}}`` wrapper is tolerated as well because it is a
        # common mistake, but scripts in the wild use the array form.
        raw = lua_to_py(dialog)
        if isinstance(raw, list):
            controls, spec = raw, {}
        elif isinstance(raw, dict):
            controls, spec = raw.get("controls") or [], raw
        else:
            controls, spec = [], {}
        button_spec = lua_to_py(buttons)
        ids: dict[str, Any] = {}
        if isinstance(button_spec, dict):
            # ``display(dialog, button_ids)``: argument 3 maps wx ids onto labels.
            ids, button_spec = button_spec, None
        if isinstance(button_ids, dict) and not ids:
            ids = lua_to_py(button_ids) or {}
        names: list[str] = []
        if isinstance(button_spec, list):
            names = [f"Button {i + 1}" if b is None else str(b) for i, b in enumerate(button_spec)]
        elif isinstance(button_spec, dict):
            names = [str(label) for label in button_spec.values()]
        names = [name for name in names if name]
        if not names:
            # Aegisub falls back to the stock OK/Cancel pair.
            names = ["OK", "Cancel"]
        keys: list[str] = [str(key) for key, label in ids.items()
                           if isinstance(label, str) and label.strip()]
        defaults: dict[str, Any] = {}
        if isinstance(controls, list):
            for control in controls:
                if not isinstance(control, dict):
                    continue
                key = control.get("name") or control.get("id") or control.get("key")
                if key:
                    defaults[str(key)] = control.get("value", "")
        record = {"kind": "display", "title": spec.get("title", ""), "buttons": names,
                  "keys": keys, "controls": controls}
        return self._answer_dialog(record, defaults, names)

    @staticmethod
    def _button_index(button: Any, names: list[str]) -> int | None:
        """Resolve a 1-based index or a button label to an index, ``None`` = close."""
        if isinstance(button, bool):
            return 1 if button else None
        if isinstance(button, str):
            return names.index(button) + 1 if button in names else None
        try:
            index = int(button)
        except (TypeError, ValueError):
            return None
        return index if 1 <= index <= len(names) else None

    @staticmethod
    def _button_label(index: int | None, names: list[str]) -> Any:
        """What Aegisub returns: the pressed button's label, or ``false``."""
        if index is None:
            return False
        return names[index - 1]

    def _answer_dialog(self, record: dict[str, Any], defaults: dict[str, Any], names: list[str]) -> Any:
        self.dialogs.append(record)
        mode = self.dialog_mode
        answers = self.dialog_answers
        title = str(record.get("title") or "")
        explicit = answers.get(title) if title else None
        if explicit is None:
            explicit = answers.get(str(record.get("kind")))
        if explicit is None and len(answers) == 1:
            explicit = next(iter(answers.values()))
        if explicit is not None and not isinstance(explicit, (dict, int, float, str, bool)):
            explicit = None
        if isinstance(explicit, dict):
            values = dict(defaults)
            values.update(explicit.get("values", {}) or {})
            index = self._button_index(explicit.get("button", 1), names)
        elif isinstance(explicit, (int, float, str, bool)):
            values = dict(defaults)
            index = self._button_index(explicit, names)
        elif mode == "defaults":
            values, index = dict(defaults), 1
        elif mode == "answers":
            self._dialog_unanswered(record, title)
            values, index = dict(defaults), None
        else:
            self._dialog_unanswered(record, title)
            values, index = dict(defaults), None
        record["button"] = index
        record["values"] = values
        return (self._button_label(index, names), py_to_lua(values))

    def _dialog_unanswered(self, record: dict[str, Any], title: str) -> None:
        """Remember a dialog that has no configured answer.

        The run is *not* aborted from inside the Lua call: a Python exception
        raised across the lupa boundary inside ``xpcall`` loses its message (Lua
        reports a bare "error in error handling"), so the macro carries on as if
        the user had closed the dialog and :meth:`_finish` reports the reason.
        """
        self.dialog_failures.append(
            f"the script opened a dialog ({title or record.get('kind')!r}) that has no configured answer; "
            "pass dialog_answers for it, or set dialog_mode='defaults' to accept the defaults"
        )

    def _api_dialog_open(self, name: Any, default_dir: Any = None, default_file: Any = None, filters: Any = None) -> Any:
        record = {"kind": "open", "title": str(name or ""), "filters": lua_to_py(filters)}
        answer = self.dialog_answers.get(str(name or "")) or self.dialog_answers.get("open")
        if isinstance(answer, str):
            self.dialogs.append(record)
            return answer
        self.dialogs.append(record)
        if self.dialog_mode == "defaults":
            return ""
        self._dialog_unanswered(record, str(name or ""))
        return None

    def _api_dialog_save(self, name: Any, default_dir: Any = None, default_file: Any = None, filters: Any = None) -> Any:
        record = {"kind": "save", "title": str(name or ""), "filters": lua_to_py(filters)}
        answer = self.dialog_answers.get(str(name or "")) or self.dialog_answers.get("save")
        self.dialogs.append(record)
        if isinstance(answer, str):
            return answer
        if self.dialog_mode == "defaults" and default_file:
            return default_file
        self._dialog_unanswered(record, str(name or ""))
        return None

    def _api_text_extents(self, style: Any, text: Any) -> tuple[float, float, float, float]:
        spec = lua_to_py(style)
        if not isinstance(spec, dict):
            raise LuaError("aegisub.text_extents: first argument must be a style table")
        return FM.text_extents_lua(str(text), spec)

    def _api_video_size(self) -> Any:
        if not self.video_size:
            return None
        width, height = self.video_size
        aspect = width / height if height else 0.0
        return (width, height, aspect, 0)

    def _api_project_properties(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "video_file": self.video or "",
            "audio_file": self.audio or "",
            "video_size": list(self.video_size) if self.video_size else [],
            "keyframes": list(self.keyframes) if self.keyframes else False,
            "timecodes": list(self.timecodes) if self.timecodes else False,
            "fps": self.fps if self.fps is not None else 0,
            "frames": 0,
            "automation_scripts": list(self.automation_scripts),
            "path": self.project_path or getattr(self.doc, "path", "") or "",
        }
        if self.fps and self.keyframes:
            properties["frames"] = int(self.keyframes[-1])
        return properties

    def _api_frame_from_ms(self, ms: Any) -> Any:
        if not self.fps:
            return None
        return int(round(_float_of(ms, 0.0) / 1000.0 * self.fps))

    def _api_ms_from_frame(self, frame: Any) -> Any:
        if not self.fps:
            return None
        return int(round(_int_of(frame, 0) / self.fps * 1000.0))

    def _api_decode_path(self, path: Any) -> str:
        return self.decode_path(str(path))

    def _api_encode_path(self, path: Any) -> str:
        return "?dummy:" + json.dumps(str(path), ensure_ascii=False)

    # Aegisub's karaoke tags are ``\k``, ``\K``, ``\kf``, ``\ko``
    # (``ass_override.cpp``); ``AssKaraoke`` matches any tag whose name starts
    # with ``\k`` case-insensitively and rewrites ``\K`` to ``\kf``.
    _KARAOKE_TAG_RE = _pyre.compile(r"^\\[kK](?:f|o)?")
    # Tag names are either all letters (``\p``, ``\kf``, ``\alpha``) or a
    # leading run of digits plus letters (``\1c``, ``\3a``).
    _TAG_NAME_RE = _pyre.compile(r"^\\(?:[0-9]+[a-zA-Z]*|[a-zA-Z]+)")
    _TAG_NUMBER_RE = _pyre.compile(r"[-+]?[0-9]*\.?[0-9]+")
    _BLOCK_RE = _pyre.compile(r"\{([^{}]*)\}")

    @staticmethod
    def _split_override_tags(block: str) -> list[str]:
        """Split an override block into tags, keeping ``\\t(...)`` whole.

        Aegisub attaches the tags nested inside ``\\t(...)`` to the ``\\t`` tag
        rather than treating them as siblings; splitting naively on ``\\`` would
        tear them apart and could even fake a karaoke boundary.
        """
        tags: list[str] = []
        current = ""
        depth = 0
        for char in block:
            if char == "\\" and depth == 0:
                if current:
                    tags.append(current)
                current = "\\"
                continue
            if char == "(":
                depth += 1
            elif char == ")" and depth:
                depth -= 1
            current += char
        if current:
            tags.append(current)
        return tags

    def _karaoke_syllables(self, line: Any) -> list[dict[str, Any]]:
        """Syllables of a dialogue line, as Aegisub's ``AssKaraoke`` sees them.

        Mirrors ``AssKaraoke::ParseSyllables`` (``src/ass_karaoke.cpp``) with
        ``auto_split = normalize = false`` -- the exact configuration
        ``LuaAssFile::LuaParseKaraokeData`` uses:

        * a syllable's duration comes from the karaoke tag *preceding* its
          text, and a line with no karaoke tags yields a single syllable
          holding the whole visible text (duration 0);
        * ``start_time`` / ``end_time`` are milliseconds relative to the line
          start and ``tag`` is the tag name *with* the backslash (``\\k``,
          ``\\kf``, ``\\ko``; ``\\K`` is normalized to ``\\kf``);
        * ``text`` re-inserts the syllable's non-karaoke override tags (and
          drawing commands) while ``text_stripped`` keeps plain text only.
        """
        table = lua_to_py(line)
        if not isinstance(table, dict):
            raise LuaError("aegisub.parse_karaoke_data: expected a line table")
        text = str(table.get("text", ""))

        syllables: list[dict[str, Any]] = []
        syl_text = ""
        groups: list[tuple[int, str]] = []
        duration = 0
        tag_type = "\\k"
        cursor = 0
        drawing = False

        def add_group(rendered: str) -> None:
            # Aegisub keeps override tags in a map keyed by the plain-text
            # offset, so groups that land on the same offset concatenate.
            offset = len(syl_text)
            if groups and groups[-1][0] == offset:
                groups[-1] = (offset, groups[-1][1] + rendered)
            else:
                groups.append((offset, rendered))

        def flush(final: bool = False) -> None:
            nonlocal syl_text, groups, duration, tag_type, cursor
            if not final and duration <= 0 and not syl_text:
                # "Don't bother including zero duration zero length syls"
                return
            rendered: list[str] = []
            position = 0
            for offset, group in groups:
                rendered.append(syl_text[position:offset])
                rendered.append(group)
                position = offset
            rendered.append(syl_text[position:])
            syllables.append(
                {
                    "duration": duration,
                    "start_time": cursor,
                    "end_time": cursor + duration,
                    "tag": tag_type,
                    "text": "".join(rendered),
                    "text_stripped": syl_text,
                }
            )
            cursor += duration
            syl_text = ""
            groups = []
            duration = 0
            tag_type = "\\k"

        position = 0
        for match in self._BLOCK_RE.finditer(text):
            plain = text[position:match.start()]
            position = match.end()
            if plain:
                if drawing:
                    add_group(plain)
                else:
                    syl_text += plain
            merged = ""
            for tag in self._split_override_tags(match.group(1)):
                karaoke = self._KARAOKE_TAG_RE.match(tag)
                if karaoke:
                    name = karaoke.group(0)
                    if name == "\\K":
                        name = "\\kf"
                    value = self._TAG_NUMBER_RE.match(tag[karaoke.end():])
                    if merged:
                        add_group("{" + merged + "}")
                        merged = ""
                    flush()
                    tag_type = name
                    duration = int(float(value.group(0)) * 10) if value else 0
                    continue
                named = self._TAG_NAME_RE.match(tag)
                if named and named.group(0) == "\\p":
                    value = self._TAG_NUMBER_RE.match(tag[named.end():])
                    drawing = bool(value) and float(value.group(0)) > 0
                merged += tag
            if merged:
                add_group("{" + merged + "}")
        plain = text[position:]
        if plain:
            if drawing:
                add_group(plain)
            else:
                syl_text += plain
        flush(final=True)
        return syllables

    def _api_parse_karaoke_data(self, line: Any) -> dict[int, dict[str, Any]]:
        """Lua-facing ``aegisub.parse_karaoke_data``.

        The returned table is *0-based*: index 0 is an empty placeholder
        syllable and the real syllables follow at 1..n, so ``#kara`` is the
        syllable count.  Aegisub does exactly this (``LuaParseKaraokeData``:
        "2.1.x stored everything before the first syllable at index zero ...
        scripts may rely on kara[0] existing so add an empty syllable"), and it
        is load-bearing here: karaskel's ``preproc_line_text`` walks
        ``for i = 0, #kara`` and dereferences ``kara[i]`` with no nil guard.
        """
        placeholder = {
            "duration": 0,
            "start_time": 0,
            "end_time": 0,
            "tag": "",
            "text": "",
            "text_stripped": "",
        }
        data: dict[int, dict[str, Any]] = {0: placeholder}
        for index, syllable in enumerate(self._karaoke_syllables(line), start=1):
            data[index] = syllable
        return data

    def _api_get_configuration(self, key: Any = None) -> Any:
        if key is None:
            return dict(self.config)
        return self.config.get(str(key), "")

    # -- path resolution ---------------------------------------------------
    def resolve_include(self, name: str) -> Path | None:
        candidate = Path(name).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for directory in self.include_dirs:
            target = (directory / name).expanduser()
            if target.exists():
                return target
        return None

    def decode_path(self, spec: str) -> str:
        """Turn an Aegisub path specifier (``?script/x.lua``) into a real path."""
        if not spec.startswith("?"):
            return str(Path(spec).expanduser())
        body = spec[1:]
        head, _, tail = body.partition("/")
        folder: str | None = None
        if head == "script":
            base = Path(self.project_path).expanduser().parent if self.project_path else Path.cwd()
            folder = str(base)
        elif head in {"data", "user"}:
            folder = str(Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "aegisub")
        elif head == "temp":
            folder = tempfile.gettempdir()
        elif head == "video":
            folder = str(Path(self.video).parent) if self.video else None
        elif head == "audio":
            folder = str(Path(self.audio).parent) if self.audio else None
        elif head == "dummy":
            return tail or ""
        if folder is None:
            return spec
        return str(Path(folder) / tail) if tail else folder

    # -- loading -----------------------------------------------------------
    def _run_chunk(self, source: str, name: str = "<string>") -> Any:
        # Lua source is handed over as UTF-8 *bytes* (latin-1 reinterpretation),
        # exactly like Aegisub reading a .lua file: literals keep their bytes and
        # scripts that slice multi-byte text behave identically.
        loaded = self.lua.eval("loadstring or load")(to_lua_str(source), "@" + name)
        # Lua 5.1's loadstring (and 5.2+'s load) return ``nil, errmsg`` when the
        # chunk does not compile; lupa hands multiple Lua return values back as
        # a Python tuple, so a compile failure arrives as a 2-tuple and a plain
        # ``chunk is None`` check would then blow up with a confusing
        # ``'tuple' object is not callable``.
        chunk: Any = loaded
        message = ""
        if isinstance(loaded, tuple):
            chunk = loaded[0] if loaded else None
            message = str(loaded[1]) if len(loaded) > 1 and loaded[1] is not None else ""
        if chunk is None:
            detail = f": {message}" if message else " (chunk did not compile)"
            raise LuaError(f"{name}: Lua syntax error{detail}")
        try:
            return chunk()
        except lupa.LuaError as exc:  # pragma: no cover - surfaced to the caller
            raise LuaError(f"{name}: {exc}") from None

    def load_script(self, source: str, *, name: str = "<string>") -> LuaEngine:
        """Load (execute) an automation script, registering its macros/filters."""
        self.scripts.append(name)
        before = len(self.macros), len(self.filters)
        try:
            self._run_chunk(source, name)
        except LuaError as exc:
            raise LuaError(f"failed to load automation script {name}: {exc}") from None
        for item in self.macros[before[0] :] + self.filters[before[1] :]:
            item.script = name
        return self

    def load_file(self, path: str | os.PathLike[str]) -> LuaEngine:
        target = Path(path).expanduser()
        if not target.exists():
            raise FileNotFoundError(f"automation script not found: {target}")
        return self.load_script(target.read_text(encoding="utf-8", errors="surrogateescape"), name=str(target))

    # -- inspection --------------------------------------------------------
    def list_macros(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.macros]

    def list_filters(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.filters]

    def find(self, name: str, kind: str | None = None) -> RegisteredItem:
        pool = {"macro": self.macros, "filter": self.filters}.get(kind) if kind else self.macros + self.filters
        pool = pool or []
        for item in pool:
            if item.name == name:
                return item
        lowered = name.strip().lower()
        for item in pool:
            if item.name.lower() == lowered:
                return item
        available = ", ".join(sorted({item.name for item in pool})) or "none"
        raise MacroNotFound(f"{kind or 'macro'} {name!r} is not registered; available: {available}")

    # -- line tables -------------------------------------------------------
    def line_class(self, section: Section | None, entry: Entry) -> str:
        if isinstance(entry, EventEntry):
            return CLASS_DIALOGUE
        if isinstance(entry, StyleEntry):
            return CLASS_STYLE
        if isinstance(entry, DataEntry):
            if entry.kind.lower().startswith("style"):
                return CLASS_STYLE
            return CLASS_UNKNOWN
        kind = (section.kind if section is not None else KIND_INFO) or KIND_INFO
        if kind == KIND_INFO:
            raw = getattr(entry, "raw", "") or ""
            key, sep, _value = raw.partition(":")
            if sep and key.strip() and not key.strip().startswith((";", "!")):
                return CLASS_INFO
        return CLASS_UNKNOWN

    def line_section_name(self, section: Section | None) -> str:
        if section is None:
            return "Script Info"
        header = (section.header or "").strip()
        if header.startswith("[") and header.endswith("]"):
            return header[1:-1].strip()
        return header or section.kind

    @staticmethod
    def _read_spec(entry: DataEntry, spec: dict[str, tuple[str, str]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field, (column, kind) in spec.items():
            if kind == "bool":
                if entry.has(column) or field == "comment":
                    out[field] = _bool_of(entry.get(column, "0"))
                continue
            if not entry.has(column):
                continue
            raw = entry.get(column, "")
            if kind == "int":
                out[field] = _int_of(raw, 0)
            elif kind == "num":
                out[field] = _float_of(raw, 0.0)
            elif kind == "raw":
                out[field] = raw
            elif kind == "time":
                # raw is ASS timestamp text ("0:00:01.00"); A4 exposes ms.
                out[field] = _time_value(raw, 0)
            else:
                out[field] = raw
        return out

    def line_table(self, section: Section | None, entry: Entry) -> dict[str, Any]:
        """The Lua table representation of one physical line."""
        klass = self.line_class(section, entry)
        table: dict[str, Any] = {"class": klass, "section": self.line_section_name(section)}
        if isinstance(entry, RawEntry):
            table["raw"] = entry.raw
        else:
            table["raw"] = entry.render() if hasattr(entry, "render") else getattr(entry, "raw", "")
            table["malformed"] = bool(getattr(entry, "malformed", False))
        if klass == CLASS_INFO:
            raw = getattr(entry, "raw", table.get("raw", ""))
            key, _sep, value = str(raw).partition(":")
            table["key"] = key.strip()
            table["value"] = value.strip()
        elif klass == CLASS_STYLE and isinstance(entry, DataEntry):
            table.update(self._read_spec(entry, STYLE_SPEC))
            extra: dict[str, str] = {}
            for column in entry.order:
                canonical = canonical_style_key(column)
                if any(canonical == spec_column for spec_column, _ in STYLE_SPEC.values()):
                    continue
                extra[canonical] = entry.get(canonical, "")
            if extra:
                table["extra"] = extra
        elif klass == CLASS_DIALOGUE and isinstance(entry, EventEntry):
            table.update(self._read_spec(entry, DIALOGUE_SPEC))
            table["comment"] = entry.is_comment
            for alias in ("margin_v", "margin_t", "margin_b"):
                if entry.has("MarginV"):
                    table[alias] = _int_of(entry.get("MarginV", "0"), 0)
        return table

    # -- writes ------------------------------------------------------------
    def _column_values(self, payload: dict[str, Any], spec: dict[str, tuple[str, str]]) -> dict[str, str]:
        columns: dict[str, str] = {}
        for field, value in payload.items():
            if field in {"class", "section", "raw", "malformed", "extra", "key", "value", "comment"}:
                # ``comment`` toggles Dialogue/Comment via ``entry.kind`` (see
                # ``apply_line_table``); it is not a real ASS column, so it must
                # never reach ``DataEntry.set`` (which would append a bogus field
                # name to ``order`` and emit an extra trailing column).
                continue
            entry_spec = spec.get(field)
            if entry_spec is None:
                continue
            column, kind = entry_spec
            if kind == "bool":
                columns[column] = "-1" if _bool_of(value) else "0"
            elif kind == "int":
                columns[column] = str(_int_of(value, 0))
            elif kind == "num":
                columns[column] = _num_str(_float_of(value, 0.0))
            elif kind == "time":
                columns[column] = U.format_time(_time_value(value, 0))
            else:
                columns[column] = "" if value is None else str(value)
        extra = payload.get("extra")
        if isinstance(extra, dict):
            for key, value in extra.items():
                columns[canonical_style_key(str(key))] = "" if value is None else str(value)
        return columns

    def apply_line_table(self, section: Section | None, entry: Entry, payload: dict[str, Any], *, position: int | None = None) -> None:
        """Write a line table back onto an existing line (A4 copy-back semantics)."""
        klass = self.line_class(section, entry)
        target = str(payload.get("class") or klass)
        if target != klass:
            self._replace_line_kind(section, entry, payload, target, position)
            return
        if klass == CLASS_INFO:
            key = str(payload.get("key", "") or "")
            value = str(payload.get("value", "") or "")
            current = getattr(entry, "raw", "")
            key_now, _sep, value_now = str(current).partition(":")
            if key.strip() == key_now.strip() and value.strip() == value_now.strip():
                return
            new_raw = f"{key.strip()}: {value.strip()}"
            self._set_raw_line(section, entry, new_raw, position)
            return
        if klass == CLASS_UNKNOWN:
            raw = payload.get("raw")
            if raw is None:
                return
            self._set_raw_line(section, entry, str(raw), position)
            return
        if not isinstance(entry, DataEntry):
            return
        spec = STYLE_SPEC if klass == CLASS_STYLE else DIALOGUE_SPEC
        columns = self._column_values(payload, spec)
        if klass == CLASS_DIALOGUE and "comment" in payload:
            want_comment = _bool_of(payload["comment"])
            if want_comment != entry.is_comment:
                entry.kind = "Comment" if want_comment else "Dialogue"
                entry.lead = f"{entry.kind}:"
        changed = False
        for column, value in columns.items():
            if not entry.has(column):
                # The section's Format line does not carry this column: ASS has
                # a fixed column set per section, so drop it (Aegisub likewise
                # only emits fields present in the Format) instead of appending
                # a stray extra field to the rendered line.
                continue
            if entry.get(column, "") != value:
                entry.set(column, value)
                changed = True
        if changed or entry.kind:
            self.doc.mark_dirty()

    def _replace_line_kind(self, section: Section | None, entry: Entry, payload: dict[str, Any], klass: str, position: int | None) -> None:
        text = self.line_text_for(klass, payload, section, entry)
        if position is None:
            position = self._position_of(entry)
        self.doc.remove_lines([position])
        self.doc.insert_lines(position, [text])

    def _set_raw_line(self, section: Section | None, entry: Entry, text: str, position: int | None) -> None:
        if isinstance(entry, RawEntry):
            if entry.raw != text:
                entry.raw = text
                self.doc.mark_dirty()
            return
        if position is None:
            position = self._position_of(entry)
        if position is None:
            raise LuaError("cannot locate line for raw write")
        self.doc.remove_lines([position])
        self.doc.insert_lines(position, [text])

    def _position_of(self, entry: Entry) -> int | None:
        for index, (_section, candidate) in enumerate(self.doc.all_lines()):
            if candidate is entry:
                return index
        return None

    def line_text_for(self, klass: str, payload: dict[str, Any], section: Section | None = None, template: Entry | None = None) -> str:
        """Render an A4 line table into an ASS line of the right kind."""
        if klass == CLASS_INFO:
            key = str(payload.get("key", "") or "Title")
            value = str(payload.get("value", "") or "")
            return f"{key}: {value}"
        if klass == CLASS_UNKNOWN:
            return str(payload.get("raw", "") or "")
        if klass == CLASS_STYLE:
            order = self._section_format(section, KIND_STYLES)
            values = self._column_values(payload, STYLE_SPEC)
            name = str(payload.get("name", payload.get("Name", "Default")) or "Default")
            defaults = {
                "Name": name,
                "Fontname": "Arial",
                "Fontsize": "48",
                "PrimaryColour": "&H00FFFFFF&",
                "SecondaryColour": "&H000000FF&",
                "OutlineColour": "&H00000000&",
                "BackColour": "&H80000000&",
                "Bold": "0",
                "Italic": "0",
                "Underline": "0",
                "StrikeOut": "0",
                "ScaleX": "100",
                "ScaleY": "100",
                "Spacing": "0",
                "Angle": "0",
                "BorderStyle": "1",
                "Outline": "2",
                "Shadow": "1",
                "Alignment": "2",
                "MarginL": "20",
                "MarginR": "20",
                "MarginV": "20",
                "AlphaLevel": "0",
                "Encoding": "1",
            }
            cells = [values.get(column, defaults.get(column, "0")) for column in order]
            return "Style: " + ",".join(cells)
        order = self._section_format(section, KIND_EVENTS)
        values = self._column_values(payload, DIALOGUE_SPEC)
        defaults = {"Layer": "0", "Start": "0:00:00.00", "End": "0:00:00.00", "Style": "Default", "Name": "", "MarginL": "0", "MarginR": "0", "MarginV": "0", "Effect": "", "Text": ""}
        cells = [values.get(column, defaults.get(column, "")) for column in order]
        comment = _bool_of(payload.get("comment", False))
        return ("Comment: " if comment else "Dialogue: ") + ",".join(cells)

    def _section_format(self, section: Section | None, kind: str) -> list[str]:
        sections: list[Section] = []
        if section is not None and section.kind == kind:
            sections.append(section)
        sections.extend(sec for sec in self.doc.sections if sec.kind == kind and sec is not section)
        for sec in sections:
            for entry in sec.entries:
                if isinstance(entry, DataEntry) and entry.kind.lower() == "format":
                    return list(entry.order)
        if kind == KIND_EVENTS:
            return [part.strip() for part in EVENT_FORMAT.split(":", 1)[1].split(",")]
        header = (sections[0].header if sections else "") or ""
        if "V4+" in header or not header:
            return [part.strip() for part in V4P_STYLE_FORMAT.split(":", 1)[1].split(",")]
        return [part.strip() for part in V4_STYLE_FORMAT.split(":", 1)[1].split(",")]

    def _parse_line_entry(self, section: Section | None, text: str, klass: str) -> Entry:
        """Parse *text* into a real entry using the target section's own layout.

        Lines created by a script must read back with the class the script asked
        for, so dialogue/style lines are built through the real parser (section
        header, format order, comment/dialogue lead) instead of being kept as
        opaque raw text that would later report ``class == "unknown"``.
        """
        if klass in (CLASS_UNKNOWN, CLASS_INFO) or not text.strip():
            return RawEntry(text)
        header = (section.header if section is not None else None) or ""
        if not header:
            header = "[V4+ Styles]" if klass == CLASS_STYLE else "[Events]"
        fmt: list[str] = []
        if section is not None:
            for entry in section.entries:
                if isinstance(entry, DataEntry) and entry.kind.lower() == "format":
                    fmt.append(entry.render())
        if not fmt:
            fmt.append(V4P_STYLE_FORMAT if klass == CLASS_STYLE else EVENT_FORMAT)
        try:
            tmp = AssDocument.from_text("\n".join([header, *fmt, text]) + "\n")
            lines = tmp.all_lines()
            if lines and not isinstance(lines[-1][1], RawEntry):
                return lines[-1][1]
        except Exception:  # pragma: no cover - defensive: fall back to raw
            pass
        return RawEntry(text)

    def create_line(self, payload: dict[str, Any], *, position: int | None = None) -> None:
        """Create a line (append when *position* is None, else insert before it)."""
        klass = str(payload.get("class") or CLASS_DIALOGUE)
        if position is None:
            # ``subs.append(line)`` in A4 appends to the flat end of the entry
            # list (cf. AssFile's single entry vector): the new entry lands
            # after *every* existing physical line, even when the file ends with
            # a section Aegisub does not know.  ``line.class`` comes from the
            # entry type, not the enclosing section, so the entry is still built
            # typed -- through the real parser, using the section that owns the
            # class -- and merely *stored* in the last container.
            if klass == CLASS_INFO:
                section = self.doc.section(KIND_INFO) or self.doc.ensure_section(KIND_INFO, header=_SCRIPT_HEADER)
            elif klass == CLASS_STYLE:
                section = self.doc.style_section(create=True)
            else:
                section = (self.doc.event_sections(create=True) or [None])[0]
            text = self.line_text_for(klass, payload, section)
            entry = self._parse_line_entry(section, text, klass)
            containers = self.doc.line_containers()
            if containers:
                container = typing.cast("list[Entry]", containers[-1][1])
            else:  # pragma: no cover - a document always has a preamble
                container = typing.cast("list[Entry]", self.doc.preamble)
            container.append(entry)
            self.doc.mark_dirty()
            return
        total = len(self.doc.all_lines())
        target = min(max(position, 1), total + 1)
        section: Section | None = None
        if target <= total:
            section, _entry = self.doc.all_lines()[target - 1]
        elif self.doc.sections:
            section = self.doc.sections[-1]
        text = self.line_text_for(klass, payload, section)
        self.doc.insert_lines(target - 1, [text])

    # -- running -----------------------------------------------------------
    def _snapshot_state(self) -> list[str]:
        out: list[str] = []
        for _section, entry in self.doc.all_lines():
            if isinstance(entry, RawEntry):
                out.append(entry.raw)
            else:
                out.append(entry.render() if hasattr(entry, "render") else getattr(entry, "raw", ""))
        return out

    def _finish(self, result: MacroResult, before: list[str], started: float, snapshot: str, dry_run: bool) -> MacroResult:
        after = self._snapshot_state()
        result.lines_before = len(before)
        result.lines_after = len(after)
        result.structure_changed = len(before) != len(after)
        if not result.structure_changed:
            result.changed_lines = [i + 1 for i, (a, b) in enumerate(zip(before, after)) if a != b]
        result.progress_title = self.progress_title
        result.progress_task = self.progress_task
        result.progress = self.progress
        result.undo_points = list(self.undo_points)
        result.dialogs = list(self.dialogs)
        result.unsupported = list(self.unsupported)
        result.selection = list(self.selection)
        result.active_line = self.selection[-1] if self.selection else 0
        result.cancelled = self._cancelled
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        result.output = list(self.output)
        result.dialog_failures = list(self.dialog_failures)
        if result.dialog_failures:
            # A dialog the script opened was never answered, so the script ran as
            # if the user had closed it.  That is never a clean success.
            note = "; ".join(result.dialog_failures)
            result.error = f"{result.error}\n{note}" if result.error else note
            result.ok = False
        if result.error and CANCEL_MARKER in result.error:
            result.ok = False
            result.cancelled = True
            result.error = "cancelled by the script (aegisub.cancel)"
        if dry_run and self.doc.to_text() != snapshot:
            self.doc.restore(snapshot)
        result.dry_run = dry_run
        return result

    def run_macro(
        self,
        name: str,
        *,
        selection: Sequence[int] | None = None,
        config: dict[str, Any] | None = None,
        dry_run: bool = False,
        raise_errors: bool = False,
    ) -> MacroResult:
        """Run a registered macro.

        With ``raise_errors=True`` a failed macro raises :class:`LuaError`
        (message carries the Lua source line and a Lua traceback) instead of
        returning ``MacroResult(ok=False)`` -- callers that must not silently
        succeed can use this.
        """
        item = self.find(name, "macro") if name else self.find_first_macro()
        self._reset_run_state()
        if selection is not None:
            self.selection = [int(i) for i in selection]
        if config:
            self.config.update(config)
        result = MacroResult(name=item.name, kind="macro", ok=True)
        before = self._snapshot_state()
        snapshot = self.doc.snapshot()
        g = self.lua.globals()
        subs = self._build_subs()
        active_line = self.selection[-1] if self.selection else 0
        sel = self.lua_value(list(self.selection))
        # aegisub-mcp convenience: the A4 callback parameters are also exposed as
        # globals so probes/one-off macros can reach them by the names used in
        # Aegisub's documentation (``subtitles``, ``subs``, ``selected_lines``,
        # ``sel``, ``active_line``, ``config``).
        g.subs = subs
        g.subtitles = subs
        g.sel = sel
        g.selected_lines = sel
        g.active_line = active_line
        g.config = self.lua_value(dict(self.config))
        started = time.perf_counter()
        # Aegisub's optional 4th register_macro argument is the validation
        # function: it decides whether the macro may run at all for the current
        # subtitles/selection (called with the same three arguments as the macro
        # -- src/auto4_lua.cpp, LuaCommand::Validate).  A GUI greys the menu
        # entry out; headless the run is refused with a real error so a caller
        # cannot mistake "not applicable" for "ran fine".
        if item.is_valid is not None:
            # ``__protected_call`` returns ``(ok, err, <what the fn returned>)``,
            # so the validation *result* is slot 2 -- reading slot 0/1 here used to
            # make every validate-gated macro fail with "not applicable" even when
            # the script said yes (``valid`` was the pcall status and
            # ``valid_err`` the error, always nil on success).
            valid_outcome = self.lua.globals()["__protected_call"](item.is_valid, subs, sel, active_line)
            if isinstance(valid_outcome, tuple):
                called_ok, call_error = valid_outcome[0], valid_outcome[1]
                valid_returns: tuple[Any, ...] = tuple(valid_outcome[2:])
            else:  # pragma: no cover - defensive: lupa returns a tuple for multi-value calls
                called_ok, call_error, valid_returns = bool(valid_outcome), None, ()
            if not called_ok:
                result.ok = False
                result.error = (
                    f"macro {item.name!r} was refused: its register_macro validation "
                    f"function raised {call_error}"
                )
                finished = self._finish(result, before, started, snapshot, dry_run)
                if raise_errors:
                    raise LuaError(finished.error)
                return finished
            if not (valid_returns and lua_to_py(valid_returns[0])):
                result.ok = False
                result.error = (
                    f"macro {item.name!r} was not applicable: its register_macro validation "
                    "function returned false for the current subtitles/selection"
                )
                finished = self._finish(result, before, started, snapshot, dry_run)
                if raise_errors:
                    raise LuaError(finished.error)
                return finished
        # Automation 4 calls a macro as
        # ``fn(subtitles, selected_lines, active_line)`` (see upstream
        # ``automation/v4-docs/basic-function-interface.txt`` and real scripts
        # such as ``automation/autoload/strip-tags.lua`` / ``macro-1-edgeblur.lua``).
        # ``selected_lines`` is an array table of 1-based line indices;
        # ``active_line`` is the index of the active line (0 when nothing is
        # active).  A macro that only declares ``function(subs)`` simply ignores
        # the extra arguments, exactly as it does in Aegisub.
        outcome = self.lua.globals()["__protected_call"](item.fn, subs, sel, active_line)
        if isinstance(outcome, tuple):
            ok, err = outcome[0], outcome[1]
            returns: tuple[Any, ...] = tuple(outcome[2:])
        else:  # pragma: no cover - defensive: lupa returns a tuple for multi-value calls
            ok, err, returns = bool(outcome), None, ()
        if not ok:
            result.ok = False
            result.error = str(err)
        # Aegisub reads up to two values back from a macro: a new active line
        # index and a new selection table (src/auto4_lua.cpp,
        # LuaCommand::operator()).  Apply them to the engine's selection before
        # _finish() snapshots it, so a caller sees the selection the macro
        # asked for instead of the one it started with.
        active_out = self._apply_macro_returns(returns) if ok else 0
        if g.subs is not None and getattr(g.subs, "refresh", None):
            try:
                g.subs.refresh()
            except lupa.LuaError:  # pragma: no cover - defensive
                pass
        finished = self._finish(result, before, started, snapshot, dry_run)
        if active_out:
            finished.active_line = active_out
        if raise_errors and not finished.ok:
            raise LuaError(finished.error or f"macro {item.name!r} failed")
        return finished

    def _apply_macro_returns(self, returns: tuple[Any, ...]) -> int:
        """Apply a macro's return values the way Aegisub does.

        Returns the new active line index (0 when the macro did not ask for
        one).  Out-of-range values are dropped, mirroring the bounds checks in
        ``LuaCommand::operator()`` (src/auto4_lua.cpp).
        """
        if not returns:
            return 0
        count = len(self._snapshot_state())
        active = 0
        candidate = lua_to_py(returns[0])
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            index = int(candidate)
            if 1 <= index <= count:
                active = index
        if len(returns) > 1:
            table = lua_to_py(returns[1])
            if isinstance(table, list):
                wanted = {
                    int(value)
                    for value in table
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and 1 <= int(value) <= count
                }
                if wanted:
                    self.selection = sorted(wanted)
        return active

    def find_first_macro(self) -> RegisteredItem:
        if not self.macros:
            raise MacroNotFound("no macros registered by the loaded script(s)")
        return self.macros[0]

    def run_filter(
        self,
        name: str,
        *,
        selection: Sequence[int] | None = None,
        config: dict[str, Any] | None = None,
        dry_run: bool = False,
        raise_errors: bool = False,
    ) -> MacroResult:
        item = self.find(name, "filter") if name else self.find_first_filter()
        self._reset_run_state()
        if selection is not None:
            self.selection = [int(i) for i in selection]
        if config:
            self.config.update(config)
        result = MacroResult(name=item.name, kind="filter", ok=True)
        before = self._snapshot_state()
        snapshot = self.doc.snapshot()
        g = self.lua.globals()
        subs = self._build_subs()
        sel = self.lua_value(list(self.selection))
        cfg = self.lua_value(dict(self.config))
        g.subs = subs
        g.subtitles = subs
        g.sel = sel
        g.selected_lines = sel
        g.config = cfg
        active_line = self.selection[-1] if self.selection else 0
        g.active_line = active_line
        if item.options_provider is not None:
            # register_filter's 5th argument opens the filter's option window in
            # the GUI; headless there is nothing to open, so say so instead of
            # pretending the user configured anything.
            self._mark_unsupported(
                f"aegisub.register_filter options window of {item.name!r} is GUI-only; "
                "the filter ran with the stored/default configuration"
            )
        started = time.perf_counter()
        # Automation 4 calls a filter as ``fn(subtitles, config)`` (upstream
        # ``automation/autoload/cleantags-autoload.lua`` =>
        # ``cleantags_filter(subtitles, config)``; ``kara-templater.lua`` =>
        # ``filter_apply_templates(subs, config)``), where ``config`` is the
        # stored "dialog result" for that filter.
        #
        # aegisub-mcp callers (and the bundled ``tests/lua/smoke.lua`` filter,
        # declared ``function(subtitles, selected, active)``) additionally rely
        # on the selected line indices being available in that same slot, so the
        # second argument is a superset table: array part = selected line
        # indices (``ipairs`` keeps working), named fields = the filter config
        # (``config.<option>`` keeps working).  The third argument is the active
        # line index, as for macros.
        active_line = self.selection[-1] if self.selection else 0
        filter_config = self.lua_value(list(self.selection))
        for key, value in self.config.items():
            filter_config[str(key)] = self.lua_value(value)
        outcome = self.lua.globals()["__protected_call"](item.fn, subs, filter_config, active_line)
        # ``__protected_call`` returns ``(ok, err, <whatever fn returned>)`` and
        # the documented Automation 4 filter contract has filters returning the
        # (possibly new) subtitles table -- upstream ``cleantags_filter`` and
        # ``filter_apply_templates`` both do.  Unpacking exactly two values made
        # any filter that returned something die with "too many values to
        # unpack", so go by index like :meth:`run_macro` does.  The returned
        # table itself is *not* re-applied: Automation 4 filters edit
        # ``subtitles`` in place and Aegisub only reads a new active
        # line/selection out of the return values.
        if isinstance(outcome, tuple):
            ok, err = outcome[0], outcome[1]
            returns: tuple[Any, ...] = tuple(outcome[2:])
        else:  # pragma: no cover - defensive: lupa returns a tuple for multi-value calls
            ok, err, returns = bool(outcome), None, ()
        if not ok:
            result.ok = False
            result.error = str(err)
        active_out = self._apply_macro_returns(returns) if ok else 0
        if g.subs is not None and getattr(g.subs, "refresh", None):
            try:
                g.subs.refresh()
            except lupa.LuaError:  # pragma: no cover - defensive
                pass
        finished = self._finish(result, before, started, snapshot, dry_run)
        if active_out:
            finished.active_line = active_out
        if raise_errors and not finished.ok:
            raise LuaError(finished.error or f"filter {item.name!r} failed")
        return finished

    def find_first_filter(self) -> RegisteredItem:
        if not self.filters:
            raise MacroNotFound("no filters registered by the loaded script(s)")
        return sorted(self.filters, key=lambda item: (item.priority, item.order))[0]

    def run_filters(
        self,
        *,
        selection: Sequence[int] | None = None,
        config: dict[str, Any] | None = None,
        dry_run: bool = False,
        raise_errors: bool = False,
    ) -> list[MacroResult]:
        """Run every registered filter in Aegisub order (priority, then order)."""
        results: list[MacroResult] = []
        for item in sorted(self.filters, key=lambda entry: (entry.priority, entry.order)):
            results.append(
                self.run_filter(
                    item.name,
                    selection=selection,
                    config=config,
                    dry_run=dry_run,
                    raise_errors=raise_errors,
                )
            )
            if not results[-1].ok:
                break
        return results

    def _reset_run_state(self) -> None:
        self.output = []
        self.undo_points = []
        self.dialogs = []
        self.progress_title = None
        self.progress_task = None
        self.progress = None
        self._cancelled = False

    # -- karaoke data (host helper used by tools) --------------------------
    def parse_karaoke_data(self, line: dict[str, Any]) -> list[dict[str, Any]]:
        # Python-side accessor: a 1-based list, times relative to the line.
        return self._karaoke_syllables(line)


# ---------------------------------------------------------------------------
# CLI: python -m aegisub_mcp.lua.engine script.lua --ass file.ass --macro Name
# ---------------------------------------------------------------------------


def run_script_cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m aegisub_mcp.lua.engine", description="Run an Aegisub automation script headlessly")
    parser.add_argument("script", help="automation script (.lua)")
    parser.add_argument("--ass", dest="ass_path", help="subtitle file to load")
    parser.add_argument("--macro", help="macro to run (default: first registered)")
    parser.add_argument("--filter", help="filter to run")
    parser.add_argument("--all-filters", action="store_true", help="run every registered filter")
    parser.add_argument("--selection", help="comma separated 1-based line selection")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-to", help="write the result here")
    parser.add_argument("--list", action="store_true", help="only list macros/filters")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--keyframes")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    doc = AssDocument.load(args.ass_path) if args.ass_path else AssDocument.from_text("[Script Info]\nTitle: (empty)\n\n[Events]\n" + EVENT_FORMAT + "\n")
    selection = [int(part) for part in args.selection.split(",") if part.strip()] if args.selection else []
    engine = LuaEngine(
        doc,
        selection=selection,
        fps=args.fps,
        keyframes=[int(part) for part in args.keyframes.split(",")] if args.keyframes else None,
        project_path=args.ass_path,
    )
    engine.load_file(args.script)
    if args.list:
        payload = {"macros": engine.list_macros(), "filters": engine.list_filters()}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    results: list[MacroResult] = []
    if args.filter:
        results.append(engine.run_filter(args.filter, dry_run=args.dry_run))
    elif args.all_filters:
        results = engine.run_filters(dry_run=args.dry_run)
    else:
        results.append(engine.run_macro(args.macro or "", dry_run=args.dry_run))
    if args.save_to and not args.dry_run:
        doc.save(args.save_to)
    payload = [result.to_dict() for result in results]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for result in payload:
            print(f"{result['kind']} {result['name']}: {'ok' if result['ok'] else 'FAILED'}")
            if result["error"]:
                print(result["error"])
            if result["log"]:
                print(result["log"])
            print(f"  lines {result['lines_before']} -> {result['lines_after']}, changed {len(result['changed_lines'])}")
    return 0 if all(result["ok"] for result in payload) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_script_cli())
