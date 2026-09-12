"""Shared state and helpers for the aegisub-mcp tool layer.

The MCP server is single-user and long-lived, so tools operate on *open
documents* held by one :class:`Workspace`.  Documents keep their original bytes
until something changes, so a file that is opened and saved untouched is
byte-identical (see :mod:`aegisub_mcp.asscore.document`).

Conventions used by every tool module in this package
-----------------------------------------------------
* Tool names are prefixed ``ass_`` and live in the module that owns the feature.
* Line indices in tool I/O are **0-based** and refer to ``doc.events()`` order,
  comments included.  ``resolve_indices()`` turns the many selection spellings
  into that list, so tools never parse user selectors themselves.
* Every tool returns a JSON-serialisable ``dict``.  User errors raise
  :class:`ToolError`; the server turns that into ``{"error": "..."}``.
* Tools must not read or write files directly — go through the workspace so
  dirty/undo bookkeeping stays correct.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..asscore import assutil as U
from ..asscore.document import AssDocument, EventEntry

__all__ = [
    "ToolError",
    "Workspace",
    "workspace",
    "ok",
    "fail",
    "resolve_indices",
    "event_dict",
    "doc_summary",
    "jsonable",
    "iter_events",
    "entry_at",
    "default_automation_dirs",
    "default_output_dir",
]

MAX_UNDO = 50
FORMAT_MARKER = "aegisub-mcp"


class ToolError(RuntimeError):
    """User-facing error; the message is returned to the caller verbatim."""


# --------------------------------------------------------------------------- ids


def _slug(path: str | os.PathLike[str]) -> str:
    name = Path(path).name or "untitled.ass"
    return name


# --------------------------------------------------------------------- workspace


class Workspace:
    """Open documents, selections and per-session settings."""

    def __init__(self) -> None:
        self._docs: dict[str, AssDocument] = {}
        self._paths: dict[str, Path | None] = {}
        self._order: list[str] = []
        self._counter = 0
        self._undo: dict[str, list[str]] = {}
        self._redo: dict[str, list[str]] = {}
        self.current: str | None = None
        self.selection: list[int] = []
        self.keyframes: list[int] = []
        self.video: dict[str, Any] | None = None
        self.audio: str | None = None
        self.automation_dirs: list[Path] = default_automation_dirs()
        self.output_dir: Path = default_output_dir()
        self.dialog_answers: dict[str, Any] | None = None
        self.script_config: dict[str, Any] = {}
        self.lock = threading.RLock()

    # -- registry ----------------------------------------------------------

    def register(self, doc: AssDocument, *, path: str | os.PathLike[str] | None = None,
                 doc_id: str | None = None) -> str:
        with self.lock:
            if doc_id is None:
                self._counter += 1
                base = _slug(path) if path else "untitled.ass"
                doc_id = f"{base}@{self._counter}" if base in self._docs else base
            self._docs[doc_id] = doc
            self._paths[doc_id] = Path(path).expanduser() if path else None
            if doc_id not in self._order:
                self._order.append(doc_id)
            self.current = doc_id
            self.selection = []
            return doc_id

    def open(self, path: str | os.PathLike[str], *, doc_id: str | None = None) -> str:
        target = Path(path).expanduser()
        if not target.is_file():
            raise ToolError(f"file not found: {target}")
        with self.lock:
            existing = self.find_by_path(target)
            if existing is not None and doc_id is None:
                self.current = existing
                return existing
            try:
                doc = AssDocument.load(str(target))
            except Exception as exc:  # noqa: BLE001
                raise ToolError(f"could not parse {target}: {exc}") from exc
            return self.register(doc, path=target, doc_id=doc_id)

    def find_by_path(self, path: str | os.PathLike[str]) -> str | None:
        target = Path(path).expanduser().resolve()
        for doc_id, stored in self._paths.items():
            if stored and stored.resolve() == target:
                return doc_id
        return None

    def new(self, *, play_res: tuple[int, int] = (1920, 1080),
            doc_id: str | None = None) -> str:
        return self.register(AssDocument.new(play_res=play_res), doc_id=doc_id)

    def ids(self) -> list[str]:
        return list(self._order)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def resolve_id(self, doc_id: str | None = None) -> str:
        if doc_id:
            if doc_id not in self._docs:
                raise ToolError(
                    f"unknown document {doc_id!r}; open documents: {', '.join(self.ids()) or '(none)'}"
                )
            return doc_id
        if self.current and self.current in self._docs:
            return self.current
        if self._order:
            self.current = self._order[0]
            return self.current
        raise ToolError("no document is open — call ass_open or ass_new first")

    def get(self, doc_id: str | None = None) -> AssDocument:
        return self._docs[self.resolve_id(doc_id)]

    def doc_id_for(self, doc: AssDocument) -> str:
        for did, item in self._docs.items():
            if item is doc:
                return did
        raise ToolError("document is not open in this workspace")

    def path(self, doc_id: str | None = None) -> Path | None:
        return self._paths.get(self.resolve_id(doc_id))

    def set_path(self, doc_id: str | None, path: str | os.PathLike[str]) -> None:
        self._paths[self.resolve_id(doc_id)] = Path(path).expanduser()

    def close(self, doc_id: str | None = None, *, save: bool = False) -> str:
        did = self.resolve_id(doc_id)
        if save:
            self.save(did)
        with self.lock:
            self._docs.pop(did, None)
            self._paths.pop(did, None)
            self._undo.pop(did, None)
            self._redo.pop(did, None)
            if did in self._order:
                self._order.remove(did)
            if self.current == did:
                self.current = self._order[0] if self._order else None
        return did

    # -- persistence -------------------------------------------------------

    def save(self, doc_id: str | None = None, path: str | os.PathLike[str] | None = None,
             *, encoding: str | None = None, bom: bool | None = None,
             newline: str | None = None, backup: bool = False) -> dict[str, Any]:
        did = self.resolve_id(doc_id)
        doc = self._docs[did]
        target = Path(path).expanduser() if path else self._paths.get(did)
        if target is None:
            raise ToolError("no path known for this document — pass a path or use save_as")
        if bom is not None:
            doc.has_bom = bool(bom)
        if newline:
            doc.newline = newline
        try:
            doc.save(str(target), encoding=encoding, create_backup=backup)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"could not save {target}: {exc}") from exc
        self._paths[did] = target
        return {"doc_id": did, "path": str(target), "bytes": target.stat().st_size,
                "dirty": bool(doc.dirty), "encoding": doc.encoding,
                "has_bom": bool(doc.has_bom), "newline": repr(doc.newline)}

    # -- undo --------------------------------------------------------------

    def snapshot(self, doc_id: str | None = None) -> str:
        did = self.resolve_id(doc_id)
        doc = self._docs[did]
        with self.lock:
            stack = self._undo.setdefault(did, [])
            stack.append(doc.snapshot())
            del stack[:-MAX_UNDO]
            self._redo[did] = []
        return did

    def undo(self, doc_id: str | None = None) -> bool:
        did = self.resolve_id(doc_id)
        stack = self._undo.get(did) or []
        if not stack:
            return False
        doc = self._docs[did]
        with self.lock:
            self._redo.setdefault(did, []).append(doc.snapshot())
            doc.restore(stack.pop())
        return True

    def redo(self, doc_id: str | None = None) -> bool:
        did = self.resolve_id(doc_id)
        stack = self._redo.get(did) or []
        if not stack:
            return False
        doc = self._docs[did]
        with self.lock:
            self._undo.setdefault(did, []).append(doc.snapshot())
            doc.restore(stack.pop())
        return True

    def undo_depth(self, doc_id: str | None = None) -> tuple[int, int]:
        did = self.resolve_id(doc_id)
        return len(self._undo.get(did) or []), len(self._redo.get(did) or [])

    # -- settings ----------------------------------------------------------

    def add_automation_dir(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path).expanduser()
        if not target.exists():
            raise ToolError(f"automation directory not found: {target}")
        if target not in self.automation_dirs:
            self.automation_dirs.append(target)
        return target

    def set_output_dir(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        self.output_dir = target
        return target


def default_automation_dirs() -> list[Path]:
    """Aegisub automation search paths: user, then XDG, then our vendored bundle."""
    home = Path.home()
    candidates = [
        home / ".aegisub" / "automation",
        home / ".config" / "aegisub" / "automation",
        Path(__file__).resolve().parent.parent / "lua" / "automation",
    ]
    return [p for p in candidates if p.exists()] or [home / ".aegisub" / "automation"]


def default_output_dir() -> Path:
    env = os.environ.get("AEGISUB_MCP_OUT")
    if env:
        return Path(env).expanduser()
    target = Path.cwd() / "aegisub-mcp-out"
    return target


workspace = Workspace()


# ------------------------------------------------------------------ selection


def _int(value: Any, *, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolError(f"{what} must be an integer, got {value!r}") from None


def iter_events(doc: AssDocument) -> list[EventEntry]:
    return doc.events()


def entry_at(doc: AssDocument, index: int) -> EventEntry:
    events = doc.events()
    if not (0 <= index < len(events)):
        raise ToolError(f"line index {index} out of range (document has {len(events)} lines)")
    return events[index]


def resolve_indices(spec: Any, doc: AssDocument, *, selection: Sequence[int] | None = None,
                    default_all: bool = True) -> list[int]:
    """Turn a selection spelling into a sorted list of unique 0-based indices.

    Accepted forms:

    ``None`` / ``"all"``      every line (or the session selection when
                              ``default_all`` is False)
    ``"selection"``           the session selection set by ``ass_select``
    ``"current"``             the first index of the session selection
    ``int`` / ``[int, ...]``  explicit 0-based indices
    ``"0-4,7"``               ranges and single indices mixed in one string
    ``{"range": [a, b]}``     inclusive range
    ``{"style": "X"}``        lines using style X (case-insensitive)
    ``{"actor": "X"}``        lines spoken by X
    ``{"effect": "X"}``       lines with effect X
    ``{"kind": "dialogue"}``  ``dialogue`` or ``comment`` only
    ``{"text_contains": s}``  plain text match (override tags ignored)
    ``{"text_regex": re}``    regular-expression match on the raw text
    ``{"drawing": true}``     lines that contain an active ``\\p`` drawing
    ``{"karaoke": true}``     lines carrying ``\\k`` tags
    ``{"start_ms": [a, b]}``  lines overlapping a time window (ms)
    ``{"cps_min": n}``/``{"cps_max": n}``/``{"duration_min": ms}``/``{"duration_max": ms}``
    ``{"empty": true}``       lines whose plain text is empty

    Conditions inside a dict are ANDed; a list of specs is the union of each
    spec's result.
    """
    from ..asscore import karaoke as K
    from ..asscore import tags as T

    events = doc.events()
    total = len(events)

    def match_condition(index: int, cond: dict) -> bool:
        entry = events[index]
        text = entry.text
        for key, value in cond.items():
            key = str(key)
            if key == "range":
                first, last = _pair(value, "range")
                if not (first <= index <= last):
                    return False
            elif key == "style":
                if entry.get("Style", "").lower() != str(value).lower():
                    return False
            elif key in ("actor", "name"):
                if entry.get("Name", "").lower() != str(value).lower():
                    return False
            elif key == "effect":
                if entry.get("Effect", "").lower() != str(value).lower():
                    return False
            elif key == "kind":
                want = str(value).lower()
                is_comment = entry.kind == "Comment"
                if want.startswith("c") and not is_comment:
                    return False
                if want.startswith("d") and is_comment:
                    return False
            elif key == "text_contains":
                needle = str(value).lower()
                if needle not in T.plain_text(text).lower():
                    return False
            elif key == "text_regex":
                import re as _re
                try:
                    pattern = _re.compile(str(value))
                except _re.error as exc:
                    raise ToolError(f"invalid text_regex: {exc}") from None
                if not pattern.search(text):
                    return False
            elif key == "drawing":
                want = bool(value)
                if bool(T.drawing_state(text) > 0) != want:
                    return False
            elif key == "karaoke":
                want = bool(value)
                if K.parse_karaoke_has_tags(text) != want:
                    return False
            elif key == "start_ms":
                lo, hi = _pair(value, "start_ms")
                if not (lo <= entry.start_ms <= hi):
                    return False
            elif key == "overlap_ms":
                lo, hi = _pair(value, "overlap_ms")
                if entry.end_ms < lo or entry.start_ms > hi:
                    return False
            elif key == "cps_min":
                if duration_cps(entry) < float(value):
                    return False
            elif key == "cps_max":
                if duration_cps(entry) > float(value):
                    return False
            elif key == "duration_min":
                if entry.duration_ms < float(value):
                    return False
            elif key == "duration_max":
                if entry.duration_ms > float(value):
                    return False
            elif key == "empty":
                if bool(not T.plain_text(text).strip()) != bool(value):
                    return False
            elif key == "index":
                if index not in _int_list(value, "index"):
                    return False
            else:
                raise ToolError(f"unknown selection condition {key!r}")
        return True

    def resolve(spec: Any) -> list[int]:
        if spec is None:
            if default_all:
                return list(range(total))
            return list(selection or [])
        if isinstance(spec, str):
            token = spec.strip()
            if token in ("", "all", "*"):
                return list(range(total))
            if token == "selection":
                return list(selection or [])
            if token == "current":
                sel = list(selection or [])
                return sel[:1]
            if token in ("none", "empty_selection"):
                return []
            return _parse_index_string(token, total)
        if isinstance(spec, bool):
            raise ToolError("selection must not be a bare boolean")
        if isinstance(spec, int):
            return [spec]
        if isinstance(spec, dict):
            if "index" in spec and len(spec) == 1:
                return _int_list(spec["index"], "index")
            if "indices" in spec and len(spec) == 1:
                return _int_list(spec["indices"], "indices")
            else:
                return [i for i in range(total) if match_condition(i, spec)]
        if isinstance(spec, (list, tuple)):
            if not spec:
                return []
            if all(isinstance(x, (int, str)) and not isinstance(x, bool) for x in spec):
                if all(isinstance(x, int) for x in spec):
                    out = [int(x) for x in spec]
                else:
                    out = []
                    for item in spec:
                        if isinstance(item, int):
                            out.append(item)
                        else:
                            out.extend(_parse_index_string(str(item), total))
                return out
            out: list[int] = []
            for item in spec:
                out.extend(resolve(item))
            return out
        raise ToolError(f"unsupported selection spec: {spec!r}")

    result: list[int] = []
    for index in resolve(spec):
        if not (0 <= index < total):
            raise ToolError(f"line index {index} out of range (document has {total} lines)")
        if index not in result:
            result.append(index)
    result.sort()
    return result


def _pair(value: Any, what: str) -> tuple[float, float]:
    if isinstance(value, dict):
        value = [value.get("lo"), value.get("hi")]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ToolError(f"{what} must be [lo, hi], got {value!r}")


def _int_list(value: Any, what: str) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [_int(v, what=what) for v in value]
    return [_int(value, what=what)]


def _parse_index_string(token: str, total: int) -> list[int]:
    out: list[int] = []
    for chunk in token.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk[1:]:
            lower, _, upper = chunk.partition("-")
            try:
                first, last = int(lower), int(upper)
            except ValueError:
                raise ToolError(f"bad range {chunk!r} in selection {token!r}") from None
            step = 1 if last >= first else -1
            out.extend(range(first, last + step, step))
        else:
            out.append(_int(chunk, what=f"index in {token!r}"))
    return out


def duration_cps(entry: EventEntry) -> float:
    """Characters per second using plain text (override tags excluded)."""
    from ..asscore import tags as T
    text = T.plain_text(entry.text).replace("\n", " ").replace("\\N", " ")
    duration = max(entry.duration_ms, 0) / 1000.0
    if duration <= 0:
        return float("inf")
    return len(text.strip()) / duration


# ------------------------------------------------------------------- serialise


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def event_dict(entry: EventEntry, *, index: int | None = None,
               full: bool = True) -> dict[str, Any]:
    from ..asscore import tags as T
    data: dict[str, Any] = {
        "kind": entry.kind,
        "start_ms": entry.start_ms,
        "end_ms": entry.end_ms,
        "duration_ms": entry.duration_ms,
        "start": entry.get("Start"),
        "end": entry.get("End"),
        "style": entry.get("Style"),
        "actor": entry.get("Name"),
        "effect": entry.get("Effect"),
        "text": entry.text,
        "plain_text": T.plain_text(entry.text),
        "comment": entry.kind == "Comment",
        "cps": round(duration_cps(entry), 2),
    }
    if full:
        data["layer"] = entry.get("Layer", entry.get("Marked", "0"))
        data["margin_l"] = entry.get("MarginL")
        data["margin_r"] = entry.get("MarginR")
        data["margin_v"] = entry.get("MarginV")
        data["drawing"] = T.drawing_state(entry.text) > 0
    if index is not None:
        data["index"] = index
    return data


def doc_summary(doc: AssDocument, doc_id: str | None = None,
                path: Any = None) -> dict[str, Any]:
    events = doc.events()
    comments = sum(1 for e in events if e.kind == "Comment")
    play_x, play_y = doc.play_res
    return {
        "doc_id": doc_id,
        "path": str(path) if path else None,
        "dirty": bool(doc.dirty),
        "script_type": doc.script_type,
        "play_res_x": play_x,
        "play_res_y": play_y,
        "fps": doc.fps(),
        "lines": len(events),
        "dialogue": len(events) - comments,
        "comments": comments,
        "styles": len(doc.styles()),
        "sections": [section.kind for section in doc.sections],
        "encoding": getattr(doc, "encoding", None),
        "has_bom": getattr(doc, "has_bom", None),
        "newline": repr(getattr(doc, "newline", "\n")),
    }


def ok(**payload: Any) -> dict[str, Any]:
    return jsonable(payload)


def fail(message: str) -> dict[str, Any]:
    return {"error": message}
