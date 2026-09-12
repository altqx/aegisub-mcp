"""Project/file management and line (event) access tools for the ASS workspace.

Every public function in this module is an MCP tool: a module-level ``ass_*``
callable with full type hints that returns a JSON-serialisable ``dict``.

Conventions used everywhere in this module
------------------------------------------

* **Indices are 0-based** and follow ``AssDocument.events()`` order, i.e. the
  order the lines appear in the file, comments included.  The same numbering is
  used by ``ass_list_lines``, ``ass_get_line``, ``ass_add_line`` and every
  selection helper.
* **Selections** are turned into indices with ``base.resolve_indices``.  The
  accepted spellings are: ``None``/``"all"`` for every line, ``"selection"``
  for the set parked by :func:`ass_select`, ``"current"`` for its first index,
  an ``int``, a list of ints, ``"0-4,7"`` style strings and condition dicts such
  as ``{"style": "Default"}``, ``{"actor": "Alice"}``, ``{"kind": "comment"}``,
  ``{"text_contains": "hi"}``, ``{"drawing": True}`` or ``{"start_ms": [a, b]}``.
* **Times** are always milliseconds.  Tools that take a time also accept an
  ``"h:mm:ss.cc"`` string (see :func:`time_to_ms`).  Times are clamped into the
  document timebase ``0 .. MAX_MS``.
* Every mutating tool calls ``workspace.snapshot()`` *before* it changes the
  document, so :func:`ass_undo` can roll it back.  User errors are raised as
  ``ToolError``; no raw library exception escapes.
"""

from __future__ import annotations

import codecs
import functools
import hashlib
import re
import time as _time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..asscore import assutil as U
from ..asscore import karaoke as K
from ..asscore import tags as T
from ..asscore.document import (
    KIND_FONTS,
    KIND_GRAPHICS,
    AssDocument,
    EventEntry,
    Section,
    read_text_with_encoding,
)
from .base import (
    ToolError,
    doc_summary,
    entry_at,
    event_dict,
    jsonable,
    ok,
    resolve_indices,
    workspace,
)

__all__ = ["register"]

#: Largest timestamp Aegisub itself accepts (9:59:59.99).  Anything above is
#: clamped so a line always stays inside the document timebase.
MAX_MS = 35_999_990

_NUMERIC_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")
_FIELD_RE = re.compile(r"^\s*(?:fontname|filename)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SRT_TS_RE = re.compile(
    r"^\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
    r"(?:\s+\S.*)?$"
)
_BLOCK_RE = re.compile(r"\{([^{}]*)\}")

_TEXT_FIELD = {"text": "Text"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def time_to_ms(value: Any, *, what: str = "time") -> int:
    """Coerce ``value`` to integer milliseconds.

    Accepts an ``int``/``float`` (already milliseconds), a plain numeric string
    (also milliseconds) or an ASS timestamp such as ``"0:00:01.50"`` /
    ``"1:02:03.45"``.  Raises ``ToolError`` for anything else.  (``tools/base``
    does not actually export a ``time_to_ms`` despite the project brief, so the
    helper lives here.)
    """
    if isinstance(value, bool):
        raise ToolError(f"{what}: expected milliseconds or a time string, got a boolean")
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    if isinstance(value, str):
        if _NUMERIC_RE.match(value):
            return int(round(float(value.strip())))
        parsed = U.try_parse_time(value)
        if parsed is None:
            raise ToolError(
                f"{what}: cannot parse {value!r} as a time "
                "(use 'h:mm:ss.cc' or plain milliseconds)"
            )
        return parsed
    raise ToolError(f"{what}: expected milliseconds or a time string, got {type(value).__name__}")


def ms_to_time(ms: int | float) -> str:
    """Format integer milliseconds as an ASS timestamp (``0:00:01.50``)."""
    return U.format_time(int(ms))


def _field_ms(value: Any, what: str) -> int:
    """``time_to_ms`` plus clamping into the document timebase."""
    return max(0, min(MAX_MS, time_to_ms(value, what=what)))


def _as_int(value: Any, default: int = 0) -> int:
    text = str(value).strip()
    if text.startswith("Marked="):          # SSA stores the layer as "Marked=N"
        text = text[len("Marked="):]
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _current_doc() -> AssDocument:
    return workspace.get(None)


def _line_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the optional field payload shared by the update tools."""
    values: dict[str, Any] = {}
    for key in ("start_ms", "end_ms"):
        if payload.get(key) is not None:
            values[key] = _field_ms(payload[key], key)
    for key in ("text", "style", "actor", "effect"):
        if payload.get(key) is not None:
            values[key] = str(payload[key])
    if payload.get("layer") is not None:
        values["layer"] = _as_int(payload["layer"])
    for key in ("margin_l", "margin_r", "margin_v"):
        if payload.get(key) is not None:
            values[key] = _as_int(payload[key])
    if payload.get("comment") is not None:
        values["comment"] = bool(payload["comment"])
    return values


def _layer_field(entry: EventEntry) -> str:
    return "Layer" if entry.has("Layer") else ("Marked" if entry.has("Marked") else "Layer")


def _layer_value(field: str, layer: int) -> str:
    """Render a layer for ``field``.

    SSA stores the value of its ``Marked`` field as ``Marked=N`` (the whole
    ``Marked=N`` token is the value), so new/edited SSA lines must use that
    spelling to stay consistent with the lines already in the file.
    """
    if field == "Marked":
        return f"Marked={layer}"
    return str(layer)


def _entry_actor(entry: EventEntry) -> str:
    if entry.has("Name"):
        return entry.get("Name")
    if entry.has("Actor"):
        return entry.get("Actor")
    return ""


def _event_lead_separator(doc: AssDocument) -> str:
    """Return the separator this document writes right after an event colon.

    Aegisub emits ``Dialogue: 0,...`` (one space) but compact files exist that
    write ``Dialogue:0,...``; both spellings mean the same thing.  The parser
    keeps the separator inside the first field of every line it reads, so this
    is only consulted for lines the tool builds from scratch.  When the document
    has no event line to copy the Aegisub spelling is used.
    """
    for other in doc.events():
        if not isinstance(other, EventEntry):
            continue
        lead = other.lead or ""
        raw = other.raw or ""
        if not lead or not raw.startswith(lead):
            continue
        tail = raw[len(lead):]
        if tail:
            return " " if tail[0] == " " else ""
    return " "


def _set_event_lead(entry: EventEntry, doc: AssDocument, kind: str) -> None:
    """Spell ``kind`` the way this document spells its event lines.

    A line parsed from a file already carries the source's separator inside its
    first field, so only the class word may be replaced there — copying the
    document's separator onto it too would double the whitespace.  A line built
    from scratch has nothing to inherit, so it adopts the file's convention.
    """
    separator = "" if entry.raw else _event_lead_separator(doc)
    entry.lead = f"{kind}:{separator}"


def _apply_update(entry: EventEntry, values: dict[str, Any]) -> list[str]:
    """Apply already-validated field values; returns the field names touched."""
    touched: list[str] = []
    if values.get("start_ms") is not None:
        entry.set("Start", ms_to_time(values["start_ms"]))
        touched.append("start_ms")
    if values.get("end_ms") is not None:
        entry.set("End", ms_to_time(values["end_ms"]))
        touched.append("end_ms")
    if values.get("text") is not None:
        entry.set("Text", values["text"])
        touched.append("text")
    if values.get("style") is not None:
        entry.set("Style", values["style"])
        touched.append("style")
    if values.get("actor") is not None:
        entry.set("Name" if entry.has("Name") else "Actor", values["actor"])
        touched.append("actor")
    if values.get("effect") is not None:
        entry.set("Effect", values["effect"])
        touched.append("effect")
    if values.get("layer") is not None:
        field = _layer_field(entry)
        entry.set(field, _layer_value(field, values["layer"]))
        touched.append("layer")
    if values.get("comment") is not None:
        kind = "Comment" if values["comment"] else "Dialogue"
        entry.kind = kind
        _set_event_lead(entry, _current_doc(), kind)
        touched.append("comment")
    for key, field in (("margin_l", "MarginL"), ("margin_r", "MarginR"), ("margin_v", "MarginV")):
        if values.get(key) is not None:
            entry.set(field, str(values[key]))
            touched.append(key)
    return touched


def _safe_cps(entry: EventEntry) -> float | None:
    seconds = entry.duration_ms / 1000.0
    if seconds <= 0:
        return None
    return round(U.reading_speed_cps(entry.text, entry.duration_ms), 2)


def _is_drawing(text: str) -> bool:
    """True when any part of ``text`` is drawn in ``\\p`` mode.

    ``tags.drawing_state`` only reports the state at the *end* of the line, which
    is ``0`` for the very common ``{\\p1}path{\\p0}`` shape — so look at every
    segment instead.
    """
    return any(T.parse(text).drawing_flags())


def _line_dict(
    entry: EventEntry,
    index: int | None = None,
    *,
    plain_text: bool = True,
    tags_summary: bool = False,
) -> dict[str, Any]:
    data = event_dict(entry, index=index, full=True)
    data["cps"] = _safe_cps(entry)
    if not plain_text:
        data.pop("plain_text", None)
    if tags_summary:
        data["tags_summary"] = jsonable(T.tag_summary(entry.text))
    return data


def _doc_and_id(doc_id: str | None) -> tuple[str, AssDocument]:
    did = workspace.resolve_id(doc_id)
    return did, workspace.get(did)


# --------------------------------------------------------------------------- #
# document construction / IO
# --------------------------------------------------------------------------- #


def _check_encoding(encoding: str) -> None:
    """Raise ``ToolError`` when ``encoding`` is not a known codec."""
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise ToolError(f"unknown encoding {encoding!r}: {exc}") from None


def _read_text_strict(target: Path, encoding: str) -> str:
    """Read ``target`` and decode it strictly with ``encoding``.

    Raises ``ToolError`` for an unknown codec, an unreadable file or bytes that
    do not decode with the requested encoding.  The library helper
    ``read_text_with_encoding`` is deliberately not used here: it decodes with
    ``errors="surrogateescape"`` and therefore never reports a bad byte.
    """
    _check_encoding(encoding)
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise ToolError(f"could not read {target}: {exc}") from None
    try:
        return data.decode(encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise ToolError(f"could not decode {target} as {encoding!r}: {exc}") from None


def _load_with_encoding(target: Path, encoding: str) -> AssDocument:
    """Load ``target`` decoding it strictly with an explicit ``encoding``.

    ``AssDocument.load`` always sniffs the encoding; when the caller insists on
    one we decode the file ourselves (see :func:`_read_text_strict`) and rebuild
    the document with the same newline / BOM / trailing-newline bookkeeping
    ``load`` uses, so byte fidelity is preserved.
    """
    text = _read_text_strict(target, encoding)

    has_bom = text.startswith("\ufeff")
    if has_bom:
        text = text[1:]
    crlf = text.count("\r\n")
    lf_only = text.count("\n") - crlf
    cr_only = text.count("\r") - crlf
    if crlf and crlf >= lf_only and crlf >= cr_only:
        newline = "\r\n"
    elif cr_only and cr_only > lf_only:
        newline = "\r"
    else:
        newline = "\n"
    trailing = text.endswith(("\n", "\r"))
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if trailing:
        normalized = normalized[:-1]
    try:
        return AssDocument(
            normalized,
            path=str(target),
            encoding=encoding,
            newline=newline,
            trailing_newline=trailing,
            has_bom=has_bom,
        )
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"could not parse {target}: {exc}") from None


def ass_new_document(
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    doc_id: str | None = None,
    script_type: str = "v4.00+",
) -> dict[str, Any]:
    """Create (and register) an empty ASS document.

    Args:
        play_res_x / play_res_y: PlayResX / PlayResY of the new script.
        doc_id: optional registry id; one is generated when omitted.
        script_type: ``"v4.00+"`` (ASS, default) or ``"v4.00"`` (SSA).

    Returns the same summary as :func:`ass_document_info` minus the document
    statistics: ``doc_id``, ``path``, ``dirty``, ``script_type``,
    ``play_res_x``, ``play_res_y``, ``fps``, ``lines``, ``dialogue``,
    ``comments``, ``styles``, ``sections``, ``encoding``, ``has_bom``,
    ``newline``.  Line indices are 0-based.
    """
    try:
        x, y = int(play_res_x), int(play_res_y)
    except (TypeError, ValueError):
        raise ToolError("play_res_x and play_res_y must be integers") from None
    if x <= 0 or y <= 0:
        raise ToolError(f"play resolution must be positive, got {x}x{y}")
    st = str(script_type).strip()
    if st not in ("v4.00+", "v4.00", "v4+", "v4"):
        raise ToolError(f"unsupported script_type {script_type!r}; use 'v4.00+' (ASS) or 'v4.00' (SSA)")
    try:
        doc = AssDocument.new(play_res=(x, y), script_type=st)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"could not create a new document: {exc}") from None
    did = workspace.register(doc, doc_id=doc_id)
    return doc_summary(doc, did, None)


def ass_open(path: str, doc_id: str | None = None, encoding: str | None = None) -> dict[str, Any]:
    """Open an ASS/SSA file and make it the current document.

    Args:
        path: file to open.  A missing file raises ``ToolError``.
        doc_id: optional registry id for the opened document.
        encoding: force this encoding instead of the automatic sniff; the file
            must decode with it or ``ToolError`` is raised.

    Returns the document summary (``doc_id``, ``path``, ``dirty``,
    ``script_type``, ``play_res_x``, ``play_res_y``, ``fps``, ``lines``,
    ``dialogue``, ``comments``, ``styles``, ``sections``, ``encoding``,
    ``has_bom``, ``newline``).  Line indices are 0-based.
    """
    target = Path(str(path)).expanduser()
    if not target.is_file():
        raise ToolError(f"file not found: {target}")
    if encoding is None:
        did = workspace.open(target, doc_id=doc_id)
    else:
        existing = workspace.find_by_path(target)
        if existing is not None and doc_id is None:
            # already open: still honour the requested encoding by checking the
            # codec exists and the file decodes with it, then keep the in-memory
            # document
            _read_text_strict(target, encoding)
            workspace.current = existing
            did = existing
        else:
            doc = _load_with_encoding(target, encoding)
            did = workspace.register(doc, path=target, doc_id=doc_id)
    doc = workspace.get(did)
    return doc_summary(doc, did, workspace.path(did))


def ass_save(
    doc_id: str | None = None,
    path: str | None = None,
    encoding: str | None = None,
    bom: bool | None = None,
    newline: str | None = None,
    create_backup: bool = False,
) -> dict[str, Any]:
    """Write a document to disk and report exactly what was written.

    Args:
        doc_id: document to save; the current one when omitted.
        path: destination; the document's known path when omitted.
        encoding: encoding override for this write.
        bom: force the UTF-8/UTF-16 BOM on/off (``None`` keeps the current one).
        newline: ``"\\n"`` / ``"\\r\\n"`` / ``"\\r"`` override.
        create_backup: copy the previous file to ``<path>.bak`` first.

    Returns ``{"doc_id", "path", "bytes_written", "sha256", "changed",
    "encoding", "has_bom", "newline", "backup"}``.  ``changed`` is true when the
    bytes written differ from whatever was on the destination beforehand (a
    brand new destination counts as changed); ``has_bom`` is derived from the
    bytes actually written.  Line indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    target_path = str(path) if path is not None else None
    probe = Path(target_path).expanduser() if target_path else workspace.path(did)
    previous: bytes | None = None
    if probe is not None and probe.is_file():
        try:
            previous = probe.read_bytes()
        except OSError as exc:
            raise ToolError(f"could not read {probe}: {exc}") from None
    try:
        result = workspace.save(
            did,
            target_path,
            encoding=encoding,
            bom=bom,
            newline=newline,
            backup=bool(create_backup),
        )
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"could not save document: {exc}") from None
    target = Path(result["path"])
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise ToolError(f"saved to {target} but could not read it back: {exc}") from None
    effective_encoding = encoding or doc.encoding
    if encoding:
        # keep the in-memory document in sync with the bytes now on disk so a
        # later save without an explicit encoding reproduces the same file
        doc.encoding = encoding
    sha = hashlib.sha256(data).hexdigest()
    bom_written = data[:3] == codecs.BOM_UTF8 or data[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)
    return ok(
        doc_id=did,
        path=str(target),
        bytes_written=len(data),
        sha256=sha,
        changed=previous != data,
        encoding=effective_encoding,
        has_bom=bom_written,
        newline=result["newline"],
        backup=f"{target}.bak" if create_backup else None,
    )


def ass_save_all() -> dict[str, Any]:
    """Save every open document that has a known path.

    Returns ``{"saved": [<result of ass_save>, ...], "count",
    "skipped": [doc_id, ...], "errors": [{"doc_id", "error"}, ...],
    "current"}``.  Documents created in memory and never saved are reported in
    ``skipped`` rather than failing the whole call.  Line indices are 0-based.
    """
    saved: list[dict[str, Any]] = []
    skipped: list[str] = []
    errors: list[dict[str, str]] = []
    for did in workspace.ids():
        if workspace.path(did) is None:
            skipped.append(did)
            continue
        try:
            saved.append(ass_save(did))
        except ToolError as exc:
            errors.append({"doc_id": did, "error": str(exc)})
    return ok(saved=saved, count=len(saved), skipped=skipped, errors=errors,
              current=workspace.current)


def ass_close(doc_id: str | None = None, save: bool = False) -> dict[str, Any]:
    """Close a document (optionally saving it first).

    Args:
        doc_id: document to close; the current one when omitted.
        save: save before closing (requires a known path).

    Returns ``{"doc_id", "closed", "saved", "path", "remaining", "current"}``.
    Line indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    path = workspace.path(did)
    if save and path is None:
        raise ToolError("no path known for this document — save it with ass_save(path=...) first")
    closed = workspace.close(did, save=bool(save))
    return ok(doc_id=closed, closed=True, saved=bool(save),
              path=str(path) if path else None, remaining=workspace.ids(),
              current=workspace.current)


def ass_list_documents() -> dict[str, Any]:
    """List the open documents.

    Returns ``{"documents": [{"doc_id", "path", "dirty", "lines", "encoding",
    "current"}...], "ids": [...], "current", "count"}``.  Line indices are
    0-based.
    """
    docs = []
    for did in workspace.ids():
        doc = workspace.get(did)
        docs.append({
            "doc_id": did,
            "path": str(workspace.path(did)) if workspace.path(did) else None,
            "dirty": bool(doc.dirty),
            "lines": len(doc.events()),
            "encoding": doc.encoding,
            "current": did == workspace.current,
        })
    return ok(documents=docs, ids=workspace.ids(), current=workspace.current, count=len(docs))


def ass_select_document(doc_id: str) -> dict[str, Any]:
    """Make ``doc_id`` the current document (clearing the session selection).

    Returns the document summary; see :func:`ass_open`.  Line indices are
    0-based.
    """
    did = workspace.resolve_id(doc_id)
    workspace.current = did
    workspace.selection = []
    doc = workspace.get(did)
    return doc_summary(doc, did, workspace.path(did))


def _attachment_names(doc: AssDocument, kind: str) -> list[str]:
    names: list[str] = []
    for line in doc.attachments(kind):
        match = _FIELD_RE.match(line)
        if match:
            names.append(match.group(1))
        elif line.strip() and not line.strip().startswith(("!!", "0!", "#!", "%#")):
            names.append(line.strip())
    return names


def ass_document_info(doc_id: str | None = None) -> dict[str, Any]:
    """Detailed information about one document.

    Args:
        doc_id: document to describe; the current one when omitted.

    Returns the document summary (see :func:`ass_open`) extended with
    ``section_kinds``, ``section_headers``, ``section_order`` (aliases of each
    other, in file order), ``style_names``, ``font_names``, ``graphic_names``,
    ``event_count``, ``comment_count``, ``dialogue_count``, ``duration_ms``
    (sum of dialogue durations), ``span_ms`` (last end minus first start),
    ``start_ms``/``end_ms`` of the first/last line, ``info`` (Script Info keys),
    ``format_order``, ``wrapping``, ``attachments`` and ``extradata_count``.
    Line indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    events = doc.events()
    comments = [e for e in events if e.is_comment]
    dialogues = [e for e in events if not e.is_comment]
    info = doc_summary(doc, did, workspace.path(did))
    starts = [e.start_ms for e in events]
    ends = [e.end_ms for e in events]
    info.update({
        "section_kinds": [s.kind for s in doc.sections],
        "section_headers": [s.header for s in doc.sections],
        "section_order": [s.header for s in doc.sections],
        "style_names": [s.name for s in doc.styles()],
        "font_names": _attachment_names(doc, KIND_FONTS),
        "graphic_names": _attachment_names(doc, KIND_GRAPHICS),
        "attachments": {
            "fonts": doc.attachments(KIND_FONTS),
            "graphics": doc.attachments(KIND_GRAPHICS),
        },
        "extradata_count": len(doc.extradata()),
        "event_count": len(events),
        "comment_count": len(comments),
        "dialogue_count": len(dialogues),
        "duration_ms": sum(max(0, e.duration_ms) for e in dialogues),
        "span_ms": (max(ends) - min(starts)) if events else 0,
        "start_ms": min(starts) if starts else 0,
        "end_ms": max(ends) if ends else 0,
        "info": doc.info_all(),
        "format_order": doc.format_order(),
        "wrapping": doc.wrapping,
    })
    return ok(**info)


# --------------------------------------------------------------------------- #
# session selection
# --------------------------------------------------------------------------- #


def ass_select(spec: Any, mode: str = "set") -> dict[str, Any]:
    """Update the session selection used by ``"selection"``.

    Args:
        spec: any selection spelling accepted by ``base.resolve_indices`` —
            ``"all"``/``None``, ``"0-4,7"``, ``5``, ``[0, 2]``,
            ``{"style": "Default"}``, ``{"text_contains": "hi"}``, ...
        mode: ``"set"`` (replace), ``"add"``, ``"remove"`` or ``"toggle"``.

    Returns ``{"doc_id", "mode", "selection": [<0-based indices>], "count"}``.
    Line indices are 0-based.
    """
    did, doc = _doc_and_id(None)
    action = str(mode).strip().lower()
    if action not in ("set", "add", "remove", "toggle"):
        raise ToolError(f"unknown selection mode {mode!r}; use set, add, remove or toggle")
    incoming = resolve_indices(spec, doc, selection=workspace.selection)
    current = list(workspace.selection)
    if action == "set":
        result = incoming
    elif action == "add":
        result = sorted(set(current) | set(incoming))
    elif action == "remove":
        result = sorted(set(current) - set(incoming))
    else:
        result = sorted(set(current) ^ set(incoming))
    workspace.selection = result
    return ok(doc_id=did, mode=action, selection=result, count=len(result))


def ass_get_selection() -> dict[str, Any]:
    """Read back the session selection.

    Returns ``{"doc_id", "selection": [<0-based indices>], "count",
    "lines": [<line dicts, same shape as ass_list_lines>]}`` and drops indices
    that are out of range for the current document.  Line indices are 0-based.
    """
    did, doc = _doc_and_id(None)
    total = len(doc.events())
    selection = [i for i in workspace.selection if 0 <= i < total]
    events = doc.events()
    return ok(
        doc_id=did,
        selection=selection,
        count=len(selection),
        lines=[_line_dict(events[i], i, tags_summary=False) for i in selection],
    )


# --------------------------------------------------------------------------- #
# reading lines
# --------------------------------------------------------------------------- #


def ass_list_lines(
    selection: Any = None,
    doc_id: str | None = None,
    offset: int = 0,
    limit: int | None = 200,
    include_tags_summary: bool = False,
    include_plain_text: bool = True,
) -> dict[str, Any]:
    """List lines with paging.

    Args:
        selection: selection spelling (see module docstring); ``None`` means all
            lines.
        doc_id: document to read; the current one when omitted.
        offset: number of selected lines to skip (0-based).
        limit: maximum number of lines to return; ``None`` returns the rest.
        include_tags_summary: add a ``tags_summary`` dict (tag counts, drawing
            flag, karaoke/clip/transform lists) per line.
        include_plain_text: include the ``plain_text`` field (tags stripped).

    Returns ``{"doc_id", "total", "offset", "limit", "returned", "indices",
    "lines"}`` where ``lines`` are ``event_dict`` style dicts: ``index``,
    ``kind``, ``start``, ``end``, ``start_ms``, ``end_ms``, ``duration_ms``,
    ``style``, ``actor``, ``effect``, ``layer``, ``margin_l/r/v``, ``text``
    (raw, tags included), ``plain_text``, ``comment``, ``drawing``, ``cps``.
    Indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    total = len(indices)
    start = _as_int(offset)
    if start < 0:
        raise ToolError("offset must be >= 0")
    if limit is None:
        window = indices[start:]
    else:
        count = _as_int(limit)
        if count < 0:
            raise ToolError("limit must be >= 0 or None")
        window = indices[start:start + count]
    events = doc.events()
    return ok(
        doc_id=did,
        total=total,
        offset=start,
        limit=None if limit is None else _as_int(limit),
        returned=len(window),
        indices=window,
        lines=[
            _line_dict(events[i], i, plain_text=bool(include_plain_text),
                       tags_summary=bool(include_tags_summary))
            for i in window
        ],
    )


def ass_get_line(index: int, doc_id: str | None = None) -> dict[str, Any]:
    """Full detail for a single line (0-based ``index``).

    Returns the ``ass_list_lines`` line dict extended with:

    * ``fields`` — every field of the line exactly as stored.
    * ``tags_summary`` — override-tag counts/structure.
    * ``drawing`` — ``{"active", "state", "segments", "drawing_segments",
      "prefix", "path", "suffix"}``.
    * ``karaoke`` — ``{"has_karaoke", "kinds", "total_ms", "syllables"}`` with
      per-syllable start/end times resolved against the line timing.
    * ``style``/``resolved_style`` — the style name, and the same name only when
      that style actually exists in the document (else ``None``).
    * ``timing`` — ``{"start_ms", "end_ms", "duration_ms", "start", "end",
      "cps", "characters", "lines"}``.

    Line indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    idx = _as_int(index)
    entry = entry_at(doc, idx)
    data = _line_dict(entry, idx, tags_summary=True)
    text = entry.text
    flags = T.parse(text).drawing_flags()
    is_drawing = any(flags)
    drawing: dict[str, Any] = {
        "active": is_drawing,
        "state": T.drawing_state(text),
        "segments": len(flags),
        "drawing_segments": len([flag for flag in flags if flag]),
    }
    if is_drawing:
        prefix, path, suffix = T.drawing_parts(text)
        drawing.update({"prefix": prefix, "path": path, "suffix": suffix})
    has_karaoke = K.parse_karaoke_has_tags(text)
    data.update({
        "doc_id": did,
        "fields": entry.fields_dict(),
        "drawing": drawing,
        "karaoke": {
            "has_karaoke": has_karaoke,
            "kinds": K.karaoke_tag_kinds(text),
            "total_ms": K.karaoke_total_ms(text),
            "syllables": (
                K.karaoke_timings(text, entry.start_ms, entry.end_ms) if has_karaoke else []
            ),
        },
        "style": entry.get("Style"),
        "resolved_style": entry.get("Style") if doc.get_style(entry.get("Style")) else None,
        "timing": {
            "start_ms": entry.start_ms,
            "end_ms": entry.end_ms,
            "duration_ms": entry.duration_ms,
            "start": entry.get("Start"),
            "end": entry.get("End"),
            "cps": _safe_cps(entry),
            "characters": len(T.plain_text(text)),
            "lines": U.line_count_of(text),
        },
    })
    return ok(**data)


# --------------------------------------------------------------------------- #
# writing lines
# --------------------------------------------------------------------------- #


def _insert_spec_keys(order: Sequence[str]) -> dict[str, str]:
    """Map generic field names onto the event format actually in use."""
    def pick(*names: str) -> str:
        for name in names:
            if name in order:
                return name
        return names[0]

    return {
        "layer": pick("Layer", "Marked"),
        "start": pick("Start"),
        "end": pick("End"),
        "style": pick("Style"),
        "actor": pick("Name", "Actor"),
        "effect": pick("Effect"),
        "margin_l": pick("MarginL"),
        "margin_r": pick("MarginR"),
        "margin_v": pick("MarginV"),
        "text": pick("Text"),
    }


def _add_one(doc: AssDocument, spec: dict[str, Any], *, what: str = "line") -> EventEntry:
    """Create one event from a spec dict (shared by add/duplicate/split/import)."""
    start_raw = spec.get("start_ms", spec.get("start"))
    end_raw = spec.get("end_ms", spec.get("end"))
    if start_raw is None or end_raw is None:
        raise ToolError(f"{what}: start and end times are required")
    start = _field_ms(start_raw, f"{what} start")
    end = _field_ms(end_raw, f"{what} end")
    if end < start:
        raise ToolError(
            f"{what}: end ({ms_to_time(end)}) is before start ({ms_to_time(start)})"
        )
    order = doc.format_order()
    keys = _insert_spec_keys(order)
    fields = {
        keys["layer"]: _layer_value(keys["layer"], _as_int(spec.get("layer", 0))),
        keys["start"]: ms_to_time(start),
        keys["end"]: ms_to_time(end),
        keys["style"]: str(spec.get("style") or "Default"),
        keys["actor"]: str(spec.get("actor", spec.get("name", "")) or ""),
        keys["margin_l"]: str(_as_int(spec.get("margin_l", 0))),
        keys["margin_r"]: str(_as_int(spec.get("margin_r", 0))),
        keys["margin_v"]: str(_as_int(spec.get("margin_v", 0))),
        keys["effect"]: str(spec.get("effect", "") or ""),
        keys["text"]: "" if spec.get("text") is None else str(spec["text"]),
    }
    try:
        entry = doc.add_event(**fields)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"{what}: could not add the line: {exc}") from None
    if bool(spec.get("comment", False)):
        # NOTE: add_event(kind="Comment") calls entry.set("Layer", ...) which
        # appends a bogus "Layer" field to SSA-style ("Marked") event lines, so
        # the kind is set afterwards instead.
        entry.kind = "Comment"
    # A line built from scratch has no source text to inherit, so it copies the
    # document's own ``Dialogue:``/``Comment:`` spelling (see _set_event_lead).
    _set_event_lead(entry, doc, entry.kind)
    doc.dirty = True
    return entry


def _index_of(doc: AssDocument, entry: EventEntry) -> int:
    for i, candidate in enumerate(doc.events()):
        if candidate is entry:
            return i
    raise ToolError("the line is no longer part of the document")


def _spec_from_entry(entry: EventEntry, **overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "start_ms": entry.start_ms,
        "end_ms": entry.end_ms,
        "text": entry.text,
        "style": entry.get("Style") or "Default",
        "actor": _entry_actor(entry),
        "effect": entry.get("Effect"),
        "layer": entry.get("Layer") if entry.has("Layer") else entry.get("Marked", "0"),
        "comment": entry.is_comment,
        "margin_l": entry.get("MarginL", "0"),
        "margin_r": entry.get("MarginR", "0"),
        "margin_v": entry.get("MarginV", "0"),
    }
    for key, value in overrides.items():
        if value is not None:
            spec[key] = value
    return spec


def ass_add_line(
    start_ms: Any,
    end_ms: Any,
    text: str,
    style: str = "Default",
    actor: str = "",
    effect: str = "",
    layer: int = 0,
    comment: bool = False,
    margin_l: int = 0,
    margin_r: int = 0,
    margin_v: int = 0,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Append a new line.

    ``start_ms``/``end_ms`` accept milliseconds or time strings such as
    ``"0:00:01.50"``; they are clamped into the document timebase and
    ``end < start`` is an error.  Lines are appended at the end of the document
    and the returned ``index`` is its **0-based** position.  The call is
    snapshot-backed, so :func:`ass_undo` reverts it.

    Returns ``{"doc_id", "index", "line": <line dict>}``.
    """
    did, doc = _doc_and_id(doc_id)
    spec = {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "style": style,
        "actor": actor,
        "effect": effect,
        "layer": layer,
        "comment": comment,
        "margin_l": margin_l,
        "margin_r": margin_r,
        "margin_v": margin_v,
    }
    # validate before snapshotting so a failed call leaves no undo step behind
    _field_ms(start_ms, "start_ms")
    _field_ms(end_ms, "end_ms")
    workspace.snapshot(did)
    entry = _add_one(doc, spec, what="add_line")
    index = _index_of(doc, entry)
    return ok(doc_id=did, index=index, line=_line_dict(entry, index))


def ass_add_lines(lines: list[dict[str, Any]], doc_id: str | None = None) -> dict[str, Any]:
    """Append several lines in one call.

    ``lines`` is a list of dicts using the :func:`ass_add_line` keys
    (``start_ms``/``end_ms``/``text``/``style``/``actor``/``effect``/``layer``/
    ``comment``/``margin_l``/``margin_r``/``margin_v``) plus an optional
    ``index`` — the **0-based** position the new line is inserted at.  Insertion
    positions refer to the line list as it is being built (the previous inserts
    of this same call are already in place); lines without ``index`` go to the
    end.  All times accept milliseconds or time strings.

    Returns ``{"doc_id", "indices": [<0-based index per input dict>],
    "count", "lines": [<line dict>, ...]}``.
    """
    did, doc = _doc_and_id(doc_id)
    if not isinstance(lines, list) or not lines:
        raise ToolError("lines must be a non-empty list of dicts")
    specs: list[tuple[dict[str, Any], int | None, str]] = []
    for position, item in enumerate(lines):
        what = f"lines[{position}]"
        if not isinstance(item, dict):
            raise ToolError(f"{what} must be a dict")
        spec = dict(item)
        insert_at = spec.pop("index", None)
        _field_ms(spec.get("start_ms", spec.get("start")), f"{what} start")
        _field_ms(spec.get("end_ms", spec.get("end")), f"{what} end")
        specs.append((spec, None if insert_at is None else _as_int(insert_at), what))
    workspace.snapshot(did)
    order = list(doc.events())
    created: list[EventEntry] = []
    for spec, insert_at, what in specs:
        entry = _add_one(doc, spec, what=what)
        if insert_at is None:
            order.append(entry)
        else:
            position = max(0, min(insert_at, len(order)))
            order.insert(position, entry)
        created.append(entry)
    doc.reorder_events(order)
    indices = [_index_of(doc, entry) for entry in created]
    events = doc.events()
    return ok(doc_id=did, indices=indices, count=len(indices),
              lines=[_line_dict(events[i], i) for i in indices])


def ass_update_line(
    index: int,
    doc_id: str | None = None,
    start_ms: Any = None,
    end_ms: Any = None,
    text: str | None = None,
    style: str | None = None,
    actor: str | None = None,
    effect: str | None = None,
    layer: int | None = None,
    comment: bool | None = None,
    margin_l: int | None = None,
    margin_r: int | None = None,
    margin_v: int | None = None,
) -> dict[str, Any]:
    """Change individual fields of one line; only the fields you pass change.

    ``index`` is **0-based**.  Times accept milliseconds or time strings.
    ``comment=True`` turns the line into a ``Comment:``, ``False`` back into a
    ``Dialogue:``.  The call is snapshot-backed (:func:`ass_undo` reverts it).

    Returns ``{"doc_id", "index", "changed": [field names], "line": <dict>}``.
    """
    did, doc = _doc_and_id(doc_id)
    idx = _as_int(index)
    entry = entry_at(doc, idx)
    payload = {
        "start_ms": start_ms, "end_ms": end_ms, "text": text, "style": style,
        "actor": actor, "effect": effect, "layer": layer, "comment": comment,
        "margin_l": margin_l, "margin_r": margin_r, "margin_v": margin_v,
    }
    values = _line_values(payload)
    if not values:
        raise ToolError("nothing to update — pass at least one field")
    new_start = values.get("start_ms", entry.start_ms)
    new_end = values.get("end_ms", entry.end_ms)
    if new_end < new_start:
        raise ToolError(
            f"end ({ms_to_time(new_end)}) is before start ({ms_to_time(new_start)})"
        )
    workspace.snapshot(did)
    touched = _apply_update(entry, values)
    doc.dirty = True
    return ok(doc_id=did, index=idx, changed=touched, line=_line_dict(entry, idx))


def ass_update_lines(
    selection: Any,
    doc_id: str | None = None,
    start_ms: Any = None,
    end_ms: Any = None,
    text: str | None = None,
    style: str | None = None,
    actor: str | None = None,
    effect: str | None = None,
    layer: int | None = None,
    comment: bool | None = None,
    margin_l: int | None = None,
    margin_r: int | None = None,
    margin_v: int | None = None,
    pad_ms: int | None = None,
    clamp_to_avoid_overlap: bool = False,
) -> dict[str, Any]:
    """Change the same fields on every selected line.

    Args:
        selection: selection spelling (see module docstring); ``None`` = all.
        doc_id: document to edit; the current one when omitted.
        start_ms/end_ms/text/style/actor/effect/layer/comment/margin_l/margin_r/
            margin_v: applied only to the lines you pass, as in
            :func:`ass_update_line`.
        pad_ms: widen every selected line by this many ms on both sides
            (``start -= pad_ms``, ``end += pad_ms``); negative values trim it.
        clamp_to_avoid_overlap: after the edit, pull each selected line's times
            back so it no longer overlaps an *unselected* line (best effort;
            worst case a 10 ms line is left in the gap).

    Returns ``{"doc_id", "changed": [<0-based indices>], "count",
    "clamped": [<0-based indices>], "fields": [field names],
    "lines": [<dict>, ...]}``.  Snapshot-backed.  Line indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    payload = {
        "start_ms": start_ms, "end_ms": end_ms, "text": text, "style": style,
        "actor": actor, "effect": effect, "layer": layer, "comment": comment,
        "margin_l": margin_l, "margin_r": margin_r, "margin_v": margin_v,
    }
    values = _line_values(payload)
    pad = None if pad_ms is None else _as_int(pad_ms)
    if not values and pad is None and not clamp_to_avoid_overlap:
        raise ToolError("nothing to update — pass at least one field, pad_ms or clamp_to_avoid_overlap")
    events = doc.events()
    targets = [events[i] for i in indices]
    for entry in targets:
        new_start = values.get("start_ms", entry.start_ms)
        new_end = values.get("end_ms", entry.end_ms)
        if pad is not None:
            new_start = max(0, new_start - pad)
            new_end = min(MAX_MS, max(new_start, new_end + pad))
        if new_end < new_start:
            raise ToolError(
                f"line {_index_of(doc, entry)}: end ({ms_to_time(new_end)}) is before "
                f"start ({ms_to_time(new_start)})"
            )
    workspace.snapshot(did)
    changed: list[int] = []
    fields: list[str] = []
    for entry in targets:
        line_values = dict(values)
        if pad is not None:
            line_values["start_ms"] = max(0, entry.start_ms - pad)
            line_values["end_ms"] = min(MAX_MS, max(line_values["start_ms"], entry.end_ms + pad))
        touched = _apply_update(entry, line_values)
        if touched:
            changed.append(_index_of(doc, entry))
            for name in touched:
                if name not in fields:
                    fields.append(name)
    doc.dirty = True
    clamped: list[int] = []
    if clamp_to_avoid_overlap:
        clamped = _clamp_overlaps(doc, indices)
    fresh = doc.events()
    return ok(
        doc_id=did,
        changed=sorted(changed),
        count=len(changed),
        clamped=clamped,
        fields=fields,
        lines=[_line_dict(fresh[i], i) for i in indices],
    )


def _clamp_overlaps(doc: AssDocument, indices: Sequence[int]) -> list[int]:
    """Shrink selected lines so they stop overlapping *any* other line.

    Every other event counts as a boundary — including the other selected lines,
    so a selected run is trimmed against itself and the trailing line of a run is
    left alone when nothing follows it.
    """
    events = doc.events()
    touched: list[int] = []
    for i in sorted(set(indices)):
        if not 0 <= i < len(events):
            continue
        entry = events[i]
        start, end = entry.start_ms, entry.end_ms
        new_start, new_end = start, end
        for j, other in enumerate(events):
            if j == i:
                continue
            other_start, other_end = other.start_ms, other.end_ms
            if other_start >= end or other_end <= start:
                continue
            if other_start <= new_start:
                # the other line starts at or before ours: push our start past it
                new_start = max(new_start, other_end)
            elif other_end >= new_end:
                # the other line ends at or after ours: pull our end back to it
                new_end = min(new_end, other_start)
            else:
                # the other line sits strictly inside ours; the overlap cannot be
                # removed without dropping our line, so trim the front and leave it
                new_start = max(new_start, other_end)
        if new_start >= new_end:
            new_end = min(MAX_MS, new_start + 10)
        if (new_start, new_end) != (start, end):
            entry.set("Start", ms_to_time(new_start))
            entry.set("End", ms_to_time(new_end))
            touched.append(i)
    if touched:
        doc.dirty = True
    return touched


def ass_delete_lines(selection: Any, doc_id: str | None = None) -> dict[str, Any]:
    """Delete the selected lines.

    ``selection`` is any selection spelling; ``None`` means every line (use with
    care).  Returns ``{"doc_id", "deleted": [<0-based indices>], "count",
    "remaining"}``.  Snapshot-backed.
    """
    did, doc = _doc_and_id(doc_id)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    if not indices:
        raise ToolError("nothing selected — the selection is empty")
    workspace.snapshot(did)
    removed = doc.remove_events(indices)
    return ok(doc_id=did, deleted=indices, count=removed, remaining=len(doc.events()))


def ass_duplicate_lines(
    selection: Any,
    doc_id: str | None = None,
    offset_ms: int = 0,
    insert_after: bool = True,
) -> dict[str, Any]:
    """Duplicate the selected lines.

    Args:
        selection: selection spelling; ``None`` = all lines.
        offset_ms: shift the copies in time by this many ms.
        insert_after: put each copy right after (``True``) or right before
            (``False``) its source.

    Copies keep every field of the source (style, actor, effect, layer, margins,
    comment flag, tags).  Returns ``{"doc_id", "indices": [<0-based index of
    each copy>], "count", "lines": [...]}``.  Snapshot-backed.  Indices are
    0-based.
    """
    did, doc = _doc_and_id(doc_id)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    if not indices:
        raise ToolError("nothing selected — the selection is empty")
    offset = _as_int(offset_ms)
    events = doc.events()
    sources = [events[i] for i in indices]
    workspace.snapshot(did)
    order = list(events)
    created: list[EventEntry] = []
    for source in sources:
        start = max(0, min(MAX_MS, source.start_ms + offset))
        end = max(start, min(MAX_MS, source.end_ms + offset))
        spec = _spec_from_entry(source, start_ms=start, end_ms=end)
        entry = _add_one(doc, spec, what="duplicate")
        position = order.index(source) + (1 if insert_after else 0)
        order.insert(position, entry)
        created.append(entry)
    doc.reorder_events(order)
    new_indices = sorted(_index_of(doc, entry) for entry in created)
    fresh = doc.events()
    return ok(doc_id=did, indices=new_indices, count=len(new_indices),
              lines=[_line_dict(fresh[i], i) for i in new_indices])


def ass_move_lines(selection: Any, target_index: int, doc_id: str | None = None) -> dict[str, Any]:
    """Move the selected lines to another position in the line order.

    ``target_index`` is the **0-based** slot in the *remaining* lines (the
    selection is removed first) where the moved block is inserted; it is clamped
    to the list length, so a large value appends.  Returns ``{"doc_id",
    "moved": [old 0-based indices], "target_index", "selection": [new 0-based
    indices], "count"}``.  Snapshot-backed.
    """
    did, doc = _doc_and_id(doc_id)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    if not indices:
        raise ToolError("nothing selected — the selection is empty")
    target = _as_int(target_index)
    events = doc.events()
    picked = [events[i] for i in indices]
    rest = [e for i, e in enumerate(events) if i not in set(indices)]
    position = max(0, min(target, len(rest)))
    workspace.snapshot(did)
    doc.reorder_events(rest[:position] + picked + rest[position:])
    new_indices = sorted(_index_of(doc, entry) for entry in picked)
    return ok(doc_id=did, moved=indices, target_index=position,
              selection=new_indices, count=len(new_indices))


# --------------------------------------------------------------------------- #
# structural line edits
# --------------------------------------------------------------------------- #


def _override_prefix(text: str, raw_index: int) -> str:
    """Render every override block that sits before ``raw_index``.

    Only tag blocks count as state to carry over, so a plain-text prefix yields
    an empty string while ``{\\i1}abc{\\i0}`` split after ``abc`` yields
    ``{\\i1}{\\i0}`` (the net effect is the same as the original line).
    """
    parts: list[str] = []
    cursor = 0
    for segment in T.parse(text).segments:
        if cursor >= raw_index:
            break
        if isinstance(segment, T.TagBlock):
            parts.append(segment.render())
        else:
            cursor += len(segment.render())
    return "".join(parts)


def _split_text(text: str, at_ms: int, start_ms: int, end_ms: int) -> tuple[str, str]:
    """Split ``text`` at the visible character matching the time fraction.

    Returns ``(head, tail)``: ``head`` keeps the original text up to the cut,
    ``tail`` repeats every override block that preceded the cut so the second
    half starts with the same styling state.
    """
    positions = T.char_positions(text)
    if not positions:
        return text, ""
    total = len(positions)
    span = max(1, end_ms - start_ms)
    fraction = (at_ms - start_ms) / span
    cut = int(round(total * fraction))
    # keep at least one visible character on each side when the line allows it
    cut = max(1, min(cut, total - 1)) if total > 1 else total
    raw_index = positions[cut][0] if cut < total else len(text)
    head, tail = text[:raw_index], text[raw_index:]
    prefix = _override_prefix(text, raw_index)
    if prefix and not tail.startswith(prefix):
        tail = prefix + tail
    return head, tail


def ass_split_line(index: int, at_ms: Any, doc_id: str | None = None) -> dict[str, Any]:
    """Split one line in two at a time strictly inside its span.

    ``index`` is **0-based**.  ``at_ms`` accepts milliseconds or a time string
    and must satisfy ``start < at_ms < end``, otherwise ``ToolError``.  The cut
    sits at the visible character whose position matches the time fraction (each
    half keeps at least one visible character).  The first half keeps the
    original line's text up to the cut; the second half is a new line right
    after it that repeats every override block preceding the cut (so ``{\\i1}``
    prefixes survive) and keeps style, actor, effect, layer and margins.

    Returns ``{"doc_id", "index", "second_index", "at_ms", "first": <dict>,
    "second": <dict>}``.  Snapshot-backed.
    """
    did, doc = _doc_and_id(doc_id)
    idx = _as_int(index)
    entry = entry_at(doc, idx)
    start, end = entry.start_ms, entry.end_ms
    at = _field_ms(at_ms, "at_ms")
    if not (start < at < end):
        raise ToolError(
            f"at_ms {ms_to_time(at)} is outside the line span "
            f"{ms_to_time(start)}..{ms_to_time(end)}"
        )
    head, tail = _split_text(entry.text, at, start, end)
    before = doc.events()
    workspace.snapshot(did)
    entry.set("End", ms_to_time(at))
    entry.set("Text", head)
    doc.dirty = True
    spec = _spec_from_entry(entry, start_ms=at, end_ms=end, text=tail)
    second = _add_one(doc, spec, what="split_line")
    order = list(before)
    position = order.index(entry)
    order.insert(position + 1, second)
    doc.reorder_events(order)
    first_index = _index_of(doc, entry)
    second_index = _index_of(doc, second)
    events = doc.events()
    return ok(
        doc_id=did,
        index=first_index,
        second_index=second_index,
        at_ms=at,
        first=_line_dict(events[first_index], first_index),
        second=_line_dict(events[second_index], second_index),
    )


def ass_merge_lines(selection: Any, separator: str = r"\N", doc_id: str | None = None) -> dict[str, Any]:
    """Merge the selected lines into the earliest one.

    The selected lines are sorted by start (then end) time and their raw texts
    are joined with ``separator`` (default the literal ``\\N``) into the
    earliest-starting line, which keeps its style, actor, effect, layer and
    margins; its span becomes the union of the merged spans.  The other
    selected lines are removed.  Needs at least two lines.

    Returns ``{"doc_id", "index": <0-based index of the merged line *after*
    the merge>, "removed": [<pre-merge 0-based indices of the lines folded
    away>], "removed_count", "start_ms", "end_ms", "text", "line": <dict>}``.
    Snapshot-backed.
    """
    did, doc = _doc_and_id(doc_id)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    if len(indices) < 2:
        raise ToolError(f"select at least two lines to merge (got {len(indices)})")
    events = doc.events()
    pairs = sorted(((i, events[i]) for i in indices), key=lambda p: (p[1].start_ms, p[1].end_ms, p[0]))
    target = pairs[0][1]
    start = min(entry.start_ms for _, entry in pairs)
    end = max(entry.end_ms for _, entry in pairs)
    separator = r"\N" if separator is None else str(separator)
    merged_text = separator.join(entry.text for _, entry in pairs)
    workspace.snapshot(did)
    target.set("Start", ms_to_time(start))
    target.set("End", ms_to_time(end))
    target.set("Text", merged_text)
    doc.dirty = True
    removed = doc.remove_events([i for i, _ in pairs[1:]])
    index = _index_of(doc, target)
    fresh = doc.events()
    return ok(doc_id=did, index=index, removed=[i for i, _ in pairs[1:]],
              removed_count=removed, start_ms=start, end_ms=end, text=merged_text,
              line=_line_dict(fresh[index], index))


# --------------------------------------------------------------------------- #
# bulk text edits
# --------------------------------------------------------------------------- #

_FIELD_MAP = {
    "text": "Text",
    "start": "Start",
    "end": "End",
    "style": "Style",
    "actor": "Name",
    "name": "Name",
    "effect": "Effect",
}


def ass_find_replace(
    pattern: str,
    replacement: str,
    selection: Any = None,
    doc_id: str | None = None,
    regex: bool = False,
    case_sensitive: bool = True,
    fields: Sequence[str] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Find and replace text across the selected lines.

    Args:
        pattern: literal text (``regex=False``) or a Python regular expression
            (``regex=True``, compiled with ``re.UNICODE``; back-references like
            ``\\1`` work in ``replacement``).
        replacement: replacement string (used literally when ``regex=False``).
        selection: selection spelling; ``None`` = every line.
        doc_id: document to edit; the current one when omitted.
        regex: treat ``pattern`` as a regular expression.
        case_sensitive: ``False`` makes the match case-insensitive.
        fields: which fields to touch; defaults to ``["text"]``.  Any of
            ``text``, ``start``, ``end``, ``style``, ``actor``/``name``,
            ``effect``.
        dry_run: count only, change nothing.
        limit: stop after this many replacements in total.

    Returns ``{"doc_id", "pattern", "replacement", "fields", "regex",
    "dry_run", "total": <replacements made>, "changed": [<0-based indices>],
    "lines": [{"index", "counts": {field: n}, "count", "text"}...],
    "limit"}``.  Snapshot-backed unless ``dry_run``.  Indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    if pattern is None or str(pattern) == "":
        raise ToolError("pattern must not be empty")
    wanted = [str(f).strip().lower() for f in (fields or ["text"])]
    unknown = [f for f in wanted if f not in _FIELD_MAP]
    if unknown:
        raise ToolError(
            f"unknown field(s) {', '.join(sorted(set(unknown)))}; "
            f"use {', '.join(sorted(set(_FIELD_MAP)))}"
        )
    cap: int | None = None
    if limit is not None:
        cap = _as_int(limit)
        if cap < 0:
            raise ToolError("limit must be >= 0 or None")
    flags = re.UNICODE | (0 if case_sensitive else re.IGNORECASE)
    if regex:
        try:
            compiled = re.compile(str(pattern), flags)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from None
    else:
        compiled = re.compile(re.escape(str(pattern)), flags)
    substitute = str(replacement) if regex else (lambda _match: str(replacement))

    indices = resolve_indices(selection, doc, selection=workspace.selection)
    events = doc.events()
    plan: list[tuple[int, str, str, int]] = []  # (index, field_key, new_value, count)
    total = 0
    for i in indices:
        entry = events[i]
        for key in wanted:
            field = _FIELD_MAP[key]
            value = entry.get(field)
            if not value:
                continue
            found = len(compiled.findall(value))
            if not found:
                continue
            allowed = found
            if cap is not None:
                if total >= cap:
                    continue
                allowed = min(found, cap - total)
            if allowed <= 0:
                continue
            if dry_run:
                plan.append((i, key, value, allowed))
                total += allowed
                continue
            new_value, applied = compiled.subn(substitute, value, count=allowed)
            plan.append((i, key, new_value, applied))
            total += applied
    changed: list[int] = []
    per_line: dict[int, dict[str, int]] = {}
    new_text: dict[int, str] = {}
    for i, key, value, count in plan:
        per_line.setdefault(i, {})
        per_line[i][key] = per_line[i].get(key, 0) + count
    # A dry run reports each line's current text on purpose: nothing has been
    # written yet, so there is no new text to show.  The real run below replaces
    # it with the post-edit text read back from the document.
    if not dry_run and total:
        workspace.snapshot(did)
        for i, key, value, _count in plan:
            events[i].set(_FIELD_MAP[key], value)
            if i not in changed:
                changed.append(i)
        doc.dirty = True
        fresh = doc.events()
        new_text = {i: fresh[i].text for i in changed}
    created: list[dict[str, Any]] = []
    for i in sorted(per_line):
        text = new_text.get(i, events[i].text)
        created.append({
            "index": i,
            "counts": per_line[i],
            "count": sum(per_line[i].values()),
            "text": text,
        })
    return ok(
        doc_id=did,
        pattern=str(pattern),
        replacement=str(replacement),
        fields=wanted,
        regex=bool(regex),
        case_sensitive=bool(case_sensitive),
        dry_run=bool(dry_run),
        total=total,
        changed=sorted(changed),
        lines=created,
        limit=cap,
    )


def ass_set_comment(
    selection: Any,
    comment: bool = True,
    drop: bool = False,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Convert lines between ``Dialogue:`` and ``Comment:``.

    Args:
        selection: selection spelling; ``None`` = every line.
        comment: ``True`` converts to ``Comment:``, ``False`` back to
            ``Dialogue:``.
        drop: delete the selected lines instead of converting them.

    Returns ``{"doc_id", "changed": [<0-based indices>], "count", "comment",
    "dropped"}``.  Snapshot-backed.  Indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    if not indices:
        raise ToolError("nothing selected — the selection is empty")
    workspace.snapshot(did)
    if drop:
        removed = doc.remove_events(indices)
        return ok(doc_id=did, changed=indices, count=removed, comment=bool(comment), dropped=True)
    events = doc.events()
    kind = "Comment" if comment else "Dialogue"
    changed: list[int] = []
    for i in indices:
        entry = events[i]
        if entry.kind == kind:
            continue
        entry.kind = kind
        _set_event_lead(entry, doc, kind)
        changed.append(i)
    if changed:
        doc.dirty = True
    return ok(doc_id=did, changed=changed, count=len(changed), comment=bool(comment), dropped=False)


def ass_sort_lines(
    selection: Any = None,
    doc_id: str | None = None,
    keys: Sequence[str] | None = None,
    reverse: bool = False,
) -> dict[str, Any]:
    """Sort the selected lines in place (``document.reorder_events``).

    Args:
        selection: selection spelling; ``None`` = every line.
        keys: sort keys, applied in order; defaults to ``["start", "end"]``.
            One of ``start``, ``end``, ``duration``, ``style``, ``actor``/
            ``name``, ``effect``, ``layer``, ``text``, ``comment``.
        reverse: sort descending.

    Only the selected lines move: they are re-shuffled among the slots they
    already occupy, so unselected lines keep their positions.

    Returns ``{"doc_id", "sorted", "selection": [<0-based indices sorted>],
    "count", "moved": {old 0-based index: new 0-based index}}``.
    """
    did, doc = _doc_and_id(doc_id)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    if len(indices) < 2:
        return ok(doc_id=did, sorted=False, selection=indices, count=len(indices), moved={})
    use_keys = [str(k).strip().lower() for k in (keys or ["start", "end"])]
    events = list(doc.events())

    def key_of(entry: EventEntry) -> tuple:
        parts: list[Any] = []
        for name in use_keys:
            if name == "start":
                parts.append(entry.start_ms)
            elif name == "end":
                parts.append(entry.end_ms)
            elif name == "duration":
                parts.append(entry.duration_ms)
            elif name == "style":
                parts.append(entry.get("Style").lower())
            elif name in ("actor", "name"):
                parts.append(_entry_actor(entry).lower())
            elif name == "effect":
                parts.append(entry.get("Effect").lower())
            elif name == "layer":
                parts.append(_as_int(entry.get("Layer", entry.get("Marked", "0"))))
            elif name == "text":
                parts.append(T.plain_text(entry.text).lower())
            elif name == "comment":
                parts.append(1 if entry.is_comment else 0)
            else:
                raise ToolError(
                    f"unknown sort key {name!r}; use start, end, duration, style, "
                    "actor, effect, layer, text or comment"
                )
        return tuple(parts)

    chosen = sorted((events[i] for i in indices), key=key_of, reverse=bool(reverse))
    workspace.snapshot(did)
    order = list(events)
    for position, entry in zip(indices, chosen):
        order[position] = entry
    doc.reorder_events(order)
    fresh = doc.events()
    new_position = {id(entry): i for i, entry in enumerate(fresh)}
    moved = {old: new_position[id(events[old])] for old in indices}
    return ok(doc_id=did, sorted=True, selection=indices, count=len(indices), moved=moved)


# --------------------------------------------------------------------------- #
# undo / redo
# --------------------------------------------------------------------------- #


def ass_undo(doc_id: str | None = None) -> dict[str, Any]:
    """Undo the last snapshot-backed change of a document.

    Returns ``{"doc_id", "undone", "undo_depth", "redo_depth", "dirty"}``.
    ``undone`` is ``False`` when the undo stack is empty.  Line indices are
    0-based.
    """
    did, doc = _doc_and_id(doc_id)
    undone = workspace.undo(did)
    depth = workspace.undo_depth(did)
    return ok(doc_id=did, undone=undone, undo_depth=depth[0],
              redo_depth=depth[1], dirty=bool(workspace.get(did).dirty))


def ass_redo(doc_id: str | None = None) -> dict[str, Any]:
    """Redo the last undone change.

    Returns ``{"doc_id", "redone", "undo_depth", "redo_depth", "dirty"}``;
    ``redone`` is ``False`` when the redo stack is empty.  Line indices are
    0-based.
    """
    did, doc = _doc_and_id(doc_id)
    redone = workspace.redo(did)
    depth = workspace.undo_depth(did)
    return ok(doc_id=did, redone=redone, undo_depth=depth[0],
              redo_depth=depth[1], dirty=bool(workspace.get(did).dirty))


def ass_undo_history(doc_id: str | None = None) -> dict[str, Any]:
    """List the undo steps available for a document.

    The workspace keeps raw document snapshots and no labels, so the entries are
    positional: step 1 is the **oldest** available snapshot, the highest step is
    what :func:`ass_undo` would restore next.  Each entry carries a summary of
    the snapshot (line count, first start time, line count difference from the
    current state) so it is still possible to tell the steps apart.

    Returns ``{"doc_id", "undo_depth", "redo_depth", "entries": [{"step",
    "label", "lines", "first_start"}...], "redo_entries": [...],
    "next_undo", "next_redo"}``.  Line indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    undo_stack = list(getattr(workspace, "_undo", {}).get(did) or [])
    redo_stack = list(getattr(workspace, "_redo", {}).get(did) or [])

    def describe(index: int, snapshot: str, prefix: str) -> dict[str, Any]:
        parsed = AssDocument.from_text(snapshot)
        events = parsed.events()
        return {
            "step": index + 1,
            "label": f"{prefix} step {index + 1}",
            "lines": len(events),
            "first_start": ms_to_time(events[0].start_ms) if events else None,
        }

    entries = [describe(i, snap, "undo") for i, snap in enumerate(undo_stack)]
    redo_entries = [describe(i, snap, "redo") for i, snap in enumerate(redo_stack)]
    current_lines = len(doc.events())
    return ok(
        doc_id=did,
        undo_depth=len(undo_stack),
        redo_depth=len(redo_stack),
        entries=entries,
        redo_entries=redo_entries,
        labels=[e["label"] for e in entries],
        redo_labels=[e["label"] for e in redo_entries],
        current_lines=current_lines,
        next_undo=entries[-1]["label"] if entries else None,
        next_redo=redo_entries[-1]["label"] if redo_entries else None,
    )


# --------------------------------------------------------------------------- #
# export / import
# --------------------------------------------------------------------------- #


def _srt_time(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _export_slug(doc: AssDocument, did: str) -> str:
    name = Path(doc.path).stem if doc.path else Path(did).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name) or "document"


def ass_export_text(
    selection: Any = None,
    doc_id: str | None = None,
    format: str = "txt",
    line_separator: str = r"\N",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Export selected lines as plain text, TSV, SRT or an ASS fragment.

    Args:
        selection: selection spelling; ``None`` = every line.
        doc_id: document to export; the current one when omitted.
        format: ``"txt"`` (one line of plain text each), ``"tsv"`` (start, end,
            style, actor, effect, kind, plain text), ``"srt"`` (SubRip) or
            ``"ass-fragment"`` (the raw ``Dialogue:``/``Comment:`` lines).
        line_separator: replacement for the ASS in-line hard break ``\\N`` (and
            ``\\n``) in ``txt``/``tsv`` output.  The default keeps the ASS escape
            for lossless re-import; pass ``"\\n"`` for one output line per
            visual line, or any other string (``"|"``) to flatten explicitly.
            ``srt`` always uses real newlines (required by the format) and
            ``ass-fragment`` keeps the original text untouched.
        output_path: file to write; when omitted the file is written into
            ``workspace.output_dir`` (created if needed).

    ``txt`` and ``srt`` skip comment lines; ``tsv`` and ``ass-fragment`` keep
    them.  Returns ``{"doc_id", "path", "format", "media_type", "text", "bytes",
    "line_count", "selection": [<0-based indices>]}`` where ``line_count`` is the
    number of exported records (SRT blocks, TSV rows, text lines, events).
    """
    did, doc = _doc_and_id(doc_id)
    fmt = str(format).strip().lower()
    if fmt not in ("txt", "tsv", "srt", "ass-fragment", "ass_fragment", "fragment"):
        raise ToolError(f"unknown export format {format!r}; use txt, tsv, srt or ass-fragment")
    if fmt == "ass_fragment":
        fmt = "ass-fragment"
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    events = doc.events()
    separator = r"\N" if line_separator is None else str(line_separator)

    def plain(text: str) -> str:
        """Visible text with override tags gone and ``\\h`` as a space."""
        return T.plain_text(text).replace(r"\h", " ")

    def wrapped(text: str) -> str:
        """Plain text with the ASS hard break turned into a real newline (SRT)."""
        return plain(text).replace(r"\N", "\n").replace(r"\n", "\n")

    def separated(text: str) -> str:
        """Plain text with in-line breaks replaced by ``line_separator``."""
        return plain(text).replace(r"\N", separator).replace(r"\n", separator)

    def flattened(text: str) -> str:
        """Plain text squeezed onto one TSV field using ``separator``."""
        joined = separated(text)
        return joined.replace("\t", " ").replace("\n", " ")

    chunks: list[str] = []
    line_count = 0
    if fmt == "txt":
        for i in indices:
            entry = events[i]
            if entry.is_comment:
                continue
            chunks.append(separated(entry.text))
            line_count += 1
        text = "\n".join(chunks)
        extension, media = "txt", "text/plain"
    elif fmt == "tsv":
        rows = ["start\tend\tstyle\tactor\teffect\tkind\ttext"]
        for i in indices:
            entry = events[i]
            rows.append("\t".join([
                entry.get("Start"), entry.get("End"), entry.get("Style"),
                _entry_actor(entry), entry.get("Effect"), entry.kind,
                flattened(entry.text),
            ]))
            line_count += 1
        text = "\n".join(rows)
        extension, media = "tsv", "text/tab-separated-values"
    elif fmt == "srt":
        blocks: list[str] = []
        for i in indices:
            entry = events[i]
            if entry.is_comment:
                continue
            line_count += 1
            body = wrapped(entry.text)
            blocks.append(
                f"{line_count}\n{_srt_time(entry.start_ms)} --> {_srt_time(entry.end_ms)}\n{body}"
            )
        text = "\n\n".join(blocks)
        extension, media = "srt", "application/x-subrip"
    else:  # ass-fragment
        chunks = [events[i].render() for i in indices]
        line_count = len(chunks)
        text = "\n".join(chunks)
        extension, media = "ass", "text/plain"
    if text:
        text = text + "\n"

    if output_path is None:
        target = Path(workspace.output_dir) / f"export-{_export_slug(doc, did)}-{int(_time.time())}.{extension}"
    else:
        target = Path(str(output_path)).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="")
    except OSError as exc:
        raise ToolError(f"could not write {target}: {exc}") from None
    return ok(
        doc_id=did,
        path=str(target),
        format=fmt,
        media_type=media,
        text=text,
        bytes=len(text.encode("utf-8")),
        line_count=line_count,
        selection=indices,
    )


def ass_import_srt(
    path: str,
    doc_id: str | None = None,
    style: str = "Default",
    offset_ms: int = 0,
) -> dict[str, Any]:
    """Import a SubRip (``.srt``) file as new ASS lines.

    Handles ``,`` and ``.`` decimal separators, multi-line blocks, CRLF and a
    UTF-8 BOM; multi-line subtitle bodies become ``\\N`` hard breaks.

    Args:
        path: the ``.srt`` file to read.
        doc_id: document to append to; the current one when omitted.
        style: ASS style for the imported lines.
        offset_ms: shift every imported line in time.

    Returns ``{"doc_id", "path", "count", "indices": [<0-based indices>],
    "lines": [<dict>], "skipped": <blocks without a timestamp>, "style",
    "offset_ms"}``.  Snapshot-backed.
    """
    target = Path(str(path)).expanduser()
    if not target.is_file():
        raise ToolError(f"file not found: {target}")
    did, doc = _doc_and_id(doc_id)
    try:
        raw = read_text_with_encoding(str(target))
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"could not read {target}: {exc}") from None
    text = raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    offset = _as_int(offset_ms)
    specs: list[tuple[int, int, str]] = []
    skipped = 0
    for block in re.split(r"\n\s*\n", text):
        rows = [line for line in block.split("\n") if line.strip()]
        if not rows:
            continue
        timestamp_at = next((i for i, line in enumerate(rows) if "-->" in line), None)
        if timestamp_at is None:
            skipped += 1
            continue
        match = _SRT_TS_RE.match(rows[timestamp_at])
        if not match:
            raise ToolError(f"malformed SRT timestamp: {rows[timestamp_at]!r}")
        start = _field_ms(U.parse_time(match.group(1)) + offset, "srt start")
        end = _field_ms(U.parse_time(match.group(2)) + offset, "srt end")
        if end < start:
            end = start
        body = U.escape_ass_text("\n".join(rows[timestamp_at + 1:]), hard_breaks=True)
        specs.append((start, end, body))
    if not specs:
        raise ToolError(f"no usable SubRip blocks found in {target} (skipped {skipped})")
    workspace.snapshot(did)
    created: list[EventEntry] = []
    for start, end, body in specs:
        created.append(_add_one(
            doc,
            {"start_ms": start, "end_ms": end, "text": body, "style": style},
            what="srt block",
        ))
    indices = sorted(_index_of(doc, entry) for entry in created)
    events = doc.events()
    return ok(doc_id=did, path=str(target), count=len(indices), indices=indices,
              skipped=skipped, style=style, offset_ms=offset,
              lines=[_line_dict(events[i], i) for i in indices])


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def ass_stats(doc_id: str | None = None) -> dict[str, Any]:
    """Statistics for a document.

    Returns ``{"doc_id", "path", "lines", "dialogue", "comments", "characters",
    "words", "total_duration_ms", "average_cps", "mean_line_cps", "min_line_cps",
    "max_line_cps", "slowest": {"index", "cps"} | None, "style_histogram",
    "style_names", "actor_histogram", "actors", "over_cps_25", "lines_over_25",
    "empty_lines", "drawing_lines", "karaoke_lines", "duration_ms",
    "span_ms"}``.  ``average_cps`` is total characters over total dialogue
    duration; the per-line figures only consider lines with a positive duration.
    Indices are 0-based.
    """
    did, doc = _doc_and_id(doc_id)
    events = doc.events()
    dialogues = [e for e in events if not e.is_comment]
    comments = [e for e in events if e.is_comment]
    total_duration = sum(max(0, e.duration_ms) for e in dialogues)
    characters = sum(len(T.plain_text(e.text)) for e in dialogues)
    words = sum(len(T.plain_text(e.text).split()) for e in dialogues)
    cps_values: list[tuple[int, float]] = []
    for i, entry in enumerate(events):
        if entry.is_comment or entry.duration_ms <= 0:
            continue
        cps = _safe_cps(entry)
        if cps is not None:
            cps_values.append((i, cps))
    style_histogram: dict[str, int] = {}
    actor_histogram: dict[str, int] = {}
    for entry in events:
        style = entry.get("Style") or ""
        style_histogram[style] = style_histogram.get(style, 0) + 1
        actor = _entry_actor(entry).strip()
        if actor:
            actor_histogram[actor] = actor_histogram.get(actor, 0) + 1
    over = sorted((i, cps) for i, cps in cps_values if cps > 25.0)
    starts = [e.start_ms for e in events]
    ends = [e.end_ms for e in events]
    slowest = max(cps_values, key=lambda item: item[1]) if cps_values else None
    values = [cps for _, cps in cps_values]
    return ok(
        doc_id=did,
        path=str(workspace.path(did)) if workspace.path(did) else None,
        lines=len(events),
        dialogue=len(dialogues),
        comments=len(comments),
        characters=characters,
        words=words,
        total_duration_ms=total_duration,
        duration_ms=total_duration,
        span_ms=(max(ends) - min(starts)) if events else 0,
        average_cps=round(characters / (total_duration / 1000.0), 2) if total_duration > 0 else 0.0,
        mean_line_cps=round(sum(values) / len(values), 2) if values else 0.0,
        min_line_cps=round(min(values), 2) if values else 0.0,
        max_line_cps=round(max(values), 2) if values else 0.0,
        slowest={"index": slowest[0], "cps": slowest[1]} if slowest else None,
        style_histogram=style_histogram,
        style_names=sorted(style_histogram),
        actor_histogram=actor_histogram,
        actors=sorted(actor_histogram),
        over_cps_25=len(over),
        lines_over_25=[i for i, _ in over],
        empty_lines=[i for i, e in enumerate(events) if not T.plain_text(e.text).strip()],
        drawing_lines=[i for i, e in enumerate(events) if _is_drawing(e.text)],
        karaoke_lines=[i for i, e in enumerate(events) if K.parse_karaoke_has_tags(e.text)],
    )


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def _guarded(fn: Any) -> Any:
    """Wrap ``fn`` so no exception other than ``ToolError`` can escape.

    The asscore layers raise plain library errors (``TimeParseError``,
    ``UnicodeDecodeError``, ``OSError`` …) for malformed input; the tool contract
    only allows ``ToolError``, so anything else is re-raised with the real class
    and message preserved in the text.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{fn.__name__}: {type(exc).__name__}: {exc}") from None

    return wrapper


def _wrap_tools() -> None:
    """Apply :func:`_guarded` to every ``ass_*`` tool defined above."""
    for name, obj in list(globals().items()):
        if name.startswith("ass_") and callable(obj) and not isinstance(obj, type):
            globals()[name] = _guarded(obj)


_wrap_tools()


def register(mcp: Any, ws: Any = None) -> list[str]:
    """Register every ``ass_*`` tool of this module with a FastMCP instance.

    ``ws`` is accepted for symmetry with the other tool modules; the tools use
    the ``workspace`` singleton from ``tools.base``.  Returns the sorted list of
    registered tool names.
    """
    names: list[str] = []
    for name, obj in sorted(globals().items()):
        if not name.startswith("ass_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != __name__:
            continue
        mcp.tool()(obj)
        names.append(name)
    return sorted(names)
