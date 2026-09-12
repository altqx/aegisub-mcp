"""Timing, frame mapping, timecodes and QC tools.

Every public function follows the project tool contract documented in
:mod:`aegisub_mcp.tools.base`:

* the name starts with ``ass_`` and the function is registered by
  :func:`register`,
* it returns a JSON-serialisable ``dict``,
* user mistakes raise :class:`~aegisub_mcp.tools.base.ToolError` — a raw
  exception never escapes,
* documents are reached through the module-level ``workspace`` singleton and
  **every mutating tool calls** ``workspace.snapshot(doc_id)`` **before** it
  writes, so ``ass_undo`` always steps back one tool call.

Line indices are 0-based and follow ``doc.events()`` order, comments included
(the same order ``AssDocument.events()`` uses).  A selection of ``None`` — or
``[]``, or ``""`` — means *every line in the document*; that is deliberate and
is what makes a bare ``ass_qc()`` check the whole file even when
``workspace.selection`` is empty.

Times are stored as ASS timestamps with centisecond resolution, so a requested
millisecond value that is not a multiple of 10 ms is rounded on write; tools
report the values that were actually stored, never the values that were
requested.
"""

from __future__ import annotations

import bisect
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..asscore import assutil as U
from ..asscore import render as R
from ..asscore import tags as T
from .base import ToolError, jsonable, ok, resolve_indices, workspace

# --------------------------------------------------------------------- constants

#: A gap shorter than this (ms) between two lines on the same layer counts as a
#: flicker in :func:`ass_qc`.  Aegisub has no user-visible setting for it.
TINY_GAP_MS = 100

#: Every override tag libass knows about, used by :func:`ass_qc`.
KNOWN_TAGS: frozenset[str] = frozenset(
    T.DRAWING_TAGS
    | T.KARAOKE_TAGS
    | T.CLIP_TAGS
    | T.ANIMATION_TAGS
    | T.LAYOUT_TAGS
    | T.COLOR_TAGS
    | T.STYLE_TAGS
    | T.RESET_TAGS
)

#: Two consecutive frame steps closer than this (ms) are treated as the same
#: frame rate when a v2 timecodes file is split into constant-rate runs: v2 files
#: are written with 6 decimals but centisecond-rounded ASS times are far coarser,
#: so a strict comparison would split every real file per frame.
_V2_RATE_TOLERANCE_MS = 1.0

_MS_ONLY_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_V1_HEADER_RE = re.compile(r"^#?\s*timecode\s+format\s+v1\b", re.I)
_V2_HEADER_RE = re.compile(r"^#?\s*timecode\s+format\s+v2\b", re.I)

SILENCE_NOTE = (
    "Changes are only written when dry_run is False; every response states whether "
    "anything was applied."
)


# ------------------------------------------------------------------ small helpers


def _round_half_away(value: float) -> int:
    """Round to the nearest integer, halves away from zero (Aegisub's rule)."""
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def _as_indices(spec: Any, doc) -> list[int]:
    """Normalise a selection spelling; ``None``/``[]``/``""`` mean *all lines*."""
    if spec is None:
        spec = "all"
    elif isinstance(spec, (list, tuple, set)) and not spec:
        spec = "all"
    elif isinstance(spec, str) and not spec.strip():
        spec = "all"
    return resolve_indices(spec, doc)


def _to_ms(value: Any, *, what: str) -> int:
    """Accept milliseconds (int/float) or an ASS time string (``0:00:01.50``)."""
    if isinstance(value, bool):
        raise ToolError(f"{what} must be a time, got {value!r}")
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    text = str(value).strip()
    if not text:
        raise ToolError(f"{what} must be a time, got an empty string")
    if _MS_ONLY_RE.match(text):
        return int(round(float(text)))
    try:
        return U.parse_time(text)
    except U.TimeParseError:
        raise ToolError(
            f"{what}: cannot parse {value!r} as a time "
            "(use 'h:mm:ss.cc' or a number of milliseconds)"
        ) from None


def _write_time(entry, field: str, ms: int | float) -> tuple[int, str]:
    """Write ``ms`` into an event field; return what was actually stored."""
    text = U.format_time(int(round(ms)))
    entry.set(field, text)
    return U.parse_time(text), text


def _event_times(entry, index: int) -> tuple[int, int]:
    """Start/end in ms, turning a malformed stamp into a clear ToolError."""
    try:
        return entry.start_ms, entry.end_ms
    except U.TimeParseError as exc:
        raise ToolError(f"line {index} has an unparseable time: {exc}") from None


def _layer_of(entry) -> tuple[tuple[str, Any], str]:
    """Return ``(group key, display)`` for the line's layer/marked column."""
    raw = entry.get("Layer") or entry.get("Marked") or "0"
    text = str(raw).strip()
    marked = re.match(r"(?i)^marked\s*=\s*(.*)$", text)
    if marked:
        text = marked.group(1).strip()
    try:
        return ("int", int(text)), text
    except ValueError:
        return ("str", text.lower()), text


def _layer_groups(events: Sequence[Any]) -> dict[tuple[str, Any], list[int]]:
    groups: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for i, ev in enumerate(events):
        key, _display = _layer_of(ev)
        groups[key].append(i)
    for key in groups:
        groups[key].sort(key=lambda i: (events[i].start_ms, events[i].end_ms, i))
    return groups


def _doc_optional(doc_id: str | None):
    """The requested document, or ``None`` when none is open (fps fallbacks)."""
    try:
        return workspace.get(doc_id)
    except ToolError:
        return None


def _probe_fps(path: str | None) -> float | None:
    if not path:
        return None
    try:
        info = R.probe_video(path)
    except Exception:  # noqa: BLE001 - probing is best effort
        return None
    value = info.get("fps")
    try:
        fps = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def _resolve_fps(fps: Any, doc=None) -> tuple[float, str]:
    """Frame rate resolution order: argument, workspace video/script, document.

    Returns ``(fps, source)`` and raises a ToolError that explains exactly what
    is missing when none of the sources has one.
    """
    if fps is not None and fps != "":
        try:
            value = float(fps)
        except (TypeError, ValueError):
            raise ToolError(f"fps must be a number, got {fps!r}") from None
        if not (value > 0):
            raise ToolError(f"fps must be positive, got {value!r}")
        return value, "argument"

    video = workspace.video
    if isinstance(video, dict):
        candidate = video.get("fps")
        if candidate:
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value, "workspace.video"
    config = workspace.script_config or {}
    for key in ("fps", "FPS", "framerate", "frame_rate"):
        candidate = config.get(key)
        if candidate:
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value, "workspace.script_config"

    if doc is not None:
        raw = doc.info_get("FPS")
        if raw:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value, "document FPS"

    probed = _probe_fps(_workspace_video_path())
    if probed:
        return probed, "workspace.video(probe)"

    raise ToolError(
        "no frame rate available: pass fps=..., open a video (workspace video fps), "
        "or add an 'FPS' line to [Script Info]"
    )


def _workspace_video_path() -> str | None:
    video = workspace.video
    if isinstance(video, dict):
        candidate = video.get("path") or video.get("file")
        return str(candidate) if candidate else None
    if isinstance(video, str) and video:
        return video
    return None


def _frame_of(ms: float, fps: float, mode: str = "nearest") -> int:
    exact = (float(ms) / 1000.0) * float(fps)
    if mode == "floor":
        return int(math.floor(exact))
    if mode == "ceil":
        return int(math.ceil(exact))
    if mode == "nearest":
        return _round_half_away(exact)
    raise ToolError(f"mode must be 'nearest', 'floor' or 'ceil', got {mode!r}")


def _ms_of_frame(frame: float, fps: float) -> int:
    return _round_half_away((float(frame) / float(fps)) * 1000.0)


def _check_mode(value: str, allowed: Iterable[str], what: str) -> str:
    text = str(value).strip().lower()
    if text not in set(allowed):
        raise ToolError(f"{what} must be one of {sorted(allowed)}, got {value!r}")
    return text


def _fields_for(which: str) -> list[str]:
    text = _check_mode(which, ("start", "end", "both"), "which")
    return ["Start", "End"] if text == "both" else (["Start"] if text == "start" else ["End"])


# --------------------------------------------------------------- character maths


def _plain_visible(text: str) -> str:
    """Plain text with line breaks normalised; override tags removed."""
    plain = T.plain_text(text)
    return plain.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", "\u00a0")


def _char_stats(text: str) -> tuple[int, int, bool]:
    """``(characters, lines, drawing)`` — characters excludes line breaks."""
    if T.is_drawing(text):
        return 0, 1, True
    plain = _plain_visible(text)
    characters = len(plain.replace("\n", ""))
    lines = plain.count("\n") + 1
    return characters, lines, False


def _cps(characters: int, duration_ms: int) -> float | None:
    if duration_ms <= 0:
        return None
    return characters / (duration_ms / 1000.0)


def _tag_names_deep(text: str, depth: int = 0) -> list[str]:
    names: list[str] = []
    for tag in T.parse(text).tags():
        names.append(tag.name)
        if tag.name == "t" and depth < 3:
            try:
                inner = tag.transform_parts() or {}
            except Exception:  # noqa: BLE001 - malformed transform arguments
                inner = {}
            body = inner.get("tags") or ""
            if body:
                names.extend(_tag_names_deep(body, depth + 1))
    return names


def _unclosed_block_index(text: str) -> int | None:
    i = 0
    length = len(text)
    while i < length:
        if text[i] == "{":
            end = text.find("}", i + 1)
            if end < 0:
                return i
            i = end + 1
        else:
            i += 1
    return None


def _line_facts(index: int, entry) -> dict[str, Any]:
    text = entry.text
    times_ok = True
    try:
        start = entry.start_ms
    except U.TimeParseError:
        start = 0
        times_ok = False
    try:
        end = entry.end_ms
    except U.TimeParseError:
        end = 0
        times_ok = False
    characters, lines, drawing = _char_stats(text)
    duration = end - start
    return {
        "index": index,
        "kind": entry.kind,
        "comment": entry.is_comment,
        "layer_key": _layer_of(entry)[0],
        "layer": _layer_of(entry)[1],
        "style": entry.get("Style"),
        "start_ms": start,
        "end_ms": end,
        "duration_ms": duration,
        "times_ok": times_ok,
        "text": text,
        "characters": characters,
        "lines": lines,
        "drawing": drawing,
        "cps": _cps(characters, duration) if times_ok else None,
    }


# --------------------------------------------------------------------- overlaps


def _times_map(
    indices: Sequence[int], events: Sequence[Any]
) -> tuple[dict[int, tuple[int, int]], list[dict[str, Any]]]:
    """``index -> (start_ms, end_ms)``, plus the lines whose times do not parse.

    QC and the overlap checker must survive a malformed file (``recoverable.ass``
    has ``not-a-time`` in a Start column), so unparseable lines are reported
    instead of raising.
    """
    times: dict[int, tuple[int, int]] = {}
    skipped: list[dict[str, Any]] = []
    for i in indices:
        try:
            times[i] = (events[i].start_ms, events[i].end_ms)
        except U.TimeParseError as exc:
            skipped.append({"index": i, "reason": str(exc)})
    return times, skipped


def _overlap_pairs(
    indices: Sequence[int],
    events: Sequence[Any],
    *,
    layer_strict: bool,
    tolerate_ms: int,
    times: dict[int, tuple[int, int]],
    include_comments: bool = False,
) -> list[dict[str, Any]]:
    """Overlapping pairs of *dialogue* lines, ordered by start time.

    Two lines may legitimately overlap when they sit on different layers, so
    with ``layer_strict`` only lines of the same layer are compared.
    """
    candidates = [
        i for i in indices if i in times and (include_comments or not events[i].is_comment)
    ]
    groups: dict[Any, list[int]] = defaultdict(list)
    for i in candidates:
        groups[_layer_of(events[i])[0] if layer_strict else "all"].append(i)

    pairs: list[dict[str, Any]] = []
    for members in groups.values():
        members.sort(key=lambda i: (times[i][0], times[i][1], i))
        for pos, a in enumerate(members):
            a_start, a_end = times[a]
            for b in members[pos + 1:]:
                b_start, b_end = times[b]
                if b_start >= a_end:
                    break
                overlap = min(a_end, b_end) - max(a_start, b_start)
                if overlap > tolerate_ms:
                    pairs.append(
                        {
                            "a": a,
                            "b": b,
                            "layer": _layer_of(events[a])[1],
                            "a_start_ms": a_start,
                            "a_end_ms": a_end,
                            "b_start_ms": b_start,
                            "b_end_ms": b_end,
                            "overlap_ms": overlap,
                        }
                    )
    pairs.sort(key=lambda p: (p["a_start_ms"], p["b_start_ms"], p["a"], p["b"]))
    return pairs


def _gap_pairs(
    indices: Sequence[int],
    events: Sequence[Any],
    times: dict[int, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Tiny gaps between consecutive dialogue lines on the same layer."""
    groups: dict[Any, list[int]] = defaultdict(list)
    for i in indices:
        if i not in times or events[i].is_comment:
            continue
        groups[_layer_of(events[i])[0]].append(i)
    out: list[dict[str, Any]] = []
    for key, members in groups.items():
        members.sort(key=lambda i: (times[i][0], times[i][1], i))
        for prev, nxt in zip(members, members[1:]):
            gap = times[nxt][0] - times[prev][1]
            if 0 <= gap < TINY_GAP_MS:
                out.append({"a": prev, "b": nxt, "layer": _layer_of(events[prev])[1], "gap_ms": gap})
    out.sort(key=lambda p: (times[p["a"]][0], p["a"]))
    return out


# ------------------------------------------------------------------------- shift


def ass_shift_times(
    selection: Any = None,
    offset_ms: Any = 0,
    doc_id: str | None = None,
    clamp: bool = False,
    only_selected: bool = True,
) -> dict[str, Any]:
    """Shift the start/end of lines by ``offset_ms`` (negative shifts allowed).

    Args:
        selection: which lines to shift — ``None``/``"all"``, an index, a list of
            indices, ``"0-4,7"`` or any selector ``base.resolve_indices`` takes.
        offset_ms: milliseconds to add; negative values move lines earlier.
        doc_id: document id; defaults to the current document.
        clamp: when True the shift is limited so that no selected line starts
            before zero.  The *same* (reduced) offset is applied to every line so
            the relative timing of the selection is preserved.
        only_selected: when False every line in the document is shifted and
            ``selection`` is ignored.

    Returns:
        ``{"doc_id", "requested_offset_ms", "applied_offset_ms", "clamped",
        "only_selected", "indices", "count", "applied", "changes": [{"index",
        "start_ms", "end_ms", "new_start_ms", "new_end_ms", "start", "end"}]}``
        where the ``new_*`` values are what the document actually holds after the
        write (ASS stores centiseconds).
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    offset = _to_ms(offset_ms, what="offset_ms")
    indices = _as_indices(selection, doc) if only_selected else list(range(len(events)))
    indices = [i for i in indices if 0 <= i < len(events)]
    if not indices:
        return ok(
            doc_id=workspace.resolve_id(doc_id),
            requested_offset_ms=offset,
            applied_offset_ms=0,
            clamped=False,
            only_selected=only_selected,
            indices=[],
            count=0,
            applied=False,
            changes=[],
            note="no lines selected",
        )

    effective = offset
    clamped = False
    if clamp and offset < 0:
        earliest = min(_event_times(events[i], i)[0] for i in indices)
        if earliest + offset < 0:
            effective = -earliest
            clamped = True

    before = [(i, *_event_times(events[i], i)) for i in indices]
    if effective == 0:
        return ok(
            doc_id=workspace.resolve_id(doc_id),
            requested_offset_ms=offset,
            applied_offset_ms=0,
            clamped=clamped,
            only_selected=only_selected,
            indices=list(indices),
            count=0,
            applied=False,
            changes=[],
            note="offset is zero after clamping — nothing to change",
        )

    workspace.snapshot(doc_id)
    changes = []
    for index, start, end in before:
        entry = events[index]
        new_start, start_text = _write_time(entry, "Start", start + effective)
        new_end, end_text = _write_time(entry, "End", end + effective)
        changes.append(
            {
                "index": index,
                "start_ms": start,
                "end_ms": end,
                "new_start_ms": new_start,
                "new_end_ms": new_end,
                "start": start_text,
                "end": end_text,
            }
        )
    doc.dirty = True
    return ok(
        doc_id=workspace.resolve_id(doc_id),
        requested_offset_ms=offset,
        applied_offset_ms=effective,
        clamped=clamped,
        only_selected=only_selected,
        indices=list(indices),
        count=len(changes),
        applied=True,
        changes=changes,
    )


def ass_scale_times(
    selection: Any = None,
    factor: Any = 1.0,
    doc_id: str | None = None,
    origin: Any = None,
    round_ms: bool = True,
) -> dict[str, Any]:
    """Scale the timing of the selection around an origin.

    ``new = origin + (old - origin) * factor`` is applied to both the start and
    the end of every selected line, so the selection stretches or compresses
    while the origin stays put.

    Args:
        selection: lines to scale (see :func:`ass_shift_times`).
        factor: multiplier, must be > 0.
        doc_id: document id.
        origin: ``None``/``"first"`` (start of the earliest selected line),
            ``"document"`` (0), or an explicit millisecond value / time string.
        round_ms: round the computed times to whole milliseconds before writing.

    Returns:
        ``{"doc_id", "factor", "origin_ms", "origin", "round_ms", "indices",
        "count", "changes": [{"index", "start_ms", "end_ms", "new_start_ms",
        "new_end_ms"}], "span": {"before": {"min_start_ms", "max_end_ms",
        "span_ms"}, "after": {...}}}``
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    try:
        factor_value = float(factor)
    except (TypeError, ValueError):
        raise ToolError(f"factor must be a number, got {factor!r}") from None
    if not (factor_value > 0):
        raise ToolError(f"factor must be greater than 0, got {factor_value!r}")

    indices = _as_indices(selection, doc)
    if not indices:
        raise ToolError("no lines selected to scale")
    before = [(i, *_event_times(events[i], i)) for i in indices]

    if origin is None or (isinstance(origin, str) and origin.strip().lower() in ("first", "selection")):
        origin_ms = min(item[1] for item in before)
        origin_label = "first"
    elif isinstance(origin, str) and origin.strip().lower() in ("document", "doc", "zero"):
        origin_ms = 0
        origin_label = "document"
    else:
        origin_ms = _to_ms(origin, what="origin")
        origin_label = origin_ms

    changes = []
    for index, start, end in before:
        new_start = origin_ms + (start - origin_ms) * factor_value
        new_end = origin_ms + (end - origin_ms) * factor_value
        if round_ms:
            new_start = _round_half_away(new_start)
            new_end = _round_half_away(new_end)
        if new_end < new_start:
            raise ToolError(
                f"scaling line {index} would end before it starts "
                f"({new_start} > {new_end}); check the factor and origin"
            )
        changes.append(
            {
                "index": index,
                "start_ms": start,
                "end_ms": end,
                "new_start_ms": new_start,
                "new_end_ms": new_end,
            }
        )

    span_before = {
        "min_start_ms": min(c["start_ms"] for c in changes),
        "max_end_ms": max(c["end_ms"] for c in changes),
    }
    span_after = {
        "min_start_ms": min(c["new_start_ms"] for c in changes),
        "max_end_ms": max(c["new_end_ms"] for c in changes),
    }
    span_before["span_ms"] = span_before["max_end_ms"] - span_before["min_start_ms"]
    span_after["span_ms"] = span_after["max_end_ms"] - span_after["min_start_ms"]
    moved = any(change["new_start_ms"] != change["start_ms"] or change["new_end_ms"] != change["end_ms"] for change in changes)

    workspace.snapshot(doc_id)
    for change in changes:
        entry = events[change["index"]]
        change["new_start_ms"], start_text = _write_time(entry, "Start", change["new_start_ms"])
        change["new_end_ms"], end_text = _write_time(entry, "End", change["new_end_ms"])
        change["start"] = start_text
        change["end"] = end_text
    doc.dirty = True
    return ok(
        doc_id=workspace.resolve_id(doc_id),
        factor=factor_value,
        origin=origin_label,
        origin_ms=origin_ms,
        round_ms=round_ms,
        indices=list(indices),
        count=len(changes),
        applied=moved,
        changes=changes,
        span={"before": span_before, "after": span_after},
        note=None if moved else "factor and origin left every selected time unchanged",
    )


def ass_set_times(
    index: int,
    doc_id: str | None = None,
    start_ms: Any = None,
    end_ms: Any = None,
    duration_ms: Any = None,
) -> dict[str, Any]:
    """Set the start, end and/or duration of one line.

    Args:
        index: 0-based line index.
        doc_id: document id.
        start_ms: new start — milliseconds or a time string such as
            ``"0:00:01.50"``.
        end_ms: new end (milliseconds or time string).  Mutually exclusive with
            ``duration_ms``.
        duration_ms: new duration; the end becomes ``start + duration_ms``.

    At least one of the three must be given.  An end before the start is
    rejected with a ToolError naming both values.

    Returns:
        ``{"doc_id", "index", "before": {"start_ms", "end_ms", "duration_ms"},
        "after": {...}, "changed": ["Start", ...], "applied"}``
    """
    doc = workspace.get(doc_id)
    if isinstance(index, bool) or not isinstance(index, int):
        raise ToolError(f"index must be an integer, got {index!r}")
    events = doc.events()
    if not (0 <= index < len(events)):
        raise ToolError(f"line index {index} out of range (document has {len(events)} lines)")
    if start_ms is None and end_ms is None and duration_ms is None:
        raise ToolError("pass start_ms, end_ms or duration_ms (nothing to set)")
    if end_ms is not None and duration_ms is not None:
        raise ToolError("pass either end_ms or duration_ms, not both")

    entry = events[index]
    old_start, old_end = _event_times(entry, index)
    new_start = _to_ms(start_ms, what="start_ms") if start_ms is not None else old_start
    if duration_ms is not None:
        duration = _to_ms(duration_ms, what="duration_ms")
        new_end = new_start + duration
    elif end_ms is not None:
        new_end = _to_ms(end_ms, what="end_ms")
    else:
        new_end = old_end

    if new_end < new_start:
        raise ToolError(
            f"line {index}: end {U.format_time(new_end)} ({new_end} ms) is before "
            f"start {U.format_time(new_start)} ({new_start} ms)"
        )

    start_text = U.format_time(new_start)
    end_text = U.format_time(new_end)
    stored_start = U.parse_time(start_text)
    stored_end = U.parse_time(end_text)
    changed = []
    if stored_start != old_start:
        changed.append("Start")
    if stored_end != old_end:
        changed.append("End")
    if not changed:
        # Nothing to write: the requested times (after centisecond rounding) are
        # already stored, so no snapshot is taken and the document is untouched.
        return ok(
            doc_id=workspace.resolve_id(doc_id),
            index=index,
            applied=False,
            changed=[],
            before={"start_ms": old_start, "end_ms": old_end, "duration_ms": old_end - old_start},
            after={
                "start_ms": stored_start,
                "end_ms": stored_end,
                "duration_ms": stored_end - stored_start,
            },
            requested={"start_ms": new_start, "end_ms": new_end},
            rounded_to_centiseconds=stored_start != new_start or stored_end != new_end,
            note="the stored times are already what was requested",
        )

    workspace.snapshot(doc_id)
    entry.set("Start", start_text)
    entry.set("End", end_text)
    doc.dirty = True
    return ok(
        doc_id=workspace.resolve_id(doc_id),
        index=index,
        applied=True,
        changed=changed,
        before={"start_ms": old_start, "end_ms": old_end, "duration_ms": old_end - old_start},
        after={
            "start_ms": stored_start,
            "end_ms": stored_end,
            "duration_ms": stored_end - stored_start,
        },
        requested={"start_ms": new_start, "end_ms": new_end},
        rounded_to_centiseconds=stored_start != new_start or stored_end != new_end,
    )


def ass_set_durations(
    selection: Any = None,
    min_ms: Any = None,
    max_ms: Any = None,
    mode: str = "stretch",
    doc_id: str | None = None,
    dry_run: bool = True,
    allow_overlap: bool = False,
) -> dict[str, Any]:
    """Pull every selected line into the ``[min_ms, max_ms]`` duration window.

    Only the edge named by ``mode`` moves — ``"stretch"`` (default) keeps the
    start and moves the end, ``"start"`` keeps the end and moves the start.  A
    line is never moved past the neighbouring line on the same layer unless
    ``allow_overlap`` is True, and never past ``keep`` gaps (there are none by
    default, so the limit is the neighbour's exact start).

    Args:
        selection: lines to adjust.
        min_ms: minimum duration; shorter lines are lengthened.
        max_ms: maximum duration; longer lines are shortened.
        mode: ``"stretch"``/``"end"`` (move the end) or ``"start"`` (move the
            start).
        doc_id: document id.
        dry_run: when True (the default) nothing is written — the response shows
            exactly what would happen.
        allow_overlap: allow the new edge to cross the neighbouring line.

    Returns:
        ``{"doc_id", "dry_run", "applied", "mode", "window": {"min_ms",
        "max_ms"}, "allow_overlap", "count", "changes": [{"index", "field",
        "old_ms", "new_ms", "old", "new", "reason"}], "blocked": [{"index",
        "reason", "wanted_ms", "limit_ms"}]}``
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    resolved_mode = "end" if str(mode).strip().lower() in ("stretch", "end") else (
        "start" if str(mode).strip().lower() == "start" else None
    )
    if resolved_mode is None:
        raise ToolError(f"mode must be 'stretch' or 'start', got {mode!r}")
    if min_ms is None and max_ms is None:
        raise ToolError("pass min_ms and/or max_ms (nothing to enforce)")
    low = _to_ms(min_ms, what="min_ms") if min_ms is not None else None
    high = _to_ms(max_ms, what="max_ms") if max_ms is not None else None
    if low is not None and high is not None and low > high:
        raise ToolError(f"min_ms ({low}) is greater than max_ms ({high})")

    indices = _as_indices(selection, doc)
    times: dict[int, tuple[int, int]] = {}
    for i, ev in enumerate(events):
        times[i] = _event_times(ev, i)
    groups = _layer_groups(events)

    changes: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for index in indices:
        start, end = times[index]
        duration = end - start
        if low is not None and duration < low:
            target = low
            reason = f"duration {duration} ms is under min_ms {low} ms"
        elif high is not None and duration > high:
            target = high
            reason = f"duration {duration} ms is over max_ms {high} ms"
        else:
            continue
        key = _layer_of(events[index])[0]
        if resolved_mode == "end":
            wanted = start + target
            limit = None
            if not allow_overlap:
                neighbours = [
                    times[other][0]
                    for other in groups[key]
                    if other != index and times[other][0] > start
                ]
                if neighbours:
                    limit = min(neighbours)
                    wanted = min(wanted, limit)
            if wanted <= start or wanted == end:
                blocked.append(
                    {
                        "index": index,
                        "reason": "no room before the next line on this layer",
                        "wanted_ms": start + target,
                        "limit_ms": limit,
                    }
                )
                continue
            field, old_ms, new_ms = "End", end, wanted
        else:
            wanted = end - target
            limit = None
            if not allow_overlap:
                neighbours = [
                    times[other][1]
                    for other in groups[key]
                    if other != index and times[other][0] <= start
                ]
                if neighbours:
                    limit = max(neighbours)
                    wanted = max(wanted, limit)
            if wanted >= end or wanted == start:
                blocked.append(
                    {
                        "index": index,
                        "reason": "no room after the previous line on this layer",
                        "wanted_ms": end - target,
                        "limit_ms": limit,
                    }
                )
                continue
            field, old_ms, new_ms = "Start", start, wanted
        changes.append(
            {
                "index": index,
                "field": field,
                "old_ms": old_ms,
                "new_ms": new_ms,
                "old": U.format_time(old_ms),
                "new": U.format_time(new_ms),
                "reason": reason + (" (clamped to the neighbouring line)" if new_ms != (start + target if field == "End" else end - target) else ""),
            }
        )
        if field == "End":
            times[index] = (start, new_ms)
        else:
            times[index] = (new_ms, end)

    applied = False
    if changes and not dry_run:
        workspace.snapshot(doc_id)
        for change in changes:
            entry = events[change["index"]]
            stored, text = _write_time(entry, change["field"], change["new_ms"])
            change["new_ms"] = stored
            change["new"] = text
        doc.dirty = True
        applied = True

    return ok(
        doc_id=workspace.resolve_id(doc_id),
        dry_run=bool(dry_run),
        applied=applied,
        mode=resolved_mode,
        window={"min_ms": low, "max_ms": high},
        allow_overlap=bool(allow_overlap),
        indices=list(indices),
        count=len(changes),
        changes=changes,
        blocked=blocked,
    )


# ------------------------------------------------------------------- frame maths


def ass_frame_from_ms(ms: Any = 0, fps: Any = None, doc_id: str | None = None) -> dict[str, Any]:
    """Convert a time in milliseconds to a frame number.

    Args:
        ms: milliseconds (int/float) or a time string.
        fps: frame rate; resolution order is ``fps`` argument, workspace video
            (dict ``fps`` or a probed video path), workspace script config,
            document ``FPS`` in [Script Info], then a ToolError explaining what
            is missing.
        doc_id: document id (only used as an fps source).

    Returns:
        ``{"ms", "seconds", "fps", "fps_source", "frame", "frame_exact",
        "frame_floor", "frame_ceil"}``
    """
    value = _to_ms(ms, what="ms")
    rate, source = _resolve_fps(fps, _doc_optional(doc_id))
    exact = (value / 1000.0) * rate
    return ok(
        ms=value,
        seconds=round(value / 1000.0, 6),
        fps=rate,
        fps_source=source,
        frame=_round_half_away(exact),
        frame_exact=round(exact, 6),
        frame_floor=int(math.floor(exact)),
        frame_ceil=int(math.ceil(exact)),
    )


def ass_ms_from_frame(frame: Any = 0, fps: Any = None, doc_id: str | None = None) -> dict[str, Any]:
    """Convert a frame number to its start time in milliseconds.

    See :func:`ass_frame_from_ms` for the fps resolution order.

    Returns:
        ``{"frame", "fps", "fps_source", "ms", "seconds"}``
    """
    try:
        value = float(frame)
    except (TypeError, ValueError):
        raise ToolError(f"frame must be a number, got {frame!r}") from None
    rate, source = _resolve_fps(fps, _doc_optional(doc_id))
    ms = _ms_of_frame(value, rate)
    return ok(
        frame=value,
        fps=rate,
        fps_source=source,
        ms=ms,
        seconds=round(ms / 1000.0, 6),
    )


def ass_snap_to_frames(
    selection: Any = None,
    fps: Any = None,
    mode: str = "nearest",
    doc_id: str | None = None,
    which: str = "both",
) -> dict[str, Any]:
    """Snap line start/end times to whole video frames.

    Args:
        selection: lines to snap.
        fps: frame rate (see :func:`ass_frame_from_ms`).
        mode: ``"nearest"`` (default), ``"floor"`` or ``"ceil"``.
        doc_id: document id.
        which: ``"both"`` (default), ``"start"`` or ``"end"``.

    Returns:
        ``{"doc_id", "fps", "fps_source", "mode", "which", "considered",
        "count", "changes": [{"index", "field", "old_ms", "new_ms", "old",
        "new", "frame"}], "applied"}`` — ``new_ms`` is what the document now
        holds, ``count`` is the number of fields that actually moved and
        ``considered`` the number that were inspected.
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    snap_mode = _check_mode(mode, ("nearest", "floor", "ceil"), "mode")
    fields = _fields_for(which)
    rate, source = _resolve_fps(fps, doc)
    indices = _as_indices(selection, doc)

    plans: list[dict[str, Any]] = []
    for index in indices:
        start, end = _event_times(events[index], index)
        wanted = {"Start": start, "End": end}
        frames = {}
        for field in fields:
            frame = _frame_of(wanted[field], rate, snap_mode)
            frames[field] = frame
            wanted[field] = _ms_of_frame(frame, rate)
        if wanted["End"] < wanted["Start"]:
            wanted["End"] = wanted["Start"]
        for field in fields:
            old = start if field == "Start" else end
            plans.append(
                {
                    "index": index,
                    "field": field,
                    "old_ms": old,
                    "new_ms": wanted[field],
                    "frame": frames[field],
                }
            )

    # only real moves are written: a snap that lands on the same millisecond
    # must not dirty the document or push a useless undo step
    real = [plan for plan in plans if plan["new_ms"] != plan["old_ms"]]
    applied = False
    changes = []
    if real:
        workspace.snapshot(doc_id)
        for plan in real:
            stored, text = _write_time(events[plan["index"]], plan["field"], plan["new_ms"])
            changes.append(
                {
                    "index": plan["index"],
                    "field": plan["field"],
                    "old_ms": plan["old_ms"],
                    "new_ms": stored,
                    "old": U.format_time(plan["old_ms"]),
                    "new": text,
                    "frame": plan["frame"],
                }
            )
        doc.dirty = True
        applied = any(c["old_ms"] != c["new_ms"] for c in changes)

    return ok(
        doc_id=workspace.resolve_id(doc_id),
        fps=rate,
        fps_source=source,
        mode=snap_mode,
        which=str(which).strip().lower(),
        indices=list(indices),
        considered=len(plans),
        count=len(changes),
        applied=applied,
        changes=changes,
    )


def ass_load_keyframes(
    path: Any = None,
    doc_id: str | None = None,
    frames: Any = None,
    times_ms: Any = None,
) -> dict[str, Any]:
    """Load video keyframes into ``workspace.keyframes`` (milliseconds).

    Args:
        path: media file.  When omitted the workspace video is probed with
            ``render.keyframes``.
        doc_id: document id; only used as an fps source when ``frames`` is given.
        frames: explicit frame numbers to convert (needs an fps).
        times_ms: explicit keyframe times in milliseconds — used verbatim.

    Returns:
        ``{"source", "path", "count", "keyframes_ms", "first_ms", "last_ms"}``
    """
    if times_ms is not None:
        values = [_to_ms(v, what="times_ms") for v in times_ms]
        source = "times_ms"
        target = None
    elif frames is not None:
        rate, _src = _resolve_fps(None, _doc_optional(doc_id))
        values = [_ms_of_frame(float(f), rate) for f in frames]
        source = "frames"
        target = None
    else:
        if path:
            target = str(Path(str(path)).expanduser())
        else:
            target = _workspace_video_path()
        if not target:
            raise ToolError(
                "no video to probe: pass path=... or set a workspace video first"
            )
        if not Path(target).is_file():
            raise ToolError(f"video not found: {target}")
        try:
            seconds = R.keyframes(target)
        except R.RenderError as exc:
            raise ToolError(f"could not list keyframes of {target}: {exc}") from None
        values = [_round_half_away(float(s) * 1000.0) for s in seconds]
        source = "render.keyframes"

    values = sorted({int(v) for v in values})
    workspace.keyframes = list(values)
    return ok(
        source=source,
        path=str(target) if target else None,
        count=len(values),
        keyframes_ms=values,
        first_ms=values[0] if values else None,
        last_ms=values[-1] if values else None,
    )


def ass_snap_to_keyframes(
    selection: Any = None,
    which: str = "both",
    mode: str = "nearest",
    forward_only: bool = False,
    max_distance_ms: Any = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Snap line times to keyframes loaded in ``workspace.keyframes``.

    Args:
        selection: lines to snap.
        which: ``"both"`` (default), ``"start"`` or ``"end"``.
        mode: ``"nearest"`` (default), ``"previous"`` or ``"next"``.
        forward_only: only consider keyframes at or after the line's time (never
            move a time earlier).  Contradicts ``mode="previous"``.
        max_distance_ms: skip a snap when it would move the time further than
            this.
        doc_id: document id.

    Returns:
        ``{"doc_id", "mode", "which", "forward_only", "max_distance_ms",
        "keyframes_loaded", "considered", "count", "applied", "changes":
        [{"index", "field", "old_ms", "new_ms", "old", "new", "delta_ms",
        "keyframe_ms"}], "skipped": [{"index", "field", "old_ms", "reason"}]}``
        — ``count``/``changes`` only list fields that really moved.
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    snap_mode = _check_mode(mode, ("nearest", "previous", "next"), "mode")
    fields = _fields_for(which)
    if forward_only and snap_mode == "previous":
        raise ToolError("forward_only cannot be combined with mode='previous'")
    keyframes = sorted({int(k) for k in (workspace.keyframes or [])})
    if not keyframes:
        raise ToolError("no keyframes loaded — call ass_load_keyframes first")
    distance = _to_ms(max_distance_ms, what="max_distance_ms") if max_distance_ms is not None else None
    indices = _as_indices(selection, doc)

    plans: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index in indices:
        start, end = _event_times(events[index], index)
        times = {"Start": start, "End": end}
        new_times = {}
        for field in fields:
            old = times[field]
            pool = [k for k in keyframes if k >= old] if forward_only else keyframes
            if not pool:
                skipped.append({"index": index, "field": field, "old_ms": old,
                                "reason": "no keyframe in the requested direction"})
                new_times[field] = old
                continue
            if snap_mode == "nearest":
                target = min(pool, key=lambda k: (abs(k - old), k))
            elif snap_mode == "next":
                after = [k for k in pool if k >= old]
                target = min(after) if after else None
            else:  # previous
                before = [k for k in pool if k <= old]
                target = max(before) if before else None
            if target is None:
                skipped.append({"index": index, "field": field, "old_ms": old,
                                "reason": "no keyframe in the requested direction"})
                new_times[field] = old
                continue
            if distance is not None and abs(target - old) > distance:
                skipped.append(
                    {"index": index, "field": field, "old_ms": old,
                     "reason": f"nearest keyframe is {abs(target - old)} ms away, "
                               f"over max_distance_ms {distance}"}
                )
                new_times[field] = old
                continue
            new_times[field] = target
        if new_times.get("End", end) < new_times.get("Start", start):
            new_times["End"] = new_times.get("Start", start)
        for field in fields:
            plans.append(
                {
                    "index": index,
                    "field": field,
                    "old_ms": times[field],
                    "new_ms": new_times[field],
                }
            )

    # same rule as ass_snap_to_frames: an exact match is not a change
    real = [plan for plan in plans if plan["new_ms"] != plan["old_ms"]]
    applied = False
    changes: list[dict[str, Any]] = []
    if real:
        workspace.snapshot(doc_id)
        for plan in real:
            stored, text = _write_time(events[plan["index"]], plan["field"], plan["new_ms"])
            changes.append(
                {
                    "index": plan["index"],
                    "field": plan["field"],
                    "old_ms": plan["old_ms"],
                    "new_ms": stored,
                    "old": U.format_time(plan["old_ms"]),
                    "new": text,
                    "delta_ms": stored - plan["old_ms"],
                    "keyframe_ms": plan["new_ms"],
                }
            )
        doc.dirty = True
        applied = any(c["old_ms"] != c["new_ms"] for c in changes)

    return ok(
        doc_id=workspace.resolve_id(doc_id),
        mode=snap_mode,
        which=str(which).strip().lower(),
        forward_only=bool(forward_only),
        max_distance_ms=distance,
        keyframes_loaded=len(keyframes),
        indices=list(indices),
        considered=len(plans),
        count=len(changes),
        applied=applied,
        changes=changes,
        skipped=skipped,
    )


# ------------------------------------------------------------------------ silence


def _silence_path(video: Any) -> str:
    if video:
        target = str(Path(str(video)).expanduser())
        if not Path(target).is_file():
            raise ToolError(f"media file not found: {target}")
        return target
    target = _workspace_video_path() or workspace.audio
    if not target:
        raise ToolError("no media to analyse: pass video=... or set a workspace video/audio")
    if not Path(target).is_file():
        raise ToolError(f"media file not found: {target}")
    return target


def _silence_plan(
    selection: Any,
    video: Any,
    noise_db: float,
    min_silence_s: float,
    doc_id: str | None,
    shift_only: Any,
    max_shift_ms: Any,
) -> dict[str, Any]:
    """Shared engine for the two silence tools: never writes anything."""
    doc = workspace.get(doc_id)
    events = doc.events()
    side = "both" if shift_only is None else _check_mode(shift_only, ("start", "both"), "shift_only")
    limit = _to_ms(max_shift_ms, what="max_shift_ms") if max_shift_ms is not None else None
    target = _silence_path(video)
    try:
        raw = R.silencedetect(target, noise_db=float(noise_db), min_duration_s=float(min_silence_s))
    except R.RenderError as exc:
        raise ToolError(f"silencedetect failed on {target}: {exc}") from None
    silences = []
    open_silences = []
    for item in raw:
        start = item.get("start_ms")
        end = item.get("end_ms")
        if start is None or end is None:
            # ffmpeg omits the end when the file finishes inside a silence
            open_silences.append(int(round(start)) if start is not None else None)
            continue
        silences.append(
            {
                "start_ms": int(round(start)),
                "end_ms": int(round(end)),
                "duration_ms": int(round(item.get("duration_ms") or (end - start))),
            }
        )
    silences.sort(key=lambda s: s["start_ms"])

    suggestions = []
    for index in _as_indices(selection, doc):
        if events[index].is_comment:
            continue
        start, end = _event_times(events[index], index)
        for silence in silences:
            if silence["start_ms"] <= start < silence["end_ms"]:
                delta = silence["end_ms"] - start
                if limit is not None and delta > limit:
                    break
                new_start = silence["end_ms"]
                if side == "start":
                    if new_start >= end:
                        break
                    suggestions.append(
                        {
                            "index": index,
                            "start_ms": start,
                            "end_ms": end,
                            "new_start_ms": new_start,
                            "new_end_ms": end,
                            "delta_ms": delta,
                            "silence": silence,
                            "reason": "line starts inside a silent stretch; end kept",
                        }
                    )
                else:
                    suggestions.append(
                        {
                            "index": index,
                            "start_ms": start,
                            "end_ms": end,
                            "new_start_ms": new_start,
                            "new_end_ms": end + delta,
                            "delta_ms": delta,
                            "silence": silence,
                            "reason": "line starts inside a silent stretch; shifted by delta",
                        }
                    )
                break
    return {
        "path": target,
        "noise_db": float(noise_db),
        "min_silence_s": float(min_silence_s),
        "shift_only": side,
        "max_shift_ms": limit,
        "silences": silences,
        "open_silences": open_silences,
        "suggestions": suggestions,
    }


def ass_align_to_silence(
    selection: Any = None,
    video: Any = None,
    noise_db: float = -45,
    min_silence_s: float = 0.15,
    doc_id: str | None = None,
    shift_only: Any = None,
    max_shift_ms: Any = None,
) -> dict[str, Any]:
    """Report audio silence intervals and the per-line shifts they suggest.

    This tool **never modifies the document** — it exists so an agent can look
    before it leaps; :func:`ass_align_lines_to_silence` applies the same plan
    when ``dry_run=False``.

    Args:
        selection: lines to consider.
        video: media file; defaults to the workspace video/audio.
        noise_db: silence threshold in dBFS (passed to ``silencedetect``).
        min_silence_s: shortest silence to report.
        doc_id: document id.
        shift_only: ``None``/``"both"`` shift start and end together, ``"start"``
            moves only the start (shortening the line).
        max_shift_ms: ignore suggestions longer than this.

    Returns:
        ``{"doc_id", "path", "noise_db", "min_silence_s", "shift_only",
        "max_shift_ms", "silences": [{"start_ms", "end_ms", "duration_ms"}],
        "suggestions": [{"index", "start_ms", "end_ms", "new_start_ms",
        "new_end_ms", "delta_ms", "silence", "reason"}], "count", "applied":
        False, "note"}``
    """
    plan = _silence_plan(selection, video, noise_db, min_silence_s, doc_id, shift_only, max_shift_ms)
    return ok(
        doc_id=workspace.resolve_id(doc_id),
        **plan,
        count=len(plan["suggestions"]),
        applied=False,
        dry_run=True,
        note="report only — the document was not modified",
    )


def ass_align_lines_to_silence(
    selection: Any = None,
    video: Any = None,
    noise_db: float = -45,
    min_silence_s: float = 0.15,
    doc_id: str | None = None,
    dry_run: bool = True,
    shift_only: Any = None,
    max_shift_ms: Any = None,
) -> dict[str, Any]:
    """Move line starts out of audio silence (ffmpeg ``silencedetect``).

    A line whose start falls inside a silent interval is pushed to the end of
    that interval: ``shift_only="both"`` (default) moves the end by the same
    delta, ``"start"`` keeps the end where it is and shortens the line.  Lines
    that start on audio are left alone and are not reported as suggestions.

    Args:
        selection: lines to consider.
        video: media file; defaults to the workspace video/audio.
        noise_db: silence threshold in dBFS.
        min_silence_s: shortest silence to act on.
        doc_id: document id.
        dry_run: when True (the default) the plan is returned and nothing is
            written.
        shift_only: ``None``/``"both"`` or ``"start"``.
        max_shift_ms: ignore suggestions longer than this.

    Returns:
        The same plan as :func:`ass_align_to_silence` plus ``"dry_run"`` and
        ``"applied"`` so the caller always knows whether the document changed.
    """
    plan = _silence_plan(selection, video, noise_db, min_silence_s, doc_id, shift_only, max_shift_ms)
    suggestions = plan["suggestions"]
    applied = False
    if suggestions and not dry_run:
        doc = workspace.get(doc_id)
        events = doc.events()
        workspace.snapshot(doc_id)
        for item in suggestions:
            entry = events[item["index"]]
            stored_start, start_text = _write_time(entry, "Start", item["new_start_ms"])
            stored_end, end_text = _write_time(entry, "End", item["new_end_ms"])
            item["new_start_ms"] = stored_start
            item["new_end_ms"] = stored_end
            item["start"] = start_text
            item["end"] = end_text
        doc.dirty = True
        applied = True
    if applied:
        note = f"applied {len(suggestions)} shift(s) from {len(plan['silences'])} silent interval(s)"
    elif dry_run:
        note = (
            f"dry run: {len(suggestions)} suggestion(s) from {len(plan['silences'])} "
            "silent interval(s) — document unchanged"
        )
    else:
        note = (
            f"nothing applied: 0 of {len(plan['silences'])} silent interval(s) start a line "
            "— document unchanged"
        )
    return ok(
        doc_id=workspace.resolve_id(doc_id),
        **plan,
        count=len(suggestions),
        applied=applied,
        dry_run=bool(dry_run),
        note=note,
    )


# ---------------------------------------------------------------------- timecodes


def _timecodes() -> dict[str, Any] | None:
    return getattr(workspace, "timecodes", None)


def _v1_pieces(default_fps: float, overrides: list[tuple[int, int | None, float]], path: str):
    """Turn v1 overrides into contiguous ``(start, end, fps)`` pieces."""
    pieces: list[tuple[int, int | None, float]] = []
    cursor: int | None = 0
    for start, end, fps, lineno in overrides:
        if cursor is None:
            raise ToolError(f"{path}:{lineno}: override starts after an override that runs to the end")
        if start < cursor:
            raise ToolError(
                f"{path}:{lineno}: v1 override starts at frame {start}, inside the previous "
                f"override (which ends at {cursor}); overlapping overrides are not supported"
            )
        if start > cursor:
            pieces.append((cursor, start, default_fps))
        pieces.append((start, end, fps))
        cursor = None if end is None else end
    if cursor is not None:
        pieces.append((cursor, None, default_fps))
    if not pieces:
        pieces.append((0, None, default_fps))
    return pieces


def _build_segments(pieces: list[tuple[int, int | None, float]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    elapsed = 0.0
    for start, end, fps in pieces:
        start_ms = elapsed
        if end is None:
            end_ms = None
        else:
            end_ms = elapsed + (end - start) * 1000.0 / fps
            elapsed = end_ms
        segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_ms": round(start_ms, 3),
                "end_ms": None if end_ms is None else round(end_ms, 3),
                "fps": round(fps, 6),
            }
        )
    return segments


def _v2_segments(times: list[float], path: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    total = len(times)
    i = 0
    while i < total:
        if i + 1 >= total:
            # trailing frame with no successor: keep the previous rate
            fps = segments[-1]["fps"] if segments else None
            segments.append(
                {
                    "start_frame": i,
                    "end_frame": total,
                    "start_ms": round(times[i], 3),
                    "end_ms": None,
                    "fps": fps,
                }
            )
            break
        delta = times[i + 1] - times[i]
        if delta <= 0:
            raise ToolError(
                f"{path}: frame times must increase, but frame {i + 1} ({times[i + 1]}) "
                f"is not after frame {i} ({times[i]})"
            )
        j = i + 1
        while j + 1 < total and abs((times[j + 1] - times[j]) - delta) < _V2_RATE_TOLERANCE_MS:
            j += 1
        if j + 1 < total:
            # frame j has a different step, so it opens the next segment
            segments.append(
                {
                    "start_frame": i,
                    "end_frame": j,
                    "start_ms": round(times[i], 3),
                    "end_ms": round(times[j], 3),
                    "fps": round(_run_fps(times, i, j), 6),
                }
            )
            i = j
        else:
            segments.append(
                {
                    "start_frame": i,
                    "end_frame": total,
                    "start_ms": round(times[i], 3),
                    "end_ms": None,
                    "fps": round(_run_fps(times, i, total - 1), 6),
                }
            )
            i = total
    return segments


def _run_fps(times: list[float], first: int, last: int) -> float:
    """Average rate of frames ``first..last`` (mean frame duration)."""
    frames = last - first
    if frames <= 0:
        return 0.0
    return frames * 1000.0 / (times[last] - times[first])


def ass_read_timecodes(path: Any) -> dict[str, Any]:
    """Parse an Aegisub timecodes file (v1 or v2) and store it on the workspace.

    Args:
        path: path to the ``.txt``/``.timecodes`` file.

    Returns:
        ``{"path", "version", "default_fps", "fps_changes", "times_ms",
        "frame_count", "duration_ms", "segments": [{"start_frame", "end_frame",
        "start_ms", "end_ms", "fps"}]}``.  v1 files have ``default_fps`` and
        ``fps_changes`` (``[{"start_frame", "end_frame", "fps"}]``); v2 files
        have the per-frame ``times_ms`` list.  The result is also kept on
        ``workspace.timecodes`` so the conversion tools can honour it.

    Malformed input raises a ToolError naming the offending 1-based line number.
    """
    target = Path(str(path)).expanduser()
    if not target.is_file():
        raise ToolError(f"timecodes file not found: {target}")
    try:
        raw = target.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ToolError(f"could not read {target}: {exc}") from None

    lines = raw.splitlines()
    header_index = None
    version = None
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if _V1_HEADER_RE.match(stripped):
                header_index, version = lineno, 1
                break
            if _V2_HEADER_RE.match(stripped):
                header_index, version = lineno, 2
                break
            continue
        raise ToolError(
            f"{target}:{lineno}: expected '# timecode format v1' or "
            f"'# timecode format v2' on the first content line"
        )
    if version is None:
        raise ToolError(
            f"{target}: no '# timecode format v1' or '# timecode format v2' header found"
        )

    body = [
        (lineno, line.strip())
        for lineno, line in enumerate(lines, start=1)
        if lineno > header_index and line.strip() and not line.strip().startswith("#")
    ]

    result: dict[str, Any] = {
        "path": str(target),
        "version": version,
        "default_fps": None,
        "fps_changes": [],
        "times_ms": [],
        "frame_count": None,
        "duration_ms": None,
        "segments": [],
        "line_count": len(lines),
    }

    if version == 1:
        if not body:
            raise ToolError(f"{target}: missing the default frame rate for a v1 file")
        lineno, text = body[0]
        try:
            default_fps = float(text)
        except ValueError:
            raise ToolError(
                f"{target}:{lineno}: expected the default frame rate of a v1 file, got {text!r}"
            ) from None
        if not (default_fps > 0):
            raise ToolError(f"{target}:{lineno}: default frame rate must be positive, got {text!r}")
        overrides: list[tuple[int, int | None, float, int]] = []
        for lineno, text in body[1:]:
            fields = [f.strip() for f in text.split(",")]
            if len(fields) == 2:
                start_raw, fps_raw = fields
                end_raw = None
            elif len(fields) == 3:
                start_raw, end_raw, fps_raw = fields
            else:
                raise ToolError(
                    f"{target}:{lineno}: expected 'start,fps' or 'start,end,fps', got {text!r}"
                )
            try:
                start_frame = int(float(start_raw))
                end_frame = None if end_raw in (None, "", "0") else int(float(end_raw))
                fps = float(fps_raw)
            except ValueError:
                raise ToolError(
                    f"{target}:{lineno}: could not parse the override {text!r} "
                    "(expected numbers: start,end,fps)"
                ) from None
            if start_frame < 0 or (end_frame is not None and end_frame < start_frame):
                raise ToolError(f"{target}:{lineno}: invalid frame range {text!r}")
            if not (fps > 0):
                raise ToolError(f"{target}:{lineno}: frame rate must be positive, got {fps!r}")
            overrides.append((start_frame, end_frame, fps, lineno))
        overrides.sort(key=lambda item: item[0])
        result["default_fps"] = round(default_fps, 6)
        result["fps_changes"] = [
            {"start_frame": s, "end_frame": e, "fps": round(f, 6)} for s, e, f, _ln in overrides
        ]
        result["segments"] = _build_segments(_v1_pieces(default_fps, overrides, str(target)))
        if result["segments"]:
            last = result["segments"][-1]
            result["duration_ms"] = last["end_ms"]
    else:
        times: list[float] = []
        for lineno, text in body:
            try:
                times.append(float(text))
            except ValueError:
                raise ToolError(
                    f"{target}:{lineno}: expected a frame timestamp in milliseconds, got {text!r}"
                ) from None
        if not times:
            raise ToolError(f"{target}: v2 timecodes file contains no frame times")
        result["times_ms"] = [round(t, 3) for t in times]
        result["frame_count"] = len(times)
        result["segments"] = _v2_segments(times, str(target))
        last_fps = result["segments"][-1]["fps"] if result["segments"] else None
        if times:
            step = 1000.0 / last_fps if last_fps else 0.0
            result["duration_ms"] = round(times[-1] + step, 3)
        result["default_fps"] = result["segments"][0]["fps"] if result["segments"] else None
        if len(result["segments"]) > 1:
            # constant-rate runs, usable as v1 overrides if this file is
            # later written back out as v1
            result["fps_changes"] = [
                {
                    "start_frame": segment["start_frame"],
                    "end_frame": segment["end_frame"],
                    "fps": segment["fps"],
                }
                for segment in result["segments"]
                if segment["fps"] is not None
            ]

    workspace.timecodes = result
    workspace.timecodes_path = str(target)
    return ok(**result)


def _timecodes_frame_count(doc, fps: float) -> int:
    loaded = _timecodes()
    if loaded and loaded.get("frame_count"):
        return int(loaded["frame_count"])
    if loaded and loaded.get("times_ms"):
        return len(loaded["times_ms"])
    events = doc.events() if doc else []
    ends: list[int] = []
    for event in events:
        try:
            ends.append(event.end_ms)
        except U.TimeParseError:
            continue
    end = max(ends, default=0)
    if end <= 0:
        raise ToolError(
            "cannot tell how many frames to write: the document has no timed lines "
            "and no timecodes file is loaded"
        )
    return int(math.floor(end * fps / 1000.0)) + 1


def ass_write_timecodes(
    path: Any = None,
    fps: Any = None,
    v2: bool = True,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Write an Aegisub timecodes file (v1 or v2) into ``workspace.output_dir``.

    Args:
        path: destination.  Relative names land inside ``workspace.output_dir``;
            an absolute path is honoured as given.  Defaults to
            ``<script name>.timecodes``.
        fps: frame rate.  Resolution order: this argument, the default rate of a
            loaded timecodes file, workspace video/script info, the document's
            ``FPS``; a ToolError explains what is missing.
        v2: True (default) writes a v2 file (one timestamp per frame), False
            writes v1 (default rate plus overrides).
        doc_id: document id.

    Returns:
        ``{"path", "version", "fps", "fps_source", "frame_count", "lines",
        "bytes", "preview": [first lines], "used_timecodes"}``
    """
    doc = _doc_optional(doc_id)
    loaded = _timecodes()
    fps_from_loaded = False
    if fps is None and loaded and loaded.get("default_fps"):
        rate, source = float(loaded["default_fps"]), "loaded timecodes"
        fps_from_loaded = True
    else:
        rate, source = _resolve_fps(fps, doc)

    if path is None:
        doc_path = workspace.path(doc_id) if doc is not None else None
        stem = doc_path.stem if doc_path else "untitled"
        target = Path(workspace.output_dir) / f"{stem}.timecodes"
    else:
        candidate = Path(str(path)).expanduser()
        target = candidate if candidate.is_absolute() else Path(workspace.output_dir) / candidate

    frame_count = _timecodes_frame_count(doc, rate)
    times_from_loaded = False
    if loaded and (loaded.get("times_ms") or loaded.get("segments")) and v2:
        # honour the loaded timecodes, whatever their version: v2 reads its
        # timestamps back, v1 is replayed segment by segment
        times = [
            round(float(_timecodes_lookup_frame(float(i))[0]), 6) for i in range(frame_count)
        ]
        times_from_loaded = True
    elif loaded and loaded.get("times_ms"):
        times = [float(t) for t in loaded["times_ms"]][:frame_count]
        frame_count = len(times)
    else:
        times = [i * 1000.0 / rate for i in range(frame_count)]

    if v2:
        body = [f"{value:.6f}" for value in times]
    else:
        body = [f"{rate:.6f}"]
        changes = (loaded or {}).get("fps_changes") or []
        for change in changes:
            start = int(change["start_frame"])
            end = change.get("end_frame")
            try:
                change_fps = float(change["fps"])
            except (TypeError, ValueError):
                continue
            if abs(change_fps - rate) < 1e-9 and end is None:
                continue
            body.append(f"{start},{0 if end is None else int(end)},{change_fps:.6f}")

    content = "# timecode format v%s\n%s\n" % ("2" if v2 else "1", "\n".join(body))
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"could not write {target}: {exc}") from None
    preview = content.splitlines()[:6]
    return ok(
        path=str(target),
        version=2 if v2 else 1,
        fps=rate,
        fps_source=source,
        frame_count=frame_count,
        lines=len(body) + 1,
        bytes=len(content.encode("utf-8")),
        preview=preview,
        used_timecodes=bool(times_from_loaded or fps_from_loaded),
        times_source=(
            "loaded timecodes"
            if times_from_loaded
            else f"uniform grid at {rate:g} fps"
        ),
    )


def _timecodes_lookup_ms(value: float) -> tuple[float, str]:
    """Map a time in ms to a frame number using the loaded timecodes."""
    loaded = _timecodes()
    version = loaded.get("version") if loaded else None
    if version == 2 and loaded.get("times_ms"):
        times = loaded["times_ms"]
        if value <= times[0]:
            return 0.0, "timecodes v2"
        if value >= times[-1]:
            last_fps = loaded["segments"][-1]["fps"] if loaded.get("segments") else None
            if last_fps:
                return (
                    (len(times) - 1) + (value - times[-1]) / (1000.0 / last_fps),
                    "timecodes v2 (extrapolated)",
                )
            return float(len(times) - 1), "timecodes v2"
        index = bisect.bisect_right(times, value) - 1
        span = times[index + 1] - times[index]
        return index + ((value - times[index]) / span if span else 0.0), "timecodes v2"
    if version == 1 and loaded.get("segments"):
        segments = loaded["segments"]
        chosen = segments[0]
        for segment in segments:
            if segment["start_ms"] <= value:
                chosen = segment
            if segment.get("end_ms") is not None and value < segment["end_ms"]:
                chosen = segment
                break
        fps = chosen["fps"]
        if not fps:
            return float(chosen["start_frame"]), "timecodes v1"
        return (
            chosen["start_frame"] + (value - chosen["start_ms"]) / (1000.0 / fps),
            "timecodes v1",
        )
    raise ToolError("no timecodes file is loaded — call ass_read_timecodes first")


def _timecodes_lookup_frame(value: float) -> tuple[float, str]:
    """Map a frame number to its start time in ms using the loaded timecodes."""
    loaded = _timecodes()
    version = loaded.get("version") if loaded else None
    if version == 2 and loaded.get("times_ms"):
        times = loaded["times_ms"]
        if value < 0:
            raise ToolError(f"frame must not be negative, got {value}")
        if value < len(times):
            return float(times[int(value)]), "timecodes v2"
        last_fps = loaded["segments"][-1]["fps"] if loaded.get("segments") else None
        if last_fps:
            return times[-1] + (value - (len(times) - 1)) * (1000.0 / last_fps), "timecodes v2 (extrapolated)"
        return float(times[-1]), "timecodes v2"
    if version == 1 and loaded.get("segments"):
        segments = loaded["segments"]
        chosen = segments[-1]
        for segment in segments:
            if segment["start_frame"] <= value and (
                segment.get("end_frame") is None or value < segment["end_frame"]
            ):
                chosen = segment
                break
        fps = chosen["fps"]
        if not fps:
            raise ToolError("the loaded timecodes segment has no frame rate")
        return chosen["start_ms"] + (value - chosen["start_frame"]) * (1000.0 / fps), "timecodes v1"
    raise ToolError("no timecodes file is loaded — call ass_read_timecodes first")


def ass_frame_from_timecodes(frame: Any = 0, doc_id: str | None = None) -> dict[str, Any]:
    """The start time of ``frame`` according to the loaded timecodes file.

    Honours ``workspace.timecodes`` (written by :func:`ass_read_timecodes`) and
    falls back to the document/workspace frame rate when no file is loaded;
    ``method`` says which of the two was used.  Both coordinates are returned so
    the pair is unambiguous.

    Args:
        frame: frame number.
        doc_id: document id, only needed for the fps fallback.

    Returns:
        ``{"frame", "ms", "seconds", "method", "input", "input_kind", "exact_ms"}``
    """
    try:
        value = float(frame)
    except (TypeError, ValueError):
        raise ToolError(f"frame must be a number, got {frame!r}") from None
    if _timecodes():
        exact, method = _timecodes_lookup_frame(value)
    else:
        rate, source = _resolve_fps(None, _doc_optional(doc_id))
        exact, method = (value / rate) * 1000.0, f"fps ({source})"
    return ok(
        input=value,
        input_kind="frame",
        frame=value,
        ms=_round_half_away(exact),
        exact_ms=round(exact, 3),
        seconds=round(exact / 1000.0, 6),
        method=method,
        used_timecodes=bool(_timecodes()),
    )


def ass_ms_from_timecodes(ms: Any = 0, doc_id: str | None = None) -> dict[str, Any]:
    """The frame that contains ``ms`` according to the loaded timecodes file.

    Mirror of :func:`ass_frame_from_timecodes`; ``method`` reports whether the
    loaded timecodes or the fallback frame rate was used, and ``ms`` is the start
    time of the resulting frame.

    Args:
        ms: milliseconds (int/float) or a time string.
        doc_id: document id, only needed for the fps fallback.

    Returns:
        ``{"ms", "frame", "frame_exact", "frame_floor", "frame_ceil",
        "frame_start_ms", "method", "input", "input_kind"}``
    """
    value = _to_ms(ms, what="ms")
    if _timecodes():
        exact, method = _timecodes_lookup_ms(float(value))
        frame_start, _m = _timecodes_lookup_frame(float(_round_half_away(exact)))
    else:
        rate, source = _resolve_fps(None, _doc_optional(doc_id))
        exact = (value / 1000.0) * rate
        method = f"fps ({source})"
        frame_start = _ms_of_frame(_round_half_away(exact), rate)
    return ok(
        input=value,
        input_kind="ms",
        ms=value,
        frame=_round_half_away(exact),
        frame_exact=round(exact, 6),
        frame_floor=int(math.floor(exact)),
        frame_ceil=int(math.ceil(exact)),
        frame_start_ms=frame_start,
        method=method,
        used_timecodes=bool(_timecodes()),
    )


# ----------------------------------------------------------------------------- QC


def ass_qc(
    selection: Any = None,
    doc_id: str | None = None,
    cps_warn: float = 20.0,
    cps_max: float = 25.0,
    min_duration_ms: int = 300,
    max_duration_ms: int = 10000,
    max_chars: int = 42,
    max_lines: int = 2,
    check_overlaps: bool = True,
    check_gaps: bool = True,
    check_empty: bool = True,
    check_styles: bool = True,
    check_tags: bool = True,
) -> dict[str, Any]:
    """QA a selection of lines and return a structured issue report.

    ``cps`` is the number of visible characters — override tags are stripped and
    line breaks are not counted — divided by the line's duration in seconds.
    Drawing lines (``{\\\\p1}``) carry no readable text, so they are skipped by the
    text checks and reported once each as an ``info`` issue with code
    ``drawing_line``.

    Args:
        selection: lines to check; ``None``/``[]`` means every line.
        doc_id: document id.
        cps_warn: cps at or above which a line is ``cps_high`` (warning).
        cps_max: cps at or above which a line is ``cps_extreme`` (error).
        min_duration_ms: shorter lines are ``too_short`` (warning).
        max_duration_ms: longer lines are ``too_long`` (warning).
        max_chars: more visible characters is ``too_many_chars`` (warning).
        max_lines: more \\N-breaks is ``too_many_lines`` (warning).
        check_overlaps: report ``overlap_same_layer`` for lines on the same layer
            (different layers never overlap by design).
        check_gaps: report ``gap_tiny`` for same-layer gaps below 100 ms.
        check_empty: report ``empty_text``.
        check_styles: report ``style_missing`` for styles the document lacks.
        check_tags: report ``unclosed_override_block`` and ``unknown_tag``.

    Returns:
        ``{"doc_id", "parameters": {...}, "summary": {"lines_checked",
        "dialogue", "comments", "drawings", "issues", "errors", "warnings",
        "infos", "by_code": {...}, "ok", "clean"}, "issues": [{"code",
        "severity", "index", "message", "details"}]}`` — ``ok`` is False when any
        error was found, ``clean`` is True only when there are no issues at all.
        Issues are sorted by line index then code.
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    indices = _as_indices(selection, doc)
    styles = {name.strip().lower() for name in doc.style_names()}
    issues: list[dict[str, Any]] = []

    def add(code: str, severity: str, index: int | None, message: str, **details: Any) -> None:
        issues.append(
            {
                "code": code,
                "severity": severity,
                "index": index,
                "message": message,
                "details": jsonable(details),
            }
        )

    dialogues = 0
    comments = 0
    drawings = 0
    line_times: dict[int, tuple[int, int]] = {}
    unparseable: list[int] = []
    for index in indices:
        entry = events[index]
        facts = _line_facts(index, entry)
        if facts["comment"]:
            comments += 1
        else:
            dialogues += 1
        if facts["drawing"]:
            drawings += 1
            add("drawing_line", "info", index, f"line {index} is a drawing", kind=entry.kind)

        if not facts["times_ok"]:
            unparseable.append(index)
            add(
                "unparseable_time",
                "error",
                index,
                f"line {index} has a Start/End that is not a timestamp",
                start=entry.get("Start"),
                end=entry.get("End"),
            )
            continue

        start, end = facts["start_ms"], facts["end_ms"]
        line_times[index] = (start, end)
        duration = facts["duration_ms"]
        if duration < 0:
            add(
                "negative_duration",
                "error",
                index,
                f"line {index} ends before it starts ({duration} ms)",
                start_ms=start,
                end_ms=end,
                duration_ms=duration,
            )
        elif duration == 0:
            add(
                "zero_duration",
                "error",
                index,
                f"line {index} has zero duration",
                start_ms=start,
                end_ms=end,
            )
        else:
            if duration < min_duration_ms:
                add(
                    "too_short",
                    "warning",
                    index,
                    f"line {index} lasts {duration} ms (under {min_duration_ms} ms)",
                    duration_ms=duration,
                    min_duration_ms=min_duration_ms,
                )
            if duration > max_duration_ms:
                add(
                    "too_long",
                    "warning",
                    index,
                    f"line {index} lasts {duration} ms (over {max_duration_ms} ms)",
                    duration_ms=duration,
                    max_duration_ms=max_duration_ms,
                )

        if not facts["drawing"]:
            if facts["characters"] == 0:
                if check_empty:
                    add(
                        "empty_text",
                        "info" if facts["comment"] else "warning",
                        index,
                        f"line {index} has no visible text",
                        kind=entry.kind,
                    )
            else:
                cps = facts["cps"]
                if cps is not None:
                    if cps >= cps_max:
                        add(
                            "cps_extreme",
                            "error",
                            index,
                            f"line {index} reads at {cps:.2f} cps (max {cps_max})",
                            cps=round(cps, 3),
                            cps_max=cps_max,
                            characters=facts["characters"],
                            duration_ms=duration,
                        )
                    elif cps >= cps_warn:
                        add(
                            "cps_high",
                            "warning",
                            index,
                            f"line {index} reads at {cps:.2f} cps (warn {cps_warn})",
                            cps=round(cps, 3),
                            cps_warn=cps_warn,
                            characters=facts["characters"],
                            duration_ms=duration,
                        )
                if facts["characters"] > max_chars:
                    add(
                        "too_many_chars",
                        "warning",
                        index,
                        f"line {index} has {facts['characters']} characters (max {max_chars})",
                        characters=facts["characters"],
                        max_chars=max_chars,
                    )
            if facts["lines"] > max_lines:
                add(
                    "too_many_lines",
                    "warning",
                    index,
                    f"line {index} has {facts['lines']} lines (max {max_lines})",
                    lines=facts["lines"],
                    max_lines=max_lines,
                )

        if check_styles:
            style = entry.get("Style").strip()
            if style and style.lower() not in styles:
                add(
                    "style_missing",
                    "error",
                    index,
                    f"line {index} uses undefined style {style!r}",
                    style=style,
                    styles=sorted(styles),
                )

        if check_tags:
            unclosed = _unclosed_block_index(facts["text"])
            if unclosed is not None:
                add(
                    "unclosed_override_block",
                    "error",
                    index,
                    f"line {index} has an override block that is never closed",
                    raw_index=unclosed,
                )
            unknown = sorted({name for name in _tag_names_deep(facts["text"]) if name not in KNOWN_TAGS})
            if unknown:
                add(
                    "unknown_tag",
                    "warning",
                    index,
                    f"line {index} uses unknown override tag(s): {', '.join('\\\\' + n for n in unknown)}",
                    tags=unknown,
                    known_tags=sorted(KNOWN_TAGS),
                )

    if check_overlaps:
        for pair in _overlap_pairs(
            indices, events, layer_strict=True, tolerate_ms=0, times=line_times
        ):
            add(
                "overlap_same_layer",
                "warning",
                pair["a"],
                f"lines {pair['a']} and {pair['b']} overlap by {pair['overlap_ms']} ms "
                f"on layer {pair['layer']}",
                a=pair["a"],
                b=pair["b"],
                layer=pair["layer"],
                overlap_ms=pair["overlap_ms"],
            )
    if check_gaps:
        for gap in _gap_pairs(indices, events, line_times):
            add(
                "gap_tiny",
                "warning",
                gap["a"],
                f"lines {gap['a']} and {gap['b']} are only {gap['gap_ms']} ms apart "
                f"on layer {gap['layer']}",
                a=gap["a"],
                b=gap["b"],
                layer=gap["layer"],
                gap_ms=gap["gap_ms"],
                threshold_ms=TINY_GAP_MS,
            )

    issues.sort(key=lambda item: (item["index"] if item["index"] is not None else -1, item["code"]))
    by_code: dict[str, int] = defaultdict(int)
    by_severity = {"info": 0, "warning": 0, "error": 0}
    for issue in issues:
        by_code[issue["code"]] += 1
        by_severity[issue["severity"]] = by_severity.get(issue["severity"], 0) + 1

    return ok(
        doc_id=workspace.resolve_id(doc_id),
        parameters={
            "cps_warn": cps_warn,
            "cps_max": cps_max,
            "min_duration_ms": min_duration_ms,
            "max_duration_ms": max_duration_ms,
            "max_chars": max_chars,
            "max_lines": max_lines,
            "check_overlaps": bool(check_overlaps),
            "check_gaps": bool(check_gaps),
            "check_empty": bool(check_empty),
            "check_styles": bool(check_styles),
            "check_tags": bool(check_tags),
            "tiny_gap_ms": TINY_GAP_MS,
        },
        summary={
            "lines_checked": len(indices),
            "dialogue": dialogues,
            "comments": comments,
            "drawings": drawings,
            "issues": len(issues),
            "errors": by_severity["error"],
            "warnings": by_severity["warning"],
            "infos": by_severity["info"],
            "by_code": dict(sorted(by_code.items())),
            "ok": by_severity["error"] == 0,
            "clean": not issues,
        },
        unparseable_lines=unparseable,
        issues=issues,
    )


def ass_check_overlaps(
    selection: Any = None,
    doc_id: str | None = None,
    layer_strict: bool = True,
    tolerate_ms: Any = 0,
) -> dict[str, Any]:
    """List overlapping pairs of lines, ordered by start time.

    With ``layer_strict=True`` (the default) only lines on the same layer are
    compared — two lines on different layers are *meant* to coincide.  Comment
    lines never render, so they are not considered.

    Args:
        selection: lines to compare.
        doc_id: document id.
        layer_strict: compare within each layer only.
        tolerate_ms: ignore overlaps of this length or less.

    Returns:
        ``{"doc_id", "layer_strict", "tolerate_ms", "count", "pairs":
        [{"a", "b", "layer", "a_start_ms", "a_end_ms", "b_start_ms",
        "b_end_ms", "overlap_ms"}]}``
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    tolerance = _to_ms(tolerate_ms, what="tolerate_ms")
    indices = _as_indices(selection, doc)
    times, skipped = _times_map(indices, events)
    pairs = _overlap_pairs(
        indices,
        events,
        layer_strict=bool(layer_strict),
        tolerate_ms=tolerance,
        times=times,
    )
    return ok(
        doc_id=workspace.resolve_id(doc_id),
        layer_strict=bool(layer_strict),
        tolerate_ms=tolerance,
        indices=list(indices),
        count=len(pairs),
        pairs=pairs,
        skipped=skipped,
    )


def ass_fix_timing(
    selection: Any = None,
    doc_id: str | None = None,
    dry_run: bool = True,
    min_duration_ms: Any = 300,
    target_cps: Any = None,
    max_cps: Any = None,
    avoid_overlap: bool = True,
    keep_gaps_ms: Any = 0,
) -> dict[str, Any]:
    """Compute timing fixes and apply them only when ``dry_run`` is False.

    Only end times move.  A line shorter than ``min_duration_ms`` is extended to
    it; a line reading faster than ``max_cps``/``target_cps`` is extended so that
    its characters per second fall back to that figure.  With ``avoid_overlap``
    an extension stops at the next line's start on the same layer, minus
    ``keep_gaps_ms``; when there is no room the fix is reported as blocked
    instead of being applied.

    Args:
        selection: lines to fix.
        doc_id: document id.
        dry_run: True (default) returns the plan and writes nothing.
        min_duration_ms: shortest acceptable duration.
        target_cps: extend faster lines until they read at this rate.
        max_cps: same, used when ``target_cps`` is not given.
        avoid_overlap: stop extensions at the neighbouring line.
        keep_gaps_ms: keep this much room before the neighbouring line.

    Returns:
        ``{"doc_id", "dry_run", "applied", "parameters": {...}, "count",
        "changes": [{"index", "field", "old_ms", "new_ms", "old", "new",
        "reason"}], "blocked": [{"index", "reason", "wanted_ms", "limit_ms"}]}``
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    minimum = _to_ms(min_duration_ms, what="min_duration_ms")
    target = None if target_cps is None else float(target_cps)
    ceiling = None if max_cps is None else float(max_cps)
    if target is not None and not (target > 0):
        raise ToolError(f"target_cps must be positive, got {target_cps!r}")
    if ceiling is not None and not (ceiling > 0):
        raise ToolError(f"max_cps must be positive, got {max_cps!r}")
    gap = _to_ms(keep_gaps_ms, what="keep_gaps_ms")
    if gap < 0:
        raise ToolError(f"keep_gaps_ms must not be negative, got {gap}")

    indices = _as_indices(selection, doc)
    times = {i: _event_times(ev, i) for i, ev in enumerate(events)}
    groups = _layer_groups(events)

    changes: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for index in indices:
        entry = events[index]
        start, end = times[index]
        duration = end - start
        characters = _char_stats(entry.text)[0]
        wanted = end
        reasons: list[str] = []
        if duration < minimum:
            wanted = max(wanted, start + minimum)
            reasons.append(f"duration {duration} ms is under min_duration_ms {minimum} ms")
        for rate, label in ((target, "target_cps"), (ceiling, "max_cps")):
            if rate is None or characters == 0 or duration <= 0:
                continue
            current = characters / (duration / 1000.0)
            if current > rate:
                needed = start + _round_half_away(characters / rate * 1000.0)
                if needed > wanted:
                    wanted = needed
                reasons.append(f"reads at {current:.2f} cps, over {label} {rate}")
        if wanted <= end or not reasons:
            continue

        limit = None
        if avoid_overlap:
            key = _layer_of(entry)[0]
            neighbours = [
                times[other][0] for other in groups[key] if other != index and times[other][0] > start
            ]
            if neighbours:
                limit = min(neighbours) - gap
                if wanted > limit:
                    wanted = limit
        if wanted <= start or wanted == end:
            blocked.append(
                {
                    "index": index,
                    "reason": "; ".join(reasons) + " — no room before the next line on this layer",
                    "wanted_ms": wanted,
                    "limit_ms": limit,
                }
            )
            continue
        changes.append(
            {
                "index": index,
                "field": "End",
                "old_ms": end,
                "new_ms": wanted,
                "old": U.format_time(end),
                "new": U.format_time(wanted),
                "reason": "; ".join(reasons),
            }
        )
        times[index] = (start, wanted)

    applied = False
    if changes and not dry_run:
        workspace.snapshot(doc_id)
        for change in changes:
            stored, text = _write_time(events[change["index"]], "End", change["new_ms"])
            change["new_ms"] = stored
            change["new"] = text
        doc.dirty = True
        applied = True

    return ok(
        doc_id=workspace.resolve_id(doc_id),
        dry_run=bool(dry_run),
        applied=applied,
        parameters={
            "min_duration_ms": minimum,
            "target_cps": target,
            "max_cps": ceiling,
            "avoid_overlap": bool(avoid_overlap),
            "keep_gaps_ms": gap,
        },
        indices=list(indices),
        count=len(changes),
        changes=changes,
        blocked=blocked,
    )


def ass_reading_speed(index: int, doc_id: str | None = None) -> dict[str, Any]:
    """Reading speed of a single line.

    Args:
        index: 0-based line index.
        doc_id: document id.

    Returns:
        ``{"doc_id", "index", "kind", "start_ms", "end_ms", "duration_ms",
        "characters", "lines", "drawing", "cps", "plain_text"}`` — ``cps`` is
        ``None`` when the line has no duration (and 0.0 when it has no text).
    """
    doc = workspace.get(doc_id)
    if isinstance(index, bool) or not isinstance(index, int):
        raise ToolError(f"index must be an integer, got {index!r}")
    events = doc.events()
    if not (0 <= index < len(events)):
        raise ToolError(f"line index {index} out of range (document has {len(events)} lines)")
    entry = events[index]
    start, end = _event_times(entry, index)
    characters, lines, drawing = _char_stats(entry.text)
    cps = _cps(characters, end - start)
    return ok(
        doc_id=workspace.resolve_id(doc_id),
        index=index,
        kind=entry.kind,
        start_ms=start,
        end_ms=end,
        duration_ms=end - start,
        characters=characters,
        lines=lines,
        drawing=drawing,
        cps=None if cps is None else round(cps, 3),
        plain_text=_plain_visible(entry.text),
    )


def ass_cps(selection: Any = None, doc_id: str | None = None) -> dict[str, Any]:
    """Characters per second for every selected line plus the totals.

    Args:
        selection: lines to measure; ``None``/``[]`` means every line.
        doc_id: document id.

    Returns:
        ``{"doc_id", "count", "lines": [{"index", "kind", "characters",
        "duration_ms", "cps", "drawing", "start_ms", "end_ms"}], "totals":
        {"characters", "duration_ms", "average_cps", "max_cps", "worst_index"}}``
    """
    doc = workspace.get(doc_id)
    events = doc.events()
    indices = _as_indices(selection, doc)
    rows = []
    for index in indices:
        entry = events[index]
        start, end = _event_times(entry, index)
        characters, _lines, drawing = _char_stats(entry.text)
        cps = _cps(characters, end - start)
        rows.append(
            {
                "index": index,
                "kind": entry.kind,
                "start_ms": start,
                "end_ms": end,
                "duration_ms": end - start,
                "characters": characters,
                "drawing": drawing,
                "cps": None if cps is None else round(cps, 3),
            }
        )
    total_chars = sum(row["characters"] for row in rows)
    total_ms = sum(max(0, row["duration_ms"]) for row in rows)
    best = max(
        (row for row in rows if row["cps"] is not None),
        key=lambda row: row["cps"],
        default=None,
    )
    return ok(
        doc_id=workspace.resolve_id(doc_id),
        count=len(rows),
        lines=rows,
        totals={
            "characters": total_chars,
            "duration_ms": total_ms,
            "average_cps": round(total_chars / (total_ms / 1000.0), 3) if total_ms > 0 else None,
            "max_cps": best["cps"] if best else None,
            "worst_index": best["index"] if best else None,
        },
    )


# ------------------------------------------------------------------------ factory


def register(mcp, ws=None) -> list[str]:
    """Register every ``ass_*`` function in this module with FastMCP.

    Args:
        mcp: an object exposing ``tool()`` (typed as not to import FastMCP here).
        ws: accepted for API symmetry with the other tool modules; the tools use
            the module-level workspace singleton.

    Returns:
        The sorted list of registered tool names.
    """
    names: list[str] = []
    for name in sorted(globals()):
        obj = globals()[name]
        if name.startswith("ass_") and callable(obj):
            mcp.tool()(obj)
            names.append(name)
    return sorted(names)
