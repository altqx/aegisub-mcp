"""Karaoke tools: syllable splitting, ``\\k`` generation, retiming and templating.

Everything here is a thin, JSON-returning wrapper around
:mod:`aegisub_mcp.asscore.karaoke`.  The unit of ``\\k`` is the **centisecond**;
every function in this module takes and returns milliseconds unless a name says
``_cs``.

Conventions
-----------
* Text arguments accept **either** a raw string (``text=``) or a line index
  (``index=`` / a selection) resolved against an open document.  Every result
  says which was used in ``"text_source"`` (``"text"`` or ``"line"``).
* Line indices are 0-based in ``doc.events()`` order (comments included).
* Every mutation calls ``workspace.snapshot(doc_id)`` first.
* Centisecond maths is exact: durations are distributed so that they sum to the
  line duration in centiseconds; a syllable list whose durations do **not** match
  the line duration is always *reported* (``discrepancy_cs`` / ``matches``) and
  never silently stretched.
* ``not supported`` / ``relaxed`` / ``skipped`` keys exist so a caller can tell
  the difference between "done" and "approximated".
"""

from __future__ import annotations

import csv
import functools
import io
import math
import re
from pathlib import Path

from ..asscore import assutil as U
from ..asscore import karaoke as K
from ..asscore import tags as T
from ..asscore.document import EventEntry
from .base import ToolError, entry_at, ok, resolve_indices, workspace

__all__ = [
    "ass_karaoke_split",
    "ass_karaoke_get",
    "ass_karaoke_generate",
    "ass_karaoke_set_timings",
    "ass_karaoke_retime",
    "ass_karaoke_shift",
    "ass_karaoke_scale",
    "ass_karaoke_set_kind",
    "ass_karaoke_remove",
    "ass_karaoke_auto_timings",
    "ass_karaoke_styles",
    "ass_karaoke_template",
    "ass_karaoke_export",
    "register",
]

_CS_PER_MS = 10.0
_K_NAMES = sorted(K.KARAOKE_TAGS_ALL)
_SPLIT_MODES = ("marker", "char", "word", "regex")
_WEIGHT_MODES = ("char", "char_class", "even")
_LINK_MODES = ("none", "syl", "char")

#: ``mode`` spellings accepted by :func:`ass_karaoke_auto_timings`.
_AUTO_MODES = {"syllable": "marker", "syl": "marker", "marker": "marker",
               "char": "char", "character": "char", "word": "word",
               "regex": "regex"}

#: Conventional karaoke template styles (see :func:`ass_karaoke_styles`).
_KARAOKE_STYLE_BASE = {
    "Fontname": "Arial",
    "Fontsize": 60,
    "PrimaryColour": "&H00FFFFFF&",    # sung / active highlight
    "SecondaryColour": "&H000000FF&",  # unsung / before the sweep
    "OutlineColour": "&H00000000&",
    "BackColour": "&H80000000&",
    "Bold": -1,
    "Italic": 0,
    "Underline": 0,
    "StrikeOut": 0,
    "ScaleX": 100,
    "ScaleY": 100,
    "Spacing": 0,
    "Angle": 0,
    "BorderStyle": 1,
    "Outline": 2,
    "Shadow": 1,
    "MarginL": 10,
    "MarginR": 10,
    "MarginV": 10,
    "Encoding": 1,
}
_KARAOKE_STYLE_VARIANTS = (
    ("", {"Alignment": 2}),              # bottom — the usual karaoke line
    ("_2", {"Alignment": 8}),            # top
    ("_3", {"Alignment": 5}),            # middle (secondary effect layer)
)
_TEMPLATE_MARKER = "kara-templater-subset"


# --------------------------------------------------------------------------- guard


def _guard(fn):
    """Turn any unexpected exception into a :class:`ToolError`."""

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{fn.__name__} failed: {type(exc).__name__}: {exc}") from exc

    return inner


# --------------------------------------------------------------------------- helpers


def _ms_to_cs(ms: float) -> int:
    """Milliseconds -> centiseconds (same rounding as ``Syllable.duration_cs``)."""
    return int(round(float(ms) / _CS_PER_MS))


def _ms_to_time(ms: float) -> str:
    """Milliseconds -> ``h:mm:ss.cc`` (ASS times are centisecond-resolution)."""
    return U.format_time(int(round(ms)))


def _time_to_ms(value) -> int:
    return U.parse_time(value)


def _split_mode(mode: str, *, where: str) -> str:
    name = (mode or "").strip().lower()
    if name not in _SPLIT_MODES:
        raise ToolError(f"{where}: mode must be one of {', '.join(_SPLIT_MODES)}, got {mode!r}")
    return name


def _split_pieces(text: str, mode: str, marker: str, pattern: str | None) -> list[str]:
    if mode == "regex" and not pattern:
        raise ToolError("mode='regex' needs pattern=")
    try:
        return K.split_syllables(text, mode=mode, marker=marker, pattern=pattern,
                                 keep_marker=False)
    except ValueError as exc:  # asscore raises ValueError for a missing pattern
        raise ToolError(str(exc)) from None


def _piece_parts(piece: str) -> tuple[str, str]:
    """``(leading override blocks, rest)`` of one syllable piece."""
    parsed = T.parse(piece)
    prefix = ""
    idx = 0
    for seg in parsed.segments:
        if not isinstance(seg, T.TagBlock):
            break
        prefix += seg.render()
        idx += 1
    return prefix, piece[len(prefix):]


def _piece_info(piece: str, index: int) -> dict:
    prefix, rest = _piece_parts(piece)
    visible = T.plain_text(piece)
    stripped = visible.lstrip()
    lead_ws = visible[: len(visible) - len(stripped)] if stripped else visible
    return {
        "index": index,
        "raw": piece,
        "text": visible,
        "prefix": prefix,
        "prefix_tags": [t.name for t in T.parse_tags(prefix)],
        "leading_whitespace": lead_ws,
        "trailing_whitespace": visible[len(visible.rstrip()):] or "",
        "char_count": len(visible),
        "has_tags": bool(prefix) or bool(rest) and rest != visible,
    }


def _expected_visible(text: str, mode: str, marker: str, pattern: str | None) -> str:
    """Visible text that the syllables should concatenate to."""
    plain = T.plain_text(text)
    if mode == "marker" and marker:
        return plain.replace(marker, "")
    if mode == "regex" and pattern:
        try:
            return re.sub(pattern, "", plain)
        except re.error as exc:
            raise ToolError(f"invalid pattern {pattern!r}: {exc}") from None
    return plain


def _resolve_text(text, index, doc_id, *, where: str) -> dict:
    """Accept a raw string or a line index; say which was used."""
    if isinstance(text, bool):
        raise ToolError(f"{where}: text must be a string or an index, got {text!r}")
    if isinstance(text, int):
        if index is not None:
            raise ToolError(f"{where}: pass text= or index=, not both")
        index, text = text, None
    if text is not None:
        if not isinstance(text, str):
            raise ToolError(f"{where}: text must be a string, got {type(text).__name__}")
        return {"text": text, "text_source": "text", "doc": None, "doc_id": None,
                "entry": None, "index": None}
    if index is None:
        raise ToolError(f"{where}: pass either text=<string> or index=<line index> (with doc_id)")
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    entry = entry_at(doc, int(index))
    return {"text": entry.text, "text_source": "line", "doc": doc, "doc_id": did,
            "entry": entry, "index": int(index)}


def _targets(selection, index, doc, *, where: str, default_all: bool = False) -> list[int]:
    """Resolve a selection/index pair into 0-based line indices."""
    if index is not None and selection is None:
        return [int(index)]
    if selection is None and index is None:
        if not workspace.selection:
            raise ToolError(
                f"{where}: no lines selected — pass index=, selection=, or select lines "
                "in the session first"
            )
        selection = None
        default_all = False
    try:
        return resolve_indices(selection, doc, selection=workspace.selection,
                               default_all=default_all)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"{where}: bad selection {selection!r}: {exc}") from exc


def _line_span(entry: EventEntry) -> tuple[int, int]:
    return entry.start_ms, entry.end_ms


def _allocate(weights: list[float], span_ms: int, *, min_cs: int = 1) -> tuple[list[int], int]:
    """Distribute ``span_ms`` over ``weights`` so the centisecond sum is exact.

    Returns ``(durations_cs, total_cs)``.  Uses
    :func:`aegisub_mcp.asscore.karaoke.distribute_by_length` for the base
    distribution and then rebalances so that ``sum(durations_cs) == total_cs``
    exactly (the asscore helper can drift by more than one centisecond when the
    per-syllable rounding all rounds up).
    """
    total_cs = _ms_to_cs(span_ms)
    n = len(weights)
    if n == 0:
        return [], total_cs
    floor_cs = max(0, int(min_cs)) * n
    if floor_cs > total_cs:
        raise ToolError(
            f"span of {total_cs} cs cannot hold {n} syllable(s) at min_cs={min_cs} "
            f"(needs at least {floor_cs} cs)"
        )
    base = K.distribute_by_length(["" for _ in weights], int(span_ms), weights=list(weights))
    if len(base) < n:
        base = list(base) + [0] * (n - len(base))
    cs = [max(int(min_cs), int(c)) for c in base[:n]]
    diff = total_cs - sum(cs)
    guard = 0
    while diff > 0:
        for i in sorted(range(n), key=lambda k: (-cs[k], k)):
            if diff <= 0:
                break
            cs[i] += 1
            diff -= 1
        guard += 1
        if guard > total_cs + 2:  # pragma: no cover - defensive
            break
    while diff < 0:
        i = max(range(n), key=lambda k: (cs[k], -k))
        if cs[i] <= min_cs:
            raise ToolError(
                f"cannot distribute {total_cs} cs over {n} syllable(s) with min_cs={min_cs}"
            )
        cs[i] -= 1
        diff += 1
    return cs, total_cs


def _weights_for(texts: list[str], weights: str) -> list[float]:
    name = (weights or "").strip().lower()
    if name not in _WEIGHT_MODES:
        raise ToolError(f"weights must be one of {', '.join(_WEIGHT_MODES)}, got {weights!r}")
    if name == "even":
        return [1.0] * len(texts)
    if name == "char_class":
        return list(K.syllable_weights_by_char_class(texts))
    return [max(1.0, float(len(T.plain_text(t)) or 1)) for t in texts]


def _k_tag(kind: str, cs: int) -> str:
    return "\\" + kind + str(max(0, int(cs)))


def _check_kind(kind: str) -> str:
    value = (kind or "").strip()
    canon = K.KARAOKE_ALIASES.get(value, value.lower())
    if canon not in K.KARAOKE_TAG_KINDS:
        raise ToolError(
            f"kind must be one of {', '.join(K.KARAOKE_TAG_KINDS)} (or 'K'), got {kind!r}"
        )
    return canon


def _units(text: str) -> list[dict]:
    """Tag-preserving karaoke units of ``text``.

    One unit per ``\\k``-family tag: ``{"prefix", "text", "kind", "cs"}`` where
    ``prefix`` holds every override block (minus the karaoke tags) that precedes
    the unit and ``text`` is its visible text.  A line without karaoke tags
    yields a single untimed unit.
    """
    parsed = T.parse(text)
    units: list[dict] = []
    pending = ""
    current: dict | None = None
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            cleaned = T.remove_tags(seg.render(), _K_NAMES)
            if cleaned:
                pending += cleaned
            for tag in seg.tags:
                if tag.name in K.KARAOKE_TAGS_ALL:
                    if current is not None:
                        units.append(current)
                    kind = K.KARAOKE_ALIASES.get(tag.name, tag.name)
                    current = {"prefix": pending, "text": "", "kind": kind,
                               "cs": max(0, tag.int_arg(0))}
                    pending = ""
            continue
        if current is None:
            current = {"prefix": pending, "text": "", "kind": "k", "cs": 0}
            pending = ""
        current["text"] += seg.text
    if current is not None:
        units.append(current)
    return units


def _unit_pieces(text: str, mode: str, marker: str, pattern: str | None) -> list[dict]:
    """Units for a rewrite: existing karaoke when present, else a fresh split."""
    if K.parse_karaoke_has_tags(text):
        return [u for u in _units(text) if u["text"]]
    pieces = _split_pieces(text, mode, marker, pattern)
    return [{"prefix": prefix, "text": plain, "kind": "k", "cs": 0}
            for prefix, piece in ((_piece_parts(p)[0], p) for p in pieces)
            for plain in [T.plain_text(piece)]]


def _build_karaoke(units: list[dict], cs: list[int], kinds: list[str], *,
                   link: str = "none") -> str:
    """Rebuild override text from units + durations (``link`` placement rules)."""
    out: list[str] = []
    for i, unit in enumerate(units):
        tag = _k_tag(kinds[i], cs[i])
        prefix, text = unit["prefix"], unit["text"]
        if link == "syl" and (prefix or i == 0):
            out.append(prefix + "{" + tag + "}" + text)
        elif i == 0:
            out.append(T.prepend_tags(prefix + text, tag))
        else:
            out.append("{" + tag + "}" + prefix + text)
    return "".join(out)


def _char_units(units: list[dict]) -> list[dict]:
    """Split units into per-character units, keeping combining marks attached."""
    out: list[dict] = []
    for unit in units:
        chars = K.split_syllables(unit["text"], mode="char") if unit["text"] else [""]
        for j, ch in enumerate(chars):
            out.append({"prefix": unit["prefix"] if j == 0 else "",
                        "text": ch, "kind": unit["kind"], "cs": 0})
    return out


def _summary(text: str, line_start_ms: int, line_end_ms: int) -> dict:
    """Aggregate of a line's karaoke: durations, exactness, kind mix."""
    units = _units(text)
    has_tags = K.parse_karaoke_has_tags(text)
    sum_cs = sum(u["cs"] for u in units) if has_tags else 0
    line_cs = _ms_to_cs(max(0, line_end_ms - line_start_ms))
    kinds: dict[str, int] = {}
    for unit in units:
        if unit["cs"] or has_tags:
            kinds[unit["kind"]] = kinds.get(unit["kind"], 0) + 1
    return {
        "has_karaoke_tags": has_tags,
        "syllable_count": len(units) if has_tags else 0,
        "duration_sum_cs": sum_cs,
        "line_duration_cs": line_cs,
        "discrepancy_cs": sum_cs - line_cs,
        "matches_line_duration": sum_cs == line_cs if has_tags else None,
        "kind_counts": kinds,
        "kinds": sorted(kinds),
    }


def _set_text(entry: EventEntry, doc, text: str) -> None:
    entry.set("Text", text)
    doc.dirty = True


def _set_times(entry: EventEntry, doc, start_ms=None, end_ms=None) -> None:
    if start_ms is not None:
        entry.set("Start", _ms_to_time(start_ms))
    if end_ms is not None:
        entry.set("End", _ms_to_time(end_ms))
    doc.dirty = True


def _discrepancy_message(sum_cs: int, line_cs: int) -> str:
    if sum_cs == line_cs:
        return f"syllable durations sum to {sum_cs} cs, exactly the line duration"
    return (f"syllable durations sum to {sum_cs} cs but the line lasts {line_cs} cs "
            f"(discrepancy {sum_cs - line_cs:+d} cs) — reported, not stretched")


# --------------------------------------------------------------------------- split


@_guard
def ass_karaoke_split(text=None, index=None, doc_id=None, mode: str = "marker",
                      marker: str = "|", pattern: str | None = None,
                      keep_bom: bool = False) -> dict:
    """Split a karaoke line into syllables.

    Arguments
    ---------
    ``text`` / ``index`` + ``doc_id``
        Either the raw override text, or a 0-based line index in an open
        document.  ``"text_source"`` in the result says which was used.
    ``mode``   ``"marker"`` (split on ``marker``, the usual ``|`` workflow),
               ``"char"`` (one visible character per syllable, combining marks
               stay attached to their base character), ``"word"`` (whitespace)
               or ``"regex"`` (split on ``pattern``).
    ``marker`` the marker string for ``mode="marker"``.  Default ``"|"``.
    ``pattern`` regex for ``mode="regex"`` (a capture-free split pattern).
    ``keep_bom`` keep a leading U+FEFF in the text (default: strip it).

    Returns
    -------
    ``{"text_source": "text"|"line", "index", "doc_id", "mode", "marker",
       "pattern", "had_bom", "plain_text", "expected_visible_text",
       "reconstruction": {"ok", "reconstructed", "expected"},
       "syllable_count", "syllables": [{"index", "raw", "text", "prefix",
       "prefix_tags", "leading_whitespace", "trailing_whitespace",
       "char_count"}]}``
    """
    where = "ass_karaoke_split"
    mode = _split_mode(mode, where=where)
    src = _resolve_text(text, index, doc_id, where=where)
    body = src["text"]
    had_bom = body.startswith("\ufeff")
    if had_bom and not keep_bom:
        body = body[1:]
    pieces = _split_pieces(body, mode, marker, pattern)
    syllables = [_piece_info(piece, i) for i, piece in enumerate(pieces)]
    reconstructed = "".join(s["text"] for s in syllables)
    expected = _expected_visible(body, mode, marker, pattern)
    return ok(
        ok_=True,
        text_source=src["text_source"],
        index=src["index"],
        doc_id=src["doc_id"],
        mode=mode,
        marker=marker,
        pattern=pattern,
        moved_marker=(mode == "marker"),
        had_bom=had_bom,
        plain_text=T.plain_text(body),
        expected_visible_text=expected,
        reconstruction={
            "ok": reconstructed == expected,
            "reconstructed": reconstructed,
            "expected": expected,
        },
        syllable_count=len(syllables),
        syllables=syllables,
    )


# --------------------------------------------------------------------------- get


@_guard
def ass_karaoke_get(index: int, doc_id: str | None = None) -> dict:
    """Read the parsed karaoke of one line.

    Arguments
    ---------
    ``index``  0-based line index (``doc.events()`` order) of an open document.
    ``doc_id`` open document id (defaults to the current one).

    Returns
    -------
    ``{"text_source": "line", "index", "doc_id", "line": {...},
       "has_karaoke_tags", "untimed", "timings_source", "syllables":
       [{"index", "text", "raw", "prefix", "kind", "duration_cs", "duration_ms",
       "start_ms", "end_ms", "start", "end"}],
       "duration_sum_cs", "duration_sum_ms", "line_duration_cs",
       "discrepancy_cs", "matches_line_duration", "message", "kind_counts",
       "kinds"}``

    ``start_ms``/``end_ms`` are absolute (the line's ``Start`` is time zero
    internally, the line's own start is added).  ``discrepancy_cs`` is
    ``sum(syllable durations) - line duration`` in centiseconds and is reported
    explicitly, never corrected here.
    """
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    entry = entry_at(doc, int(index))
    text = entry.text
    start_ms, end_ms = _line_span(entry)
    has_tags = K.parse_karaoke_has_tags(text)
    rows = K.karaoke_timings(text, line_start_ms=start_ms, line_end_ms=end_ms)
    parsed = K.parse_karaoke(text)
    syllables = []
    for i, row in enumerate(rows):
        syl = parsed[i] if i < len(parsed) else None
        syllables.append({
            "index": i,
            "text": row.get("text", ""),
            "raw": row.get("raw", ""),
            "prefix": syl.prefix if syl is not None else "",
            "kind": row.get("kind", "k"),
            "duration_cs": int(row.get("duration_cs", 0)),
            "duration_ms": int(row.get("duration_ms", 0)),
            "start_ms": int(row.get("start_ms", start_ms)),
            "end_ms": int(row.get("end_ms", start_ms)),
            "start": _ms_to_time(row.get("start_ms", start_ms)),
            "end": _ms_to_time(row.get("end_ms", start_ms)),
        })
    if has_tags:
        sum_cs = sum(s["duration_cs"] for s in syllables)
        source = "karaoke_tags"
    else:
        sum_cs = sum(s["duration_cs"] for s in syllables)
        source = "line_span"
    line_cs = _ms_to_cs(max(0, end_ms - start_ms))
    kinds: dict[str, int] = {}
    for syl in syllables:
        kinds[syl["kind"]] = kinds.get(syl["kind"], 0) + 1
    return ok(
        ok_=True,
        text_source="line",
        index=int(index),
        doc_id=did,
        line={"start_ms": start_ms, "end_ms": end_ms, "start": entry.get("Start"),
              "end": entry.get("End"), "duration_ms": entry.duration_ms,
              "style": entry.get("Style"), "actor": entry.get("Name"),
              "plain_text": T.plain_text(text)},
        has_karaoke_tags=has_tags,
        untimed=not has_tags,
        timings_source=source,
        syllables=syllables,
        syllable_count=len(syllables),
        duration_sum_cs=sum_cs,
        duration_sum_ms=sum_cs * 10,
        line_duration_cs=line_cs,
        line_duration_ms=max(0, end_ms - start_ms),
        discrepancy_cs=sum_cs - line_cs,
        matches_line_duration=sum_cs == line_cs,
        message=_discrepancy_message(sum_cs, line_cs),
        kind_counts=kinds,
        kinds=sorted(kinds),
    )


# --------------------------------------------------------------------------- generate


@_guard
def ass_karaoke_generate(selection=None, index=None, doc_id=None, start_ms: int | None = None,
                         end_ms: int | None = None, mode: str = "marker", kind: str = "k",
                         link: str = "none", weights: str = "char",
                         replace_existing: bool = True, min_cs: int = 1,
                         snap_to_line: bool = True, marker: str = "|",
                         pattern: str | None = None) -> dict:
    """Generate ``\\k`` tags across a line's syllables.

    Arguments
    ---------
    ``selection`` / ``index``
        Which lines to rewrite.  ``index`` names one line; otherwise
        ``selection`` goes through :func:`base.resolve_indices` (``None`` uses
        the session selection, and an error is raised when nothing is selected).
    ``start_ms`` / ``end_ms``
        Span the durations are distributed over.  When omitted the line's own
        ``Start``/``End`` are used.  The line's own times are **never modified**
        by this tool (use :func:`ass_karaoke_retime` for that).
    ``mode``    ``marker`` / ``char`` / ``word`` / ``regex`` (``pattern`` for regex).
    ``kind``    karaoke tag kind: ``k``, ``kf``, ``ko`` or ``kt``.
    ``link``    how tags attach to the syllables:
                ``"none"`` one tag per syllable, placed before the syllable's own
                override tags;
                ``"syl"`` the same count but each tag is placed *after* the
                syllable's leading override tags;
                ``"char"`` one tag per visible character (combining marks stay
                with their base character) — the line span is split per character.
    ``weights`` ``"char"`` (visible length), ``"char_class"`` (CJK/Latin/space
                weighting, see ``karaoke.syllable_weights_by_char_class``) or
                ``"even"``.
    ``replace_existing``
        ``True`` (default) rewrites existing karaoke tags.  ``False`` leaves any
        line that already carries karaoke tags untouched and reports it in
        ``skipped``.
    ``min_cs``  minimum duration per syllable; a span too short to honour it
                raises :class:`ToolError` instead of silently producing zeros.
    ``snap_to_line``
        Clamp an explicitly requested span to the line's own span when it would
        overshoot (reported in ``snapped``).

    Returns
    -------
    ``{"text_source": "line", "doc_id", "mode", "kind", "link", "weights",
       "min_cs", "exact_sum_guaranteed": True, "sum_within_one_cs": True,
       "lines": [{"index", "start_ms", "end_ms", "span_ms", "span_cs",
       "snapped", "skipped", "reason", "durations_cs", "duration_sum_cs",
       "line_duration_cs", "discrepancy_cs", "matches_line_duration",
       "message", "syllables", "text", "old_text"}], "lines_changed": n}``

    The centisecond durations always sum to exactly ``span_cs`` (and therefore
    to the line duration when the span came from the line) — better than the
    one-centisecond tolerance required, so ``exact_sum_guaranteed`` is always
    ``True`` when a line is written.
    """
    where = "ass_karaoke_generate"
    mode = _split_mode(mode, where=where)
    kind = _check_kind(kind)
    if link not in _LINK_MODES:
        raise ToolError(f"link must be one of {', '.join(_LINK_MODES)}, got {link!r}")
    if mode == "regex" and not pattern:
        raise ToolError("mode='regex' needs pattern=")
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = _targets(selection, index, doc, where=where)
    if not indices:
        raise ToolError(f"{where}: selection matched no lines")
    results: list[dict] = []
    changed = 0
    snapshotted = False
    for line_index in indices:
        entry = entry_at(doc, line_index)
        text = entry.text
        line_start, line_end = _line_span(entry)
        if replace_existing is False and K.parse_karaoke_has_tags(text):
            results.append({"index": line_index, "skipped": True, "old_text": text,
                            "reason": "line already has karaoke tags and replace_existing=False",
                            "start_ms": line_start, "end_ms": line_end})
            continue
        span_start = line_start if start_ms is None else int(start_ms)
        span_end = line_end if end_ms is None else int(end_ms)
        snapped = False
        if snap_to_line and (start_ms is not None or end_ms is not None):
            if span_start < line_start or span_end > line_end:
                # clamp the requested span into the line's own span
                span_start = max(line_start, min(span_start, line_end))
                span_end = min(line_end, max(span_end, line_start))
                snapped = True
        span_ms = max(0, span_end - span_start)
        # syllables come from the marker/char/word split of the (karaoke-stripped)
        # source, exactly like karaoke.generate_k does.
        base = K.remove_karaoke(text, drop_markers=(mode != "marker"))
        pieces = [_strip_k(p) for p in _split_pieces(base, mode, marker, pattern)]
        if not pieces:
            results.append({"index": line_index, "skipped": True, "old_text": text,
                            "reason": "no syllables found", "start_ms": span_start,
                            "end_ms": span_end})
            continue
        units = [{"prefix": _piece_parts(p)[0], "text": T.plain_text(p), "kind": kind, "cs": 0}
                 for p in pieces]
        if link == "char":
            units = _char_units(units)
            if not units:  # pragma: no cover - defensive
                units = [{"prefix": "", "text": "", "kind": kind, "cs": 0}]
        w = _weights_for([u["text"] for u in units], weights)
        cs, span_cs = _allocate(w, span_ms, min_cs=min_cs)
        kinds = [kind] * len(units)
        new_text = _build_karaoke(units, cs, kinds, link=link)
        if not snapshotted:
            workspace.snapshot(did)
            snapshotted = True
        _set_text(entry, doc, new_text)
        changed += 1
        line_cs = _ms_to_cs(max(0, line_end - line_start))
        results.append({
            "index": line_index,
            "skipped": False,
            "old_text": text,
            "text": new_text,
            "start_ms": span_start,
            "end_ms": span_end,
            "span_ms": span_ms,
            "span_cs": span_cs,
            "snapped": snapped,
            "min_cs": min_cs,
            "syllables": [{"index": i, "text": u["text"], "prefix": u["prefix"],
                           "kind": kinds[i], "duration_cs": cs[i],
                           "duration_ms": cs[i] * 10}
                          for i, u in enumerate(units)],
            "durations_cs": cs,
            "duration_sum_cs": sum(cs),
            "line_duration_cs": line_cs,
            "discrepancy_cs": sum(cs) - line_cs,
            "matches_line_duration": sum(cs) == line_cs,
            "message": _discrepancy_message(sum(cs), line_cs),
        })
    return ok(
        ok_=True,
        text_source="line",
        doc_id=did,
        mode=mode,
        kind=kind,
        link=link,
        weights=weights,
        min_cs=min_cs,
        exact_sum_guaranteed=True,
        sum_within_one_cs=True,
        lines_changed=changed,
        lines=results,
    )


def _strip_k(piece: str) -> str:
    return T.remove_tags(piece, _K_NAMES)


# --------------------------------------------------------------------------- timings


@_guard
def ass_karaoke_set_timings(selection=None, index=None, doc_id=None, timings=None,
                            unit: str = "ms", mode: str = "marker", kind: str | None = None,
                            strict: bool = True, marker: str = "|",
                            pattern: str | None = None) -> dict:
    """Write explicit per-syllable durations or absolute times.

    Arguments
    ---------
    ``timings``  list of durations, list of ``{"duration_ms"|"duration_cs"|"start_ms"+
                 "end_ms"}`` dicts, or — when ``unit`` is an absolute spelling —
                 one start time per syllable, a list of ``[start, end]`` pairs, or
                 ``n + 1`` boundary times.
    ``unit``     ``"ms"`` / ``"cs"`` (values are **durations**), or
                 ``"absolute_ms"`` / ``"absolute_cs"`` / ``"absolute"`` /
                 ``"times"`` (values are **absolute times**).  Absolute times must
                 be monotonically increasing.
    ``kind``     force every syllable to this kind; ``None`` (default) keeps each
                 syllable's existing kind.
    ``strict``   ``True`` (default): a count mismatch raises :class:`ToolError`
                 naming both counts.  ``False`` pads with zeros / drops extras.
    ``mode``/``marker``/``pattern``
                 how to split a line that has **no** karaoke tags yet.

    Returns
    -------
    ``{"text_source": "line", "doc_id", "unit", "kind", "strict",
       "lines": [{"index", "syllable_count", "timing_count", "padded",
       "old_text", "text", "syllables": [{"text", "kind", "duration_cs",
       "duration_ms"}], "duration_sum_cs", "line_duration_cs",
       "discrepancy_cs", "matches_line_duration", "message"}],
       "lines_changed": n, "reported_not_stretched": True}``

    Durations that do not add up to the line duration are reported in
    ``discrepancy_cs``/``message``; they are never stretched to fit.
    """
    where = "ass_karaoke_set_timings"
    if timings is None:
        raise ToolError(f"{where}: timings= is required")
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = _targets(selection, index, doc, where=where)
    if not indices:
        raise ToolError(f"{where}: selection matched no lines")
    if kind is not None:
        kind = _check_kind(kind)
    unit_name = (unit or "ms").strip().lower()
    absolute = unit_name in ("absolute", "times", "absolute_ms", "abs_ms")
    if unit_name not in ("ms", "cs", "absolute", "times", "absolute_ms", "absolute_cs",
                         "abs_ms", "abs_cs"):
        raise ToolError(f"{where}: unit must be ms, cs, absolute_ms or absolute_cs, got {unit!r}")
    unit_cs = unit_name.endswith("cs") or unit_name == "cs"
    values = _normalise_timings(timings, absolute=absolute, unit_cs=unit_cs, where=where)
    results: list[dict] = []
    changed = 0
    snapshotted = False
    for line_index in indices:
        entry = entry_at(doc, line_index)
        text = entry.text
        pieces = _unit_pieces(text, mode, marker, pattern)
        count = len(pieces)
        expected_counts = {count, count + 1} if absolute else {count}
        if values and len(values) not in expected_counts:
            if strict:
                raise ToolError(
                    f"{where}: timings count {len(values)} does not match syllable count "
                    f"{count} for line {line_index} (pass strict=False to pad/truncate)"
                )
            padded = True
        else:
            padded = False
        # absolute -> durations
        if absolute and values:
            cs_values = _absolute_to_durations(values, count, entry)
        else:
            cs_values = [int(round(v / 10.0)) if not unit_cs else int(v) for v in values]
        cs_values = [max(0, int(v)) for v in cs_values]
        if len(cs_values) < count:
            cs_values = cs_values + [0] * (count - len(cs_values))
        cs_values = cs_values[:count]
        try:
            new_text = K.apply_timings(text, cs_values, kind=kind, mode=mode,
                                       marker=marker, pattern=pattern)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{where}: could not apply timings to line {line_index}: {exc}") from exc
        if not snapshotted:
            workspace.snapshot(did)
            snapshotted = True
        _set_text(entry, doc, new_text)
        changed += 1
        final = _units(new_text)
        final_cs = [u["cs"] for u in final if u["text"]] or [u["cs"] for u in final]
        sum_cs = sum(final_cs)
        line_cs = _ms_to_cs(max(0, entry.end_ms - entry.start_ms))
        results.append({
            "index": line_index,
            "syllable_count": count,
            "timing_count": len(values),
            "padded": padded,
            "old_text": text,
            "text": new_text,
            "syllables": [{"text": u["text"], "kind": u["kind"], "duration_cs": u["cs"],
                           "duration_ms": u["cs"] * 10} for u in final if u["text"]],
            "durations_cs": final_cs,
            "duration_sum_cs": sum_cs,
            "line_duration_cs": line_cs,
            "discrepancy_cs": sum_cs - line_cs,
            "matches_line_duration": sum_cs == line_cs,
            "message": _discrepancy_message(sum_cs, line_cs),
        })
    return ok(
        ok_=True,
        text_source="line",
        doc_id=did,
        unit=unit_name,
        kind=kind,
        strict=strict,
        reported_not_stretched=True,
        lines_changed=changed,
        lines=results,
    )


def _normalise_timings(timings, *, absolute: bool, unit_cs: bool, where: str) -> list:
    if isinstance(timings, (str, bytes)) or not hasattr(timings, "__iter__"):
        raise ToolError(f"{where}: timings must be a list, got {timings!r}")
    out: list = []
    for item in timings:
        if isinstance(item, dict):
            if "duration_cs" in item or "duration_ms" in item:
                out.append({"duration_ms": item.get("duration_ms"),
                            "duration_cs": item.get("duration_cs")})
            elif "start_ms" in item or "end_ms" in item:
                out.append([int(item.get("start_ms", 0)), int(item.get("end_ms", 0))])
            else:
                raise ToolError(f"{where}: timing dict needs duration_ms/duration_cs "
                                f"or start_ms/end_ms, got {item!r}")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append([int(item[0]), int(item[1])])
        else:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                raise ToolError(f"{where}: timing {item!r} is not a number") from None
    if absolute:
        for i, item in enumerate(out):
            if isinstance(item, dict):
                raise ToolError(f"{where}: without absolute times use duration_ms/"
                                f"duration_cs dicts, got {item!r}")
        flat: list[int] = []
        for item in out:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(int(item))
        for i in range(1, len(flat)):
            if flat[i] <= flat[i - 1]:
                raise ToolError(
                    f"{where}: absolute times must be monotonically increasing — "
                    f"value {i} ({flat[i]}) is not greater than {flat[i - 1]}"
                )
        return out
    return out


def _absolute_to_durations(values, count: int, entry: EventEntry) -> list[int]:
    """Absolute times (starts, [start,end] pairs or n+1 boundaries) -> cs durations."""
    if all(isinstance(v, list) for v in values) and values:
        cs = [_ms_to_cs(v[1] - v[0]) for v in values]
        return cs
    flat = [int(v) for v in values]
    if len(flat) == count + 1:
        cs = [_ms_to_cs(flat[i + 1] - flat[i]) for i in range(count)]
        return cs
    starts = list(flat)
    if len(starts) < count:
        starts = starts + [entry.end_ms] * (count - len(starts))
    ends = starts[1:count] + [max(entry.end_ms, starts[count - 1] if count else entry.end_ms)]
    return [_ms_to_cs(ends[i] - starts[i]) for i in range(count)]


# --------------------------------------------------------------------------- retime


@_guard
def ass_karaoke_retime(selection, doc_id: str | None = None, mode: str = "proportional",
                       new_start_ms: int | None = None, new_end_ms: int | None = None,
                       shift_ms: int | None = None, factor: float | None = None,
                       min_cs: int = 1, dry_run: bool = False) -> dict:
    """Retime existing karaoke to a new span, reporting each syllable before/after.

    Arguments
    ---------
    ``selection``  lines to retime (``None`` uses the session selection).
    ``mode``       ``"proportional"`` keeps the existing relative syllable lengths,
                   ``"even"`` gives every syllable the same duration.
    ``new_start_ms`` / ``new_end_ms``
                   new span.  When given, the line's own ``Start``/``End`` are set
                   to them as well (that is what "retime the line" means); when
                   both are omitted the line's current span is used, optionally
                   moved by ``shift_ms`` or stretched by ``factor``.
    ``shift_ms``   move the whole span by this many ms (no length change).
    ``factor``     scale the span length about the line start.
    ``min_cs``     minimum syllable duration; an impossible span raises
                   :class:`ToolError`.
    ``dry_run``    compute and report without writing anything.

    Returns
    -------
    ``{"doc_id", "mode", "dry_run", "lines": [{"index", "old_text", "text",
       "old_start_ms", "old_end_ms", "start_ms", "end_ms", "span_cs",
       "times_changed", "syllables": [{"index", "text", "kind", "before_cs",
       "after_cs", "before_ms", "after_ms", "delta_cs"}],
       "duration_sum_cs", "line_duration_cs", "discrepancy_cs",
       "matches_line_duration", "message"}], "lines_changed": n,
       "reported_not_stretched": True}``

    The new centisecond durations sum to the new span exactly; any mismatch
    against the line duration is reported rather than stretched.
    """
    where = "ass_karaoke_retime"
    mode_name = (mode or "proportional").strip().lower()
    if mode_name not in ("proportional", "even"):
        raise ToolError(f"{where}: mode must be proportional or even, got {mode!r}")
    if shift_ms is not None and factor is not None:
        raise ToolError(f"{where}: pass shift_ms or factor, not both")
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = _targets(selection, None, doc, where=where)
    if not indices:
        raise ToolError(f"{where}: selection matched no lines")
    results: list[dict] = []
    changed = 0
    snapshotted = False
    for line_index in indices:
        entry = entry_at(doc, line_index)
        text = entry.text
        if not K.parse_karaoke_has_tags(text):
            raise ToolError(
                f"{where}: line {line_index} has no karaoke tags to retime "
                "(use ass_karaoke_generate first)"
            )
        units = [u for u in _units(text) if u["text"]]
        line_start, line_end = _line_span(entry)
        if new_start_ms is not None or new_end_ms is not None:
            target_start = line_start if new_start_ms is None else int(new_start_ms)
            target_end = line_end if new_end_ms is None else int(new_end_ms)
            times_changed = (target_start != line_start) or (target_end != line_end)
        elif shift_ms is not None:
            target_start, target_end = line_start + int(shift_ms), line_end + int(shift_ms)
            times_changed = False
        elif factor is not None:
            target_start = line_start
            target_end = line_start + int(round((line_end - line_start) * float(factor)))
            times_changed = False
        else:
            target_start, target_end = line_start, line_end
            times_changed = False
        if target_end < target_start:
            raise ToolError(f"{where}: line {line_index}: end {target_end} is before start "
                            f"{target_start}")
        span_ms = target_end - target_start
        if mode_name == "even":
            weights = [1.0] * len(units)
        else:
            weights = [max(1.0, float(u["cs"])) for u in units]
        cs, span_cs = _allocate(weights, span_ms, min_cs=min_cs)
        kinds = [u["kind"] for u in units]
        new_text = _build_karaoke(units, cs, kinds)
        before = [u["cs"] for u in units]
        if not dry_run:
            if not snapshotted:
                workspace.snapshot(did)
                snapshotted = True
            _set_text(entry, doc, new_text)
            if times_changed:
                _set_times(entry, doc, start_ms=target_start, end_ms=target_end)
            changed += 1
        line_cs = _ms_to_cs(max(0, (target_end if times_changed else line_end) -
                                (target_start if times_changed else line_start)))
        results.append({
            "index": line_index,
            "old_text": text,
            "text": new_text,
            "old_start_ms": line_start,
            "old_end_ms": line_end,
            "start_ms": target_start,
            "end_ms": target_end,
            "span_ms": span_ms,
            "span_cs": span_cs,
            "times_changed": times_changed,
            "syllables": [{"index": i, "text": u["text"], "kind": kinds[i],
                           "before_cs": before[i], "after_cs": cs[i],
                           "before_ms": before[i] * 10, "after_ms": cs[i] * 10,
                           "delta_cs": cs[i] - before[i]}
                          for i, u in enumerate(units)],
            "durations_cs": cs,
            "duration_sum_cs": sum(cs),
            "line_duration_cs": line_cs,
            "discrepancy_cs": sum(cs) - line_cs,
            "matches_line_duration": sum(cs) == line_cs,
            "message": _discrepancy_message(sum(cs), line_cs),
        })
    return ok(ok_=True, doc_id=did, mode=mode_name, dry_run=bool(dry_run),
              reported_not_stretched=True, lines_changed=changed, lines=results)


# --------------------------------------------------------------------------- shift/scale


@_guard
def ass_karaoke_shift(selection, shift_ms: int, doc_id: str | None = None,
                      shift_times: bool = False) -> dict:
    """Shift every karaoke duration by ``shift_ms`` (negative trims, floor 0).

    ``shift_times=True`` also moves the line's own ``Start``/``End`` by the same
    amount.  Returns ``{"doc_id", "shift_ms", "shift_times", "lines": [{"index",
    "old_text", "text", "syllables": [{"index", "text", "kind", "before_cs",
    "after_cs"}], "duration_sum_cs", "line_duration_cs", "discrepancy_cs",
    "matches_line_duration", "message", "start_ms", "end_ms"}],
    "lines_changed": n, "clamped_cs": n}`` — durations never go below zero, so
    the sum is reported rather than stretched.
    """
    where = "ass_karaoke_shift"
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = _targets(selection, None, doc, where=where)
    if not indices:
        raise ToolError(f"{where}: selection matched no lines")
    results: list[dict] = []
    changed = 0
    clamped = 0
    snapshotted = False
    for line_index in indices:
        entry = entry_at(doc, line_index)
        text = entry.text
        before = [u["cs"] for u in _units(text) if u["text"]]
        try:
            new_text = K.shift_karaoke(text, int(shift_ms))
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{where}: line {line_index}: {exc}") from exc
        after = [u["cs"] for u in _units(new_text) if u["text"]]
        clamped += sum(1 for b, a in zip(before, after) if b + _ms_to_cs(int(shift_ms)) != a)
        if not snapshotted:
            workspace.snapshot(did)
            snapshotted = True
        _set_text(entry, doc, new_text)
        if shift_times:
            _set_times(entry, doc, start_ms=entry.start_ms + int(shift_ms),
                       end_ms=entry.end_ms + int(shift_ms))
        changed += 1
        sum_cs = sum(after)
        line_cs = _ms_to_cs(max(0, entry.end_ms - entry.start_ms))
        results.append({
            "index": line_index,
            "old_text": text,
            "text": new_text,
            "start_ms": entry.start_ms,
            "end_ms": entry.end_ms,
            "syllables": [{"index": i, "text": u["text"], "kind": u["kind"],
                           "before_cs": before[i] if i < len(before) else None,
                           "after_cs": u["cs"]}
                          for i, u in enumerate([u for u in _units(new_text) if u["text"]])],
            "duration_sum_cs": sum_cs,
            "line_duration_cs": line_cs,
            "discrepancy_cs": sum_cs - line_cs,
            "matches_line_duration": sum_cs == line_cs,
            "message": _discrepancy_message(sum_cs, line_cs),
        })
    return ok(ok_=True, doc_id=did, shift_ms=int(shift_ms), shift_times=bool(shift_times),
              clamped_cs=clamped, lines_changed=changed, lines=results)


@_guard
def ass_karaoke_scale(selection, factor: float, doc_id: str | None = None,
                      min_cs: int = 1, scale_times: bool = False) -> dict:
    """Multiply every karaoke duration by ``factor`` (never below ``min_cs``).

    ``scale_times=True`` also scales the line's own span about its start (the
    ``Start`` time is kept).  Returns the same shape as :func:`ass_karaoke_shift`
    with ``factor`` instead of ``shift_ms``; rounding/clamping is reported via
    ``duration_sum_cs``/``discrepancy_cs``/``message``, never stretched.
    """
    where = "ass_karaoke_scale"
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        raise ToolError(f"{where}: factor must be a number, got {factor!r}") from None
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = _targets(selection, None, doc, where=where)
    if not indices:
        raise ToolError(f"{where}: selection matched no lines")
    results: list[dict] = []
    changed = 0
    snapshotted = False
    for line_index in indices:
        entry = entry_at(doc, line_index)
        text = entry.text
        before = [u["cs"] for u in _units(text) if u["text"]]
        try:
            new_text = K.scale_karaoke(text, factor, min_cs=int(min_cs))
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{where}: line {line_index}: {exc}") from exc
        after = [u["cs"] for u in _units(new_text) if u["text"]]
        if not snapshotted:
            workspace.snapshot(did)
            snapshotted = True
        _set_text(entry, doc, new_text)
        if scale_times:
            new_end = entry.start_ms + int(round((entry.end_ms - entry.start_ms) * factor))
            _set_times(entry, doc, end_ms=new_end)
        changed += 1
        sum_cs = sum(after)
        line_cs = _ms_to_cs(max(0, entry.end_ms - entry.start_ms))
        results.append({
            "index": line_index,
            "old_text": text,
            "text": new_text,
            "start_ms": entry.start_ms,
            "end_ms": entry.end_ms,
            "syllables": [{"index": i, "text": u["text"], "kind": u["kind"],
                           "before_cs": before[i] if i < len(before) else None,
                           "after_cs": u["cs"]}
                          for i, u in enumerate([u for u in _units(new_text) if u["text"]])],
            "duration_sum_cs": sum_cs,
            "line_duration_cs": line_cs,
            "discrepancy_cs": sum_cs - line_cs,
            "matches_line_duration": sum_cs == line_cs,
            "message": _discrepancy_message(sum_cs, line_cs),
        })
    return ok(ok_=True, doc_id=did, factor=factor, min_cs=int(min_cs),
              scale_times=bool(scale_times), lines_changed=changed, lines=results)


# --------------------------------------------------------------------------- kind


@_guard
def ass_karaoke_set_kind(selection, kind: str = "k", doc_id: str | None = None,
                         only_matching: str | None = None) -> dict:
    """Convert karaoke tags to another kind (``k``, ``kf``, ``ko``, ``kt``).

    ``only_matching`` limits the conversion to syllables whose current kind is
    that value (e.g. only ``k`` -> ``kf``).  Durations, text and every other
    override tag are untouched.  Returns
    ``{"doc_id", "kind", "only_matching", "lines": [{"index", "old_text",
    "text", "before_kinds", "after_kinds", "changed"}], "lines_changed": n,
    "tags_converted": n}``.
    """
    where = "ass_karaoke_set_kind"
    kind = _check_kind(kind)
    only = None if only_matching is None else _check_kind(only_matching)
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = _targets(selection, None, doc, where=where)
    if not indices:
        raise ToolError(f"{where}: selection matched no lines")
    results: list[dict] = []
    changed = 0
    converted = 0
    snapshotted = False
    for line_index in indices:
        entry = entry_at(doc, line_index)
        text = entry.text
        before_kinds = [u["kind"] for u in _units(text) if u["text"]]
        new_text, hits = _convert_kinds(text, kind, only)
        after_kinds = [u["kind"] for u in _units(new_text) if u["text"]]
        row_changed = hits > 0 and new_text != text
        if row_changed:
            if not snapshotted:
                workspace.snapshot(did)
                snapshotted = True
            _set_text(entry, doc, new_text)
            changed += 1
            converted += hits
        results.append({"index": line_index, "old_text": text, "text": new_text,
                        "before_kinds": before_kinds, "after_kinds": after_kinds,
                        "tags_converted": hits, "changed": row_changed})
    return ok(ok_=True, doc_id=did, kind=kind, only_matching=only,
              tags_converted=converted, lines_changed=changed, lines=results)


def _convert_kinds(text: str, kind: str, only_matching: str | None) -> tuple[str, int]:
    """Rewrite karaoke tags in place, keeping every other tag and duration."""
    parsed = T.parse(text)
    out: list[str] = []
    hits = 0
    for seg in parsed.segments:
        if not isinstance(seg, T.TagBlock):
            out.append(seg.render())
            continue
        new_tags = []
        for tag in seg.tags:
            if tag.name in K.KARAOKE_TAGS_ALL:
                current = K.KARAOKE_ALIASES.get(tag.name, tag.name)
                if only_matching is None or current == only_matching:
                    if current != kind or tag.raw != _k_tag(kind, tag.int_arg(0)):
                        hits += 1
                    new_tags.append(T.Tag(name=kind, arg=tag.arg, paren=False, dirty=True))
                    continue
            new_tags.append(tag)
        if new_tags:
            out.append(T.TagBlock(tags=new_tags, dirty=True).render())
    return "".join(out), hits


# --------------------------------------------------------------------------- remove


@_guard
def ass_karaoke_remove(selection, doc_id: str | None = None, drop_markers: bool = False,
                       keep_times: bool = False) -> dict:
    """Remove ``\\k``/``\\kf``/``\\ko``/``\\kt`` tags (``karaoke.remove_karaoke``).

    ``drop_markers=True`` also removes the ``|`` syllable markers — they are kept
    by default because they are the source data :func:`ass_karaoke_generate`
    re-splits on.

    ``keep_times`` controls the line's own times.  ``True`` leaves ``Start``/
    ``End`` exactly as they were; ``False`` (default) retightens the line's
    ``End`` to ``Start + karaoke extent`` when the karaoke extent is shorter
    than the line (the trailing silence the karaoke timing defined is dropped).

    Returns ``{"doc_id", "drop_markers", "keep_times", "lines": [{"index",
    "old_text", "text", "removed_tags": n, "old_start_ms", "old_end_ms",
    "start_ms", "end_ms", "times_retightened"}], "lines_changed": n,
    "tags_removed": n}``.
    """
    where = "ass_karaoke_remove"
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = _targets(selection, None, doc, where=where)
    if not indices:
        raise ToolError(f"{where}: selection matched no lines")
    results: list[dict] = []
    changed = 0
    removed_total = 0
    snapshotted = False
    for line_index in indices:
        entry = entry_at(doc, line_index)
        text = entry.text
        old_start, old_end = _line_span(entry)
        units = [u for u in _units(text) if u["text"]]
        karaoke_ms = sum(u["cs"] for u in units) * 10
        before_count = len(T.parse_tags(text))
        removed = sum(1 for t in T.parse_tags(text) if t.name in K.KARAOKE_TAGS_ALL)
        markers = 0
        if drop_markers:
            markers = T.plain_text(text).count("|")
        try:
            new_text = K.remove_karaoke(text, drop_markers=drop_markers)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{where}: line {line_index}: {exc}") from exc
        new_start, new_end = old_start, old_end
        retightened = False
        if not keep_times and removed and karaoke_ms > 0 and karaoke_ms < (old_end - old_start):
            new_end = old_start + karaoke_ms
            retightened = True
        if new_text != text or retightened:
            if not snapshotted:
                workspace.snapshot(did)
                snapshotted = True
            _set_text(entry, doc, new_text)
            if retightened:
                _set_times(entry, doc, end_ms=new_end)
            changed += 1
        removed_total += removed + markers
        results.append({
            "index": line_index,
            "old_text": text,
            "text": new_text,
            "removed_tags": removed,
            "removed_markers": markers,
            "replaced_tags": max(0, before_count - len(T.parse_tags(new_text))),
            "old_start_ms": old_start,
            "old_end_ms": old_end,
            "start_ms": new_start,
            "end_ms": new_end,
            "times_retightened": retightened,
        })
    return ok(ok_=True, doc_id=did, drop_markers=bool(drop_markers),
              keep_times=bool(keep_times), tags_removed=removed_total,
              lines_changed=changed, lines=results)


# --------------------------------------------------------------------------- auto


@_guard
def ass_karaoke_auto_timings(index: int, doc_id: str | None = None, mode: str = "syllable",
                             weights: str = "char_class", start_ms: int | None = None,
                             end_ms: int | None = None, apply: bool = False,
                             kind: str = "k", marker: str = "|",
                             pattern: str | None = None) -> dict:
    """Propose karaoke timings for one line from its duration (no write by default).

    Arguments
    ---------
    ``index``  the line to analyse (0-based).
    ``mode``   ``"syllable"`` (split on ``marker``), ``"char"``, ``"word"`` or
               ``"regex"`` (needs ``pattern``).
    ``weights`` ``"char_class"`` (default), ``"char"`` or ``"even"``.
    ``start_ms`` / ``end_ms``
               span to distribute; both default to the line's own times.
    ``apply``  ``False`` (default) only proposes the timings and the text that
               *would* be written.  ``True`` writes the generated ``\\k`` tags to
               the line (after ``workspace.snapshot``).  ``apply`` is a documented
               extension of the required signature: the default is a pure read.
    ``kind``   tag kind to generate when applying.

    Returns
    -------
    ``{"doc_id", "index", "mode", "split_mode", "weights", "kind", "applied",
       "start_ms", "end_ms", "span_ms", "span_cs", "syllables": [{"index",
       "text", "weight", "duration_cs", "duration_ms"}], "durations_cs",
       "duration_sum_cs", "line_duration_cs", "discrepancy_cs",
       "matches_line_duration", "message", "proposed_text", "old_text"}``
    """
    where = "ass_karaoke_auto_timings"
    mode_name = (mode or "syllable").strip().lower()
    if mode_name not in _AUTO_MODES:
        raise ToolError(f"{where}: mode must be one of {', '.join(sorted(_AUTO_MODES))}, "
                        f"got {mode!r}")
    split_mode = _AUTO_MODES[mode_name]
    kind = _check_kind(kind)
    if split_mode == "regex" and not pattern:
        raise ToolError(f"{where}: mode='regex' needs pattern=")
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    entry = entry_at(doc, int(index))
    text = entry.text
    line_start, line_end = _line_span(entry)
    span_start = line_start if start_ms is None else int(start_ms)
    span_end = line_end if end_ms is None else int(end_ms)
    if span_end < span_start:
        raise ToolError(f"{where}: end_ms is before start_ms")
    base = K.remove_karaoke(text, drop_markers=(split_mode != "marker"))
    pieces = _split_pieces(base, split_mode, marker, pattern)
    if not pieces:
        raise ToolError(f"{where}: line {index} has no syllables to time")
    texts = [T.plain_text(p) for p in pieces]
    w = _weights_for(texts, weights)
    cs, span_cs = _allocate(w, span_end - span_start, min_cs=1)
    proposed = K.generate_k(base, kind=kind, durations_cs=cs, mode=split_mode,
                            marker=marker, pattern=pattern)
    applied = False
    if apply:
        workspace.snapshot(did)
        _set_text(entry, doc, proposed)
        applied = True
    line_cs = _ms_to_cs(max(0, line_end - line_start))
    return ok(
        ok_=True,
        doc_id=did,
        index=int(index),
        mode=mode_name,
        split_mode=split_mode,
        weights=weights,
        kind=kind,
        applied=applied,
        start_ms=span_start,
        end_ms=span_end,
        span_ms=span_end - span_start,
        span_cs=span_cs,
        syllables=[{"index": i, "text": texts[i], "weight": round(w[i], 4),
                    "duration_cs": cs[i], "duration_ms": cs[i] * 10}
                   for i in range(len(texts))],
        durations_cs=cs,
        duration_sum_cs=sum(cs),
        line_duration_cs=line_cs,
        discrepancy_cs=sum(cs) - line_cs,
        matches_line_duration=sum(cs) == line_cs,
        message=_discrepancy_message(sum(cs), line_cs),
        proposed_text=proposed,
        old_text=text,
    )


# --------------------------------------------------------------------------- styles


@_guard
def ass_karaoke_styles(doc_id: str | None = None, create: bool = True,
                       prefix: str = "Karaoke") -> dict:
    """List (and optionally create) the conventional karaoke template styles.

    The standard set is ``Karaoke``, ``Karaoke_2`` and ``Karaoke_3`` — the usual
    Aegisub template trio — built from one base: Arial 60, white
    ``PrimaryColour``, blue ``SecondaryColour`` (so ``\\kf``/``\\ko`` sweeps from
    blue into white), black outline, ``&H80000000&`` shadow, bold, BorderStyle 1,
    Outline 2, Shadow 1, margins 10, Encoding 1.  The variants differ only in
    ``Alignment``: ``Karaoke`` bottom (2), ``Karaoke_2`` top (8), ``Karaoke_3``
    middle (5), which is how the tutorial template sets stack their layers.
    ``prefix`` renames the trio (``prefix``, ``prefix_2``, ``prefix_3``).

    ``create=False`` reports what exists without touching the document.

    Returns ``{"doc_id", "prefix", "created": [names], "existing": [names],
    "styles": [{"name", "exists", "created_now", "values": {...}}]}``.
    """
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    wanted = []
    for suffix, overrides in _KARAOKE_STYLE_VARIANTS:
        spec = dict(_KARAOKE_STYLE_BASE)
        spec.update(overrides)
        spec["Name"] = f"{prefix}{suffix}"
        wanted.append((spec["Name"], spec))
    styles: list[dict] = []
    created: list[str] = []
    existing: list[str] = []
    snapshotted = False
    for name, spec in wanted:
        found = doc.get_style(name)
        if found is not None:
            existing.append(name)
            styles.append({"name": name, "exists": True, "created_now": False,
                           "values": {k: found.get(k) for k in spec if k != "Name"}})
            continue
        if not create:
            styles.append({"name": name, "exists": False, "created_now": False,
                           "documented_values": spec})
            continue
        if not snapshotted:
            workspace.snapshot(did)
            snapshotted = True
        doc.add_style(spec)
        created.append(name)
        styles.append({"name": name, "exists": True, "created_now": True,
                       "values": {k: spec[k] for k in spec if k != "Name"}})
    return ok(ok_=True, doc_id=did, prefix=prefix, create=bool(create),
              standard_names=[name for name, _ in wanted],
              created=created, existing=existing, styles=styles,
              style_names=doc.style_names())


# --------------------------------------------------------------------------- template


_TEMPLATE_VARS = ("syl", "sdur")
_TEMPLATE_VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_TEMPLATE_UNSUPPORTED = [
    "full templater class blocks (`template syl`, `template char`, `template line`, "
    "`template pre-line`) — only one plain template line is expanded",
    "`code` lines, `once` lines, `mixin` lines and Lua expressions",
    "every variable except $syl and $sdur (e.g. $start, $end, $mid, $kdur, "
    "$orgline.*, $syl.* attributes, $li, $si)",
    "inline-fx and the \\t / \\k application rules of the real templater",
    "retiming of template lines and the karaoke effect library (chorus, lead-in/out, "
    "fades, `template syl char` layers)",
]


@_guard
def ass_karaoke_template(selection, doc_id: str | None = None, template_line=None,
                         template_index: int | None = None, style: str | None = None,
                         mode: str = "marker", replace_existing: bool = False,
                         create_styles: bool = True, dry_run: bool = False) -> dict:
    """Expand a karaoke template line into one generated line per syllable.

    This is a **documented subset** of the Aegisub karaoke templater.  Only
    ``$syl`` (the syllable text) and ``$sdur`` (the syllable duration in
    centiseconds) are substituted; the template's other override tags (including
    ``\\t`` transforms) are copied verbatim into every generated line.  Classes,
    ``code``/``once``/``mixin`` lines, the rest of the variable set and the
    effect library are **not** supported — see ``unsupported`` in the result.

    Arguments
    ---------
    ``selection``   the lines whose syllables are expanded.
    ``template_line`` / ``template_index``
                    the template: a raw string, or a 0-based line index.  With a
                    real template index the generated lines are inserted directly
                    after it and copy its Layer, Actor, Effect, kind, style and
                    margins; with a raw string they are appended after the last
                    selected line.
    ``style``       style for the generated lines (default: the template's style,
                    else the ``Karaoke`` style).
    ``mode``        how to split the target line when it has no karaoke tags yet
                    (``marker`` / ``char`` / ``word`` / ``regex``).
    ``replace_existing``
                    remove previously generated lines (same Effect marker) that sit
                    immediately after the template line before inserting.
    ``create_styles``
                    create the standard karaoke styles when ``style`` names one
                    that does not exist yet.
    ``dry_run``     report the lines that would be generated without inserting.

    Returns
    -------
    ``{"doc_id", "template_subset": True, "subset_of_aegisub_karaoke_templater":
       True, "supported": [...], "unsupported": [...], "variables_used": [...],
       "unknown_variables": [...], "template": {...}, "dry_run",
       "inserted_count": n, "replaced_lines": n, "generated": [{"target_index",
       "syllable_index", "text", "syllable", "duration_cs", "start_ms", "end_ms",
       "start", "end", "layer", "actor", "style", "kind",
       "inserted_index"}], "generated_indices": [...]}``
    """
    where = "ass_karaoke_template"
    split_mode = _split_mode(mode, where=where)
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = _targets(selection, None, doc, where=where)
    if not indices:
        raise ToolError(f"{where}: selection matched no lines")

    # -- template ---------------------------------------------------------
    if isinstance(template_line, int) and template_index is None:
        template_index, template_line = template_line, None
    template_entry = None
    if template_index is not None:
        template_entry = entry_at(doc, int(template_index))
        template_text = template_entry.text
    elif isinstance(template_line, str):
        template_text = template_line
    else:
        raise ToolError(f"{where}: pass template_line=<string> or template_index=<line index>")
    if create_styles:
        wanted_style = style or (template_entry.get("Style") if template_entry else "") or "Karaoke"
        if doc.get_style(wanted_style) is None:
            ass_karaoke_styles(doc_id=did, create=True,
                               prefix=wanted_style.rstrip("_123") or "Karaoke")
    target_style = style or (template_entry.get("Style") if template_entry else "") or "Karaoke"
    layer = template_entry.get("Layer", "0") if template_entry else "0"
    actor = template_entry.get("Name", "") if template_entry else ""
    marker = (template_entry.get("Effect", "") if template_entry else "") or _TEMPLATE_MARKER
    kind = (template_entry.kind if template_entry else "Dialogue") or "Dialogue"

    found_vars = sorted(set(_TEMPLATE_VAR_RE.findall(template_text)))
    unknown = [v for v in found_vars if v not in _TEMPLATE_VARS]
    variables_used = [v for v in found_vars if v in _TEMPLATE_VARS]
    if not variables_used:
        raise ToolError(
            f"{where}: the template line has no $syl or $sdur placeholder "
            f"(found {found_vars or 'none'})"
        )

    # -- expand -----------------------------------------------------------
    generated: list[dict] = []
    pending_entries: list[EventEntry] = []
    for target_index in indices:
        entry = entry_at(doc, target_index)
        text = entry.text
        line_start, line_end = _line_span(entry)
        timed = K.parse_karaoke_has_tags(text)
        if timed:
            rows = K.karaoke_timings(text, line_start_ms=line_start, line_end_ms=line_end)
            pieces = [{"text": r["text"], "kind": r.get("kind", "k"),
                       "start_ms": int(r["start_ms"]), "end_ms": int(r["end_ms"]),
                       "duration_cs": int(r["duration_cs"])} for r in rows]
        else:
            base = K.remove_karaoke(text)
            raw_pieces = _split_pieces(base, split_mode, "|", None)
            texts = [T.plain_text(p) for p in raw_pieces]
            if not texts:
                raise ToolError(f"{where}: line {target_index} has no syllables")
            w = _weights_for(texts, "char_class")
            cs, span_cs = _allocate(w, max(0, line_end - line_start), min_cs=1)
            t = line_start
            pieces = []
            for i, piece_text in enumerate(texts):
                pieces.append({"text": piece_text, "kind": "k", "start_ms": t,
                               "end_ms": t + cs[i] * 10, "duration_cs": cs[i]})
                t += cs[i] * 10
        for i, syl in enumerate(pieces):
            rendered = _render_template(template_text, syl["text"], syl["duration_cs"])
            row = {
                "target_index": target_index,
                "syllable_index": i,
                "text": rendered,
                "syllable": syl["text"],
                "duration_cs": syl["duration_cs"],
                "start_ms": int(syl["start_ms"]),
                "end_ms": int(syl["end_ms"]),
                "start": _ms_to_time(syl["start_ms"]),
                "end": _ms_to_time(syl["end_ms"]),
                "layer": layer,
                "actor": actor,
                "style": target_style,
                "kind": kind,
                "timings_source": "karaoke_tags" if timed else "generated",
                "inserted_index": None,
            }
            generated.append(row)
            if not dry_run:
                pending_entries.append(_make_event(doc, kind, {
                    "Layer": layer, "Start": row["start"], "End": row["end"],
                    "Style": target_style, "Name": actor, "MarginL": 0,
                    "MarginR": 0, "MarginV": 0, "Effect": marker, "Text": rendered,
                }))
    if not generated:
        raise ToolError(f"{where}: nothing to generate")

    replaced = 0
    inserted_indices: list[int] = []
    if not dry_run:
        workspace.snapshot(did)
        anchor = template_entry
        if anchor is None:
            anchor = entry_at(doc, indices[-1])
        section, position = _find_entry(doc, anchor)
        if section is None:  # pragma: no cover - defensive
            raise ToolError(f"{where}: could not locate the anchor line in the document")
        if replace_existing:
            j = position + 1
            while j < len(section.entries):
                nxt = section.entries[j]
                if isinstance(nxt, EventEntry) and nxt.get("Effect", "") == marker:
                    del section.entries[j]
                    replaced += 1
                    continue
                break
        for offset, new_entry in enumerate(pending_entries):
            section.entries.insert(position + 1 + offset, new_entry)
        doc.dirty = True
        wanted = {id(e): i for i, e in enumerate(doc.events())}
        for row, new_entry in zip(generated, pending_entries):
            row["inserted_index"] = wanted.get(id(new_entry))
            if row["inserted_index"] is not None:
                inserted_indices.append(row["inserted_index"])
    return ok(
        ok_=True,
        doc_id=did,
        template_subset=True,
        subset_of_aegisub_karaoke_templater=True,
        supported=[
            "$syl substitution (syllable visible text)",
            "$sdur substitution (syllable duration in centiseconds)",
            "the template line's other override tags and \\t transforms copied verbatim",
            "correct absolute start/end times taken from the target line's karaoke tags "
            "(generated from the line span when the target is untimed)",
            "layer and actor copied from the template line",
            "generated lines inserted directly after the template line",
        ],
        unsupported=list(_TEMPLATE_UNSUPPORTED),
        variables_used=variables_used,
        unknown_variables=unknown,
        unknown_variables_kept_literal=bool(unknown),
        template={
            "index": None if template_entry is None else int(template_index),
            "text": template_text,
            "style": target_style,
            "layer": layer,
            "actor": actor,
            "kind": kind,
            "effect_marker": marker,
       },
        dry_run=bool(dry_run),
        replaced_lines=replaced,
        generated_count=len(generated),
        inserted_count=0 if dry_run else len(pending_entries),
        generated=generated,
        generated_indices=inserted_indices,
    )


def _render_template(template_text: str, syllable: str, duration_cs: int) -> str:
    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name == "syl":
            return syllable
        if name == "sdur":
            return str(int(duration_cs))
        return match.group(0)  # unknown variables stay literal

    return _TEMPLATE_VAR_RE.sub(repl, template_text)


def _find_entry(doc, entry: EventEntry):
    for section in doc.event_sections(create=False):
        for i, candidate in enumerate(section.entries):
            if candidate is entry:
                return section, i
    return None, -1


def _make_event(doc, kind: str, fields: dict) -> EventEntry:
    order = doc.format_order()
    entry = EventEntry(f"{kind}:", kind, order, ["" for _ in order])
    defaults = {
        "Layer": 0, "Start": "0:00:00.00", "End": "0:00:00.00", "Style": "Default",
        "Name": "", "MarginL": 0, "MarginR": 0, "MarginV": 0, "Effect": "", "Text": "",
    }
    for key in order:
        if key in fields and fields[key] is not None:
            entry.set(key, fields[key])
        elif key in defaults:
            entry.set(key, defaults[key])
    return entry


# --------------------------------------------------------------------------- export


@_guard
def ass_karaoke_export(selection=None, doc_id: str | None = None, format: str = "srv2",
                       include_untimed: bool = False) -> dict:
    """Export syllable timings for other karaoke tools.

    Writes into ``workspace.output_dir`` and returns both the path and the
    content.  The file is named ``<document stem>_karaoke.<ext>``.

    Arguments
    ---------
    ``selection``  lines to export.  ``None`` (default) exports every line that
                   carries karaoke tags; pass a selection to override.
    ``format``     ``"srv2"`` (default) tab-separated interchange with a ``#``
                   header and columns ``line, syl, start_ms, end_ms, dur_cs,
                   dur_ms, kind, text``; ``"txt"`` human-readable
                   ``<line_index> <h:mm:ss.cc> <h:mm:ss.cc> <text>`` rows;
                   ``"csv"`` the same columns as srv2 as RFC 4180 CSV.
    ``include_untimed``
                   also export lines without karaoke tags as a single syllable
                   covering the line span.

    Returns
    -------
    ``{"doc_id", "format", "path", "filename", "content", "lines": n,
       "syllables": n, "bytes": n, "output_dir": "..."}``
    """
    where = "ass_karaoke_export"
    fmt = (format or "srv2").strip().lower()
    if fmt not in ("srv2", "txt", "csv"):
        raise ToolError(f"{where}: format must be srv2, txt or csv, got {format!r}")
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    if selection is None:
        indices = [i for i, e in enumerate(doc.events())
                   if K.parse_karaoke_has_tags(e.text) or include_untimed]
    else:
        indices = resolve_indices(selection, doc, selection=workspace.selection)
    rows: list[dict] = []
    lines_used = 0
    for line_index in indices:
        entry = entry_at(doc, line_index)
        text = entry.text
        start_ms, end_ms = _line_span(entry)
        timed = K.parse_karaoke_has_tags(text)
        if timed:
            timing_rows = K.karaoke_timings(text, line_start_ms=start_ms, line_end_ms=end_ms)
        elif include_untimed:
            timing_rows = [{"index": 0, "text": T.plain_text(text), "kind": "k",
                            "start_ms": start_ms, "end_ms": end_ms,
                            "duration_ms": max(0, end_ms - start_ms),
                            "duration_cs": _ms_to_cs(max(0, end_ms - start_ms))}]
        else:
            continue
        lines_used += 1
        for i, row in enumerate(timing_rows):
            rows.append({
                "line": line_index,
                "syl": i,
                "start_ms": int(row.get("start_ms", start_ms)),
                "end_ms": int(row.get("end_ms", start_ms)),
                "dur_cs": int(row.get("duration_cs", 0)),
                "dur_ms": int(row.get("duration_ms", 0)),
                "kind": row.get("kind", "k"),
                "text": str(row.get("text", "")),
                "style": entry.get("Style"),
                "actor": entry.get("Name"),
            })
    content = _export_content(rows, fmt)
    out_dir = workspace.set_output_dir(workspace.output_dir)
    stem = Path(workspace.path(did).name).stem if workspace.path(did) else "untitled"
    filename = f"{stem}_karaoke.{fmt}"
    path = Path(out_dir) / filename
    path.write_text(content, encoding="utf-8", newline="")
    return ok(ok_=True, doc_id=did, format=fmt, path=str(path), filename=filename,
              output_dir=str(out_dir), content=content, lines=lines_used,
              syllables=len(rows), bytes=len(content.encode("utf-8")))


def _clean_cell(value: str) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _export_content(rows: list[dict], fmt: str) -> str:
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["line", "syl", "start_ms", "end_ms", "dur_cs", "dur_ms",
                         "kind", "text"])
        for row in rows:
            writer.writerow([row["line"], row["syl"], row["start_ms"], row["end_ms"],
                             row["dur_cs"], row["dur_ms"], row["kind"], row["text"]])
        return buf.getvalue()
    if fmt == "txt":
        out = ["# aegisub-mcp karaoke export (txt)",
               "# <line_index> <start> <end> <syllable_text>"]
        for row in rows:
            out.append(f'{row["line"]}\t{_ms_to_time(row["start_ms"])}\t'
                       f'{_ms_to_time(row["end_ms"])}\t{_clean_cell(row["text"])}')
        return "\n".join(out) + "\n"
    out = ["# srv2 karaoke timing export (aegisub-mcp)",
           "# columns: line\tsyl\tstart_ms\tend_ms\tdur_cs\tdur_ms\tkind\ttext"]
    for row in rows:
        out.append("\t".join([
            str(row["line"]), str(row["syl"]), str(row["start_ms"]), str(row["end_ms"]),
            str(row["dur_cs"]), str(row["dur_ms"]), str(row["kind"]),
            _clean_cell(row["text"]),
        ]))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- register


def register(mcp, ws=None):
    """Register every ``ass_*`` callable in this module with ``mcp``.

    ``ws`` is accepted for symmetry with the other tool modules but unused: the
    tools operate on the process-wide :data:`aegisub_mcp.tools.base.workspace`
    singleton.  Returns the sorted list of registered tool names.
    """
    names: list[str] = []
    for name in sorted(globals()):
        obj = globals()[name]
        if name.startswith("ass_") and callable(obj):
            mcp.tool()(obj)
            names.append(name)
    return names
