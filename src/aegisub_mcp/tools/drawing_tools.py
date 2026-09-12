"""Drawing, clipping and font tools for the Aegisub MCP server.

This module exposes ASS vector drawings (``\\p``), vector/rectangular clips
(``\\clip`` / ``\\iclip``) and font inspection as MCP tools.  It is a thin,
strictly typed layer on top of :mod:`aegisub_mcp.asscore.drawing`,
:mod:`aegisub_mcp.asscore.tags` and :mod:`aegisub_mcp.asscore.measure`; all
geometry and font logic lives there.

Coordinate spaces (read this before using the tools)
----------------------------------------------------
Two spaces appear throughout:

``script resolution``
    The document's ``PlayResX``/``PlayResY`` pixel grid.  Rectangular clip
    coordinates and every ``normalised`` value returned by this module are in
    script resolution.

``scale space`` (the line's ``\\pN`` level)
    A drawing written while ``\\pN`` is active is *authored* in units that are
    rendered at ``2 ** (1 - N)`` script pixels each.  So ``\\p1`` means one
    drawing unit is one script pixel, ``\\p2`` halves the drawing, ``\\p3``
    quarters it, and so on.  ``\\clip(N, ...)`` uses exactly the same level
    convention (a clip with no explicit scale uses the line's current ``\\p``
    level).

**Unless a docstring says otherwise, every drawing tool reads and writes
coordinates in the line's own scale space** -- i.e. the numbers you see in the
file.  Use :func:`ass_convert_clip_scale` or :func:`ass_get_clips` /
:func:`ass_set_clip` when you need script-resolution numbers.

Rounding
--------
ASS drawing coordinates are, by convention, integers.  Every drawing this
module produces is serialised with **integer coordinates using round-half-away-
from-zero** (:data:`COORD_DIGITS` = 0); :func:`_round_coord` documents the rule.
Input is never silently rewritten beyond that rounding, and no subpath is ever
dropped -- :func:`ass_split_drawing` covers pre-``m`` orphan commands in their
own part so ``split -> join`` is lossless.

Naming
------
Every tool that reports a bounding-box centre returns it twice: ``center``
(used internally) and ``centre`` (the spelling used in the tool specification).
Both always carry the same value.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..asscore import drawing as D
from ..asscore import measure as M
from ..asscore import tags as T
from .base import ToolError, entry_at, ok, resolve_indices, workspace

__all__ = [
    "register",
    "ass_get_drawing",
    "ass_set_drawing",
    "ass_drawing_info",
    "ass_transform_drawing",
    "ass_drawing_bbox",
    "ass_drawing_to_svg",
    "ass_svg_to_drawing",
    "ass_split_drawing",
    "ass_join_drawings",
    "ass_get_clips",
    "ass_set_clip",
    "ass_remove_clip",
    "ass_convert_clip_scale",
    "ass_scale_drawing",
    "ass_list_fonts",
    "ass_match_font",
    "ass_fonts_with_char",
    "ass_font_coverage",
    "ass_glyph_check",
    "ass_fonts_used",
]

#: Decimals kept when serialising an ASS drawing; 0 = integer coordinates.
COORD_DIGITS = 0

#: Letters that only exist in SVG path syntax, used to tell a ``spec`` string
#: that is an SVG path from one that is an ASS drawing (both share ``m``/``l``).
_SVG_MARKERS = set("HVQTZAhvqtza")

_CLIP_NAMES = ("clip", "iclip")

#: Numeric token matcher shared by the SVG viewBox / path parsers.
_NUMBER_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


# ---------------------------------------------------------------------------
# internal helpers (never registered: they do not start with ``ass_``)
# ---------------------------------------------------------------------------


def _tool(fn):
    """Wrap a tool so no raw exception can escape to the MCP client.

    :class:`~aegisub_mcp.tools.base.ToolError` is passed through untouched;
    every other exception becomes ``ToolError(str(exc))``.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - contract: never leak raw errors
            raise ToolError(f"{fn.__name__}: {exc}") from exc

    return wrapper


def _round_coord(value: float, digits: int = COORD_DIGITS) -> float:
    """Round *value* half-away-from-zero to *digits* decimals.

    Python's built-in ``round`` uses banker's rounding (``round(0.5) == 0``),
    which would make two drawings that differ by half a unit land on different
    integers depending on parity.  ASS authors expect ``0.5 -> 1`` and
    ``-0.5 -> -1``, which is what this implements.
    """
    v = float(value)
    if digits <= 0:
        return float(math.floor(v + 0.5)) if v >= 0 else float(math.ceil(v - 0.5))
    factor = 10 ** int(digits)
    scaled = v * factor
    return (math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)) / factor


def _num(value: float, digits: int = COORD_DIGITS) -> str:
    """Format a coordinate, dropping a pointless fractional part."""
    rounded = _round_coord(value, digits)
    if digits <= 0:
        return str(int(rounded))
    text = f"{rounded:.{int(digits)}f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_drawing(drawing: D.Drawing, digits: int = COORD_DIGITS) -> str:
    """Serialise a :class:`Drawing` with explicit rounding (see module docstring)."""
    parts: list[str] = []
    for cmd in drawing.commands:
        if not cmd.points:
            parts.append(cmd.kind)
            continue
        numbers = " ".join(_num(v, digits) for point in cmd.points for v in point)
        parts.append(f"{cmd.kind} {numbers}")
    return " ".join(parts)


def _round_drawing(drawing: D.Drawing, digits: int = COORD_DIGITS) -> D.Drawing:
    """Return a copy of *drawing* with every coordinate rounded explicitly.

    Uses the half-away-from-zero rule of :func:`_round_coord` (not Python's
    banker's rounding) so ``1.25 -> 1`` and ``2.5 -> 3`` deterministically.
    """
    return D.Drawing([
        D.Command(cmd.kind, [(_round_coord(x, digits), _round_coord(y, digits)) for x, y in cmd.points])
        for cmd in drawing.commands
    ])


def _as_list(bbox: tuple[float, float, float, float] | None) -> list[float] | None:
    return None if bbox is None else [float(v) for v in bbox]


def _size_of(bbox) -> list[float]:
    return [float(v) for v in D.bbox_size(bbox)]


def _center_of(bbox) -> list[float]:
    return [float(v) for v in D.bbox_center(bbox)]


def _command_dict(cmd: D.Command) -> dict[str, Any]:
    return {
        "kind": cmd.kind,
        "args": [float(v) for point in cmd.points for v in point],
        "points": [[float(x), float(y)] for x, y in cmd.points],
        "text": cmd.text(),
    }


def _parse_drawing(drawing_text: str) -> D.Drawing:
    """Parse drawing text, turning parse failures into a user-facing error."""
    text = "" if drawing_text is None else str(drawing_text).strip()
    if not text:
        raise ToolError("empty drawing text")
    try:
        parsed = D.parse_drawing(text)
    except ValueError as exc:
        raise ToolError(f"could not parse drawing: {exc}") from exc
    if not parsed.commands:
        raise ToolError(f"no ASS drawing commands found in {text[:60]!r}")
    return parsed


def _parse_drawing_allow_empty(drawing_text: str) -> D.Drawing:
    text = "" if drawing_text is None else str(drawing_text).strip()
    return D.Drawing([]) if not text else _parse_drawing(text)


def _scale_factor(level: float) -> float:
    """Script pixels per drawing unit for a ``\\p``/clip scale *level*.

    Level ``N`` renders the drawing at ``2 ** (1 - N)`` script pixels per unit.
    """
    return 2.0 ** (1.0 - float(level))


def _p_scale(text: str, default: int = 1) -> int:
    """Last ``\\p`` level in *text* (``0`` after a ``\\r``); *default* if absent."""
    level = int(default)
    for tag in T.parse_tags(text):
        if tag.name == "p":
            level = tag.int_arg(default)
        elif tag.name == "r":
            level = 0
    return level


def _looks_like_line(text: str) -> bool:
    """True when *text* is a full override line rather than bare path data."""
    return "{" in text or re.search(r"\\p\s*\d", text) is not None


def _opaque_parts(text: str) -> tuple[str, str, str]:
    """Split a line into ``(prefix, drawing, suffix)`` treating tag blocks as opaque.

    This is a hardened variant of :func:`tags.drawing_parts` used by every tool in
    this module.  The stock helper appends *every* segment while ``\\p`` is active,
    including later tag blocks, so a vector clip such as
    ``{\\p1}m 0 0 l 1 0{\\clip(1,m 2 2 l 3 3)}`` leaks the clip's ``m 2 2 l 3 3``
    text into the reported path data (and a write-back would then destroy the clip).
    Here only real text segments are collected, so the path data is exactly the
    characters between the ``\\pN`` block and the following tag block.
    """
    segments = T.parse(text).segments
    levels: list[int] = []
    level = 0
    for seg in segments:
        if isinstance(seg, T.TagBlock):
            for tag in seg.tags:
                if tag.name == "p":
                    level = tag.int_arg(0)
        levels.append(level)

    def render(seg: Any) -> str:
        return seg.render() if isinstance(seg, T.TagBlock) else seg.text

    first = None
    for i, seg in enumerate(segments):
        if not isinstance(seg, T.TagBlock) and levels[i] > 0:
            first = i
            break
    if first is None:
        return text, "", ""

    last = first
    for j in range(first, len(segments)):
        if isinstance(segments[j], T.TagBlock) or levels[j] <= 0:
            break
        last = j
    prefix = "".join(render(s) for s in segments[:first])
    drawing = "".join(s.text for s in segments[first:last + 1] if isinstance(s, T.TextSegment))
    suffix = "".join(render(s) for s in segments[last + 1:])
    return prefix, drawing.strip(), suffix


def _split_drawing_parts(text: str) -> tuple[str, str, str]:
    """``(prefix, drawing, suffix)`` for a line, via ``tags.drawing_parts`` + repair."""
    prefix, drawing, suffix = T.drawing_parts(text)
    if "{" in drawing or "}" in drawing:
        # tags.drawing_parts absorbed a later tag block into the path data.
        return _opaque_parts(text)
    return prefix, drawing, suffix


def _line_source(index: int, doc_id: str | None) -> dict[str, Any]:
    doc = workspace.get(doc_id)
    entry = entry_at(doc, index)
    raw = entry.text
    prefix, drawing, suffix = _split_drawing_parts(raw)
    scale = _p_scale(prefix or raw)
    return {
        "doc": doc,
        "doc_id": workspace.doc_id_for(doc),
        "index": int(index),
        "line": raw,
        "prefix": prefix,
        "drawing": drawing,
        "suffix": suffix,
        "scale": scale,
        "has_drawing": bool(drawing.strip()),
        "source": "line",
    }


def _resolve_input(index: int | None, text: str | None, doc_id: str | None) -> dict[str, Any]:
    """Resolve the ``(text= | index=+doc_id=)`` pair shared by every drawing tool."""
    if index is not None and text is not None:
        raise ToolError("pass either text= or index= (+ doc_id=), not both")
    if index is None and text is None:
        raise ToolError("pass text= (raw drawing/line) or index= (+ doc_id=)")

    if text is not None:
        raw = str(text)
        if _looks_like_line(raw):
            prefix, drawing, suffix = _split_drawing_parts(raw)
            return {
                "doc": None,
                "doc_id": doc_id,
                "index": None,
                "line": raw,
                "prefix": prefix,
                "drawing": drawing,
                "suffix": suffix,
                "scale": _p_scale(prefix or raw),
                "has_drawing": bool(drawing.strip()),
                "source": "text",
            }
        stripped = raw.strip()
        return {
            "doc": None,
            "doc_id": doc_id,
            "index": None,
            "line": None,
            "prefix": "",
            "drawing": stripped,
            "suffix": "",
            "scale": 1,
            "has_drawing": bool(stripped),
            "source": "text",
        }

    if index is None:
        raise ToolError("pass a line index= (with doc_id=) or raw drawing text=")
    return _line_source(int(index), doc_id)


def _require_drawing(info: dict[str, Any]) -> D.Drawing:
    if not info["has_drawing"]:
        where = f"line {info['index']}" if info["index"] is not None else "the given text"
        raise ToolError(f"{where} contains no \\p drawing")
    return _parse_drawing(info["drawing"])


def _clip_dict(tag: T.Tag, line_scale: float) -> dict[str, Any]:
    """Describe one ``\\clip``/``\\iclip`` tag, including script-resolution coords."""
    inverse = tag.name == "iclip"
    try:
        spec = D.parse_clip_arg(tag.arg)
    except ValueError as exc:
        raise ToolError(f"could not parse \\{tag.name}({tag.arg}): {exc}") from exc

    explicit = spec["scale"]
    level = float(explicit) if explicit is not None else float(line_scale)
    factor = _scale_factor(level)
    data: dict[str, Any] = {
        "tag": tag.name,
        "inverse": inverse,
        "raw": tag.arg,
        "kind": "rect" if spec["type"] == "rect" else "vector",
        "scale": None if explicit is None else float(explicit),
        "effective_scale": level,
        "scale_explicit": explicit is not None,
        "units_per_script_pixel": factor,
    }
    if spec["type"] == "rect":
        coords = [float(v) for v in spec["coords"]]
        data["coords"] = list(coords)
        data["normalised"] = [c * factor for c in coords]
        x1, y1, x2, y2 = data["normalised"]
        data["bbox"] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    else:
        drawing: D.Drawing = spec["drawing"]
        box = D.bbox(drawing)
        data["drawing"] = _format_drawing(drawing)
        data["coords"] = [[float(x), float(y)] for x, y in drawing.points()]
        data["normalised"] = [[x * factor, y * factor] for x, y in drawing.points()]
        data["bbox"] = None if box is None else [v * factor for v in box]
        data["normalised_bbox"] = data["bbox"]
        data["bbox"] = None if box is None else [float(v) for v in box]
    return data


def _line_clips(text: str, line_scale: float | None = None) -> list[dict[str, Any]]:
    level = _p_scale(text) if line_scale is None else line_scale
    return [
        _clip_dict(tag, level)
        for tag in T.parse_tags(text)
        if tag.name in _CLIP_NAMES
    ]


def _clips_of(info: dict[str, Any]) -> list[dict[str, Any]]:
    return _line_clips(info["line"], info["scale"]) if info["line"] else []


def _rewrite_clips(text: str, transform) -> str:
    """Rewrite every clip tag with ``transform(name, arg) -> new_arg | None``.

    ``None`` drops the tag.  Unlike :func:`tags.set_tag` this keeps *every*
    clip on the line, so multiple clips survive.
    """
    parsed = T.parse(text)
    out: list[str] = []
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            body: list[str] = []
            for tag in seg.tags:
                if tag.name in _CLIP_NAMES:
                    new_arg = transform(tag.name, tag.arg)
                    if new_arg is None:
                        continue
                    body.append(f"\\{tag.name}({new_arg})")
                else:
                    body.append(tag.render())
            if body:
                out.append("{" + "".join(body) + "}")
        else:
            out.append(seg.text)
    return "".join(out)


def _parse_rect(text: str) -> list[float]:
    numbers = [float(tok) for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))]
    if len(numbers) != 4:
        raise ToolError(f"rectangle needs 4 coordinates (x1,y1,x2,y2), got {len(numbers)}")
    return numbers


def _looks_like_rect(text: str) -> bool:
    try:
        return len(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))) == 4 and not any(
            ch.isalpha() for ch in str(text)
        )
    except Exception:  # pragma: no cover - defensive
        return False


def _looks_like_svg(text: str) -> bool:
    return any(ch in _SVG_MARKERS for ch in str(text))


def _selection_spec(selection: Any, doc) -> Any:
    """Explicit selection wins, else the session selection, else every line."""
    if selection is not None:
        return selection
    if workspace.selection:
        return "selection"
    return None


def _resolve_text_source(text: str | None, index: int | None, doc_id: str | None):
    """Resolve a ``(text | index)`` pair for the font tools."""
    if index is not None and text is None:
        info = _line_source(int(index), doc_id)
        return info["line"], info
    if text is None:
        raise ToolError("pass text= or index= (+ doc_id=)")
    raw = str(text)
    return raw, {
        "doc": None,
        "doc_id": doc_id,
        "index": None,
        "line": raw,
        "source": "text",
    }


def _style_font(doc, style_name: str) -> dict[str, Any]:
    style = doc.get_style(style_name) if doc is not None else None
    if style is None:
        raise ToolError(f"style {style_name!r} not found in the script")
    return {
        "style": style_name,
        "family": (style.get("Fontname") or "").strip(),
        "bold": str(style.get("Bold", "0")).strip() not in ("", "0", "false", "False"),
        "italic": str(style.get("Italic", "0")).strip() not in ("", "0", "false", "False"),
    }


def _measure_rect(ass_text: str, time_ms: float) -> dict[str, Any] | None:
    try:
        result = M.measure_render(ass_text, time_ms)
    except Exception as exc:  # noqa: BLE001 - measurement is best-effort
        return {"error": str(exc), "rect": None}
    return {
        "rect": result.get("rect"),
        "bbox": result.get("bbox"),
        "time_ms": result.get("time_ms"),
        "ink_pixels": result.get("ink_pixels"),
        "empty": result.get("empty"),
    }


# ---------------------------------------------------------------------------
# drawings
# ---------------------------------------------------------------------------


@_tool
def ass_get_drawing(
    index: int | None = None,
    text: str | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Inspect the drawing part of a line, or a raw drawing string.

    Pass exactly one of:

    * ``index`` + ``doc_id`` -- the line at that 0-based index (``doc.events()``
      order) is split with ``tags.drawing_parts`` and its drawing is inspected;
    * ``text`` -- either a full override line (anything containing ``{`` or a
      ``\\p`` tag, split the same way) or bare drawing path data.

    ``scale`` is the ``\\p`` level in effect for the drawing, so **the returned
    commands, bbox, size, centre and path length are all in that scale space**
    (units as written in the file); multiply by
    ``2 ** (1 - scale)`` for script pixels.  ``normalised_*`` mirrors of the
    bbox/size are provided already converted.

    Returns a dict with ``source`` (``"line"`` or ``"text"``), ``doc_id``,
    ``index``, ``drawing`` (the raw drawing text), ``scale``, ``has_drawing``,
    ``commands`` (list of ``{kind, args, points, text}``), ``bbox``,
    ``normalised_bbox``, ``size``, ``normalised_size``, ``center`` (aliased as
    ``centre``), ``subpath_count``, ``point_count``, ``path_length``,
    ``svg_path`` and ``clips`` (see :func:`ass_get_clips` for that shape).
    """
    info = _resolve_input(index, text, doc_id)
    drawing = _parse_drawing_allow_empty(info["drawing"])
    box = D.bbox(drawing) if drawing.commands else None
    factor = _scale_factor(info["scale"])
    subpaths = D.split_subpaths(drawing) if drawing.commands else []
    return ok(
        source=info["source"],
        doc_id=info["doc_id"],
        index=info["index"],
        drawing=info["drawing"],
        scale=info["scale"],
        has_drawing=info["has_drawing"],
        commands=[_command_dict(cmd) for cmd in drawing.commands],
        bbox=_as_list(box),
        normalised_bbox=None if box is None else [v * factor for v in box],
        size=_size_of(box),
        normalised_size=[v * factor for v in _size_of(box)],
        center=_center_of(box),
        centre=_center_of(box),
        subpath_count=len(subpaths),
        point_count=len(drawing.points()),
        path_length=D.path_length(drawing) if drawing.commands else 0.0,
        svg_path=D.to_svg_path(drawing) if drawing.commands else "",
        clips=_clips_of(info),
        coordinate_space="line scale space (\\p level); multiply by 2**(1-scale) for script pixels",
    )


@_tool
def ass_set_drawing(
    index: int,
    drawing_text: str,
    doc_id: str | None = None,
    scale: int | None = None,
    keep_tags: bool = True,
) -> dict[str, Any]:
    """Replace the drawing part of line *index*.

    ``drawing_text`` is parsed and re-serialised with integer coordinates, so
    the stored text is canonical (this is the only rewriting applied).
    ``scale``, when given, sets the line's ``\\p`` level (``\\p<scale>``); when
    omitted an existing ``\\p`` tag is left alone and a missing one is added as
    ``\\p1`` so the drawing actually renders.

    ``keep_tags=True`` keeps every non-drawing override tag and anything after
    the drawing (typically the closing ``{\\p0}``); ``keep_tags=False`` rebuilds
    the line as ``{\\p<scale>}<drawing>`` and discards all other text.

    Returns ``{doc_id, index, source, old_drawing, drawing, scale, text}``.
    """
    doc = workspace.get(doc_id)
    entry = entry_at(doc, int(index))
    old = entry.text
    prefix, old_drawing, suffix = _split_drawing_parts(old)
    new_drawing = _format_drawing(_parse_drawing(drawing_text))
    level = int(scale) if scale is not None else _p_scale(prefix) or 1
    if level <= 0:
        raise ToolError(f"drawing scale must be >= 1 for a drawing, got {level}")

    if keep_tags:
        if not prefix:
            prefix = f"{{\\p{level}}}"
            suffix = ""
        else:
            prefix = T.set_tag(prefix, "p", str(level))
    else:
        prefix, suffix = f"{{\\p{level}}}", ""
    new_text = prefix + new_drawing + suffix

    workspace.snapshot(workspace.doc_id_for(doc))
    entry.set("Text", new_text)
    return ok(
        doc_id=workspace.doc_id_for(doc),
        index=int(index),
        source="line",
        old_drawing=old_drawing,
        drawing=new_drawing,
        scale=level,
        keep_tags=bool(keep_tags),
        text=new_text,
    )


@_tool
def ass_drawing_info(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Parse a drawing only (no measurement, no line needed).

    Coordinates are in the drawing's own scale space (see the module
    docstring).  Returns ``{source, doc_id, index, drawing, scale, commands,
    subpaths: [{index, bbox, size, text, point_count}], bbox, size, center
    (aliased as ``centre``), subpath_count, point_count, path_length}``.
    """
    info = _resolve_input(index, text, doc_id)
    drawing = _require_drawing(info)
    box = D.bbox(drawing)
    subpaths = D.split_subpaths(drawing)
    return ok(
        source=info["source"],
        doc_id=info["doc_id"],
        index=info["index"],
        drawing=info["drawing"],
        scale=info["scale"],
        commands=[_command_dict(cmd) for cmd in drawing.commands],
        subpaths=[
            {
                "index": i,
                "text": _format_drawing(part),
                "bbox": _as_list(D.bbox(part)),
                "size": _size_of(D.bbox(part)),
                "point_count": len(part.points()),
            }
            for i, part in enumerate(subpaths)
        ],
        bbox=_as_list(box),
        size=_size_of(box),
        center=_center_of(box),
        centre=_center_of(box),
        subpath_count=len(subpaths),
        point_count=len(drawing.points()),
        path_length=D.path_length(drawing),
        coordinate_space="line scale space (\\p level)",
    )


@_tool
def ass_transform_drawing(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
    action: str = "translate",
    dx: float = 0.0,
    dy: float = 0.0,
    factor: float = 1.0,
    angle_deg: float = 0.0,
    origin_x: float | None = None,
    origin_y: float | None = None,
    axis: str = "x",
    in_place: bool = False,
) -> dict[str, Any]:
    """Transform a drawing and return the new path data (or write it back).

    Coordinates in and out are the drawing's own scale-space units (see the
    module docstring); results are serialised with integer coordinates.

    ``action`` and the parameters it uses:

    ``translate``       ``dx``, ``dy`` -- shift every point.
    ``scale``           ``factor`` about ``(origin_x, origin_y)``; the origin
                        defaults to the drawing's bbox centre so it stays put.
    ``scale_to_size``   ``dx``, ``dy`` = target width/height; uniform scale
                        (aspect ratio kept), drawing moved into ``(0,0,w,h)``.
    ``stretch_to_bbox`` ``dx``, ``dy`` = target width/height; independent axes
                        (aspect ratio deliberately broken).
    ``rotate``          ``angle_deg`` clockwise on screen about the origin
                        (bbox centre by default).
    ``mirror``          ``axis="x"`` flips left/right, ``axis="y"`` top/bottom
                        about the origin (bbox centre by default).
    ``center``/``centre_at_origin``  move the bbox centre to ``(origin_x,
                        origin_y)``, default ``(0, 0)``.
    ``reverse``         reverse the winding of every subpath.
    ``flatten``         replace Bézier/B-spline curves with polylines.
    ``simplify``        Douglas-Peucker with tolerance ``factor`` (default 1.0).
    ``round``           snap coordinates to the nearest multiple of ``factor``
                        (``factor=1.0`` = integers, the default).

    ``in_place=True`` with ``index`` writes the result back into the line (after
    ``workspace.snapshot``) and returns the new line text; otherwise no document
    is touched and ``drawing`` holds the new path data.

    Returns ``{source, doc_id, index, action, drawing, bbox, size, center,
    centre, changed, written}`` (``centre`` duplicates ``center``).
    """
    info = _resolve_input(index, text, doc_id)
    drawing = _require_drawing(info)
    origin = None
    if origin_x is not None or origin_y is not None:
        origin = (0.0 if origin_x is None else float(origin_x),
                  0.0 if origin_y is None else float(origin_y))
    center = D.bbox_center(D.bbox(drawing))
    key = str(action or "").strip().lower()

    if key == "translate":
        result = D.translate(drawing, dx, dy)
    elif key == "scale":
        result = D.scale(drawing, float(factor), origin=origin or center)
    elif key == "scale_to_size":
        result = D.stretch_drawing(drawing, dx, dy, keep_aspect=True, align="center")
    elif key == "stretch_to_bbox":
        result = D.stretch_drawing(drawing, dx, dy, keep_aspect=False)
    elif key == "rotate":
        result = D.rotate(drawing, angle_deg, origin=origin or center)
    elif key == "mirror":
        if axis not in ("x", "y"):
            raise ToolError(f"mirror axis must be 'x' or 'y', got {axis!r}")
        result = D.mirror(drawing, horizontal=(axis == "x"), vertical=(axis == "y"),
                          origin=origin or center)
    elif key in ("centre_at_origin", "center_at_origin"):
        result = D.translate(drawing, (0.0 if origin_x is None else origin_x) - center[0],
                             (0.0 if origin_y is None else origin_y) - center[1])
    elif key == "reverse":
        result = D.reverse(drawing)
    elif key == "flatten":
        result = D.flatten(drawing)
    elif key == "simplify":
        result = D.simplify(drawing, float(factor) if factor else 1.0)
    elif key == "round":
        step = float(factor) or 1.0
        result = D.Drawing(
            [D.Command(cmd.kind, [(_round_coord(x / step) * step, _round_coord(y / step) * step)
                                  for x, y in cmd.points])
             for cmd in drawing.commands]
        )
    else:
        raise ToolError(f"unknown drawing action {action!r}; expected one of translate, scale, "
                        "scale_to_size, rotate, mirror, reverse, flatten, simplify, round, "
                        "stretch_to_bbox, centre_at_origin")

    new_text = _format_drawing(result)
    new_box = D.bbox(result)
    written = False
    line_text: str | None = None
    if in_place and info["index"] is not None:
        doc = info["doc"] or workspace.get(info["doc_id"])
        entry = entry_at(doc, info["index"])
        workspace.snapshot(workspace.doc_id_for(doc))
        line_text = info["prefix"] + new_text + info["suffix"]
        entry.set("Text", line_text)
        written = True
    return ok(
        source=info["source"],
        doc_id=info["doc_id"],
        index=info["index"],
        action=key,
        old_drawing=info["drawing"],
        drawing=new_text,
        text=line_text,
        bbox=_as_list(new_box),
        size=_size_of(new_box),
        center=_center_of(new_box),
        centre=_center_of(new_box),
        subpath_count=len(D.split_subpaths(result)),
        point_count=len(result.points()),
        changed=bool(new_text != info["drawing"]),
        written=written,
        in_place=written,
        coordinate_space="line scale space (\\p level)",
    )


@_tool
def ass_drawing_bbox(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Bounding box / size / centre of a drawing.

    Coordinates are in the drawing's own scale space (``\\p`` level).  The box
    is the control polygon box -- Bézier/B-spline handles included -- matching
    what VSFilter/Aegisub use for positioning.

    Returns ``{source, doc_id, index, scale, bbox, size, center, centre,
    normalised_bbox, normalised_size, point_count}``.
    """
    info = _resolve_input(index, text, doc_id)
    drawing = _require_drawing(info)
    box = D.bbox(drawing)
    factor = _scale_factor(info["scale"])
    return ok(
        source=info["source"],
        doc_id=info["doc_id"],
        index=info["index"],
        scale=info["scale"],
        bbox=_as_list(box),
        size=_size_of(box),
        center=_center_of(box),
        centre=_center_of(box),
        normalised_bbox=None if box is None else [v * factor for v in box],
        normalised_size=[v * factor for v in _size_of(box)],
        point_count=len(drawing.points()),
        coordinate_space="line scale space (\\p level)",
    )


@_tool
def ass_drawing_to_svg(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
    path: str | None = None,
    padding: float = 0.0,
) -> dict[str, Any]:
    """Export a drawing as a standalone ``.svg`` file in ``workspace.output_dir``.

    The path data is the drawing's absolute SVG equivalent (scale space units,
    one SVG user unit per drawing unit).  ``padding`` grows the viewBox on every
    side so strokes near the edge are not clipped.

    ``path`` is a filename (resolved inside ``workspace.output_dir``) or an
    absolute path; the default is ``drawing.svg`` / ``drawing_<index>.svg`` and
    is overwritten if it exists.

    Returns ``{path, d, view_box, width, height, bbox, source, doc_id, index}``.
    """
    info = _resolve_input(index, text, doc_id)
    drawing = _require_drawing(info)
    box = D.bbox(drawing)
    if box is None:
        raise ToolError("drawing has no points, nothing to export")
    padded = D.bbox_expand(box, float(padding)) or box
    width = padded[2] - padded[0]
    height = padded[3] - padded[1]
    view_box = f"{_num(padded[0], 2)} {_num(padded[1], 2)} {_num(width, 2)} {_num(height, 2)}"
    d = D.to_svg_path(drawing)
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" '
        f'width="{_num(width, 2)}" height="{_num(height, 2)}">\n'
        f'  <path d="{d}" fill="#ffffff" stroke="none"/>\n'
        "</svg>\n"
    )

    out_dir = Path(workspace.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"drawing_{info['index']}.svg" if info["index"] is not None else "drawing.svg"
    target = Path(path) if path else Path(default_name)
    if not target.is_absolute():
        target = out_dir / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8")

    return ok(
        path=str(target),
        d=d,
        view_box=view_box,
        width=width,
        height=height,
        bbox=_as_list(box),
        padding=float(padding),
        source=info["source"],
        doc_id=info["doc_id"],
        index=info["index"],
    )


@_tool
def ass_svg_to_drawing(
    svg_path: str | None = None,
    d: str | None = None,
    view_box: str | Sequence[float] | None = None,
    scale: float = 1.0,
    round_to: int = 1,
) -> dict[str, Any]:
    """Convert SVG path data (or a whole ``.svg`` file) into ASS drawing text.

    Either ``d`` (one or more ``d`` attributes, space separated) or ``svg_path``
    (the file is read, its ``viewBox`` becomes the origin shift and every
    ``<path d="...">`` is converted).  ``view_box`` given as ``"x y w h"`` or
    ``[x, y, w, h]`` shifts the paths so the box origin becomes ``(0, 0)``.

    ``scale`` multiplies every coordinate; ``round_to`` is the number of decimal
    places kept when serialising (``0`` = integer coordinates).

    SVG and ASS both grow ``y`` downwards, so nothing is flipped.  Supported
    commands: ``M L H V C S Q T A Z`` (relative forms too); quadratics are
    promoted to cubics and arcs to cubic segments.

    Returns ``{drawing, bbox, size, center, path_count, view_box, scale,
    round_to, source}``.
    """
    origin = (0.0, 0.0)
    view: list[float] | None = None
    paths: list[str] = []
    source: str

    if svg_path and d:
        raise ToolError("pass either svg_path= or d=, not both")
    if not svg_path and not d:
        raise ToolError("pass svg_path= or d=")

    if svg_path:
        svg_file = Path(str(svg_path)).expanduser()
        if not svg_file.is_file():
            raise ToolError(f"svg file not found: {svg_file}")
        content = svg_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r'viewBox\s*=\s*"([^"]+)"', content)
        if match:
            nums = [float(tok) for tok in re.findall(_NUMBER_RE, match.group(1))]
            if len(nums) >= 2:
                view = nums[:4] if len(nums) >= 4 else nums
                origin = (view[0], view[1])
        paths = re.findall(r'<path[^>]*\sd\s*=\s*"([^"]*)"', content)
        if not paths:
            raise ToolError(f"no <path d=\"...\"> elements found in {svg_file}")
        source = str(svg_file)
    else:
        paths = [str(d)]
        source = "d"

    if view_box is not None:
        if isinstance(view_box, str):
            numbers = [float(tok) for tok in re.findall(_NUMBER_RE, view_box)]
        else:
            numbers = [float(v) for v in view_box]
        if len(numbers) < 2:
            raise ToolError("view_box needs at least x and y")
        origin = (numbers[0], numbers[1])
        view = numbers[:4] if len(numbers) >= 4 else numbers

    factor = float(scale)
    translate = (-origin[0] * factor, -origin[1] * factor)
    commands: list[D.Command] = []
    for path_data in paths:
        drawing = D.from_svg_path(path_data, scale=factor, translate=translate)
        commands.extend(drawing.commands)
    result = D.Drawing(commands)
    text = _format_drawing(result, int(round_to))
    box = D.bbox(result)
    return ok(
        drawing=text,
        bbox=_as_list(box),
        size=_size_of(box),
        center=_center_of(box),
        centre=_center_of(box),
        path_count=len(paths),
        view_box=view,
        origin=[origin[0], origin[1]],
        scale=factor,
        round_to=int(round_to),
        source=source,
    )


@_tool
def ass_split_drawing(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Split a drawing into one string per subpath (no subpath is ever dropped).

    Commands appearing before the first ``m``/``n`` form their own leading part,
    so ``split`` followed by :func:`ass_join_drawings` is lossless.
    Coordinates stay in the drawing's own scale space.

    Returns ``{source, doc_id, index, scale, count, subpaths: [str, ...],
    parts: [str, ...] (alias of subpaths), point_counts: [int, ...],
    bboxes: [...]}``.
    """
    info = _resolve_input(index, text, doc_id)
    drawing = _require_drawing(info)
    parts = D.split_subpaths(drawing)
    texts = [_format_drawing(part) for part in parts]
    return ok(
        source=info["source"],
        doc_id=info["doc_id"],
        index=info["index"],
        scale=info["scale"],
        count=len(parts),
        subpaths=texts,
        parts=texts,
        point_counts=[len(part.points()) for part in parts],
        bboxes=[_as_list(D.bbox(part)) for part in parts],
    )


@_tool
def ass_join_drawings(parts: Iterable[str] | str | None = None, doc_id: str | None = None) -> dict[str, Any]:
    """Join several drawing strings into one (the inverse of ``ass_split_drawing``).

    ``parts`` is a list of drawing strings (a single string is treated as a
    one-element list).  Every part must parse; empty strings are rejected so a
    subpath can never be silently lost.  Coordinates are serialised as integers.

    Returns ``{drawing, count, subpath_count, point_count, bbox, size, doc_id}``.
    """
    if parts is None:
        raise ToolError("pass parts= as a list of drawing strings")
    if isinstance(parts, str):
        parts = [parts]
    items = list(parts)
    if not items:
        raise ToolError("parts must contain at least one drawing")
    commands: list[D.Command] = []
    for i, part in enumerate(items):
        parsed = _parse_drawing_allow_empty(part)
        if not parsed.commands:
            raise ToolError(f"part {i} is empty; refusing to drop it silently")
        commands.extend(parsed.commands)
    joined = D.Drawing([cmd.copy() for cmd in commands])
    box = D.bbox(joined)
    return ok(
        drawing=_format_drawing(joined),
        count=len(items),
        subpath_count=len(D.split_subpaths(joined)),
        point_count=len(joined.points()),
        bbox=_as_list(box),
        size=_size_of(box),
        doc_id=doc_id,
    )


# ---------------------------------------------------------------------------
# clips
# ---------------------------------------------------------------------------


@_tool
def ass_get_clips(
    index: int | None = None,
    text: str | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """List every ``\\clip`` / ``\\iclip`` tag on a line (or in raw text).

    Each entry has ``tag`` (``"clip"``/``"iclip"``), ``inverse``, ``raw``
    argument, ``kind`` (``"rect"`` or ``"vector"``), the explicit ``scale``
    (``None`` when omitted) and the ``effective_scale`` actually used, plus
    ``coords`` as authored in the clip's scale space and ``normalised``
    coordinates converted to **script resolution** (multiplied by
    ``2 ** (1 - effective_scale)``).  ``bbox`` is the box in scale space and
    ``normalised_bbox`` the same box in script resolution.

    Returns ``{source, doc_id, index, line_scale, count, clips: [...]}``.
    """
    info = _resolve_input(index, text, doc_id)
    clips = _clips_of(info)
    return ok(
        source=info["source"],
        doc_id=info["doc_id"],
        index=info["index"],
        line_scale=info["scale"],
        count=len(clips),
        clips=clips,
    )


def _build_clip_tag(
    spec: str | None,
    rect: str | Sequence[float] | None,
    drawing_text: str | None,
    svg_path: str | None,
    svg_d: str | None,
    scale: float | None,
    inverse: bool,
) -> dict[str, Any]:
    """Build the override tag text for one clip, plus its parsed description."""
    if rect is not None:
        coords = ([float(v) for v in rect] if not isinstance(rect, str) else _parse_rect(rect))
        if len(coords) != 4:
            raise ToolError("rect needs 4 coordinates (x1,y1,x2,y2)")
        authored = [float(c) for c in coords]
        if scale is not None:
            # Coordinates are in the named \p/scale level; convert to script pixels.
            factor = _scale_factor(scale)
            script = [c * factor for c in authored]
        else:
            script = list(authored)
        tag = D.rect_to_clip(script[0], script[1], script[2], script[3], inverse=inverse)
        return {
            "kind": "rect",
            "tag": tag,
            "authored_coords": authored,
            "script_coords": script,
            "input_scale": None if scale is None else float(scale),
        }

    if drawing_text is not None:
        drawing = _parse_drawing(drawing_text)
        text = _format_drawing(drawing)
    elif svg_d is not None:
        converted = ass_svg_to_drawing(d=svg_d, scale=1.0, round_to=0)
        text = converted["drawing"]
    elif svg_path is not None:
        converted = ass_svg_to_drawing(svg_path=svg_path, scale=1.0, round_to=0)
        text = converted["drawing"]
    elif spec is not None and not _looks_like_rect(spec):
        raw = str(spec)
        if _looks_like_svg(raw):
            converted = ass_svg_to_drawing(d=raw, scale=1.0, round_to=0)
            text = converted["drawing"]
        else:
            text = _format_drawing(_parse_drawing(raw))
    else:
        raise ToolError("no clip geometry given: pass rect=, drawing_text=, svg_path=, svg_d= or spec=")

    level = 1.0 if scale is None else float(scale)
    tag = D.drawing_to_clip(text, scale=level, inverse=inverse)
    return {
        "kind": "vector",
        "tag": tag,
        "drawing": text,
        "scale": level,
    }


@_tool
def ass_set_clip(
    selection: Any = None,
    spec: str | None = None,
    doc_id: str | None = None,
    inverse: bool = False,
    scale: float | None = None,
    mode: str = "replace",
    rect: str | Sequence[float] | None = None,
    drawing_text: str | None = None,
    svg_path: str | None = None,
    svg_d: str | None = None,
) -> dict[str, Any]:
    """Set, add or remove a clip on the selected lines.

    ``selection`` accepts anything :func:`base.resolve_indices` understands
    (``None`` = the session selection, or every line when none is set).

    Geometry is taken from the first one of these that is given: ``rect``
    (``"x1,y1,x2,y2"`` or a 4-list, in **script resolution** unless ``scale``
    names the scale space those numbers are written in), ``drawing_text``
    (ASS path data), ``svg_path`` / ``svg_d`` (SVG path data) or ``spec``
    (a rectangle string, an ASS drawing, or an SVG path -- SVG is detected by
    its ``H/V/Q/T/A/Z`` command letters and the result is converted).

    ``scale`` for a vector clip sets the explicit ``\\clip(<scale>,...)`` level;
    when omitted a vector clip is written at level 1 (script resolution).
    ``inverse=True`` emits ``\\iclip``.

    ``mode``:

    * ``"replace"`` -- strip every existing clip from the line, then insert.
    * ``"add"`` -- keep existing clips, insert the new one at the front.
    * ``"remove"`` -- strip every clip tag (``\\clip`` and ``\\iclip``) from the
      lines; the geometry arguments are ignored.

    Returns ``{doc_id, mode, kind, tag, scale, inverse, lines: [{index, changed,
    text, removed, clip}]}``.
    """
    doc = workspace.get(doc_id)
    key = str(mode or "replace").strip().lower()
    if key not in ("replace", "add", "remove"):
        raise ToolError(f"mode must be 'replace', 'add' or 'remove', got {mode!r}")
    indices = resolve_indices(_selection_spec(selection, doc), doc)

    built = None
    if key != "remove":
        built = _build_clip_tag(spec, rect, drawing_text, svg_path, svg_d, scale, inverse)

    snapshot_done = False
    lines: list[dict[str, Any]] = []
    for index in indices:
        entry = entry_at(doc, index)
        text = entry.text
        removed = 0
        if key in ("replace", "remove"):
            strip_all = key == "remove"

            def _drop(name, arg, _inv=inverse, _all=strip_all):
                nonlocal removed
                if _all or (name == "iclip") == _inv:
                    removed += 1
                    return None
                return arg
            text = _rewrite_clips(text, _drop)
        if key != "remove" and built is not None:
            if key == "add":
                # A dedicated leading block: merging into the first block would
                # make the tag engine collapse two \clip tags of the same name.
                text = "{" + built["tag"] + "}" + text
            else:
                text = T.prepend_tags(text, built["tag"])
        changed = text != entry.text
        if changed:
            if not snapshot_done:
                workspace.snapshot(workspace.doc_id_for(doc))
                snapshot_done = True
            entry.set("Text", text)
        lines.append({
            "index": index,
            "changed": changed,
            "removed": removed,
            "tag": None if built is None else built["tag"],
            "text": text,
        })
    return ok(
        doc_id=workspace.doc_id_for(doc),
        mode=key,
        kind=None if built is None else built["kind"],
        tag=None if built is None else built["tag"],
        geometry=None if built is None else built,
        scale=None if built is None else built.get("scale", built.get("input_scale")),
        inverse=bool(inverse),
        lines=lines,
        changed=sum(1 for line in lines if line["changed"]),
    )


@_tool
def ass_remove_clip(
    selection: Any = None,
    doc_id: str | None = None,
    include_inverse: bool = True,
) -> dict[str, Any]:
    """Strip clip tags from the selected lines.

    ``include_inverse=True`` (the default) removes ``\\iclip`` as well as
    ``\\clip``; ``False`` keeps inverse clips.  Blocks left empty by the removal
    are dropped, which is expected for a remove operation.

    Returns ``{doc_id, include_inverse, changed, lines: [{index, removed,
    changed, text}]}``.
    """
    doc = workspace.get(doc_id)
    indices = resolve_indices(_selection_spec(selection, doc), doc)
    snapshot_done = False
    lines: list[dict[str, Any]] = []
    for index in indices:
        entry = entry_at(doc, index)
        removed = 0

        def _drop(name, arg, _all=include_inverse):
            nonlocal removed
            if name == "clip" or _all:
                removed += 1
                return None
            return arg

        text = _rewrite_clips(entry.text, _drop)
        changed = text != entry.text
        if changed:
            if not snapshot_done:
                workspace.snapshot(workspace.doc_id_for(doc))
                snapshot_done = True
            entry.set("Text", text)
        lines.append({"index": index, "removed": removed, "changed": changed, "text": text})
    return ok(
        doc_id=workspace.doc_id_for(doc),
        include_inverse=bool(include_inverse),
        changed=sum(1 for line in lines if line["changed"]),
        lines=[line for line in lines if line["changed"]],
    )


def _convert_one_clip(name: str, arg: str, line_scale: float, target: float) -> tuple[str, dict[str, Any]]:
    """Return ``(new_arg, info)`` converting one vector clip to *target* level."""
    spec = D.parse_clip_arg(arg)
    if spec["type"] != "drawing":
        raise ToolError("only vector clips have a scale to convert")
    source = float(spec["scale"]) if spec["scale"] is not None else float(line_scale)
    if source <= 0:
        raise ToolError(f"clip has a non-positive scale {source}")
    ratio = _scale_factor(source) / _scale_factor(target)
    scaled = _round_drawing(D.scale(spec["drawing"], ratio))
    new_arg = D.format_clip_arg({
        "type": "drawing",
        "scale": float(target),
        "coords": None,
        "drawing": scaled,
        "raw": "",
    })
    return new_arg, {
        "source_scale": source,
        "target_scale": float(target),
        "ratio": ratio,
        "coordinates": [[float(x), float(y)] for x, y in scaled.points()],
    }


@_tool
def ass_convert_clip_scale(
    selection: Any = None,
    target_scale: float = 1.0,
    doc_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rewrite vector clip coordinates from their current ``\\p`` scale to *target_scale*.

    A clip made at level ``S`` renders each unit at ``2 ** (1 - S)`` script
    pixels; to keep the rendered result identical at level ``T`` every
    coordinate is multiplied by ``2 ** (T - S)`` and rounded to the nearest
    integer.  The clip is rewritten with an explicit ``\\clip(T,...)`` level.

    The tool **proves** the conversion by measuring the rendered ink bounding
    box with ``measure.measure_render`` before and after (at the midpoint of the
    first selected line) and reporting both, plus ``converged``.  Measurement
    failing (no ffmpeg) is reported as ``measurement.error`` instead of an
    exception.

    Both measurements carry an ``empty`` flag: if the line renders no ink at
    all (for example a drawing with no ``\\pos``/``\\an`` whose geometry lands
    off-screen) then ``before`` and ``after`` are both empty and ``converged``
    is true for a trivial reason.  A caller that wants a real proof must check
    ``not measurement["before"]["empty"]`` first, which is what the test suite
    does.

    ``dry_run=True`` computes everything and reports the measurement proof
    without touching the document.

    Returns ``{doc_id, target_scale, dry_run, converted, lines: [{index,
    source_scale, target_scale, ratio, before_arg, after_arg, text}],
    measurement: {before, after, delta, converged, time_ms}}``.
    """
    doc = workspace.get(doc_id)
    indices = resolve_indices(_selection_spec(selection, doc), doc)
    events = doc.events()
    before_text = doc.to_text()

    changes: list[dict[str, Any]] = []
    for index in indices:
        entry = events[index]
        text = entry.text
        line_scale = _p_scale(text)
        found: list[dict[str, Any]] = []

        def _convert(name, arg, _ls=line_scale):
            new_arg, info = _convert_one_clip(name, arg, _ls, float(target_scale))
            found.append({"before_arg": arg, "after_arg": new_arg, **info})
            return new_arg

        if any(tag.name in _CLIP_NAMES for tag in T.parse_tags(text)):
            try:
                new_text = _rewrite_clips(text, _convert)
            except ToolError:
                continue
            if found and new_text != text:
                changes.append({
                    "index": index,
                    "text": new_text,
                    "clips": found,
                    "source_scale": found[0]["source_scale"],
                    "target_scale": float(target_scale),
                    "ratio": found[0]["ratio"],
                })

    clone = doc.clone()
    for change in changes:
        clone.events()[change["index"]].set("Text", change["text"])
    after_text = clone.to_text()

    sample_ms = 0.0
    if indices:
        first = events[indices[0]]
        sample_ms = (first.start_ms + first.end_ms) / 2.0
    measurement: dict[str, Any] = {"time_ms": sample_ms}
    if changes:
        before = _measure_rect(before_text, sample_ms)
        after = _measure_rect(after_text, sample_ms)
        measurement["before"] = before
        measurement["after"] = after
        rect_a, rect_b = (before or {}).get("rect"), (after or {}).get("rect")
        if rect_a and rect_b:
            delta = max(
                abs(rect_a[k] - rect_b[k]) for k in ("x", "y", "x1", "y1")
            )
            measurement["delta"] = delta
            measurement["converged"] = delta <= 2.0
        else:
            measurement["delta"] = None
            measurement["converged"] = rect_a is None and rect_b is None
    else:
        measurement["converged"] = True

    if changes and not dry_run:
        workspace.snapshot(workspace.doc_id_for(doc))
        for change in changes:
            events[change["index"]].set("Text", change["text"])

    return ok(
        doc_id=workspace.doc_id_for(doc),
        target_scale=float(target_scale),
        dry_run=bool(dry_run),
        converted=len(changes),
        applied=bool(changes) and not dry_run,
        lines=[
            {
                "index": change["index"],
                "source_scale": change["source_scale"],
                "target_scale": change["target_scale"],
                "ratio": change["ratio"],
                "clips": change["clips"],
                "text": change["text"],
            }
            for change in changes
        ],
        measurement=measurement,
    )


@_tool
def ass_scale_drawing(
    selection: Any = None,
    factor: float = 1.0,
    doc_id: str | None = None,
    include_clips: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scale the drawing part of each selected line (and its clips) by *factor*.

    This is the classic "make everything on the line bigger" helper.  The
    drawing part and every clip are scaled about **one common origin** -- the
    drawing's bbox centre when the line has a drawing, otherwise the first
    clip's own centre -- so the drawing and its clips keep their relative
    position instead of each drifting toward its own centre.  Coordinates remain
    in each clip's own scale space (the ``\\clip(N,...)`` level is preserved) and
    are rounded to integers with the module's explicit half-away-from-zero rule.

    ``include_clips=False`` scales only the ``\\p`` drawing.  ``dry_run=True``
    reports the new text without touching the document.

    Returns ``{doc_id, factor, include_clips, dry_run, changed, applied, lines:
    [{index, old_text, text, drawing_scaled, clips_scaled}]}``.
    """
    doc = workspace.get(doc_id)
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        raise ToolError(f"factor must be a number, got {factor!r}") from None
    if factor <= 0:
        raise ToolError(f"factor must be > 0, got {factor}")

    indices = resolve_indices(_selection_spec(selection, doc), doc)
    events = doc.events()
    changes: list[dict[str, Any]] = []
    for index in indices:
        entry = events[index]
        old_text = entry.text
        text = old_text
        prefix, drawing_text, suffix = _split_drawing_parts(text)
        drawing_scaled = False
        clips_scaled = 0
        origin: tuple[float, float] | None = None

        if drawing_text.strip():
            parsed = D.parse_drawing(drawing_text)
            origin = D.bbox_center(D.bbox(parsed))
            text = prefix + _format_drawing(D.scale(parsed, factor, origin=origin)) + suffix
            drawing_scaled = True

        if include_clips:
            def _scale_clip(name, arg):
                nonlocal clips_scaled, origin
                spec = D.parse_clip_arg(arg)
                if spec["type"] == "rect":
                    x1, y1, x2, y2 = (float(v) for v in spec["coords"])
                    if origin is None:
                        origin = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    ox, oy = origin
                    spec["coords"] = [
                        _round_coord(ox + (x1 - ox) * factor),
                        _round_coord(oy + (y1 - oy) * factor),
                        _round_coord(ox + (x2 - ox) * factor),
                        _round_coord(oy + (y2 - oy) * factor),
                    ]
                else:
                    if origin is None:
                        origin = D.bbox_center(D.bbox(spec["drawing"]))
                    spec["drawing"] = _round_drawing(
                        D.scale(spec["drawing"], factor, origin=origin)
                    )
                clips_scaled += 1
                return D.format_clip_arg(spec)

            text = _rewrite_clips(text, _scale_clip)

        if text != old_text:
            changes.append({
                "index": index,
                "old_text": old_text,
                "text": text,
                "drawing_scaled": drawing_scaled,
                "clips_scaled": clips_scaled,
            })

    if changes and not dry_run:
        workspace.snapshot(workspace.doc_id_for(doc))
        for change in changes:
            events[change["index"]].set("Text", change["text"])

    return ok(
        doc_id=workspace.doc_id_for(doc),
        factor=factor,
        include_clips=bool(include_clips),
        dry_run=bool(dry_run),
        changed=len(changes),
        applied=bool(changes) and not dry_run,
        lines=changes,
    )


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------


@_tool
def ass_list_fonts(pattern: str | None = None, limit: int = 100) -> dict[str, Any]:
    """List installed fonts, optionally filtered by a case-insensitive substring.

    ``pattern`` is a plain substring matched against family, style and file --
    not a fontconfig pattern expression.  ``limit`` caps the number of entries.

    Returns ``{pattern, limit, count, fonts: [{family, style, file, index,
    family_raw}, ...]}``.
    """
    limit = int(limit)
    if limit < 0:
        raise ToolError("limit must be >= 0")
    fonts = M.fonts_list(pattern, limit=limit)
    return ok(pattern=pattern, limit=limit, count=len(fonts), fonts=fonts)


@_tool
def ass_match_font(family: str, bold: bool = False, italic: bool = False) -> dict[str, Any]:
    """What fontconfig actually resolves for ``family`` (with bold/italic).

    The first candidate is the face that would be used; when its family differs
    from the request the subtitle will be rendered in a **substitute**.

    Returns ``{requested, bold, italic, resolved, file, style, substituted,
    match, candidates}``.
    """
    if not str(family or "").strip():
        raise ToolError("family must be a non-empty font family name")
    matches = M.fonts_match(str(family), bold=bool(bold), italic=bool(italic), limit=5)
    if not matches:
        return ok(
            requested=str(family), bold=bool(bold), italic=bool(italic),
            resolved=None, file=None, style=None, substituted=True,
            match=None, candidates=[],
        )
    first = matches[0]
    return ok(
        requested=str(family),
        bold=bool(bold),
        italic=bool(italic),
        resolved=first["family"],
        file=first["file"],
        style=first["style"],
        substituted=bool(first["substituted"]),
        match=first,
        candidates=matches,
    )


@_tool
def ass_fonts_with_char(char: str) -> dict[str, Any]:
    """Which installed family names contain the glyph for *char*.

    ``char`` must be exactly one character.  Returns ``{char, codepoint,
    codepoint_hex, count, families}`` (families sorted case-insensitively).
    """
    if not isinstance(char, str) or len(char) != 1:
        raise ToolError(f"char must be a single character, got {char!r}")
    families = M.fonts_with_char(char)
    return ok(
        char=char,
        codepoint=ord(char),
        codepoint_hex=f"U+{ord(char):04X}",
        count=len(families),
        families=families,
    )


@_tool
def ass_font_coverage(
    text: str,
    family: str | None = None,
    bold: bool = False,
    italic: bool = False,
) -> dict[str, Any]:
    """Characters of *text* missing from *family* (default: the system sans).

    The family is resolved through fontconfig first, so a substituted request is
    reported honestly.  Missing characters come back with their codepoints, plus
    up to a few installed families that do cover them.

    Returns ``{text, requested, default_used, bold, italic, resolved_family,
    substituted, missing: [{char, codepoint, codepoint_hex, fallbacks}],
    missing_count, missing_chars, missing_codepoints, covered, checked,
    coverage_source, font}``.  ``requested`` is the family the check actually ran
    against, so it is never ``None``: when ``family`` is omitted it is the system
    default sans family and ``default_used`` is ``True``.
    """
    requested = str(family).strip() if family else None
    default_used = not requested
    resolved = requested or M.default_sans_family()
    requested = resolved
    substituted = False
    if requested:
        matches = M.fonts_match(requested, bold=bool(bold), italic=bool(italic), limit=1)
        if matches:
            resolved = matches[0]["family"]
            substituted = bool(matches[0]["substituted"])
    result = M.glyph_check(str(text), resolved)
    missing = [
        {
            "char": char,
            "codepoint": ord(char),
            "codepoint_hex": f"U+{ord(char):04X}",
            "fallbacks": result["fallbacks"].get(char, []),
        }
        for char in result["missing"]
    ]
    return ok(
        text=str(text),
        requested=requested,
        default_used=default_used,
        bold=bool(bold),
        italic=bool(italic),
        resolved_family=resolved,
        substituted=substituted,
        font=result["font"],
        missing=missing,
        missing_count=len(missing),
        missing_chars=result["missing"],
        missing_codepoints=[ord(c) for c in result["missing"]],
        covered=result["covered"],
        checked=result["checked"],
        coverage_source=result["coverage_source"],
    )


@_tool
def ass_glyph_check(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
    style: str | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Missing glyphs for a line's text or for a style's font.

    The string checked is ``text`` when given, otherwise the **plain text** of
    line ``index`` (override tags removed).  The font comes from ``family`` when
    given, otherwise from the style named by ``style`` (its ``Fontname``,
    ``Bold`` and ``Italic``) -- when neither is given the style of the line is
    used, and failing that the system default sans.

    Returns ``{source, doc_id, index, style, requested, resolved_family,
    substituted, bold, italic, text, missing: [{char, codepoint, codepoint_hex,
    fallbacks}], missing_count, missing_chars, missing_codepoints, checked,
    covered, coverage_source}``.
    """
    doc = workspace.get(doc_id) if (index is not None or (doc_id is not None)) else None
    plain = None
    style_info: dict[str, Any] | None = None

    if text is not None:
        raw = str(text)
        if "{" in raw or re.search(r"\\[a-z]", raw):
            plain = T.plain_text(raw)
        else:
            plain = raw
        source = "text"
    elif index is not None:
        if doc is None:
            raise ToolError("index requires an open document (pass doc_id=)")
        entry = entry_at(doc, int(index))
        plain = T.plain_text(entry.text)
        source = "line"
        style_name = style or entry.get("Style")
        if style_name:
            style_info = _style_font(doc, style_name)
            if style is None:
                style = style_name
    else:
        raise ToolError("pass text= or index= (+ doc_id=)")

    if family:
        requested = str(family).strip()
        bold = italic = False
    elif style is not None:
        style_info = style_info or _style_font(doc, style)
        requested = style_info["family"]
        bold, italic = style_info["bold"], style_info["italic"]
    else:
        requested = None
        bold = italic = False

    resolved = requested or M.default_sans_family()
    substituted = False
    if requested:
        matches = M.fonts_match(requested, bold=bold, italic=italic, limit=1)
        if matches:
            resolved = matches[0]["family"]
            substituted = bool(matches[0]["substituted"])

    result = M.glyph_check(plain or "", resolved)
    missing = [
        {
            "char": char,
            "codepoint": ord(char),
            "codepoint_hex": f"U+{ord(char):04X}",
            "fallbacks": result["fallbacks"].get(char, []),
        }
        for char in result["missing"]
    ]
    return ok(
        source=source,
        doc_id=workspace.doc_id_for(doc) if doc is not None else doc_id,
        index=index,
        style=style,
        style_info=style_info,
        requested=requested,
        resolved_family=resolved,
        substituted=substituted,
        bold=bool(bold),
        italic=bool(italic),
        text=plain or "",
        missing=missing,
        missing_count=len(missing),
        missing_chars=result["missing"],
        missing_codepoints=[ord(c) for c in result["missing"]],
        checked=result["checked"],
        covered=result["covered"],
        coverage_source=result["coverage_source"],
    )


_FONTNAME_RE = re.compile(r"^\s*fontname\s*:\s*(?P<name>.+?)\s*$", re.IGNORECASE)


def _resolve_family(family: str) -> dict[str, Any]:
    """Resolve one family through fontconfig, flagging substitution."""
    name = str(family or "").strip()
    if not name:
        return {"family": name, "installed": False, "substituted": True,
                "resolved": None, "file": None, "style": None, "error": "empty family name"}
    matches = M.fonts_match(name, limit=1)
    if not matches:
        return {"family": name, "installed": False, "substituted": True,
                "resolved": None, "file": None, "style": None}
    first = matches[0]
    installed = first["family"].casefold() == name.casefold()
    return {
        "family": name,
        "installed": installed,
        "substituted": not installed,
        "resolved": first["family"],
        "file": first["file"],
        "style": first["style"],
    }


@_tool
def ass_fonts_used(doc_id: str | None = None) -> dict[str, Any]:
    """Every font the script asks for, plus embedded font attachments.

    Styles contribute their ``Fontname`` (with the ``Bold``/``Italic`` flags
    that affect which face fontconfig resolves); the ``[Fonts]`` section
    contributes each ``fontname:`` attachment.  Every family is resolved through
    fontconfig so **families that are not installed** and **families that
    resolve to a substitute** are flagged explicitly.

    Returns ``{doc_id, families: [{family, installed, substituted, resolved,
    file, style, bold, italic, styles: [names], attachment}], attachments:
    [names], not_installed: [names], substituted: [names], used_count}``.
    """
    doc = workspace.get(doc_id)
    families: dict[str, dict[str, Any]] = {}
    for style in doc.styles():
        name = (style.get("Fontname") or "").strip()
        if not name:
            continue
        key = name.casefold()
        entry = families.setdefault(key, {
            "family": name,
            "styles": [],
            "bold": False,
            "italic": False,
            "attachment": False,
        })
        entry["styles"].append(style.name)
        if str(style.get("Bold", "0")).strip() not in ("", "0"):
            entry["bold"] = True
        if str(style.get("Italic", "0")).strip() not in ("", "0"):
            entry["italic"] = True

    attachments: list[str] = []
    for raw in doc.attachments("fonts"):
        match = _FONTNAME_RE.match(raw.splitlines()[0] if raw else "")
        if not match:
            continue
        filename = match.group("name")
        attachments.append(filename)
        stem = Path(filename).stem
        key = stem.casefold()
        entry = families.setdefault(key, {
            "family": stem,
            "styles": [],
            "bold": False,
            "italic": False,
            "attachment": True,
        })
        entry["attachment"] = True
        entry.setdefault("attachment_file", filename)

    resolved: list[dict[str, Any]] = []
    for entry in families.values():
        embedded = bool(entry["attachment"])
        info = _resolve_family(entry["family"])
        info.update({
            "styles": entry["styles"],
            "bold": entry["bold"],
            "italic": entry["italic"],
            "attachment": embedded,
            "embedded": embedded,
        })
        # An embedded font ships inside the .ass file, so fontconfig substituting
        # it is not a rendering problem: only report substitution for styles.
        if embedded:
            info["substituted"] = False
        if "attachment_file" in entry:
            info["attachment_file"] = entry["attachment_file"]
        resolved.append(info)
    resolved.sort(key=lambda item: item["family"].casefold())

    return ok(
        doc_id=workspace.doc_id_for(doc),
        families=resolved,
        attachments=attachments,
        not_installed=[f["family"] for f in resolved if not f["installed"]],
        not_installed_style_families=[
            f["family"] for f in resolved if not f["installed"] and not f["embedded"]
        ],
        substituted=[f["family"] for f in resolved if f.get("substituted")],
        used_count=len(resolved),
    )


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def register(mcp, ws):  # noqa: ARG001 - ws kept for the project-wide contract
    """Register every ``ass_*`` tool with *mcp* and return their sorted names."""
    names: list[str] = []
    for name, value in list(globals().items()):
        if name.startswith("ass_") and callable(value):
            mcp.tool()(value)
            names.append(name)
    return sorted(names)
