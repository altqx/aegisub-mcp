"""Rendered-ink measurement and font probing for ASS scripts.

Two capabilities live here:

**Measurement.**  :func:`ink_bbox` turns a raw 8-bit grayscale raster (the exact
buffer :func:`aegisub_mcp.asscore.render.render_gray` returns) into the bounding
box of every pixel that is not background.  :func:`measure_line_render` and
:func:`measure_render` wrap that with libass rendering so a caller can ask
"how big is this line, in script coordinates?" and get numbers back.

**Font probing.**  :func:`fonts_list`, :func:`fonts_match`,
:func:`fonts_with_char` and :func:`glyph_check` shell out to ``fc-list`` and
``fc-match`` so the caller can tell which font a style will actually resolve to
and which characters are about to be rendered by a fallback face.

Everything is standard library plus ``subprocess`` -- no Pillow, no fontTools,
no numpy.

Measurement caveats (read before trusting a number)
---------------------------------------------------
* **Ink box != font metric extents.**  The box describes *painted pixels*, not
  the font's ascent/descent/advance box.  A line of "ao" has no ink in the
  descender region, and a line of "TY" has none below the baseline, even though
  both advance the same line height.  Layout metrics (line height, baseline
  offset, ``\\k`` timing geometry) cannot be derived from an ink box.
* **Outline, shadow and blur count as ink** unless the style disables them
  (``Outline=0``), because libass rasterises them too.  With the default
  ``Outline=2`` the box is up to 2 px larger on each side than the glyph
  outline.
* **Antialiasing**: pixels are compared against ``threshold`` (default 16).  A
  raised threshold trims the faint edge of the antialiased border and shrinks
  the box; a threshold of 1 includes nearly everything.  Compare boxes only
  when rendered with the same threshold.
* **Clipping**: text positioned at or beyond a frame edge is clipped by libass,
  and the measured box is then clipped too.  ``measure_line_render`` positions
  the line at ``\\pos(0,0)`` with ``\\an7`` (top-left anchor) by default, which
  is the tightest possible placement; pass ``extra_tags`` with a small offset
  (for example ``{\\an7\\pos(4,4)}``) and subtract the offset if a style might
  paint outside the top/left edge.
* **Ink vs. duration**: ``ink_bbox`` measures one raster.  Fades, ``\\move``,
  ``\\t`` and animation mean a single frame is not the whole story -- sample
  several timestamps with :func:`measure_render`.

Glyph-check caveat
------------------
:func:`glyph_check` approximates Aegisub's own "characters not in font" check.
Aegisub asks the *loaded* face (harfbuzz/freetype cmap, including fontconfig
substitution and its own fallback chain, plus VSFilter-style quirks) whether a
codepoint is present, and only inspects characters that actually appear in the
line's text after override-tag parsing.  This module instead intersects the
fontconfig ``charset`` property of the family name, which is a good but not
identical approximation: it ignores the font's OpenType feature coverage,
ignores per-character shaping substitutions, knows nothing about which style
Aegisub would actually select, and reports families rather than files.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import render as _render
from .document import EVENT_FORMAT_ASS, STYLE_FORMAT_V4P, canonical_style_key, default_style_spec

__all__ = [
    "MeasureError",
    "ink_bbox",
    "measure_line_render",
    "measure_render",
    "fonts_list",
    "fonts_match",
    "fonts_with_char",
    "glyph_check",
    "reading_speed",
    "DEFAULT_TAGS",
]


class MeasureError(RuntimeError):
    """Raised when a measurement cannot be produced or a probe fails."""


#: Default override pair used by :func:`measure_line_render`: top-left anchor at
#: the origin, so the measured ink box is directly interpretable as an offset
#: from ``(0, 0)``.
DEFAULT_TAGS = r"{\an7\pos(0,0)}"

#: 256-entry translation table cache keyed by threshold.
_TABLE_CACHE: Dict[int, bytes] = {}


def _threshold_table(threshold: int) -> bytes:
    table = _TABLE_CACHE.get(threshold)
    if table is None:
        if threshold <= 0:
            table = bytes([1] * 256)
        else:
            table = bytes(1 if value >= threshold else 0 for value in range(256))
        _TABLE_CACHE[threshold] = table
    return table


def ink_bbox(
    gray: "bytes | bytearray",
    width: int,
    height: int,
    threshold: int = 16,
) -> Optional[Dict[str, Any]]:
    """Bounding box of every pixel at or above ``threshold``.

    Args:
        gray: row-major 8-bit grayscale raster, ``width * height`` bytes (the
            layout produced by ``-pix_fmt gray -f rawvideo``).  Extra trailing
            bytes are ignored; a short buffer raises :class:`ValueError`.
        width, height: raster dimensions.  Non-positive values return ``None``.
        threshold: minimum pixel value (0-255) that counts as ink.  The default
            of 16 ignores the black/near-black background and antialiasing
            noise; 1 counts anything not exactly black.

    Returns:
        ``None`` when no pixel reaches the threshold, otherwise a dict with
        ``x0``, ``y0``, ``x1``, ``y1`` (``x1``/``y1`` are **exclusive**),
        ``width``/``height`` (the size of the box, i.e. ``x1 - x0`` /
        ``y1 - y0``), ``ink_pixels``, ``coverage`` (ink pixels divided by the
        raster area) and, for reference, ``image_width``/``image_height`` and
        ``threshold``.
    """
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        return None
    needed = w * h
    if len(gray) < needed:
        raise ValueError(
            f"grayscale buffer too short: need {needed} bytes for {w}x{h}, got {len(gray)}"
        )

    table = _threshold_table(int(threshold))
    x0 = w
    x1 = -1
    y0 = -1
    y1 = -1
    ink = 0
    for y in range(h):
        row = bytes(gray[y * w : (y + 1) * w]).translate(table)
        count = row.count(1)
        if not count:
            continue
        if y0 < 0:
            y0 = y
        y1 = y
        ink += count
        first = row.find(1)
        last = w - 1 - row[::-1].find(1)
        if first < x0:
            x0 = first
        if last > x1:
            x1 = last

    if y0 < 0:
        return None
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1 + 1,
        "y1": y1 + 1,
        "width": x1 + 1 - x0,
        "height": y1 + 1 - y0,
        "ink_pixels": ink,
        "coverage": ink / float(needed),
        "image_width": w,
        "image_height": h,
        "threshold": int(threshold),
    }


def _coerce_style(style: Optional[Dict[str, Any]], name: str = "Default") -> Dict[str, Any]:
    """Merge a caller style dict over the toolkit's default V4+ style spec."""
    spec = default_style_spec(name, Alignment=7, MarginL=0, MarginR=0, MarginV=0, Shadow=0)
    for key, value in (style or {}).items():
        canonical = canonical_style_key(str(key))
        spec[canonical] = value
    return spec


def _style_line(spec: Dict[str, Any]) -> str:
    fields: List[str] = []
    for field_name in STYLE_FORMAT_V4P:
        value = spec.get(field_name)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        fields.append("" if value is None else str(value))
    return "Style: " + ",".join(fields)


def _escape_text(text: str) -> str:
    """Turn real line breaks into ASS ``\\N`` and drop stray carriage returns."""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\N")


def build_line_ass(
    line_text: str,
    style: Optional[Dict[str, Any]] = None,
    play_res: Sequence[int] = (1920, 1080),
    extra_tags: Optional[str] = None,
) -> str:
    """Build a minimal single-line ASS document (helper used for measurement).

    The document has one ``[V4+ Styles]`` entry and one ``Dialogue`` covering
    ``0:00:00.00`` to ``0:00:10.00`` whose text is ``extra_tags`` followed by
    ``line_text``.  ``WrapStyle: 2`` disables automatic word wrapping so the
    measured box really is one line.
    """
    x, y = int(play_res[0]), int(play_res[1])
    if x <= 0 or y <= 0:
        raise MeasureError(f"play_res must be positive, got {tuple(play_res)!r}")
    spec = _coerce_style(style)
    tags = DEFAULT_TAGS if extra_tags is None else str(extra_tags)
    lines = [
        "[Script Info]",
        "Title: aegisub-mcp measure_line_render",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        f"PlayResX: {x}",
        f"PlayResY: {y}",
        "",
        "[V4+ Styles]",
        "Format: " + ", ".join(STYLE_FORMAT_V4P),
        _style_line(spec),
        "",
        "[Events]",
        "Format: " + ", ".join(EVENT_FORMAT_ASS),
        f"Dialogue: 0,0:00:00.00,0:00:10.00,{spec['Name']},,0,0,0,,{tags}{_escape_text(line_text)}",
        "",
    ]
    return "\n".join(lines)


def measure_line_render(
    line_text: str,
    style: Optional[Dict[str, Any]] = None,
    play_res: Sequence[int] = (1920, 1080),
    extra_tags: Optional[str] = None,
    threshold: int = 16,
    timeout: float = 60,
) -> Dict[str, Any]:
    """Render one line and measure the ink it leaves on a black frame.

    The line is rendered at exactly ``play_res``, so **one pixel is one script
    unit** and the returned box can be used directly as script coordinates.
    The default ``extra_tags`` is :data:`DEFAULT_TAGS`
    (``{\\an7\\pos(0,0)}``), which anchors the line's top-left corner at the
    frame origin.

    Args:
        line_text: the text of the line (raw text; newlines become ``\\N``).
        style: optional V4+ style fields; unknown spellings are normalised with
            the document module's ``canonical_style_key``.  Defaults come from
            ``default_style_spec`` with ``Alignment=7``, zero margins and no
            shadow.
        play_res: ``(PlayResX, PlayResY)`` for the generated document *and* the
            render size, i.e. the coordinate space of the result.
        extra_tags: override-tag block placed before the text, as a plain
            string.  Pass ``""`` for no tags at all (then the style alignment
            and margins decide the position).
        threshold: ink threshold, see :func:`ink_bbox`.
        timeout: ffmpeg timeout in seconds.

    Returns:
        dict with ``text``, ``style`` (the effective style dict),
        ``play_res``, ``render_width``/``render_height``, ``scale`` (script
        units per pixel, always 1.0 here), ``bbox`` (raw :func:`ink_bbox`
        result or ``None``), ``rect`` (the same box in **script coordinates**
        as ``{x, y, x1, y1, width, height}``, or ``None``), ``empty``,
        ``ink_pixels``, ``coverage``, ``ass_text`` and ``ffmpeg_cmd``.
    """
    x, y = int(play_res[0]), int(play_res[1])
    ass_text = build_line_ass(line_text, style, (x, y), extra_tags)
    # t=1s: inside the generated Dialogue (0s..10s) and far from any edge case.
    gray, gw, gh = _render.render_gray(ass_text, 1000, x, y, timeout=timeout)
    bbox = ink_bbox(gray, gw, gh, threshold)
    return _measurement_result(
        bbox=bbox,
        play_res=(x, y),
        render_size=(gw, gh),
        time_ms=1000.0,
        extra={"text": str(line_text), "style": _coerce_style(style), "ass_text": ass_text},
    )


def measure_render(
    ass: "str | os.PathLike[str]",
    time_ms: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
    threshold: int = 16,
    **kw: Any,
) -> Dict[str, Any]:
    """Measure the ink of a full ASS document rendered at ``time_ms``.

    Useful to verify a typesetting result numerically: render, then compare the
    returned ``rect`` (script coordinates) against the intended geometry.  A
    single frame is measured, so animated or fading lines should be sampled at
    several timestamps.

    ``width``/``height`` default to the document's own PlayRes (1:1 script
    coordinates).  When an explicit size is given the frame is rendered at
    PlayRes and rescaled (see :mod:`aegisub_mcp.asscore.render`), and the
    returned ``scale`` converts pixels back to script units.

    Returns:
        dict with ``time_ms``, ``play_res``, ``render_width``,
        ``render_height``, ``scale`` (``[sx, sy]``), ``bbox`` (pixels),
        ``rect`` (script coordinates), ``empty``, ``ink_pixels`` and
        ``coverage``.
    """
    with _render._ass_source(ass) as (_path, text):
        play = _render._effective_play_res(text, None) or (_render.DEFAULT_WIDTH, _render.DEFAULT_HEIGHT)
    rw = int(width) if width else play[0]
    rh = int(height) if height else play[1]
    gray, gw, gh = _render.render_gray(ass, time_ms, rw, rh, **kw)
    bbox = ink_bbox(gray, gw, gh, threshold)
    return _measurement_result(
        bbox=bbox,
        play_res=play,
        render_size=(gw, gh),
        time_ms=float(time_ms),
    )


def _measurement_result(
    bbox: Optional[Dict[str, Any]],
    play_res: Tuple[int, int],
    render_size: Tuple[int, int],
    time_ms: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sx = play_res[0] / float(render_size[0])
    sy = play_res[1] / float(render_size[1])
    result: Dict[str, Any] = {
        "time_ms": time_ms,
        "play_res": [play_res[0], play_res[1]],
        "render_width": render_size[0],
        "render_height": render_size[1],
        "scale": [sx, sy],
        "bbox": bbox,
        "rect": None,
        "empty": bbox is None,
        "ink_pixels": 0,
        "coverage": 0.0,
    }
    if bbox is not None:
        result["rect"] = {
            "x": bbox["x0"] * sx,
            "y": bbox["y0"] * sy,
            "x1": bbox["x1"] * sx,
            "y1": bbox["y1"] * sy,
            "width": bbox["width"] * sx,
            "height": bbox["height"] * sy,
        }
        result["ink_pixels"] = bbox["ink_pixels"]
        result["coverage"] = bbox["coverage"]
    if extra:
        result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Font probing (fontconfig command line tools)
# ---------------------------------------------------------------------------
_FONT_FIELDS = "%{file}\t%{index}\t%{family}\t%{style}\n"


def _fc_binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MeasureError(
            f"{name} not found on PATH; font probing requires fontconfig's {name} "
            "(install the fontconfig tools package)"
        )
    return found


def _run_fc(cmd: Sequence[str], timeout: float = 60) -> str:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MeasureError(f"cannot run {cmd[0]!r}: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise MeasureError(f"{cmd[0]} failed, exit {proc.returncode}: {err}")
    return (proc.stdout or b"").decode("utf-8", "replace")


def _primary_family(raw: str) -> str:
    """First name out of a comma-separated fontconfig family list."""
    return (raw or "").split(",")[0].strip()


def _split_font_line(line: str) -> Optional[Dict[str, Any]]:
    parts = line.split("\t")
    if len(parts) < 4:
        return None
    path, index, family_raw, style = parts[0], parts[1], parts[2], parts[3]
    try:
        index_value: Optional[int] = int(index)
    except (TypeError, ValueError):
        index_value = None
    return {
        "family": _primary_family(family_raw),
        "style": style.strip(),
        "file": path,
        "index": index_value,
        "family_raw": family_raw,
    }


def fonts_list(pattern: Optional[str] = None, limit: int = 2000) -> List[Dict[str, Any]]:
    """List installed fonts via ``fc-list``.

    Args:
        pattern: case-insensitive substring filter applied to the family, style
            and file of every entry (this is a plain substring match, *not* a
            fontconfig pattern expression).  ``None`` returns everything.
        limit: maximum number of entries to return.

    Returns:
        List of dicts with ``family`` (primary family name), ``style``, ``file``,
        ``index`` (face index inside a collection, or ``None``) and
        ``family_raw`` (the full, possibly localised, comma-separated family
        property).  Sorted by family, style and file.
    """
    out = _run_fc([_fc_binary("fc-list"), "--format", _FONT_FIELDS])
    entries: List[Dict[str, Any]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parsed = _split_font_line(line)
        if parsed:
            entries.append(parsed)
    if pattern:
        needle = str(pattern).casefold()
        entries = [
            e
            for e in entries
            if needle in e["family"].casefold()
            or needle in e["style"].casefold()
            or needle in e["file"].casefold()
        ]
    entries.sort(key=lambda e: (e["family"].casefold(), e["style"].casefold(), e["file"]))
    return entries[: max(0, int(limit))]


def fonts_match(
    family: str,
    bold: bool = False,
    italic: bool = False,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Resolve a style's font request through fontconfig.

    Thin wrapper around ``fc-match -s``: the first entry is the face fontconfig
    would actually use.  Comparing its ``family`` with the requested ``family``
    is how a caller detects **font substitution** -- when they differ, the
    subtitle will not render in the font the style asks for.

    Args:
        family: requested family name.  ``:``, ``,`` and backslashes are
            stripped because they would be parsed as fontconfig pattern syntax.
        bold, italic: appended to the query as ``:bold`` / ``:italic``.
        limit: number of ranked candidates to return.

    Returns:
        List of dicts with ``family``, ``style``, ``file``, ``index``,
        ``family_raw``, ``requested`` (the query as issued) and ``substituted``
        (True when the returned family differs from the request,
        case-insensitively).
    """
    clean = re.sub(r"[:,\\\\]", " ", str(family)).strip()
    if not clean:
        raise MeasureError("fonts_match requires a non-empty family name")
    query = clean
    if bold:
        query += ":bold"
    if italic:
        query += ":italic"
    cmd = [_fc_binary("fc-match"), "-s", "--format", _FONT_FIELDS, query]
    results: List[Dict[str, Any]] = []
    for line in _run_fc(cmd).splitlines():
        if not line.strip():
            continue
        parsed = _split_font_line(line)
        if not parsed:
            continue
        parsed["requested"] = str(family)
        parsed["substituted"] = parsed["family"].casefold() != str(family).casefold()
        results.append(parsed)
    return results[: max(0, int(limit))]


def _charset_token(char: str) -> str:
    if not isinstance(char, str) or len(char) != 1:
        raise MeasureError(f"expected a single character, got {char!r}")
    return f"{ord(char):04X}"


def fonts_with_char(char: str) -> List[str]:
    """Family names of installed fonts whose charset covers ``char``.

    Uses ``fc-list ':charset=XXXX'`` -- the four-hex-digit ``U+0E01`` style
    token, verified to work on this host's fontconfig (``:charset=0E01``).

    Note:
        The result is built from the ``family`` property, so a cover provided by
        a face buried in a family with a different primary name is reported under
        that primary name only.
    """
    token = _charset_token(char)
    out = _run_fc([_fc_binary("fc-list"), "--format", "%{family}\n", f":charset={token}"])
    families = {_primary_family(line) for line in out.splitlines() if line.strip()}
    families.discard("")
    return sorted(families, key=str.casefold)


def _charset_ranges(family: str) -> Optional[List[Tuple[int, int]]]:
    """Parsed ``charset`` property of every face of ``family``.

    Returns ``None`` when fontconfig reports no charset data at all (in which
    case callers should fall back to per-character queries).
    """
    clean = re.sub(r"[:,\\\\]", " ", str(family)).strip()
    if not clean:
        return None
    out = _run_fc([_fc_binary("fc-list"), "--format", "%{charset}\n", f":family={clean}"])
    ranges: List[Tuple[int, int]] = []
    for line in out.splitlines():
        for token in line.split():
            lo_s, _, hi_s = token.partition("-")
            try:
                lo = int(lo_s, 16)
                hi = int(hi_s, 16) if hi_s else lo
            except ValueError:
                continue
            ranges.append((lo, hi))
    if not ranges:
        return None
    ranges.sort()
    merged: List[Tuple[int, int]] = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _in_ranges(ranges: Sequence[Tuple[int, int]], codepoint: int) -> bool:
    for lo, hi in ranges:
        if lo <= codepoint <= hi:
            return True
        if lo > codepoint:
            break
    return False


def default_sans_family() -> str:
    """Family name fontconfig returns for the generic ``sans-serif`` request."""
    out = _run_fc([_fc_binary("fc-match"), "--format", "%{family}\n", "sans-serif"])
    first = _primary_family((out or "").splitlines()[0] if out.strip() else "")
    return first or "DejaVu Sans"


def glyph_check(
    text: str,
    font_family: Optional[str] = None,
    limit_fallbacks: int = 6,
) -> Dict[str, Any]:
    """Approximate Aegisub's glyph coverage check for ``text``.

    For every distinct non-space character, asks whether ``font_family`` (or the
    system default sans-serif face when ``None``) covers it, using the
    fontconfig ``charset`` property.  Characters that are not covered are
    reported in ``missing`` together with up to ``limit_fallbacks`` installed
    families that *do* cover them, which is exactly the information needed to
    decide whether to change the style's font or accept a fallback.

    This is an approximation of Aegisub's own glyph check -- see the module
    docstring for the precise differences.

    Args:
        text: the text to inspect (override tags, if any, are not parsed out).
        font_family: family to test; ``None`` selects the default sans-serif.
        limit_fallbacks: maximum fallback families reported per character.

    Returns:
        dict with ``font`` (the family actually tested), ``missing`` (list of
        characters, in first-appearance order), ``fallbacks`` (mapping each
        missing character to a list of covering families), ``checked`` (number
        of distinct non-space characters inspected), ``covered`` (the distinct
        characters that were covered) and ``coverage_source`` -- ``"charset"``
        when the font's charset property was usable, ``"query"`` when each
        character had to be probed individually.
    """
    family = str(font_family).strip() if font_family else default_sans_family()
    ranges = _charset_ranges(family)
    source = "charset" if ranges is not None else "query"

    missing: List[str] = []
    fallbacks: Dict[str, List[str]] = {}
    covered: List[str] = []
    checked = 0
    for char in dict.fromkeys(str(text)):
        if char.isspace():
            continue
        checked += 1
        if ranges is not None:
            is_covered = _in_ranges(ranges, ord(char))
        else:
            is_covered = bool(_family_covers(family, char))
        if is_covered:
            covered.append(char)
            continue
        missing.append(char)
        fallbacks[char] = fonts_with_char(char)[: max(0, int(limit_fallbacks))]

    return {
        "font": family,
        "missing": missing,
        "fallbacks": fallbacks,
        "covered": covered,
        "checked": checked,
        "coverage_source": source,
    }


def _family_covers(family: str, char: str) -> List[str]:
    """Families matching both ``family`` and the character's charset."""
    clean = re.sub(r"[:,\\\\]", " ", str(family)).strip()
    out = _run_fc(
        [
            _fc_binary("fc-list"),
            "--format",
            "%{family}\n",
            f":family={clean}:charset={_charset_token(char)}",
        ]
    )
    return sorted({_primary_family(line) for line in out.splitlines() if line.strip()})


def reading_speed(text_chars: int, duration_ms: int) -> float:
    """Characters per second for a line.

    The classic fansub sanity check: roughly 17-21 cps is comfortable for Latin
    script, and above ~25 cps a line is hard to read.

    Args:
        text_chars: number of characters in the line.
        duration_ms: display duration in milliseconds.

    Returns:
        ``text_chars / (duration_ms / 1000)``, or ``math.inf`` when the
        duration is zero/negative (an instantaneous line has unbounded speed).
    """
    duration = float(duration_ms)
    if duration <= 0:
        return math.inf
    return float(text_chars) / (duration / 1000.0)
