"""Font metrics for Aegisub-compatible text extents.

Aegisub's ``aegisub.text_extents(style, text)`` returns the *layout* extents of
a string as rendered with a style::

    width, height, descent, ext_lead

``width`` is the advance width of the widest visual line, ``height`` is the
line height of the block (ascent + descent per line, plus the line gap between
lines), ``descent`` is the distance below the baseline and ``ext_lead`` is the
external leading (the font's line gap).  Aegisub gets these numbers from the
same font tables a shaping engine uses, so we read them straight out of the
font file instead of guessing from rasterised ink.

Semantics implemented here
--------------------------

* Vertical metrics follow **HarfBuzz** (``hb_font_get_h_extents``), which is
  what Pango/wxGTK — and therefore Aegisub on Linux — and libass/FreeType all
  end up using.  For each of ascender/descender/line gap HarfBuzz picks the
  OS/2 ``sTypo*`` value only when ``fsSelection`` bit 7 (``USE_TYPO_METRICS``)
  is set, and otherwise the ``hhea`` value::

      (use_typo_metrics() && OS2.sTypoX) || hhea.x

  ``usWinAscent``/``usWinDescent`` are *not* used for layout (HarfBuzz never
  consults them); they are only a last-resort fallback for fonts whose ``hhea``
  metrics are degenerate.  See :func:`select_vertical_metrics`.
* Horizontal advance is the ``hmtx`` advance of the glyph the character maps
  to through the best ``cmap`` subtable, summed per character and scaled by
  ``size * scale_x / units_per_em``.
* Inter-character kerning comes from the legacy ``kern`` table **and only when
  the style's ``Spacing`` is zero** — Aegisub does the same ("If there's
  inter-character spacing, kerning info must not be used").
* ``Spacing`` is added once per character, *including the last one*, matching
  Aegisub's ``width += (a + spacing) * scaling`` loop.
* ``width`` is scaled by ``ScaleX``, ``height``/``descent``/``ext_lead`` by
  ``ScaleY`` — again matching Aegisub's compensation block.

Argument order
--------------

Aegisub documents ``aegisub.text_extents(style, text)`` — style first.  This
module's internal convention is ``(text, style)``, but :func:`text_extents`,
:func:`text_extents_lua` and friends accept **both orders** and disambiguate on
type, because a dict is never a string.  Anything that is not a
``(str|bytes, Mapping)`` pair raises :class:`FontMetricsError` rather than
leaking an ``AttributeError``.

Deviations from Aegisub's own fontconfig/HarfBuzz measurement
-------------------------------------------------------------

These are deliberate and are the numbers callers should expect to differ by:

* **Hinting / rasterisation.**  No rasteriser is involved: no hinting, no
  subpixel positioning, no device-pixel rounding.  Aegisub measures through
  ``wxDC::GetTextExtent``, i.e. real hinted layout.  Our widths are exact
  rational font units, so they can differ from Aegisub by fractions of a pixel.
* **Aegisub's height is normalised to the font size.**  Aegisub's non-Windows
  path divides every measurement by the *measured line height*
  (``scaling = fontsize / lheight`` with ``fontsize = style->fontsize * 64``),
  which makes its reported ``height`` equal ``style.fontsize * ScaleY / 100``
  regardless of the font's real ascent/descent, and makes its ``width``
  proportional to ``sum(advances) / (ascent + descent)`` rather than to
  ``sum(advances) / units_per_em``.  We instead report the font's true metrics
  scaled to the requested size.  Matching the quirk would make ``height``
  useless, so it is not done; for DejaVu Sans the two widths differ by roughly
  ``units_per_em / (ascent + descent)`` (~0.86x).
* **Kerning source.**  Only the legacy ``kern`` table is read.  GPOS ``kern``
  feature lookups (what modern fonts such as Noto use, and what HarfBuzz
  actually applies) are not parsed, so for fonts with GPOS-only kerning our
  width is slightly wide.  The ``kern`` subtable format 2/3 class-based pairs
  fontTools exposes as an expanded pair map are read; ``GPOS`` is not.
* **Shaping.**  Advances are summed per character.  Complex scripts needing
  contextual shaping (Arabic, Indic) or mark positioning/ligatures and
  combining marks (Thai) can differ from libass, because libass runs HarfBuzz
  and we do not.  Ligatures (``liga``) are likewise not applied.
* **Fallback fonts.**  A fontconfig *substitution* for an unknown family is
  honoured (same substitute the renderer picks), but per-character fallback for
  characters the font lacks is not modelled: an unmapped character contributes
  the ``.notdef`` advance (usually non-zero) instead of being borrowed from
  another family.  Use :func:`font_coverage` to detect that case.
* **Embedded attachments.**  Aegisub can use a font embedded in an ``[Fonts]``
  attachment; this module only reads fonts already on disk.  Pass an explicit
  ``font_file`` in the style dict to point at an extracted attachment.
* **Newlines and tags.**  Aegisub documents ``text`` as plain single-line text
  and treats formatting codes as verbatim characters.  :func:`text_extents`
  interprets them by default (tags stripped, ``\\N``/``\\n``/``\\h`` resolved,
  see :func:`visible_text`) and reports the widest visual line, which is a
  superset; pass ``raw=True`` for Aegisub's literal-text behaviour.
* **Collections.**  Only face 0 of a ``.ttc`` collection is read.

Callers that need rasterised truth should use ``aegisub_mcp.asscore.measure``
instead.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import tags as T

__all__ = [
    "DEFAULT_FAMILY",
    "DEFAULT_FONT_SIZE",
    "Extents",
    "FontMetrics",
    "FontMetricsError",
    "USE_TYPO_METRICS",
    "char_advances",
    "coerce_text_style",
    "extents_report",
    "font_coverage",
    "fontconfig_available",
    "fonts_with_char",
    "list_fonts",
    "load_metrics",
    "match_font",
    "metrics_for_style",
    "next_line_width",
    "select_vertical_metrics",
    "text_extents",
    "text_extents_lua",
    "visible_text",
]


class FontMetricsError(RuntimeError):
    """Raised when a font cannot be resolved or read, or when arguments are bad."""


DEFAULT_FONT_SIZE = 48.0
"""Font size used when a style carries no usable ``fontsize`` (Aegisub's default)."""

DEFAULT_FAMILY = "Arial"
"""Family used when a style carries no usable ``fontname`` (Aegisub's default)."""

# OS/2 fsSelection bits we care about.
USE_TYPO_METRICS = 0x80
FS_ITALIC = 0x01
FS_BOLD = 0x20

_HARD_SPACE = "\u00a0"

# --------------------------------------------------------------------------
# argument, style-key and number handling
# --------------------------------------------------------------------------

#: Style spellings accepted for each field.  Aegisub's Lua style table uses the
#: lowercase no-underscore spellings (``fontname``, ``fontsize``, ``scalex``,
#: ``scaley``); the ASS storage format and this package's own helpers use the
#: other spellings.  Both are accepted so a style table can be fed straight in.
_FONT_KEYS = ("font", "Fontname", "fontname", "font_name", "FontName")
_SIZE_KEYS = ("font_size", "fontsize", "Fontsize", "FontSize", "size")
_SCALE_X_KEYS = ("scale_x", "scalex", "ScaleX")
_SCALE_Y_KEYS = ("scale_y", "scaley", "ScaleY")
_SPACING_KEYS = ("spacing", "Spacing")
_BOLD_KEYS = ("bold", "Bold")
_ITALIC_KEYS = ("italic", "Italic")
_FONT_FILE_KEYS = ("font_file", "font_file_path", "fontfile", "fontFile")


def _pick(style: Mapping, keys: tuple[str, ...]) -> Any:
    """First *truthy* value among ``keys``, mirroring ``a or b or default``.

    Zero and the empty string count as absent, exactly as the original
    ``style.get("font_size") or style.get("Fontsize") or 48.0`` chain did.
    """
    for key in keys:
        value = style.get(key)
        if value:
            return value
    return None


def _style_number(style: Mapping, keys: tuple[str, ...], default: float, what: str) -> float:
    """Read a numeric style field, raising a readable error instead of ValueError."""
    value = _pick(style, keys)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FontMetricsError(
            f"style {what} must be a number, got {value!r} ({type(value).__name__})"
        ) from exc


def _as_text(value: Any) -> str:
    """Coerce a text argument to ``str``; Lua strings arrive as UTF-8 bytes."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FontMetricsError(f"text is not valid UTF-8: {exc}") from exc
    raise FontMetricsError(f"text must be a str or UTF-8 bytes, got {type(value).__name__}")


def _as_style(value: Any) -> Mapping:
    """Coerce a style argument to a mapping, with a readable failure mode."""
    if isinstance(value, Mapping):
        return value
    raise FontMetricsError(
        f"style must be a mapping of ASS style fields, got {type(value).__name__}"
    )


def coerce_text_style(a: Any, b: Any) -> tuple[str, Mapping]:
    """Accept ``(text, style)`` or Aegisub's documented ``(style, text)``.

    Returns ``(text, style)``.  The two are unambiguous by type — a style is a
    mapping and text is not — so both orders are supported.  Anything else
    raises :class:`FontMetricsError`.
    """
    if isinstance(a, Mapping) and not isinstance(b, Mapping):
        a, b = b, a
    return _as_text(a), _as_style(b)


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "-1"}
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


# --------------------------------------------------------------------------
# fontconfig
# --------------------------------------------------------------------------


def fontconfig_available() -> bool:
    """True when ``fc-match`` and ``fc-list`` are on PATH."""
    return shutil.which("fc-match") is not None


def _fc_match(family: str, bold: bool = False, italic: bool = False, *, index: int = 0) -> str | None:
    """Resolve a family to a font file with fontconfig; None when unavailable."""
    if not fontconfig_available():
        return None
    pattern = family or "sans-serif"
    if bold and italic:
        pattern += ":bold:italic"
    elif bold:
        pattern += ":bold"
    elif italic:
        pattern += ":italic"
    cmd = ["fc-match", "-f", "%{file}", pattern]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    path = out.stdout.strip().splitlines()[index] if out.stdout.strip() else ""
    if path and Path(path).is_file():
        return path
    return None


def match_font(family: str, bold: bool = False, italic: bool = False) -> dict:
    """What fontconfig resolves for a family.

    Returns a dict with ``requested``, ``file``, ``family``, ``style``,
    ``substituted`` (True when the resolved family differs from the request) and
    ``reason`` when nothing could be resolved.  An unknown family therefore
    yields a *substitution* rather than an error, which is what Aegisub and
    libass do too; :func:`metrics_for_style` only raises when fontconfig is
    missing or returns no file at all.
    """
    result: dict = {"requested": family, "file": None, "family": None, "style": None, "substituted": False}
    if not fontconfig_available():
        result["reason"] = "fontconfig (fc-match) is not installed"
        return result
    pattern = family or "sans-serif"
    if bold:
        pattern += ":bold"
    if italic:
        pattern += ":italic"
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}\t%{family}\t%{style}", pattern],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        result["reason"] = "fc-match failed"
        return result
    first = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if not first:
        result["reason"] = "fontconfig returned no match"
        return result
    parts = first.split("\t")
    result["file"] = parts[0] or None
    result["family"] = parts[1] if len(parts) > 1 else None
    result["style"] = parts[2] if len(parts) > 2 else None
    resolved_families = {f.strip().lower() for f in (result["family"] or "").split(",") if f.strip()}
    result["substituted"] = bool(resolved_families) and family.strip().lower() not in resolved_families
    return result


# --------------------------------------------------------------------------
# font tables
# --------------------------------------------------------------------------


def select_vertical_metrics(head: Any, hhea: Any, os2: Any) -> tuple[int, int, int, bool]:
    """Ascent, descent, line gap and whether OS/2 typo metrics were used.

    Mirrors HarfBuzz' ``hb_font_get_h_extents`` selection exactly::

        (use_typo_metrics() && OS2.sTypoX) || hhea.x

    where ``use_typo_metrics()`` is bit 7 of ``OS/2.fsSelection``.  Note that
    HarfBuzz takes ``sTypoAscender`` even when it is zero if the bit is set
    (the C macro it expands to always reports success), so no zero-check is
    applied inside the typo branch.

    ``usWinAscent``/``usWinDescent`` are *not* consulted for normal fonts.
    They are used only when the selected source leaves ascent and descent both
    zero, which means the font's ``hhea`` table is degenerate; without that net
    a broken font would report a zero-height block.

    Returns ``(ascent, descent, line_gap, used_typo_metrics)`` with ``descent``
    a *positive* distance below the baseline.
    """
    # hhea values; ascent/descent are required in every valid font.
    ascent = int(getattr(hhea, "ascent", 0) or 0)
    descent = abs(int(getattr(hhea, "descent", 0) or 0))
    line_gap = int(getattr(hhea, "lineGap", 0) or 0)

    os2_typo = bool(os2 is not None and int(getattr(os2, "fsSelection", 0) or 0) & USE_TYPO_METRICS)
    if os2_typo:
        ascent = int(getattr(os2, "sTypoAscender", 0) or 0)
        descent = abs(int(getattr(os2, "sTypoDescender", 0) or 0))
        line_gap = int(getattr(os2, "sTypoLineGap", 0) or 0)

    if not ascent and not descent:
        # Degenerate/broken hhea: fall back to the Windows metrics rather than
        # reporting a zero-height block.  Documented deviation from HarfBuzz.
        win_ascent = int(getattr(os2, "usWinAscent", 0) or 0) if os2 is not None else 0
        win_descent = int(getattr(os2, "usWinDescent", 0) or 0) if os2 is not None else 0
        if win_ascent or win_descent:
            ascent, descent = win_ascent, win_descent
            os2_typo = False

    return ascent, descent, line_gap, os2_typo


@dataclass(frozen=True)
class FontMetrics:
    """Static metrics of one font face, in font units (except *_em values)."""

    path: str
    family: str
    subfamily: str
    units_per_em: int
    ascent: int
    descent: int  # positive distance below the baseline
    line_gap: int
    underline_position: int
    underline_thickness: int
    glyph_count: int
    advances: dict[int, int]  # glyph id -> advance in font units
    cmap: dict[int, int]  # codepoint -> glyph id
    kern: dict[tuple[int, int], int]  # (left gid, right gid) -> adjustment
    is_bold: bool
    is_italic: bool
    use_typo_metrics: bool = False
    win_ascent: int = 0
    win_descent: int = 0
    os2_version: int | None = None

    def glyph_for(self, char: str) -> int:
        """Glyph id for a character, 0 (notdef) when unmapped."""
        return self.cmap.get(ord(char), 0)

    def advance_units(self, char: str) -> int:
        """Advance width of one character in font units."""
        return self.advances.get(self.glyph_for(char), 0)

    def units_to_px(self, value: float, size_px: float) -> float:
        """Scale font units to pixels."""
        return value * size_px / self.units_per_em

    @property
    def metrics_source(self) -> str:
        """``"os2-typo"`` or ``"hhea"`` — which table the vertical metrics came from."""
        return "os2-typo" if self.use_typo_metrics else "hhea"


def _name(font, name_id: int) -> str:  # pragma: no cover - trivial wrapper
    try:
        record = font["name"].getDebugName(name_id)
    except Exception:  # noqa: BLE001 - malformed name tables are common
        return ""
    return record or ""


@functools.lru_cache(maxsize=64)
def load_metrics(path: str) -> FontMetrics:
    """Read and cache the metrics of a font file.

    Raises :class:`FontMetricsError` when the file is missing, unreadable or not
    a font, and when ``fontTools`` is not installed.
    """
    try:
        from fontTools.ttLib import TTFont  # optional-but-declared dependency
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise FontMetricsError("fontTools is required for text extents: pip install fonttools") from exc

    font_path = Path(path)
    if not font_path.is_file():
        raise FontMetricsError(f"font file not found: {path}")
    try:
        with TTFont(str(font_path), lazy=True, fontNumber=0) as font:
            head = font["head"]
            hhea = font["hhea"]
            upem = int(getattr(head, "unitsPerEm", 0) or 0) or 1000
            os2 = font.get("OS/2")
            ascent, descent, line_gap, use_typo = select_vertical_metrics(head, hhea, os2)
            win_ascent = int(getattr(os2, "usWinAscent", 0) or 0) if os2 is not None else 0
            win_descent = int(getattr(os2, "usWinDescent", 0) or 0) if os2 is not None else 0
            os2_version = int(getattr(os2, "version", 0)) if os2 is not None else None
            post = font.get("post")
            upos = int(getattr(post, "underlinePosition", 0) or 0) if post is not None else 0
            uthick = int(getattr(post, "underlineThickness", 0) or 0) if post is not None else 0

            cmap: dict[int, int] = {}
            best = font.getBestCmap()
            order = font.getGlyphOrder()
            gid_of = {name: idx for idx, name in enumerate(order)}
            if best:
                for codepoint, glyph_name in best.items():
                    gid = gid_of.get(glyph_name)
                    if gid is not None:
                        cmap[codepoint] = gid

            hmtx = font["hmtx"]
            advances: dict[int, int] = {}
            for gid in range(len(order)):
                try:
                    advance = hmtx[order[gid]][0]
                except (KeyError, IndexError):  # pragma: no cover - broken fonts
                    advance = 0
                advances[gid] = int(advance)

            kern: dict[tuple[int, int], int] = {}
            kern_table = font.get("kern")
            if kern_table is not None:
                for subtable in getattr(kern_table, "kernTables", []) or []:
                    pairs = getattr(subtable, "kernTable", None) or {}
                    try:
                        items = pairs.items()
                    except AttributeError:  # pragma: no cover - unexpected subtable
                        continue
                    for (left, right), value in items:
                        kern[(gid_of.get(left, 0), gid_of.get(right, 0))] = int(value)

            fs_selection = int(getattr(os2, "fsSelection", 0) or 0) if os2 is not None else 0
            is_bold = bool(fs_selection & FS_BOLD)
            is_italic = bool(fs_selection & FS_ITALIC)
            subfamily = _name(font, 2)
            if not is_bold and "bold" in subfamily.lower():
                is_bold = True
            if not is_italic and ("italic" in subfamily.lower() or "oblique" in subfamily.lower()):
                is_italic = True

            return FontMetrics(
                path=str(font_path),
                family=_name(font, 1) or font_path.stem,
                subfamily=subfamily,
                units_per_em=upem,
                ascent=ascent,
                descent=descent,
                line_gap=line_gap,
                underline_position=upos,
                underline_thickness=uthick,
                glyph_count=len(order),
                advances=advances,
                cmap=cmap,
                kern=kern,
                is_bold=is_bold,
                is_italic=is_italic,
                use_typo_metrics=use_typo,
                win_ascent=win_ascent,
                win_descent=win_descent,
                os2_version=os2_version,
            )
    except FontMetricsError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a readable error
        raise FontMetricsError(f"cannot read font {path}: {exc}") from exc


def metrics_for_style(style: Mapping) -> FontMetrics:
    """Resolve the font of an ASS style dict and return its metrics.

    Accepts Aegisub's Lua style-table spellings (``fontname``, ``fontsize``) as
    well as this package's (``font``, ``font_size``).  An explicit ``font_file``
    wins over the family so an extracted ``[Fonts]`` attachment can be measured.
    """
    style = _as_style(style)
    family = str(_pick(style, _FONT_KEYS) or DEFAULT_FAMILY)
    bold = _truthy(_pick(style, _BOLD_KEYS))
    italic = _truthy(_pick(style, _ITALIC_KEYS))
    explicit = _pick(style, _FONT_FILE_KEYS)
    if explicit:
        explicit_path = Path(str(explicit))
        if explicit_path.is_file():
            return load_metrics(str(explicit_path))
        raise FontMetricsError(f"style font_file does not exist: {explicit!r}")
    match = match_font(family, bold, italic)
    if not match.get("file"):
        raise FontMetricsError(
            f"cannot resolve font {family!r}: {match.get('reason') or 'no match'}"
        )
    return load_metrics(str(match["file"]))


# --------------------------------------------------------------------------
# extents
# --------------------------------------------------------------------------


@dataclass
class Extents:
    """Layout extents of a text run, in script pixels."""

    width: float
    height: float
    descent: float
    ext_lead: float
    ascent: float
    font_file: str
    font_family: str
    size: float
    lines: list[float]  # width of each visual line
    scale_x: float
    scale_y: float
    spacing: float
    line_count: int = 1
    raw: bool = False

    def as_tuple(self) -> tuple[float, float, float, float]:
        """The Aegisub return order: ``width, height, descent, ext_lead``.

        This is exactly what ``aegisub.text_extents`` returns to Lua, so
        ``aegisub.text_extents_lua`` can hand it straight back.
        """
        return (self.width, self.height, self.descent, self.ext_lead)

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "descent": self.descent,
            "ext_lead": self.ext_lead,
            "ascent": self.ascent,
            "lines": self.lines,
            "line_count": self.line_count,
            "font_file": self.font_file,
            "font_family": self.font_family,
            "font_size": self.size,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "spacing": self.spacing,
            "raw": self.raw,
        }


def visible_text(text: str, *, raw: bool = False) -> str:
    """Text as the renderer lays it out: tags stripped, escapes resolved.

    With ``raw=True`` the string is returned untouched, which is Aegisub's
    documented behaviour ("formatting codes are not interpreted and will be
    taken as verbatim text").
    """
    if raw:
        return text
    stripped = T.strip_tags(text, keep_drawing=False)
    stripped = stripped.replace("\\h", _HARD_SPACE).replace("\\N", "\n").replace("\\n", "\n")
    stripped = stripped.replace("\\{", "{").replace("\\}", "}")
    return stripped


def _layout(
    text: str,
    met: FontMetrics,
    *,
    size: float,
    scale_x: float,
    spacing: float,
    raw: bool,
) -> list[tuple[float, list[tuple[str, float]]]]:
    """Per-line ``(width, [(char, advance), ...])`` for already-coerced input.

    Kerning is applied only when ``spacing`` is zero — see :func:`text_extents`.
    """
    kern_ok = not spacing
    out: list[tuple[float, list[tuple[str, float]]]] = []
    for line in visible_text(text, raw=raw).split("\n"):
        total = 0.0
        chars: list[tuple[str, float]] = []
        previous_gid: int | None = None
        for char in line:
            gid = met.glyph_for(char)
            advance = met.units_to_px(met.advances.get(gid, 0), size) * scale_x
            if previous_gid is not None and kern_ok:
                adjust = met.kern.get((previous_gid, gid))
                if adjust:
                    advance += met.units_to_px(adjust, size) * scale_x
            advance += spacing
            previous_gid = gid
            chars.append((char, advance))
            total += advance
        out.append((total, chars))
    return out


def char_advances(
    text: str,
    style: Mapping,
    *,
    metrics: FontMetrics | None = None,
    raw: bool = False,
) -> list[tuple[str, float]]:
    """Per-character advance widths in pixels, kerning applied.

    Newlines are reported with an advance of zero; callers lay out lines
    themselves (see :func:`text_extents`).
    """
    text, style = coerce_text_style(text, style)
    met = metrics or metrics_for_style(style)
    size = _style_number(style, _SIZE_KEYS, DEFAULT_FONT_SIZE, "font_size")
    scale_x = _style_number(style, _SCALE_X_KEYS, 100.0, "scale_x") / 100.0
    spacing = _style_number(style, _SPACING_KEYS, 0.0, "spacing")
    out: list[tuple[str, float]] = []
    for index, (_width, chars) in enumerate(
        _layout(text, met, size=size, scale_x=scale_x, spacing=spacing, raw=raw)
    ):
        if index:
            out.append(("\n", 0.0))
        out.extend(chars)
    return out


def text_extents(
    text: str,
    style: Mapping,
    *,
    metrics: FontMetrics | None = None,
    apply_scale: bool = True,
    lines: int | None = None,
    raw: bool = False,
) -> Extents:
    """Aegisub-compatible ``text_extents``.

    ``text``/``style`` may be given in either order (see :func:`coerce_text_style`).
    ``style`` is a mapping with Aegisub's Lua style-table keys (``fontname``,
    ``fontsize``, ``bold``, ``italic``, ``scalex``, ``scaley``, ``spacing``) or
    this package's (``font``, ``font_size``, ``scale_x``, ``scale_y``), plus an
    optional ``font_file`` to force a specific face.

    ``apply_scale=False`` ignores ``ScaleX``/``ScaleY``.  When ``lines`` is
    given the block is treated as that many visual lines even if the text has no
    explicit breaks (that is what libass does for auto-wrapped lines).  ``raw``
    disables tag stripping, matching Aegisub's literal-text behaviour.

    Semantics of the four Aegisub values:

    ``width``
        Advance width in pixels of the **widest** visual line, scaled by
        ``ScaleX``.  Aegisub itself only ever measures one line.
    ``height``
        Line height of the block: ``(ascent + descent)`` per line plus the
        line gap between lines, scaled by ``ScaleY``.
    ``descent``
        Distance in pixels below the baseline to the bottom of the font's
        descent, scaled by ``ScaleY``.
    ``ext_lead``
        External leading — the font's line gap — scaled by ``ScaleY``.
    """
    text, style = coerce_text_style(text, style)
    met = metrics or metrics_for_style(style)
    size = _style_number(style, _SIZE_KEYS, DEFAULT_FONT_SIZE, "font_size")
    scale_x = _style_number(style, _SCALE_X_KEYS, 100.0, "scale_x") / 100.0
    scale_y = _style_number(style, _SCALE_Y_KEYS, 100.0, "scale_y") / 100.0
    spacing = _style_number(style, _SPACING_KEYS, 0.0, "spacing")
    if not apply_scale:
        scale_x = scale_y = 1.0

    requested_lines = 0
    if lines is not None:
        try:
            requested_lines = int(lines)
        except (TypeError, ValueError) as exc:
            raise FontMetricsError(f"lines must be an integer, got {lines!r}") from exc
        if requested_lines < 0:
            raise FontMetricsError(f"lines must not be negative, got {lines!r}")

    layout = _layout(text, met, size=size, scale_x=scale_x, spacing=spacing, raw=raw)
    per_line = [width for width, _chars in layout]

    ascent = met.units_to_px(met.ascent, size) * scale_y
    descent = met.units_to_px(met.descent, size) * scale_y
    ext_lead = met.units_to_px(met.line_gap, size) * scale_y
    line_count = max(requested_lines, len(per_line), 1)
    return Extents(
        width=max(per_line) if per_line else 0.0,
        height=(ascent + descent) * line_count + ext_lead * (line_count - 1),
        descent=descent,
        ext_lead=ext_lead,
        ascent=ascent,
        font_file=met.path,
        font_family=met.family,
        size=size,
        lines=per_line,
        scale_x=scale_x,
        scale_y=scale_y,
        spacing=spacing,
        line_count=line_count,
        raw=raw,
    )


def text_extents_lua(text: str, style: Mapping) -> tuple[float, float, float, float]:
    """Tuple form, matching what the Lua shim hands back to Aegisub scripts.

    The tuple is ``(width, height, descent, ext_lead)`` in that order, exactly
    as ``aegisub.text_extents`` returns them to Lua.  Accepts either argument
    order.
    """
    ext = text_extents(text, style)
    return ext.as_tuple()


# --------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------


def font_coverage(font_path: str, text: str) -> dict:
    """Characters of ``text`` that the face cannot render."""
    met = load_metrics(font_path)
    text = _as_text(text)
    missing: list[str] = []
    present: list[str] = []
    seen: set[str] = set()
    for char in text:
        if char in {"\n", " ", "\t"} or char in seen:
            continue
        seen.add(char)
        if ord(char) in met.cmap:
            present.append(char)
        else:
            missing.append(char)
    return {
        "font_file": font_path,
        "font_family": met.family,
        "missing": missing,
        "missing_codepoints": [f"U+{ord(c):04X}" for c in missing],
        "present_count": len(present),
    }


def _fc_charset(char: str) -> str:
    """fontconfig charset query for a single character."""
    if not isinstance(char, str) or len(char) != 1:
        raise FontMetricsError(f"expected a single character, got {char!r}")
    return f"{ord(char):x}"


def fonts_with_char(char: str, *, limit: int = 20) -> list[str]:
    """Installed families that can render a character, via fontconfig.

    ``char`` must be exactly one character; anything else raises
    :class:`FontMetricsError` (an empty query is a programming error, not an
    empty result).  Returns ``[]`` when fontconfig is unavailable.
    """
    if not isinstance(char, str) or len(char) != 1:
        raise FontMetricsError(f"fonts_with_char expects a single character, got {char!r}")
    if not fontconfig_available():
        return []
    try:
        out = subprocess.run(
            ["fc-list", f":charset={_fc_charset(char)}", "-f", "%{family}\n"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return []
    families: list[str] = []
    for raw in out.stdout.splitlines():
        for family in raw.split(","):
            family = family.strip()
            if family and family not in families:
                families.append(family)
    return families[:limit]


def list_fonts(pattern: str | None = None, *, limit: int = 500) -> list[dict]:
    """Installed fonts, optionally filtered by a case-insensitive substring."""
    if not fontconfig_available():
        return []
    try:
        out = subprocess.run(
            ["fc-list", "-f", "%{family}\t%{style}\t%{file}\n"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return []
    needle = pattern.lower() if pattern else None
    fonts: list[dict] = []
    for raw in out.stdout.splitlines():
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        families = [f.strip() for f in parts[0].split(",") if f.strip()]
        family = families[0] if families else parts[0].strip()
        if needle and needle not in family.lower():
            continue
        fonts.append({"family": family, "families": families, "style": parts[1].strip(), "file": parts[2].strip()})
        if len(fonts) >= limit:
            break
    return fonts


def extents_report(text: str, style: Mapping, *, raw: bool = False) -> dict:
    """Extents plus the font actually used and a per-line breakdown."""
    text, style = coerce_text_style(text, style)
    met = metrics_for_style(style)
    ext = text_extents(text, style, metrics=met, raw=raw)
    data = ext.as_dict()
    per_char: list[dict] = []
    for index, (_width, chars) in enumerate(
        _layout(
            text,
            met,
            size=ext.size,
            scale_x=ext.scale_x,
            spacing=ext.spacing,
            raw=raw,
        )
    ):
        if index:
            per_char.append({"char": "\n", "advance": 0.0})
        per_char.extend({"char": char, "advance": round(advance, 4)} for char, advance in chars)
    data["per_char"] = per_char[:512]
    data["plain_text"] = visible_text(text, raw=raw)
    data["font_units_per_em"] = met.units_per_em
    data["font_metrics_source"] = met.metrics_source
    return data


def next_line_width(text: str, style: Mapping) -> float:
    """Helper used by typesetting tools: width of the last visual line."""
    ext = text_extents(text, style)
    return ext.lines[-1] if ext.lines else 0.0
