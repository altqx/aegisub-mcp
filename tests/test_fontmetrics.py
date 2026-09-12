"""Number-level tests for :mod:`aegisub_mcp.asscore.fontmetrics`.

Every expected value in this file is derived *independently* from the font's own
tables with fontTools (``head``/``hhea``/``OS/2``/``hmtx``/``cmap``/``kern``),
never from the module under test — so a passing run proves the module's numbers,
not that it agrees with itself.

Tests that need a specific font or fontconfig skip cleanly (``pytest.mark.skipif``)
when the asset is missing, so the file is green on a bare host too.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aegisub_mcp.asscore import fontmetrics as FM

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SIZE = 48.0

# Well-known families.  DejaVu Sans exercises the hhea metric path and has a
# legacy kern table; FreeSerif sets OS/2 USE_TYPO_METRICS *and* has a non-zero
# sTypoLineGap; Noto Sans Thai is the Thai-coverage font used in the report.
DEJAVU = "DejaVu Sans"
THAI = "Noto Sans Thai"
SERIF = "FreeSerif"
LIBERATION = "Liberation Serif"

HAVE_FONTTOOLS = importlib.util.find_spec("fontTools") is not None
HAVE_FC = shutil.which("fc-match") is not None

#: Extra on-disk locations to try when fontconfig is not installed.
FALLBACK_PATHS = {
    DEJAVU: (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ),
    THAI: (
        "/home/krapau/.local/share/fonts/NotoSansThai.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansThai.ttf",
    ),
    SERIF: ("/usr/share/fonts/truetype/freefont/FreeSerif.ttf",),
    LIBERATION: ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",),
}


def _fc_file(family: str) -> str | None:
    """``fc-match`` path for a family, or None."""
    if not HAVE_FC:
        return None
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}\n", family],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line and Path(line).is_file():
            return line
    return None


def find_font(family: str) -> str | None:
    """Locate a font file by family, via fontconfig then known fallback paths."""
    resolved = _fc_file(family)
    if resolved:
        return resolved
    for candidate in FALLBACK_PATHS.get(family, ()):
        if Path(candidate).is_file():
            return candidate
    return None


DEJAVU_PATH = find_font(DEJAVU)
THAI_PATH = find_font(THAI)
SERIF_PATH = find_font(SERIF)
LIBERATION_PATH = find_font(LIBERATION)

requires_fonttools = pytest.mark.skipif(not HAVE_FONTTOOLS, reason="fontTools is not installed")
requires_fontconfig = pytest.mark.skipif(not HAVE_FC, reason="fontconfig (fc-match) is absent")
requires_dejavu = pytest.mark.skipif(DEJAVU_PATH is None, reason=f"{DEJAVU} font file is absent")
requires_thai = pytest.mark.skipif(THAI_PATH is None, reason=f"{THAI} font file is absent")
requires_serif = pytest.mark.skipif(SERIF_PATH is None, reason=f"{SERIF} font file is absent")
requires_liberation = pytest.mark.skipif(LIBERATION_PATH is None, reason=f"{LIBERATION} is absent")
requires_latin_font = pytest.mark.skipif(DEJAVU_PATH is None, reason="no Latin test font available")


# --------------------------------------------------------------------------
# independent expectations, straight from the font tables
# --------------------------------------------------------------------------


def raw_tables(path: str):
    """Open a font with fontTools (single face)."""
    from fontTools.ttLib import TTFont

    return TTFont(path, lazy=True, fontNumber=0)


def expected_vertical(path: str) -> tuple[int, int, int, int]:
    """``(units_per_em, ascent, descent, line_gap)`` in font units.

    Reimplements HarfBuzz' selection rule from the raw tables: OS/2 ``sTypo*``
    wins only when ``fsSelection`` bit 7 (USE_TYPO_METRICS) is set, otherwise
    ``hhea``.  This is the ground truth the module must reproduce.
    """
    with raw_tables(path) as font:
        upem = int(font["head"].unitsPerEm)
        hhea = font["hhea"]
        os2 = font.get("OS/2")
        fs_selection = int(getattr(os2, "fsSelection", 0) or 0) if os2 is not None else 0
        if os2 is not None and fs_selection & 0x80:
            return (
                upem,
                int(os2.sTypoAscender),
                abs(int(os2.sTypoDescender)),
                int(os2.sTypoLineGap),
            )
        return upem, int(hhea.ascent), abs(int(hhea.descent)), int(hhea.lineGap)


def expected_hm_advances(path: str, text: str) -> dict[str, int]:
    """Per-character ``hmtx`` advance in font units (0 for unmapped chars)."""
    with raw_tables(path) as font:
        order = font.getGlyphOrder()
        gid_of = {name: index for index, name in enumerate(order)}
        cmap = font.getBestCmap() or {}
        hmtx = font["hmtx"]
        advances: dict[str, int] = {}
        for char in text:
            name = cmap.get(ord(char))
            gid = gid_of.get(name, 0)
            advances[char] = int(hmtx[order[gid]][0])
        return advances


def expected_kern_pairs(path: str) -> dict[tuple[str, str], int]:
    """Every kern pair in the font, keyed by glyph *name* pair."""
    pairs: dict[tuple[str, str], int] = {}
    with raw_tables(path) as font:
        table = font.get("kern")
        if table is None:
            return pairs
        for subtable in getattr(table, "kernTables", []) or []:
            for pair, value in (getattr(subtable, "kernTable", None) or {}).items():
                pairs[pair] = int(value)
    return pairs


def expected_width(
    path: str,
    text: str,
    size: float,
    *,
    scale_x: float = 1.0,
    spacing: float = 0.0,
    kern: bool = True,
) -> float:
    """Advance width of ``text`` in pixels, computed from the raw tables.

    Mirrors Aegisub's accumulation: ``width += (advance + spacing) * scaling``,
    with the kern adjustment folded into the right-hand character's advance.
    """
    with raw_tables(path) as font:
        upem = int(font["head"].unitsPerEm)
        order = font.getGlyphOrder()
        gid_of = {name: index for index, name in enumerate(order)}
        cmap = font.getBestCmap() or {}
        hmtx = font["hmtx"]
        pairs = expected_kern_pairs(path) if kern else {}
        total = 0.0
        previous: str | None = None
        for char in text:
            name = cmap.get(ord(char))
            gid = gid_of.get(name, 0)
            glyph_name = order[gid]
            advance = hmtx[glyph_name][0] * size / upem * scale_x
            if previous is not None:
                adjust = pairs.get((previous, glyph_name))
                if adjust:
                    advance += adjust * size / upem * scale_x
            total += advance + spacing
            previous = glyph_name
        return total


def latin_kern_sample() -> tuple[str, int] | None:
    """A printable-ASCII kern pair from DejaVu Sans, or None if there is none."""
    if DEJAVU_PATH is None:
        return None
    with raw_tables(DEJAVU_PATH) as font:
        cmap = font.getBestCmap() or {}
        rev = {name: chr(cp) for cp, name in cmap.items() if 32 < cp < 127}
    for (left, right), value in expected_kern_pairs(DEJAVU_PATH).items():
        if value and left in rev and right in rev:
            return rev[left] + rev[right], value
    return None


def styled(path: str | None, size: float = SIZE, **extra) -> dict:
    """A style dict pinned to an explicit font file (no fontconfig needed)."""
    style = {"font_file": str(path), "font_size": size}
    style.update(extra)
    return style


# --------------------------------------------------------------------------
# dependency-lightness and import hygiene
# --------------------------------------------------------------------------


def test_import_is_lazy_and_pulls_no_heavy_modules():
    """Importing the module must not import fontTools or the assutil helper.

    The module is declared dependency-light: fontTools is imported inside
    ``load_metrics`` and fontconfig is shelled out to, so a bare interpreter can
    import it.  Run in a subprocess to observe a genuinely fresh ``sys.modules``.
    """
    code = (
        "import sys\n"
        "from aegisub_mcp.asscore import fontmetrics\n"
        "bad = [m for m in ('fontTools', 'aegisub_mcp.asscore.assutil', 'numpy')\n"
        "       if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(SRC))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"import pulled in {result.stdout.strip()}"


def test_module_exposes_aegisub_extents_api():
    for name in (
        "text_extents",
        "text_extents_lua",
        "metrics_for_style",
        "load_metrics",
        "match_font",
        "fontconfig_available",
        "font_coverage",
        "char_advances",
        "extents_report",
        "fonts_with_char",
        "next_line_width",
        "Extents",
        "FontMetrics",
    ):
        assert hasattr(FM, name), name


# --------------------------------------------------------------------------
# vertical-metric selection (pure, no font file needed)
# --------------------------------------------------------------------------


class _Head:
    unitsPerEm = 2048


class _Hhea:
    def __init__(self, ascent, descent, line_gap):
        self.ascent = ascent
        self.descent = descent
        self.lineGap = line_gap


class _OS2:
    def __init__(self, fs_selection=0, typo=(0, 0, 0), win=(0, 0)):
        self.fsSelection = fs_selection
        self.sTypoAscender, self.sTypoDescender, self.sTypoLineGap = typo
        self.usWinAscent, self.usWinDescent = win


def test_hhea_metrics_used_when_use_typo_metrics_bit_is_clear():
    hhea = _Hhea(1901, -483, 12)
    os2 = _OS2(fs_selection=0x40, typo=(800, -200, 100), win=(2000, 700))
    assert FM.select_vertical_metrics(_Head(), hhea, os2) == (1901, 483, 12, False)


def test_os2_typo_metrics_used_when_bit7_set():
    hhea = _Hhea(1901, -483, 12)
    os2 = _OS2(fs_selection=FM.USE_TYPO_METRICS, typo=(800, -200, 100), win=(2000, 700))
    assert FM.select_vertical_metrics(_Head(), hhea, os2) == (800, 200, 100, True)


def test_win_metrics_are_the_last_resort_for_degenerate_hhea():
    hhea = _Hhea(0, 0, 0)
    os2 = _OS2(fs_selection=0, typo=(0, 0, 0), win=(2000, 700))
    assert FM.select_vertical_metrics(_Head(), hhea, os2) == (2000, 700, 0, False)


def test_missing_os2_still_yields_hhea_metrics():
    hhea = _Hhea(1901, -483, 0)
    assert FM.select_vertical_metrics(_Head(), hhea, None) == (1901, 483, 0, False)


# --------------------------------------------------------------------------
# unit scaling against the font's own tables
# --------------------------------------------------------------------------


@requires_fonttools
@requires_dejavu
@pytest.mark.parametrize("size", [16.0, 36.0, 48.0, 72.5])
def test_upem_and_vertical_metrics_match_font_tables(size):
    upem, ascent, descent, gap = expected_vertical(DEJAVU_PATH)
    metrics = FM.load_metrics(DEJAVU_PATH)
    assert metrics.units_per_em == upem
    assert (metrics.ascent, metrics.descent, metrics.line_gap) == (ascent, descent, gap)

    extents = FM.text_extents("Hello", styled(DEJAVU_PATH, size))
    assert extents.ascent == pytest.approx(ascent * size / upem, rel=1e-12)
    assert extents.descent == pytest.approx(descent * size / upem, rel=1e-12)
    assert extents.ext_lead == pytest.approx(gap * size / upem, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_height_is_ascent_plus_descent_for_one_line():
    upem, ascent, descent, gap = expected_vertical(DEJAVU_PATH)
    extents = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE))
    assert extents.height == pytest.approx((ascent + descent) * SIZE / upem, rel=1e-12)
    assert extents.line_count == 1


@requires_fonttools
@requires_serif
def test_os2_typo_font_uses_typo_metrics_and_line_gap():
    """FreeSerif sets USE_TYPO_METRICS, so sTypo* (with a real line gap) must win."""
    upem, ascent, descent, gap = expected_vertical(SERIF_PATH)
    metrics = FM.load_metrics(SERIF_PATH)
    assert metrics.use_typo_metrics is True
    assert metrics.metrics_source == "os2-typo"
    assert (metrics.ascent, metrics.descent, metrics.line_gap) == (ascent, descent, gap)
    assert gap > 0, "test font must exercise a non-zero line gap"
    extents = FM.text_extents("Ay", styled(SERIF_PATH, SIZE))
    assert extents.height == pytest.approx((ascent + descent) * SIZE / upem, rel=1e-12)
    assert extents.ext_lead == pytest.approx(gap * SIZE / upem, rel=1e-12)
    assert extents.ext_lead > 0


@requires_fonttools
@requires_liberation
def test_hhea_font_with_line_gap_reports_ext_lead():
    upem, ascent, descent, gap = expected_vertical(LIBERATION_PATH)
    assert gap > 0, "test font must exercise a non-zero line gap"
    extents = FM.text_extents("Ay", styled(LIBERATION_PATH, SIZE))
    assert extents.ext_lead == pytest.approx(gap * SIZE / upem, rel=1e-12)
    assert extents.height == pytest.approx((ascent + descent) * SIZE / upem, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_descent_and_ext_lead_scale_with_scaley():
    upem, ascent, descent, gap = expected_vertical(DEJAVU_PATH)
    extents = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE, scale_y=150.0))
    assert extents.scale_y == pytest.approx(1.5)
    assert extents.descent == pytest.approx(descent * SIZE / upem * 1.5, rel=1e-12)
    assert extents.ext_lead == pytest.approx(gap * SIZE / upem * 1.5, rel=1e-12)
    assert extents.height == pytest.approx((ascent + descent) * SIZE / upem * 1.5, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_apply_scale_false_ignores_scalex_and_scaley():
    plain = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE), apply_scale=False)
    scaled = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE, scale_x=200.0, scale_y=200.0))
    assert plain.scale_x == 1.0 and plain.scale_y == 1.0
    assert scaled.width == pytest.approx(plain.width * 2, rel=1e-12)


# --------------------------------------------------------------------------
# horizontal advances
# --------------------------------------------------------------------------


@requires_fonttools
@requires_dejavu
@pytest.mark.parametrize("text", ["H", "Hello", "Hello, World!", "AVATAR", "iiiii"])
def test_width_equals_sum_of_hmtx_advances(text):
    expected = expected_width(DEJAVU_PATH, text, SIZE)
    extents = FM.text_extents(text, styled(DEJAVU_PATH, SIZE))
    assert extents.width == pytest.approx(expected, rel=1e-12)
    assert extents.width > 0


@requires_fonttools
@requires_dejavu
def test_single_glyph_width_matches_its_own_advance():
    """DejaVu Sans 'H' has a 1540-unit advance; 1540/2048*48 == 36.09375 px."""
    table = expected_hm_advances(DEJAVU_PATH, "H")
    upem, _a, _d, _g = expected_vertical(DEJAVU_PATH)
    assert table["H"] == 1540, "DejaVu Sans 'H' advance is a documented constant"
    extents = FM.text_extents("H", styled(DEJAVU_PATH, SIZE))
    assert extents.width == pytest.approx(1540 * SIZE / 2048, rel=1e-12)
    assert extents.width == pytest.approx(36.09375, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_width_is_positive_and_monotonic_in_prefix_length():
    words = ["", "H", "He", "Hel", "Hell", "Hello"]
    widths = [FM.text_extents(word, styled(DEJAVU_PATH, SIZE)).width for word in words]
    assert widths[0] == 0.0
    assert all(width > 0 for width in widths[1:])
    assert widths == sorted(widths)
    assert widths[-1] > widths[1]


@requires_fonttools
@requires_dejavu
def test_latex_text_is_wider_than_its_english_word():
    """A longer string is strictly wider (non-zero, additive advances)."""
    short = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE)).width
    long = FM.text_extents("Hello, World! This is a longer line.", styled(DEJAVU_PATH, SIZE)).width
    assert long > short


@requires_fonttools
@requires_dejavu
def test_width_scales_linearly_with_font_size():
    at48 = FM.text_extents("Hello", styled(DEJAVU_PATH, 48.0)).width
    at96 = FM.text_extents("Hello", styled(DEJAVU_PATH, 96.0)).width
    assert at96 == pytest.approx(at48 * 2, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_width_scales_with_scalex():
    expected = expected_width(DEJAVU_PATH, "Hello", SIZE, scale_x=0.75)
    extents = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE, scale_x=75.0))
    assert extents.scale_x == pytest.approx(0.75)
    assert extents.width == pytest.approx(expected, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_spacing_is_added_once_per_character_including_the_last():
    """Aegisub: ``width += (advance + spacing) * scaling`` for every character."""
    text = "Hello"
    expected = expected_width(DEJAVU_PATH, text, SIZE, spacing=2.0)
    extents = FM.text_extents(text, styled(DEJAVU_PATH, SIZE, spacing=2.0))
    assert extents.width == pytest.approx(expected, rel=1e-12)
    bare = FM.text_extents(text, styled(DEJAVU_PATH, SIZE)).width
    assert extents.width == pytest.approx(bare + 2.0 * len(text), rel=1e-9)


@requires_fonttools
@requires_dejavu
def test_kerning_applies_only_when_spacing_is_zero():
    """Aegisub: "If there's inter-character spacing, kerning info must not be used"."""
    sample = latin_kern_sample()
    if sample is None:
        pytest.skip("test font exposes no kern pair between printable ASCII glyphs")
    text, adjustment = sample
    assert adjustment != 0

    kerned = FM.text_extents(text, styled(DEJAVU_PATH, SIZE))
    unkerned_expected = expected_width(DEJAVU_PATH, text, SIZE, kern=False)
    assert kerned.width == pytest.approx(expected_width(DEJAVU_PATH, text, SIZE), rel=1e-12)
    assert kerned.width != pytest.approx(unkerned_expected, rel=1e-12)

    spaced = FM.text_extents(text, styled(DEJAVU_PATH, SIZE, spacing=3.0))
    assert spaced.width == pytest.approx(
        expected_width(DEJAVU_PATH, text, SIZE, spacing=3.0, kern=False), rel=1e-12
    )


@requires_fonttools
@requires_dejavu
def test_char_advances_match_hmtx_and_sum_to_width():
    text = "Hello"
    upem, _a, _d, _g = expected_vertical(DEJAVU_PATH)
    advances = FM.char_advances(text, styled(DEJAVU_PATH, SIZE))
    assert [char for char, _adv in advances] == list(text)
    table = expected_hm_advances(DEJAVU_PATH, text)
    kern = latin_kern_sample()
    for char, advance in advances:
        assert advance == pytest.approx(table[char] * SIZE / upem, rel=1e-12)
    total = sum(advance for _char, advance in advances)
    assert FM.text_extents(text, styled(DEJAVU_PATH, SIZE)).width == pytest.approx(total, rel=1e-12)


# --------------------------------------------------------------------------
# empty / whitespace / multi-line
# --------------------------------------------------------------------------


@requires_fonttools
@requires_dejavu
def test_empty_string_is_zero_width_with_a_real_line_height():
    upem, ascent, descent, gap = expected_vertical(DEJAVU_PATH)
    extents = FM.text_extents("", styled(DEJAVU_PATH, SIZE))
    assert extents.width == 0.0
    assert extents.lines == [0.0]
    assert extents.line_count == 1
    assert extents.height == pytest.approx((ascent + descent) * SIZE / upem, rel=1e-12)
    assert extents.descent > 0
    assert extents.as_tuple()[0] == 0.0


@requires_fonttools
@requires_dejavu
def test_whitespace_string_has_nonzero_width_from_the_space_advance():
    expected = expected_width(DEJAVU_PATH, "   ", SIZE)
    extents = FM.text_extents("   ", styled(DEJAVU_PATH, SIZE))
    assert expected > 0
    assert extents.width == pytest.approx(expected, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_newlines_split_lines_and_width_is_the_widest():
    upem, ascent, descent, gap = expected_vertical(DEJAVU_PATH)
    extents = FM.text_extents("Hi\nHello", styled(DEJAVU_PATH, SIZE))
    assert extents.line_count == 2
    assert len(extents.lines) == 2
    assert extents.lines[0] == pytest.approx(expected_width(DEJAVU_PATH, "Hi", SIZE), rel=1e-12)
    assert extents.lines[1] == pytest.approx(expected_width(DEJAVU_PATH, "Hello", SIZE), rel=1e-12)
    assert extents.width == max(extents.lines)
    assert extents.height == pytest.approx(
        (ascent + descent) * SIZE / upem * 2 + gap * SIZE / upem, rel=1e-12
    )


@requires_fonttools
@requires_serif
def test_multiline_height_includes_the_line_gap_between_lines():
    upem, ascent, descent, gap = expected_vertical(SERIF_PATH)
    assert gap > 0
    extents = FM.text_extents("A\\NB\\NC", styled(SERIF_PATH, SIZE))
    assert extents.line_count == 3
    single = (ascent + descent) * SIZE / upem
    assert extents.height == pytest.approx(single * 3 + gap * SIZE / upem * 2, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_explicit_line_count_overrides_layout():
    """libass passes a wrapped line count; height must follow it."""
    upem, ascent, descent, gap = expected_vertical(DEJAVU_PATH)
    extents = FM.text_extents("Hi", styled(DEJAVU_PATH, SIZE), lines=4)
    assert extents.line_count == 4
    assert extents.width == pytest.approx(expected_width(DEJAVU_PATH, "Hi", SIZE), rel=1e-12)
    assert extents.height == pytest.approx(
        (ascent + descent) * SIZE / upem * 4 + gap * SIZE / upem * 3, rel=1e-12
    )


@requires_fonttools
@requires_dejavu
def test_next_line_width_returns_the_last_line():
    extents = FM.text_extents("Hi\nHello", styled(DEJAVU_PATH, SIZE))
    assert FM.next_line_width("Hi\nHello", styled(DEJAVU_PATH, SIZE)) == pytest.approx(
        extents.lines[-1], rel=1e-12
    )


# --------------------------------------------------------------------------
# tags and escapes
# --------------------------------------------------------------------------


@requires_fonttools
@requires_dejavu
def test_override_tags_are_stripped_by_default_and_kept_when_raw():
    style = styled(DEJAVU_PATH, SIZE)
    stripped = FM.text_extents(r"{\b1}Hi", style)
    raw = FM.text_extents(r"{\b1}Hi", style, raw=True)
    assert stripped.width == pytest.approx(expected_width(DEJAVU_PATH, "Hi", SIZE), rel=1e-12)
    assert raw.width == pytest.approx(expected_width(DEJAVU_PATH, r"{\b1}Hi", SIZE), rel=1e-12)
    assert raw.width > stripped.width
    assert raw.raw is True and stripped.raw is False


@requires_fonttools
@requires_dejavu
def test_visible_text_resolves_escapes_and_strips_tags():
    assert FM.visible_text(r"{\i1}ab{\i0}\Ncd") == "ab\ncd"
    assert FM.visible_text("a\\hb") == "a\u00a0b"
    assert FM.visible_text(r"{\b1}Hi", raw=True) == r"{\b1}Hi"


# --------------------------------------------------------------------------
# Thai coverage
# --------------------------------------------------------------------------


THAI_TEXT = "สวัสดี"


@requires_fonttools
@requires_thai
def test_thai_width_matches_noto_sans_thai_tables():
    upem, _a, _d, _g = expected_vertical(THAI_PATH)
    expected = expected_width(THAI_PATH, THAI_TEXT, SIZE)
    extents = FM.text_extents(THAI_TEXT, styled(THAI_PATH, SIZE))
    assert expected > 0
    assert extents.width == pytest.approx(expected, rel=1e-12)
    assert extents.width > 0
    metric = FM.load_metrics(THAI_PATH)
    assert metric.family.lower().startswith("noto sans thai") or "thai" in metric.path.lower()
    assert all(0x0E00 <= ord(char) <= 0x0E7F for char in THAI_TEXT)


@requires_fonttools
@requires_thai
def test_thai_advance_is_nonzero_and_scales_with_size():
    small = FM.text_extents(THAI_TEXT, styled(THAI_PATH, 24.0))
    large = FM.text_extents(THAI_TEXT, styled(THAI_PATH, 48.0))
    assert small.width > 0
    assert large.width == pytest.approx(small.width * 2, rel=1e-12)
    assert large.height > small.height


@requires_fonttools
@requires_dejavu
@requires_thai
def test_thai_text_is_wider_than_latin_at_the_same_length():
    latin = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE)).width
    thai = FM.text_extents(THAI_TEXT, styled(THAI_PATH, SIZE)).width
    assert thai > 0 and latin > 0 and thai != latin


@requires_fonttools
@requires_thai
def test_thai_font_lacks_an_emoji_and_coverage_reports_it():
    report = FM.font_coverage(THAI_PATH, "\U0001F600" + THAI_TEXT)
    assert "\U0001F600" in report["missing"]
    assert "U+1F600" in report["missing_codepoints"]
    assert report["present_count"] > 0


@requires_fonttools
@requires_dejavu
def test_font_coverage_ignores_spaces_and_newlines():
    report = FM.font_coverage(DEJAVU_PATH, "ab cd\nef")
    assert report["missing"] == []
    assert report["present_count"] == 6  # a b c d e f, spaces/newline skipped
    assert sorted(report["missing_codepoints"]) == []


# --------------------------------------------------------------------------
# Aegisub argument order and style-key spellings
# --------------------------------------------------------------------------


@requires_fonttools
@requires_dejavu
def test_aegisub_documented_argument_order_style_first():
    """``aegisub.text_extents(style, text)`` — style first, per Aegisub's docs."""
    style = styled(DEJAVU_PATH, SIZE)
    text_first = FM.text_extents("Hello", style)
    style_first = FM.text_extents(style, "Hello")
    assert style_first.as_tuple() == text_first.as_tuple()
    assert FM.text_extents_lua(style, "Hello") == text_first.as_tuple()


@requires_fonttools
@requires_dejavu
def test_aegisub_lua_style_table_spellings_are_honoured():
    """The Lua style table uses ``fontname``/``fontsize``/``scalex``/``scaley``.

    Silently ignoring these produced the wrong font *and* the wrong size, which
    is the defect these tests were written to pin down.
    """
    lua_style = {"font_file": DEJAVU_PATH, "fontname": DEJAVU, "fontsize": 36, "scalex": 100, "scaley": 100}
    extents = FM.text_extents("Hello", lua_style)
    assert extents.size == 36.0
    assert extents.width == pytest.approx(expected_width(DEJAVU_PATH, "Hello", 36.0), rel=1e-12)

    underscorized = FM.text_extents("Hello", styled(DEJAVU_PATH, 36.0))
    assert extents.width == pytest.approx(underscorized.width, rel=1e-12)

    sized = FM.text_extents("Hello", {"font_file": DEJAVU_PATH, "fontname": DEJAVU, "fontsize": 72})
    assert sized.size == 72.0
    assert sized.width == pytest.approx(extents.width * 2, rel=1e-12)


@requires_fonttools
@requires_dejavu
def test_unknown_style_keys_and_missing_size_use_aegisub_defaults():
    extents = FM.text_extents("Hello", {"font_file": DEJAVU_PATH, "whatever": 1})
    assert extents.size == FM.DEFAULT_FONT_SIZE
    assert extents.scale_x == 1.0 and extents.scale_y == 1.0
    assert extents.spacing == 0.0


@requires_fonttools
@requires_dejavu
def test_non_numeric_font_size_raises_fontmetrics_error():
    with pytest.raises(FM.FontMetricsError):
        FM.text_extents("Hello", {"font_file": DEJAVU_PATH, "font_size": "not-a-number"})


@requires_fonttools
@requires_dejavu
def test_negative_line_count_is_rejected():
    with pytest.raises(FM.FontMetricsError):
        FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE), lines=-1)


@requires_fonttools
@requires_dejavu
def test_utf8_bytes_text_is_accepted():
    """Lua hands strings across as UTF-8 bytes."""
    extents = FM.text_extents("Hello".encode(), styled(DEJAVU_PATH, SIZE))
    assert extents.width == pytest.approx(expected_width(DEJAVU_PATH, "Hello", SIZE), rel=1e-12)


def test_non_mapping_style_raises_fontmetrics_error():
    with pytest.raises(FM.FontMetricsError):
        FM.text_extents("Hello", "DejaVu Sans")


def test_non_string_text_raises_fontmetrics_error():
    with pytest.raises(FM.FontMetricsError):
        FM.text_extents(123, {"font": DEJAVU, "font_size": SIZE})


# --------------------------------------------------------------------------
# unknown / missing fonts
# --------------------------------------------------------------------------


def test_missing_font_file_raises():
    style = {"font_file": "/nonexistent/path/NoSuchFont.ttf", "font_size": SIZE}
    with pytest.raises(FM.FontMetricsError):
        FM.metrics_for_style(style)
    with pytest.raises(FM.FontMetricsError):
        FM.text_extents("Hello", style)


def test_load_metrics_missing_path_raises():
    with pytest.raises(FM.FontMetricsError):
        FM.load_metrics("/nonexistent/path/NoSuchFont.ttf")


def test_load_metrics_non_font_file_raises(tmp_path):
    junk = tmp_path / "not-a-font.ttf"
    junk.write_bytes(b"this is definitely not a font file\n" * 20)
    with pytest.raises(FM.FontMetricsError):
        FM.load_metrics(str(junk))


def test_load_metrics_directory_raises(tmp_path):
    with pytest.raises(FM.FontMetricsError):
        FM.load_metrics(str(tmp_path))


def test_unresolvable_family_raises_when_fontconfig_is_absent(monkeypatch):
    monkeypatch.setattr(FM, "fontconfig_available", lambda: False)
    with pytest.raises(FM.FontMetricsError) as excinfo:
        FM.metrics_for_style({"font": "DejaVu Sans", "font_size": SIZE})
    assert "DejaVu Sans" in str(excinfo.value)


def test_match_font_reports_no_match_without_fontconfig(monkeypatch):
    monkeypatch.setattr(FM, "fontconfig_available", lambda: False)
    result = FM.match_font("DejaVu Sans")
    assert result["file"] is None
    assert "fontconfig" in (result.get("reason") or "")


@requires_fontconfig
def test_unknown_family_is_substituted_rather_than_crashing():
    """fontconfig always substitutes; Aegisub/libass do the same.

    The contract is therefore: an unknown family yields a documented
    *substitution* (with ``substituted`` flagged), not an exception.
    """
    result = FM.match_font("ThisFamilyDoesNotExistZZZ")
    assert result["file"] is not None
    assert result["substituted"] is True
    extents = FM.text_extents("Hello", {"font": "ThisFamilyDoesNotExistZZZ", "font_size": SIZE})
    assert extents.width > 0
    assert extents.descent > 0


@requires_fontconfig
@requires_fonttools
def test_known_family_resolves_to_its_own_file():
    result = FM.match_font(DEJAVU)
    assert result["file"] is not None
    assert result["substituted"] is False
    metrics = FM.metrics_for_style({"font": DEJAVU, "font_size": SIZE})
    assert Path(metrics.path).is_file()


@requires_fonttools
@requires_thai
@requires_fontconfig
def test_thai_family_resolves_and_measures():
    metrics = FM.metrics_for_style({"fontname": THAI, "fontsize": SIZE})
    assert Path(metrics.path).is_file()
    extents = FM.text_extents(THAI_TEXT, {"fontname": THAI, "fontsize": SIZE})
    assert extents.width > 0
    assert extents.width == pytest.approx(expected_width(metrics.path, THAI_TEXT, SIZE), rel=1e-12)


def test_fontconfig_available_is_a_bool():
    assert isinstance(FM.fontconfig_available(), bool)


# --------------------------------------------------------------------------
# tuple / dict shape — the aegisub.text_extents contract
# --------------------------------------------------------------------------


@requires_fonttools
@requires_dejavu
def test_tuple_order_is_width_height_descent_ext_lead():
    extents = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE))
    width, height, descent, ext_lead = extents.as_tuple()
    assert (width, height, descent, ext_lead) == (
        extents.width,
        extents.height,
        extents.descent,
        extents.ext_lead,
    )
    assert width > 0
    assert height > 0
    assert descent > 0
    assert ext_lead >= 0
    assert height >= descent * 2  # ascent alone must be positive
    assert FM.text_extents_lua("Hello", styled(DEJAVU_PATH, SIZE)) == (width, height, descent, ext_lead)


@requires_fonttools
@requires_dejavu
def test_dict_uses_the_documented_aegisub_key_names():
    data = FM.text_extents("Hello", styled(DEJAVU_PATH, SIZE)).as_dict()
    for key in ("width", "height", "descent", "ext_lead"):
        assert key in data
    assert data["line_count"] == 1
    assert data["font_size"] == SIZE
    assert len(data["lines"]) == 1


@requires_fonttools
@requires_dejavu
def test_extents_report_carries_font_provenance_and_per_char_advances():
    report = FM.extents_report("Hi", styled(DEJAVU_PATH, SIZE))
    assert report["font_units_per_em"] == expected_vertical(DEJAVU_PATH)[0]
    assert report["font_metrics_source"] in {"hhea", "os2-typo"}
    assert report["plain_text"] == "Hi"
    assert [entry["char"] for entry in report["per_char"]] == ["H", "i"]
    table = expected_hm_advances(DEJAVU_PATH, "Hi")
    for entry in report["per_char"]:
        expected = table[entry["char"]] * SIZE / report["font_units_per_em"]
        # extents_report rounds per-character advances to 4 decimals for display.
        assert entry["advance"] == pytest.approx(round(expected, 4), rel=1e-12)


@requires_fonttools
@requires_thai
def test_thai_font_metrics_source_matches_its_os2_flag():
    upem, ascent, descent, gap = expected_vertical(THAI_PATH)
    metrics = FM.load_metrics(THAI_PATH)
    assert (metrics.units_per_em, metrics.ascent, metrics.descent, metrics.line_gap) == (
        upem,
        ascent,
        descent,
        gap,
    )
    expected_source = "os2-typo" if metrics.use_typo_metrics else "hhea"
    assert metrics.metrics_source == expected_source


# --------------------------------------------------------------------------
# font enumeration helpers
# --------------------------------------------------------------------------


def test_fonts_with_char_requires_exactly_one_character():
    with pytest.raises(FM.FontMetricsError):
        FM.fonts_with_char("")
    with pytest.raises(FM.FontMetricsError):
        FM.fonts_with_char("ab")


@requires_fontconfig
def test_fonts_with_char_finds_a_font_for_thai():
    families = FM.fonts_with_char("\u0E2A")
    assert families, "no installed font claims coverage of U+0E2A (THAI LETTER SO)"
    assert all(isinstance(name, str) and name for name in families)


@requires_fontconfig
def test_list_fonts_returns_files_that_exist():
    fonts = FM.list_fonts("DejaVu", limit=10)
    assert fonts
    for entry in fonts:
        assert "DejaVu".lower() in entry["family"].lower()
        assert Path(entry["file"]).is_file()


def test_list_fonts_without_fontconfig_is_empty(monkeypatch):
    monkeypatch.setattr(FM, "fontconfig_available", lambda: False)
    assert FM.list_fonts() == []
    assert FM.fonts_with_char("A") == []
