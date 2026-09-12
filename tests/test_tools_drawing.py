"""Tests for :mod:`aegisub_mcp.tools.drawing_tools`.

Covers the drawing / clipping / font tool surface: input resolution
(``index=``+``doc_id=`` vs raw ``text=``), hand-computed geometry, SVG
interop, clip scale conversion convergence (verified by rendering with
libass) and the font tools.

Every expected number that can be derived by hand is derived by hand:
a 100x100 square must report bbox ``[0, 0, 100, 100]``, size ``[100, 100]``,
centre ``[50, 50]`` and perimeter ``400``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegisub_mcp.asscore import measure as M
from aegisub_mcp.tools import drawing_tools as DT
from aegisub_mcp.tools.base import ToolError, workspace

FIXTURES = Path(__file__).parent / "fixtures" / "real"
BASIC = FIXTURES / "basic.ass"
LEGACY_FONTS = FIXTURES / "legacy-attachments-real.ass"

# A closed 100x100 square: 4 segments of 100 units -> perimeter 400.
SQUARE = "m 0 0 l 100 0 l 100 100 l 0 100 c"
# Two disjoint squares, 10x10 at the origin and 50x50 at (100, 100).
TWO_SQUARES = "m 0 0 l 10 0 10 10 0 10 c m 100 100 l 150 100 150 150 100 150 c"

ALL_TOOLS = [
    "ass_convert_clip_scale",
    "ass_drawing_bbox",
    "ass_drawing_info",
    "ass_drawing_to_svg",
    "ass_font_coverage",
    "ass_fonts_used",
    "ass_fonts_with_char",
    "ass_get_clips",
    "ass_get_drawing",
    "ass_glyph_check",
    "ass_join_drawings",
    "ass_list_fonts",
    "ass_match_font",
    "ass_remove_clip",
    "ass_scale_drawing",
    "ass_set_clip",
    "ass_set_drawing",
    "ass_split_drawing",
    "ass_svg_to_drawing",
    "ass_transform_drawing",
]


@pytest.fixture(autouse=True)
def clean_workspace(tmp_path, monkeypatch):
    """Isolate every test from workspace state left by other modules."""
    reset = getattr(workspace, "reset", None)
    if callable(reset):
        reset()
    for did in list(workspace.ids()):
        workspace.close(did)
    workspace.selection = []
    monkeypatch.setattr(workspace, "output_dir", tmp_path / "out")
    yield
    for did in list(workspace.ids()):
        workspace.close(did)
    workspace.selection = []


def _j(payload) -> str:
    """Assert the payload is JSON-serialisable and return it re-encoded."""
    return json.dumps(payload)


def _doc(play_res=(640, 480), doc_id="doc"):
    did = workspace.new(play_res=play_res, doc_id=doc_id)
    return did, workspace.get(did)


def _drawing_line(doc, text, start_ms=0, end_ms=2000, style="Default"):
    entry = doc.add_event(start_ms=start_ms, end_ms=end_ms, style=style, text=text)
    # document.add_event() only honours field names (``Style=``); the friendly
    # ``style=`` keyword is silently dropped, so set the field explicitly.
    entry.set("Style", style)
    return entry


def _rect(result):
    return result["rect"]


def _render_rect(text, time_ms):
    """Render and reduce the ink rectangle to x/y/width/height.

    ``measure.measure_render()`` reports six keys (x, y, width, height and the
    redundant x1/y1 corner); the four-key subset is what the tests compare.
    """
    measured = M.measure_render(text, time_ms)["rect"]
    return {key: measured[key] for key in ("x", "y", "width", "height")}


# ---------------------------------------------------------------------------
# byte-exact round-trip through the tool layer
# ---------------------------------------------------------------------------


def test_round_trip_bytes_unchanged(tmp_path):
    """Open a real fixture, read through the tools, save: identical bytes."""
    original = BASIC.read_bytes()
    did = workspace.open(str(BASIC), doc_id="basic")
    doc = workspace.get(did)
    assert doc.dirty is False

    # Exercise the read path: this line has no drawing, so nothing is mutated.
    info = DT.ass_get_drawing(index=0, doc_id=did)
    assert info["source"] == "line"
    assert info["has_drawing"] is False
    assert info["drawing"] == ""
    assert doc.dirty is False

    out = tmp_path / "copy.ass"
    saved = workspace.save(did, out)
    assert saved["bytes"] == len(original)
    assert out.read_bytes() == original
    assert out.read_bytes()[:3] == b"\xef\xbb\xbf"  # BOM preserved
    assert b"\r\n" in out.read_bytes()  # CRLF preserved


REAL_FIXTURES = sorted(p.name for p in FIXTURES.iterdir() if p.is_file())


@pytest.mark.parametrize("fixture", REAL_FIXTURES)
def test_every_real_fixture_round_trips_byte_exact(fixture, tmp_path):
    """Each file in tests/fixtures/real/ survives open -> tools -> save.

    None of the real fixtures carries a drawing or a clip, so the drawing read
    path must report an empty drawing for every line instead of raising, and
    the file must still come back out byte for byte (BOM, CRLF and all).
    """
    path = FIXTURES / fixture
    original = path.read_bytes()
    did = workspace.open(str(path), doc_id=fixture)
    doc = workspace.get(did)
    assert doc.dirty is False

    for index in range(len(doc.events())):
        info = DT.ass_get_drawing(index=index, doc_id=did)
        assert info["source"] == "line"
        assert info["has_drawing"] is False
        assert info["drawing"] == ""
        assert DT.ass_get_clips(index=index, doc_id=did)["clips"] == []

    out = tmp_path / fixture
    workspace.save(did, out)
    assert out.read_bytes() == original


def test_split_returns_parts_alias():
    """``ass_split_drawing`` exposes the same list as ``subpaths`` and ``parts``."""
    split = DT.ass_split_drawing(text=TWO_SQUARES)
    assert split["parts"] == split["subpaths"]
    assert split["count"] == 2
    assert DT.ass_join_drawings(split["parts"])["subpath_count"] == 2


def test_centre_is_an_alias_of_center():
    """The spec spells it ``centre``; both keys must carry the same value."""
    assert DT.ass_get_drawing(text=SQUARE)["centre"] == [50.0, 50.0]
    assert DT.ass_get_drawing(text=SQUARE)["centre"] == DT.ass_get_drawing(text=SQUARE)["center"]
    assert DT.ass_drawing_info(text=SQUARE)["centre"] == [50.0, 50.0]
    assert DT.ass_drawing_bbox(text=SQUARE)["centre"] == [50.0, 50.0]
    moved = DT.ass_transform_drawing(text=SQUARE, action="translate", dx=10.0)
    assert moved["centre"] == moved["center"] == [60.0, 50.0]


def test_no_such_line_raises_tool_error():
    did, _doc_ref = _doc()
    with pytest.raises(ToolError):
        DT.ass_get_drawing(index=99, doc_id=did)


# ---------------------------------------------------------------------------
# input resolution reports which source was used
# ---------------------------------------------------------------------------


def test_get_drawing_from_line_reports_source_and_hand_computed_geometry():
    did, doc = _doc()
    _drawing_line(doc, "{\\an7\\pos(0,0)\\p1}" + SQUARE)
    result = DT.ass_get_drawing(index=0, doc_id=did)

    assert result["source"] == "line"
    assert result["index"] == 0
    assert result["scale"] == 1
    assert result["has_drawing"] is True
    # 100 by 100 square anchored at the origin.
    assert result["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert result["size"] == [100.0, 100.0]
    assert result["center"] == [50.0, 50.0]
    assert result["path_length"] == 400.0
    assert result["point_count"] == 4
    assert result["subpath_count"] == 1
    assert result["svg_path"] == "M 0 0 L 100 0 L 100 100 L 0 100 Z"
    assert result["drawing"] == SQUARE
    _j(result)


def test_get_drawing_from_text_reports_text_source():
    result = DT.ass_get_drawing(text=SQUARE)
    assert result["source"] == "text"
    assert result["index"] is None
    assert result["doc_id"] is None
    assert result["bbox"] == [0.0, 0.0, 100.0, 100.0]


def test_get_drawing_accepts_a_whole_line_in_text():
    result = DT.ass_get_drawing(text="{\\p2}" + SQUARE + "{\\p0}")
    assert result["source"] == "text"
    assert result["drawing"] == SQUARE
    assert result["scale"] == 2


def test_get_drawing_reports_clip_tags_on_the_line():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE + "{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")
    result = DT.ass_get_drawing(index=0, doc_id=did)
    # The clip block must not leak into the reported path data.
    assert result["drawing"] == SQUARE
    assert result["subpath_count"] == 1
    assert len(result["clips"]) == 1
    assert result["clips"][0]["kind"] == "vector"
    assert result["clips"][0]["drawing"] == "m 20 20 l 80 20 80 80 20 80"


def test_get_drawing_rejects_ambiguous_input():
    with pytest.raises(ToolError):
        DT.ass_get_drawing(index=0, text=SQUARE)
    with pytest.raises(ToolError):
        DT.ass_get_drawing()


def test_get_drawing_without_document_raises():
    with pytest.raises(ToolError):
        DT.ass_get_drawing(index=0, doc_id="nope")


# ---------------------------------------------------------------------------
# drawing_info: commands, per-subpath boxes, totals
# ---------------------------------------------------------------------------


def test_drawing_info_commands_and_subpath_boxes_by_hand():
    info = DT.ass_drawing_info(text=TWO_SQUARES)
    assert info["source"] == "text"
    # parser coalesces a run of coordinates into one command per letter run
    assert [c["kind"] for c in info["commands"]] == ["m", "l", "c", "m", "l", "c"]
    assert [len(c["points"]) for c in info["commands"]] == [1, 3, 0, 1, 3, 0]
    assert info["commands"][0]["args"] == [0.0, 0.0]
    assert info["commands"][3]["args"] == [100.0, 100.0]  # the second move
    assert info["subpath_count"] == 2
    assert [s["bbox"] for s in info["subpaths"]] == [
        [0.0, 0.0, 10.0, 10.0],
        [100.0, 100.0, 150.0, 150.0],
    ]
    assert [s["point_count"] for s in info["subpaths"]] == [4, 4]
    assert info["bbox"] == [0.0, 0.0, 150.0, 150.0]
    assert info["size"] == [150.0, 150.0]
    assert info["center"] == [75.0, 75.0]
    _j(info)


def test_drawing_info_from_line_index():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + TWO_SQUARES + "{\\p0}")
    info = DT.ass_drawing_info(index=0, doc_id=did)
    assert info["source"] == "line"
    assert info["subpath_count"] == 2
    assert info["bbox"] == [0.0, 0.0, 150.0, 150.0]


def test_drawing_info_on_line_without_drawing_raises():
    did, doc = _doc()
    _drawing_line(doc, "just text")
    with pytest.raises(ToolError):
        DT.ass_drawing_info(index=0, doc_id=did)


def test_drawing_bbox_tool_matches_info():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE)
    bbox = DT.ass_drawing_bbox(index=0, doc_id=did)
    info = DT.ass_drawing_info(index=0, doc_id=did)
    assert bbox["bbox"] == info["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert bbox["center"] == [50.0, 50.0]
    assert bbox["size"] == [100.0, 100.0]
    assert bbox["point_count"] == 4


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def test_transform_scale_keeps_aspect_ratio():
    scaled = DT.ass_transform_drawing(text=SQUARE, action="scale", factor=2.5)
    assert scaled["action"] == "scale"
    assert scaled["in_place"] is False
    assert scaled["size"] == [250.0, 250.0]
    ratio = scaled["size"][0] / scaled["size"][1]
    assert ratio == pytest.approx(1.0, abs=1e-9)

    smaller = DT.ass_transform_drawing(text=SQUARE, action="scale", factor=0.5)
    assert smaller["size"] == [50.0, 50.0]
    # Scaling happens about the drawing's own bbox centre by default.
    assert smaller["center"] == [50.0, 50.0]
    assert scaled["center"] == [50.0, 50.0]
    wider = DT.ass_transform_drawing(text=SQUARE, action="scale", factor=3.0)
    assert wider["bbox"] == [-100.0, -100.0, 200.0, 200.0]


def test_transform_scale_to_size_fits_within_the_requested_box():
    result = DT.ass_transform_drawing(text=SQUARE, action="scale_to_size", dx=200.0, dy=50.0)
    width, height = result["size"]
    assert width == pytest.approx(50.0, abs=0.5)
    assert height == pytest.approx(50.0, abs=0.5)  # uniform: limited by height


def test_transform_stretch_to_bbox_fills_exactly():
    result = DT.ass_transform_drawing(text=SQUARE, action="stretch_to_bbox", dx=200.0, dy=50.0)
    assert result["size"] == pytest.approx([200.0, 50.0], abs=0.5)


def test_transform_translate_shifts_the_box():
    result = DT.ass_transform_drawing(text=SQUARE, action="translate", dx=10.0, dy=-20.0)
    assert result["bbox"] == [10.0, -20.0, 110.0, 80.0]
    assert result["drawing"] == "m 10 -20 l 110 -20 l 110 80 l 10 80 c"


def test_transform_rotate_about_default_centre_keeps_size_then_grows():
    quarter = DT.ass_transform_drawing(text=SQUARE, action="rotate", angle_deg=90.0)
    assert quarter["size"] == pytest.approx([100.0, 100.0], abs=0.5)
    assert quarter["center"] == pytest.approx([50.0, 50.0], abs=0.5)
    eighth = DT.ass_transform_drawing(text=SQUARE, action="rotate", angle_deg=45.0)
    assert eighth["size"][0] == pytest.approx(141.42, abs=0.5)


def test_transform_mirror_keeps_extents():
    mirrored = DT.ass_transform_drawing(text=SQUARE, action="mirror", axis="x")
    assert mirrored["size"] == pytest.approx([100.0, 100.0], abs=0.5)
    assert mirrored["center"] == pytest.approx([50.0, 50.0], abs=0.5)
    flipped = DT.ass_transform_drawing(text=SQUARE, action="mirror", axis="y")
    assert flipped["size"] == pytest.approx([100.0, 100.0], abs=0.5)


def test_transform_centre_at_origin_and_explicit_origin():
    moved = DT.ass_transform_drawing(text=SQUARE, action="centre_at_origin")
    assert moved["center"] == pytest.approx([0.0, 0.0], abs=0.5)
    moved_explicit = DT.ass_transform_drawing(
        text=SQUARE, action="centre_at_origin", origin_x=100.0, origin_y=200.0
    )
    assert moved_explicit["center"] == pytest.approx([100.0, 200.0], abs=0.5)
    scaled_about_origin = DT.ass_transform_drawing(
        text=SQUARE, action="scale", factor=2.0, origin_x=0.0, origin_y=0.0
    )
    assert scaled_about_origin["bbox"] == [0.0, 0.0, 200.0, 200.0]


def test_transform_round_reverse_flatten_simplify():
    rounded = DT.ass_transform_drawing(text="m 0 0 l 13 0 13 7 0 7 c", action="round", factor=5.0)
    # same command letter, one coordinate list (the canonical ASS serialisation)
    assert rounded["drawing"] == "m 0 0 l 15 0 15 5 0 5 c"

    for action in ("reverse", "flatten", "simplify"):
        result = DT.ass_transform_drawing(text=TWO_SQUARES, action=action)
        assert result["action"] == action
        assert result["subpath_count"] == 2  # never silently drops subpaths
        assert DT.ass_drawing_info(text=result["drawing"])["bbox"] == pytest.approx(
            [0.0, 0.0, 150.0, 150.0], abs=1.0
        )

    simplified = DT.ass_transform_drawing(
        text="m 0 0 l 0 0 l 100 0 100 0 100 100 0 100 c", action="simplify", factor=1.0
    )
    assert simplified["point_count"] < 8


def test_transform_unknown_action_raises():
    with pytest.raises(ToolError):
        DT.ass_transform_drawing(text=SQUARE, action="explode")


def test_transform_in_place_writes_back_and_preserves_clips():
    did, doc = _doc()
    _drawing_line(doc, "{\\an7\\pos(0,0)\\p1}" + SQUARE + "{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")

    result = DT.ass_transform_drawing(
        index=0, doc_id=did, action="translate", dx=10.0, dy=0.0, in_place=True
    )
    assert result["in_place"] is True
    assert result["written"] is True
    line = doc.events()[0].text
    assert line.startswith("{\\an7\\pos(0,0)\\p1}m 10 0 l 110 0 l 110 100 l 10 100 c")
    # the vector clip must survive the write-back untouched
    assert line.endswith("{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")
    assert doc.events()[0].text == result["text"]

    # the mutation went through snapshot(), so undo restores the old line
    assert workspace.undo(did) is True
    assert doc.events()[0].text.endswith("m 0 0 l 100 0 l 100 100 l 0 100 c{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")


def test_transform_in_place_false_does_not_touch_the_document():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE)
    before = doc.events()[0].text
    DT.ass_transform_drawing(index=0, doc_id=did, action="translate", dx=5.0)
    assert doc.events()[0].text == before


def test_transform_in_place_without_an_index_only_returns_text():
    # Spec: in_place only writes back when there is a line to write to; with raw
    # text there is nothing to write, so the new drawing is returned instead.
    result = DT.ass_transform_drawing(text=SQUARE, in_place=True)
    assert result["in_place"] is False
    assert result["written"] is False
    assert result["index"] is None
    assert result["drawing"] == SQUARE
    assert result["text"] is None


# ---------------------------------------------------------------------------
# split / join
# ---------------------------------------------------------------------------


def test_split_and_join_never_drop_subpaths():
    split = DT.ass_split_drawing(text=TWO_SQUARES)
    assert split["count"] == 2
    assert split["subpaths"] == [
        "m 0 0 l 10 0 10 10 0 10 c",
        "m 100 100 l 150 100 150 150 100 150 c",
    ]
    assert split["bboxes"][1] == [100.0, 100.0, 150.0, 150.0]

    joined = DT.ass_join_drawings(split["subpaths"])
    assert joined["count"] == 2
    assert joined["subpath_count"] == 2
    assert joined["bbox"] == [0.0, 0.0, 150.0, 150.0]
    assert DT.ass_join_drawings(split["subpaths"], doc_id="ignored")["doc_id"] == "ignored"

    # split(join(split(x))) is stable and loses nothing
    again = DT.ass_split_drawing(text=joined["drawing"])
    assert again["subpaths"] == split["subpaths"]


def test_split_many_subpaths_are_all_reported():
    many = " ".join(f"m {i * 10} 0 l {i * 10 + 5} 0 {i * 10 + 5} 5 {i * 10} 5 c" for i in range(6))
    split = DT.ass_split_drawing(text=many)
    assert split["count"] == 6
    assert sum(split["point_counts"]) == 24


def test_split_from_line_index():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + TWO_SQUARES + "{\\p0}")
    split = DT.ass_split_drawing(index=0, doc_id=did)
    assert split["source"] == "line"
    assert split["count"] == 2


def test_join_requires_parts():
    with pytest.raises(ToolError):
        DT.ass_join_drawings([])


# ---------------------------------------------------------------------------
# SVG interop
# ---------------------------------------------------------------------------


def test_svg_square_becomes_move_and_line_commands():
    result = DT.ass_svg_to_drawing(d="M 0 0 L 100 0 L 100 100 L 0 100 Z")
    assert result["drawing"] == "m 0 0 l 100 0 l 100 100 l 0 100 c"
    assert result["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert result["size"] == [100.0, 100.0]
    assert result["path_count"] == 1
    assert result["source"] == "d"
    _j(result)


def test_svg_scaled_conversion():
    result = DT.ass_svg_to_drawing(d="M 0 0 L 10 0 L 10 10 Z", scale=10.0)
    assert result["drawing"] == "m 0 0 l 100 0 l 100 100 c"


def test_svg_file_with_view_box_is_read(tmp_path):
    svg = tmp_path / "square.svg"
    svg.write_text(
        '<?xml version="1.0"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">\n'
        '  <path d="M 0 0 L 200 0 L 200 100 L 0 100 Z" fill="#fff"/>\n'
        "</svg>\n",
        encoding="utf-8",
    )
    result = DT.ass_svg_to_drawing(svg_path=str(svg))
    assert result["drawing"] == "m 0 0 l 200 0 l 200 100 l 0 100 c"
    assert result["path_count"] == 1
    assert result["view_box"] == [0.0, 0.0, 200.0, 100.0]
    assert result["bbox"] == [0.0, 0.0, 200.0, 100.0]


def test_svg_file_with_offset_view_box_is_translated(tmp_path):
    svg = tmp_path / "offset.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="50 50 100 100">'
        '<path d="M 50 50 L 150 50 L 150 150 L 50 150 Z"/></svg>',
        encoding="utf-8",
    )
    result = DT.ass_svg_to_drawing(svg_path=str(svg))
    assert result["bbox"] == [0.0, 0.0, 100.0, 100.0]


def test_drawing_to_svg_writes_into_output_dir(tmp_path):
    result = DT.ass_drawing_to_svg(text=SQUARE, padding=10.0)
    written = Path(result["path"])
    assert written.parent == workspace.output_dir
    assert written.exists()
    body = written.read_text(encoding="utf-8")
    assert body.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert 'viewBox="-10 -10 120 120"' in body
    assert 'd="M 0 0 L 100 0 L 100 100 L 0 100 Z"' in body
    assert result["d"] == "M 0 0 L 100 0 L 100 100 L 0 100 Z"
    assert result["view_box"] == "-10 -10 120 120"
    assert result["width"] == 120.0
    assert result["height"] == 120.0
    _j(result)


def test_drawing_to_svg_round_trips_through_svg_to_drawing(tmp_path):
    exported = DT.ass_drawing_to_svg(text=SQUARE, path="square.svg")
    back = DT.ass_svg_to_drawing(svg_path=exported["path"])
    # the exported viewBox is offset by the padding, so the geometry comes back
    # at the padded origin
    assert DT.ass_drawing_info(text=back["drawing"])["size"] == [100.0, 100.0]
    assert DT.ass_split_drawing(text=back["drawing"])["count"] == 1


# ---------------------------------------------------------------------------
# set_drawing
# ---------------------------------------------------------------------------


def test_set_drawing_replaces_the_drawing_part():
    did, doc = _doc()
    _drawing_line(doc, "{\\an7\\pos(10,20)\\p1}" + SQUARE + "{\\p0}hello")
    result = DT.ass_set_drawing(0, "m 0 0 l 50 0 50 50 0 50 c", doc_id=did)
    assert result["old_drawing"] == SQUARE
    line = doc.events()[0].text
    assert line == "{\\an7\\pos(10,20)\\p1}m 0 0 l 50 0 50 50 0 50 c{\\p0}hello"
    assert result["scale"] == 1
    assert result["keep_tags"] is True


def test_set_drawing_sets_or_prefixes_the_scale():
    did, doc = _doc()
    _drawing_line(doc, "{\\an7\\p1}" + SQUARE)
    result = DT.ass_set_drawing(0, SQUARE, doc_id=did, scale=3)
    assert result["scale"] == 3
    assert doc.events()[0].text == "{\\an7\\p3}" + SQUARE

    # a line with no tags at all gets a {\pN} block prepended
    fresh = doc.add_event(start_ms=2500, end_ms=4000, style="Default", text="")
    result = DT.ass_set_drawing(1, SQUARE, doc_id=did)
    assert result["scale"] == 1
    assert fresh.text.startswith("{\\p1}" + SQUARE)


def test_set_drawing_keep_tags_false_drops_other_tags():
    did, doc = _doc()
    _drawing_line(doc, "{\\an7\\pos(10,20)\\p1}" + SQUARE)
    result = DT.ass_set_drawing(0, SQUARE, doc_id=did, keep_tags=False)
    assert result["keep_tags"] is False
    assert doc.events()[0].text == "{\\p1}" + SQUARE


def test_set_drawing_validates_and_normalises_coordinates():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE)
    result = DT.ass_set_drawing(0, "m 0.4 0.6 l 10 0 10 10 0 10 c", doc_id=did)
    # integer ASS coordinates: 0.4 -> 0, 0.6 -> 1
    assert result["drawing"] == "m 0 1 l 10 0 10 10 0 10 c"

    with pytest.raises(ToolError):
        DT.ass_set_drawing(0, "this is not a drawing", doc_id=did)


def test_set_drawing_keeps_clips_on_the_line():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE + "{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")
    DT.ass_set_drawing(0, "m 0 0 l 5 0 5 5 0 5 c", doc_id=did)
    assert doc.events()[0].text == "{\\p1}m 0 0 l 5 0 5 5 0 5 c{\\clip(1,m 20 20 l 80 20 80 80 20 80)}"


# ---------------------------------------------------------------------------
# clips
# ---------------------------------------------------------------------------


def test_get_clips_rect_and_vector_normalised_coordinates():
    line = "{\\p1}" + SQUARE + "{\\clip(10,20,30,40)\\iclip(2,m 0 0 l 100 0 100 100 0 100)}"
    result = DT.ass_get_clips(text=line)
    assert result["source"] == "text"
    assert result["line_scale"] == 1
    assert result["count"] == 2

    rect = result["clips"][0]
    assert rect["tag"] == "clip"
    assert rect["kind"] == "rect"
    assert rect["inverse"] is False
    assert rect["raw"] == "10,20,30,40"
    assert rect["coords"] == [10.0, 20.0, 30.0, 40.0]
    assert rect["normalised"] == [10.0, 20.0, 30.0, 40.0]

    vector = result["clips"][1]
    assert vector["tag"] == "iclip"
    assert vector["kind"] == "vector"
    assert vector["inverse"] is True
    assert vector["scale"] == 2.0
    # a \p2 / \clip(2,...) coordinate covers half a script pixel
    assert vector["units_per_script_pixel"] == 0.5
    assert vector["normalised"] == [[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]
    assert vector["normalised_bbox"] == [0.0, 0.0, 50.0, 50.0]
    _j(result)


def test_get_clips_from_line_index():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE + "{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")
    result = DT.ass_get_clips(index=0, doc_id=did)
    assert result["source"] == "line"
    assert result["count"] == 1
    assert result["clips"][0]["effective_scale"] == 1.0


def test_set_clip_rect_interpreted_in_the_named_scale_space():
    did, doc = _doc(play_res=(640, 480))
    _drawing_line(doc, "{\\an7\\pos(0,0)\\p1}" + "m 0 0 l 640 0 640 480 0 480 c")
    result = DT.ass_set_clip([0], None, doc_id=did, rect="10,10,20,20", scale=2)
    assert result["kind"] == "rect"
    # \p2 units are half a script pixel: 10..20 in \p2 space is 5..10 on screen
    assert result["geometry"]["authored_coords"] == [10.0, 10.0, 20.0, 20.0]
    assert result["geometry"]["script_coords"] == [5.0, 5.0, 10.0, 10.0]
    assert result["tag"] == "\\clip(5,5,10,10)"
    assert "\\clip(5,5,10,10)" in doc.events()[0].text

    # and the rendering agrees: the ink is limited to the 5..10 box
    ink = _render_rect(doc.to_text(), 1000)
    assert ink == pytest.approx(
        {"x": 5.0, "y": 5.0, "width": 5.0, "height": 5.0}, abs=1.0
    )


def test_set_clip_vector_matches_rectangle_rendering():
    did, doc = _doc(play_res=(640, 480))
    _drawing_line(doc, "{\\an7\\pos(0,0)\\p1}m 0 0 l 640 0 640 480 0 480 c")
    DT.ass_set_clip([0], "m 100 50 l 300 50 300 150 100 150", doc_id=did)
    line = doc.events()[0].text
    assert "\\clip(1,m 100 50 l 300 50 300 150 100 150)" in line
    measured = M.measure_render(doc.to_text(), 1000)
    assert measured["rect"]["x"] == pytest.approx(100.0, abs=1.0)
    assert measured["rect"]["width"] == pytest.approx(200.0, abs=1.0)


def test_set_clip_modes_add_replace_remove_and_inverse():
    did, doc = _doc(play_res=(640, 480))
    _drawing_line(doc, "{\\an7\\pos(0,0)\\p1}m 0 0 l 640 0 640 480 0 480 c")

    DT.ass_set_clip(0, "m 100 50 l 300 50 300 150 100 150", doc_id=did)
    assert doc.events()[0].text.count("\\clip") == 1

    DT.ass_set_clip(0, "m 10 10 l 20 10 20 20 10 20", doc_id=did, mode="add")
    assert doc.events()[0].text.count("\\clip") == 2

    DT.ass_set_clip(0, None, doc_id=did, mode="replace", rect="0,0,50,50")
    assert doc.events()[0].text.count("\\clip") == 1
    assert "\\clip(0,0,50,50)" in doc.events()[0].text

    DT.ass_set_clip(0, "m 5 5 l 60 5 60 60 5 60", doc_id=did, inverse=True)
    assert "\\iclip(1,m 5 5 l 60 5 60 60 5 60)" in doc.events()[0].text

    removed = DT.ass_set_clip(0, None, doc_id=did, mode="remove")
    assert removed["lines"][0]["removed"] >= 1
    assert "clip" not in doc.events()[0].text


def test_set_clip_from_svg_path():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE)
    result = DT.ass_set_clip([0], None, doc_id=did, svg_d="M 10 10 L 60 10 L 60 60 L 10 60 Z")
    assert result["kind"] == "vector"
    assert "\\clip(1,m 10 10 l 60 10 l 60 60 l 10 60 c)" in doc.events()[0].text


def test_set_clip_requires_geometry():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE)
    with pytest.raises(ToolError):
        DT.ass_set_clip([0], None, doc_id=did)
    with pytest.raises(ToolError):
        DT.ass_set_clip([0], None, doc_id=did, rect="not,numbers")


def test_remove_clip_and_include_inverse():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE + "{\\clip(1,m 0 0 l 10 0 10 10 0 10)\\iclip(2,m 0 0 l 5 0 5 5 0 5)}")

    result = DT.ass_remove_clip([0], doc_id=did, include_inverse=False)
    assert result["changed"] == 1
    text = doc.events()[0].text
    assert "\\clip(" not in text
    assert "\\iclip(2,m 0 0 l 5 0 5 5 0 5)" in text

    DT.ass_remove_clip([0], doc_id=did)
    assert "clip" not in doc.events()[0].text
    assert doc.events()[0].text == "{\\p1}" + SQUARE


def test_remove_clip_on_clean_line_is_a_no_op():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE)
    result = DT.ass_remove_clip([0], doc_id=did)
    assert result["changed"] == 0
    assert doc.events()[0].text == "{\\p1}" + SQUARE


# ---------------------------------------------------------------------------
# clip scale conversion, verified by rendering
# ---------------------------------------------------------------------------


def test_convert_clip_scale_converges_when_rendered():
    r"""Rewriting \clip(1,..) as \clip(3,..) must not move a single pixel."""
    did, doc = _doc(play_res=(640, 480))
    line = (
        "{\\an7\\pos(0,0)\\p1}m 0 0 l 640 0 640 480 0 480 c"
        "{\\clip(1,m 100 50 l 300 50 300 150 100 150)}"
    )
    _drawing_line(doc, line)

    before_text = doc.to_text()
    before = _render_rect(before_text, 1000)
    assert before == pytest.approx(
        {"x": 100.0, "y": 50.0, "width": 200.0, "height": 100.0}, abs=1.0
    )

    result = DT.ass_convert_clip_scale([0], 3, doc_id=did)
    assert result["converted"] == 1
    assert result["applied"] is True
    entry = result["lines"][0]
    assert entry["source_scale"] == 1.0
    assert entry["target_scale"] == 3.0
    assert entry["ratio"] == 4.0  # 2**(target - source)
    assert entry["clips"][0]["after_arg"] == "3,m 400 200 l 1200 200 1200 600 400 600"

    after_text = doc.to_text()
    assert "\\clip(3,m 400 200 l 1200 200 1200 600 400 600)" in after_text
    after = _render_rect(after_text, 1000)

    # both renderings are reported by the tool itself -- and both must contain
    # ink, otherwise "converged" would be true for a trivial reason
    assert result["measurement"]["before"]["empty"] is False
    assert result["measurement"]["after"]["empty"] is False
    assert result["measurement"]["before"]["ink_pixels"] > 0
    assert result["measurement"]["after"]["ink_pixels"] == result["measurement"]["before"]["ink_pixels"]
    assert result["measurement"]["before"]["rect"]["x"] == pytest.approx(100.0, abs=1.0)
    assert result["measurement"]["after"]["rect"]["x"] == pytest.approx(100.0, abs=1.0)
    assert result["measurement"]["converged"] is True

    # the assertion the spec asks for: rendered ink is identical within tolerance
    for key in ("x", "y", "width", "height"):
        assert after[key] == pytest.approx(before[key], abs=1.0), f"{key} moved"


def test_convert_clip_scale_dry_run_leaves_the_document_alone():
    did, doc = _doc(play_res=(640, 480))
    _drawing_line(doc, "{\\an7\\pos(0,0)\\p1}m 0 0 l 640 0 640 480 0 480 c{\\clip(1,m 100 50 l 300 50 300 150 100 150)}")
    before = doc.events()[0].text
    result = DT.ass_convert_clip_scale([0], 2, doc_id=did, dry_run=True)
    assert result["dry_run"] is True
    assert result["applied"] is False
    assert doc.events()[0].text == before
    assert result["lines"][0]["clips"][0]["after_arg"] == "2,m 200 100 l 600 100 600 300 200 300"


def test_convert_clip_scale_uses_the_line_scale_when_the_clip_has_none():
    did, doc = _doc()
    _drawing_line(doc, "{\\an7\\p4}" + SQUARE + "{\\clip(m 10 10 l 20 10 20 20 10 20)}")
    result = DT.ass_convert_clip_scale([0], 1, doc_id=did)
    entry = result["lines"][0]
    assert entry["source_scale"] == 4.0
    assert entry["ratio"] == pytest.approx(0.125)  # 2**(1-4)
    # 10 * 0.125 = 1.25 -> 1 and 20 * 0.125 = 2.5 -> 3 (explicit half-away-from-zero)
    assert entry["clips"][0]["after_arg"] == "1,m 1 1 l 3 1 3 3 1 3"


def test_convert_clip_scale_ignores_lines_without_vector_clips():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE + "{\\clip(10,20,30,40)}")
    result = DT.ass_convert_clip_scale([0], 3, doc_id=did)
    assert result["converted"] == 0
    assert doc.events()[0].text.endswith("{\\clip(10,20,30,40)}")


# ---------------------------------------------------------------------------
# scale everything on the line
# ---------------------------------------------------------------------------


def test_scale_drawing_scales_drawing_and_clip_together():
    did, doc = _doc(play_res=(640, 480))
    _drawing_line(doc, "{\\an7\\pos(0,0)\\p1}" + SQUARE + "{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")

    result = DT.ass_scale_drawing([0], 2.0, doc_id=did)
    assert result["applied"] is True
    assert result["changed"] == 1
    entry = result["lines"][0]
    assert entry["drawing_scaled"] is True
    assert entry["clips_scaled"] == 1

    info = DT.ass_drawing_info(index=0, doc_id=did)
    assert info["size"] == [200.0, 200.0]
    clips = DT.ass_get_clips(index=0, doc_id=did)
    assert len(clips["clips"]) == 1  # the clip was scaled, not dropped
    # Everything scales about the one common origin -- the drawing's bbox centre
    # (50, 50): the clip's 20..80 box doubles to -10..110, keeping its centre.
    assert clips["clips"][0]["normalised_bbox"] == pytest.approx([-10.0, -10.0, 110.0, 110.0], abs=0.5)


def test_scale_drawing_dry_run_and_clip_opt_out():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE + "{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")
    before = doc.events()[0].text

    dry = DT.ass_scale_drawing([0], 3.0, doc_id=did, dry_run=True)
    assert dry["applied"] is False
    assert doc.events()[0].text == before
    # the dry run reports the *proposed* text: drawing and clip both tripled
    # about the drawing's centre (50, 50), i.e. 20..80 -> -40..140
    assert "m -100 -100 l 200 -100 l 200 200 l -100 200 c" in dry["lines"][0]["text"]
    assert "\\clip(1,m -40 -40 l 140 -40 140 140 -40 140)" in dry["lines"][0]["text"]

    no_clips = DT.ass_scale_drawing([0], 3.0, doc_id=did, include_clips=False)
    assert no_clips["lines"][0]["clips_scaled"] == 0
    assert "\\clip(1,m 20 20 l 80 20 80 80 20 80)" in doc.events()[0].text


def test_scale_drawing_rejects_bad_factor():
    did, doc = _doc()
    _drawing_line(doc, "{\\p1}" + SQUARE)
    with pytest.raises(ToolError):
        DT.ass_scale_drawing([0], 0, doc_id=did)
    with pytest.raises(ToolError):
        DT.ass_scale_drawing([0], -2, doc_id=did)


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------


def test_list_fonts_filters_case_insensitively():
    family = M.default_sans_family()
    needle = family.split()[0]
    result = DT.ass_list_fonts(needle.lower(), limit=5)
    assert result["pattern"] == needle.lower()
    assert 0 < result["count"] <= 5
    for font in result["fonts"]:
        assert needle.lower() in font["family"].lower()
        assert Path(font["file"]).exists()
    assert DT.ass_list_fonts(None, limit=1)["count"] == 1
    _j(result)


def test_match_font_reports_installed_and_substituted():
    installed = DT.ass_match_font(M.default_sans_family())
    assert installed["substituted"] is False
    assert installed["resolved"]
    assert Path(installed["file"]).exists()

    missing = DT.ass_match_font("Definitely Not An Installed Family 12345")
    assert missing["substituted"] is True
    assert missing["resolved"] != "Definitely Not An Installed Family 12345"
    assert len(missing["candidates"]) >= 1

    bold = DT.ass_match_font(M.default_sans_family(), bold=True)
    assert "bold" in bold["match"]["style"].lower() or "bold" in bold["resolved"].lower()


def test_fonts_with_char_finds_families():
    result = DT.ass_fonts_with_char("A")
    assert result["char"] == "A"
    assert result["codepoint"] == 65
    assert result["codepoint_hex"] == "U+0041"
    assert result["count"] > 0
    assert "DejaVu Sans" in result["families"]

    with pytest.raises(ToolError):
        DT.ass_fonts_with_char("")
    with pytest.raises(ToolError):
        DT.ass_fonts_with_char("AB")


def test_font_coverage_reports_missing_characters_and_codepoints():
    family = M.default_sans_family()
    text = "AB\u0e2a\u0e27"  # Thai letters are absent from DejaVu Sans
    expected = M.glyph_check(text, family)
    assert expected["missing"], "fixture assumption: DejaVu Sans has no Thai glyphs"

    result = DT.ass_font_coverage(text, family=family)
    assert result["requested"] == family
    assert result["missing_count"] == len(expected["missing"])
    assert result["missing_chars"] == list(expected["missing"])
    assert result["missing_codepoints"] == [ord(c) for c in expected["missing"]]
    for item in result["missing"]:
        assert ord(item["char"]) == item["codepoint"]
        assert item["codepoint_hex"] == f"U+{item['codepoint']:04X}"
        assert isinstance(item["codepoint"], int)
    assert "A" not in result["missing_chars"]
    assert "B" not in result["missing_chars"]
    _j(result)


def test_font_coverage_defaults_to_the_default_sans_family():
    result = DT.ass_font_coverage("hello")
    assert result["requested"] == M.default_sans_family()
    assert result["missing_count"] == 0


def test_glyph_check_for_a_line_and_for_a_style():
    did, doc = _doc()
    doc.add_style({
        "Name": "Thai", "Fontname": "Definitely Not An Installed Family 12345",
        "Fontsize": 48, "Bold": 0, "Italic": 0,
        "PrimaryColour": "&H00FFFFFF", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H00000000", "BackColour": "&H00000000",
        "BorderStyle": 1, "Outline": 2, "Shadow": 0, "Alignment": 2,
        "MarginL": 10, "MarginR": 10, "MarginV": 10, "Encoding": 1,
        "ScaleX": 100, "ScaleY": 100, "Spacing": 0, "Angle": 0,
    })
    _drawing_line(doc, "hello \u0e2a\u0e27", style="Thai")

    by_style = DT.ass_glyph_check(index=0, doc_id=did, style="Thai")
    assert by_style["style"] == "Thai"
    assert by_style["style_info"]["family"] == "Definitely Not An Installed Family 12345"
    assert by_style["substituted"] is True
    assert by_style["missing_chars"] == ["\u0e2a", "\u0e27"]
    assert by_style["missing_codepoints"] == [3626, 3623]
    assert by_style["text"] == "hello \u0e2a\u0e27"

    by_family = DT.ass_glyph_check(text="AB\u0e2a", family=by_style["resolved_family"])
    assert by_family["missing_chars"] == ["\u0e2a"]
    assert by_family["source"] == "text"

    from_line = DT.ass_glyph_check(index=0, doc_id=did)
    assert from_line["source"] == "line"
    assert from_line["style_info"]["family"] == "Definitely Not An Installed Family 12345"

    clean = DT.ass_glyph_check(index=0, doc_id=did, family=M.default_sans_family())
    assert clean["requested"] == M.default_sans_family()
    # every default sans covers ASCII, whatever the Thai glyph situation is
    latin = DT.ass_glyph_check(text="hello", family=M.default_sans_family())
    assert latin["missing_count"] == 0
    assert latin["source"] == "text"


def test_glyph_check_unknown_style_raises():
    did, doc = _doc()
    _drawing_line(doc, "hello")
    with pytest.raises(ToolError):
        DT.ass_glyph_check(index=0, doc_id=did, style="NoSuchStyle")


def test_fonts_used_flags_missing_and_substituted_families():
    did, doc = _doc()
    doc.add_style({
        "Name": "Good", "Fontname": M.default_sans_family(),
        "Fontsize": 48, "Bold": 0, "Italic": 0,
        "PrimaryColour": "&H00FFFFFF", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H00000000", "BackColour": "&H00000000",
        "BorderStyle": 1, "Outline": 2, "Shadow": 0, "Alignment": 2,
        "MarginL": 10, "MarginR": 10, "MarginV": 10, "Encoding": 1,
        "ScaleX": 100, "ScaleY": 100, "Spacing": 0, "Angle": 0,
    })
    doc.add_style({
        "Name": "Bad", "Fontname": "Definitely Not An Installed Family 12345",
        "Fontsize": 48, "Bold": -1, "Italic": 0,
        "PrimaryColour": "&H00FFFFFF", "SecondaryColour": "&H000000FF",
        "OutlineColour": "&H00000000", "BackColour": "&H00000000",
        "BorderStyle": 1, "Outline": 2, "Shadow": 0, "Alignment": 2,
        "MarginL": 10, "MarginR": 10, "MarginV": 10, "Encoding": 1,
        "ScaleX": 100, "ScaleY": 100, "Spacing": 0, "Angle": 0,
    })

    result = DT.ass_fonts_used(doc_id=did)
    by_family = {f["family"]: f for f in result["families"]}
    assert M.default_sans_family() in by_family
    assert by_family[M.default_sans_family()]["installed"] is True
    assert by_family[M.default_sans_family()]["styles"] == ["Good"]

    bad = by_family["Definitely Not An Installed Family 12345"]
    assert bad["installed"] is False
    assert bad["substituted"] is True
    assert bad["styles"] == ["Bad"]
    assert bad["bold"] is True
    assert "Definitely Not An Installed Family 12345" in result["not_installed"]
    assert "Definitely Not An Installed Family 12345" in result["not_installed_style_families"]
    assert "Definitely Not An Installed Family 12345" in result["substituted"]
    assert M.default_sans_family() not in result["not_installed"]
    _j(result)


def test_fonts_used_lists_embedded_attachments():
    did = workspace.open(str(LEGACY_FONTS), doc_id="legacy")
    result = DT.ass_fonts_used(doc_id=did)
    assert "Sample_0.ttf" in result["attachments"]
    embedded = [f for f in result["families"] if f["embedded"]]
    assert len(embedded) == 1
    assert embedded[0]["attachment_file"] == "Sample_0.ttf"
    # an embedded font is not a substitution problem
    assert embedded[0]["substituted"] is False
    assert embedded[0]["family"] not in result["substituted"]


# ---------------------------------------------------------------------------
# registration + JSON safety
# ---------------------------------------------------------------------------


def test_register_registers_every_tool_once():
    class FakeMCP:
        def __init__(self):
            self.registered = []

        def tool(self):
            def decorator(fn):
                self.registered.append(fn.__name__)
                return fn
            return decorator

    mcp = FakeMCP()
    names = DT.register(mcp, workspace)
    assert names == sorted(ALL_TOOLS)
    assert sorted(mcp.registered) == sorted(ALL_TOOLS)
    assert len(mcp.registered) == len(set(mcp.registered)) == 20
    for name in names:
        assert callable(getattr(DT, name))


def test_all_tool_results_are_json_serialisable(tmp_path):
    did, doc = _doc(play_res=(640, 480))
    _drawing_line(doc, "{\\an7\\pos(0,0)\\p1}" + SQUARE + "{\\clip(1,m 20 20 l 80 20 80 80 20 80)}")
    results = [
        DT.ass_get_drawing(index=0, doc_id=did),
        DT.ass_drawing_info(index=0, doc_id=did),
        DT.ass_drawing_bbox(text=SQUARE),
        DT.ass_drawing_to_svg(text=SQUARE, padding=2.0),
        DT.ass_svg_to_drawing(d="M 0 0 L 10 0 L 10 10 Z"),
        DT.ass_split_drawing(index=0, doc_id=did),
        DT.ass_join_drawings(DT.ass_split_drawing(index=0, doc_id=did)["subpaths"]),
        DT.ass_get_clips(index=0, doc_id=did),
        DT.ass_list_fonts(M.default_sans_family().split()[0], limit=2),
        DT.ass_match_font(M.default_sans_family()),
        DT.ass_fonts_with_char("A"),
        DT.ass_font_coverage("hello"),
        DT.ass_glyph_check(text="hello"),
        DT.ass_fonts_used(doc_id=did),
        DT.ass_transform_drawing(text=SQUARE, action="rotate", angle_deg=30.0),
    ]
    for payload in results:
        assert isinstance(payload, dict)
        assert json.loads(json.dumps(payload)) == payload
