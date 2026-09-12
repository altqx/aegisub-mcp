"""Tests for :mod:`aegisub_mcp.asscore.drawing`.

Run with::

    cd /home/krapau/Work/aegisub-mcp && .venv/bin/python -m pytest tests/test_drawing.py -q

Focused, deterministic coverage of the public API: parsing/serialisation,
bounding boxes, affine maths, structure operations, clip override forms and the
SVG bridge (including arc conversion accuracy).
"""

from __future__ import annotations

import math

import pytest

from aegisub_mcp.asscore import drawing as D

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _cmd(kind: str, *points: tuple[float, float]):
    return D.Command(kind, list(points))


def _square(size: float = 100.0) -> D.Drawing:
    return D.Drawing(
        [
            _cmd("m", (0, 0)),
            _cmd("l", (size, 0), (size, size), (0, size)),
            _cmd("c"),
        ]
    )


def _polylines(drawing: D.Drawing, steps: int = 64) -> list[list[tuple[float, float]]]:
    """Flatten ``drawing`` into absolute polylines (one list per subpath)."""
    out: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    pen = None
    for cmd in D.flatten(drawing, steps).commands:
        if cmd.kind in ("m", "n"):
            if current:
                out.append(current)
            current = [cmd.points[0]]
            pen = cmd.points[0]
        elif cmd.kind == "l":
            if pen is not None and (not current or current[-1] != pen):
                current.append(pen)
            current.extend(cmd.points)
            pen = cmd.points[-1]
        elif cmd.kind == "c":
            if current or pen:
                out.append(current)
                current = []
    if current:
        out.append(current)
    return out


def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _flat(points) -> list[float]:
    """Flatten a point sequence to ``[x0, y0, x1, y1, ...]`` for approx checks."""
    return [value for point in points for value in point]


def _dist_to_polyline(p, poly) -> float:
    if len(poly) == 1:
        return _dist(p, poly[0])
    best = float("inf")
    for a, b in zip(poly, poly[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            best = min(best, _dist(p, a))
            continue
        t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq))
        best = min(best, math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy)))
    return best


def _cubic_point(p0, c1, c2, p1, t: float):
    mt = 1 - t
    a, b, c, d = mt**3, 3 * mt**2 * t, 3 * mt * t**2, t**3
    return (
        a * p0[0] + b * c1[0] + c * c2[0] + d * p1[0],
        a * p0[1] + b * c1[1] + c * c2[1] + d * p1[1],
    )


def _assert_points_close(actual, expected, abs_tol=1e-6):
    assert len(actual) == len(expected), f"{actual} != {expected}"
    for (ax, ay), (ex, ey) in zip(actual, expected):
        assert ax == pytest.approx(ex, abs=abs_tol)
        assert ay == pytest.approx(ey, abs=abs_tol)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


def test_drawing_text_canonical_example():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("l", (100, 0), (100, 100)), _cmd("c")])
    assert drawing.text() == "m 0 0 l 100 0 100 100 c"


def test_command_uppercase_kind_is_normalised():
    assert _cmd("M", (1, 2)).kind == "m"
    assert _cmd("C").kind == "c"
    assert D.Command("S", [(1, 1), (2, 2), (3, 3)]).kind == "s"


def test_command_rejects_unknown_kind():
    with pytest.raises(ValueError):
        D.Command("x", [(0, 0)])
    with pytest.raises(ValueError):
        D.Command("", [])


def test_command_arity_rules():
    with pytest.raises(ValueError):
        D.Command("b", [(0, 0), (1, 1)])  # needs exactly 3
    with pytest.raises(ValueError):
        D.Command("b", [(0, 0), (1, 1), (2, 2), (3, 3)])
    with pytest.raises(ValueError):
        D.Command("l", [])  # needs at least 1
    with pytest.raises(ValueError):
        D.Command("s", [(0, 0), (1, 1)])  # needs at least 3
    with pytest.raises(ValueError):
        D.Command("m", [(0, 0), (1, 1)])  # exactly 1
    with pytest.raises(ValueError):
        D.Command("c", [(0, 0)])  # none
    assert _cmd("s", (0, 0), (1, 1), (2, 2), (3, 3)).kind == "s"


def test_text_rounds_and_trims_numbers():
    drawing = D.Drawing([_cmd("m", (0.0, -0.0)), _cmd("l", (100.456, 0.5)), _cmd("c")])
    assert drawing.text() == "m 0 0 l 100.46 0.5 c"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parse_basic_and_roundtrip():
    text = "m 0 0 l 100 0 100 100 c"
    drawing = D.parse_drawing(text)
    assert drawing.commands == [
        D.Command("m", [(0, 0)]),
        D.Command("l", [(100, 0), (100, 100)]),
        D.Command("c", []),
    ]
    assert drawing.text() == text


def test_parse_tolerates_whitespace_and_newlines():
    text = "  m\t0   0\n\n  l 10 0\r\n   20 20\nc "
    drawing = D.parse_drawing(text)
    assert drawing.text() == "m 0 0 l 10 0 20 20 c"


def test_parse_accepts_leading_dot_and_negative_numbers():
    drawing = D.parse_drawing("m -.5 .5 l -3.25 1e1")
    assert drawing.commands[0].points == [(-0.5, 0.5)]
    assert drawing.commands[1].points == [(-3.25, 10.0)]
    assert drawing.text() == "m -0.5 0.5 l -3.25 10"


def test_parse_multi_subpath_and_close_without_numbers():
    drawing = D.parse_drawing("m 0 0 l 10 0 10 10 c m 20 20 b 30 20 30 30 20 30 c")
    kinds = [c.kind for c in drawing.commands]
    assert kinds == ["m", "l", "c", "m", "b", "c"]


def test_parse_bezier_forms_b_s_p():
    drawing = D.parse_drawing("m 0 0 b 1 1 2 2 3 3 s 4 4 5 5 6 6 p 7 7 8 8 9 9 10 10")
    assert [c.kind for c in drawing.commands] == ["m", "b", "s", "p"]
    assert drawing.commands[1].points == [(1, 1), (2, 2), (3, 3)]
    assert len(drawing.commands[2].points) == 3
    assert len(drawing.commands[3].points) == 4


def test_parse_multiple_moveto_pairs_split_into_commands():
    drawing = D.parse_drawing("m 0 0 10 10 l 20 20")
    assert [c.kind for c in drawing.commands] == ["m", "m", "l"]


def test_parse_uppercase_letters_are_accepted():
    # Uppercase is folded to the lowercase command: 'C' is a close, not a bezier.
    assert D.parse_drawing("M 0 0 L 10 0 10 10 C 20 20").text() == "m 0 0 l 10 0 10 10 c"
    assert D.parse_drawing("M 0 0 B 1 1 2 2 3 3").text() == "m 0 0 b 1 1 2 2 3 3"


def test_parse_without_command_letter_raises():
    with pytest.raises(ValueError):
        D.parse_drawing("")
    with pytest.raises(ValueError):
        D.parse_drawing("1 2 3 4")
    with pytest.raises(ValueError):
        D.parse_drawing("   ,,, ")


def test_parse_ignores_malformed_trailing_numbers():
    assert D.parse_drawing("m 0 0 l 10 10 20").text() == "m 0 0 l 10 10"
    # Unknown letters simply terminate the running command.
    assert D.parse_drawing("m 0 0 l 10 10 z 99 99").text() == "m 0 0 l 10 10"
    # 'b' keeps only its first three points; too-short splines are dropped.
    assert D.parse_drawing("m 0 0 b 1 1 2 2 3 3 4 4").text() == "m 0 0 b 1 1 2 2 3 3"
    assert D.parse_drawing("m 0 0 s 1 1 2 2").text() == "m 0 0"
    assert D.parse_drawing("m 0 0 l").text() == "m 0 0"


def test_parse_realistic_aegisub_drawing():
    text = (
        "m 0 0 l 34 0 35 3 35 23 36 26 37 27 38 27 39 26 40 23 40 3 41 0 75 0 "
        "b 72 12 66 15 60 15 l 54 15 48 12 45 0 c"
    )
    drawing = D.parse_drawing(text)
    assert drawing.text() == text
    assert drawing.text() == D.parse_drawing(drawing.text()).text()


# --------------------------------------------------------------------------
# Bounding boxes
# --------------------------------------------------------------------------


def test_bbox_uses_all_control_points_including_n():
    drawing = D.Drawing(
        [
            _cmd("m", (10, 20)),
            _cmd("b", (-30, 5), (40, 90), (50, 60)),
            _cmd("n", (100, -7)),
            _cmd("l", (0, 0)),
        ]
    )
    assert D.bbox(drawing) == (-30.0, -7.0, 100.0, 90.0)


def test_bbox_of_points_and_empty_cases():
    assert D.bbox_of_points([(1, 2), (3, 0)]) == (1.0, 0.0, 3.0, 2.0)
    assert D.bbox_of_points([]) is None
    assert D.bbox(D.Drawing()) is None
    assert D.bbox(D.Drawing([_cmd("c")])) is None


def test_bbox_union_and_intersect():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 20.0, 5.0)
    assert D.bbox_union([a, b, None]) == (0.0, 0.0, 20.0, 10.0)
    assert D.bbox_union([]) is None
    assert D.bbox_intersect(a, b) == (5.0, 5.0, 10.0, 5.0)
    assert D.bbox_intersect(a, (20.0, 20.0, 30.0, 30.0)) is None
    assert D.bbox_intersect(None, a) is None


def test_bbox_contains_size_center_expand():
    outer = (0.0, 0.0, 100.0, 50.0)
    inner = (10.0, 10.0, 20.0, 20.0)
    assert D.bbox_contains(outer, inner) is True
    assert D.bbox_contains(inner, outer) is False
    assert D.bbox_contains(outer, (10.0, 10.0, 200.0, 20.0)) is False
    assert D.bbox_contains(None, inner) is False
    assert D.bbox_size(outer) == (100.0, 50.0)
    assert D.bbox_size(None) == (0.0, 0.0)
    assert D.bbox_center(outer) == (50.0, 25.0)
    assert D.bbox_center(None) == (0.0, 0.0)
    assert D.bbox_expand(outer, 5) == (-5.0, -5.0, 105.0, 55.0)
    assert D.bbox_expand(None, 5) is None


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


def test_transform_matrix_maths():
    drawing = D.Drawing([_cmd("m", (1, 1)), _cmd("l", (2, 3))])
    out = D.transform(drawing, (2.0, 0.0, 0.0, 3.0, 10.0, 20.0))
    assert out.commands[0].points == [(12.0, 23.0)]
    assert out.commands[1].points == [(14.0, 29.0)]


def test_transform_svg_axis_convention():
    # x' = a*x + c*y + e ; y' = b*x + d*y + f  ->  shear keeps y, shifts x by y.
    out = D.transform(D.Drawing([_cmd("m", (2, 5))]), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0))
    assert out.commands[0].points == [(7.0, 5.0)]
    # Control points are transformed, so command structure survives untouched.
    assert [c.kind for c in D.transform(_square(), (2, 0, 0, 2, 0, 0)).commands] == ["m", "l", "c"]


def test_translate_moves_every_point():
    out = D.translate(D.Drawing([_cmd("m", (1, 2)), _cmd("b", (3, 4), (5, 6), (7, 8))]), -1, 10)
    assert out.commands[0].points == [(0.0, 12.0)]
    assert out.commands[1].points == [(2.0, 14.0), (4.0, 16.0), (6.0, 18.0)]
    assert D.bbox(out) == (0.0, 12.0, 6.0, 18.0)


def test_scale_uniform_and_about_origin():
    drawing = D.Drawing([_cmd("m", (1, 1)), _cmd("l", (2, 3))])
    assert D.scale(drawing, 2).commands[1].points == [(4.0, 6.0)]
    assert D.scale(drawing, 2, 3).commands[1].points == [(4.0, 9.0)]
    about = D.scale(drawing, 2, 3, origin=(1.0, 1.0))
    assert about.commands[0].points == [(1.0, 1.0)]  # origin is a fixed point
    assert about.commands[1].points == [(3.0, 7.0)]


def test_rotate_positive_is_clockwise_in_screen_coordinates():
    drawing = D.Drawing([_cmd("m", (1, 0))])
    # ASS y grows downwards, so +90 degrees takes (1,0) to (0,1): clockwise on screen.
    assert _flat(D.rotate(drawing, 90).commands[0].points) == pytest.approx([0.0, 1.0], abs=1e-9)
    assert _flat(D.rotate(drawing, 180).commands[0].points) == pytest.approx([-1.0, 0.0], abs=1e-9)
    # Same 90 degree turn about a non-trivial origin.
    assert _flat(D.rotate(drawing, 90, origin=(5, 0)).commands[0].points) == pytest.approx(
        [5.0, -4.0], abs=1e-9
    )


def test_rotate_origin_point_is_fixed_and_length_preserved():
    drawing = D.Drawing([_cmd("m", (10, 5)), _cmd("l", (30, 25))])
    out = D.rotate(drawing, 37, origin=(10, 5))
    assert _flat(out.commands[0].points) == pytest.approx([10.0, 5.0], abs=1e-9)
    assert _dist(out.commands[1].points[0], (10, 5)) == pytest.approx(
        _dist((30, 25), (10, 5)), abs=1e-9
    )


def test_mirror_horizontal_vertical_and_origin():
    drawing = D.Drawing([_cmd("m", (3, 2))])
    assert D.mirror(drawing, horizontal=True).commands[0].points == [(-3.0, 2.0)]
    assert D.mirror(drawing, horizontal=False, vertical=True).commands[0].points == [(3.0, -2.0)]
    assert D.mirror(drawing, horizontal=True, vertical=True).commands[0].points == [(-3.0, -2.0)]
    mirrored = D.mirror(drawing, horizontal=True, origin=(10.0, 10.0))
    assert mirrored.commands[0].points == [(17.0, 2.0)]


def test_offset_to_origin_sets_bbox_minimum():
    drawing = D.Drawing([_cmd("m", (10, 20)), _cmd("l", (30, 40)), _cmd("c")])
    out = D.offset_to_origin(drawing)
    assert D.bbox(out) == (0.0, 0.0, 20.0, 20.0)
    shifted = D.offset_to_origin(drawing, target=(5.0, 6.0))
    assert D.bbox(shifted) == (5.0, 6.0, 25.0, 26.0)
    assert D.offset_to_origin(D.Drawing()).commands == []


# --------------------------------------------------------------------------
# stretch_drawing
# --------------------------------------------------------------------------


def test_stretch_without_keep_aspect_fills_target_exactly():
    drawing = _square(50.0)
    out = D.stretch_drawing(drawing, 200, 100)
    assert D.bbox(out) == (0.0, 0.0, 200.0, 100.0)


def test_stretch_keep_aspect_centers_inside_target():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("l", (50, 100)), _cmd("c")])
    out = D.stretch_drawing(drawing, 200, 200, keep_aspect=True)
    # min(200/50, 200/100) = 2 -> 100x200, centred horizontally.
    assert D.bbox(out) == (50.0, 0.0, 150.0, 200.0)


def test_stretch_keep_aspect_align_variants():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("l", (50, 100)), _cmd("c")])
    # scale = min(300/50, 700/100) = 6 -> 300x600, i.e. 100 units of vertical slack.
    assert D.bbox(D.stretch_drawing(drawing, 300, 700, keep_aspect=True)) == (
        0.0,
        50.0,
        300.0,
        650.0,
    )
    top_right = D.stretch_drawing(drawing, 300, 700, keep_aspect=True, align="topright")
    assert D.bbox(top_right) == (0.0, 0.0, 300.0, 600.0)
    bottom_left = D.stretch_drawing(drawing, 300, 700, keep_aspect=True, align="bottom-left")
    assert D.bbox(bottom_left) == (0.0, 100.0, 300.0, 700.0)
    with pytest.raises(ValueError):
        D.stretch_drawing(drawing, 10, 10, keep_aspect=True, align="sideways")


def test_stretch_empty_drawing_is_a_copy():
    empty = D.Drawing([_cmd("c")])
    out = D.stretch_drawing(empty, 100, 100)
    assert out.commands == empty.commands
    assert out is not empty


# --------------------------------------------------------------------------
# Structure: split / flatten / simplify / reverse / round / length
# --------------------------------------------------------------------------


def test_split_subpaths_starts_each_group_with_m_or_n():
    drawing = D.Drawing(
        [
            _cmd("m", (0, 0)),
            _cmd("l", (10, 0)),
            _cmd("m", (20, 20)),
            _cmd("b", (1, 1), (2, 2), (3, 3)),
            _cmd("c"),
            _cmd("n", (40, 40)),
            _cmd("l", (50, 50)),
        ]
    )
    parts = D.split_subpaths(drawing)
    assert len(parts) == 3
    assert [p.commands[0].kind for p in parts] == ["m", "m", "n"]
    assert parts[1].text() == "m 20 20 b 1 1 2 2 3 3 c"
    assert sum(len(p.commands) for p in parts) == len(drawing.commands)


def test_flatten_bezier_hits_analytic_points():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("b", (0, 100), (100, 100), (100, 0))])
    flat = D.flatten(drawing, bezier_steps=2)
    assert [c.kind for c in flat.commands] == ["m", "l"]
    # t=0.5 -> (50, 75); t=1 -> the end point exactly.
    assert flat.commands[1].points == pytest.approx([(50.0, 75.0), (100.0, 0.0)], abs=1e-9)


def test_flatten_tracks_pen_and_is_deterministic():
    drawing = D.Drawing(
        [
            _cmd("m", (10, 20)),
            _cmd("b", (20, 20), (30, 30), (40, 20)),
            _cmd("l", (50, 20)),
        ]
    )
    flat = D.flatten(drawing, bezier_steps=4)
    assert len(flat.commands[1].points) == 4
    assert flat.commands[1].points[-1] == pytest.approx((40.0, 20.0))
    assert flat.text() == D.flatten(drawing, 4).text()


def test_flatten_bspline_starts_and_ends_on_control_points():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("s", (10, 0), (20, 10), (30, 0))])
    flat = D.flatten(drawing, bezier_steps=8)
    points = flat.commands[1].points
    assert points[-1] == pytest.approx((30.0, 0.0), abs=1e-9)
    assert _dist(points[0], (0.0, 0.0)) < 1.0  # first sample is just off the start
    # Extended spline 'p' behaves like 's' in this implementation.
    p_drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("p", (10, 0), (20, 10), (30, 0))])
    assert D.flatten(p_drawing, 8).text() == flat.text()


def test_flatten_preserves_other_commands():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("n", (5, 5)), _cmd("l", (6, 6)), _cmd("c")])
    assert D.flatten(drawing).commands == drawing.commands


def test_simplify_reduces_point_count_and_stays_within_tolerance():
    drawing = D.Drawing(
        [
            _cmd("m", (0, 0)),
            _cmd("l", (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 5)),
        ]
    )
    original = list(drawing.commands[1].points)
    simplified = D.simplify(drawing, tolerance=0.5)
    reduced = list(simplified.commands[1].points)
    assert len(reduced) < len(original)
    assert reduced == [(5.0, 0.0), (5.0, 5.0)]
    # Every original vertex stays within tolerance of the retained polyline.
    poly = [(0.0, 0.0)] + reduced
    for point in original:
        assert _dist_to_polyline(point, poly) <= 0.5 + 1e-9


def test_simplify_reduces_noisy_bezier_within_tolerance():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("b", (0, 100), (100, 100), (100, 0))])
    flat = D.flatten(drawing, bezier_steps=32)
    simplified = D.simplify(drawing, tolerance=0.5)
    assert len(simplified.commands[1].points) < len(flat.commands[1].points)
    poly = [(0.0, 0.0)] + simplified.commands[1].points
    for point in flat.commands[1].points:
        assert _dist_to_polyline(point, poly) <= 0.5 + 1e-6


def test_simplify_preserves_subpaths_and_close():
    drawing = D.Drawing(
        [
            _cmd("m", (0, 0)),
            _cmd("l", (10, 0), (20, 0)),
            _cmd("c"),
            _cmd("m", (50, 50)),
            _cmd("l", (60, 50), (70, 50)),
            _cmd("c"),
        ]
    )
    out = D.simplify(drawing, tolerance=0.1)
    assert [c.kind for c in out.commands] == ["m", "l", "c", "m", "l", "c"]
    assert [c.kind for c in D.split_subpaths(out)[1].commands] == ["m", "l", "c"]


def test_reverse_polyline_is_exact():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("l", (10, 0), (10, 10))])
    out = D.reverse(drawing)
    assert out.text() == "m 10 10 l 10 0 0 0"


def test_reverse_bezier_keeps_geometry_and_length():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("b", (10, 0), (20, 10), (20, 20))])
    out = D.reverse(drawing)
    assert out.text() == "m 20 20 b 20 10 10 0 0 0"
    forward = _polylines(drawing)[0]
    backward = _polylines(out)[0]
    assert backward == pytest.approx(list(reversed(forward)), abs=1e-6)
    assert D.path_length(out) == pytest.approx(D.path_length(drawing), rel=1e-9)


def test_reverse_keeps_close_and_subpaths():
    drawing = D.Drawing(
        [
            _cmd("m", (0, 0)),
            _cmd("l", (10, 0), (10, 10)),
            _cmd("c"),
            _cmd("m", (20, 20)),
            _cmd("l", (30, 20)),
        ]
    )
    out = D.reverse(drawing)
    text = out.text()
    assert text.startswith("m 10 10 l 10 0 0 0 c")
    assert text.endswith("m 30 20 l 20 20")


def test_reverse_empty_and_closeless_subpaths():
    assert D.reverse(D.Drawing()).commands == []
    single = D.Drawing([_cmd("m", (5, 5))])
    assert D.reverse(single).text() == "m 5 5"


def test_round_drawing_rounds_coordinates():
    drawing = D.Drawing([_cmd("m", (0.1234, -0.9876)), _cmd("l", (100.005, 100.004))])
    out = D.round_drawing(drawing, digits=2)
    assert out.commands[0].points == [(0.12, -0.99)]
    assert out.commands[1].points == [(100.0, 100.0)]
    assert D.round_drawing(drawing, 1).commands[0].points == [(0.1, -1.0)]


def test_path_length_closed_square_and_open_polyline():
    assert D.path_length(_square(100.0)) == pytest.approx(400.0)
    open_line = D.Drawing([_cmd("m", (0, 0)), _cmd("l", (3, 0), (3, 4))])
    assert D.path_length(open_line) == pytest.approx(7.0)


def test_path_length_of_bezier_and_jumps():
    curve = D.Drawing([_cmd("m", (0, 0)), _cmd("b", (0, 100), (100, 100), (100, 0))])
    assert D.path_length(curve, bezier_steps=64) == pytest.approx(200.0, rel=0.02)
    # 'm'/'n' jumps contribute no length.
    two = D.Drawing([_cmd("m", (0, 0)), _cmd("l", (10, 0)), _cmd("n", (500, 500)), _cmd("l", (510, 500))])
    assert D.path_length(two) == pytest.approx(20.0)


# --------------------------------------------------------------------------
# Clip forms
# --------------------------------------------------------------------------


def test_parse_clip_rect_plain_and_scaled():
    plain = D.parse_clip_arg("10,20,300,400")
    assert plain["type"] == "rect"
    assert plain["scale"] is None
    assert plain["coords"] == [10.0, 20.0, 300.0, 400.0]
    assert plain["drawing"] is None
    assert plain["raw"] == "10,20,300,400"

    scaled = D.parse_clip_arg("0.5,10,20,300,400")
    assert scaled["type"] == "rect"
    assert scaled["scale"] == 0.5
    assert scaled["coords"] == [10.0, 20.0, 300.0, 400.0]


def test_parse_clip_drawing_with_and_without_scale():
    with_scale = D.parse_clip_arg("1,m 0 0 l 100 0 100 100")
    assert with_scale["type"] == "drawing"
    assert with_scale["scale"] == 1.0
    assert with_scale["coords"] is None
    assert with_scale["drawing"].text() == "m 0 0 l 100 0 100 100"

    without = D.parse_clip_arg("m 0 0 l 100 0 100 100")
    assert without["scale"] is None
    assert without["drawing"].text() == "m 0 0 l 100 0 100 100"


def test_parse_clip_whitespace_newlines_and_parentheses():
    spec = D.parse_clip_arg("\n  ( 1,\n   m 0 0\n   l 10 0 10 10  c )\n")
    assert spec["type"] == "drawing"
    assert spec["scale"] == 1.0
    assert spec["drawing"].text() == "m 0 0 l 10 0 10 10 c"
    rect = D.parse_clip_arg("(1, 2, 3, 4)")
    assert rect["coords"] == [1.0, 2.0, 3.0, 4.0]


def test_parse_clip_invalid_arguments_raise():
    with pytest.raises(ValueError):
        D.parse_clip_arg("")
    with pytest.raises(ValueError):
        D.parse_clip_arg("1,2,3")  # wrong rect arity
    with pytest.raises(ValueError):
        D.parse_clip_arg("1,2,x,4")
    with pytest.raises(ValueError):
        D.parse_clip_arg("1,2,3,m 0 0 l 1 1")  # numbers before drawing


def test_format_clip_arg_roundtrips_both_types():
    for arg in ("10,20,300,400", "0.5,10,20,300,400", "1,m 0 0 l 100 0 100 100", "m 0 0 l 5 5 c"):
        spec = D.parse_clip_arg(arg)
        text = D.format_clip_arg(spec)
        reparsed = D.parse_clip_arg(text)
        assert reparsed["type"] == spec["type"]
        assert reparsed["scale"] == spec["scale"]
        assert reparsed["coords"] == spec["coords"]
        if spec["drawing"] is not None:
            assert reparsed["drawing"].text() == spec["drawing"].text()
    assert D.format_clip_arg(D.parse_clip_arg("10,20,300,400")) == "10,20,300,400"
    assert D.format_clip_arg(D.parse_clip_arg("1,m 0 0 l 100 0 100 100")) == "1,m 0 0 l 100 0 100 100"


def test_format_clip_arg_rejects_bad_specs():
    with pytest.raises(ValueError):
        D.format_clip_arg({"type": "rect", "coords": [1, 2]})
    with pytest.raises(ValueError):
        D.format_clip_arg({"type": "nope"})


def test_drawing_to_clip_full_tags():
    drawing = _square(100.0)
    assert D.drawing_to_clip(drawing) == "\\clip(1,m 0 0 l 100 0 100 100 0 100 c)"
    assert D.drawing_to_clip(drawing, inverse=True) == "\\iclip(1,m 0 0 l 100 0 100 100 0 100 c)"
    assert D.drawing_to_clip("m 0 0 l 100 0 100 100", scale=1.0) == "\\clip(1,m 0 0 l 100 0 100 100)"
    assert D.drawing_to_clip("m 0 0 l 100 0 100 100", scale=None) == "\\clip(m 0 0 l 100 0 100 100)"
    assert D.drawing_to_clip("m 0 0 l 1 0", scale=0.5) == "\\clip(0.5,m 0 0 l 1 0)"


def test_rect_to_clip_tags_and_roundtrip():
    assert D.rect_to_clip(10, 20, 300, 400) == "\\clip(10,20,300,400)"
    assert D.rect_to_clip(10, 20, 300, 400, inverse=True) == "\\iclip(10,20,300,400)"
    assert D.rect_to_clip(10, 20, 300, 400, scale=0.5) == "\\clip(0.5,10,20,300,400)"
    spec = D.parse_clip_arg(D.rect_to_clip(1, 2, 3, 4)[len("\\clip(") : -1])
    assert spec["coords"] == [1.0, 2.0, 3.0, 4.0]


def test_clip_roundtrip_through_drawing_to_clip():
    spec = D.parse_clip_arg(D.drawing_to_clip("m 0 0 l 100 0 100 100 c", scale=2)[6:-1])
    assert spec["scale"] == 2.0
    assert spec["drawing"].text() == "m 0 0 l 100 0 100 100 c"


# --------------------------------------------------------------------------
# SVG: serialisation
# --------------------------------------------------------------------------


def test_to_svg_path_basic_commands():
    assert D.to_svg_path(_square(100.0)) == "M 0 0 L 100 0 100 100 0 100 Z"


def test_to_svg_path_n_closes_previous_subpath():
    drawing = D.Drawing(
        [
            _cmd("m", (0, 0)),
            _cmd("l", (10, 0)),
            _cmd("n", (20, 20)),
            _cmd("l", (30, 20)),
            _cmd("c"),
        ]
    )
    assert D.to_svg_path(drawing) == "M 0 0 L 10 0 Z M 20 20 L 30 20 Z"


def test_to_svg_path_empty_drawing():
    assert D.to_svg_path(D.Drawing()) == ""
    assert D.to_svg_path(D.Drawing([_cmd("m", (1, 2))])) == "M 1 2"


def test_to_svg_path_emits_spline_as_polyline():
    drawing = D.Drawing([_cmd("m", (0, 0)), _cmd("s", (10, 0), (20, 10), (30, 0))])
    svg = D.to_svg_path(drawing)
    assert svg.startswith("M 0 0 L ")
    assert "C" not in svg


# --------------------------------------------------------------------------
# SVG: parsing
# --------------------------------------------------------------------------


def test_from_svg_cubic_roundtrip_is_exact():
    path = "M 0 0 C 10 0 20 10 20 20"
    drawing = D.from_svg_path(path)
    assert drawing.text() == "m 0 0 b 10 0 20 10 20 20"
    assert D.to_svg_path(drawing) == path


def test_from_svg_quadratic_uses_standard_control_rule():
    drawing = D.from_svg_path("M 0 0 Q 10 0 20 20")
    assert drawing.text() == "m 0 0 b 6.67 0 13.33 6.67 20 20"
    exact = D.from_svg_path("M 0 0 Q 10 0 20 20")
    assert exact.commands[1].points[0] == pytest.approx((20.0 / 3.0, 0.0), abs=1e-9)
    assert exact.commands[1].points[1] == pytest.approx((40.0 / 3.0, 20.0 / 3.0), abs=1e-9)
    # Round-tripping through SVG text preserves the curve shape.
    again = D.from_svg_path(D.to_svg_path(exact))
    _assert_points_close(again.commands[1].points, exact.commands[1].points, abs_tol=1e-3)


def test_from_svg_quadratic_roundtrip_accuracy():
    path = "M 0 0 Q 50 100 100 0"
    original = D.from_svg_path(path)
    reserialised = D.from_svg_path(D.to_svg_path(original))
    _assert_points_close(reserialised.commands[1].points, original.commands[1].points, abs_tol=1e-3)


def test_from_svg_relative_commands_and_hv():
    drawing = D.from_svg_path("m 10 10 h 80 v 80 h -80 z")
    assert drawing.text() == "m 10 10 l 90 10 l 90 90 l 10 90 c"


def test_from_svg_lowercase_cubic_and_implicit_repeat():
    drawing = D.from_svg_path("M 0 0 c 10 0 20 10 20 20 10 0 20 10 20 20")
    assert drawing.text() == "m 0 0 b 10 0 20 10 20 20 b 30 20 40 30 40 40"


def test_from_svg_smooth_s_reflects_control_point():
    drawing = D.from_svg_path("M 0 0 C 0 10 10 10 10 0 S 20 -10 20 0")
    assert drawing.commands[2].points == pytest.approx(
        [(10.0, -10.0), (20.0, -10.0), (20.0, 0.0)], abs=1e-9
    )


def test_from_svg_smooth_s_without_previous_curve_uses_current_point():
    drawing = D.from_svg_path("M 0 0 S 10 10 20 0")
    assert drawing.commands[1].points == pytest.approx([(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)])


def test_from_svg_smooth_t_reflects_quadratic_control():
    drawing = D.from_svg_path("M 0 0 Q 5 10 10 0 T 20 0")
    c1, c2, end = drawing.commands[2].points
    assert c1 == pytest.approx((40.0 / 3.0, -20.0 / 3.0), abs=1e-9)
    assert c2 == pytest.approx((50.0 / 3.0, -20.0 / 3.0), abs=1e-9)
    assert end == pytest.approx((20.0, 0.0))


def test_from_svg_arc_accuracy_90_degrees():
    drawing = D.from_svg_path("M 1 0 A 1 1 0 0 1 0 1")
    cubics = [(c.points[0], c.points[1], c.points[2]) for c in drawing.commands[1:]]
    assert len(cubics) == 1
    start = drawing.commands[0].points[0]
    worst = 0.0
    for c1, c2, end in cubics:
        for i in range(41):
            point = _cubic_point(start, c1, c2, end, i / 40)
            worst = max(worst, abs(math.hypot(*point) - 1.0))
        start = end
    assert worst < 0.01  # under 1 % of the radius
    assert cubics[-1][2] == pytest.approx((0.0, 1.0), abs=1e-9)


def test_from_svg_arc_accuracy_full_circle_and_large_flag():
    # Two 180-degree halves traced with opposite sweep flags form a full circle.
    drawing = D.from_svg_path("M 1 0 A 1 1 0 0 1 -1 0 A 1 1 0 0 1 1 0")
    start = drawing.commands[0].points[0]
    segments = [(c.points[0], c.points[1], c.points[2]) for c in drawing.commands[1:]]
    assert len(segments) == 4  # 180 degrees -> two 90-degree cubics each
    worst = 0.0
    for c1, c2, end in segments:
        for i in range(41):
            point = _cubic_point(start, c1, c2, end, i / 40)
            worst = max(worst, abs(math.hypot(*point) - 1.0))
        start = end
    assert worst < 0.01
    assert D.path_length(drawing, bezier_steps=64) == pytest.approx(2 * math.pi, rel=0.01)


def test_from_svg_arc_large_arc_flag_changes_route():
    small = D.from_svg_path("M 1 0 A 1 1 0 0 1 0 -1")
    large = D.from_svg_path("M 1 0 A 1 1 0 1 1 0 -1")
    assert D.path_length(small, 64) < D.path_length(large, 64)
    assert D.path_length(large, 64) > 3.0


def test_from_svg_arc_degenerate_becomes_line():
    drawing = D.from_svg_path("M 0 0 A 0 5 0 0 1 10 0")
    assert drawing.text() == "m 0 0 l 10 0"
    coincident = D.from_svg_path("M 5 5 A 5 5 0 0 1 5 5")
    assert coincident.text() == "m 5 5"


def test_from_svg_rotated_arc_stays_on_ellipse():
    drawing = D.from_svg_path("M 3 0 A 3 1 45 0 1 0 1")
    box = D.bbox(drawing)
    assert box is not None
    # All control points stay near the rotated ellipse's bounding region.
    assert box[2] - box[0] < 8.0
    assert box[3] - box[1] < 8.0


def test_from_svg_scale_translate_and_flip_y():
    scaled = D.from_svg_path("M 0 0 L 10 20", scale=2.0)
    assert scaled.commands[1].points == [(20.0, 40.0)]
    moved = D.from_svg_path("M 0 0 L 10 20", scale=2.0, translate=(5.0, 7.0))
    assert moved.commands[1].points == [(25.0, 47.0)]
    flipped = D.from_svg_path("M 0 10 L 20 30", flip_y=True)
    assert flipped.commands[0].points == [(0.0, -10.0)]
    assert flipped.commands[1].points == [(20.0, -30.0)]
    flipped_moved = D.from_svg_path("M 0 10 L 20 30", flip_y=True, translate=(5.0, 100.0))
    assert flipped_moved.commands[1].points == [(25.0, 70.0)]


def test_from_svg_unparsable_raises_with_offset():
    with pytest.raises(ValueError) as excinfo:
        D.from_svg_path("M 0 0 L")
    assert "offset" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        D.from_svg_path("X 1 2")
    assert "offset" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        D.from_svg_path("10 10")
    assert "offset" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        D.from_svg_path("M 0 0 Z 5 5")
    assert "offset" in str(excinfo.value)
    with pytest.raises(ValueError):
        D.from_svg_path(None)


# --------------------------------------------------------------------------
# SVG -> ASS typesetting entry point
# --------------------------------------------------------------------------


def test_svg_path_to_ass_contain_fit_viewbox_to_1080p():
    path = "M 0 0 L 100 0 L 100 100 L 0 100 Z"
    drawing = D.svg_path_to_ass_drawing(path, source_size=(100, 100), target_size=(1920, 1080))
    # min(19.2, 10.8) = 10.8 -> 1080x1080 centred horizontally (dx = 420).
    assert drawing.text() == "m 420 0 l 1500 0 l 1500 1080 l 420 1080 c"
    assert D.bbox(drawing) == (420.0, 0.0, 1500.0, 1080.0)


def test_svg_path_to_ass_contain_centers_both_axes():
    drawing = D.svg_path_to_ass_drawing(
        "M 0 0 L 200 0 L 200 50 L 0 50 Z", source_size=(200, 50), target_size=(100, 100)
    )
    # s = min(0.5, 2) = 0.5 -> 100x25 centred vertically at dy = 37.5.
    assert D.bbox(drawing) == (0.0, 37.5, 100.0, 62.5)


def test_svg_path_to_ass_stretch_fills_target():
    drawing = D.svg_path_to_ass_drawing(
        "M 0 0 L 100 0 L 100 200 L 0 200 Z", source_size=(100, 200), target_size=(400, 400), fit="stretch"
    )
    assert D.bbox(drawing) == (0.0, 0.0, 400.0, 400.0)


def test_svg_path_to_ass_pad_and_offset():
    drawing = D.svg_path_to_ass_drawing(
        "M 0 0 L 100 0 L 100 100 L 0 100 Z",
        source_size=(100, 100),
        target_size=(1920, 1080),
        pad=10,
        offset=(5, 7),
    )
    # usable 1900x1060 -> s = 10.6, dx = 10 + 420 = 430, dy = 10 (+ offset 5,7).
    assert D.bbox(drawing) == (435.0, 17.0, 1495.0, 1077.0)
    with pytest.raises(ValueError):
        D.svg_path_to_ass_drawing(
            "M 0 0 L 1 0", source_size=(10, 10), target_size=(10, 10), pad=5
        )


def test_svg_path_to_ass_fit_none_and_missing_sizes():
    path = "M 0 0 L 10 20"
    assert D.svg_path_to_ass_drawing(path, fit="none").text() == "m 0 0 l 10 20"
    assert D.svg_path_to_ass_drawing(path, fit="none", offset=(3, 4)).text() == "m 3 4 l 13 24"
    # No sizes given -> only the offset is applied.
    assert D.svg_path_to_ass_drawing(path, offset=(1, 2)).text() == "m 1 2 l 11 22"
    with pytest.raises(ValueError):
        D.svg_path_to_ass_drawing(path, source_size=(1, 1), target_size=(1, 1), fit="squash")


def test_svg_path_to_ass_flip_y_and_scaling_pipeline():
    # A y-up 10x10 viewBox dropped into a 100x100 target, flipped to ASS's y-down.
    drawing = D.svg_path_to_ass_drawing(
        "M 0 0 L 10 0 L 10 10 Z", source_size=(10, 10), target_size=(100, 100), flip_y=True
    )
    assert D.bbox(drawing) == (0.0, 0.0, 100.0, 100.0)
