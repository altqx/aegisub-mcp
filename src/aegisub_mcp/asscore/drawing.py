"""ASS vector drawings: command model, geometry, clip forms and the SVG bridge.

An ASS drawing (the ``\\p`` "drawing" mode of a subtitle event) is a tiny
vector language::

    m 0 0 l 100 0 100 100 c

This module is the single place where drawings are parsed, measured,
transformed, converted to/from SVG path data and wrapped in clip overrides.
It is dependency-free (stdlib + :mod:`.assutil`) and deterministic.

Coordinate system
-----------------
ASS lives in *screen* coordinates: ``x`` grows to the right, ``y`` grows
**downwards**, the origin is the top-left of the script resolution.  SVG uses
the same (y-down) convention, which is why the SVG bridge is nearly symmetric;
``flip_y`` exists only for sources that use a mathematical (y-up) axis.

Point semantics
---------------
Every drawing command except ``c`` carries control points.  For ``b``/``s``/``p``
the curve *starts at the current pen position* (the previous command's last
point) and ends at the command's last point; this is why :func:`flatten` and
:func:`reverse` have to track a pen.  ``m`` starts a new subpath, ``n`` moves
the pen without drawing, ``l`` appends a polyline, ``b`` is a cubic Bézier,
``s`` a cubic B-spline, ``p`` the "extended" B-spline variant, ``c`` closes the
current subpath.

Deliberate approximations
-------------------------
* ``s``/``p`` are sampled as clamped *uniform* cubic B-splines (see
  :func:`flatten`); VSFilter's exact terminal-tangent handling for ``p`` is not
  reproduced.
* :func:`reverse` is exact for ``l`` and ``b``; reversing ``s``/``p`` reverses
  the control point order and is therefore approximate.
* :func:`to_svg_path` emits ``s``/``p`` as sampled ``L`` polylines because SVG
  has no uniform B-spline primitive.
* Elliptical SVG arcs are converted to cubic Béziers with at most 90° per
  segment (the classic ``4/3·tan(θ/4)`` handle length, max radial error ≈ 0.03 %).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import assutil

__all__ = [
    # data model
    "Command",
    "Drawing",
    "parse_drawing",
    # bbox helpers
    "bbox",
    "bbox_of_points",
    "bbox_union",
    "bbox_intersect",
    "bbox_contains",
    "bbox_size",
    "bbox_center",
    "bbox_expand",
    # transforms
    "transform",
    "translate",
    "scale",
    "rotate",
    "mirror",
    "offset_to_origin",
    "stretch_drawing",
    # structure / geometry
    "split_subpaths",
    "flatten",
    "simplify",
    "reverse",
    "round_drawing",
    "path_length",
    # clip forms
    "parse_clip_arg",
    "format_clip_arg",
    "drawing_to_clip",
    "rect_to_clip",
    # svg
    "to_svg_path",
    "from_svg_path",
    "svg_path_to_ass_drawing",
]

#: Degrees of ASS precision used when serialising drawing coordinates.
COORD_PRECISION = 2

#: Default number of subdivisions used per curve segment when flattening.
DEFAULT_BEZIER_STEPS = 16

#: Decimals used when emitting SVG path data (a touch more than ASS needs so
#: arc handles survive a round-trip).
SVG_PRECISION = 4

#: The seven ASS drawing commands, in canonical lowercase form.
DRAWING_KINDS = ("m", "n", "l", "b", "s", "p", "c")

#: Minimum / maximum number of points per command (``None`` = unbounded).
_ARITY: dict[str, tuple[int, int | None]] = {
    "m": (1, 1),
    "n": (1, 1),
    "l": (1, None),
    "b": (3, 3),
    "s": (3, None),
    "p": (3, None),
    "c": (0, 0),
}

# A number as accepted by ASS/Aegisub: ``0``, ``12``, ``.5``, ``-3.25``, ``1e3``.
_NUM_PATTERN = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(_NUM_PATTERN)
# A command letter or a number, in order; everything else is a separator.
_TOKEN_RE = re.compile(r"([A-Za-z])|(" + _NUM_PATTERN + r")")


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Command:
    """A single drawing command.

    ``kind`` is one of ``'m'``, ``'n'``, ``'l'``, ``'b'``, ``'s'``, ``'p'``,
    ``'c'`` (case-insensitive on input, always stored lowercase).  ``points`` is
    a list of ``(x, y)`` float tuples whose permitted length depends on the
    kind (``b`` needs exactly 3, ``s``/``p`` need 3 or more, ``l`` 1 or more,
    ``m``/``n`` exactly 1, ``c`` none).
    """

    kind: str
    points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        kind = self.kind.lower() if isinstance(self.kind, str) else self.kind
        if kind not in _ARITY:
            raise ValueError(
                f"unknown drawing command {self.kind!r}; expected one of {''.join(DRAWING_KINDS)}"
            )
        self.kind = kind
        points: list[tuple[float, float]] = []
        for pt in self.points:
            try:
                x, y = pt
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"point {pt!r} is not an (x, y) pair") from exc
            points.append((float(x), float(y)))
        low, high = _ARITY[kind]
        if len(points) < low or (high is not None and len(points) > high):
            if high is None:
                expected = f"{low} or more"
            elif low == high:
                expected = str(low)
            else:
                expected = f"{low}..{high}"
            raise ValueError(
                f"command {kind!r} takes {expected} point(s), got {len(points)}"
            )
        self.points = points

    def copy(self) -> "Command":
        return Command(self.kind, list(self.points))

    def text(self) -> str:
        """Serialise this command to canonical ASS form (no trailing space)."""
        if not self.points:
            return self.kind
        nums = " ".join(assutil.round_coord(v, COORD_PRECISION) for pt in self.points for v in pt)
        return f"{self.kind} {nums}"


@dataclass
class Drawing:
    """An ordered list of :class:`Command` making up one ASS drawing."""

    commands: list[Command] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.commands = [c if isinstance(c, Command) else Command(*c) for c in self.commands]

    def text(self) -> str:
        """Canonical ASS serialisation: ``m 0 0 l 100 0 100 100 c``.

        Numbers are formatted with :func:`assutil.round_coord` (2 decimals,
        integers without a fractional part), commands separated by a single
        space, no leading or trailing whitespace.
        """
        return " ".join(cmd.text() for cmd in self.commands)

    def copy(self) -> "Drawing":
        return Drawing([c.copy() for c in self.commands])

    def points(self) -> list[tuple[float, float]]:
        """Every control point in command order (including ``n``)."""
        return [p for cmd in self.commands for p in cmd.points]

    def is_empty(self) -> bool:
        return not self.commands or not self.points()


# --------------------------------------------------------------------------
# Parsing / serialisation
# --------------------------------------------------------------------------


def _commands_from_numbers(kind: str, nums: Sequence[float]) -> list[Command]:
    """Build commands for one letter out of the numbers that followed it.

    Tolerance rules: a trailing unpaired number is dropped, ``b`` keeps only its
    first three points, ``s``/``p``/``l`` need their minimum arity or are
    discarded.
    """
    if kind == "c":
        return [Command("c", [])]
    pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    if not pairs:
        return []
    if kind in ("m", "n"):
        return [Command(kind, [p]) for p in pairs]
    if kind == "l":
        return [Command("l", pairs)]
    if kind == "b":
        return [Command("b", pairs[:3])] if len(pairs) >= 3 else []
    if kind in ("s", "p"):
        return [Command(kind, pairs)] if len(pairs) >= 3 else []
    return []  # pragma: no cover - unreachable, kind is validated earlier


def parse_drawing(text: str) -> Drawing:
    """Parse ASS drawing text into a :class:`Drawing`.

    Tolerant by design, matching what real Aegisub files contain:

    * arbitrary whitespace, tabs and newlines between tokens;
    * numbers such as ``.5``, ``-3.25``, ``1e3``;
    * repeated ``m`` subpaths, ``c`` with or without trailing numbers;
    * unparsable trailing numbers or extra points are ignored rather than
      raising (``l 10 10 20`` drops the odd ``20``).

    The only hard failure is input that contains no drawing command letter at
    all (including the empty string): that raises :class:`ValueError`.
    """
    if text is None:
        raise ValueError("drawing text must be a string, got None")
    src = str(text)
    commands: list[Command] = []
    kind: str | None = None
    nums: list[float] = []
    saw_command = False

    def flush() -> None:
        nonlocal nums
        pending, nums = nums, []
        if kind is None:
            return
        if kind == "c":
            commands.append(Command("c", []))
        elif pending:
            commands.extend(_commands_from_numbers(kind, pending))

    for match in _TOKEN_RE.finditer(src):
        letter, number = match.group(1), match.group(2)
        if letter is not None:
            flush()
            lowered = letter.lower()
            if lowered in _ARITY:
                kind = lowered
                saw_command = True
            else:
                # Unknown letters only terminate the running command.
                kind = None
        elif kind is not None:
            nums.append(float(number))
    flush()

    if not saw_command:
        raise ValueError(f"no drawing command letter found in {text!r}")
    return Drawing(commands)


# --------------------------------------------------------------------------
# Bounding boxes
# --------------------------------------------------------------------------


def bbox_of_points(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    """Axis-aligned bbox ``(x1, y1, x2, y2)`` of points, or ``None`` if empty."""
    xs: list[float] = []
    ys: list[float] = []
    for x, y in points:
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def bbox(drawing: Drawing) -> tuple[float, float, float, float] | None:
    """Bounding box of a drawing, or ``None`` when it has no points.

    Every control point counts — including Bézier/B-spline handles and ``n``
    moves — because that is what VSFilter/Aegisub use when positioning a
    drawing (the box of the *control polygon*, not of the curve).
    """
    return bbox_of_points(drawing.points())


def bbox_union(
    bboxes: Iterable[tuple[float, float, float, float] | None],
) -> tuple[float, float, float, float] | None:
    """Smallest box containing all boxes; ``None`` entries are ignored."""
    found = [b for b in bboxes if b is not None]
    if not found:
        return None
    return (
        min(b[0] for b in found),
        min(b[1] for b in found),
        max(b[2] for b in found),
        max(b[3] for b in found),
    )


def bbox_intersect(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Overlap of two boxes, or ``None`` when they are disjoint / either is None.

    Touching edges (zero-area overlap) still count as an intersection.
    """
    if a is None or b is None:
        return None
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x1 > x2 or y1 > y2:
        return None
    return (x1, y1, x2, y2)


def bbox_contains(
    outer: tuple[float, float, float, float] | None,
    inner: tuple[float, float, float, float] | None,
) -> bool:
    """True when ``inner`` lies fully inside ``outer`` (either None -> False)."""
    if outer is None or inner is None:
        return False
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def bbox_size(bbox: tuple[float, float, float, float] | None) -> tuple[float, float]:
    """``(width, height)``; an empty/None box is ``(0.0, 0.0)``."""
    if bbox is None:
        return (0.0, 0.0)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def bbox_center(bbox: tuple[float, float, float, float] | None) -> tuple[float, float]:
    """Centre point; an empty/None box gives ``(0.0, 0.0)``."""
    if bbox is None:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_expand(
    bbox: tuple[float, float, float, float] | None, pad: float
) -> tuple[float, float, float, float] | None:
    """Grow (or shrink, with negative ``pad``) every side; ``None`` stays ``None``."""
    if bbox is None:
        return None
    pad = float(pad)
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


# --------------------------------------------------------------------------
# Affine transforms
# --------------------------------------------------------------------------


def transform(
    drawing: Drawing, matrix: tuple[float, float, float, float, float, float]
) -> Drawing:
    """Apply an affine matrix to every control point.

    ``matrix`` is the SVG/PostScript 6-tuple ``(a, b, c, d, e, f)`` used in the
    same order as SVG's ``matrix(a b c d e f)``::

        x' = a*x + c*y + e
        y' = b*x + d*y + f

    Control points are transformed, so Béziers and B-splines keep their shape
    exactly (affine maps commute with the curve evaluation).
    """
    a, b, c, d, e, f = (float(v) for v in matrix)
    return Drawing(
        [
            Command(cmd.kind, [(a * x + c * y + e, b * x + d * y + f) for x, y in cmd.points])
            for cmd in drawing.commands
        ]
    )


def translate(drawing: Drawing, dx: float, dy: float) -> Drawing:
    """Shift the drawing by ``(dx, dy)``."""
    return transform(drawing, (1.0, 0.0, 0.0, 1.0, float(dx), float(dy)))


def scale(
    drawing: Drawing,
    sx: float,
    sy: float | None = None,
    origin: tuple[float, float] | None = None,
) -> Drawing:
    """Scale about ``origin`` (default ``(0, 0)``); ``sy`` defaults to ``sx``."""
    sx = float(sx)
    sy = sx if sy is None else float(sy)
    ox, oy = (0.0, 0.0) if origin is None else (float(origin[0]), float(origin[1]))
    return transform(drawing, (sx, 0.0, 0.0, sy, ox - sx * ox, oy - sy * oy))


def rotate(
    drawing: Drawing,
    degrees: float,
    origin: tuple[float, float] | None = None,
) -> Drawing:
    """Rotate about ``origin`` (default ``(0, 0)``).

    **Convention:** positive angles rotate *clockwise on screen*.  Because ASS
    (like SVG) has ``y`` growing downwards, the SVG matrix
    ``(cos, sin, -sin, cos)`` looks clockwise to the viewer: rotating ``(1, 0)``
    by ``+90`` yields ``(0, 1)``, i.e. straight down.
    """
    ox, oy = (0.0, 0.0) if origin is None else (float(origin[0]), float(origin[1]))
    rad = math.radians(float(degrees))
    cos, sin = math.cos(rad), math.sin(rad)
    a, b, c, d = cos, sin, -sin, cos
    e = ox - (a * ox + c * oy)
    f = oy - (b * ox + d * oy)
    return transform(drawing, (a, b, c, d, e, f))


def mirror(
    drawing: Drawing,
    horizontal: bool = True,
    vertical: bool = False,
    origin: tuple[float, float] | None = None,
) -> Drawing:
    """Mirror about ``origin`` (default ``(0, 0)``).

    ``horizontal=True`` flips left↔right (``x -> -x``), ``vertical=True`` flips
    top↔bottom (``y -> -y``); both together are a 180° rotation.  In ASS screen
    coordinates a "horizontal mirror" therefore flips the *x* axis.
    """
    sx = -1.0 if horizontal else 1.0
    sy = -1.0 if vertical else 1.0
    return scale(drawing, sx, sy, origin)


def offset_to_origin(drawing: Drawing, target: tuple[float, float] = (0.0, 0.0)) -> Drawing:
    """Translate so the drawing's bbox minimum lands on ``target``.

    An empty drawing (no points) is returned unchanged.
    """
    box = bbox(drawing)
    if box is None:
        return drawing.copy()
    tx, ty = (float(target[0]), float(target[1]))
    dx, dy = tx - box[0], ty - box[1]
    if dx == 0.0 and dy == 0.0:
        return drawing.copy()
    return translate(drawing, dx, dy)


def _resolve_align(align: str) -> tuple[str, str]:
    """Normalise an align keyword into ``(horizontal, vertical)`` axis anchors."""
    key = str(align or "center").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "center": ("center", "center"),
        "centre": ("center", "center"),
        "middle": ("center", "center"),
        "c": ("center", "center"),
        "left": ("min", "center"),
        "west": ("min", "center"),
        "w": ("min", "center"),
        "right": ("max", "center"),
        "east": ("max", "center"),
        "e": ("max", "center"),
        "top": ("center", "min"),
        "north": ("center", "min"),
        "n": ("center", "min"),
        "bottom": ("center", "max"),
        "south": ("center", "max"),
        "s": ("center", "max"),
        "topleft": ("min", "min"),
        "northwest": ("min", "min"),
        "nw": ("min", "min"),
        "topright": ("max", "min"),
        "northeast": ("max", "min"),
        "ne": ("max", "min"),
        "bottomleft": ("min", "max"),
        "southwest": ("min", "max"),
        "sw": ("min", "max"),
        "bottomright": ("max", "max"),
        "southeast": ("max", "max"),
        "se": ("max", "max"),
    }
    if key not in aliases:
        raise ValueError(f"unknown align {align!r}")
    return aliases[key]


def stretch_drawing(
    drawing: Drawing,
    target_w: float,
    target_h: float,
    keep_aspect: bool = False,
    align: str = "center",
) -> Drawing:
    """Fit the drawing into ``(0, 0, target_w, target_h)``.

    Without ``keep_aspect`` the result bbox is exactly
    ``(0, 0, target_w, target_h)``.  With ``keep_aspect=True`` the drawing is
    scaled uniformly (the limiting axis wins) and placed inside that box
    according to ``align`` (``'center'``, ``'left'``, ``'topright'``,
    ``'bottom'``, ... — see :func:`_resolve_align`).

    Degenerate source boxes (zero width and/or height, e.g. a single vertical
    line) keep a factor of ``1.0`` on that axis instead of dividing by zero.
    """
    box = bbox(drawing)
    if box is None:
        return drawing.copy()
    src_w, src_h = bbox_size(box)
    tw, th = float(target_w), float(target_h)
    horizontal, vertical = _resolve_align(align)

    if not keep_aspect:
        sx = tw / src_w if src_w else 1.0
        sy = th / src_h if src_h else 1.0
        dx = 0.0
        dy = 0.0
    else:
        ratios = [r for r in ((tw / src_w if src_w else None), (th / src_h if src_h else None)) if r]
        s = min(ratios) if ratios else 1.0
        sx = sy = s
        scaled_w, scaled_h = src_w * sx, src_h * sy
        if horizontal == "center":
            dx = (tw - scaled_w) / 2.0
        elif horizontal == "max":
            dx = tw - scaled_w
        else:
            dx = 0.0
        if vertical == "center":
            dy = (th - scaled_h) / 2.0
        elif vertical == "max":
            dy = th - scaled_h
        else:
            dy = 0.0

    return transform(drawing, (sx, 0.0, 0.0, sy, dx - sx * box[0], dy - sy * box[1]))


# --------------------------------------------------------------------------
# Subpaths and curve evaluation
# --------------------------------------------------------------------------


def split_subpaths(drawing: Drawing) -> list[Drawing]:
    """Split a drawing at every ``m``/``n``, each result starting with one.

    Commands before the first ``m``/``n`` (unusual, but they happen in broken
    files) form their own leading group and keep their original letters.
    """
    groups: list[list[Command]] = []
    current: list[Command] = []
    for cmd in drawing.commands:
        if cmd.kind in ("m", "n") and current:
            groups.append(current)
            current = []
        current.append(cmd)
    if current:
        groups.append(current)
    return [Drawing(list(g)) for g in groups]


def _cubic_point(p0, c1, c2, p1, t: float) -> tuple[float, float]:
    """Point on the cubic Bézier ``p0 -> c1 -> c2 -> p1`` at ``t``."""
    mt = 1.0 - t
    a = mt * mt * mt
    b = 3.0 * mt * mt * t
    c = 3.0 * mt * t * t
    d = t * t * t
    return (
        a * p0[0] + b * c1[0] + c * c2[0] + d * p1[0],
        a * p0[1] + b * c1[1] + c * c2[1] + d * p1[1],
    )


def sample_cubic(p0, c1, c2, p1, steps: int = DEFAULT_BEZIER_STEPS) -> list[tuple[float, float]]:
    """Uniform samples of a cubic Bézier, excluding ``p0`` and including ``p1``."""
    steps = max(1, int(steps))
    return [_cubic_point(p0, c1, c2, p1, i / steps) for i in range(1, steps + 1)]


def _bspline_point(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    """Uniform cubic B-spline basis evaluation for one segment."""
    t2 = t * t
    t3 = t2 * t
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * t3 - 6.0 * t2 + 4.0
    b2 = -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0
    b3 = t3
    return (
        (b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]) / 6.0,
        (b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]) / 6.0,
    )


def sample_bspline(
    start: tuple[float, float],
    points: Sequence[tuple[float, float]],
    steps: int = DEFAULT_BEZIER_STEPS,
) -> list[tuple[float, float]]:
    """Sample an ASS ``s``/``p`` spline as a polyline.

    The control polygon is clamped at both ends by tripling the first and last
    control point, which makes the curve start exactly at ``start`` (the pen
    position carried over from the previous command) and end exactly at
    ``points[-1]`` — the behaviour ASS users expect, and the reason ``l``
    reversal and bbox tests are well-defined.  Samples exclude ``start`` and
    include the final point.
    """
    steps = max(1, int(steps))
    if not points:
        return []
    control = [start, start, start] + list(points) + [points[-1], points[-1]]
    out: list[tuple[float, float]] = []
    for i in range(len(control) - 3):
        p0, p1, p2, p3 = control[i], control[i + 1], control[i + 2], control[i + 3]
        for j in range(1, steps + 1):
            out.append(_bspline_point(p0, p1, p2, p3, j / steps))
    return out


def flatten(drawing: Drawing, bezier_steps: int = DEFAULT_BEZIER_STEPS) -> Drawing:
    """Convert every curve command into plain ``l`` polylines.

    ``b`` is sampled exactly (``bezier_steps`` subdivisions per curve).  ``s``
    and ``p`` are sampled as a uniform cubic B-spline with clamped end control
    points (see :func:`sample_bspline`) — an accepted approximation of
    VSFilter's spline handling; ``p``'s extra terminal tangent controls are
    treated exactly like ``s``.  ``m``/``n``/``l``/``c`` survive unchanged, and
    the pen position is tracked across commands (including the reset to the
    subpath start after ``c``).
    """
    steps = max(1, int(bezier_steps))
    out: list[Command] = []
    pen: tuple[float, float] | None = None
    subpath_start: tuple[float, float] | None = None

    for cmd in drawing.commands:
        kind = cmd.kind
        if kind in ("m", "n"):
            out.append(cmd.copy())
            pen = cmd.points[0]
            if kind == "m":
                subpath_start = pen
        elif kind == "l":
            out.append(cmd.copy())
            pen = cmd.points[-1]
        elif kind == "c":
            out.append(cmd.copy())
            pen = subpath_start
        elif kind == "b":
            start = pen if pen is not None else cmd.points[0]
            c1, c2, end = cmd.points
            out.append(Command("l", sample_cubic(start, c1, c2, end, steps)))
            pen = end
        elif kind in ("s", "p"):
            start = pen if pen is not None else cmd.points[0]
            out.append(Command("l", sample_bspline(start, cmd.points, steps)))
            pen = cmd.points[-1]
    return Drawing(out)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _point_segment_distance(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return _dist(p, a)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _douglas_peucker(points: list[tuple[float, float]], tolerance: float):
    """Douglas-Peucker decimation; first and last points are always kept."""
    if len(points) < 3 or tolerance <= 0.0:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        worst = 0.0
        index = i0
        for i in range(i0 + 1, i1):
            d = _point_segment_distance(points[i], points[i0], points[i1])
            if d > worst:
                worst = d
                index = i
        if worst > tolerance:
            keep[index] = True
            stack.append((i0, index))
            stack.append((index, i1))
    return [p for p, kept in zip(points, keep) if kept]


def simplify(drawing: Drawing, tolerance: float = 0.5) -> Drawing:
    """Douglas-Peucker simplification of a (curve) drawing.

    The drawing is :func:`flatten`-ed first, so curves become polylines; the
    result only contains ``m``/``n``/``l``/``c``.  Subpath structure and ``c``
    are preserved, and every dropped vertex is guaranteed to lie within
    ``tolerance`` of the retained polyline.
    """
    flat = flatten(drawing)
    tol = max(0.0, float(tolerance))
    out: list[Command] = []
    pen: tuple[float, float] | None = None
    for cmd in flat.commands:
        if cmd.kind in ("m", "n"):
            out.append(cmd.copy())
            pen = cmd.points[0]
        elif cmd.kind == "c":
            out.append(cmd.copy())
        elif cmd.kind == "l":
            start = pen if pen is not None else cmd.points[0]
            kept = _douglas_peucker([start] + list(cmd.points), tol)
            new_points = kept[1:] or [cmd.points[-1]]
            out.append(Command("l", new_points))
            pen = new_points[-1]
        else:  # pragma: no cover - flatten never leaves curves behind
            out.append(cmd.copy())
    return Drawing(out)


def _reverse_subpath(cmds: list[Command]) -> list[Command]:
    """Reverse one subpath's drawing order (see :func:`reverse`)."""
    if not cmds:
        return []
    closed = cmds[-1].kind == "c"
    head = cmds[0]
    if head.kind not in ("m", "n"):
        # No explicit start point: reversing would be guesswork, keep as-is.
        return [c.copy() for c in cmds]
    body = [c for c in cmds[1:] if c.kind != "c"]
    if not body:
        return [Command(head.kind, [head.points[0]])]

    pen = head.points[0]
    segments: list[tuple[Command, tuple[float, float]]] = []
    for cmd in body:
        segments.append((cmd, pen))
        pen = cmd.points[-1]

    out: list[Command] = [Command(head.kind, [pen])]
    for cmd, start in reversed(segments):
        if cmd.kind == "l":
            out.append(Command("l", list(reversed(cmd.points))[1:] + [start]))
        elif cmd.kind == "b":
            c1, c2, _end = cmd.points
            out.append(Command("b", [c2, c1, start]))
        elif cmd.kind in ("s", "p"):
            # Approximate: control order reversed, original start becomes the end.
            out.append(Command(cmd.kind, list(reversed(cmd.points))[1:] + [start]))
        else:  # pragma: no cover - no other kinds can appear inside a subpath
            out.append(cmd.copy())
    if closed:
        out.append(Command("c", []))
    return out


def reverse(drawing: Drawing) -> Drawing:
    """Reverse the point order of every subpath.

    Exact for ``l`` (the point list is reversed) and ``b`` (the two control
    points are swapped and the original start point becomes the new end).
    **Approximate for ``s``/``p``**: the control point order is reversed, but
    B-splines are not symmetric under that operation, so the reversed spline
    is only a close, not identical, shape.
    """
    out: list[Command] = []
    for group in split_subpaths(drawing):
        out.extend(_reverse_subpath(group.commands))
    return Drawing(out)


def round_drawing(drawing: Drawing, digits: int = 2) -> Drawing:
    """Round every coordinate to ``digits`` decimals (in place value, not text)."""
    digits = int(digits)
    return Drawing(
        [
            Command(cmd.kind, [(round(x, digits), round(y, digits)) for x, y in cmd.points])
            for cmd in drawing.commands
        ]
    )


def path_length(drawing: Drawing, bezier_steps: int = DEFAULT_BEZIER_STEPS) -> float:
    """Total outline length of the (flattened) drawing.

    ``m``/``n`` jump the pen without adding length; a ``c`` adds the closing
    segment back to the subpath start, so a closed square measures its full
    perimeter.
    """
    flat = flatten(drawing, bezier_steps)
    total = 0.0
    pen: tuple[float, float] | None = None
    start: tuple[float, float] | None = None
    for cmd in flat.commands:
        if cmd.kind in ("m", "n"):
            pen = cmd.points[0]
            if cmd.kind == "m":
                start = pen
        elif cmd.kind == "l":
            if pen is not None and cmd.points:
                total += _dist(pen, cmd.points[0])
            for i in range(1, len(cmd.points)):
                total += _dist(cmd.points[i - 1], cmd.points[i])
            pen = cmd.points[-1]
        elif cmd.kind == "c":
            if pen is not None and start is not None:
                total += _dist(pen, start)
            pen = start
    return total


# --------------------------------------------------------------------------
# Clip override forms
# --------------------------------------------------------------------------

_DRAWING_LETTER_RE = re.compile("[mnlbspc]", re.IGNORECASE)


def _clip_num(value: float) -> str:
    return assutil.round_coord(value, 3)


def _coords_num(value: float) -> str:
    return assutil.round_coord(value, COORD_PRECISION)


def _parse_number_list(text: str) -> list[float]:
    """Parse a comma/whitespace separated number list, raising on junk."""
    numbers: list[float] = []
    pos = 0
    while pos < len(text):
        ch = text[pos]
        if ch in " \t\r\n,":
            pos += 1
            continue
        match = _NUMBER_RE.match(text, pos)
        if not match:
            raise ValueError(f"invalid clip argument: unexpected {ch!r} at offset {pos}")
        numbers.append(float(match.group(0)))
        pos = match.end()
    return numbers


def parse_clip_arg(arg: str) -> dict:
    """Parse the inside of a ``\\clip``/``\\iclip`` override tag.

    Returns ``{'type', 'scale', 'coords', 'drawing', 'raw'}`` where ``type`` is
    ``'rect'`` or ``'drawing'``, ``scale`` is a float or ``None`` when omitted,
    ``coords`` is ``[x1, y1, x2, y2]`` for rectangles, ``drawing`` is a
    :class:`Drawing` for drawings and ``raw`` is the untouched input.

    All real-world forms are accepted::

        clip(x1,y1,x2,y2)
        clip(scale,x1,y1,x2,y2)
        clip(1,m 0 0 l 100 0 100 100)
        clip(m 0 0 l 100 0 100 100)      # scale omitted
        clip(\\n  1,\\n  m 0 0 ... )        # whitespace / newlines

    A surrounding pair of parentheses is tolerated so callers can pass either
    the raw tag body or the whole ``(...)`` group.
    """
    raw = "" if arg is None else str(arg)
    text = raw.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    spec = {"type": None, "scale": None, "coords": None, "drawing": None, "raw": raw}
    if not text:
        raise ValueError("empty clip argument")

    match = _DRAWING_LETTER_RE.search(text)
    if match is None:
        numbers = _parse_number_list(text)
        if len(numbers) == 4:
            spec.update(type="rect", coords=numbers)
        elif len(numbers) == 5:
            spec.update(type="rect", scale=numbers[0], coords=numbers[1:])
        else:
            raise ValueError(
                f"clip rectangle needs 4 coordinates (or scale + 4), got {len(numbers)}"
            )
        return spec

    prefix = text[: match.start()]
    prefix_numbers = _NUMBER_RE.findall(prefix)
    if len(prefix_numbers) == 1:
        spec["scale"] = float(prefix_numbers[0])
    elif len(prefix_numbers) > 1:
        raise ValueError(f"unexpected numbers before the clip drawing: {prefix!r}")
    drawing = parse_drawing(text[match.start() :])
    spec.update(type="drawing", drawing=drawing)
    return spec


def format_clip_arg(spec: dict, inverse: bool = False) -> str:
    """Serialise a clip spec back to argument text (no tag syntax around it).

    ``inverse`` is accepted for API symmetry and has no effect on the text: the
    inversion is expressed by the tag *name* (``\\iclip``), not its argument.

    A rectangle formats as ``x1,y1,x2,y2`` (or ``scale,x1,y1,x2,y2``), a
    drawing as ``m 0 0 ...`` (or ``scale,m 0 0 ...``).  Round-trips through
    :func:`parse_clip_arg`.
    """
    kind = (spec or {}).get("type")
    if kind == "rect":
        coords = spec.get("coords") or []
        if len(coords) != 4:
            raise ValueError("clip rectangle spec needs exactly 4 coordinates")
        parts = [_coords_num(c) for c in coords]
        scale = spec.get("scale")
        if scale is not None:
            return ",".join([_clip_num(scale)] + parts)
        return ",".join(parts)
    if kind == "drawing":
        drawing = spec.get("drawing")
        if isinstance(drawing, Drawing):
            text = drawing.text()
        elif drawing is None:
            raise ValueError("clip drawing spec has no drawing")
        else:
            text = str(drawing)
        scale = spec.get("scale")
        if scale is None:
            return text
        return f"{_clip_num(scale)},{text}"
    raise ValueError(f"unknown clip spec type {kind!r}")


def _clip_tag(arg: str, inverse: bool) -> str:
    return f"{r'\iclip' if inverse else r'\clip'}({arg})"


def drawing_to_clip(drawing_or_text, scale: float | None = 1.0, inverse: bool = False) -> str:
    """Build a full ``\\clip(...)``/``\\iclip(...)`` override tag for a drawing.

    ``drawing_or_text`` may be a :class:`Drawing`, a list of :class:`Command`
    or an already serialised string.  ``scale=None`` omits the scale (plain
    ``\\clip(m ...)``); the default ``1.0`` produces Aegisub's explicit form::

        \\clip(1,m 0 0 l 100 0 100 100)
    """
    if isinstance(drawing_or_text, Drawing):
        text = drawing_or_text.text()
    elif isinstance(drawing_or_text, str):
        text = drawing_or_text
    else:
        text = Drawing(list(drawing_or_text)).text()
    spec = {"type": "drawing", "scale": scale, "coords": None, "drawing": text, "raw": text}
    return _clip_tag(format_clip_arg(spec, inverse=inverse), inverse)


def rect_to_clip(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float | None = None,
    inverse: bool = False,
) -> str:
    """Build a rectangular ``\\clip(x1,y1,x2,y2)`` override tag (``\\iclip`` when inverse)."""
    spec = {
        "type": "rect",
        "scale": scale,
        "coords": [x1, y1, x2, y2],
        "drawing": None,
        "raw": "",
    }
    return _clip_tag(format_clip_arg(spec, inverse=inverse), inverse)


# --------------------------------------------------------------------------
# SVG bridge
# --------------------------------------------------------------------------

_SVG_COMMANDS = set("MmLlHhVvCcSsQqTtAaZz")
_SVG_NUM_RE = re.compile(_NUM_PATTERN)
_SVG_SEPARATORS = " \t\r\n,"


def _svg_num(value: float) -> str:
    return assutil.round_coord(value, SVG_PRECISION)


def to_svg_path(drawing: Drawing) -> str:
    """Serialise a drawing as absolute SVG path data (``M``/``L``/``C``/``Z``).

    ``m`` becomes ``M``, ``l`` becomes ``L`` (all its points, SVG implicit
    repetition), ``b`` becomes ``C``, ``c`` becomes ``Z`` and ``n`` becomes
    ``M`` after closing the previous subpath with ``Z``.  ``s``/``p`` splines
    have no SVG equivalent and are emitted as sampled ``L`` polylines
    (``DEFAULT_BEZIER_STEPS`` subdivisions per segment) — a documented
    approximation.
    """
    parts: list[str] = []
    pen: tuple[float, float] | None = None
    open_subpath = False

    def coords(points: Sequence[tuple[float, float]]) -> str:
        return " ".join(f"{_svg_num(x)} {_svg_num(y)}" for x, y in points)

    for cmd in drawing.commands:
        kind = cmd.kind
        if kind == "m":
            parts.append(f"M {coords(cmd.points)}")
            pen = cmd.points[0]
            open_subpath = True
        elif kind == "n":
            if open_subpath:
                parts.append("Z")
            parts.append(f"M {coords(cmd.points)}")
            pen = cmd.points[0]
            open_subpath = True
        elif kind == "l":
            parts.append(f"L {coords(cmd.points)}")
            pen = cmd.points[-1]
            open_subpath = True
        elif kind == "b":
            parts.append(f"C {coords(cmd.points)}")
            pen = cmd.points[-1]
            open_subpath = True
        elif kind in ("s", "p"):
            start = pen if pen is not None else cmd.points[0]
            sampled = sample_bspline(start, cmd.points, DEFAULT_BEZIER_STEPS)
            parts.append(f"L {coords(sampled)}")
            pen = cmd.points[-1]
            open_subpath = True
        elif kind == "c":
            if open_subpath:
                parts.append("Z")
                open_subpath = False
    return " ".join(parts)


def _svg_skip_separators(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in _SVG_SEPARATORS:
        pos += 1
    return pos


def _svg_read_number(text: str, pos: int, command: str) -> tuple[float, int]:
    pos = _svg_skip_separators(text, pos)
    match = _SVG_NUM_RE.match(text, pos)
    if not match:
        raise ValueError(
            f"unparsable SVG path at offset {pos}: expected a number for command {command!r}"
        )
    return float(match.group(0)), match.end()


def _svg_read_flag(text: str, pos: int, command: str) -> tuple[int, int]:
    """Arc flags are single ``0``/``1`` characters (``a1 1 0 011 1`` is legal)."""
    pos = _svg_skip_separators(text, pos)
    if pos < len(text) and text[pos] in "01":
        return int(text[pos]), pos + 1
    raise ValueError(f"unparsable SVG path at offset {pos}: expected an arc flag (0 or 1)")


def _angle_between(ux: float, uy: float, vx: float, vy: float) -> float:
    dot = ux * vx + uy * vy
    norm = math.hypot(ux, uy) * math.hypot(vx, vy)
    if norm == 0.0:
        return 0.0
    angle = math.acos(max(-1.0, min(1.0, dot / norm)))
    if ux * vy - uy * vx < 0.0:
        angle = -angle
    return angle


def arc_to_cubics(p0, rx, ry, x_axis_rotation, large_arc, sweep, p1):
    """Convert an SVG elliptical arc into cubic Béziers (SVG impl. notes F.6).

    Returns a list of ``(c1, c2, end)`` tuples (the start is the caller's
    current point), each spanning at most 90°.  Returns ``None`` when the arc
    degenerates to a straight line (zero radii or coincident endpoints), which
    the caller renders as a line.  Handle length uses ``4/3·tan(θ/4)``, the
    standard approximation with a maximum radial error of ~0.03 % for a 90°
    segment.
    """
    x1, y1 = p0
    x2, y2 = p1
    if x1 == x2 and y1 == y2:
        return []
    rx = abs(float(rx))
    ry = abs(float(ry))
    if rx == 0.0 or ry == 0.0:
        return None

    phi = math.radians(float(x_axis_rotation) % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        factor = math.sqrt(lam)
        rx *= factor
        ry *= factor

    numerator = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    if denominator == 0.0:
        return None
    coefficient = math.sqrt(max(0.0, numerator / denominator))
    if bool(large_arc) == bool(sweep):
        coefficient = -coefficient
    cxp = coefficient * rx * y1p / ry
    cyp = -coefficient * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    start_angle = _angle_between(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    delta = _angle_between(
        (x1p - cxp) / rx,
        (y1p - cyp) / ry,
        (-x1p - cxp) / rx,
        (-y1p - cyp) / ry,
    )
    if not sweep and delta > 0.0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0.0:
        delta += 2.0 * math.pi

    segments = max(1, int(math.ceil(abs(delta) / (math.pi / 2.0) - 1e-9)))
    step = delta / segments
    handle = 4.0 / 3.0 * math.tan(step / 4.0)

    def point(t: float) -> tuple[float, float]:
        return (
            cx + rx * math.cos(t) * cos_phi - ry * math.sin(t) * sin_phi,
            cy + rx * math.cos(t) * sin_phi + ry * math.sin(t) * cos_phi,
        )

    def derivative(t: float) -> tuple[float, float]:
        return (
            -rx * math.sin(t) * cos_phi - ry * math.cos(t) * sin_phi,
            -rx * math.sin(t) * sin_phi + ry * math.cos(t) * cos_phi,
        )

    out: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    t1 = start_angle
    for _ in range(segments):
        t2 = t1 + step
        e1, e2 = point(t1), point(t2)
        d1, d2 = derivative(t1), derivative(t2)
        c1 = (e1[0] + handle * d1[0], e1[1] + handle * d1[1])
        c2 = (e2[0] - handle * d2[0], e2[1] - handle * d2[1])
        out.append((c1, c2, e2))
        t1 = t2
    return out


def _svg_to_ass_commands(path: str) -> list[Command]:
    """Parse SVG path data into absolute ASS commands (``m``/``l``/``b``/``c``)."""
    out: list[Command] = []
    pos = 0
    length = len(path)
    cx = cy = 0.0
    sub_x = sub_y = 0.0
    command: str | None = None
    prev_command: str | None = None
    prev_cubic_control: tuple[float, float] | None = None
    prev_quad_control: tuple[float, float] | None = None
    subpath_open = False

    while True:
        pos = _svg_skip_separators(path, pos)
        if pos >= length:
            break
        ch = path[pos]
        if ch.isalpha():
            if ch not in _SVG_COMMANDS:
                raise ValueError(f"unparsable SVG path at offset {pos}: unknown command {ch!r}")
            command = ch
            pos += 1
            if command in "Zz":
                if subpath_open:
                    out.append(Command("c", []))
                    subpath_open = False
                cx, cy = sub_x, sub_y
                prev_command = command
                prev_cubic_control = prev_quad_control = None
                continue
        else:
            if command is None:
                raise ValueError(
                    f"unparsable SVG path at offset {pos}: expected a command letter, got {ch!r}"
                )
            if command in "Zz":
                raise ValueError(
                    f"unparsable SVG path at offset {pos}: unexpected number after 'Z'"
                )

        assert command is not None
        relative = command.islower()
        upper = command.upper()

        if upper == "M":
            x, pos = _svg_read_number(path, pos, command)
            y, pos = _svg_read_number(path, pos, command)
            if relative:
                cx, cy = cx + x, cy + y
            else:
                cx, cy = x, y
            out.append(Command("m", [(cx, cy)]))
            sub_x, sub_y = cx, cy
            subpath_open = True
            # Further coordinate pairs after M are implicit linetos.
            command = "l" if relative else "L"
            prev_command = "M"
            prev_cubic_control = prev_quad_control = None

        elif upper == "L":
            x, pos = _svg_read_number(path, pos, command)
            y, pos = _svg_read_number(path, pos, command)
            cx, cy = (cx + x, cy + y) if relative else (x, y)
            out.append(Command("l", [(cx, cy)]))
            prev_command = command
            prev_cubic_control = prev_quad_control = None

        elif upper == "H":
            x, pos = _svg_read_number(path, pos, command)
            cx = cx + x if relative else x
            out.append(Command("l", [(cx, cy)]))
            prev_command = command
            prev_cubic_control = prev_quad_control = None

        elif upper == "V":
            y, pos = _svg_read_number(path, pos, command)
            cy = cy + y if relative else y
            out.append(Command("l", [(cx, cy)]))
            prev_command = command
            prev_cubic_control = prev_quad_control = None

        elif upper == "C":
            x1, pos = _svg_read_number(path, pos, command)
            y1, pos = _svg_read_number(path, pos, command)
            x2, pos = _svg_read_number(path, pos, command)
            y2, pos = _svg_read_number(path, pos, command)
            ex, pos = _svg_read_number(path, pos, command)
            ey, pos = _svg_read_number(path, pos, command)
            if relative:
                x1, y1 = cx + x1, cy + y1
                x2, y2 = cx + x2, cy + y2
                ex, ey = cx + ex, cy + ey
            out.append(Command("b", [(x1, y1), (x2, y2), (ex, ey)]))
            prev_cubic_control = (x2, y2)
            prev_quad_control = None
            prev_command = command
            cx, cy = ex, ey

        elif upper == "S":
            x2, pos = _svg_read_number(path, pos, command)
            y2, pos = _svg_read_number(path, pos, command)
            ex, pos = _svg_read_number(path, pos, command)
            ey, pos = _svg_read_number(path, pos, command)
            if relative:
                x2, y2 = cx + x2, cy + y2
                ex, ey = cx + ex, cy + ey
            if prev_command is not None and prev_command.upper() in ("C", "S") and prev_cubic_control:
                x1 = 2.0 * cx - prev_cubic_control[0]
                y1 = 2.0 * cy - prev_cubic_control[1]
            else:
                x1, y1 = cx, cy
            out.append(Command("b", [(x1, y1), (x2, y2), (ex, ey)]))
            prev_cubic_control = (x2, y2)
            prev_quad_control = None
            prev_command = command
            cx, cy = ex, ey

        elif upper == "Q":
            qx, pos = _svg_read_number(path, pos, command)
            qy, pos = _svg_read_number(path, pos, command)
            ex, pos = _svg_read_number(path, pos, command)
            ey, pos = _svg_read_number(path, pos, command)
            if relative:
                qx, qy = cx + qx, cy + qy
                ex, ey = cx + ex, cy + ey
            c1 = (cx + 2.0 / 3.0 * (qx - cx), cy + 2.0 / 3.0 * (qy - cy))
            c2 = (ex + 2.0 / 3.0 * (qx - ex), ey + 2.0 / 3.0 * (qy - ey))
            out.append(Command("b", [c1, c2, (ex, ey)]))
            prev_quad_control = (qx, qy)
            prev_cubic_control = None
            prev_command = command
            cx, cy = ex, ey

        elif upper == "T":
            ex, pos = _svg_read_number(path, pos, command)
            ey, pos = _svg_read_number(path, pos, command)
            if relative:
                ex, ey = cx + ex, cy + ey
            if prev_command is not None and prev_command.upper() in ("Q", "T") and prev_quad_control:
                qx = 2.0 * cx - prev_quad_control[0]
                qy = 2.0 * cy - prev_quad_control[1]
            else:
                qx, qy = cx, cy
            c1 = (cx + 2.0 / 3.0 * (qx - cx), cy + 2.0 / 3.0 * (qy - cy))
            c2 = (ex + 2.0 / 3.0 * (qx - ex), ey + 2.0 / 3.0 * (qy - ey))
            out.append(Command("b", [c1, c2, (ex, ey)]))
            prev_quad_control = (qx, qy)
            prev_cubic_control = None
            prev_command = command
            cx, cy = ex, ey

        elif upper == "A":
            rx, pos = _svg_read_number(path, pos, command)
            ry, pos = _svg_read_number(path, pos, command)
            rotation, pos = _svg_read_number(path, pos, command)
            large_arc, pos = _svg_read_flag(path, pos, command)
            sweep, pos = _svg_read_flag(path, pos, command)
            ex, pos = _svg_read_number(path, pos, command)
            ey, pos = _svg_read_number(path, pos, command)
            if relative:
                ex, ey = cx + ex, cy + ey
            segments = arc_to_cubics((cx, cy), rx, ry, rotation, large_arc, sweep, (ex, ey))
            if segments is None:
                out.append(Command("l", [(ex, ey)]))
                prev_cubic_control = prev_quad_control = None
            else:
                for c1, c2, end in segments:
                    out.append(Command("b", [c1, c2, end]))
                prev_cubic_control = segments[-1][1] if segments else None
                prev_quad_control = None
            prev_command = command
            cx, cy = ex, ey

        else:  # pragma: no cover - every letter is covered above
            raise ValueError(f"unparsable SVG path at offset {pos}: unsupported command {ch!r}")

    return out


def from_svg_path(
    path_data: str,
    scale: float = 1.0,
    translate: tuple[float, float] = (0.0, 0.0),
    flip_y: bool = False,
) -> Drawing:
    """Convert SVG path data into an ASS :class:`Drawing`.

    Supports ``M/m L/l H/h V/v C/c S/s Q/q T/t A/a Z/z``.  Relative commands
    are resolved to absolute, quadratics (``Q``/``T``) are promoted to cubics
    with the standard control point rule, smooth ``S``/``T`` use reflected
    control points, and elliptical arcs are expanded into cubic Bézier
    segments of at most 90° each.

    **Y axis:** SVG and ASS both grow downwards, so nothing is flipped by
    default.  ``flip_y=True`` maps ``y -> -y`` *before* the translation is
    applied (for sources authored in a mathematical y-up space)::

        x' = scale * x + tx
        y' = -scale * y + ty      # flip_y=True

    Unparsable path data raises :class:`ValueError` naming the offending
    character offset.
    """
    if path_data is None:
        raise ValueError("SVG path data must be a string, got None")
    drawing = Drawing(_svg_to_ass_commands(str(path_data)))
    tx, ty = float(translate[0]), float(translate[1])
    factor = float(scale)
    matrix = (factor, 0.0, 0.0, -factor if flip_y else factor, tx, ty)
    return transform(drawing, matrix)


def _positive_size(size) -> tuple[float, float] | None:
    """Return ``(w, h)`` when both dimensions are > 0, else ``None``."""
    try:
        w, h = float(size[0]), float(size[1])
    except (TypeError, ValueError, IndexError):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return (w, h)


def svg_path_to_ass_drawing(
    path_data: str,
    source_size: tuple[float, float] = (0.0, 0.0),
    target_size: tuple[float, float] = (0.0, 0.0),
    fit: str = "contain",
    pad: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    flip_y: bool = False,
) -> Drawing:
    """Main typesetting entry point: SVG path -> ASS drawing ready to position.

    The path is converted with :func:`from_svg_path` (no scaling), then:

    1. if ``source_size`` and ``target_size`` are both non-zero, the source box
       is fitted into the target box (minus ``pad`` on every side):

       * ``fit='contain'`` — uniform scale (``min`` of the axis ratios), centred
         inside the padded target box, aspect ratio preserved;
       * ``fit='stretch'`` — independent x/y scales filling the padded box;
       * ``fit='none'`` — no fitting at all; ``pad`` is ignored;
    2. ``offset`` is added on top of the resulting translation.

    **Y axis:** the path is read in SVG coordinates (y down, origin top-left of
    the source box), which matches ASS.  ``flip_y=True`` negates y *before*
    the fit/offset translation; when ``source_size`` is known the flip is taken
    about the source box (``y -> source_h - y``) so the flipped path still fills
    the source box and composes correctly with ``fit``.  Without a
    ``source_size`` the flip is a plain ``y -> -y``.  The path is assumed to be
    expressed relative to the source box origin: for a viewBox with a non-zero
    origin, pass ``offset=(-min_x, -min_y)`` (or pre-translate the path)
    yourself.
    """
    mode = (fit or "contain").strip().lower()
    if mode not in ("contain", "stretch", "none"):
        raise ValueError(f"unknown fit mode {fit!r}; expected 'contain', 'stretch' or 'none'")

    ox, oy = float(offset[0]), float(offset[1])
    pad = float(pad)
    source = _positive_size(source_size)
    target = _positive_size(target_size)

    # A y-up source (flip_y) is mirrored inside the source box, i.e. y -> h - y.
    flip_translate = (0.0, source[1]) if (flip_y and source is not None) else (0.0, 0.0)
    drawing = from_svg_path(
        path_data, scale=1.0, translate=flip_translate, flip_y=flip_y
    )

    if mode == "none" or source is None or target is None:
        if ox or oy:
            drawing = translate(drawing, ox, oy)
        return drawing

    sw, sh = source
    tw, th = target
    usable_w = tw - 2.0 * pad
    usable_h = th - 2.0 * pad
    if usable_w <= 0.0 or usable_h <= 0.0:
        raise ValueError(f"pad {pad} leaves no room inside target_size {target_size}")

    if mode == "stretch":
        sx = usable_w / sw
        sy = usable_h / sh
        dx, dy = pad, pad
    else:  # contain
        s = min(usable_w / sw, usable_h / sh)
        sx = sy = s
        dx = pad + (usable_w - sw * s) / 2.0
        dy = pad + (usable_h - sh * s) / 2.0

    return transform(drawing, (sx, 0.0, 0.0, sy, dx + ox, dy + oy))
