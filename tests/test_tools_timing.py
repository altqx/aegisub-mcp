"""Tests for :mod:`aegisub_mcp.tools.timing` — timing, frames, timecodes and QC.

Every code path is exercised through the public ``ass_*`` entry points.  The
real-world fixtures in ``tests/fixtures/real`` cover file-level behaviour (byte
fidelity, malformed times, the overlap in ``basic.ass``); hand-built documents
written into ``tmp_path`` cover the maths: per-layer overlaps, cps on text with
override tags, clamped shifts, frame/keyframe snapping and the timecodes parser.

No test writes into a fixture: mutations happen in memory or on a ``tmp_path``
copy, and the silence tools are driven with a monkeypatched ``silencedetect`` so
the suite stays independent of ffmpeg and of any media file.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from aegisub_mcp.tools import lines as L
from aegisub_mcp.tools import timing as T
from aegisub_mcp.tools.base import ToolError, workspace

FIX = pathlib.Path(__file__).parent / "fixtures" / "real"
FIXTURES = sorted(FIX.glob("*.ass")) + sorted(FIX.glob("*.ssa"))
BASIC = FIX / "basic.ass"
RECOVERABLE = FIX / "recoverable.ass"

#: every tool the brief requires, plus the two silence entry points
REQUIRED_TOOLS = {
    "ass_shift_times",
    "ass_scale_times",
    "ass_set_times",
    "ass_set_durations",
    "ass_frame_from_ms",
    "ass_ms_from_frame",
    "ass_snap_to_frames",
    "ass_load_keyframes",
    "ass_snap_to_keyframes",
    "ass_align_to_silence",
    "ass_align_lines_to_silence",
    "ass_read_timecodes",
    "ass_write_timecodes",
    "ass_frame_from_timecodes",
    "ass_ms_from_timecodes",
    "ass_qc",
    "ass_check_overlaps",
    "ass_fix_timing",
    "ass_reading_speed",
    "ass_cps",
}

#: the QC codes the brief demands
REQUIRED_CODES = {
    "zero_duration",
    "negative_duration",
    "cps_high",
    "cps_extreme",
    "too_short",
    "too_long",
    "too_many_chars",
    "too_many_lines",
    "overlap_same_layer",
    "gap_tiny",
    "empty_text",
    "style_missing",
    "unclosed_override_block",
    "unknown_tag",
}

#: public shortcuts that can add timing entries to ``timing.py``
STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def clean_workspace():
    """Start every test from a pristine module-level workspace singleton.

    ``timing.py`` keeps its loaded timecodes on the workspace; ``Workspace.__init__``
    does not know about that attribute, so it is removed explicitly to stop one
    test's timecodes leaking into the next.
    """
    def reset() -> None:
        workspace.__init__()
        for extra in ("timecodes", "timecodes_path"):
            workspace.__dict__.pop(extra, None)

    reset()
    yield
    reset()


def style_line(name: str = "Default") -> str:
    return (
        f"Style: {name},Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,"
        "100,100,0,0,1,2,1,2,40,40,30,1"
    )


def ass_time(value) -> str:
    """``1234`` → ``"0:00:01.23"``; time strings pass through unchanged.

    Letting the tests write milliseconds keeps the arithmetic readable; ASS
    itself only stores centisecond-precision ``H:MM:SS.cc`` timestamps.
    """
    if isinstance(value, str):
        return value
    total = int(round(float(value)))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{millis // 10:02d}"


def ev(start, end, text, layer=0, style="Default", kind="Dialogue") -> dict:
    """One event: ``ev("0:00:01.00", "0:00:02.00", "hi")`` or ``ev(1000, 2000, "hi")``."""
    return {
        "layer": layer,
        "start": ass_time(start),
        "end": ass_time(end),
        "style": style,
        "kind": kind,
        "text": text,
    }


def make_ass(events=(), styles=("Default",), fps=None) -> str:
    """Build a minimal but complete ASS document as text."""
    header = [
        "[Script Info]",
        "Title: timing test",
        "ScriptType: v4.00+",
        "PlayResX: 1280",
        "PlayResY: 720",
    ]
    if fps is not None:
        header.append(f"FPS: {fps}")
    body = [
        "",
        "[V4+ Styles]",
        STYLE_FORMAT,
    ]
    body += [style_line(name) for name in styles]
    body += [
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for item in events:
        body.append(
            f"{item['kind']}: {item['layer']},{item['start']},{item['end']},"
            f"{item['style']},,0,0,0,,{item['text']}"
        )
    return "\n".join(header + body) + "\n"


def open_text(tmp_path: pathlib.Path, text: str, name: str = "case.ass", doc_id=None) -> str:
    """Write ``text`` to ``tmp_path/name`` and open it through the tool layer."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return L.ass_open(str(path), doc_id=doc_id)["doc_id"]


def open_basic(doc_id=None) -> str:
    return L.ass_open(str(BASIC), doc_id=doc_id)["doc_id"]


def times(doc_id, index: int) -> tuple[int, int]:
    entry = workspace.get(doc_id).events()[index]
    return entry.start_ms, entry.end_ms


def issue_codes(report) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


def assert_json(result) -> None:
    assert isinstance(result, dict)
    json.dumps(result)  # raises TypeError when the result is not serialisable


class FakeMCP:
    """Minimal stand-in for FastMCP; records what ``register`` hands it."""

    def __init__(self) -> None:
        self.registered: list = []

    def tool(self):
        def decorator(fn):
            self.registered.append(fn)
            return fn

        return decorator


# --------------------------------------------------------------------------- #
# module contract
# --------------------------------------------------------------------------- #


def test_required_tools_exist_and_are_documented() -> None:
    for name in sorted(REQUIRED_TOOLS):
        fn = getattr(T, name, None)
        assert callable(fn), f"{name} is missing from tools/timing.py"
        doc = (fn.__doc__ or "").strip()
        assert doc, f"{name} has no docstring"
        assert "Returns:" in doc, f"{name} does not document its return shape"


def test_register_hands_every_ass_tool_to_fastmcp() -> None:
    mcp = FakeMCP()
    names = T.register(mcp, workspace)
    assert names == sorted(names)
    assert set(names) == set(REQUIRED_TOOLS)
    assert len(mcp.registered) == len(names)
    assert {fn.__name__ for fn in mcp.registered} == set(names)


def test_read_only_tools_return_json_serialisable_dicts(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:01.00", "hello"),
                ev("0:00:01.00", "0:00:01.50", r"{\i1}tall{\i0}"),
            ],
            fps=25,
        ),
    )
    results = [
        T.ass_qc(doc_id=doc_id),
        T.ass_cps(doc_id=doc_id),
        T.ass_check_overlaps(doc_id=doc_id),
        T.ass_reading_speed(0, doc_id=doc_id),
        T.ass_frame_from_ms(1000, fps=25, doc_id=doc_id),
        T.ass_ms_from_frame(25, fps=25, doc_id=doc_id),
        T.ass_frame_from_timecodes(2, doc_id=doc_id),
        T.ass_ms_from_timecodes(90, doc_id=doc_id),
    ]
    for result in results:
        assert_json(result)


def test_every_tool_needs_a_document_and_says_so() -> None:
    with pytest.raises(ToolError):
        T.ass_qc()
    with pytest.raises(ToolError):
        T.ass_cps()
    with pytest.raises(ToolError):
        T.ass_check_overlaps()
    with pytest.raises(ToolError):
        T.ass_shift_times(None, 100)


# --------------------------------------------------------------------------- #
# real fixtures: byte fidelity + QC
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_read_only_tools_keep_real_fixtures_byte_exact(
    path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Run the timing tools over a real file and re-save it: bytes must match.

    The two mutating tools keep their ``dry_run=True`` default, so opening a
    fixture, inspecting it and saving it back out must be byte-for-byte a no-op.
    A file whose events carry a broken timestamp (``recoverable.ass``) makes the
    duration tools refuse with a ToolError instead of guessing — that is allowed
    here precisely because nothing may be written in that case.
    """
    original = path.read_bytes()
    doc_id = L.ass_open(str(path))["doc_id"]

    T.ass_qc(doc_id=doc_id)
    T.ass_check_overlaps(doc_id=doc_id)
    try:
        T.ass_cps(doc_id=doc_id)
        T.ass_set_durations(None, min_ms=300, doc_id=doc_id)  # dry run by default
        T.ass_fix_timing(doc_id=doc_id)  # dry run by default
    except ToolError:
        # unparseable timestamps: the tools report/refuse, they never invent times
        pass

    dest = tmp_path / f"saved-{path.name}"
    saved = L.ass_save(path=str(dest), doc_id=doc_id)
    assert dest.read_bytes() == original, saved


def test_qc_on_basic_fixture_is_clean_and_counts_comments() -> None:
    """``basic.ass`` holds one dialogue and one comment, both with plain text."""
    doc_id = open_basic()
    report = T.ass_qc(doc_id=doc_id)
    assert_json(report)
    assert report["summary"]["lines_checked"] == 2
    assert report["summary"]["dialogue"] == 1
    assert report["summary"]["comments"] == 1
    assert report["summary"]["drawings"] == 0
    assert report["issues"] == []
    assert report["summary"]["clean"] is True
    assert report["summary"]["ok"] is True


def test_qc_on_recoverable_fixture_reports_unparseable_time() -> None:
    doc_id = L.ass_open(str(RECOVERABLE))["doc_id"]
    report = T.ass_qc(doc_id=doc_id)
    assert report["unparseable_lines"] == [0]
    assert "unparseable_time" in issue_codes(report)
    assert report["summary"]["errors"] == 1
    assert report["summary"]["ok"] is False


def test_reading_speed_and_cps_match_the_fixture_text() -> None:
    doc_id = open_basic()
    speed = T.ass_reading_speed(0, doc_id=doc_id)
    # "Hello {\i1}world{\i0}" -> 11 visible characters over 3 s
    assert speed["characters"] == 11
    assert speed["plain_text"] == "Hello world"
    assert speed["duration_ms"] == 3000
    assert speed["cps"] == pytest.approx(11 / 3.0, abs=1e-3)

    everything = T.ass_cps(doc_id=doc_id)
    assert everything["count"] == 2
    assert everything["totals"]["characters"] == 22
    assert everything["totals"]["worst_index"] == 1  # the comment: 11 chars in 1 s
    assert everything["lines"][0]["cps"] == pytest.approx(11 / 3.0, abs=1e-3)
    assert everything["lines"][1]["cps"] == 11.0


# --------------------------------------------------------------------------- #
# shift / scale
# --------------------------------------------------------------------------- #


def test_shift_times_moves_every_line(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:01.00", "0:00:02.00", "a"),
                ev("0:00:03.00", "0:00:04.00", "b"),
            ]
        ),
    )
    result = T.ass_shift_times(None, 500, doc_id=doc_id)
    assert result["applied_offset_ms"] == 500
    assert result["count"] == 2
    assert result["applied"] is True
    assert times(doc_id, 0) == (1500, 2500)
    assert times(doc_id, 1) == (3500, 4500)
    assert result["changes"][0]["new_start_ms"] == 1500
    assert result["changes"][0]["start"] == "0:00:01.50"


def test_shift_times_negative_offset_may_go_below_zero(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:01.00", "0:00:02.00", "a")]))
    result = T.ass_shift_times(None, -1500, doc_id=doc_id)
    assert result["clamped"] is False
    assert times(doc_id, 0) == (-500, 500)
    assert result["changes"][0]["new_start_ms"] == -500


def test_shift_times_clamp_preserves_relative_timing(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:01.00", "0:00:02.00", "a"),
                ev("0:00:03.00", "0:00:04.00", "b"),
            ]
        ),
    )
    result = T.ass_shift_times(None, -5000, clamp=True, doc_id=doc_id)
    assert result["requested_offset_ms"] == -5000
    assert result["applied_offset_ms"] == -1000  # limited by the earliest start
    assert result["clamped"] is True
    assert times(doc_id, 0) == (0, 1000)
    assert times(doc_id, 1) == (2000, 3000)  # same +2 s gap as before


def test_shift_times_only_selected_shifts_the_selection(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:01.00", "0:00:02.00", "a"),
                ev("0:00:03.00", "0:00:04.00", "b"),
            ]
        ),
    )
    result = T.ass_shift_times([1], 500, doc_id=doc_id)
    assert result["indices"] == [1]
    assert times(doc_id, 0) == (1000, 2000)
    assert times(doc_id, 1) == (3500, 4500)

    T.ass_shift_times("0-1", -500, doc_id=doc_id)
    assert times(doc_id, 0) == (500, 1500)
    assert times(doc_id, 1) == (3000, 4000)


def test_shift_times_rejects_malformed_offsets(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:01.00", "0:00:02.00", "a")]))
    with pytest.raises(ToolError):
        T.ass_shift_times(None, "later", doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_shift_times(None, True, doc_id=doc_id)


def test_scale_times_around_document_origin_reports_spans(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:01.00", "0:00:02.00", "a"),
                ev("0:00:03.00", "0:00:04.00", "b"),
            ]
        ),
    )
    result = T.ass_scale_times(None, 2.0, origin="document", doc_id=doc_id)
    assert result["origin_ms"] == 0
    assert result["span"]["before"] == {"min_start_ms": 1000, "max_end_ms": 4000, "span_ms": 3000}
    assert result["span"]["after"] == {"min_start_ms": 2000, "max_end_ms": 8000, "span_ms": 6000}
    assert times(doc_id, 0) == (2000, 4000)
    assert times(doc_id, 1) == (6000, 8000)


def test_scale_times_origin_first_and_explicit(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:01.00", "0:00:02.00", "a"),
                ev("0:00:03.00", "0:00:04.00", "b"),
            ]
        ),
    )
    first = T.ass_scale_times(None, 0.5, origin="first", doc_id=doc_id)
    assert first["origin_ms"] == 1000
    assert times(doc_id, 0) == (1000, 1500)
    assert times(doc_id, 1) == (2000, 2500)

    explicit = T.ass_scale_times(None, 2.0, origin=1000, doc_id=doc_id)
    assert explicit["origin_ms"] == 1000
    assert times(doc_id, 0) == (1000, 2000)
    assert times(doc_id, 1) == (3000, 4000)


def test_scale_times_rejects_bad_factor(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:01.00", "0:00:02.00", "a")]))
    with pytest.raises(ToolError):
        T.ass_scale_times(None, 0, doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_scale_times(None, -1.5, doc_id=doc_id)


# --------------------------------------------------------------------------- #
# set times / set durations
# --------------------------------------------------------------------------- #


def test_set_times_accepts_milliseconds_and_time_strings(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:00.00", "0:00:05.00", "a")]))
    result = T.ass_set_times(0, start_ms="0:00:01.50", end_ms=2500, doc_id=doc_id)
    assert result["before"]["start_ms"] == 0
    assert result["after"] == {"start_ms": 1500, "end_ms": 2500, "duration_ms": 1000}
    assert result["changed"] == ["Start", "End"]
    assert times(doc_id, 0) == (1500, 2500)


def test_set_times_duration_only_and_centisecond_rounding(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:01.00", "0:00:05.00", "a")]))
    result = T.ass_set_times(0, duration_ms=1234, doc_id=doc_id)
    assert result["after"]["duration_ms"] == 1230  # ASS times are centiseconds
    assert result["rounded_to_centiseconds"] is True
    assert times(doc_id, 0) == (1000, 2230)


def test_set_times_rejects_end_before_start(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:05.00", "0:00:06.00", "a")]))
    with pytest.raises(ToolError) as excinfo:
        T.ass_set_times(0, end_ms=1000, doc_id=doc_id)
    message = str(excinfo.value)
    assert "0:00:01.00" in message and "0:00:05.00" in message
    assert times(doc_id, 0) == (5000, 6000)  # untouched


def test_set_times_requires_an_argument_and_valid_values(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:01.00", "0:00:02.00", "a")]))
    with pytest.raises(ToolError):
        T.ass_set_times(0, doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_set_times(0, end_ms=1500, duration_ms=500, doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_set_times(7, start_ms=0, doc_id=doc_id)


def test_set_durations_dry_run_default_leaves_the_document_alone(
    tmp_path: pathlib.Path,
) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:00.40", "a"),
                ev("0:00:01.00", "0:00:03.00", "b"),
            ]
        ),
    )
    before = workspace.get(doc_id).to_bytes()
    result = T.ass_set_durations(None, min_ms=1000, doc_id=doc_id)
    assert result["dry_run"] is True
    assert result["applied"] is False
    assert result["count"] == 1
    assert result["changes"][0]["new_ms"] == 1000
    assert workspace.get(doc_id).to_bytes() == before
    assert workspace.undo_depth(doc_id)[0] == 0  # nothing snapshotted either


def test_set_durations_applies_and_stops_at_the_layer_neighbour(
    tmp_path: pathlib.Path,
) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:00.40", "a"),  # wants 1000, neighbour at 800
                ev("0:00:00.80", "0:00:02.00", "b"),  # fine already
                ev("0:00:00.60", "0:00:01.00", "c", layer=1),  # other layer: no limit
            ]
        ),
    )
    result = T.ass_set_durations(None, min_ms=1000, doc_id=doc_id, dry_run=False)
    assert result["applied"] is True
    assert times(doc_id, 0) == (0, 800)  # clamped to the next line on layer 0
    assert times(doc_id, 1) == (800, 2000)
    assert times(doc_id, 2) == (600, 1600)  # layer 1 is its own world
    first = result["changes"][0]
    assert first["new_ms"] == 800
    assert "clamped" in first["reason"]


def test_set_durations_blocks_when_there_is_no_room(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:00.10", "a"),
                ev("0:00:00.10", "0:00:02.00", "b"),
            ]
        ),
    )
    result = T.ass_set_durations(None, min_ms=1000, doc_id=doc_id, dry_run=False)
    assert result["count"] == 0
    assert result["blocked"][0]["index"] == 0
    assert result["blocked"][0]["limit_ms"] == 100
    assert times(doc_id, 0) == (0, 100)


def test_set_durations_allow_overlap_true_ignores_neighbours(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:00.40", "a"),
                ev("0:00:00.80", "0:00:02.00", "b"),
            ]
        ),
    )
    result = T.ass_set_durations(None, min_ms=1000, doc_id=doc_id, dry_run=False, allow_overlap=True)
    assert result["allow_overlap"] is True
    assert times(doc_id, 0) == (0, 1000)
    assert result["count"] == 1


def test_set_durations_max_ms_shortens_and_mode_start_moves_the_start(
    tmp_path: pathlib.Path,
) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:01.00", "0:00:11.00", "a")]))
    result = T.ass_set_durations(None, max_ms=4000, doc_id=doc_id, dry_run=False)
    assert times(doc_id, 0) == (1000, 5000)

    other = open_text(
        tmp_path,
        make_ass([ev("0:00:01.00", "0:00:11.00", "a")]),
        name="second.ass",
    )
    result = T.ass_set_durations(None, max_ms=4000, mode="start", doc_id=other, dry_run=False)
    assert result["mode"] == "start"
    assert times(other, 0) == (7000, 11000)
    with pytest.raises(ToolError):
        T.ass_set_durations(None, min_ms=1000, max_ms=500, doc_id=other)


def test_set_durations_rejects_an_unknown_mode(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:01.00", "0:00:02.00", "a")]))
    with pytest.raises(ToolError):
        T.ass_set_durations(None, min_ms=1000, mode="sideways", doc_id=doc_id)


# --------------------------------------------------------------------------- #
# frames and keyframes
# --------------------------------------------------------------------------- #


def test_frame_conversions_at_25fps() -> None:
    forward = T.ass_frame_from_ms(1000, fps=25)
    assert forward["frame"] == 25
    assert forward["frame_exact"] == 25.0
    assert forward["fps_source"] == "argument"

    odd = T.ass_frame_from_ms(1001, fps=25)
    assert (odd["frame_floor"], odd["frame"], odd["frame_ceil"]) == (25, 25, 26)

    back = T.ass_ms_from_frame(25, fps=25)
    assert back["ms"] == 1000

    fractional = T.ass_ms_from_frame(2.5, fps=25)
    assert fractional["ms"] == 100


def test_frame_conversions_accept_time_strings() -> None:
    assert T.ass_frame_from_ms("0:00:01.00", fps=24)["frame"] == 24
    assert T.ass_ms_from_frame("25", fps=25)["ms"] == 1000


def test_fps_resolution_order(tmp_path: pathlib.Path) -> None:
    # a document without an ``FPS`` line and no video has nothing to resolve
    bare = open_text(
        tmp_path,
        make_ass([ev("0:00:01.00", "0:00:02.00", "a")]),
        name="bare.ass",
    )
    with pytest.raises(ToolError) as excinfo:
        T.ass_frame_from_ms(1000, doc_id=bare)
    assert "no frame rate available" in str(excinfo.value)

    document = open_text(
        tmp_path,
        make_ass([ev("0:00:01.00", "0:00:02.00", "a")], fps=24),
        name="fps.ass",
    )
    from_document = T.ass_frame_from_ms(1000, doc_id=document)
    assert from_document["fps"] == 24.0
    assert from_document["fps_source"] == "document FPS"
    assert from_document["frame"] == 24

    # the workspace video wins over the document, an explicit argument wins
    # over both
    workspace.video = {"path": None, "fps": 23.976}
    from_video = T.ass_frame_from_ms(1000, doc_id=document)
    assert from_video["fps"] == pytest.approx(23.976)
    assert from_video["fps_source"] == "workspace.video"

    explicit = T.ass_frame_from_ms(1000, fps=50, doc_id=document)
    assert explicit["fps"] == 50.0
    assert explicit["fps_source"] == "argument"


def test_snap_to_frames_modes_and_fields(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev(1010, 3520, "a")]), name="snap.ass")
    nearest = T.ass_snap_to_frames(None, fps=24, doc_id=doc_id)
    # 1010 ms -> 24.24 frames -> 24 -> 1000 ms; 3520 -> 84.48 -> 84 -> 3500 ms
    assert nearest["considered"] == 2
    assert nearest["count"] == 2
    assert [(c["field"], c["old_ms"], c["new_ms"], c["frame"]) for c in nearest["changes"]] == [
        ("Start", 1010, 1000, 24),
        ("End", 3520, 3500, 84),
    ]
    assert nearest["applied"] is True
    assert times(doc_id, 0) == (1000, 3500)

    floor_doc = open_text(tmp_path, make_ass([ev(1010, 3520, "a")]), name="floor.ass")
    floor = T.ass_snap_to_frames(None, fps=24, mode="floor", doc_id=floor_doc)
    assert floor["mode"] == "floor"
    assert times(floor_doc, 0) == (1000, 3500)

    ceil_doc = open_text(tmp_path, make_ass([ev(1010, 3520, "a")]), name="ceil.ass")
    ceil = T.ass_snap_to_frames(None, fps=24, mode="ceil", which="start", doc_id=ceil_doc)
    assert ceil["considered"] == 1  # which="start" leaves the end alone
    assert [(c["new_ms"], c["frame"]) for c in ceil["changes"]] == [(1040, 25)]
    assert times(ceil_doc, 0) == (1040, 3520)

    with pytest.raises(ToolError):
        T.ass_snap_to_frames(None, fps=24, mode="sideways", doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_snap_to_frames(None, fps=24, which="middle", doc_id=doc_id)


def test_snap_to_frames_reports_no_change_as_not_applied(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev(1000, 2000, "a")]), name="aligned.ass")
    before = workspace.get(doc_id).to_bytes()
    result = T.ass_snap_to_frames(None, fps=25, doc_id=doc_id)
    # 1000 ms and 2000 ms are exactly frames 25 and 50: nothing to write
    assert result["considered"] == 2
    assert result["count"] == 0
    assert result["changes"] == []
    assert result["applied"] is False
    assert workspace.get(doc_id).to_bytes() == before
    assert workspace.undo_depth(doc_id)[0] == 0


def test_load_keyframes_from_times_frames_and_path(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:00.00", "0:00:04.00", "a")], fps=25))

    explicit = T.ass_load_keyframes(times_ms=[0, 1000, 2000], doc_id=doc_id)
    assert explicit["source"] == "times_ms"
    assert explicit["count"] == 3
    assert workspace.keyframes == [0, 1000, 2000]
    assert explicit["first_ms"] == 0 and explicit["last_ms"] == 2000

    from_frames = T.ass_load_keyframes(frames=[0, 25, 50], doc_id=doc_id)
    assert from_frames["source"] == "frames"
    assert from_frames["keyframes_ms"] == [0, 1000, 2000]
    assert workspace.keyframes == [0, 1000, 2000]  # stored on the workspace

    media = tmp_path / "clip.mkv"
    media.write_bytes(b"")
    with pytest.raises(ToolError):
        T.ass_load_keyframes(path=str(media), doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_load_keyframes(path=str(tmp_path / "missing.mkv"), doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_load_keyframes(doc_id=doc_id)  # no path and no workspace video


def test_load_keyframes_frames_needs_an_fps(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:00.00", "0:00:04.00", "a")]))
    with pytest.raises(ToolError):
        T.ass_load_keyframes(frames=[0, 10], doc_id=doc_id)


def test_snap_to_keyframes_reports_each_old_to_new(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev(900, 2100, "a")], fps=25))
    T.ass_load_keyframes(times_ms=[0, 1000, 2000, 3000], doc_id=doc_id)

    nearest = T.ass_snap_to_keyframes(None, doc_id=doc_id)
    assert nearest["keyframes_loaded"] == 4
    assert nearest["applied"] is True
    assert times(doc_id, 0) == (1000, 2000)
    assert [(c["old_ms"], c["new_ms"], c["keyframe_ms"]) for c in nearest["changes"]] == [
        (900, 1000, 1000),
        (2100, 2000, 2000),
    ]

    previous = T.ass_snap_to_keyframes(None, mode="previous", doc_id=doc_id)
    assert previous["mode"] == "previous"
    assert times(doc_id, 0) == (1000, 2000)  # already on keyframes

    nxt = T.ass_snap_to_keyframes(None, mode="next", doc_id=doc_id)
    assert nxt["count"] == 0  # nothing to do when already exact

    forward = T.ass_snap_to_keyframes(None, mode="next", forward_only=True, doc_id=doc_id)
    assert forward["forward_only"] is True


def test_snap_to_keyframes_respects_forward_only_and_max_distance(
    tmp_path: pathlib.Path,
) -> None:
    doc_id = open_text(tmp_path, make_ass([ev(900, 2100, "a")], fps=25))
    T.ass_load_keyframes(times_ms=[0, 1000, 2000, 3000], doc_id=doc_id)

    forward = T.ass_snap_to_keyframes(None, mode="next", forward_only=True, doc_id=doc_id)
    assert times(doc_id, 0) == (1000, 3000)  # never moves a time earlier

    limited = T.ass_snap_to_keyframes(None, max_distance_ms=500, doc_id=doc_id)
    assert limited["count"] == 0  # everything is within 100 ms already
    assert limited["skipped"] == []

    T.ass_load_keyframes(times_ms=[5000, 6000], doc_id=doc_id)
    far = T.ass_snap_to_keyframes(None, max_distance_ms=200, doc_id=doc_id)
    assert far["count"] == 0
    assert [(s["index"], s["field"]) for s in far["skipped"]] == [(0, "Start"), (0, "End")]
    assert all("over max_distance_ms 200" in s["reason"] for s in far["skipped"])


def test_snap_to_keyframes_needs_keyframes_and_valid_modes(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev(900, 2100, "a")], fps=25))
    with pytest.raises(ToolError) as excinfo:
        T.ass_snap_to_keyframes(None, doc_id=doc_id)
    assert "keyframe" in str(excinfo.value)

    T.ass_load_keyframes(times_ms=[0, 1000], doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_snap_to_keyframes(None, mode="sideways", doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_snap_to_keyframes(None, which="middle", doc_id=doc_id)
    with pytest.raises(ToolError):
        T.ass_snap_to_keyframes(None, mode="previous", forward_only=True, doc_id=doc_id)


# --------------------------------------------------------------------------- #
# timecodes
# --------------------------------------------------------------------------- #

V1_TIMECODES = """# timecode format v1
25.0
0,10,30.0
"""

V2_TIMECODES = """# timecode format v2
0.000
41.708
83.417
125.125
166.833
"""


def test_read_timecodes_v1(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "v1.timecodes"
    path.write_text(V1_TIMECODES, encoding="utf-8")
    result = T.ass_read_timecodes(str(path))
    assert result["version"] == 1
    assert result["default_fps"] == 25.0
    assert result["fps_changes"] == [{"start_frame": 0, "end_frame": 10, "fps": 30.0}]
    assert result["segments"] == [
        {"start_frame": 0, "end_frame": 10, "start_ms": 0.0, "end_ms": 333.333, "fps": 30.0},
        {"start_frame": 10, "end_frame": None, "start_ms": 333.333, "end_ms": None, "fps": 25.0},
    ]
    assert workspace.timecodes == result


def test_read_timecodes_v2(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "v2.timecodes"
    path.write_text(V2_TIMECODES, encoding="utf-8")
    result = T.ass_read_timecodes(str(path))
    assert result["version"] == 2
    assert result["frame_count"] == 5
    assert result["times_ms"][1] == 41.708
    assert result["default_fps"] == pytest.approx(23.976, abs=1e-3)
    assert result["segments"] == [
        {
            "start_frame": 0,
            "end_frame": 5,
            "start_ms": 0.0,
            "end_ms": None,
            "fps": pytest.approx(23.976, abs=1e-3),
        }
    ]


def test_read_timecodes_v2_splits_constant_rate_runs(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "mixed.timecodes"
    path.write_text("# timecode format v2\n0.000\n40.000\n80.000\n200.000\n", encoding="utf-8")
    result = T.ass_read_timecodes(str(path))
    assert result["segments"] == [
        {"start_frame": 0, "end_frame": 2, "start_ms": 0.0, "end_ms": 80.0, "fps": 25.0},
        {
            "start_frame": 2,
            "end_frame": 4,
            "start_ms": 80.0,
            "end_ms": None,
            "fps": pytest.approx(8.333333, abs=1e-4),
        },
    ]
    assert [(c["start_frame"], c["end_frame"], c["fps"]) for c in result["fps_changes"]] == [
        (0, 2, 25.0),
        (2, 4, pytest.approx(8.333333, abs=1e-4)),
    ]


def test_read_timecodes_reports_the_offending_line(tmp_path: pathlib.Path) -> None:
    bad_v2 = tmp_path / "bad-v2.timecodes"
    bad_v2.write_text("# timecode format v2\n0.000\nnope\n", encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        T.ass_read_timecodes(str(bad_v2))
    assert ":3:" in str(excinfo.value) and "nope" in str(excinfo.value)

    bad_v1 = tmp_path / "bad-v1.timecodes"
    bad_v1.write_text("# timecode format v1\nfast\n", encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        T.ass_read_timecodes(str(bad_v1))
    assert ":2:" in str(excinfo.value)

    # a v1 override is 'start,end,fps' (or 'start,fps' for the tail of the
    # file): anything else is malformed and names the line
    bad_override = tmp_path / "bad-override.timecodes"
    bad_override.write_text("# timecode format v1\n25\n0,10,oops\n", encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        T.ass_read_timecodes(str(bad_override))
    assert ":3:" in str(excinfo.value) and "oops" in str(excinfo.value)

    truncated = tmp_path / "truncated.timecodes"
    truncated.write_text("# timecode format v1\n25\n0\n", encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        T.ass_read_timecodes(str(truncated))
    assert ":3:" in str(excinfo.value)

    headerless = tmp_path / "headerless.timecodes"
    headerless.write_text("25.0\n0,10,30.0\n", encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        T.ass_read_timecodes(str(headerless))
    assert "timecode format" in str(excinfo.value)

    with pytest.raises(ToolError):
        T.ass_read_timecodes(str(tmp_path / "missing.timecodes"))


def test_timecodes_conversions_honour_the_loaded_file(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev(0, 1000, "a")], fps=25), name="conv.ass")
    fallback = T.ass_frame_from_timecodes(2, doc_id=doc_id)
    assert fallback["method"] == "fps (document FPS)"
    assert fallback["used_timecodes"] is False
    assert fallback["ms"] == 80

    path = tmp_path / "v1.timecodes"
    path.write_text(V1_TIMECODES, encoding="utf-8")
    T.ass_read_timecodes(str(path))

    frame = T.ass_frame_from_timecodes(2, doc_id=doc_id)
    assert frame["method"] == "timecodes v1"
    assert frame["used_timecodes"] is True
    assert frame["ms"] == 67  # 2 * 1000/30 rounded to ms
    assert frame["exact_ms"] == pytest.approx(66.667, abs=1e-3)

    late = T.ass_frame_from_timecodes(12, doc_id=doc_id)
    # frame 10 ends the 30 fps run at 333.333 ms; two 25 fps frames later
    assert late["ms"] == 413
    assert late["exact_ms"] == pytest.approx(413.333, abs=1e-3)

    at_ms = T.ass_ms_from_timecodes(90, doc_id=doc_id)
    assert at_ms["method"] == "timecodes v1"
    assert (at_ms["frame"], at_ms["frame_start_ms"]) == (3, 100)
    assert T.ass_ms_from_timecodes(400, doc_id=doc_id)["frame"] == 12  # 333.333 + 2*40


def test_write_timecodes_writes_into_the_output_dir(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:00.00", "0:00:04.00", "a")], fps=25))
    workspace.output_dir = tmp_path / "out"

    v2 = T.ass_write_timecodes(doc_id=doc_id)
    written = pathlib.Path(v2["path"])
    assert written.parent == workspace.output_dir
    assert v2["version"] == 2
    assert v2["fps"] == 25.0
    assert v2["preview"][0] == "# timecode format v2"
    lines = written.read_text(encoding="utf-8").splitlines()
    assert v2["frame_count"] == 101  # floor(4000 * 25 / 1000) + 1
    assert len(lines) == v2["frame_count"] + 1
    assert lines[1] == "0.000000"
    assert lines[2] == "40.000000"  # frame 1 at 25 fps

    reloaded = T.ass_read_timecodes(str(written))
    assert reloaded["frame_count"] == 101

    v1 = T.ass_write_timecodes(path="explicit.timecodes", v2=False, doc_id=doc_id)
    assert pathlib.Path(v1["path"]).name == "explicit.timecodes"
    assert pathlib.Path(v1["path"]).parent == workspace.output_dir
    assert v1["preview"] == ["# timecode format v1", "25.000000"]  # no overrides needed

    # a relative path is resolved inside output_dir, subdirectories included
    nested = T.ass_write_timecodes(path="sub/inner.timecodes", doc_id=doc_id)
    assert pathlib.Path(nested["path"]) == workspace.output_dir / "sub" / "inner.timecodes"
    assert pathlib.Path(nested["path"]).is_file()


def test_write_timecodes_reuses_the_loaded_rate_and_runs(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:00.00", "0:00:04.00", "a")]))
    workspace.output_dir = tmp_path / "out"
    path = tmp_path / "v1.timecodes"
    path.write_text(V1_TIMECODES, encoding="utf-8")
    T.ass_read_timecodes(str(path))

    result = T.ass_write_timecodes(v2=False, doc_id=doc_id)
    assert result["fps"] == 25.0
    assert result["fps_source"] == "loaded timecodes"
    assert result["preview"][2] == "0,10,30.000000"


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #


def build_qc_document(tmp_path: pathlib.Path, name: str = "qc.ass") -> str:
    """A document carrying every defect the brief lists."""
    return open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:00.00", "zero"),  # zero_duration
                ev("0:00:01.00", "0:00:00.50", "backwards"),  # negative_duration
                ev("0:00:02.00", "0:00:02.10", "short"),  # too_short (and cps_extreme)
                ev("0:00:03.00", "0:00:20.00", "long"),  # too_long
                ev("0:00:21.00", "0:00:22.00", "x" * 50),  # too_many_chars + cps_extreme
                ev("0:00:23.00", "0:00:24.00", "a\\Nb\\Nc"),  # too_many_lines
                ev("0:00:25.00", "0:00:26.00", "y" * 22),  # cps_high
                ev("0:00:27.00", "0:00:28.00", "over"),  # overlap_same_layer ...
                ev("0:00:27.50", "0:00:29.00", "lap"),  # ... with the previous line
                ev("0:00:29.05", "0:00:30.00", "gap"),  # gap_tiny against line 8
                ev("0:00:31.00", "0:00:32.00", ""),  # empty_text
                ev("0:00:33.00", "0:00:34.00", "hi", style="Ghost"),  # style_missing
                ev("0:00:35.00", "0:00:36.00", r"{\b1 never closed"),  # unclosed block
                ev("0:00:37.00", "0:00:38.00", r"{\bogus1}tag"),  # unknown_tag
                ev("0:00:39.00", "0:00:41.00", r"{\p1}m 0 0 l 10 0"),  # drawing_line
            ],
            styles=("Default",),
        ),
        name=name,
    )


def test_qc_reports_every_required_code(tmp_path: pathlib.Path) -> None:
    doc_id = build_qc_document(tmp_path)
    report = T.ass_qc(doc_id=doc_id)
    assert_json(report)
    codes = issue_codes(report)
    missing = REQUIRED_CODES - codes
    assert not missing, f"QC never reported {sorted(missing)}"
    assert "drawing_line" in codes

    for issue in report["issues"]:
        assert set(issue) == {"code", "severity", "index", "message", "details"}
        assert issue["severity"] in ("info", "warning", "error")
        assert isinstance(issue["message"], str) and issue["message"]

    severities = {
        code: {i["severity"] for i in report["issues"] if i["code"] == code}
        for code in REQUIRED_CODES | {"drawing_line"}
    }
    assert severities["zero_duration"] == {"error"}
    assert severities["negative_duration"] == {"error"}
    assert severities["style_missing"] == {"error"}
    assert severities["unclosed_override_block"] == {"error"}
    assert severities["cps_extreme"] == {"error"}
    assert severities["cps_high"] == {"warning"}
    assert severities["too_short"] == {"warning"}
    assert severities["too_long"] == {"warning"}
    assert severities["too_many_chars"] == {"warning"}
    assert severities["too_many_lines"] == {"warning"}
    assert severities["overlap_same_layer"] == {"warning"}
    assert severities["gap_tiny"] == {"warning"}
    assert severities["unknown_tag"] == {"warning"}
    assert severities["empty_text"] == {"warning"}
    assert severities["drawing_line"] == {"info"}
    assert report["summary"]["ok"] is False
    assert report["summary"]["clean"] is False
    assert report["summary"]["by_code"] == {
        code: len([i for i in report["issues"] if i["code"] == code]) for code in codes
    }


def test_qc_cps_ignores_override_tags_and_drawings(tmp_path: pathlib.Path) -> None:
    tagged = r"{\an8\pos(10,20)}Hello {\i1}world{\i0}"
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:01.00", tagged),  # 11 visible chars in 1 s
                ev("0:00:02.00", "0:00:04.00", tagged),  # 11 visible chars in 2 s
                ev("0:00:05.00", "0:00:06.00", r"{\p1}m 0 0 l 100 0 100 100{\p0}"),
            ]
        ),
        name="cps.ass",
    )
    report = T.ass_qc(doc_id=doc_id, cps_warn=20.0, cps_max=25.0)
    by_index: dict[int, set[str]] = {}
    for issue in report["issues"]:
        by_index.setdefault(issue["index"], set()).add(issue["code"])

    # the raw line is 38 characters long; only 11 are visible, so a tag-blind
    # counter would report 38 cps and a tag-aware one reports 11
    assert len(tagged) > 25
    assert T.ass_cps(0, doc_id=doc_id)["lines"][0]["cps"] == 11.0
    assert not ({"cps_high", "cps_extreme"} & by_index.get(0, set()))
    assert not ({"cps_high", "cps_extreme"} & by_index.get(1, set()))
    assert "too_many_chars" not in by_index.get(0, set())
    assert by_index[2] == {"drawing_line"}  # drawings are info only
    assert "empty_text" not in by_index[2]  # a drawing is not empty text
    assert report["summary"]["drawings"] == 1

    english = T.ass_reading_speed(0, doc_id=doc_id)
    assert english["characters"] == 11
    assert english["plain_text"] == "Hello world"
    assert english["cps"] == 11.0

    drawing = T.ass_reading_speed(2, doc_id=doc_id)
    assert drawing["drawing"] is True
    assert drawing["characters"] == 0


def test_qc_cps_thresholds_count_only_visible_characters(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:01.00", "twentyish"[:10]),  # 10 chars / 1 s = 10 cps
                ev("0:00:02.00", "0:00:03.00", "x" * 21),  # 21 cps
                ev("0:00:04.00", "0:00:05.00", "x" * 30),  # 30 cps
                ev("0:00:06.00", "0:00:08.00", r"{\k10}a{\k20}bb"),  # 3 chars / 2 s
            ]
        ),
    )
    report = T.ass_qc(doc_id=doc_id, cps_warn=20.0, cps_max=25.0)
    assert "cps_high" not in issue_codes_for(report, 0)
    assert issue_codes_for(report, 1) == {"cps_high"}
    assert issue_codes_for(report, 2) == {"cps_extreme"}
    assert not ({"cps_high", "cps_extreme"} & issue_codes_for(report, 3))
    assert T.ass_reading_speed(3, doc_id=doc_id)["characters"] == 3


def issue_codes_for(report, index: int) -> set[str]:
    return {i["code"] for i in report["issues"] if i["index"] == index}


def test_qc_overlap_is_per_layer(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:05.00", "layer 0", layer=0),
                ev("0:00:01.00", "0:00:06.00", "layer 1", layer=1),  # different layer: fine
            ]
        ),
    )
    assert "overlap_same_layer" not in issue_codes(T.ass_qc(doc_id=doc_id))
    assert T.ass_check_overlaps(doc_id=doc_id)["count"] == 0

    same = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:05.00", "a", layer=0),
                ev("0:00:01.00", "0:00:06.00", "b", layer=0),
            ]
        ),
        name="same.ass",
    )
    assert "overlap_same_layer" in issue_codes(T.ass_qc(doc_id=same))
    pairs = T.ass_check_overlaps(doc_id=same)
    assert pairs["count"] == 1
    assert pairs["pairs"][0]["overlap_ms"] == 4000
    assert pairs["pairs"][0]["layer"] == "0"

    loose = T.ass_check_overlaps(doc_id=doc_id, layer_strict=False)
    assert loose["layer_strict"] is False
    assert loose["count"] == 1
    assert loose["pairs"][0]["overlap_ms"] == 4000


def test_check_overlaps_ordering_and_tolerance(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:02.00", "a"),
                ev("0:00:04.00", "0:00:06.00", "b"),
                ev("0:00:01.00", "0:00:05.00", "c"),  # overlaps a (1000) and b (1000)
            ]
        ),
    )
    pairs = T.ass_check_overlaps(doc_id=doc_id)
    assert [p["overlap_ms"] for p in pairs["pairs"]] == [1000, 1000]
    assert [(p["a"], p["b"]) for p in pairs["pairs"]] == [(0, 2), (2, 1)]
    assert pairs["pairs"][0]["a_start_ms"] <= pairs["pairs"][1]["a_start_ms"]

    tolerated = T.ass_check_overlaps(doc_id=doc_id, tolerate_ms=1000)
    assert tolerated["pairs"] == []
    assert tolerated["tolerate_ms"] == 1000


def test_qc_check_flags_disable_whole_families(tmp_path: pathlib.Path) -> None:
    doc_id = build_qc_document(tmp_path, name="qc2.ass")
    report = T.ass_qc(
        doc_id=doc_id,
        check_overlaps=False,
        check_gaps=False,
        check_empty=False,
        check_styles=False,
        check_tags=False,
    )
    codes = issue_codes(report)
    for code in (
        "overlap_same_layer",
        "gap_tiny",
        "empty_text",
        "style_missing",
        "unclosed_override_block",
        "unknown_tag",
    ):
        assert code not in codes
    # the plain timing checks still ran
    assert {"zero_duration", "negative_duration", "too_short"} <= codes


def test_qc_selection_limits_the_checked_lines(tmp_path: pathlib.Path) -> None:
    doc_id = build_qc_document(tmp_path, name="qc3.ass")
    report = T.ass_qc(0, doc_id=doc_id)
    assert report["summary"]["lines_checked"] == 1
    # index 0 has zero duration, so the text checks are skipped for it
    assert issue_codes(report) == {"zero_duration"}

    ranged = T.ass_qc({"range": [0, 1]}, doc_id=doc_id)
    assert ranged["summary"]["lines_checked"] == 2

    # the empty-text line is reported on its own when it is selected
    empty = T.ass_qc(10, doc_id=doc_id)
    assert empty["summary"]["lines_checked"] == 1
    assert issue_codes(empty) == {"empty_text"}


def test_qc_zero_and_negative_duration_details(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:01.00", "0:00:01.00", "a"),
                ev("0:00:02.00", "0:00:01.00", "b"),
            ]
        ),
    )
    report = T.ass_qc(doc_id=doc_id)
    zero = next(i for i in report["issues"] if i["code"] == "zero_duration")
    negative = next(i for i in report["issues"] if i["code"] == "negative_duration")
    assert zero["details"]["start_ms"] == zero["details"]["end_ms"] == 1000
    assert negative["details"]["duration_ms"] == -1000
    assert "duration_ms" in negative["details"]


def test_qc_reports_a_tiny_gap_once_per_pair(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:01.00", "a"),
                ev("0:00:01.05", "0:00:02.00", "b"),
                ev("0:00:02.50", "0:00:03.00", "c"),  # 500 ms gap: fine
            ]
        ),
    )
    gaps = [i for i in T.ass_qc(doc_id=doc_id)["issues"] if i["code"] == "gap_tiny"]
    assert len(gaps) == 1
    assert gaps[0]["details"]["gap_ms"] == 50
    assert gaps[0]["details"]["b"] == 1


# --------------------------------------------------------------------------- #
# fix timing
# --------------------------------------------------------------------------- #


def test_fix_timing_dry_run_default_does_not_write(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:00.00", "0:00:00.10", "a")]))
    before = workspace.get(doc_id).to_bytes()
    result = T.ass_fix_timing(doc_id=doc_id)
    assert result["dry_run"] is True
    assert result["applied"] is False
    assert result["parameters"]["min_duration_ms"] == 300
    assert result["count"] == 1
    change = result["changes"][0]
    assert set(change) == {"index", "field", "old_ms", "new_ms", "old", "new", "reason"}
    assert change["field"] == "End"
    assert (change["old_ms"], change["new_ms"]) == (100, 300)
    assert "min_duration_ms" in change["reason"]
    assert workspace.get(doc_id).to_bytes() == before
    assert workspace.undo_depth(doc_id)[0] == 0


def test_fix_timing_applies_and_reports_the_change_list(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:00.10", "a"),  # extended to 300 ms
                ev("0:00:01.00", "0:00:01.20", "b"),  # extended, but capped at c
                ev("0:00:01.40", "0:00:02.00", "c"),
            ]
        ),
    )
    result = T.ass_fix_timing(doc_id=doc_id, dry_run=False, min_duration_ms=500)
    assert result["applied"] is True
    assert [(c["index"], c["old_ms"], c["new_ms"]) for c in result["changes"]] == [
        (0, 100, 500),
        (1, 1200, 1400),  # stopped at the next line rather than 1500
    ]
    assert times(doc_id, 0) == (0, 500)
    assert times(doc_id, 1) == (1000, 1400)
    assert times(doc_id, 2) == (1400, 2000)

    around = T.ass_fix_timing(doc_id=doc_id, dry_run=False, min_duration_ms=500, keep_gaps_ms=100)
    assert [(c["index"], c["new_ms"]) for c in around["changes"]] == [(1, 1300)]
    assert times(doc_id, 1) == (1000, 1300)


def test_fix_timing_target_cps_and_blocked(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:01.00", "x" * 30),  # 30 cps -> 3 s at 10 cps
                ev("0:00:05.00", "0:00:05.10", "y"),  # under the 300 ms minimum
                ev("0:00:07.00", "0:00:08.00", "z"),  # already fine
            ]
        ),
        name="target.ass",
    )
    result = T.ass_fix_timing(doc_id=doc_id, target_cps=10.0, dry_run=False)
    assert result["count"] == 2
    assert result["applied"] is True
    assert result["blocked"] == []
    assert [(c["index"], c["field"], c["old_ms"], c["new_ms"]) for c in result["changes"]] == [
        (0, "End", 1000, 3000),
        (1, "End", 5100, 5300),
    ]
    assert "target_cps" in result["changes"][0]["reason"]
    assert "min_duration_ms" in result["changes"][1]["reason"]
    assert times(doc_id, 0) == (0, 3000)
    assert times(doc_id, 1) == (5000, 5300)
    assert times(doc_id, 2) == (7000, 8000)

    # a line whose successor already starts at its end has nowhere to grow
    blocked_doc = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:00.50", "a"),
                ev("0:00:00.50", "0:00:02.50", "b"),
            ]
        ),
        name="blocked.ass",
    )
    blocked = T.ass_fix_timing(doc_id=blocked_doc, min_duration_ms=1500, dry_run=False)
    assert blocked["changes"] == []
    assert blocked["applied"] is False
    assert len(blocked["blocked"]) == 1
    assert (blocked["blocked"][0]["index"], blocked["blocked"][0]["wanted_ms"]) == (0, 500)
    assert blocked["blocked"][0]["limit_ms"] == 500
    assert "no room" in blocked["blocked"][0]["reason"]
    assert times(blocked_doc, 0) == (0, 500)

    with pytest.raises(ToolError):
        T.ass_fix_timing(doc_id=doc_id, target_cps=0)
    with pytest.raises(ToolError):
        T.ass_fix_timing(doc_id=doc_id, keep_gaps_ms=-1)


def test_fix_timing_avoids_crossing_other_lines_when_disabled(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.00", "0:00:00.10", "a"),
                ev("0:00:00.20", "0:00:02.00", "b"),
            ]
        ),
    )
    safe = T.ass_fix_timing(doc_id=doc_id, min_duration_ms=1000)
    assert safe["changes"][0]["new_ms"] == 200

    unsafe = T.ass_fix_timing(doc_id=doc_id, min_duration_ms=1000, avoid_overlap=False, dry_run=False)
    assert unsafe["changes"][0]["new_ms"] == 1000
    assert times(doc_id, 0) == (0, 1000)


# --------------------------------------------------------------------------- #
# silence alignment (silencedetect is monkeypatched: no ffmpeg, no media)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_silencedetect(monkeypatch):
    """Replace ``render.silencedetect`` with a deterministic stub."""
    calls: list[dict] = []

    def fake(path, *, noise_db=-45.0, min_duration_s=0.15, **kwargs):
        calls.append(
            {"path": path, "noise_db": noise_db, "min_duration_s": min_duration_s, **kwargs}
        )
        return [
            {"start_ms": 0.0, "end_ms": 1200.0, "duration_ms": 1200.0},
            {"start_ms": 4000.0, "end_ms": 5000.0, "duration_ms": 1000.0},
            {"start_ms": 9000.0},  # file ends inside a silence: no end_ms
        ]

    monkeypatch.setattr(T.R, "silencedetect", fake)
    return calls


def silence_document(tmp_path: pathlib.Path):
    media = tmp_path / "media.mkv"
    media.write_bytes(b"")
    doc_id = open_text(
        tmp_path,
        make_ass(
            [
                ev("0:00:00.50", "0:00:02.00", "starts in silence"),
                ev("0:00:02.50", "0:00:03.00", "starts on audio"),
                ev("0:00:04.50", "0:00:05.50", "also in silence"),
            ]
        ),
    )
    return doc_id, str(media)


def test_align_to_silence_reports_without_touching_the_document(
    tmp_path: pathlib.Path, fake_silencedetect
) -> None:
    doc_id, media = silence_document(tmp_path)
    before = workspace.get(doc_id).to_bytes()
    result = T.ass_align_to_silence(video=media, doc_id=doc_id)
    assert result["applied"] is False
    assert "not modified" in result["note"]
    assert result["silences"] == [
        {"start_ms": 0, "end_ms": 1200, "duration_ms": 1200},
        {"start_ms": 4000, "end_ms": 5000, "duration_ms": 1000},
    ]
    assert result["open_silences"] == [9000]
    assert [s["index"] for s in result["suggestions"]] == [0, 2]
    assert result["suggestions"][0]["delta_ms"] == 700
    assert result["suggestions"][0]["new_start_ms"] == 1200
    assert result["suggestions"][1]["new_end_ms"] == 6000
    assert workspace.get(doc_id).to_bytes() == before
    assert fake_silencedetect[0]["noise_db"] == -45.0


def test_align_lines_to_silence_dry_run_then_applies(
    tmp_path: pathlib.Path, fake_silencedetect
) -> None:
    doc_id, media = silence_document(tmp_path)
    before = workspace.get(doc_id).to_bytes()

    dry = T.ass_align_lines_to_silence(video=media, doc_id=doc_id)
    assert dry["dry_run"] is True
    assert dry["applied"] is False
    assert dry["count"] == 2
    assert workspace.get(doc_id).to_bytes() == before

    applied = T.ass_align_lines_to_silence(video=media, doc_id=doc_id, dry_run=False)
    assert applied["applied"] is True
    assert "applied 2 shift(s)" in applied["note"]
    assert times(doc_id, 0) == (1200, 2700)
    assert times(doc_id, 2) == (5000, 6000)
    assert times(doc_id, 1) == (2500, 3000)  # untouched: it starts on audio
    assert workspace.get(doc_id).to_bytes() != before


def test_align_lines_to_silence_shift_only_start_shortens_the_line(
    tmp_path: pathlib.Path, fake_silencedetect
) -> None:
    doc_id, media = silence_document(tmp_path)
    dry = T.ass_align_lines_to_silence(video=media, doc_id=doc_id, shift_only="start")
    assert dry["shift_only"] == "start"
    first = dry["suggestions"][0]
    assert (first["new_start_ms"], first["new_end_ms"]) == (1200, 2000)

    with pytest.raises(ToolError):
        T.ass_align_lines_to_silence(video=media, doc_id=doc_id, shift_only="end")


def test_align_lines_to_silence_respects_max_shift_and_empty_plan(
    tmp_path: pathlib.Path, fake_silencedetect
) -> None:
    doc_id, media = silence_document(tmp_path)
    limited = T.ass_align_lines_to_silence(video=media, doc_id=doc_id, max_shift_ms=100)
    assert limited["suggestions"] == []
    assert limited["count"] == 0
    assert "nothing applied" in T.ass_align_lines_to_silence(
        video=media, doc_id=doc_id, max_shift_ms=100, dry_run=False
    )["note"]


def test_silence_tools_need_a_real_media_file(tmp_path: pathlib.Path) -> None:
    doc_id = open_text(tmp_path, make_ass([ev("0:00:00.00", "0:00:01.00", "a")]))
    with pytest.raises(ToolError) as excinfo:
        T.ass_align_to_silence(doc_id=doc_id)
    assert "no media to analyse" in str(excinfo.value)

    with pytest.raises(ToolError) as excinfo:
        T.ass_align_to_silence(video=str(tmp_path / "nope.mkv"), doc_id=doc_id)
    assert "media file not found" in str(excinfo.value)


def test_silence_tools_turn_render_errors_into_tool_errors(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    doc_id, media = silence_document(tmp_path)

    def boom(*args, **kwargs):
        raise T.R.RenderError("ffmpeg is not installed")

    monkeypatch.setattr(T.R, "silencedetect", boom)
    with pytest.raises(ToolError) as excinfo:
        T.ass_align_lines_to_silence(video=media, doc_id=doc_id, dry_run=False)
    assert "silencedetect failed" in str(excinfo.value)
    assert times(doc_id, 0) == (500, 2000)  # nothing was written


# --------------------------------------------------------------------------- #
# mutation contract: snapshot before every change, dry runs never snapshot
# --------------------------------------------------------------------------- #


def test_every_mutation_snapshots_so_undo_restores_the_bytes(tmp_path: pathlib.Path) -> None:
    text = make_ass(
        [
            ev("0:00:01.00", "0:00:02.00", "a"),
            ev("0:00:05.00", "0:00:06.00", "b"),
        ]
    )

    def scenario(name: str, call, source: str = text):
        doc_id = open_text(tmp_path, source, name=name)
        original = workspace.get(doc_id).to_bytes()
        call(doc_id)
        assert workspace.get(doc_id).to_bytes() != original, f"{name} did not change anything"
        depth, _ = workspace.undo_depth(doc_id)
        assert depth >= 1, f"{name} mutated without workspace.snapshot()"
        assert workspace.undo(doc_id) is True
        assert workspace.get(doc_id).to_bytes() == original

    scenario("shift.ass", lambda d: T.ass_shift_times(None, 250, doc_id=d))
    scenario("scale.ass", lambda d: T.ass_scale_times(None, 1.5, doc_id=d))
    scenario("set.ass", lambda d: T.ass_set_times(0, start_ms=0, doc_id=d))
    scenario(
        "durations.ass",
        lambda d: T.ass_set_durations(None, min_ms=4000, doc_id=d, dry_run=False),
    )
    scenario(
        "frames.ass",
        lambda d: T.ass_snap_to_frames(None, fps=24, doc_id=d),
        # deliberately not on 24 fps frame boundaries, or there is nothing to write
        source=make_ass(
            [
                ev("0:00:01.01", "0:00:02.03", "a"),
                ev("0:00:05.05", "0:00:06.07", "b"),
            ]
        ),
    )
    scenario(
        "keyframes.ass",
        lambda d: (
            T.ass_load_keyframes(times_ms=[0, 1500, 3000], doc_id=d),
            T.ass_snap_to_keyframes(None, doc_id=d),
        ),
    )
    scenario(
        "fix.ass",
        lambda d: T.ass_fix_timing(doc_id=d, dry_run=False, min_duration_ms=3000),
    )


def test_dry_runs_never_snapshot_or_write(tmp_path: pathlib.Path) -> None:
    text = make_ass([ev("0:00:01.00", "0:00:01.50", "a")])
    doc_id = open_text(tmp_path, text, name="dry.ass")
    original = workspace.get(doc_id).to_bytes()

    T.ass_set_durations(None, min_ms=5000, doc_id=doc_id)
    T.ass_fix_timing(doc_id=doc_id, min_duration_ms=5000)
    assert workspace.undo_depth(doc_id) == (0, 0)
    assert workspace.get(doc_id).to_bytes() == original

    workspace.undo(doc_id)  # must be a harmless no-op
    assert workspace.get(doc_id).to_bytes() == original


def test_silence_apply_snapshots(tmp_path: pathlib.Path, fake_silencedetect) -> None:
    doc_id, media = silence_document(tmp_path)
    original = workspace.get(doc_id).to_bytes()
    T.ass_align_lines_to_silence(video=media, doc_id=doc_id, dry_run=False)
    assert workspace.undo_depth(doc_id)[0] == 1
    assert workspace.undo(doc_id) is True
    assert workspace.get(doc_id).to_bytes() == original
