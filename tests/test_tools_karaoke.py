"""Tool-layer tests for :mod:`aegisub_mcp.tools.karaoke_tools`.

The tools are called directly (no MCP transport): every ``ass_*`` function is a
module-level callable returning a JSON-serialisable dict, which is exactly what
``register(mcp, ws)`` hands to FastMCP.

Expectations here are hand-computed from the ASS centisecond model, e.g. a line
from ``0:00:01.00`` to ``0:00:02.50`` is 1500 ms = 150 cs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegisub_mcp.tools import base
from aegisub_mcp.tools import karaoke_tools as KT
from aegisub_mcp.tools.base import ToolError, workspace

REAL_FIXTURES = Path(__file__).parent / "fixtures" / "real"

HEADER = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: 1280\n"
    "PlayResY: 720\n"
    "\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
    "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,"
    "100,100,0,0,1,2,1,2,40,40,30,1\n"
    "\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


@pytest.fixture(autouse=True)
def fresh_workspace(tmp_path):
    """The module-level singleton is process-wide: reset it between tests."""
    ws = workspace
    ws._docs.clear()
    ws._paths.clear()
    ws._order.clear()
    ws._undo.clear()
    ws._redo.clear()
    ws._counter = 0
    ws.current = None
    ws.selection = []
    ws.output_dir = tmp_path / "output"
    ws.output_dir.mkdir(parents=True, exist_ok=True)
    yield ws
    ws._docs.clear()
    ws._paths.clear()
    ws._order.clear()
    ws._undo.clear()
    ws._redo.clear()
    ws.current = None
    ws.selection = []


def write_ass(tmp_path, dialogues, name="karaoke.ass", layer=0, style="Default",
              actor="", effect=""):
    """Write a minimal ASS file and return its path.

    ``dialogues`` is a list of ``(start, end, text)`` triples.
    """
    body = HEADER + "".join(
        f"Dialogue: {layer},{start},{end},{style},{actor},0,0,0,{effect},{text}\n"
        for start, end, text in dialogues
    )
    path = tmp_path / name
    path.write_bytes(body.encode("utf-8"))
    return path


def open_ass(tmp_path, dialogues, **kwargs):
    """Write a document and open it; returns ``(doc_id, path, doc)``."""
    path = write_ass(tmp_path, dialogues, **kwargs)
    did = workspace.open(path)
    return did, path, workspace.get(did)


def events_text(doc_id):
    return [entry.text for entry in workspace.get(doc_id).events()]


# --------------------------------------------------------------------------- split


def test_split_marker_three_syllables_with_hand_computed_counts():
    res = KT.ass_karaoke_split(text="ka|ra|o")
    assert res["text_source"] == "text"
    assert res["mode"] == "marker"
    assert res["syllable_count"] == 3
    assert [s["text"] for s in res["syllables"]] == ["ka", "ra", "o"]
    assert [s["char_count"] for s in res["syllables"]] == [2, 2, 1]
    assert [s["prefix"] for s in res["syllables"]] == ["", "", ""]
    assert [s["index"] for s in res["syllables"]] == [0, 1, 2]
    # ``plain_text`` is the raw visible text (the marker is still a character in it);
    # ``expected_visible_text`` is what the syllables must concatenate to, and the
    # reconstruction check proves the marker was consumed as a separator.
    assert res["plain_text"] == "ka|ra|o"
    assert res["expected_visible_text"] == "karao"
    assert res["reconstruction"] == {"ok": True, "reconstructed": "karao",
                                     "expected": "karao"}
    assert res["had_bom"] is False


def test_split_marker_from_line_index_keeps_tags_with_their_syllable(tmp_path):
    did, _path, _doc = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", r"{\i1}ka|ra|o")])
    res = KT.ass_karaoke_split(index=0, doc_id=did)
    assert res["text_source"] == "line"
    assert res["index"] == 0
    assert res["doc_id"] == did
    assert [s["text"] for s in res["syllables"]] == ["ka", "ra", "o"]
    assert res["syllables"][0]["prefix"] == r"{\i1}"
    assert res["syllables"][0]["prefix_tags"] == ["i"]
    assert res["syllables"][0]["raw"] == r"{\i1}ka"
    assert res["syllables"][1]["prefix"] == ""
    assert res["reconstruction"]["ok"] is True
    assert res["reconstruction"]["reconstructed"] == "karao"


def test_split_char_keeps_combining_marks_attached_thai_and_latin():
    # Thai: base ก + sara am + mai ek are ONE cluster, the next base is the next
    # syllable -> 2 syllables, 3 and 1 code points.
    #
    # NOTE (asscore bug, worked around here): karaoke._split_by_char tests
    # ``ch in _THAI_LEAD`` against a set of *ints*, so Thai leading vowels
    # (เ แ โ ใ ไ) never merge with their consonant.  That is only asserted from the
    # combining-mark side below, which does work; see the report for the upstream
    # detail.
    thai = "\u0e01\u0e33\u0e48\u0e01"
    res = KT.ass_karaoke_split(text=thai, mode="char")
    assert res["syllable_count"] == 2
    assert [s["text"] for s in res["syllables"]] == ["\u0e01\u0e33\u0e48", "\u0e01"]
    assert [s["char_count"] for s in res["syllables"]] == [3, 1]
    assert res["reconstruction"]["ok"] is True
    assert res["reconstruction"]["reconstructed"] == thai

    # Latin: decomposed "á" (a + U+0301) stays one syllable, like the precomposed
    # spelling, so no combining mark is ever orphaned.
    decomposed = "na\u0301korn"
    res_lat = KT.ass_karaoke_split(text=decomposed, mode="char")
    assert res_lat["reconstruction"]["reconstructed"] == decomposed
    assert any(s["text"] == "a\u0301" for s in res_lat["syllables"])
    precomposed = KT.ass_karaoke_split(text="n\u00e1korn", mode="char")
    assert res_lat["syllable_count"] == precomposed["syllable_count"]


def test_split_marker_thai_two_syllables():
    res = KT.ass_karaoke_split(text="\u0e01\u0e33|\u0e01\u0e48\u0e32")
    assert res["syllable_count"] == 2
    assert [s["char_count"] for s in res["syllables"]] == [2, 3]


def test_split_modes_word_and_regex():
    words = KT.ass_karaoke_split(text="ka ra o", mode="word")
    assert [s["text"] for s in words["syllables"]] == ["ka ", "ra ", "o"]

    rx = KT.ass_karaoke_split(text="ka/ra-o", mode="regex", pattern="[/-]")
    assert [s["text"] for s in rx["syllables"]] == ["ka", "ra", "o"]
    assert rx["reconstruction"]["ok"] is True


def test_split_error_paths():
    with pytest.raises(ToolError, match="mode must be one of"):
        KT.ass_karaoke_split(text="ka|ra", mode="nope")
    with pytest.raises(ToolError, match="needs pattern"):
        KT.ass_karaoke_split(text="ka|ra", mode="regex")
    with pytest.raises(ToolError, match="pass either text="):
        KT.ass_karaoke_split()
    # documented precedence: a raw string wins over an index
    both = KT.ass_karaoke_split(text="ka|ra", index=0)
    assert both["text_source"] == "text"
    assert both["syllable_count"] == 2


# --------------------------------------------------------------------------- get


def test_get_three_syllable_line_reports_the_discrepancy(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.50", r"{\k30}ka{\k20}ra{\k10}o")])
    res = KT.ass_karaoke_get(index=0, doc_id=did)
    assert res["text_source"] == "line"
    assert res["has_karaoke_tags"] is True
    assert res["untimed"] is False
    assert res["timings_source"] == "karaoke_tags"
    assert res["syllable_count"] == 3
    # 30 cs + 20 cs + 10 cs = 60 cs against a 150 cs line
    assert [s["duration_cs"] for s in res["syllables"]] == [30, 20, 10]
    assert [s["duration_ms"] for s in res["syllables"]] == [300, 200, 100]
    assert res["duration_sum_cs"] == 60
    assert res["duration_sum_ms"] == 600
    assert res["line_duration_cs"] == 150
    assert res["discrepancy_cs"] == -90
    assert res["matches_line_duration"] is False
    assert "reported, not stretched" in res["message"]
    assert "-90" in res["message"]
    # absolute times start at the line's own Start (1000 ms), not at zero
    assert [(s["start_ms"], s["end_ms"]) for s in res["syllables"]] == [
        (1000, 1300), (1300, 1500), (1500, 1600)]
    assert [s["text"] for s in res["syllables"]] == ["ka", "ra", "o"]
    assert res["kind_counts"] == {"k": 3}
    assert res["kinds"] == ["k"]
    assert json.dumps(res)  # JSON-serialisable


def test_get_untimed_line_is_reported_as_untimed(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "hello")])
    res = KT.ass_karaoke_get(index=0, doc_id=did)
    assert res["has_karaoke_tags"] is False
    assert res["untimed"] is True
    assert res["timings_source"] == "line_span"
    assert res["syllable_count"] == 1
    # an untimed line is reported as a single syllable spanning the whole line
    assert res["syllables"][0]["duration_cs"] == 150
    assert res["duration_sum_cs"] == res["line_duration_cs"] == 150
    assert res["discrepancy_cs"] == 0
    assert res["matches_line_duration"] is True
    assert res["syllables"][0]["start_ms"] == 1000
    assert res["syllables"][0]["end_ms"] == 2500


def test_get_mixes_karaoke_kinds(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:03.00", r"{\kf10}a{\ko20}b{\K30}c")])
    res = KT.ass_karaoke_get(index=0, doc_id=did)
    # \K is the Aegisub alias of \kf
    assert [s["kind"] for s in res["syllables"]] == ["kf", "ko", "kf"]
    assert res["kind_counts"] == {"kf": 2, "ko": 1}
    assert res["kinds"] == ["kf", "ko"]
    assert res["duration_sum_cs"] == 60
    assert res["line_duration_cs"] == 200
    assert res["discrepancy_cs"] == -140


def test_get_unknown_index_raises_toolerror(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra")])
    with pytest.raises(ToolError):
        KT.ass_karaoke_get(index=7, doc_id=did)


# --------------------------------------------------------------------------- generate


def test_generate_marker_three_syllables_exact_centisecond_sum(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    res = KT.ass_karaoke_generate(index=0, doc_id=did, mode="marker")
    assert res["text_source"] == "line"
    assert res["exact_sum_guaranteed"] is True
    assert res["sum_within_one_cs"] is True
    assert res["lines_changed"] == 1
    row = res["lines"][0]
    # 1500 ms span, char weights 2:2:1 -> 600/600/300 ms -> 60/60/30 cs
    assert row["span_ms"] == 1500
    assert row["span_cs"] == 150
    assert row["durations_cs"] == [60, 60, 30]
    assert row["duration_sum_cs"] == 150
    assert row["line_duration_cs"] == 150
    assert row["discrepancy_cs"] == 0
    assert row["matches_line_duration"] is True
    assert "exactly the line duration" in row["message"]
    assert row["text"] == r"{\k60}ka{\k60}ra{\k30}o"
    assert row["snapped"] is False
    # \k never touches the line's own times
    assert (row["start_ms"], row["end_ms"]) == (1000, 2500)
    assert events_text(did) == [r"{\k60}ka{\k60}ra{\k30}o"]
    assert [s["duration_ms"] for s in row["syllables"]] == [600, 600, 300]


def test_generate_rounding_remainder_goes_to_the_first_longest_syllable(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.00", "a|b|c")])
    res = KT.ass_karaoke_generate(index=0, doc_id=did)
    row = res["lines"][0]
    # 1000 ms / 3 = 333.33 ms each -> 33 cs, the missing 1 cs lands on index 0
    assert row["durations_cs"] == [34, 33, 33]
    assert row["duration_sum_cs"] == row["span_cs"] == row["line_duration_cs"] == 100
    assert row["text"] == r"{\k34}a{\k33}b{\k33}c"


def test_generate_char_link_gives_one_tag_per_visible_character(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.00", "ab|cd")])
    res = KT.ass_karaoke_generate(index=0, doc_id=did, link="char")
    row = res["lines"][0]
    assert [s["text"] for s in row["syllables"]] == ["a", "b", "c", "d"]
    assert row["durations_cs"] == [25, 25, 25, 25]
    assert row["duration_sum_cs"] == 100
    assert row["text"] == r"{\k25}a{\k25}b{\k25}c{\k25}d"


def test_generate_syl_link_puts_the_tag_after_the_syllable_prefix_tags(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", r"{\i1}ka|ra")])
    res = KT.ass_karaoke_generate(index=0, doc_id=did, link="syl")
    row = res["lines"][0]
    assert row["durations_cs"] == [75, 75]
    # the syllable's own override block stays in front of the \k tag
    assert row["text"] == r"{\i1}{\k75}ka{\k75}ra"


def test_generate_even_weights_ignore_length(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.00", "kaka|rara")])
    res = KT.ass_karaoke_generate(index=0, doc_id=did, weights="even")
    assert res["lines"][0]["durations_cs"] == [50, 50]


def test_generate_kind_and_min_cs(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.00", "ka|ra")])
    res = KT.ass_karaoke_generate(index=0, doc_id=did, kind="kf")
    assert res["kind"] == "kf"
    assert res["lines"][0]["text"] == r"{\kf50}ka{\kf50}ra"
    with pytest.raises(ToolError, match="kind must be one of"):
        KT.ass_karaoke_generate(index=0, doc_id=did, kind="zz")


def test_generate_span_too_short_for_min_cs_raises(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:01.02", "a|b|c")])
    with pytest.raises(ToolError) as excinfo:
        KT.ass_karaoke_generate(index=0, doc_id=did)
    message = str(excinfo.value)
    assert "2 cs" in message          # 20 ms span
    assert "3 syllable" in message    # three syllables at min_cs=1
    assert events_text(did) == ["a|b|c"]  # nothing was written


def test_generate_snap_to_line_clamps_an_overshooting_span(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra")])
    res = KT.ass_karaoke_generate(index=0, doc_id=did, start_ms=500, end_ms=3000,
                                  snap_to_line=True)
    row = res["lines"][0]
    assert row["snapped"] is True
    assert (row["start_ms"], row["end_ms"]) == (1000, 2500)
    assert row["span_cs"] == row["line_duration_cs"] == 150
    assert row["discrepancy_cs"] == 0
    assert row["text"] == r"{\k75}ka{\k75}ra"

    # with snapping off the same request is honoured and the mismatch reported
    res2 = KT.ass_karaoke_generate(index=0, doc_id=did, start_ms=500, end_ms=3000,
                                   snap_to_line=False)
    row2 = res2["lines"][0]
    assert row2["snapped"] is False
    assert (row2["start_ms"], row2["end_ms"]) == (500, 3000)
    assert row2["span_cs"] == 250
    assert row2["discrepancy_cs"] == 100
    assert row2["matches_line_duration"] is False
    assert "reported, not stretched" in row2["message"]


def test_generate_replace_existing_false_skips_timed_lines(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.50", r"{\k30}ka{\k20}ra")])
    res = KT.ass_karaoke_generate(index=0, doc_id=did, replace_existing=False)
    assert res["lines_changed"] == 0
    assert res["lines"][0]["skipped"] is True
    assert "replace_existing=False" in res["lines"][0]["reason"]
    assert events_text(did) == [r"{\k30}ka{\k20}ra"]


def test_generate_without_a_target_raises(tmp_path):
    open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra")])
    with pytest.raises(ToolError, match="no lines selected"):
        KT.ass_karaoke_generate()


def test_generate_then_set_timings_round_trip_loses_no_centisecond(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.53", "ka|ra|o")])
    generated = KT.ass_karaoke_generate(index=0, doc_id=did)["lines"][0]
    durations = generated["durations_cs"]
    assert sum(durations) == 153            # 1530 ms -> 153 cs exactly

    back = KT.ass_karaoke_set_timings(index=0, doc_id=did, timings=durations, unit="cs")
    row = back["lines"][0]
    assert row["durations_cs"] == durations
    assert row["duration_sum_cs"] == 153
    assert row["matches_line_duration"] is True
    assert events_text(did) == [generated["text"]]  # byte-identical after the trip


# --------------------------------------------------------------------------- set_timings


def test_set_timings_durations_report_a_mismatch_instead_of_stretching(tmp_path):
    did, _p, doc = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    res = KT.ass_karaoke_set_timings(index=0, doc_id=did, timings=[500, 400, 300])
    row = res["lines"][0]
    assert res["reported_not_stretched"] is True
    assert row["syllable_count"] == 3
    assert row["timing_count"] == 3
    assert row["padded"] is False
    assert row["durations_cs"] == [50, 40, 30]
    assert row["duration_sum_cs"] == 120
    assert row["line_duration_cs"] == 150
    assert row["discrepancy_cs"] == -30
    assert row["matches_line_duration"] is False
    assert "reported, not stretched" in row["message"]
    assert row["text"] == r"{\k50}ka{\k40}ra{\k30}o"
    assert events_text(did) == [r"{\k50}ka{\k40}ra{\k30}o"]
    # the tool only writes \k tags, never the line's own times
    entry = doc.events()[0]
    assert (entry.start_ms, entry.end_ms) == (1000, 2500)


def test_set_timings_strict_count_mismatch_names_both_counts(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    with pytest.raises(ToolError) as excinfo:
        KT.ass_karaoke_set_timings(index=0, doc_id=did, timings=[100, 200])
    message = str(excinfo.value)
    assert "timings count 2" in message
    assert "syllable count 3" in message
    assert "line 0" in message
    assert events_text(did) == ["ka|ra|o"]


def test_set_timings_non_strict_pads_with_zero(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    res = KT.ass_karaoke_set_timings(index=0, doc_id=did, timings=[500, 400],
                                     strict=False)
    row = res["lines"][0]
    assert row["padded"] is True
    assert row["syllable_count"] == 3
    assert row["timing_count"] == 2
    assert row["durations_cs"] == [50, 40, 0]
    assert row["text"] == r"{\k50}ka{\k40}ra{\k0}o"
    assert row["discrepancy_cs"] == -60


def test_set_timings_absolute_boundaries_and_monotonicity(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    res = KT.ass_karaoke_set_timings(index=0, doc_id=did, unit="absolute_ms",
                                     timings=[1000, 1300, 1600, 1900])
    row = res["lines"][0]
    assert res["unit"] == "absolute_ms"
    assert row["durations_cs"] == [30, 30, 30]
    assert row["text"] == r"{\k30}ka{\k30}ra{\k30}o"

    with pytest.raises(ToolError, match="monotonically increasing"):
        KT.ass_karaoke_set_timings(index=0, doc_id=did, unit="absolute_ms",
                                   timings=[1000, 900, 1600, 1900])


def test_set_timings_forces_kind(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    res = KT.ass_karaoke_set_timings(index=0, doc_id=did, timings=[50, 50, 50],
                                     kind="ko")
    # 50 ms is 5 cs, and the kind is forced onto every tag
    assert res["lines"][0]["text"] == r"{\ko5}ka{\ko5}ra{\ko5}o"
    assert res["lines"][0]["durations_cs"] == [5, 5, 5]
    assert res["lines"][0]["discrepancy_cs"] == -135


def test_set_timings_requires_timings(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    with pytest.raises(ToolError, match="timings= is required"):
        KT.ass_karaoke_set_timings(index=0, doc_id=did)
    with pytest.raises(ToolError, match="unit must be"):
        KT.ass_karaoke_set_timings(index=0, doc_id=did, timings=[1, 2, 3], unit="frames")


# --------------------------------------------------------------------------- retime


def test_retime_proportional_reports_before_and_after(tmp_path):
    did, _p, doc = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}ka{\k40}ra{\k40}o")])
    res = KT.ass_karaoke_retime(selection=[0], doc_id=did, new_start_ms=1000,
                                new_end_ms=1500)
    row = res["lines"][0]
    assert res["mode"] == "proportional"
    assert res["dry_run"] is False
    assert res["lines_changed"] == 1
    assert row["span_cs"] == 50
    assert row["old_start_ms"] == 1000 and row["old_end_ms"] == 2000
    assert row["start_ms"] == 1000 and row["end_ms"] == 1500
    assert row["times_changed"] is True
    # 20:40:40 over 50 cs keeps the 1:2:2 ratio -> 10:20:20
    assert [(s["before_cs"], s["after_cs"], s["delta_cs"]) for s in row["syllables"]] == [
        (20, 10, -10), (40, 20, -20), (40, 20, -20)]
    assert row["durations_cs"] == [10, 20, 20]
    assert row["duration_sum_cs"] == row["line_duration_cs"] == 50
    assert row["discrepancy_cs"] == 0
    assert row["text"] == r"{\k10}ka{\k20}ra{\k20}o"
    entry = doc.events()[0]
    assert (entry.start_ms, entry.end_ms) == (1000, 1500)


def test_retime_dry_run_writes_nothing(tmp_path):
    did, _p, _doc = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}ka{\k40}ra{\k40}o")])
    res = KT.ass_karaoke_retime(selection=[0], doc_id=did, new_end_ms=1500,
                                dry_run=True)
    assert res["dry_run"] is True
    assert res["lines_changed"] == 0
    assert res["lines"][0]["text"] == r"{\k10}ka{\k20}ra{\k20}o"  # the proposal
    assert events_text(did) == [r"{\k20}ka{\k40}ra{\k40}o"]      # the document
    entry = workspace.get(did).events()[0]
    assert (entry.start_ms, entry.end_ms) == (1000, 2000)


def test_retime_even_mode_and_shift_and_factor(tmp_path):
    did, _p, _doc = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}ka{\k40}ra{\k40}o")])
    even = KT.ass_karaoke_retime(selection=[0], doc_id=did, mode="even",
                                 new_start_ms=1000, new_end_ms=1900)
    assert even["lines"][0]["durations_cs"] == [30, 30, 30]

    shifted = KT.ass_karaoke_retime(selection=[0], doc_id=did, shift_ms=1000)
    row = shifted["lines"][0]
    assert (row["old_start_ms"], row["old_end_ms"]) == (1000, 1900)
    assert (row["start_ms"], row["end_ms"]) == (2000, 2900)
    assert row["times_changed"] is False       # only the \k durations moved
    assert sum(row["durations_cs"]) == 90

    scaled = KT.ass_karaoke_retime(selection=[0], doc_id=did, factor=2.0)
    assert scaled["lines"][0]["span_cs"] == 180

    with pytest.raises(ToolError, match="not both"):
        KT.ass_karaoke_retime(selection=[0], doc_id=did, shift_ms=100, factor=2.0)
    with pytest.raises(ToolError, match="mode must be"):
        KT.ass_karaoke_retime(selection=[0], doc_id=did, mode="bogus")


def test_retime_requires_existing_karaoke_tags(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    with pytest.raises(ToolError, match="no karaoke tags to retime"):
        KT.ass_karaoke_retime(selection=[0], doc_id=did)


# --------------------------------------------------------------------------- shift / scale


def test_shift_adds_centiseconds_and_reports_clamping(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}ka{\k30}ra")])
    res = KT.ass_karaoke_shift(selection=[0], shift_ms=100, doc_id=did)
    row = res["lines"][0]
    assert res["clamped_cs"] == 0
    assert [s["after_cs"] for s in row["syllables"]] == [30, 40]
    assert row["text"] == r"{\k30}ka{\k40}ra"
    assert row["duration_sum_cs"] == 70

    # the document now holds {\k30}ka{\k40}ra; -250 ms moves both back by 25 cs
    back = KT.ass_karaoke_shift(selection=[0], shift_ms=-250, doc_id=did)
    brow = back["lines"][0]
    assert [s["before_cs"] for s in brow["syllables"]] == [30, 40]
    assert [s["after_cs"] for s in brow["syllables"]] == [5, 15]
    assert back["clamped_cs"] == 0
    assert brow["text"] == r"{\k5}ka{\k15}ra"

    # -1000 ms cannot go below zero: both syllables clamp and say so
    clamped = KT.ass_karaoke_shift(selection=[0], shift_ms=-1000, doc_id=did)
    crow = clamped["lines"][0]
    assert [s["after_cs"] for s in crow["syllables"]] == [0, 0]
    assert clamped["clamped_cs"] == 2
    assert crow["text"] == r"{\k0}ka{\k0}ra"
    assert crow["duration_sum_cs"] == 0
    # the line itself is 1000 ms = 100 cs long, all of which is now untagged
    assert crow["line_duration_cs"] == 100
    assert crow["discrepancy_cs"] == -100      # reported, not stretched
    assert "reported, not stretched" in crow["message"]


def test_shift_times_moves_the_line_itself(tmp_path):
    did, _p, doc = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}ka{\k30}ra")])
    res = KT.ass_karaoke_shift(selection=[0], shift_ms=500, doc_id=did,
                               shift_times=True)
    assert res["shift_times"] is True
    entry = doc.events()[0]
    assert (entry.start_ms, entry.end_ms) == (1500, 2500)
    assert res["lines"][0]["start_ms"] == 1500


def test_scale_multiplies_durations_with_a_floor(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}a{\k30}b")])
    res = KT.ass_karaoke_scale(selection=[0], factor=2.0, doc_id=did)
    assert res["factor"] == 2.0
    assert [s["after_cs"] for s in res["lines"][0]["syllables"]] == [40, 60]
    assert res["lines"][0]["text"] == r"{\k40}a{\k60}b"

    floored = KT.ass_karaoke_scale(selection=[0], factor=0.01, doc_id=did, min_cs=5)
    assert [s["after_cs"] for s in floored["lines"][0]["syllables"]] == [5, 5]
    assert floored["min_cs"] == 5

    with pytest.raises(ToolError, match="factor must be a number"):
        KT.ass_karaoke_scale(selection=[0], factor="big", doc_id=did)


def test_scale_times_scales_the_line_span(tmp_path):
    did, _p, doc = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}a{\k30}b")])
    KT.ass_karaoke_scale(selection=[0], factor=1.5, doc_id=did, scale_times=True)
    entry = doc.events()[0]
    assert (entry.start_ms, entry.end_ms) == (1000, 2500)


# --------------------------------------------------------------------------- kind


def test_set_kind_converts_every_karaoke_tag(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:03.00", r"{\k10}a{\kf20}b{\ko30}c")])
    res = KT.ass_karaoke_set_kind(selection=[0], kind="kt", doc_id=did)
    assert res["kind"] == "kt"
    assert res["tags_converted"] == 3
    assert res["lines_changed"] == 1
    assert res["lines"][0]["text"] == r"{\kt10}a{\kt20}b{\kt30}c"
    assert res["lines"][0]["before_kinds"] == ["k", "kf", "ko"]
    assert res["lines"][0]["after_kinds"] == ["kt", "kt", "kt"]
    assert res["lines"][0]["changed"] is True
    assert events_text(did) == [r"{\kt10}a{\kt20}b{\kt30}c"]


def test_set_kind_only_matching_leaves_other_kinds_alone(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:03.00", r"{\k10}a{\kf20}b{\ko30}c")])
    res = KT.ass_karaoke_set_kind(selection=[0], kind="kt", doc_id=did,
                                  only_matching="k")
    assert res["only_matching"] == "k"
    assert res["tags_converted"] == 1
    assert res["lines"][0]["text"] == r"{\kt10}a{\kf20}b{\ko30}c"


def test_set_kind_accepts_the_capital_k_alias(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}ka")])
    res = KT.ass_karaoke_set_kind(selection=[0], kind="K", doc_id=did)
    assert res["kind"] == "kf"
    assert events_text(did) == [r"{\kf20}ka"]


def test_set_kind_rejects_a_bad_kind(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.00", r"{\k20}ka")])
    with pytest.raises(ToolError, match="kind must be one of"):
        KT.ass_karaoke_set_kind(selection=[0], kind="kx", doc_id=did)
    with pytest.raises(ToolError, match="kind must be one of"):
        KT.ass_karaoke_set_kind(selection=[0], kind="kt", doc_id=did, only_matching="zz")


# --------------------------------------------------------------------------- remove


def test_remove_keeps_markers_and_can_keep_times(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.50", r"{\k30}ka|{\k30}ra|{\k30}o")])
    res = KT.ass_karaoke_remove(selection=[0], doc_id=did, keep_times=True)
    row = res["lines"][0]
    assert row["text"] == "ka|ra|o"          # markers survive
    assert row["removed_tags"] == 3
    assert row["removed_markers"] == 0
    assert row["times_retightened"] is False
    assert (row["start_ms"], row["end_ms"]) == (1000, 2500)
    assert res["tags_removed"] == 3


def test_remove_can_drop_markers_and_retighten_the_line(tmp_path):
    did, _p, doc = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.50", r"{\k30}ka|{\k30}ra|{\k30}o")])
    res = KT.ass_karaoke_remove(selection=[0], doc_id=did, drop_markers=True)
    row = res["lines"][0]
    assert row["text"] == "karao"
    assert row["removed_tags"] == 3
    assert row["removed_markers"] == 2
    assert res["tags_removed"] == 5
    # 90 cs of karaoke inside a 150 cs line -> End tightens to 1000 + 900 ms
    assert row["times_retightened"] is True
    assert (row["old_end_ms"], row["end_ms"]) == (2500, 1900)
    entry = doc.events()[0]
    assert (entry.start_ms, entry.end_ms) == (1000, 1900)


def test_remove_is_a_no_op_on_an_untimed_line(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "hello")])
    res = KT.ass_karaoke_remove(selection=[0], doc_id=did)
    assert res["lines_changed"] == 0
    assert res["lines"][0]["text"] == "hello"
    assert res["lines"][0]["removed_tags"] == 0


# --------------------------------------------------------------------------- auto timings


def test_auto_timings_proposes_without_writing(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    res = KT.ass_karaoke_auto_timings(index=0, doc_id=did)
    assert res["applied"] is False
    assert res["mode"] == "syllable"
    assert res["weights"] == "char_class"
    # char_class weights 1.1 / 1.1 / 0.55 over 150 cs -> 60 / 60 / 30
    assert res["durations_cs"] == [60, 60, 30]
    assert res["duration_sum_cs"] == res["line_duration_cs"] == 150
    assert res["discrepancy_cs"] == 0
    assert res["proposed_text"] == r"{\k60}ka{\k60}ra{\k30}o"
    assert res["old_text"] == "ka|ra|o"
    assert [s["weight"] for s in res["syllables"]] == [1.1, 1.1, 0.55]
    assert events_text(did) == ["ka|ra|o"]        # nothing was written


def test_auto_timings_apply_writes_the_proposal(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    res = KT.ass_karaoke_auto_timings(index=0, doc_id=did, apply=True)
    assert res["applied"] is True
    assert events_text(did) == [res["proposed_text"]]
    assert events_text(did) == [r"{\k60}ka{\k60}ra{\k30}o"]


def test_auto_timings_even_weights_and_char_mode(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    even = KT.ass_karaoke_auto_timings(index=0, doc_id=did, weights="even")
    assert even["durations_cs"] == [50, 50, 50]

    did2, _p2, _d2 = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "kara")],
                              name="chars.ass")
    chars = KT.ass_karaoke_auto_timings(index=0, doc_id=did2, mode="char")
    assert len(chars["syllables"]) == 4
    assert chars["duration_sum_cs"] == 150
    assert chars["discrepancy_cs"] == 0
    assert sum(chars["durations_cs"]) == 150
    # every character weighs 0.55 -> 1500 / 4 = 375 ms = 37.5 cs each; the rounding
    # remainder (-2 cs) is taken off the first syllable so the sum stays exact
    assert chars["durations_cs"] == [36, 38, 38, 38]
    assert chars["proposed_text"] == r"{\k36}k{\k38}a{\k38}r{\k38}a"


# --------------------------------------------------------------------------- styles


def test_styles_create_the_conventional_trio(tmp_path):
    did, _p, doc = open_ass(tmp_path, [("0:00:01.00", "0:00:02.00", "ka|ra")])
    listing = KT.ass_karaoke_styles(doc_id=did, create=False)
    assert listing["created"] == []
    assert [s["exists"] for s in listing["styles"]] == [False, False, False]
    assert listing["standard_names"] == ["Karaoke", "Karaoke_2", "Karaoke_3"]
    assert doc.get_style("Karaoke") is None

    res = KT.ass_karaoke_styles(doc_id=did, create=True)
    assert res["created"] == ["Karaoke", "Karaoke_2", "Karaoke_3"]
    assert res["existing"] == []
    assert all(s["created_now"] for s in res["styles"])

    base_style = doc.get_style("Karaoke")
    assert base_style is not None
    assert base_style.get("Fontname") == "Arial"
    assert int(base_style.get("Fontsize")) == 60
    assert int(base_style.get("Alignment")) == 2      # bottom
    assert int(doc.get_style("Karaoke_2").get("Alignment")) == 8   # top
    assert int(doc.get_style("Karaoke_3").get("Alignment")) == 5   # middle
    assert base_style.get("SecondaryColour") == "&H000000FF&"      # kf sweeps from blue
    assert int(base_style.get("Bold")) == -1
    assert res["style_names"][:3] == ["Default", "Karaoke", "Karaoke_2"] or set(
        ["Karaoke", "Karaoke_2", "Karaoke_3"]).issubset(set(res["style_names"]))

    again = KT.ass_karaoke_styles(doc_id=did, create=True)
    assert again["created"] == []
    assert again["existing"] == ["Karaoke", "Karaoke_2", "Karaoke_3"]


def test_styles_support_a_custom_prefix(tmp_path):
    did, _p, doc = open_ass(tmp_path, [("0:00:01.00", "0:00:02.00", "ka|ra")])
    res = KT.ass_karaoke_styles(doc_id=did, create=True, prefix="Op")
    assert res["created"] == ["Op", "Op_2", "Op_3"]
    assert doc.get_style("Op_3") is not None


# --------------------------------------------------------------------------- template


def test_template_expands_a_two_syllable_line(tmp_path):
    template = r"{\k$sdur\t(0,400,\fscx130)\b1}$syl"
    did, _p, doc = open_ass(tmp_path, [
        ("0:00:00.00", "0:00:00.10", template),
        ("0:00:01.00", "0:00:01.40", "ka|ra"),
    ], layer=1, style="Karaoke", actor="Bob", effect="tmpl")

    res = KT.ass_karaoke_template(selection=[1], doc_id=did, template_index=0)
    assert res["template_subset"] is True
    assert res["subset_of_aegisub_karaoke_templater"] is True
    assert res["variables_used"] == ["sdur", "syl"]
    assert res["unknown_variables"] == []
    assert res["dry_run"] is False
    assert res["inserted_count"] == 2

    # documented subset: the unsupported list names classes, code lines and the library
    unsupported = " | ".join(res["unsupported"])
    assert "class blocks" in unsupported
    assert "`code` lines" in unsupported
    assert "effect library" in unsupported
    assert len(res["supported"]) >= 3

    assert [r["syllable"] for r in res["generated"]] == ["ka", "ra"]
    # 400 ms line, char_class weights 1.1 : 1.1 -> 20 cs each
    assert [r["duration_cs"] for r in res["generated"]] == [20, 20]
    assert [(r["start_ms"], r["end_ms"]) for r in res["generated"]] == [
        (1000, 1200), (1200, 1400)]
    assert [r["start"] for r in res["generated"]] == ["0:00:01.00", "0:00:01.20"]
    assert res["generated"][0]["text"] == r"{\k20\t(0,400,\fscx130)\b1}ka"
    assert res["generated"][1]["text"] == r"{\k20\t(0,400,\fscx130)\b1}ra"
    # layer/actor/style copied from the template line
    assert res["generated"][0]["layer"] == "1"
    assert res["generated"][0]["actor"] == "Bob"
    assert res["generated"][0]["style"] == "Karaoke"
    assert res["generated"][0]["kind"] == "Dialogue"

    # generated lines sit directly after the template line
    assert res["generated_indices"] == [1, 2]
    assert [r["inserted_index"] for r in res["generated"]] == [1, 2]
    texts = events_text(did)
    assert texts[0] == template
    assert texts[1] == res["generated"][0]["text"]
    assert texts[2] == res["generated"][1]["text"]
    assert len(texts) == 4
    assert texts[3] == "ka|ra"     # the target line itself is untouched
    entry = doc.events()[1]
    assert (entry.start_ms, entry.end_ms) == (1000, 1200)
    assert entry.get("Effect") == "tmpl"
    json.dumps(res)


def test_template_uses_existing_karaoke_durations(tmp_path):
    template = r"{\b1$syl}"
    did, _p, _doc = open_ass(tmp_path, [
        ("0:00:00.00", "0:00:00.10", template),
        ("0:00:01.00", "0:00:01.40", r"{\k30}ka{\k10}ra"),
    ], effect="tmpl")
    res = KT.ass_karaoke_template(selection=[1], doc_id=did, template_index=0)
    assert [r["duration_cs"] for r in res["generated"]] == [30, 10]
    assert [r["timings_source"] for r in res["generated"]] == [
        "karaoke_tags", "karaoke_tags"]
    assert [(r["start_ms"], r["end_ms"]) for r in res["generated"]] == [
        (1000, 1300), (1300, 1400)]
    assert [r["text"] for r in res["generated"]] == [r"{\b1ka}", r"{\b1ra}"]


def test_template_dry_run_inserts_nothing(tmp_path):
    template = r"{\b1}$syl"
    did, _p, _doc = open_ass(tmp_path, [
        ("0:00:00.00", "0:00:00.10", template),
        ("0:00:01.00", "0:00:01.40", "ka|ra"),
    ], effect="tmpl")
    res = KT.ass_karaoke_template(selection=[1], doc_id=did, template_index=0,
                                  dry_run=True)
    assert res["dry_run"] is True
    assert res["generated_count"] == 2      # the proposal
    assert res["inserted_count"] == 0       # nothing was actually inserted
    assert res["generated_indices"] == []
    assert len(events_text(did)) == 2


def test_template_replace_existing_removes_previous_output(tmp_path):
    template = r"{\b1}$syl"
    did, _p, _doc = open_ass(tmp_path, [
        ("0:00:00.00", "0:00:00.10", template),
        ("0:00:01.00", "0:00:01.40", "ka|ra"),
    ], effect="tmpl")
    KT.ass_karaoke_template(selection=[1], doc_id=did, template_index=0)
    assert len(events_text(did)) == 4       # template + 2 generated + target
    # only the generated lines carry the template's Effect marker; clear it on the
    # lyric line so the replacement pass has exactly the two generated lines to drop
    workspace.get(did).events()[3].set("Effect", "")
    # the lyric line moved to index 3 when the two generated lines were inserted
    idx = events_text(did).index("ka|ra")
    res = KT.ass_karaoke_template(selection=[idx], doc_id=did, template_index=0,
                                  replace_existing=True)
    assert res["replaced_lines"] == 2
    assert len(events_text(did)) == 4


def test_template_accepts_a_raw_template_string(tmp_path):
    did, _p, _doc = open_ass(tmp_path, [("0:00:01.00", "0:00:01.40", "ka|ra")])
    res = KT.ass_karaoke_template(selection=[0], doc_id=did,
                                  template_line=r"{\b1}$syl")
    assert res["inserted_count"] == 2
    assert [r["text"] for r in res["generated"]] == [r"{\b1}ka", r"{\b1}ra"]
    # with a raw template the generated lines follow the selected line
    assert len(events_text(did)) == 3


def test_template_requires_a_placeholder(tmp_path):
    did, _p, _doc = open_ass(tmp_path, [
        ("0:00:00.00", "0:00:00.10", r"{\b1}plain"),
        ("0:00:01.00", "0:00:01.40", "ka|ra"),
    ])
    with pytest.raises(ToolError, match="no \\$syl or \\$sdur placeholder"):
        KT.ass_karaoke_template(selection=[1], doc_id=did, template_index=0)


def test_template_reports_unknown_variables(tmp_path):
    did, _p, _doc = open_ass(tmp_path, [
        ("0:00:00.00", "0:00:00.10", r"{\b1$syl}$start"),
        ("0:00:01.00", "0:00:01.40", "ka|ra"),
    ])
    res = KT.ass_karaoke_template(selection=[1], doc_id=did, template_index=0)
    assert res["unknown_variables"] == ["start"]
    assert res["variables_used"] == ["syl"]


# --------------------------------------------------------------------------- export


def test_export_srv2_writes_into_the_output_dir(tmp_path):
    did, path, _doc = open_ass(tmp_path, [
        ("0:00:01.00", "0:00:02.50", r"{\k30}ka{\k20}ra{\k10}o"),
        ("0:00:03.00", "0:00:04.00", "plain line"),
    ])
    res = KT.ass_karaoke_export(doc_id=did, format="srv2")
    assert res["format"] == "srv2"
    assert res["filename"] == f"{path.stem}_karaoke.srv2"
    assert Path(res["path"]).parent == workspace.output_dir
    assert res["lines"] == 1          # only the timed line
    assert res["syllables"] == 3
    assert res["bytes"] == len(res["content"].encode("utf-8"))
    assert Path(res["path"]).read_text(encoding="utf-8") == res["content"]
    rows = res["content"].splitlines()
    assert rows[0].startswith("# srv2")
    assert rows[2] == "0\t0\t1000\t1300\t30\t300\tk\tka"
    assert rows[3] == "0\t1\t1300\t1500\t20\t200\tk\tra"
    assert rows[4] == "0\t2\t1500\t1600\t10\t100\tk\to"
    assert Path(res["path"]).name == res["filename"]

    untimed = KT.ass_karaoke_export(doc_id=did, include_untimed=True)
    assert untimed["lines"] == 2
    assert untimed["syllables"] == 4


def test_export_txt_and_csv_formats(tmp_path):
    did, _p, _d = open_ass(
        tmp_path, [("0:00:01.00", "0:00:02.50", r"{\k30}ka{\k20}ra{\k10}o")])
    txt = KT.ass_karaoke_export(doc_id=did, format="txt")
    assert txt["filename"].endswith("_karaoke.txt")
    txt_rows = txt["content"].splitlines()
    assert txt_rows[0].startswith("# aegisub-mcp karaoke export")
    assert txt_rows[2] == "0\t0:00:01.00\t0:00:01.30\tka"

    csv = KT.ass_karaoke_export(doc_id=did, format="csv")
    assert csv["filename"].endswith("_karaoke.csv")
    csv_rows = csv["content"].splitlines()
    assert csv_rows[0] == "line,syl,start_ms,end_ms,dur_cs,dur_ms,kind,text"
    assert csv_rows[1] == "0,0,1000,1300,30,300,k,ka"
    assert len(csv_rows) == 4

    sub = KT.ass_karaoke_export(doc_id=did, format="srv2", selection=[0])
    assert sub["lines"] == 1 and sub["syllables"] == 3

    with pytest.raises(ToolError, match="format must be srv2, txt or csv"):
        KT.ass_karaoke_export(doc_id=did, format="json")


# --------------------------------------------------------------------------- contract


def test_byte_exact_round_trip_of_a_real_fixture(tmp_path):
    """Changing nothing must reproduce the fixture byte for byte (BOM, CRLF,
    missing final newline and all)."""
    source = REAL_FIXTURES / "basic.ass"
    original = source.read_bytes()
    assert original.startswith(b"\xef\xbb\xbf")      # the fixture really has a BOM

    did = workspace.open(source)
    target = tmp_path / "roundtrip.ass"
    saved = workspace.save(did, path=str(target))
    assert saved["doc_id"] == did
    assert target.read_bytes() == original
    assert saved["bytes"] == len(original)

    # opening and saving twice does not drift either
    workspace.save(did, path=str(target))
    assert target.read_bytes() == original


def test_readonly_karaoke_tools_on_a_real_fixture_leave_it_byte_exact(tmp_path):
    """Read-only karaoke tools over a real fixture must not touch a single byte."""
    source = REAL_FIXTURES / "basic.ass"
    original = source.read_bytes()
    did = workspace.open(source)

    got = KT.ass_karaoke_get(index=0, doc_id=did)
    assert got["text_source"] == "line"
    assert got["untimed"] is True                 # the fixture has no karaoke
    assert got["line"]["plain_text"] == "Hello world"   # tags stripped from the text

    split = KT.ass_karaoke_split(index=0, doc_id=did)
    assert split["text_source"] == "line"
    assert split["reconstruction"] == {"ok": True, "reconstructed": "Hello world",
                                       "expected": "Hello world"}

    exported = KT.ass_karaoke_export(doc_id=did, include_untimed=True)
    assert exported["lines"] == 2                 # both real dialogue lines
    assert Path(exported["path"]).exists()

    target = tmp_path / "readonly.ass"
    workspace.save(did, path=str(target))
    assert target.read_bytes() == original


def test_every_mutation_snapshots_and_undo_restores(tmp_path):
    did, _p, _doc = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    assert workspace.undo_depth(did) == (0, 0)
    KT.ass_karaoke_generate(index=0, doc_id=did)
    assert events_text(did) == [r"{\k60}ka{\k60}ra{\k30}o"]
    depth, redo = workspace.undo_depth(did)
    assert depth == 1 and redo == 0        # exactly one snapshot was taken

    assert workspace.undo(did) is True
    assert events_text(did) == ["ka|ra|o"]

    workspace.redo(did)
    assert events_text(did) == [r"{\k60}ka{\k60}ra{\k30}o"]


def test_selection_from_the_session_is_used_when_no_index_is_given(tmp_path):
    did, _p, _d = open_ass(tmp_path, [
        ("0:00:01.00", "0:00:02.00", "a|b"),
        ("0:00:01.00", "0:00:02.00", "c|d"),
    ])
    workspace.selection = [1]
    res = KT.ass_karaoke_generate(doc_id=did, weights="even")
    assert res["lines_changed"] == 1
    assert res["lines"][0]["index"] == 1
    assert events_text(did) == ["a|b", r"{\k50}c{\k50}d"]


def test_selection_can_span_several_lines(tmp_path):
    did, _p, _d = open_ass(tmp_path, [
        ("0:00:01.00", "0:00:02.00", "a|b"),
        ("0:00:03.00", "0:00:04.00", "c|d"),
    ])
    res = KT.ass_karaoke_generate(selection=[0, 1], doc_id=did, weights="even")
    assert [row["index"] for row in res["lines"]] == [0, 1]
    assert all(row["duration_sum_cs"] == 100 for row in res["lines"])
    assert events_text(did) == [r"{\k50}a{\k50}b", r"{\k50}c{\k50}d"]


def test_register_exposes_every_documented_tool():
    registered: list[str] = []

    class FakeMcp:
        def tool(self):
            def deco(fn):
                registered.append(fn.__name__)
                return fn
            return deco

    names = KT.register(FakeMcp(), workspace)
    expected = {
        "ass_karaoke_split", "ass_karaoke_get", "ass_karaoke_generate",
        "ass_karaoke_set_timings", "ass_karaoke_retime", "ass_karaoke_shift",
        "ass_karaoke_scale", "ass_karaoke_set_kind", "ass_karaoke_remove",
        "ass_karaoke_auto_timings", "ass_karaoke_styles", "ass_karaoke_template",
        "ass_karaoke_export",
    }
    assert set(names) == expected
    assert names == sorted(names)
    assert set(registered) == expected
    for name in names:
        assert callable(getattr(KT, name))
        assert getattr(KT, name).__doc__


def test_results_are_plain_json_serialisable_dicts(tmp_path):
    did, _p, _d = open_ass(tmp_path, [("0:00:01.00", "0:00:02.50", "ka|ra|o")])
    results = [
        KT.ass_karaoke_split(index=0, doc_id=did),
        KT.ass_karaoke_get(index=0, doc_id=did),
        KT.ass_karaoke_generate(index=0, doc_id=did),
        KT.ass_karaoke_auto_timings(index=0, doc_id=did),
        KT.ass_karaoke_export(doc_id=did),
    ]
    for res in results:
        assert isinstance(res, dict)
        round_tripped = json.loads(json.dumps(res))
        assert round_tripped == res
