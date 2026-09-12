"""Karaoke engine tests."""

from __future__ import annotations

import pytest

from aegisub_mcp.asscore import karaoke as K
from aegisub_mcp.asscore import tags as T


def test_split_by_marker() -> None:
    pieces = K.split_syllables("ka|ra|o|ke", mode="marker")
    assert pieces == ["ka", "ra", "o", "ke"]


def test_split_by_marker_keeps_tags_with_next_syllable() -> None:
    pieces = K.split_syllables(r"{\i1}ka|ra|{\b1}o", mode="marker")
    assert pieces[0] == r"{\i1}ka"
    assert pieces[-1] == r"{\b1}o"


def test_split_by_char_keeps_combining_marks() -> None:
    decomposed = "na\u0301korn"  # á written as a + combining acute
    pieces = K.split_syllables(decomposed, mode="char")
    # the base character and its combining mark form one syllable …
    assert pieces[1] == "a\u0301"
    # … matching how the precomposed spelling splits
    assert K.split_syllables("nákorn", mode="char")[1] == "á"
    assert "".join(T.plain_text(p) for p in pieces) == decomposed


def test_split_by_char_thai_marks_stay_with_base() -> None:
    pieces = K.split_syllables("กาน", mode="char")
    assert pieces == ["กา", "น"]


def test_split_by_word_keeps_space() -> None:
    pieces = K.split_syllables("hello world", mode="word")
    assert pieces == ["hello ", "world"]


def test_split_by_regex() -> None:
    pieces = K.split_syllables("ka-ra-o", mode="regex", pattern=r"\-")
    assert pieces == ["ka", "ra", "o"]


def test_parse_karaoke_reads_durations() -> None:
    line = r"{\k20}ka{\k30}ra{\kf50}o"
    syls = K.parse_karaoke(line)
    assert [s.text for s in syls] == ["ka", "ra", "o"]
    assert [s.duration_ms for s in syls] == [200, 300, 500]
    assert [s.kind for s in syls] == ["k", "k", "kf"]
    assert syls[-1].end_ms == 1000


def test_parse_karaoke_K_alias_is_kf() -> None:
    syls = K.parse_karaoke(r"{\K25}x")
    assert syls[0].kind == "kf"
    assert syls[0].duration_cs == 25


def test_karaoke_timings_absolute() -> None:
    rows = K.karaoke_timings(r"{\k20}a{\k20}b", line_start_ms=5000)
    assert rows[0]["start_ms"] == 5000 and rows[0]["end_ms"] == 5200
    assert rows[1]["start_ms"] == 5200 and rows[1]["end_ms"] == 5400


def test_generate_k_distributes_without_losing_text() -> None:
    out = K.generate_k("ka|ra|o", start_ms=0, end_ms=1000, mode="marker")
    assert T.plain_text(out) == "karao"
    syls = K.parse_karaoke(out)
    assert sum(s.duration_cs for s in syls) == 100
    assert len(syls) == 3


def test_generate_k_replaces_existing() -> None:
    out = K.generate_k(r"{\k10}ka|ra", start_ms=0, end_ms=400, mode="marker")
    assert out.count(r"\k") == 2
    assert r"\k10" not in out
    syls = K.parse_karaoke(out)
    assert len(syls) == 2
    assert sum(s.duration_cs for s in syls) == 40


def test_generate_k_explicit_durations() -> None:
    out = K.generate_k("ka|ra", durations_cs=[12, 34], mode="marker")
    assert r"\k12" in out and r"\k34" in out


def test_set_karaoke_kind() -> None:
    out = K.set_karaoke_kind(r"{\k20}a{\k30}b", "kf")
    assert r"\kf20" in out and r"\kf30" in out
    assert r"\k20" not in out


def test_shift_and_scale_karaoke() -> None:
    shifted = K.shift_karaoke(r"{\k20}a", -50)
    assert r"\k15" in shifted
    scaled = K.scale_karaoke(r"{\k20}a", 2.0)
    assert r"\k40" in scaled


def test_retime_line_proportional_keeps_ratio() -> None:
    line = r"{\k10}a{\k30}b"
    out = K.retime_line(line, 0, 2000, mode="proportional")
    syls = K.parse_karaoke(out)
    assert sum(s.duration_cs for s in syls) == 200
    assert syls[1].duration_cs > syls[0].duration_cs


def test_retime_line_even() -> None:
    line = r"{\k10}a{\k30}b"
    out = K.retime_line(line, 0, 1000, mode="even")
    syls = K.parse_karaoke(out)
    assert [s.duration_cs for s in syls] == [50, 50]


def test_remove_karaoke_keeps_other_tags() -> None:
    out = K.remove_karaoke(r"{\i1\k20}ka|ra")
    assert out == r"{\i1}ka|ra"
    dropped = K.remove_karaoke(r"{\k20}ka|ra", drop_markers=True)
    assert "|" not in dropped and r"\k20" not in dropped


def test_apply_timings_with_dicts() -> None:
    out = K.apply_timings(r"{\k10}ab", [{"start_ms": 0, "end_ms": 250}])
    assert r"\k25" in out


def test_distribute_by_length_sums_exactly() -> None:
    cs = K.distribute_by_length(["a", "bb", "ccc"], 1000)
    assert sum(cs) == 100


def test_auto_timings_char_mode() -> None:
    out = K.auto_timings("ab", 0, 1000, mode="char")
    assert len(K.parse_karaoke(out)) == 2
    assert K.karaoke_total_ms(out) == 1000


def test_karaoke_total_and_kinds() -> None:
    line = r"{\k20}a{\kf30}b"
    assert K.karaoke_total_ms(line) == 500
    assert K.karaoke_tag_kinds(line) == ["k", "kf"]
