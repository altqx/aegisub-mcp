"""Override-tag engine tests: parse/serialize fidelity, editing, TS wrapping."""

from __future__ import annotations

import pytest

from aegisub_mcp.asscore import assutil as U
from aegisub_mcp.asscore import tags as T

ROUND_TRIP = [
    "",
    "plain text",
    r"{\pos(10,20)}hello",
    r"{\an7\pos(0,0)\fscx120\fscy120}k",
    r"{\1c&H00FF00&\3c&H202020&\bord2}tagged",
    r"{\t(0,500,\fscx120\fscy120)}grow",
    r"{\clip(1,m 0 0 l 100 0 100 100)}clipped",
    r"{\clip(10,20,30,40)}rect",
    r"{\iclip(10,20,30,40)}irect",
    r"{\clip(m 0 0 b 10 10 20 20 30 0 c)}clipdraw",
    r"{\p1}m 0 0 l 100 0 100 100 c",
    r"{\p2}m 0 0 b 10 10 20 20 30 0{\p0}back to text",
    r"{\kt10}ka{\kf20}ra{\ko30}o{\K40}ke",
    r"{\alpha&H80&\1a&HFF&}fade",
    r"{\fnMy Font\fs40\b1\i1\u1\s1\bord3\shad1\blur1.5\be1}styled",
    r"{\fad(500,500)\move(0,0,100,100,0,1000)\org(50,50)}moved",
    r"{\q2\pbo10\r}reset\rsome style\a6",
    r"a{\b1}b{\b0}c",
    r"{\fsp2\frz45\frx10\fry10\fax0.2\fay0.1}p",
    r"{\3c&HFF0000&}x{\c&H00FF00&}y",
    r"{\clip(5,1,m 0 0 l 10 0 10 10)}scaled-clip",
    r"{\fscx100\t(100,200,1.5,\clip(1,m 0 0 l 5 0 5 5))\fscy100}nested",
    r"line one\Nline two",
    r"{\an8}top{\N}bottom",
]


@pytest.mark.parametrize("text", ROUND_TRIP, ids=range(len(ROUND_TRIP)))
def test_parse_serialize_round_trip(text: str) -> None:
    assert T.serialize(T.parse(text)) == text


@pytest.mark.parametrize("text", ROUND_TRIP, ids=range(len(ROUND_TRIP)))
def test_plain_text_has_no_tags(text: str) -> None:
    plain = T.plain_text(text)
    assert "{" not in plain and "}" not in plain


def test_parse_tags_names_and_args() -> None:
    text = r"{\an7\pos(10,20)\fscx120\b1\clip(1,m 0 0 l 5 0 5 5)}x"
    assert T.tag_names(text) == ["an", "pos", "fscx", "b", "clip"]
    assert T.get_tag(text, "pos").arg == "10,20"
    assert T.get_tag(text, "b").arg == "1"
    assert T.get_tag(text, "clip").arg.startswith("1,m 0 0")
    assert T.get_tag(text, "nope") is None


def test_nested_parens_are_handled() -> None:
    text = r"{\t(0,500,\clip(1,m 0 0 l 5 0 5 5))}nested"
    tag = T.get_tag(text, "t")
    assert tag is not None
    parts = T.parse_transform(tag.arg)
    assert parts["t1"] == 0 and parts["t2"] == 500
    assert parts["tag_list"][0].name == "clip"
    assert parts["tag_list"][0].arg == "1,m 0 0 l 5 0 5 5"


def test_transform_parts_defaults() -> None:
    parts = T.parse_transform(r"\fscx120")
    assert parts["t1"] == 0 and parts["t2"] == 0 and parts["accel"] == 1.0
    assert parts["tag_list"][0].name == "fscx"


def test_strip_tags_keeps_visible_text() -> None:
    text = r"{\an8\pos(0,0)\fad(200,200)}Hello {\i1}world{\i0}!"
    assert T.strip_tags(text) == "Hello world!"
    assert T.strip_tags(text, keep=["pos", "an"]) == r"{\an8\pos(0,0)}Hello world!"
    assert T.strip_tags(text, keep_groups=["fade"]) == r"{\fad(200,200)}Hello world!"


def test_strip_tags_preserves_drawing_when_asked() -> None:
    text = r"{\p1}m 0 0 l 10 0 10 10 c"
    assert "m 0 0" in T.strip_tags(text, keep_drawing=True)
    assert "m 0 0" not in T.strip_tags(text)


def test_remove_and_set_tags() -> None:
    text = r"{\an7\pos(0,0)\fscx120}x"
    assert T.remove_tags(text, ["pos"]) == r"{\an7\fscx120}x"
    assert T.set_tag(text, "pos", "50,60") == r"{\an7\pos(50,60)\fscx120}x"
    assert T.set_tag("no tags", "an", "5") == r"{\an5}no tags"


def test_prepend_and_append_tags() -> None:
    assert T.prepend_tags("word", r"\an5") == r"{\an5}word"
    assert T.prepend_tags(r"{\pos(1,2)}word", r"\an5") == r"{\an5\pos(1,2)}word"
    assert T.append_tags("word", r"\fscx120") == r"word{\fscx120}"


def test_char_index_mapping() -> None:
    text = r"{\an7}ab{\i1}cd"
    chars = T.char_positions(text)
    assert [c for _, c in chars] == list("abcd")
    assert T.plain_len(text) == 4
    raw_at = T.plain_to_raw_index(text, 2)
    assert raw_at is not None and text[raw_at] == "c"
    assert T.raw_to_plain_index(text, raw_at) == 2
    assert T.raw_to_plain_index(text, 1) is None, "inside a tag block"


def test_iter_visible_chars_reports_state() -> None:
    text = r"{\an7}ab{\i1}c"
    chars = list(T.iter_visible_chars(text))
    assert [c for _, c, _ in chars] == list("abc")
    assert "i" in chars[-1][2] and "an" in chars[-1][2]


def test_wrap_range_keeps_surrounding_tags() -> None:
    text = r"{\fscx100}a{\fscx120}bcd"
    out, warnings = T.wrap_range(text, 1, 3, r"\fscx200")
    assert r"\fscx200" in out, "new tag must be present"
    assert T.plain_text(out) == "abcd", "visible text preserved"
    assert r"\fscx120" in out, "previous value restored after the range"
    assert isinstance(warnings, list)


def test_split_drawing_sections() -> None:
    text = r"{\p1}m 0 0 l 100 0 100 100{\p0}after"
    prefix, drawing, suffix = T.drawing_parts(text)
    assert r"\p1" in prefix
    assert drawing.startswith("m 0 0")
    assert "after" in suffix


def test_drawing_detection() -> None:
    assert T.is_drawing(r"{\p1}m 0 0 l 1 1") is True
    assert T.is_drawing(r"{\p0}plain") is False
    assert T.drawing_state(r"{\p2}m 0 0 l 1 1") == 2
    assert T.drawing_state(r"{\p1}m 0 0{\p0}text") == 0


def test_normalize_dedupes_redundant_tags() -> None:
    assert T.normalize(r"{\b1}{\b1\b1}word") == r"{\b1}word"


def test_tag_summary() -> None:
    summary = T.tag_summary(r"{\an7\pos(1,2)\k20\t(0,100,\fscx120)}x")
    assert summary["tags"] == 4
    assert summary["tag_names"]["an"] == 1
    assert "pos" in summary["tag_names"]
    assert summary["karaoke"] == ["k"]
    assert summary["transforms"] == ["0,100,\\fscx120"]
    assert summary["visible_chars"] == 1


def test_simple_tag_helpers() -> None:
    assert T.pos_tag(10, 20) == r"\pos(10,20)"
    assert T.an_tag(7) == r"\an7"
    assert T.fade_tag(200, 300) == r"\fad(200,300)"
    assert T.transform_tag(0, 500, r"\fscx120") == r"\t(0,500,\fscx120)"
    assert T.move_tag(0, 0, 10, 10, 100, 500) == r"\move(0,0,10,10,100,500)"
    assert T.color_tag("&H00FF00&", which=1) == r"\1c&H00FF00&"


def test_time_parse_and_format() -> None:
    for ms in (0, 10, 1000, 3661230, 5999990):
        assert abs(U.parse_time(U.format_time(ms)) - ms) <= 10
    assert U.parse_time("0:00:01.50") == 1500
    assert U.parse_time("1:02:03.45") == 3723450
    assert U.parse_time("-0:00:00.10") == -100
    assert U.try_parse_time("garbage") is None
    with pytest.raises(U.TimeParseError):
        U.parse_time("garbage")
    assert U.format_time(3723450) == "1:02:03.45"


def test_colour_conversion() -> None:
    assert U.parse_ass_color("&H00FF00&") == (0, 255, 0, 0)
    assert U.parse_ass_color("&HFF0000") == (0, 0, 255, 0)
    assert U.format_ass_color(0, 255, 0) == "&H00FF00&"
    assert U.parse_ass_color(U.format_ass_color(1, 2, 3, 4)) == (1, 2, 3, 4)
