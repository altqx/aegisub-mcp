"""Tests for the override-tag / typesetting tool layer (``tools.tags_tools``).

Every test calls the module-level ``ass_*`` functions directly, exactly the way
FastMCP would after ``register()``.  The workspace singleton in
``aegisub_mcp.tools.base`` is reused between tests, so each test starts by
clearing it (see :func:`reset_workspace` — ``base`` exposes no ``reset()``).

Index vocabulary used throughout
-------------------------------
* **plain index** — offset into the *visible* text: every character renders,
  override blocks and their braces do not count, but the characters of a
  drawing path *do* count (they are the visible text between ``\\p1`` and
  ``\\p0``).
* **raw index** — offset into the line text as stored in the ASS file,
  override blocks, braces and all.

Checks that must hold for every line a tool produces:

* ``tags.parse(line).render() == line`` (a byte-exact re-serialisation), and
* the braces balance, so no override block is ever left unclosed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from aegisub_mcp.asscore import measure as M
from aegisub_mcp.asscore import tags as T
from aegisub_mcp.tools import base as B
from aegisub_mcp.tools import tags_tools as TT
from aegisub_mcp.tools.base import ToolError

REAL = Path(__file__).resolve().parent / "fixtures" / "real"
BASIC_ASS = REAL / "basic.ass"
BASIC_SSA = REAL / "basic.ssa"

TOOL_NAMES = [
    "ass_add_typesetting",
    "ass_apply_tag_to_block",
    "ass_convert_tags",
    "ass_insert_tag_at",
    "ass_karaoke_tags_only",
    "ass_parse_text",
    "ass_plain_text",
    "ass_remove_tag",
    "ass_set_tag",
    "ass_strip_tags",
    "ass_swap_an_pos",
    "ass_tag_summary",
    "ass_wrap_range",
]

#: ``{\an7\pos(100,200)}Hello {\p1}<drawing>{\p0} world`` — two override blocks,
#: a drawing run and two plain runs; all offsets below are hand-counted.
LINE_TWO_BLOCKS = r"{\an7\pos(100,200)}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world"
#: ``{\an7\pos(100,200)}Hello`` — one block (raw [0, 19)) then 5 visible chars.
LINE_ONE_BLOCK = r"{\an7\pos(100,200)}Hello"
LINE_SIMPLE = r"{\an7\pos(100,200)}Hello world"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def reset_workspace() -> None:
    """Empty the ``base.workspace`` singleton in place.

    ``tools/base.py`` ships no ``reset()`` helper, so the internal containers
    are cleared directly (this is what the smoke probes did).
    """
    ws = B.workspace
    ws._docs.clear()
    ws._paths.clear()
    ws._order.clear()
    ws.current = None
    ws.selection = []
    ws._undo.clear()
    ws._redo.clear()


def make_doc(*lines: str, fixture: str = "basic.ass", **event_kwargs) -> tuple[str, Any]:
    """Open a real fixture and replace its events with ``lines``."""
    reset_workspace()
    doc_id = B.workspace.open(str(REAL / fixture))
    doc = B.workspace.get(doc_id)
    doc.remove_events(list(range(len(doc.events()))))
    fields: dict[str, Any] = {"MarginV": 0, "Style": "Default"}
    fields.update(event_kwargs)
    for text in lines:
        doc.add_event(
            text=text,
            Start="0:00:00.00",
            End="0:00:02.00",
            **fields,
        )
    return doc_id, doc


def assert_intact(line: str, balanced: bool = True) -> None:
    """A produced line must re-serialise byte-for-byte.

    ``balanced`` additionally requires the brace *counts* to match, which only
    holds for lines that did not already contain a stray ``}`` in visible text
    (the ASS format allows those, so they are simply preserved).
    """
    assert T.parse(line).render() == line, f"parser round-trip changed {line!r}"
    if balanced:
        assert line.count("{") == line.count("}"), f"unbalanced braces in {line!r}"


def assert_jsonable(result: object) -> None:
    """Every tool result has to survive ``json.dumps``."""
    json.dumps(result)


# --------------------------------------------------------------------------- #
# module contract
# --------------------------------------------------------------------------- #


def test_tools_are_module_level_ass_functions():
    for name in TOOL_NAMES:
        fn = getattr(TT, name)
        assert callable(fn), name
        assert fn.__module__ == TT.__name__, name
        assert fn.__doc__ and len(fn.__doc__) > 200, f"{name} has no real docstring"


def test_docstrings_separate_plain_and_raw_indices():
    """The index vocabulary has to be spelled out in every tool docstring."""
    for name in TOOL_NAMES:
        doc = (getattr(TT, name).__doc__ or "").lower()
        assert "plain" in doc, f"{name} does not mention plain indices"
        assert "raw" in doc, f"{name} does not mention raw indices"


def test_register_registers_every_tool_and_returns_sorted_names():
    class FakeMCP:
        def __init__(self):
            self.tools = []

        def tool(self):
            def deco(fn):
                self.tools.append(fn.__name__)
                return fn

            return deco

    mcp = FakeMCP()
    names = TT.register(mcp)
    assert names == sorted(TOOL_NAMES)
    assert sorted(mcp.tools) == sorted(TOOL_NAMES)


def test_every_result_is_json_serialisable():
    doc_id, _ = make_doc(LINE_TWO_BLOCKS)
    results = [
        TT.ass_parse_text(text=LINE_TWO_BLOCKS),
        TT.ass_parse_text(index=0, doc_id=doc_id),
        TT.ass_plain_text(text=LINE_TWO_BLOCKS),
        TT.ass_strip_tags(text=LINE_TWO_BLOCKS),
        TT.ass_set_tag(text=LINE_TWO_BLOCKS, name="bord", arg="2"),
        TT.ass_remove_tag(text=LINE_TWO_BLOCKS, names=["p"]),
        TT.ass_wrap_range(text=LINE_TWO_BLOCKS, start=0, end=3, override=r"\b1"),
        TT.ass_insert_tag_at(text=LINE_TWO_BLOCKS, plain_index=1, override=r"\b1"),
        TT.ass_apply_tag_to_block(text=LINE_TWO_BLOCKS, block="all", override=r"\b1"),
        TT.ass_add_typesetting([0], doc_id=doc_id, pos=(1, 2), an=7, in_place=False),
        TT.ass_swap_an_pos([0], doc_id=doc_id, dry_run=True),
        TT.ass_tag_summary([0], doc_id=doc_id),
        TT.ass_karaoke_tags_only(text=r"{\k20}ab"),
        TT.ass_convert_tags([0], doc_id=doc_id, dry_run=True),
    ]
    for result in results:
        assert_jsonable(result)


# --------------------------------------------------------------------------- #
# ass_parse_text
# --------------------------------------------------------------------------- #


def test_parse_text_hand_computed_two_blocks_and_a_drawing():
    result = TT.ass_parse_text(text=LINE_TWO_BLOCKS)
    assert result["source"] == "text"
    assert result["index"] is None and result["doc_id"] is None
    assert result["raw"] == LINE_TWO_BLOCKS
    assert result["raw_length"] == 59
    # the plain text keeps the drawing path characters: 6 + 18 + 6
    assert result["plain_text"] == "Hello m 0 0 l 50 0 50 50 world"
    assert result["plain_length"] == 30

    kinds = [(s["kind"], s["start"], s["end"]) for s in result["segments"]]
    assert kinds == [
        ("block", 0, 19),
        ("text", 19, 25),
        ("block", 25, 30),
        ("text", 30, 48),
        ("block", 48, 53),
        ("text", 53, 59),
    ]
    plain_spans = [(s["plain_start"], s["plain_end"]) for s in result["segments"]]
    assert plain_spans == [(0, 0), (0, 6), (6, 6), (6, 24), (24, 24), (24, 30)]

    assert [s.get("text") for s in result["segments"] if s["kind"] == "text"] == [
        "Hello ",
        "m 0 0 l 50 0 50 50",
        " world",
    ]
    blocks = [s for s in result["segments"] if s["kind"] == "block"]
    assert [b["raw"] for b in blocks] == [r"{\an7\pos(100,200)}", r"{\p1}", r"{\p0}"]
    assert [b["inner"] for b in blocks] == [r"\an7\pos(100,200)", r"\p1", r"\p0"]
    assert [b["block"] for b in blocks] == [0, 1, 2]
    assert all(b["is_override"] for b in blocks)

    assert result["tags"] == [
        {
            "name": "an",
            "argument": "7",
            "raw": r"\an7",
            "paren": False,
            "is_override": True,
            "block": 0,
        },
        {
            "name": "pos",
            "argument": "100,200",
            "raw": r"\pos(100,200)",
            "paren": True,
            "is_override": True,
            "block": 0,
        },
        {
            "name": "p",
            "argument": "1",
            "raw": r"\p1",
            "paren": False,
            "is_override": True,
            "block": 1,
        },
        {
            "name": "p",
            "argument": "0",
            "raw": r"\p0",
            "paren": False,
            "is_override": True,
            "block": 2,
        },
    ]
    assert result["summary"] == {
        "blocks": 3,
        "tags": 4,
        "tag_names": {"an": 1, "p": 2, "pos": 1},
        "tag_groups": {"drawing": 2, "layout": 2},
        # \p1 is closed again by \p0, so the line does not *end* in drawing mode:
        "has_drawing": False,
        "drawing_state": 0,
        "has_karaoke": False,
        "has_transform": False,
        "has_clip": False,
        "plain_length": 30,
    }


def test_parse_text_character_offsets_keep_plain_and_raw_apart():
    result = TT.ass_parse_text(text=LINE_TWO_BLOCKS)
    chars = result["characters"]
    assert len(chars) == 30 == result["plain_length"]
    assert [c["plain_index"] for c in chars] == list(range(30))
    assert "".join(c["char"] for c in chars) == result["plain_text"]
    assert chars[0] == {"plain_index": 0, "char": "H", "raw_index": 19}
    assert chars[5] == {"plain_index": 5, "char": " ", "raw_index": 24}
    # plain 6 is the first drawing character: raw 30, because the {\p1} block
    # (raw [25, 30)) sits between them.
    assert chars[6] == {"plain_index": 6, "char": "m", "raw_index": 30}
    assert chars[29] == {"plain_index": 29, "char": "d", "raw_index": 58}
    raw_offsets = [c["raw_index"] for c in chars]
    assert raw_offsets == sorted(raw_offsets)
    # 5 override-block characters are skipped between raw 19 and raw 30
    assert chars[6]["raw_index"] - chars[5]["raw_index"] == 6


def test_parse_text_summary_flags():
    cases = {
        "drawing": (r"{\p1}m 0 0 l 10 0 10 10", "has_drawing"),
        "karaoke": (r"{\k20}ab{\kf30}cd", "has_karaoke"),
        "transform": (r"{\t(0,500,\fscx120)}x", "has_transform"),
        "clip": (r"{\clip(0,0,10,10)}x", "has_clip"),
        "iclip": (r"{\iclip(m 0 0 l 10 0 10 10)}x", "has_clip"),
    }
    for label, (line, flag) in cases.items():
        summary = TT.ass_parse_text(text=line)["summary"]
        assert summary[flag] is True, f"{label} not detected in {line!r}"

    # an open \p1 keeps drawing mode on until the end of the line
    open_drawing = TT.ass_parse_text(text=r"{\p1}m 0 0 l 10 0 10 10")["summary"]
    assert open_drawing["has_drawing"] is True
    assert open_drawing["drawing_state"] == 1
    closed = TT.ass_parse_text(text=r"{\p1}m 0 0 l 10 0 10 10{\p0}")["summary"]
    assert closed["has_drawing"] is False and closed["drawing_state"] == 0

    empty = TT.ass_parse_text(text="plain")["summary"]
    assert (empty["blocks"], empty["tags"], empty["tag_groups"]) == (0, 0, {})
    assert empty["has_drawing"] is False and empty["has_karaoke"] is False


def test_parse_text_reports_the_line_source():
    doc_id, doc = make_doc(LINE_TWO_BLOCKS, "plain")
    result = TT.ass_parse_text(index=0, doc_id=doc_id)
    assert result["source"] == "line"
    assert result["index"] == 0
    assert result["doc_id"] == doc_id
    assert result["raw"] == doc.events()[0].text
    second = TT.ass_parse_text(index=1, doc_id=doc_id)
    assert second["raw"] == "plain" and second["plain_length"] == 5


def test_parse_text_source_errors():
    doc_id, _ = make_doc(LINE_TWO_BLOCKS)
    with pytest.raises(ToolError):
        TT.ass_parse_text()
    with pytest.raises(ToolError):
        TT.ass_parse_text(text=LINE_TWO_BLOCKS, index=0, doc_id=doc_id)
    with pytest.raises(ToolError):
        TT.ass_parse_text(index=99, doc_id=doc_id)
    # index= without doc_id falls back to the current document ...
    from_current = TT.ass_parse_text(index=0)
    assert from_current["source"] == "line" and from_current["doc_id"] == doc_id
    # ... and fails when no document is open at all
    reset_workspace()
    with pytest.raises(ToolError):
        TT.ass_parse_text(index=0)


# --------------------------------------------------------------------------- #
# ass_plain_text / ass_strip_tags
# --------------------------------------------------------------------------- #


def test_plain_text_strips_everything_by_default():
    result = TT.ass_plain_text(text=LINE_TWO_BLOCKS)
    assert result["source"] == "text" and result["index"] is None
    # the drawing path is dropped from the stripped line itself ...
    assert result["text"] == "Hello  world"
    # ... while plain_text is the plain map of the *produced* line: the drawing
    # path is hidden text there because \p1 is gone, but the two spaces and the
    # visible words are exactly what a renderer shows.  (ass_parse_text's
    # plain_text/plain_length instead count every non-tag character, which is what
    # the plain index map is built from.)
    assert result["plain_text"] == "Hello  world"
    assert result["changed"] is True


def test_plain_text_keeps_named_tags_and_groups():
    by_name = TT.ass_plain_text(text=LINE_TWO_BLOCKS, keep="pos")
    assert by_name["text"] == r"{\pos(100,200)}Hello  world"
    assert by_name["keep_names"] == ["pos"] and by_name["kept"] == ["pos"]

    by_group = TT.ass_plain_text(text=r"{\an7\pos(1,2)\clip(0,0,5,5)\k20}ab", keep="clip,karaoke")
    assert by_group["keep_groups"] == ["clip", "karaoke"]
    assert by_group["text"] == r"{\clip(0,0,5,5)\k20}ab"
    assert by_group["plain_text"] == "ab"

    # a group wins over a bare name and expands to its whole family
    both = TT.ass_plain_text(text=r"{\an7\pos(1,2)\move(0,0,1,1,0,10)}x", keep=["layout"])
    assert both["text"] == r"{\an7\pos(1,2)\move(0,0,1,1,0,10)}x"
    # unknown names are reported but simply keep nothing
    unknown = TT.ass_plain_text(text=r"{\b1}x", keep="nosuchtag")
    assert unknown["text"] == "x" and unknown["keep_names"] == ["nosuchtag"]


def test_strip_tags_modes_and_flags():
    base = TT.ass_strip_tags(text=LINE_TWO_BLOCKS, keep="pos")
    assert base["text"] == r"{\pos(100,200)}Hello  world"

    keep_drawing = TT.ass_strip_tags(text=LINE_TWO_BLOCKS, keep="pos", keep_drawing=True)
    assert keep_drawing["keep_drawing"] is True
    assert keep_drawing["text"] == r"{\pos(100,200)}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world"

    removed = TT.ass_strip_tags(text=LINE_TWO_BLOCKS, remove="p", remove_groups="layout")
    # \an7\pos(100,200) is a layout tag and \p1/\p0 were named explicitly
    assert removed["text"] == "Hello m 0 0 l 50 0 50 50 world"
    assert removed["remove_names"] == ["p"] and removed["remove_groups"] == ["layout"]

    karaoke = TT.ass_strip_tags(text=r"{\an7\k20}a{\k30}b", keep_karaoke=True)
    assert karaoke["keep_karaoke"] is True
    assert karaoke["text"] == r"{\k20}a{\k30}b"

    clipped = TT.ass_strip_tags(text=r"{\clip(0,0,9,9)\b1}x", keep_clip=True)
    assert clipped["text"] == r"{\clip(0,0,9,9)}x"

    with pytest.raises(ToolError):
        TT.ass_strip_tags(text=r"{\b1}x", keep_groups="nosuchgroup")


def test_strip_tags_in_place_writes_the_selected_lines():
    doc_id, doc = make_doc(LINE_TWO_BLOCKS, r"{\b1}keep")
    result = TT.ass_strip_tags([0], doc_id=doc_id, in_place=True)
    assert result["written"] == 1
    assert doc.events()[0].text == "Hello  world"
    assert doc.events()[1].text == r"{\b1}keep"  # untouched
    assert_intact(doc.events()[0].text)


# --------------------------------------------------------------------------- #
# ass_set_tag / ass_remove_tag
# --------------------------------------------------------------------------- #


def test_set_tag_where_variants():
    appended = TT.ass_set_tag(text=LINE_TWO_BLOCKS, name="bord", arg="2")
    assert appended["where"] == "after_first_block"
    assert appended["tag"] == r"\bord2"
    assert appended["argument"] == "2"
    assert appended["text"] == (
        r"{\an7\pos(100,200)\bord2}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world"
    )

    prepended = TT.ass_set_tag(text=LINE_TWO_BLOCKS, name="fad", arg="200,200", where="prepend")
    assert prepended["text"] == (
        r"{\fad(200,200)\an7\pos(100,200)}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world"
    )

    prepend_block = TT.ass_set_tag(
        text=LINE_TWO_BLOCKS, name="an", value=8, where="prepend_block"
    )
    assert prepend_block["text"] == (
        r"{\an8}{\an7\pos(100,200)}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world"
    )

    appended_block = TT.ass_set_tag(
        text=LINE_TWO_BLOCKS, name="be", value=1, where="append"
    )
    assert appended_block["text"].endswith(r"{\be1}")
    assert appended_block["text"].startswith(LINE_TWO_BLOCKS)

    # an existing tag is updated, not duplicated
    updated = TT.ass_set_tag(text=LINE_TWO_BLOCKS, name="an", value=4, where="prepend")
    assert updated["text"] == (
        r"{\an4\pos(100,200)}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world"
    )

    wrapped = TT.ass_set_tag(
        text=LINE_TWO_BLOCKS, name="fscx", arg="120", where="wrap"
    )
    # \fscx has no document/style default here, so the implicit default (100) is
    # restored at the end of the line instead of the previous value
    assert wrapped["text"] == (
        r"{\an7\pos(100,200)}{\fscx120}Hello {\p1}m 0 0 l 50 0 50 50{\p0} "
        r"world{\fscx100}"
    )
    for line in (
        appended, prepended, prepend_block, appended_block, updated, wrapped
    ):
        assert_intact(line["text"])

    with pytest.raises(ToolError):
        TT.ass_set_tag(text="x", name="fad", arg="1,1", where="nope")
    with pytest.raises(ToolError):
        TT.ass_set_tag(text="x", name="", arg="1")


def test_set_tag_only_if_missing():
    line = r"{\an7}text"
    skipped = TT.ass_set_tag(text=line, name="an", value=8, only_if_missing=True)
    assert skipped["skipped"] is True
    assert skipped["changed"] is False
    assert skipped["text"] == line

    applied = TT.ass_set_tag(text=line, name="fad", arg="1,1", only_if_missing=True)
    assert applied["skipped"] is False and applied["changed"] is True
    assert applied["text"] == r"{\an7\fad(1,1)}text"


def test_set_tag_rejects_braces_backslash_injection_and_newlines():
    with pytest.raises(ToolError):
        TT.ass_set_tag(text="x", name="fad", arg="{2}")
    with pytest.raises(ToolError):
        TT.ass_set_tag(text="x", name="fad", arg="2}")
    with pytest.raises(ToolError):
        TT.ass_set_tag(text="x", name="fad", arg="2\n3")
    # a backslash inside the argument is *allowed* on purpose: that is how
    # nested tags such as \t(\fscx120) are built.
    nested = TT.ass_set_tag(text="x", name="t", arg=r"0,500,\fscx120")
    assert nested["text"] == r"{\t(0,500,\fscx120)}x"
    assert_intact(nested["text"])


def test_set_tag_in_place_is_snapshot_backed():
    doc_id, doc = make_doc(LINE_TWO_BLOCKS)
    before = doc.events()[0].text
    result = TT.ass_set_tag(index=0, doc_id=doc_id, name="an", value=8, in_place=True)
    assert result["written"] == 1 and result["source"] == "line"
    assert doc.events()[0].text == (
        r"{\an8\pos(100,200)}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world"
    )
    assert B.workspace.undo(doc_id), "the mutation was not snapshot-backed"
    assert doc.events()[0].text == before


def test_remove_tag_single_name_and_list():
    one = TT.ass_remove_tag(text=LINE_TWO_BLOCKS, names="pos")
    assert one["names"] == ["pos"] and one["removed_count"] == 1
    # every removed occurrence is reported with where it was
    assert [tag["name"] for tag in one["removed"]] == ["pos"]
    assert one["removed"][0]["raw"] == r"\pos(100,200)"
    assert one["removed"][0]["argument"] == "100,200"
    assert one["removed"][0]["block"] == 0
    assert one["text"] == r"{\an7}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world"

    two = TT.ass_remove_tag(text=LINE_TWO_BLOCKS, names=["p", "pos"])
    assert two["removed_count"] == 3  # \pos, \p1 and \p0
    assert two["text"] == r"{\an7}Hello m 0 0 l 50 0 50 50 world"
    # every occurrence is gone, so no block is left empty
    assert "{}" not in two["text"]
    for result in (one, two):
        assert_intact(result["text"])

    with pytest.raises(ToolError):
        TT.ass_remove_tag(text="x", names=None)

    doc_id, doc = make_doc(LINE_TWO_BLOCKS)
    in_place = TT.ass_remove_tag(index=0, doc_id=doc_id, names="pos", in_place=True)
    assert in_place["written"] == 1 and doc.events()[0].text == one["text"]


# --------------------------------------------------------------------------- #
# ass_wrap_range / ass_insert_tag_at / ass_apply_tag_to_block
# --------------------------------------------------------------------------- #


def test_wrap_range_plain_hand_computed():
    result = TT.ass_wrap_range(text=LINE_TWO_BLOCKS, start=0, end=5, override=r"\b1")
    assert result["scope"] == "plain"
    assert result["plain_start"] == 0 and result["plain_end"] == 5
    # plain 0..5 is "Hello", raw 19..24
    assert result["raw_start"] == 19 and result["raw_end"] == 24
    assert result["text"] == (
        r"{\an7\pos(100,200)}{\b1}Hello{\b0} {\p1}m 0 0 l 50 0 50 50{\p0} world"
    )
    assert result["plain_text"] == "Hello m 0 0 l 50 0 50 50 world"
    assert_intact(result["text"])

    # the drawing path is plain 6..24 (raw 30..48) even though \p1 sits in front
    drawing = TT.ass_wrap_range(text=LINE_TWO_BLOCKS, start=6, end=24, override=r"\p2")
    assert drawing["raw_start"] == 30 and drawing["raw_end"] == 48
    # the opening block goes *after* the existing {\p1} (which is outside the
    # range) and the closing block restores \p to its previous value, 1
    assert drawing["text"] == (
        r"{\an7\pos(100,200)}Hello {\p1}{\p2}m 0 0 l 50 0 50 50{\p1}{\p0} world"
    )
    assert_intact(drawing["text"])


def test_wrap_range_plain_and_raw_agree_across_an_existing_block():
    """The documented difference: the two counters use different origins."""
    above = TT.ass_parse_text(text=LINE_ONE_BLOCK)
    assert above["raw_length"] == 24 and above["plain_length"] == 5

    # the same *text* can be named either way: plain 0..5 == raw 19..24
    by_plain = TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=0, end=5, override=r"\b1")
    by_raw = TT.ass_wrap_range(
        text=LINE_ONE_BLOCK, start=19, end=24, override=r"\b1", scope="raw"
    )
    assert by_plain["text"] == by_raw["text"]
    assert by_plain["text"] == r"{\an7\pos(100,200)}{\b1}Hello{\b0}"
    assert by_plain["raw_start"] == 19 and by_plain["raw_end"] == 24
    assert by_raw["plain_start"] == 0 and by_raw["plain_end"] == 5
    assert by_raw["scope"] == "raw"

    # but the *same numbers* select completely different things: plain 19 does
    # not exist (the line only has 5 visible characters) ...
    with pytest.raises(ToolError) as plain_err:
        TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=19, end=24, override=r"\b1")
    assert "visible text" in str(plain_err.value)
    # ... while raw 19 is the first visible character.
    assert TT.ass_wrap_range(
        text=LINE_ONE_BLOCK, start=19, end=20, override=r"\b1", scope="raw"
    )["text"] == r"{\an7\pos(100,200)}{\b1}H{\b0}ello"

    # plain 1..2 is inside the word, raw 1..2 is inside the override block
    inside_word = TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=1, end=2, override=r"\i1")
    assert inside_word["text"] == r"{\an7\pos(100,200)}H{\i1}e{\i0}llo"
    with pytest.raises(ToolError) as block_err:
        TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=1, end=2, override=r"\i1", scope="raw")
    assert "override block" in str(block_err.value)


def test_wrap_range_errors_and_in_place():
    with pytest.raises(ToolError):
        TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=6, end=7, override=r"\b1")
    with pytest.raises(ToolError):
        TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=3, end=3, override=r"\b1")
    with pytest.raises(ToolError):
        TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=0, end=2)
    with pytest.raises(ToolError):
        TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=0, end=2, scope="nope", override=r"\b1")
    # a brace *inside* the payload is refused ...
    with pytest.raises(ToolError):
        TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=0, end=2, override=r"\b\i{1}")
    # ... and so is a newline (it would split the event)
    with pytest.raises(ToolError):
        TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=0, end=2, override="\\b1\n\\b2")
    # a fully brace-wrapped payload is accepted and normalised away
    braced = TT.ass_wrap_range(text=LINE_ONE_BLOCK, start=0, end=2, override=r"{\b1}")
    assert braced["override"] == r"\b1"
    assert braced["text"] == r"{\an7\pos(100,200)}{\b1}He{\b0}llo"

    doc_id, doc = make_doc(LINE_ONE_BLOCK)
    result = TT.ass_wrap_range(
        index=0, doc_id=doc_id, start=0, end=5, override=r"\b1", in_place=True
    )
    assert result["written"] == 1 and result["source"] == "line"
    assert doc.events()[0].text == r"{\an7\pos(100,200)}{\b1}Hello{\b0}"


def test_insert_tag_at_plain_index_and_after():
    # before the character at plain_index
    before = TT.ass_insert_tag_at(text=LINE_SIMPLE, plain_index=5, override=r"\b1")
    assert before["plain_index"] == 5 and before["after"] is False
    assert before["text"] == r"{\an7\pos(100,200)}Hello{\b1} world"

    after = TT.ass_insert_tag_at(text=LINE_SIMPLE, plain_index=5, override=r"\b1", after=True)
    assert after["after"] is True
    assert after["text"] == r"{\an7\pos(100,200)}Hello {\b1}world"

    start = TT.ass_insert_tag_at(text=LINE_SIMPLE, plain_index=0, override=r"\b1")
    assert start["text"] == r"{\an7\pos(100,200)}{\b1}Hello world"

    end = TT.ass_insert_tag_at(text=LINE_SIMPLE, plain_index=11, override=r"\b1", after=True)
    assert end["text"] == r"{\an7\pos(100,200)}Hello world{\b1}"

    for result in (before, after, start, end):
        assert_intact(result["text"])

    # plain_index is a *visible* index: 12 is past the end of the 11 visible
    # characters of "Hello world"
    with pytest.raises(ToolError):
        TT.ass_insert_tag_at(text=LINE_SIMPLE, plain_index=12, override=r"\b1")
    with pytest.raises(ToolError):
        TT.ass_insert_tag_at(text=LINE_SIMPLE, plain_index=0, override="")
    # a line without any block gets one at the very start
    bare = TT.ass_insert_tag_at(text="hello world", plain_index=0, override=r"\b1")
    assert bare["text"] == r"{\b1}hello world"
    assert bare["plain_text"] == "hello world" and bare["changed"] is True

    doc_id, doc = make_doc(LINE_SIMPLE)
    written = TT.ass_insert_tag_at(
        index=0, doc_id=doc_id, plain_index=5, override=r"\b1", in_place=True
    )
    assert written["written"] == 1 and doc.events()[0].text == before["text"]


def test_apply_tag_to_block_all_and_single():
    every = TT.ass_apply_tag_to_block(text=LINE_TWO_BLOCKS, block="all", override=r"\bord2")
    assert every["block"] == "all"
    assert every["blocks_total"] == 3 and every["applied_blocks"] == [0, 1, 2]
    assert every["text"] == (
        r"{\an7\pos(100,200)\bord2}Hello {\p1\bord2}m 0 0 l 50 0 50 50"
        r"{\p0\bord2} world"
    )

    second = TT.ass_apply_tag_to_block(text=LINE_TWO_BLOCKS, block=1, override=r"\bord2")
    assert second["applied_blocks"] == [1]
    assert second["text"] == (
        r"{\an7\pos(100,200)}Hello {\p1\bord2}m 0 0 l 50 0 50 50{\p0} world"
    )
    assert_intact(every["text"]) and assert_intact(second["text"])

    with pytest.raises(ToolError):
        TT.ass_apply_tag_to_block(text=LINE_TWO_BLOCKS, block=3, override=r"\bord2")
    with pytest.raises(ToolError):
        TT.ass_apply_tag_to_block(text=LINE_TWO_BLOCKS, block=-1, override=r"\bord2")
    # a line with no block at all gets one at the start, for block=0 and "all"
    created = TT.ass_apply_tag_to_block(text="plain", block=0, override=r"\bord2")
    assert created["text"] == r"{\bord2}plain"
    assert created["applied_blocks"] == [0] and created["blocks_total"] == 1
    assert created["plain_text"] == "plain"
    every_created = TT.ass_apply_tag_to_block(text="plain", block="all", override=r"\bord2")
    assert every_created["text"] == r"{\bord2}plain"
    assert every_created["applied_blocks"] == [0]
    # block=N>0 cannot be created on an empty line
    with pytest.raises(ToolError):
        TT.ass_apply_tag_to_block(text="plain", block=2, override=r"\bord2")
    for result in (created, every_created):
        assert_intact(result["text"])


# --------------------------------------------------------------------------- #
# ass_add_typesetting
# --------------------------------------------------------------------------- #


def test_add_typesetting_builds_the_exact_tag_string():
    doc_id, doc = make_doc(LINE_TWO_BLOCKS)
    result = TT.ass_add_typesetting(
        [0], doc_id=doc_id, pos=(100, 200), an=8, fade=(200, 200), clip=(0, 0, 100, 100)
    )
    assert result["tag_string"] == r"\pos(100,200)\an8\fad(200,200)\clip(0,0,100,100)"
    assert result["override"] == r"{\pos(100,200)\an8\fad(200,200)\clip(0,0,100,100)}"
    assert result["count"] == 1 and result["written"] == 1 and result["in_place"] is True
    assert result["text"] == (
        r"{\pos(100,200)\an8\fad(200,200)\clip(0,0,100,100)}Hello "
        r"{\p1}m 0 0 l 50 0 50 50{\p0} world"
    )
    # the old \an7\pos(100,200) moved into the new block instead of stacking
    assert doc.events()[0].text == result["text"]
    assert doc.events()[0].text.count(r"\an") == 1
    assert doc.events()[0].text.count(r"\pos") == 1
    assert_intact(result["text"])


def test_add_typesetting_pieces_and_order():
    doc_id, _ = make_doc(LINE_SIMPLE)
    result = TT.ass_add_typesetting(
        [0],
        doc_id=doc_id,
        pos=(10, 20),
        an=7,
        move=(0, 0, 100, 200, 0, 2000),
        fade=(0, 255, 255, 200),
        clip={"drawing": "m 0 0 l 10 0 10 10", "inverse": True},
        org=(5, 6),
        extra_tags=r"\bord2",
    )
    assert result["tag_string"] == (
        r"\pos(10,20)\an7\move(0,0,100,200,0,2000)\fade(0,255,255,200,200,200,200)"
        r"\iclip(m 0 0 l 10 0 10 10)\org(5,6)\bord2"
    )
    assert_intact(result["lines"][0]["text"])

    # move with only four numbers, and a two-pair form
    four = TT.ass_add_typesetting([0], doc_id=doc_id, move=(1, 2, 3, 4), in_place=False)
    assert four["tag_string"] == r"\move(1,2,3,4)"
    pairs = TT.ass_add_typesetting(
        [0], doc_id=doc_id, move=((1, 2), (3, 4)), in_place=False
    )
    assert pairs["tag_string"] == r"\move(1,2,3,4)"
    # a clip drawing without a scale
    drawing = TT.ass_add_typesetting(
        [0], doc_id=doc_id, clip="m 0 0 l 5 5 0 5", in_place=False
    )
    assert drawing["tag_string"] == r"\clip(m 0 0 l 5 5 0 5)"
    # \r goes first
    reset = TT.ass_add_typesetting([0], doc_id=doc_id, an=5, reset_first=True, in_place=False)
    assert reset["tag_string"] == r"\r\an5"


def test_add_typesetting_rejects_bad_an_and_shapes():
    doc_id, _ = make_doc(LINE_SIMPLE)
    for bad in (0, 10, -1, 4.5, "8", True, [8]):
        with pytest.raises(ToolError) as err:
            TT.ass_add_typesetting([0], doc_id=doc_id, an=bad)
        assert "an" in str(err.value)
    with pytest.raises(ToolError):
        TT.ass_add_typesetting([0], doc_id=doc_id, pos=(1, 2, 3))
    with pytest.raises(ToolError):
        TT.ass_add_typesetting([0], doc_id=doc_id, org=(1,))
    for bad_fade in ((1, 2, 3), (1, 2, 3, 4, 5)):
        with pytest.raises(ToolError):
            TT.ass_add_typesetting([0], doc_id=doc_id, fade=bad_fade)
    with pytest.raises(ToolError):
        TT.ass_add_typesetting([0], doc_id=doc_id)  # nothing requested
    with pytest.raises(ToolError):
        TT.ass_add_typesetting([0], doc_id=doc_id, extra_tags="{a}b}")
    with pytest.raises(ToolError):
        TT.ass_add_typesetting([9], doc_id=doc_id, an=7)  # no such line
    # in_place=False must not touch the document
    doc = B.workspace.get(doc_id)
    before = doc.events()[0].text
    TT.ass_add_typesetting([0], doc_id=doc_id, an=7, in_place=False)
    assert doc.events()[0].text == before


def test_add_typesetting_is_snapshot_backed():
    doc_id, doc = make_doc(LINE_SIMPLE)
    before = doc.events()[0].text
    TT.ass_add_typesetting([0], doc_id=doc_id, an=8)
    assert doc.events()[0].text != before
    assert B.workspace.undo(doc_id), "typesetting was not snapshot-backed"
    assert doc.events()[0].text == before


# --------------------------------------------------------------------------- #
# ass_swap_an_pos
# --------------------------------------------------------------------------- #

#: ``\an`` -> mirrored ``\an``, plus the expected (row step, column step).
MIRROR = {
    7: (1, -2, 0),
    1: (7, +2, 0),
    9: (3, -2, 0),
    3: (9, +2, 0),
    8: (2, -2, 0),
    2: (8, +2, 0),
    4: (4, 0, 0),
    5: (5, 0, 0),
    6: (6, 0, 0),
}


def test_swap_an_pos_hand_computed_arithmetic_for_two_corners():
    """``dy = -(height / 2) * row_step`` — checked corner by corner."""
    positions = [(400, 200), (400, 300)]
    for an, (expected_an, row_step, _col_step) in ((7, MIRROR[7]), (1, MIRROR[1])):
        for x, y in positions:
            doc_id, doc = make_doc(rf"{{\an{an}\pos({x},{y})}}Test")
            result = TT.ass_swap_an_pos([0], doc_id=doc_id, dry_run=True)
            line = result["lines"][0]
            height = line["measured"]["height"]
            assert height > 0
            assert line["old_an"] == an and line["new_an"] == expected_an
            # the arithmetic under test, written out by hand
            assert line["dy"] == pytest.approx(-(height / 2.0) * row_step)
            assert line["dx"] == pytest.approx(0.0)
            expected_y = y + -(height / 2.0) * row_step
            assert line["new_pos"] == f"{x},{expected_y:g}"
            # ... and the line text that comes out of it
            expected_text = rf"{{\an{expected_an}\pos({x},{expected_y:g})}}Test"
            assert line["after"] == expected_text
            assert result["dry_run"] is True and result["written"] == 0
            assert doc.events()[0].text == rf"{{\an{an}\pos({x},{y})}}Test"

    # 7 -> 1 pushes the anchor *down* by the full line height, 1 -> 7 up
    doc_id, _ = make_doc(r"{\an7\pos(400,200)}Test")
    down = TT.ass_swap_an_pos([0], doc_id=doc_id, dry_run=True)["lines"][0]
    assert down["dy"] == pytest.approx(down["measured"]["height"])
    doc_id, _ = make_doc(r"{\an1\pos(400,300)}Test")
    up = TT.ass_swap_an_pos([0], doc_id=doc_id, dry_run=True)["lines"][0]
    assert up["dy"] == pytest.approx(-up["measured"]["height"])


def test_swap_an_pos_skips_central_alignments_and_lines_without_pos():
    doc_id, doc = make_doc(r"{\an4}x", r"{\an7}no pos here", r"{\an7\pos(10,20)}y")
    result = TT.ass_swap_an_pos([0, 1, 2], doc_id=doc_id, dry_run=True)
    rows = result["lines"]
    assert rows[0]["skipped"] is True and rows[0]["old_an"] == 4 == rows[0]["new_an"]
    assert rows[1]["skipped"] is True and "pos" in (rows[1]["reason"] or "")
    assert rows[1]["after"] == r"{\an7}no pos here"
    assert rows[2]["changed"] is True and rows[2]["new_an"] == 1
    assert result["changed"] == 1 and result["written"] == 0
    assert doc.events()[1].text == r"{\an7}no pos here"


def _measure_line_rect(doc_id: str, path: Path) -> dict:
    B.workspace.save(doc_id, str(path))
    frame = M.measure_render(str(path), 1000)
    if not frame.get("rect"):
        pytest.skip(f"libass measurement unavailable: {frame.get('empty')}")
    return frame["rect"]


def _swap_render_case(tmp_path: Path, text: str, radius: float, **kwargs) -> tuple[dict, dict, dict]:
    doc_id, _ = make_doc(text)
    path = tmp_path / f"swap_{abs(hash(text)) % 10**8}.ass"
    before = _measure_line_rect(doc_id, path)
    result = TT.ass_swap_an_pos([0], doc_id=doc_id, **kwargs)
    after = _measure_line_rect(doc_id, path)
    for key in ("x", "y", "width", "height"):
        assert abs(after[key] - before[key]) <= radius, (
            f"{text}: {key} moved {after[key] - before[key]:+.2f} (> {radius}); "
            f"before={before} after={after}"
        )
    return result, before, after


def test_swap_an_pos_render_box_coincides_vertically(tmp_path):
    """The rendered ink box must not move when \\an is mirrored."""
    for text in (
        r"{\an7\pos(400,200)}Test",
        r"{\an1\pos(400,300)}Test",
        r"{\an8\pos(400,200)}Test",
        r"{\an2\pos(400,300)}Test",
        r"{\an9\pos(400,200)}Test",
        r"{\an3\pos(400,300)}Test",
        r"{\an7\pos(640,360)}A second line with \Ntwo rows",
    ):
        # 0.5 px: the measured values are exact to the last decimal
        _swap_render_case(tmp_path, text, 0.5)


def test_swap_an_pos_render_box_coincides_horizontally(tmp_path):
    """``horizontal=True`` also mirrors the column (an 7 <-> 3, 1 <-> 9 ...).

    Tolerance 1.5 px: the anchor geometry is exact (the box's anchored edge
    lands on the same pixel), but the *opposite* edge can lose its faintest
    antialiased column when the glyphs move to a different subpixel phase.
    """
    for text in (
        r"{\an7\pos(400,200)}Test",
        r"{\an1\pos(400,300)}Test",
        r"{\an9\pos(400,200)}Test",
        r"{\an3\pos(400,300)}Test",
    ):
        _swap_render_case(tmp_path, text, 1.5, horizontal=True)


def test_swap_an_pos_margin_mode_render_and_arithmetic(tmp_path):
    """Margin mode rewrites ``\\an`` *and* the style margin, not ``\\pos``."""
    text = r"{\an7}Margin test"
    doc_id, doc = make_doc(text, MarginV=30)
    play_x, play_y = doc.play_res
    path = tmp_path / "margin.ass"
    before = _measure_line_rect(doc_id, path)
    result = TT.ass_swap_an_pos([0], doc_id=doc_id, margin_mode=True)
    line = result["lines"][0]
    after = _measure_line_rect(doc_id, path)

    assert line["margin_mode"] is True and line["margin_v"] is not None
    height = line["measured"]["height"]
    expected_margin = int(round(play_y - 30 - height))
    # hand-computed: the mirrored \an1 bottom margin keeps the same top edge
    assert line["margin_v"]["before"] == 30
    assert line["margin_v"]["after"] == expected_margin
    # \pos is absent, so only \an is rewritten in the text
    assert line["after"] == r"{\an1}Margin test"
    assert float(doc.events()[0].get("MarginV")) == pytest.approx(expected_margin)
    for key in ("x", "y", "width", "height"):
        assert abs(after[key] - before[key]) <= 0.5, (
            f"{key} moved {after[key] - before[key]:+.2f}; before={before} after={after}"
        )


def test_swap_an_pos_margin_mode_dry_run_writes_nothing_and_undo_restores(tmp_path):
    """A dry run must touch neither the text nor MarginV, and undo must restore both."""
    text = r"{\an7}Margin test"
    doc_id, doc = make_doc(text, MarginV=30)
    plan = TT.ass_swap_an_pos([0], doc_id=doc_id, margin_mode=True, dry_run=True)
    assert plan["dry_run"] is True and plan["written"] == 0
    assert plan["lines"][0]["margin_v"]["after"] != 30  # the plan is computed ...
    assert doc.events()[0].text == text  # ... but nothing is written
    assert doc.events()[0].get("MarginV") == "30"

    done = TT.ass_swap_an_pos([0], doc_id=doc_id, margin_mode=True)
    assert done["written"] == 1
    assert doc.events()[0].text == r"{\an1}Margin test"
    assert float(doc.events()[0].get("MarginV")) == done["lines"][0]["margin_v"]["after"]
    # one undo restores the text *and* the margin: both were written under the
    # same snapshot
    assert B.workspace.undo(doc_id)
    assert doc.events()[0].text == text
    assert doc.events()[0].get("MarginV") == "30"


def test_swap_an_pos_writes_in_place_when_not_dry_run():
    doc_id, doc = make_doc(r"{\an7\pos(100,200)}Test")
    before = doc.events()[0].text
    result = TT.ass_swap_an_pos([0], doc_id=doc_id)
    assert result["dry_run"] is False and result["written"] == 1
    assert doc.events()[0].text == result["lines"][0]["after"]
    assert doc.events()[0].text.startswith(r"{\an1\pos(100,")
    assert_intact(doc.events()[0].text)
    assert B.workspace.undo(doc_id), "the swap was not snapshot-backed"
    assert doc.events()[0].text == before


# --------------------------------------------------------------------------- #
# ass_tag_summary / ass_karaoke_tags_only
# --------------------------------------------------------------------------- #


def test_tag_summary_histogram_groups_and_missing_position():
    doc_id, _ = make_doc(
        LINE_TWO_BLOCKS,
        r"{\k20}ab{\kf30}cd",
        r"plain text, no block",
        r"{\b1}bold but unpositioned",
    )
    result = TT.ass_tag_summary([0, 1, 2, 3], doc_id=doc_id)
    assert result["source"] == "selection" and result["count"] == 4
    # the sanity check this tool exists for: line 1 has a block but no \pos
    assert result["missing_position"] == [1, 2, 3]
    assert result["lines"][1]["missing_position"] is True
    assert result["lines"][1]["first_block_has_pos"] is False
    assert result["lines"][2]["missing_position"] is True
    assert result["lines"][2]["blocks"] == 0 and result["lines"][2]["tags"] == 0
    assert result["lines"][0]["missing_position"] is False

    first = result["lines"][0]
    assert first["tag_names"] == {"an": 1, "p": 2, "pos": 1}
    assert first["tag_groups"] == {"drawing": 2, "layout": 2}
    assert first["blocks"] == 3 and first["tags"] == 4
    assert first["visible_chars"] == 30
    assert first["first_block_has_pos"] is True

    assert result["lines"][1]["tag_names"] == {"k": 1, "kf": 1}
    assert result["lines"][1]["tag_groups"] == {"karaoke": 2}
    assert result["lines"][1]["has_karaoke"] is True
    totals = dict(result["totals"])
    assert totals.pop("visible_chars") == (
        len("Hello m 0 0 l 50 0 50 50 world")  # line 0, the drawing stays visible
        + len("abcd")  # line 1
        + len("plain text, no block")  # line 2
        + len("bold but unpositioned")  # line 3
    )
    # line 0 ends with \p0 and line 1 is the only karaoke line, so only the
    # "karaoke" flag is present: the optional flags only appear when non-zero
    assert totals.pop("karaoke") == 1
    assert totals == {
        # 3 blocks on line 0, 2 on line 1 (\k and \kf), 1 on line 3
        "lines": 4,
        "tags": 7,
        "blocks": 6,
        "missing_position": 3,
    }


def test_tag_summary_single_line_and_text_source():
    result = TT.ass_tag_summary(text=r"{\an7\pos(1,2)}\b1}x")
    assert result["source"] == "text" and result["index"] is None
    assert result["count"] == 1
    assert result["lines"][0]["first_block_has_pos"] is True
    assert result["missing_position"] == []

    doc_id, _ = make_doc(LINE_TWO_BLOCKS)
    single = TT.ass_tag_summary(index=0, doc_id=doc_id)
    assert single["source"] == "line" and single["index"] == 0
    with pytest.raises(ToolError):
        TT.ass_tag_summary(text=LINE_TWO_BLOCKS, index=0, doc_id=doc_id)
    # no selection/index/text at all means "every line of the current document"
    all_lines = TT.ass_tag_summary()
    assert all_lines["source"] == "selection" and all_lines["count"] == 1
    # ... and fails when there is no current document to summarise
    reset_workspace()
    with pytest.raises(ToolError):
        TT.ass_tag_summary()


def test_karaoke_tags_only_reports_arguments_in_order():
    result = TT.ass_karaoke_tags_only(text=r"{\k20}ab{\kf30}cd")
    assert result["source"] == "text"
    assert result["has_karaoke"] is True and result["count"] == 2
    assert result["karaoke_names"] == ["k", "kf"]
    assert result["tags"] == [
        {"name": "k", "argument": "20", "raw": r"\k20", "plain_index": 0, "block": 0, "text": "ab"},
        {"name": "kf", "argument": "30", "raw": r"\kf30", "plain_index": 2, "block": 1, "text": "cd"},
    ]
    assert result["syllable_text"] == "abcd"
    assert result["times_ms"] == [20, 30] and result["total_ms"] == 50

    none = TT.ass_karaoke_tags_only(text=r"{\an7}plain")
    assert none["has_karaoke"] is False and none["tags"] == [] and none["total_ms"] == 0

    doc_id, _ = make_doc(r"{\K15}ab")
    from_line = TT.ass_karaoke_tags_only(index=0, doc_id=doc_id)
    assert from_line["source"] == "line" and from_line["count"] == 1
    # \K is the legacy SSA spelling of the fill karaoke tag: reported under its
    # canonical ASS name (kf) while "raw" keeps the original spelling
    assert from_line["tags"][0]["name"] == "kf"
    assert from_line["tags"][0]["raw"] == r"\K15"
    assert from_line["times_ms"] == [15]
    # index= without doc_id uses the current document ...
    assert TT.ass_karaoke_tags_only(index=0)["source"] == "line"
    # ... and fails when there is no document open at all
    reset_workspace()
    with pytest.raises(ToolError):
        TT.ass_karaoke_tags_only(index=0)


# --------------------------------------------------------------------------- #
# ass_convert_tags
# --------------------------------------------------------------------------- #


def test_convert_tags_to_ass_lists_every_replacement():
    doc_id, doc = make_doc(
        r"{\a6\K20}ab",
        r"{\a5\k30}c{\a4}d",
        r"{\an7\kf10}already modern",
        r"no tags at all",
    )
    result = TT.ass_convert_tags([0, 1, 2, 3], doc_id=doc_id)
    assert result["mode"] == "to_ass" and result["dry_run"] is False
    assert result["count"] == 4 and result["changed"] == 2
    assert result["written"] == 2 and result["replacement_count"] == 3

    first = result["lines"][0]
    assert first["before"] == r"{\a6\K20}ab" and first["after"] == r"{\an8\kf20}ab"
    assert first["changed"] is True
    assert first["replacements"] == [
        {
            "from": r"\a6",
            "to": r"\an8",
            "kind": "alignment",
            "note": r"legacy SSA \a6 -> \an8",
        },
        {
            "from": r"\K",
            "to": r"\kf",
            "kind": "karaoke",
            "note": r"legacy SSA \K -> \kf (fill karaoke)",
        },
    ]
    # SSA \a5 is the top-left alignment, i.e. ASS \an7; \a4 is *not* a valid SSA
    # alignment, so that occurrence is left exactly as it was
    assert result["lines"][1]["after"] == r"{\an7\k30}c{\a4}d"
    assert result["lines"][1]["replacements"] == [
        {
            "from": r"\a5",
            "to": r"\an7",
            "kind": "alignment",
            "note": r"legacy SSA \a5 -> \an7",
        }
    ]
    assert result["lines"][2]["changed"] is False
    assert result["lines"][2]["replacements"] == []
    assert result["lines"][3]["changed"] is False
    assert doc.events()[0].text == r"{\an8\kf20}ab"
    assert doc.events()[3].text == "no tags at all"
    for event in doc.events():
        assert_intact(event.text)


def test_convert_tags_to_ssa_and_dry_run():
    doc_id, doc = make_doc(r"{\an8\kf20}ab")
    dry = TT.ass_convert_tags([0], doc_id=doc_id, mode="to_ssa", dry_run=True)
    assert dry["dry_run"] is True and dry["written"] == 0
    assert dry["replacement_count"] == 2
    assert dry["lines"][0]["after"] == r"{\a6\K20}ab"
    assert doc.events()[0].text == r"{\an8\kf20}ab"  # untouched

    real = TT.ass_convert_tags([0], doc_id=doc_id, mode="to_ssa")
    assert real["written"] == 1 and doc.events()[0].text == r"{\a6\K20}ab"
    # round-tripping back to ASS restores the original line
    back = TT.ass_convert_tags([0], doc_id=doc_id, mode="to_ass")
    assert back["lines"][0]["after"] == r"{\an8\kf20}ab"

    with pytest.raises(ToolError):
        TT.ass_convert_tags([0], doc_id=doc_id, mode="nope")


def test_convert_tags_leaves_visible_text_alone():
    doc_id, doc = make_doc(r"{\a6}x\K y")
    TT.ass_convert_tags([0], doc_id=doc_id)
    assert doc.events()[0].text == r"{\an8}x\K y"  # \K outside a block is text


# --------------------------------------------------------------------------- #
# byte-exact round trip through the tool layer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fixture", ["basic.ass", "basic.ssa"])
def test_real_fixture_round_trips_byte_for_byte(fixture, tmp_path):
    """Open a real fixture, change nothing, save: identical bytes."""
    source = REAL / fixture
    original = source.read_bytes()
    reset_workspace()
    doc_id = B.workspace.open(str(source))
    # a read-only tool pass must not disturb anything either
    TT.ass_tag_summary([], doc_id=doc_id)
    TT.ass_parse_text(index=0, doc_id=doc_id)
    out = tmp_path / fixture
    info = B.workspace.save(doc_id, str(out))
    assert info["bytes"] == len(original)
    assert out.read_bytes() == original


def test_real_fixture_round_trips_after_a_swap(tmp_path):
    """A real fixture with a \an7 line survives a swap + save byte-exactly."""
    reset_workspace()
    doc_id = B.workspace.open(str(BASIC_ASS))
    doc = B.workspace.get(doc_id)
    doc.add_event(
        text=r"{\an7\pos(640,360)}Typeset me",
        Start="0:00:10.00",
        End="0:00:12.00",
        Style="Default",
    )
    index = len(doc.events()) - 1
    path = tmp_path / "basic.ass"
    B.workspace.save(doc_id, str(path))
    swap = TT.ass_swap_an_pos([index], doc_id=doc_id)
    assert swap["written"] == 1
    B.workspace.save(doc_id)
    text = path.read_text(encoding="utf-8-sig")
    assert rf"{{\an1\pos(640," in text.replace("\r\n", "\n")
    assert T.parse(swap["lines"][0]["after"]).render() == swap["lines"][0]["after"]


# --------------------------------------------------------------------------- #
# escaping / structural integrity of every mutation
# --------------------------------------------------------------------------- #


TORTURE_LINES = [
    "",
    "plain",
    r"{\an7\pos(100,200)}Hello {\p1}m 0 0 l 50 0 50 50{\p0} world",
    r"{\b1}bold{\b0}normal",
    r"{}empty block} stray text",
    r"a{literal}brace",
    r"{\t(0,500,\fscx120\fscy120)}animated",
    r"{\clip(m 0 0 l 10 0 10 10)\iclip(0,0,5,5)\k20\K30}mixed",
    r"\\N literal backslashes \\and \\N",
    r"{\an}tag without argument",
]

MUTATIONS = [
    lambda line: TT.ass_set_tag(text=line, name="bord", arg="2"),
    lambda line: TT.ass_set_tag(text=line, name="fad", arg="10,10", where="prepend"),
    lambda line: TT.ass_set_tag(text=line, name="an", value=4, where="prepend_block"),
    lambda line: TT.ass_set_tag(text=line, name="be", value=1, where="append"),
    lambda line: TT.ass_remove_tag(text=line, names=["p", "pos"]),
    lambda line: TT.ass_plain_text(text=line),
    lambda line: TT.ass_strip_tags(text=line, keep="pos,an", keep_drawing=True),
    lambda line: TT.ass_strip_tags(text=line),
    lambda line: TT.ass_insert_tag_at(text=line, plain_index=0, override=r"\b1"),
    lambda line: TT.ass_apply_tag_to_block(text=line, block="all", override=r"\bord1"),
    lambda line: _convert_line(line),
]


def _convert_line(line: str) -> dict:
    """Run the (selection-only) ``ass_convert_tags`` tool against one raw line."""
    doc_id, _ = make_doc(line)
    result = TT.ass_convert_tags([0], doc_id=doc_id, dry_run=True)
    # the integrity tests look at the produced text, which a conversion reports
    # as "after" (nothing is written because dry_run is set)
    return {**result, "text": result["lines"][0]["after"], "index": 0}


def _wrap_first_visible(line: str):
    plain_len = T.plain_len(line)
    if plain_len == 0:
        return None
    return TT.ass_wrap_range(text=line, start=0, end=plain_len, override=r"\b1")


def test_no_tool_ever_breaks_braces_or_backslashes():
    """Every produced line parses back byte-for-byte and keeps braces balanced."""
    for line in TORTURE_LINES:
        # a line that already carries a stray brace keeps that imbalance, so the
        # count check only applies to lines that started out balanced
        balanced = line.count("{") == line.count("}")
        for mutation in MUTATIONS:
            try:
                result = mutation(line)
            except ToolError:
                continue  # a refused edit is fine, a corrupting one is not
            text = result.get("text")
            if text is None:
                continue
            assert_intact(text, balanced=balanced)
        wrapped = _wrap_first_visible(line)
        if wrapped is not None:
            assert_intact(wrapped["text"], balanced=balanced)


def test_every_override_block_written_has_a_closing_brace():
    """No tool ever opens a block it does not close, and never nests one."""
    for line in TORTURE_LINES:
        for mutation in MUTATIONS:
            try:
                result = mutation(line)
            except ToolError:
                continue
            text = result.get("text")
            if text is None:
                continue
            depth = 0
            for char in text:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                # a stray "}" in visible text is legal and preserved; a depth of
                # more than one would mean a *nested* block, which is never valid
                assert depth <= 1, f"nested override block in {text!r}"
            # brace balance is preserved exactly: every "{" that a tool added was
            # matched by a "}", so the unmatched-brace count cannot have grown
            assert text.count("{") - text.count("}") == line.count("{") - line.count("}"), (
                f"unclosed override block in {text!r} (input was {line!r})"
            )


def test_stray_braces_in_visible_text_are_preserved():
    line = r"{\b1}a}b{literal"
    for mutation in MUTATIONS:
        try:
            result = mutation(line)
        except ToolError:
            continue
        text = result["text"]
        # the input already carries a stray "}" in visible text, so only the
        # parse round-trip is required here, not matched brace counts
        assert_intact(text, balanced=False)
        # the stray "}" of the visible text is still there and still unbalanced
        # *as text*, which is exactly what the ASS format allows
        assert text.count("}") >= 1
