"""Tests for :mod:`aegisub_mcp.tools.lines` — project/file + line access tools.

The tools are exercised end-to-end through their public ``ass_*`` entry points
against the real-world fixtures in ``tests/fixtures/real``.  Line indices in
every assertion are the 0-based ``AssDocument.events()`` indices the tools use.

Nothing in this file ever writes to a fixture: mutation tests work in memory or
re-point the document at a ``tmp_path`` copy first.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import pathlib

import pytest

from aegisub_mcp.tools import lines as L
from aegisub_mcp.tools.base import ToolError, workspace

FIX = pathlib.Path(__file__).parent / "fixtures" / "real"
FIXTURES = sorted(FIX.glob("*.ass")) + sorted(FIX.glob("*.ssa"))
BASIC = FIX / "basic.ass"
SSA = FIX / "basic.ssa"

#: every tool the brief requires
REQUIRED_TOOLS = {
    "ass_new_document", "ass_open", "ass_save", "ass_save_all", "ass_close",
    "ass_list_documents", "ass_select_document", "ass_document_info",
    "ass_select", "ass_get_selection", "ass_list_lines", "ass_get_line",
    "ass_add_line", "ass_add_lines", "ass_update_line", "ass_update_lines",
    "ass_delete_lines", "ass_duplicate_lines", "ass_move_lines",
    "ass_split_line", "ass_merge_lines", "ass_find_replace", "ass_set_comment",
    "ass_sort_lines", "ass_undo", "ass_redo", "ass_undo_history",
    "ass_export_text", "ass_import_srt", "ass_stats",
}


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def clean_workspace():
    """Every test starts from a pristine module-level workspace singleton."""
    workspace.__init__()
    yield
    workspace.__init__()


def open_basic(name: str = "basic.ass", doc_id: str | None = None) -> dict:
    """Open a real fixture in the workspace and return the summary."""
    return L.ass_open(str(FIX / name), doc_id=doc_id)


def open_copy(tmp_path: pathlib.Path, name: str = "basic.ass", target: str = "copy.ass") -> pathlib.Path:
    """Open a fixture and re-point it at a tmp_path copy so saves stay local."""
    open_basic(name)
    dest = tmp_path / target
    L.ass_save(path=str(dest))
    return dest


def text_of(index: int) -> str:
    return L.ass_get_line(index)["text"]


def assert_json(result) -> None:
    json.dumps(result)  # raises TypeError if the result is not serialisable


# --------------------------------------------------------------------------- #
# module contract
# --------------------------------------------------------------------------- #


def test_required_tools_exist_and_are_callable() -> None:
    for name in sorted(REQUIRED_TOOLS):
        fn = getattr(L, name, None)
        assert fn is not None, f"{name} is missing from tools.lines"
        assert callable(fn), f"{name} is not callable"
        assert fn.__doc__, f"{name} has no docstring"
        assert "0-based" in fn.__doc__, f"{name} docstring must document 0-based indices"


class FakeMcp:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def tool(self):
        def decorator(fn):
            self.registered.append(fn.__name__)
            return fn

        return decorator


def test_register_applies_mcp_tool_to_every_ass_function() -> None:
    mcp = FakeMcp()
    names = L.register(mcp, workspace)
    assert names == sorted(names), "register() must return a sorted list"
    assert set(names) == REQUIRED_TOOLS
    assert set(mcp.registered) == REQUIRED_TOOLS
    module_tools = {
        n for n, obj in vars(L).items()
        if n.startswith("ass_") and callable(obj) and getattr(obj, "__module__", None) == L.__name__
    }
    assert module_tools == REQUIRED_TOOLS


# --------------------------------------------------------------------------- #
# documents: new / open / save / close / list
# --------------------------------------------------------------------------- #


def test_new_document_returns_summary() -> None:
    result = L.ass_new_document()
    # base.Workspace names an unsaved document after the default file name
    assert result["doc_id"] == "untitled.ass"
    assert result["play_res_x"] == 1920 and result["play_res_y"] == 1080
    assert result["script_type"] == "v4.00+"
    assert result["lines"] == 0
    assert result["dirty"] is False
    assert result["path"] is None

    custom = L.ass_new_document(play_res_x=640, play_res_y=480, doc_id="tiny", script_type="v4.00")
    assert custom["doc_id"] == "tiny"
    assert custom["play_res_x"] == 640
    assert custom["script_type"] == "v4.00"
    assert workspace.current == "tiny"
    assert "tiny" in L.ass_list_documents()["ids"]

    with pytest.raises(ToolError):
        L.ass_new_document(script_type="v5.00")


def test_new_document_is_editable_and_savable(tmp_path: pathlib.Path) -> None:
    L.ass_new_document()
    added = L.ass_add_line(1000, 2000, "hello")
    assert added["index"] == 0
    dest = tmp_path / "fresh.ass"
    saved = L.ass_save(path=str(dest))
    assert saved["changed"] is True
    assert saved["bytes_written"] == dest.stat().st_size
    assert "[Events]" in dest.read_text(encoding="utf-8")
    assert L.ass_list_documents()["documents"][0]["dirty"] is False


def test_open_returns_document_summary() -> None:
    result = open_basic()
    assert result["doc_id"] == "basic.ass"
    assert result["path"] == str(BASIC)
    assert result["encoding"] == "utf-8"
    assert result["has_bom"] is True
    assert result["newline"] == repr("\r\n")
    assert result["play_res_x"] == 1280 and result["play_res_y"] == 720
    assert result["script_type"] == "v4.00+"
    assert result["lines"] == 2
    assert result["styles"] == 1
    assert result["dirty"] is False
    assert_json(result)


def test_open_missing_file_raises_tool_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ToolError) as excinfo:
        L.ass_open(str(tmp_path / "nope.ass"))
    assert "not found" in str(excinfo.value).lower()


def test_open_undecodable_file_raises_tool_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ToolError):
        L.ass_open(str(BASIC), encoding="definitely-not-a-codec")

    bad = tmp_path / "bad.ass"
    bad.write_bytes(b"\xff\xfe\x00")  # odd length for utf-16
    with pytest.raises(ToolError) as excinfo:
        L.ass_open(str(bad), encoding="utf-16-le")
    assert "decode" in str(excinfo.value).lower()

    garbage = tmp_path / "garbage.ass"
    garbage.write_bytes(b"\x00\x01\x02\xff\xfe binary")  # 0xff/0xfe are not ASCII
    with pytest.raises(ToolError):
        L.ass_open(str(garbage), encoding="ascii")

    # a bogus codec is rejected even when the path is already open
    L.ass_open(str(BASIC))
    with pytest.raises(ToolError):
        L.ass_open(str(BASIC), encoding="definitely-not-a-codec")


def test_open_with_explicit_encoding_keeps_bytes(tmp_path: pathlib.Path) -> None:
    result = L.ass_open(str(SSA), encoding="utf-8")
    assert result["doc_id"] == "basic.ssa"
    dest = tmp_path / "ssa.ass"
    saved = L.ass_save(path=str(dest))
    assert saved["encoding"] == "utf-8"
    assert dest.read_bytes() == SSA.read_bytes()


def test_open_two_documents_keeps_both_registered() -> None:
    first = open_basic()
    second = L.ass_open(str(SSA))
    listing = L.ass_list_documents()
    assert listing["ids"] == [first["doc_id"], second["doc_id"]]
    assert listing["current"] == second["doc_id"]
    assert [d["current"] for d in listing["documents"]] == [False, True]
    assert all(d["path"] for d in listing["documents"])


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_save_round_trip_is_byte_exact(path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The tool layer must re-save an untouched real fixture byte for byte."""
    original = path.read_bytes()
    opened = L.ass_open(str(path))
    assert opened["dirty"] is False
    dest = tmp_path / path.name
    saved = L.ass_save(path=str(dest))

    written = dest.read_bytes()
    assert written == original, f"{path.name}: tool-layer round-trip changed bytes"
    assert saved["path"] == str(dest)
    assert saved["bytes_written"] == len(original)
    assert saved["sha256"] == hashlib.sha256(original).hexdigest()
    # the destination did not exist before, so the bytes on disk did change
    assert saved["changed"] is True
    # has_bom always describes the bytes that were actually written
    assert saved["has_bom"] is written.startswith(codecs.BOM_UTF8)
    assert_json(saved)


def test_save_changed_flag_tracks_destination_bytes(tmp_path: pathlib.Path) -> None:
    """``changed`` compares the bytes written with what the destination held."""
    open_basic()
    dest = tmp_path / "same.ass"
    first = L.ass_save(path=str(dest))  # destination did not exist yet
    assert first["changed"] is True
    second = L.ass_save(path=str(dest))  # identical bytes written again
    assert second["changed"] is False
    assert second["sha256"] == first["sha256"]
    L.ass_add_line(1000, 2000, "now it differs")
    assert L.ass_save(path=str(dest))["changed"] is True
    before = dest.read_bytes()
    # this fixture is CRLF-native, so forcing LF is the byte-changing override
    L.ass_save(path=str(dest), newline="\n")
    assert dest.read_bytes() != before
    assert L.ass_save(path=str(dest), newline="\n")["changed"] is False


def test_save_reports_change_after_edit(tmp_path: pathlib.Path) -> None:
    original = BASIC.read_bytes()
    open_basic()
    L.ass_add_line(9000, 9500, "extra")
    dest = tmp_path / "edited.ass"
    saved = L.ass_save(path=str(dest))
    assert saved["changed"] is True
    assert dest.read_bytes() != original
    assert dest.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM preserved
    assert b"extra" in dest.read_bytes()


def test_save_honours_encoding_bom_and_newline(tmp_path: pathlib.Path) -> None:
    open_basic()
    dest = tmp_path / "utf16.ass"
    saved = L.ass_save(path=str(dest), encoding="utf-16-le", bom=False, newline="\n")
    assert saved["encoding"] == "utf-16-le"
    assert saved["has_bom"] is False
    raw = dest.read_bytes()
    assert raw.startswith(b"[\x00")  # utf-16-le without BOM: '[Script Info]'
    assert b"\r\x00\n\x00" not in raw

    dest2 = tmp_path / "lf.ass"
    # a second, independent load of the same file is still utf-8, so ``bom=True``
    # adds a UTF-8 BOM: the document layer only synthesises a BOM for utf-8
    # (encoding to utf-16-le never emits one)
    fresh = L.ass_open(str(BASIC), doc_id="fresh-copy")
    saved2 = L.ass_save(doc_id=fresh["doc_id"], path=str(dest2), bom=True, newline="\r\n")
    assert saved2["encoding"] == "utf-8"
    assert saved2["has_bom"] is True
    raw2 = dest2.read_bytes()
    assert raw2.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in raw2
    assert b"\n" not in raw2.replace(b"\r\n", b"")  # CRLF on every line
    assert saved2["bytes_written"] == len(raw2)


def test_save_create_backup_writes_bak(tmp_path: pathlib.Path) -> None:
    dest = open_copy(tmp_path)
    L.ass_add_line(9000, 9500, "second")
    saved = L.ass_save(path=str(dest), create_backup=True)
    backup = pathlib.Path(str(dest) + ".bak")
    assert saved["backup"] == str(backup)
    assert backup.exists()
    assert backup.read_bytes() != dest.read_bytes()


def test_save_all_saves_known_paths_and_skips_unsaved(tmp_path: pathlib.Path) -> None:
    dest = open_copy(tmp_path)               # knows a path -> saved
    L.ass_new_document(doc_id="scratch")     # never saved  -> skipped
    L.ass_add_line(1000, 2000, "edit", doc_id="basic.ass")
    result = L.ass_save_all()
    assert result["count"] == 1
    assert [s["doc_id"] for s in result["saved"]] == ["basic.ass"]
    assert result["skipped"] == ["scratch"]
    assert result["errors"] == []
    assert result["saved"][0]["path"] == str(dest)
    assert b"edit" in dest.read_bytes()
    assert L.ass_list_documents()["documents"][0]["dirty"] is False


def test_close_with_and_without_save(tmp_path: pathlib.Path) -> None:
    dest = open_copy(tmp_path)
    L.ass_add_line(9000, 9500, "flush me")
    closed = L.ass_close(save=True)
    assert closed["closed"] is True and closed["saved"] is True
    assert closed["doc_id"] == "basic.ass"
    assert closed["remaining"] == []
    assert b"flush me" in dest.read_bytes()

    open_basic()
    L.ass_add_line(9000, 9500, "discard me")
    discarded = L.ass_close(save=False)
    assert discarded["saved"] is False
    assert b"discard me" not in BASIC.read_bytes()
    assert L.ass_list_documents()["ids"] == []


def test_close_unknown_document_raises() -> None:
    with pytest.raises(ToolError):
        L.ass_close(doc_id="ghost")


def test_select_document_switches_current() -> None:
    open_basic()
    L.ass_open(str(SSA))
    assert L.ass_list_documents()["current"] == "basic.ssa"
    result = L.ass_select_document("basic.ass")
    assert result["doc_id"] == "basic.ass"
    assert workspace.current == "basic.ass"
    assert L.ass_list_documents()["current"] == "basic.ass"
    with pytest.raises(ToolError):
        L.ass_select_document("nope")


# --------------------------------------------------------------------------- #
# document info
# --------------------------------------------------------------------------- #


def test_document_info_reports_sections_styles_and_events() -> None:
    open_basic()
    info = L.ass_document_info()
    assert info["doc_id"] == "basic.ass"
    assert info["section_kinds"] == ["script_info", "styles", "events", "other"]
    assert info["section_order"] == info["section_headers"]
    assert info["style_names"] == ["Default"]
    assert info["event_count"] == 2
    assert info["comment_count"] == 1
    assert info["dialogue_count"] == 1
    assert info["duration_ms"] == 3000          # 1.00 -> 4.00
    assert info["start_ms"] == 1000
    assert info["end_ms"] == 6000
    assert info["span_ms"] == 5000
    assert info["info"]["Title"] == "Fulmen fixture"
    assert "Layer" in info["format_order"]
    assert info["font_names"] == [] and info["graphic_names"] == []
    assert_json(info)


def test_document_info_reports_attachments_for_legacy_fixture() -> None:
    L.ass_open(str(FIX / "legacy-attachments-real.ass"))
    info = L.ass_document_info()
    assert "Sample_0.ttf" in info["font_names"]


def test_document_info_accepts_explicit_doc_id() -> None:
    open_basic()
    L.ass_new_document(doc_id="empty")
    info = L.ass_document_info("basic.ass")
    assert info["doc_id"] == "basic.ass" and info["event_count"] == 2
    assert L.ass_document_info("empty")["event_count"] == 0


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #


@pytest.fixture
def three_lines() -> None:
    """A document with four lines (2 dialogues + 1 comment + 1 dialogue)."""
    L.ass_new_document(doc_id="sel")
    L.ass_add_line(1000, 2000, "first")
    L.ass_add_line(3000, 4000, "second")
    L.ass_add_line(5000, 6000, "third", comment=True)
    L.ass_add_line(7000, 8000, "fourth")


def test_select_modes_and_get_selection(three_lines) -> None:
    assert L.ass_select([0, 2])["selection"] == [0, 2]
    assert L.ass_select("1", mode="add")["selection"] == [0, 1, 2]
    assert L.ass_select("0,2", mode="remove")["selection"] == [1]
    assert L.ass_select(2, mode="toggle")["selection"] == [1, 2]
    assert L.ass_select(2, mode="toggle")["selection"] == [1]

    state = L.ass_get_selection()
    assert state["selection"] == [1]
    assert state["count"] == 1
    assert state["lines"][0]["text"] == "second"
    assert state["doc_id"] == "sel"

    everything = L.ass_select(None)
    assert everything["selection"] == [0, 1, 2, 3]
    assert L.ass_get_selection()["count"] == 4

    with pytest.raises(ToolError):
        L.ass_select(0, mode="nonsense")


def test_select_accepts_structured_specs(three_lines) -> None:
    assert L.ass_select({"kind": "comment"})["selection"] == [2]
    assert L.ass_select({"text_contains": "third"})["selection"] == [2]
    assert L.ass_select({"range": [1, 2]})["selection"] == [1, 2]
    assert L.ass_select("0-3")["selection"] == [0, 1, 2, 3]
    with pytest.raises(ToolError):
        L.ass_select(99)


# --------------------------------------------------------------------------- #
# line listing / retrieval
# --------------------------------------------------------------------------- #


def test_list_lines_paging_and_flags(three_lines) -> None:
    everything = L.ass_list_lines()
    assert everything["total"] == 4
    assert everything["offset"] == 0
    assert [line["index"] for line in everything["lines"]] == [0, 1, 2, 3]
    assert everything["lines"][0]["text"] == "first"
    assert everything["lines"][2]["comment"] is True
    assert "plain_text" in everything["lines"][0]
    assert "tags_summary" not in everything["lines"][0]

    page = L.ass_list_lines(offset=1, limit=2)
    assert page["total"] == 4 and page["offset"] == 1
    assert [line["index"] for line in page["lines"]] == [1, 2]

    with_tags = L.ass_list_lines(limit=1, include_tags_summary=True, include_plain_text=False)
    assert "plain_text" not in with_tags["lines"][0]
    assert "tags_summary" in with_tags["lines"][0]

    selection_page = L.ass_list_lines(selection=[3], offset=0, limit=10)
    assert [line["index"] for line in selection_page["lines"]] == [3]

    assert_json(everything)
    with pytest.raises(ToolError):
        L.ass_list_lines(offset=-1)


def test_get_line_details() -> None:
    open_basic()
    line = L.ass_get_line(0)
    assert line["index"] == 0
    assert line["kind"] == "Dialogue"
    assert line["start_ms"] == 1000 and line["end_ms"] == 4000
    assert line["start"] == "0:00:01.00" and line["end"] == "0:00:04.00"
    assert line["text"] == r"Hello {\i1}world{\i0}"
    assert line["plain_text"] == "Hello world"
    assert line["style"] == "Default"
    assert line["resolved_style"] == "Default"
    assert line["margin_l"] == "0" and line["margin_v"] == "0"
    assert line["fields"]["Style"] == "Default"
    assert line["drawing"]["active"] is False
    assert line["drawing"]["state"] == 0
    assert line["drawing"]["drawing_segments"] == 0
    assert line["karaoke"]["has_karaoke"] is False
    assert line["timing"]["duration_ms"] == 3000
    assert line["timing"]["characters"] == 11
    assert line["cps"] == pytest.approx(11 / 3, abs=0.01)
    assert "i" in line["tags_summary"]["tag_names"]
    assert line["tags_summary"]["has_drawing"] is False
    assert_json(line)

    comment = L.ass_get_line(1)
    assert comment["comment"] is True and comment["kind"] == "Comment"
    assert comment["timing"]["duration_ms"] == 1000

    with pytest.raises(ToolError):
        L.ass_get_line(99)
    with pytest.raises(ToolError):
        L.ass_get_line(-1)


def test_get_line_reports_drawing_and_karaoke() -> None:
    L.ass_new_document(doc_id="shapes")
    drawing = L.ass_add_line(0, 5000, r"{\p1}m 0 0 l 100 0 100 100 0 100{\p0}", comment=False)
    kara = L.ass_add_line(6000, 8000, r"{\k30}hel{\k20}lo")

    shape = L.ass_get_line(drawing["index"])
    assert shape["drawing"]["active"] is True
    # ``state`` is the \p state at the *end* of the line, which the usual
    # {\p1}path{\p0} shape resets to 0; the segment counts prove the drawing
    assert shape["drawing"]["state"] == 0
    assert shape["drawing"]["segments"] == 3
    assert shape["drawing"]["drawing_segments"] == 2
    assert shape["drawing"]["path"].startswith("m 0 0")
    assert shape["tags_summary"]["has_drawing"] is True

    karaoke = L.ass_get_line(kara["index"])
    assert karaoke["karaoke"]["has_karaoke"] is True
    assert karaoke["karaoke"]["total_ms"] == 500
    assert [s["duration_ms"] for s in karaoke["karaoke"]["syllables"]] == [300, 200]


# --------------------------------------------------------------------------- #
# adding lines
# --------------------------------------------------------------------------- #


def test_add_line_accepts_ms_and_time_strings() -> None:
    L.ass_new_document(doc_id="add")
    first = L.ass_add_line(1000, 2500, "numeric")
    assert first["index"] == 0
    assert first["line"]["start_ms"] == 1000 and first["line"]["end_ms"] == 2500

    second = L.ass_add_line("0:00:03.50", "0:00:04.25", "strings", style="Default",
                            actor="Alice", effect="fx", layer=3,
                            margin_l=10, margin_r=20, margin_v=30)
    assert second["index"] == 1
    line = second["line"]
    assert line["start"] == "0:00:03.50" and line["end_ms"] == 4250
    assert line["actor"] == "Alice" and line["effect"] == "fx"
    assert line["layer"] == "3"
    assert line["margin_l"] == "10" and line["margin_r"] == "20" and line["margin_v"] == "30"

    commented = L.ass_add_line(5000, 6000, "a comment", comment=True)
    assert commented["line"]["comment"] is True
    assert commented["line"]["kind"] == "Comment"
    assert L.ass_list_lines()["total"] == 3

    clamped = L.ass_add_line(-5000, 10_000_000, "clamped")
    assert clamped["line"]["start_ms"] == 0
    assert clamped["line"]["end_ms"] <= L.MAX_MS


def test_add_line_rejects_bad_times() -> None:
    L.ass_new_document(doc_id="bad")
    with pytest.raises(ToolError) as excinfo:
        L.ass_add_line(5000, 1000, "backwards")
    assert "before start" in str(excinfo.value)
    with pytest.raises(ToolError):
        L.ass_add_line("not-a-time", 1000, "junk")
    with pytest.raises(ToolError):
        L.ass_add_line(True, 1000, "bool")
    assert L.ass_list_lines()["total"] == 0


def test_add_lines_inserts_at_requested_indices() -> None:
    L.ass_new_document(doc_id="bulk")
    L.ass_add_line(1000, 2000, "existing")
    result = L.ass_add_lines([
        {"start_ms": 3000, "end_ms": 4000, "text": "appended"},
        {"start_ms": 500, "end_ms": 900, "text": "at front", "index": 0},
        {"start_ms": 4500, "end_ms": 5000, "text": "in middle", "index": 1, "comment": True},
    ])
    assert result["indices"] == [3, 0, 1]
    # entries are applied in order: "in middle" is inserted at position 1 of the
    # list as it exists at that moment (["at front", "existing", "appended"])
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == [
        "at front", "in middle", "existing", "appended",
    ]
    assert result["lines"][2]["comment"] is True
    assert result["count"] == 3
    assert_json(result)

    with pytest.raises(ToolError):
        L.ass_add_lines([])
    with pytest.raises(ToolError):
        L.ass_add_lines(["not a dict"])
    with pytest.raises(ToolError):
        L.ass_add_lines([{"start_ms": 5000, "end_ms": 1000, "text": "bad"}])


# --------------------------------------------------------------------------- #
# updating lines
# --------------------------------------------------------------------------- #


def test_update_line_changes_only_given_fields() -> None:
    open_basic()
    before = L.ass_get_line(0)
    result = L.ass_update_line(0, text="new text", start_ms="0:00:00.50")
    assert result["index"] == 0
    assert sorted(result["changed"]) == ["start_ms", "text"]
    line = result["line"]
    assert line["text"] == "new text"
    assert line["start_ms"] == 500
    assert line["end_ms"] == before["end_ms"]
    assert line["style"] == before["style"]
    assert line["actor"] == before["actor"]
    assert line["margin_l"] == before["margin_l"]

    only_actor = L.ass_update_line(0, actor="Bob")
    assert only_actor["changed"] == ["actor"]
    assert only_actor["line"]["actor"] == "Bob"
    assert only_actor["line"]["text"] == "new text"

    margins = L.ass_update_line(1, margin_l=5, margin_r=6, margin_v=7, layer=2, comment=False)
    assert sorted(margins["changed"]) == ["comment", "layer", "margin_l", "margin_r", "margin_v"]
    assert margins["line"]["kind"] == "Dialogue"
    assert margins["line"]["margin_l"] == "5"

    with pytest.raises(ToolError):
        L.ass_update_line(0)
    with pytest.raises(ToolError):
        L.ass_update_line(0, start_ms=9000, end_ms=1000)
    with pytest.raises(ToolError):
        L.ass_update_line(50, text="gone")


def test_update_line_keeps_end_when_start_moves_past_it() -> None:
    L.ass_new_document(doc_id="clamp")
    L.ass_add_line(1000, 2000, "x")
    with pytest.raises(ToolError):
        L.ass_update_line(0, start_ms=3000)
    assert L.ass_get_line(0)["start_ms"] == 1000


def test_update_lines_applies_to_selection(three_lines) -> None:
    L.ass_select([0, 1])
    result = L.ass_update_lines("selection", style="Hers", actor="Ann")
    assert result["changed"] == [0, 1]
    assert result["count"] == 2
    assert [line["style"] for line in result["lines"]] == ["Hers", "Hers"]
    assert L.ass_get_line(2)["style"] == "Default"

    filtered = L.ass_update_lines({"kind": "comment"}, effect="dry")
    assert filtered["changed"] == [2]
    assert L.ass_get_line(2)["effect"] == "dry"


def test_update_lines_pad_ms_extends_both_ends(three_lines) -> None:
    result = L.ass_update_lines([1], pad_ms=250)
    line = result["lines"][0]
    assert line["start_ms"] == 2750 and line["end_ms"] == 4250
    assert result["changed"] == [1]

    trimmed = L.ass_update_lines([1], pad_ms=-250)
    line = trimmed["lines"][0]
    assert line["start_ms"] == 3000 and line["end_ms"] == 4000

    first = L.ass_update_lines([0], pad_ms=5000)
    assert first["lines"][0]["start_ms"] == 0  # clamped into the timebase


def test_update_lines_clamp_to_avoid_overlap() -> None:
    L.ass_new_document(doc_id="overlap")
    L.ass_add_line(1000, 4000, "A")
    L.ass_add_line(3000, 6000, "B")
    result = L.ass_update_lines([0], text="A2", clamp_to_avoid_overlap=True)
    assert result["clamped"] == [0]
    lines = L.ass_list_lines()["lines"]
    assert lines[0]["end_ms"] == 3000          # pulled back to B's start
    assert lines[0]["start_ms"] == 1000
    assert lines[1]["start_ms"] == 3000

    L.ass_new_document(doc_id="overlap2")
    L.ass_add_line(1000, 4000, "A")
    L.ass_add_line(3000, 6000, "B")
    both = L.ass_update_lines([0, 1], clamp_to_avoid_overlap=True)
    # only line 0 needed shrinking: line 1 already starts at 3000, which is the
    # end line 0 was pulled back to
    assert both["clamped"] == [0]
    lines = L.ass_list_lines()["lines"]
    assert lines[1]["start_ms"] >= lines[0]["end_ms"]
    assert lines[0]["start_ms"] == 1000 and lines[1]["end_ms"] == 6000
    assert lines[0]["end_ms"] == 3000


def test_update_lines_requires_something_to_do(three_lines) -> None:
    with pytest.raises(ToolError):
        L.ass_update_lines([0])
    with pytest.raises(ToolError):
        L.ass_update_lines([])


# --------------------------------------------------------------------------- #
# structural edits
# --------------------------------------------------------------------------- #


def test_delete_lines_removes_selection(three_lines) -> None:
    result = L.ass_delete_lines([1, 2])
    assert result["deleted"] == [1, 2]
    assert result["count"] == 2
    assert result["remaining"] == 2
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == ["first", "fourth"]

    L.ass_select([0])
    by_selection = L.ass_delete_lines("selection")
    assert by_selection["deleted"] == [0]
    assert L.ass_list_lines()["total"] == 1


def test_duplicate_lines_offset_and_position() -> None:
    L.ass_new_document(doc_id="dup")
    L.ass_add_line(1000, 2000, "one", actor="A")
    L.ass_add_line(3000, 4000, "two")

    result = L.ass_duplicate_lines([0], offset_ms=10_000)
    assert result["indices"] == [1]
    lines = L.ass_list_lines()["lines"]
    assert [line["text"] for line in lines] == ["one", "one", "two"]
    assert lines[1]["start_ms"] == 11_000 and lines[1]["end_ms"] == 12_000
    assert lines[1]["actor"] == "A"

    before = L.ass_duplicate_lines([0], insert_after=False)
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == [
        "one", "one", "one", "two",
    ]
    # insert_after=False puts the copy right where its source was, i.e. 0-based 0
    assert before["indices"] == [0]
    assert_json(result)


def test_move_lines_reorders() -> None:
    L.ass_new_document(doc_id="move")
    for text in ("A", "B", "C"):
        L.ass_add_line(1000, 2000, text)

    result = L.ass_move_lines([0], 2)
    assert result["moved"] == [0]
    assert result["count"] == 1
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == ["B", "C", "A"]
    assert result["selection"] == [2]

    L.ass_select([2])
    moved = L.ass_move_lines("selection", 0)
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == ["A", "B", "C"]
    assert moved["selection"] == [0]

    with pytest.raises(ToolError):
        L.ass_move_lines([], 0)


def test_split_line_preserves_prefix_and_style() -> None:
    open_basic()
    result = L.ass_split_line(0, 2500)
    assert result["index"] == 0
    assert result["second_index"] == 1
    assert result["at_ms"] == 2500
    assert result["first"]["end_ms"] == 2500
    assert result["second"]["start_ms"] == 2500
    assert result["second"]["end_ms"] == 4000
    # the cut lands mid-word ("Hello world" is 11 visible chars, 6 kept): the
    # first half keeps the override block that follows it, the second half
    # re-emits the active state so it renders identically on its own
    assert result["first"]["text"] == r"Hello {\i1}"
    assert result["second"]["text"] == r"{\i1}world{\i0}"
    assert result["second"]["text"].startswith(r"{\i1}")
    assert result["second"]["style"] == "Default"
    assert result["second"]["kind"] == "Dialogue"
    assert L.ass_list_lines()["total"] == 3
    assert [line["plain_text"] for line in L.ass_list_lines()["lines"]] == [
        "Hello ", "world", "Preserve me",
    ]

    # splitting the derived karaoke-free drawing line keeps the {\p1} prefix
    L.ass_new_document(doc_id="drawsplit")
    L.ass_add_line(0, 1000, r"{\p1}m 0 0 l 100 0{\p0}")
    split = L.ass_split_line(0, "0:00:00.50")
    assert split["first"]["text"].startswith(r"{\p1}")
    assert split["second"]["text"].startswith(r"{\p1}")


def test_split_line_outside_span_raises() -> None:
    open_basic()
    for bad in (1000, 4000, 0, "0:00:10.00"):
        with pytest.raises(ToolError) as excinfo:
            L.ass_split_line(0, bad)
        assert "outside the line span" in str(excinfo.value)
    assert L.ass_list_lines()["total"] == 2


def test_merge_lines_joins_in_time_order() -> None:
    L.ass_new_document(doc_id="merge")
    L.ass_add_line(5000, 6000, "later", style="Extra", actor="Z")
    L.ass_add_line(1000, 2000, "earlier", style="Base", actor="Y")

    result = L.ass_merge_lines([0, 1], separator=chr(92) + "N")
    # the merged line is the earliest one (actor Y / style Base); the later
    # source line is dropped and the survivors shift up to 0
    assert result["index"] == 0
    assert result["removed"] == [0]
    assert result["removed_count"] == 1
    assert result["start_ms"] == 1000 and result["end_ms"] == 6000
    assert result["text"] == "earlier" + chr(92) + "N" + "later"
    assert result["line"]["style"] == "Base"
    assert result["line"]["actor"] == "Y"
    lines = L.ass_list_lines()["lines"]
    assert len(lines) == 1
    assert lines[0]["start_ms"] == 1000 and lines[0]["end_ms"] == 6000

    with pytest.raises(ToolError):
        L.ass_merge_lines([0])


# --------------------------------------------------------------------------- #
# find / replace
# --------------------------------------------------------------------------- #


def test_find_replace_literal_counts_and_changes() -> None:
    open_basic()
    result = L.ass_find_replace("world", "planet")
    assert result["total"] == 1
    assert result["dry_run"] is False
    assert result["changed"] == [0]
    assert result["lines"][0]["counts"] == {"text": 1}
    assert result["lines"][0]["count"] == 1
    assert text_of(0) == r"Hello {\i1}planet{\i0}"
    assert text_of(1) == "Preserve me"
    assert_json(result)


def test_find_replace_dry_run_changes_nothing() -> None:
    open_basic()
    result = L.ass_find_replace("world", "planet", dry_run=True)
    assert result["total"] == 1
    assert result["changed"] == []
    assert result["dry_run"] is True
    assert result["lines"][0]["text"] == r"Hello {\i1}world{\i0}"
    assert text_of(0) == r"Hello {\i1}world{\i0}"


def test_find_replace_regex_and_case_sensitivity() -> None:
    open_basic()
    result = L.ass_find_replace(r"\\i(\d)", r"\\b\1", regex=True)
    assert result["total"] == 2
    assert text_of(0) == r"Hello {\b1}world{\b0}"

    insensitive = L.ass_find_replace("HELLO", "hi", regex=False, case_sensitive=False)
    assert insensitive["total"] == 1
    assert text_of(0).startswith("hi")

    strict = L.ass_find_replace("HELLO", "hi", case_sensitive=True)
    assert strict["total"] == 0

    with pytest.raises(ToolError):
        L.ass_find_replace("(", "x", regex=True)
    with pytest.raises(ToolError):
        L.ass_find_replace("", "x")


def test_find_replace_fields_limit_and_selection() -> None:
    open_basic()
    styles = L.ass_find_replace("Default", "Main", fields=["style"])
    assert styles["total"] == 2
    assert L.ass_get_line(0)["style"] == "Main"
    assert L.ass_get_line(1)["style"] == "Main"
    assert text_of(0) == r"Hello {\i1}world{\i0}"

    scoped = L.ass_find_replace("Main", "Other", fields=["style"], selection=[1])
    assert scoped["total"] == 1
    assert L.ass_get_line(1)["style"] == "Other"
    assert L.ass_get_line(0)["style"] == "Main"

    unlimited = L.ass_find_replace("r", "R", fields=["text"], selection=[1])
    assert unlimited["total"] == 2                  # "Preserve me" has two 'r's
    assert text_of(1) == "PReseRve me"
    limited = L.ass_find_replace("e", "E", fields=["text"], selection=[1], limit=1)
    assert limited["total"] == 1                    # only the first 'e' per line
    assert text_of(1) == "PREseRve me"

    with pytest.raises(ToolError):
        L.ass_find_replace("a", "b", fields=["nope"])
    with pytest.raises(ToolError):
        L.ass_find_replace("a", "b", limit=-1)


# --------------------------------------------------------------------------- #
# comments
# --------------------------------------------------------------------------- #


def test_set_comment_toggles_kind() -> None:
    open_basic()
    result = L.ass_set_comment([0])
    assert result["changed"] == [0]
    assert result["comment"] is True and result["dropped"] is False
    assert L.ass_get_line(0)["kind"] == "Comment"
    assert L.ass_get_line(1)["kind"] == "Comment"

    back = L.ass_set_comment([0], comment=False)
    assert back["changed"] == [0]
    assert L.ass_get_line(0)["kind"] == "Dialogue"

    dropped = L.ass_set_comment([1], drop=True)
    assert dropped["dropped"] is True
    assert dropped["changed"] == [1]
    assert dropped["count"] == 1
    assert L.ass_list_lines()["total"] == 1
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == [r"Hello {\i1}world{\i0}"]

    with pytest.raises(ToolError):
        L.ass_set_comment([])


# --------------------------------------------------------------------------- #
# sorting
# --------------------------------------------------------------------------- #


def test_sort_lines_by_time_and_keys() -> None:
    L.ass_new_document(doc_id="sort")
    L.ass_add_line(9000, 9500, "last")
    L.ass_add_line(1000, 2000, "first")
    L.ass_add_line(1000, 1500, "tie-early-end")
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == [
        "last", "first", "tie-early-end",
    ]

    result = L.ass_sort_lines()
    assert result["sorted"] is True
    assert result["count"] == 3
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == [
        "tie-early-end", "first", "last",
    ]

    reverse = L.ass_sort_lines(keys=["start", "end"], reverse=True)
    assert reverse["count"] == 3
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == [
        "last", "first", "tie-early-end",
    ]

    by_text = L.ass_sort_lines(keys=["text"])
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == [
        "first", "last", "tie-early-end",
    ]
    assert_json(result)

    with pytest.raises(ToolError):
        L.ass_sort_lines(keys=["nonsense"])


# --------------------------------------------------------------------------- #
# undo / redo
# --------------------------------------------------------------------------- #


def test_undo_and_redo_restore_state() -> None:
    L.ass_new_document(doc_id="undo")
    L.ass_add_line(1000, 2000, "one")
    L.ass_add_line(3000, 4000, "two")

    history = L.ass_undo_history()
    assert history["undo_depth"] == 2
    assert history["redo_depth"] == 0
    assert history["labels"] == ["undo step 1", "undo step 2"]
    assert history["next_undo"] == "undo step 2"
    assert history["next_redo"] is None
    assert history["current_lines"] == 2
    assert_json(history)

    undone = L.ass_undo()
    assert undone["undone"] is True
    assert undone["undo_depth"] == 1 and undone["redo_depth"] == 1
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == ["one"]

    pending = L.ass_undo_history()
    assert pending["redo_labels"] == ["redo step 1"]
    assert pending["next_redo"] == "redo step 1"

    redone = L.ass_redo()
    assert redone["redone"] is True
    assert redone["redo_depth"] == 0
    assert [line["text"] for line in L.ass_list_lines()["lines"]] == ["one", "two"]
    # the redo stack is consumed by the redo above
    assert L.ass_undo_history()["redo_labels"] == []

    # nothing left to undo
    L.ass_undo()
    L.ass_undo()
    exhausted = L.ass_undo()
    assert exhausted["undone"] is False
    assert exhausted["undo_depth"] == 0
    assert L.ass_list_lines()["total"] == 0


def test_undo_restores_text_edits() -> None:
    open_basic()
    original = text_of(0)
    L.ass_update_line(0, text="changed")
    assert text_of(0) == "changed"
    L.ass_undo()
    assert text_of(0) == original


# --------------------------------------------------------------------------- #
# export / import
# --------------------------------------------------------------------------- #


def test_export_text_formats(tmp_path: pathlib.Path) -> None:
    workspace.set_output_dir(tmp_path / "out")
    open_basic()
    L.ass_add_line(7000, 7500, r"line one\Nline two")

    txt = L.ass_export_text()
    assert txt["format"] == "txt"
    assert pathlib.Path(txt["path"]).parent == tmp_path / "out"
    assert pathlib.Path(txt["path"]).exists()
    assert txt["text"] == "Hello world\nline one" + chr(92) + "Nline two\n"  # comment skipped
    assert txt["bytes"] == len(txt["text"].encode("utf-8"))
    assert txt["selection"] == [0, 1, 2]
    assert_json(txt)

    with_sep = L.ass_export_text(selection=[2], line_separator="|")
    assert with_sep["text"] == "line one|line two\n"

    real_break = L.ass_export_text(selection=[2], line_separator="\n")
    assert real_break["text"] == "line one\nline two\n"

    tsv = L.ass_export_text(format="tsv", selection=[0, 1])
    rows = tsv["text"].strip().split("\n")
    assert rows[0] == "start\tend\tstyle\tactor\teffect\tkind\ttext"
    assert rows[1].split("\t")[:3] == ["0:00:01.00", "0:00:04.00", "Default"]
    assert rows[1].split("\t")[5] == "Dialogue"
    assert rows[2].split("\t")[5] == "Comment"

    srt = L.ass_export_text(format="srt")
    assert srt["text"] == (
        "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n"
        "2\n00:00:07,000 --> 00:00:07,500\nline one\nline two\n"
    )

    fragment = L.ass_export_text(format="ass-fragment", selection=[0])
    assert fragment["text"].startswith("Dialogue: 0,0:00:01.00,0:00:04.00,Default,")
    assert fragment["text"].endswith(r"Hello {\i1}world{\i0}" + "\n")
    assert fragment["bytes"] == len(fragment["text"].encode("utf-8"))

    explicit = L.ass_export_text(format="srt", output_path=str(tmp_path / "explicit.srt"))
    assert pathlib.Path(explicit["path"]) == tmp_path / "explicit.srt"
    assert (tmp_path / "explicit.srt").read_text(encoding="utf-8") == explicit["text"]

    with pytest.raises(ToolError):
        L.ass_export_text(format="pdf")


def test_import_srt_parses_real_world_variants(tmp_path: pathlib.Path) -> None:
    srt = tmp_path / "in.srt"
    srt.write_bytes(
        b"1\r\n"
        b"00:00:01,000 --> 00:00:02.500\r\n"
        b"Hello\r\n"
        b"world\r\n"
        b"\r\n"
        b"2\r\n"
        b"00:00:03.000 --> 00:00:04,250\r\n"
        b"Dot decimal separator\r\n"
        b"\r\n"
        b"3\r\n"
        b"00:00:05.25 --> 00:00:06.75\r\n"
        b"<i>tags</i> kept literally\r\n"
    )
    L.ass_new_document(doc_id="srt")
    result = L.ass_import_srt(str(srt))
    assert result["count"] == 3
    assert result["indices"] == [0, 1, 2]
    lines = L.ass_list_lines()["lines"]
    assert lines[0]["text"] == r"Hello\Nworld"
    assert lines[0]["start_ms"] == 1000 and lines[0]["end_ms"] == 2500
    assert lines[0]["style"] == "Default"
    assert lines[1]["text"] == "Dot decimal separator"
    assert lines[1]["start_ms"] == 3000 and lines[1]["end_ms"] == 4250
    assert lines[2]["text"] == "<i>tags</i> kept literally"
    assert result["style"] == "Default"
    assert_json(result)

    offset = L.ass_import_srt(str(srt), style="Srt", offset_ms=500)
    assert offset["count"] == 3
    moved = L.ass_list_lines()["lines"][3:]
    assert moved[0]["start_ms"] == 1500 and moved[0]["end_ms"] == 3000   # 2500 + 500
    assert moved[0]["style"] == "Srt"
    assert moved[2]["start_ms"] == 5750                                  # 5250 + 500

    with pytest.raises(ToolError):
        L.ass_import_srt(str(tmp_path / "missing.srt"))

    empty = tmp_path / "empty.srt"
    empty.write_text("no timestamps here\n")
    with pytest.raises(ToolError):
        L.ass_import_srt(str(empty))


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def test_stats_reports_counts_and_histograms() -> None:
    open_basic()
    stats = L.ass_stats()
    assert stats["doc_id"] == "basic.ass"
    assert stats["lines"] == 2
    assert stats["dialogue"] == 1
    assert stats["comments"] == 1
    assert stats["characters"] == 11
    assert stats["words"] == 2
    assert stats["total_duration_ms"] == 3000
    assert stats["duration_ms"] == 3000
    assert stats["span_ms"] == 5000
    assert stats["average_cps"] == pytest.approx(11 / 3, abs=0.01)
    assert stats["min_line_cps"] == stats["max_line_cps"] == stats["mean_line_cps"]
    assert stats["style_histogram"] == {"Default": 2}
    assert stats["actors"] == []
    assert stats["over_cps_25"] == 0
    assert stats["slowest"]["index"] == 0
    assert_json(stats)


def test_stats_histogram_and_actors() -> None:
    L.ass_new_document(doc_id="stats")
    L.ass_add_line(0, 1000, "one", style="A", actor="Ann")
    L.ass_add_line(1000, 2000, "two", style="A", actor="Bob")
    L.ass_add_line(2000, 3000, "three", style="B", actor="Ann")
    L.ass_add_line(3000, 4000, "commented", comment=True, actor="Ann")

    stats = L.ass_stats()
    assert stats["lines"] == 4
    assert stats["dialogue"] == 3
    assert stats["comments"] == 1
    # both histograms count every line, comments included ("commented" keeps the
    # default style and its actor); the per-line cps figures ignore comments
    assert stats["style_histogram"] == {"A": 2, "B": 1, "Default": 1}
    assert stats["actors"] == ["Ann", "Bob"]
    assert stats["actor_histogram"] == {"Ann": 3, "Bob": 1}
    assert stats["total_duration_ms"] == 3000


# --------------------------------------------------------------------------- #
# cross-cutting behaviour
# --------------------------------------------------------------------------- #


def test_every_tool_result_is_json_serialisable() -> None:
    open_basic()
    results = [
        L.ass_list_documents(),
        L.ass_document_info(),
        L.ass_select([0]),
        L.ass_get_selection(),
        L.ass_list_lines(include_tags_summary=True),
        L.ass_get_line(0),
        L.ass_add_line(2000, 3000, "x"),
        L.ass_add_lines([{"start_ms": 4000, "end_ms": 5000, "text": "y"}]),
        L.ass_update_line(0, text="z"),
        L.ass_update_lines([0], pad_ms=10),
        L.ass_duplicate_lines([0]),
        L.ass_move_lines([0], 0),
        L.ass_split_line(0, 2500),
        L.ass_merge_lines([0, 1]),
        L.ass_find_replace("z", "q", dry_run=True),
        L.ass_set_comment([0]),
        L.ass_sort_lines(),
        L.ass_undo(),
        L.ass_redo(),
        L.ass_undo_history(),
        L.ass_delete_lines([0]),
        L.ass_export_text(format="txt"),
        L.ass_stats(),
    ]
    for result in results:
        assert isinstance(result, dict)
        assert json.loads(json.dumps(result)) == json.loads(json.dumps(result))
    assert json.dumps(results, allow_nan=False)


def test_tools_never_leave_a_raw_exception() -> None:
    """Every failure path surfaces as ToolError, not a library exception."""
    open_basic()
    calls = [
        lambda: L.ass_open("/definitely/missing.ass"),
        lambda: L.ass_get_line(1_000),
        lambda: L.ass_add_line(100, 50, "x"),
        lambda: L.ass_update_line(0),
        lambda: L.ass_split_line(0, 123_456),
        lambda: L.ass_find_replace("(", "x", regex=True),
        lambda: L.ass_export_text(format="nope"),
        lambda: L.ass_import_srt("/definitely/missing.srt"),
        lambda: L.ass_select(500),
        lambda: L.ass_close("ghost"),
        lambda: L.ass_select_document("ghost"),
        lambda: L.ass_sort_lines(keys=["nope"]),
    ]
    for call in calls:
        with pytest.raises(ToolError):
            call()


def test_mutations_are_snapshot_backed() -> None:
    """Each documented mutation adds exactly one undo step."""
    L.ass_new_document(doc_id="snap")
    assert L.ass_undo_history()["undo_depth"] == 0
    L.ass_add_line(1000, 2000, "a")
    assert L.ass_undo_history()["undo_depth"] == 1
    L.ass_update_line(0, text="b")
    assert L.ass_undo_history()["undo_depth"] == 2
    L.ass_set_comment([0])
    assert L.ass_undo_history()["undo_depth"] == 3
    L.ass_duplicate_lines([0])
    assert L.ass_undo_history()["undo_depth"] == 4
    L.ass_undo()
    L.ass_undo()
    L.ass_undo()
    # four mutations took four snapshots; three undos leave the first one pending
    assert L.ass_undo_history()["undo_depth"] == 1
    assert L.ass_list_lines()["total"] == 1
    assert text_of(0) == "a"

    L.ass_undo()
    assert L.ass_undo_history()["undo_depth"] == 0
    assert L.ass_list_lines()["total"] == 0


def test_ssa_documents_are_editable() -> None:
    """SSA uses ``Marked`` instead of ``Layer``; edits must not corrupt it."""
    L.ass_open(str(SSA))
    before = L.ass_get_line(0)
    # this fixture stores SSA's odd ``Marked=0`` value verbatim; the tool layer
    # exposes it through the ``layer`` key without rewriting it
    assert before["layer"] == "Marked=0"
    assert list(before["fields"])[0] == "Marked"
    added = L.ass_add_line(5000, 6000, "ssa line", comment=True)
    line = added["line"]
    assert line["kind"] == "Comment"
    assert line["start_ms"] == 5000
    full = L.ass_get_line(added["index"])
    assert list(full["fields"]) == ["Marked", "Start", "End", "Style", "Name",
                                    "MarginL", "MarginR", "MarginV", "Effect", "Text"]

    destination = pathlib.Path("/tmp") / "aegisub-mcp-ssa-check.ssa"
    saved = L.ass_save(path=str(destination))
    written = destination.read_text(encoding="utf-8")
    assert "Marked=0,0:00:05.00,0:00:06.00" in written
    assert "Layer:" not in written
    assert saved["bytes_written"] == len(destination.read_bytes())
    destination.unlink()


def test_bom_crlf_fixture_without_final_newline_round_trips(tmp_path: pathlib.Path) -> None:
    path = FIX / "bom-crlf-no-final.ssa"
    opened = L.ass_open(str(path))
    assert opened["has_bom"] is True
    assert opened["newline"] == repr("\r\n")
    dest = tmp_path / path.name
    saved = L.ass_save(path=str(dest))
    assert dest.read_bytes() == path.read_bytes()
    assert not dest.read_bytes().endswith(b"\n")
    assert saved["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
