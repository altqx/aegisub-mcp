"""Round-trip fidelity + document model tests against real-world fixtures."""

from __future__ import annotations

import pathlib

import pytest

from aegisub_mcp.asscore import document as D

FIX = pathlib.Path(__file__).parent / "fixtures" / "real"
FIXTURES = sorted(FIX.glob("*.ass")) + sorted(FIX.glob("*.ssa"))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_byte_exact_round_trip(path: pathlib.Path) -> None:
    """Loading and saving without edits must reproduce the file byte for byte."""
    original = path.read_bytes()
    doc = D.AssDocument.load(str(path))
    assert doc.to_bytes() == original, f"{path.name}: round-trip changed bytes"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_round_trip_is_idempotent(path: pathlib.Path) -> None:
    doc = D.AssDocument.load(str(path))
    once = doc.to_bytes()
    assert D.AssDocument.from_bytes(once).to_bytes() == once


def test_fixture_count_is_meaningful() -> None:
    assert len(FIXTURES) >= 10, "expected the full real-world fixture set"


def test_bom_crlf_no_final_newline_preserved() -> None:
    p = FIX / "bom-crlf-no-final.ssa"
    doc = D.AssDocument.load(str(p))
    assert doc.encoding == "utf-8"
    assert doc.newline == "\r\n"
    assert doc.trailing_newline is False
    assert doc.has_bom is True
    assert doc.to_bytes() == p.read_bytes()


def test_events_and_fields() -> None:
    doc = D.AssDocument.load(str(FIX / "basic.ass"))
    events = doc.events()
    assert events, "expected dialogue lines"
    ev = events[0]
    assert ev.get("Layer").strip() == "0"
    assert 0 <= ev.start_ms < ev.end_ms
    assert ev.duration_ms == ev.end_ms - ev.start_ms
    assert ev.get("Style")
    assert isinstance(ev.text, str)


def test_add_edit_remove_event_keeps_others_intact() -> None:
    doc = D.AssDocument.load(str(FIX / "basic.ass"))
    before = doc.to_bytes()
    doc2 = D.AssDocument.from_bytes(before)
    n = len(doc2.events())
    entry = doc2.add_event(kind="Dialogue", start=1000, end=2000, text="hello")
    events = doc2.events()
    assert len(events) == n + 1
    idx = next(i for i, e in enumerate(events) if e is entry)
    assert events[idx].text == "hello"
    assert events[idx].start_ms == 1000 and events[idx].end_ms == 2000
    assert doc2.remove_events([idx]) == 1
    assert len(doc2.events()) == n
    assert doc2.to_bytes() == before, "add+remove must restore original bytes"


def test_events_are_sorted_by_start() -> None:
    doc = D.AssDocument.load(str(FIX / "basic.ass"))
    starts = [e.start_ms for e in doc.sorted_events("start")]
    assert starts == sorted(starts)


def test_info_set_and_play_res() -> None:
    doc = D.AssDocument.load(str(FIX / "basic.ass"))
    x, y = doc.play_res
    assert x > 0 and y > 0
    doc.set_play_res(1280, 720)
    assert doc.play_res == (1280, 720)
    doc.info_set("Title", "changed")
    assert "Title: changed" in doc.to_text()


def test_info_round_trip_preserves_other_keys() -> None:
    doc = D.AssDocument.load(str(FIX / "basic.ass"))
    before = doc.info_all()
    doc.info_set("PlayResY", 1080)
    after = doc.info_all()
    for key, value in before.items():
        if key != "PlayResY":
            assert after.get(key) == value


def test_unknown_sections_and_legacy_attachments_preserved() -> None:
    doc = D.AssDocument.load(str(FIX / "legacy-attachments-real.ass"))
    assert doc.to_bytes() == (FIX / "legacy-attachments-real.ass").read_bytes()
    kinds = {s.kind for s in doc.sections}
    assert "fonts" in kinds or "graphics" in kinds


def test_malformed_lines_are_kept_verbatim() -> None:
    for name in ("recoverable.ass", "malformed-unknown.ssa", "unknown-fields.ass"):
        p = FIX / name
        doc = D.AssDocument.load(str(p))
        assert doc.to_bytes() == p.read_bytes(), name


def test_colon_tight_dialogue_is_an_event_like_aegisub() -> None:
    """Aegisub strips exactly ``Dialogue:``/``Comment:`` then trims each token
    (``src/ass_parser.cpp`` + ``src/ass_dialogue.cpp``), so a missing space after
    the colon must still yield a real event, not a preserved raw line."""
    p = FIX / "nospace-dialogue.ass"
    doc = D.AssDocument.load(str(p))
    events = doc.events()
    assert [e.kind for e in events] == ["Dialogue", "Comment"]
    assert [e.get("Layer") for e in events] == ["0", "0"]
    assert [e.get("Start") for e in events] == ["0:00:01.00", "0:00:03.00"]
    assert [e.get("Style") for e in events] == ["Default", "Alt"]
    assert [e.text for e in events] == ["tight line", "tight comment"]
    assert events[1].is_comment is True
    # the unknown section after [Events] must survive untouched
    assert doc.to_bytes() == p.read_bytes()


def test_redefined_formats_uses_per_entry_order() -> None:
    doc = D.AssDocument.load(str(FIX / "redefined-formats.ssa"))
    orders = [tuple(e.order) for e in doc.events()]
    assert len(set(orders)) > 1, "fixture should have more than one event format"
    for ev in doc.events():
        assert set(ev.fields_dict()) == set(ev.order)


def test_custom_and_fallback_formats_parse() -> None:
    for name in ("custom-format.ssa", "fallback-formats.ssa"):
        doc = D.AssDocument.load(str(FIX / name))
        assert doc.events(), name


def test_style_lookup_and_crud() -> None:
    doc = D.AssDocument.load(str(FIX / "basic.ass"))
    styles = doc.styles()
    assert styles, "expected styles"
    name = styles[0].name
    assert doc.get_style(name) is not None
    doc.add_style({"Name": "Extra", "Fontname": "Arial", "Fontsize": "40"})
    assert doc.get_style("Extra").get("Fontname") == "Arial"
    assert doc.get_style("extra") is not None, "style lookup is case-insensitive"
    assert doc.remove_style("Extra") is True
    assert doc.get_style("Extra") is None


def test_stats_reports_counts() -> None:
    doc = D.AssDocument.load(str(FIX / "basic.ass"))
    st = doc.stats()
    assert st["events_total"] == len(doc.events())
    assert st["styles"] == len(doc.styles())
    assert st["play_res"][0] > 0
    assert st["dialogue"] <= st["events_total"]


def test_new_document_is_valid_ass() -> None:
    doc = D.AssDocument.new(play_res=(1280, 720), title="brand new")
    doc.add_event(kind="Dialogue", start=0, end=2000, text="hello")
    text = doc.to_text()
    assert "[Script Info]" in text and "[Events]" in text
    assert "PlayResX: 1280" in text
    reloaded = D.AssDocument.from_text(text)
    assert reloaded.events()[0].text == "hello"


def test_snapshot_restore() -> None:
    doc = D.AssDocument.load(str(FIX / "basic.ass"))
    snap = doc.snapshot()
    doc.info_set("Title", "temp")
    doc.restore(snap)
    assert doc.to_bytes() == (FIX / "basic.ass").read_bytes()
