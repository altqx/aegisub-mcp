"""The Aegisub-side bridge script, executed by the real Automation 4 host.

These tests render ``assets/krapau-bridge.lua.in`` exactly as
``aegisub-mcp-bridge install`` does, load it into the engine the way Aegisub
loads an Autoload script, publish revisions from the Python side and then drive
the macros -- i.e. the whole loop is exercised without a GUI.  What they cannot
prove is Aegisub's *own* behaviour (when it calls the validation callback, what
its autosave writes); that is what the on-host Aegisub run is for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegisub_mcp.asscore.document import AssDocument
from aegisub_mcp.bridge import Bridge, parse_applied, parse_state
from aegisub_mcp.lua.engine import LuaEngine

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "real" / "basic.ass"

PULL = "krapau-bridge: Pull changes from MCP"
PUSH = "krapau-bridge: Push my document to MCP"


@pytest.fixture()
def installed(tmp_path: Path) -> dict:
    """A rendered bridge script plus its bridge directory.

    ``Bridge.ensure()`` mirrors real usage: the MCP side creates the directory
    (``ass_bridge_publish``/``install`` do) before Aegisub ever writes to it.
    The script itself also creates it on demand -- see
    ``test_script_creates_a_missing_bridge_directory``.
    """
    bridge_dir = tmp_path / "bridge"
    bridge = Bridge(bridge_dir)
    bridge.ensure()
    script = tmp_path / "krapau-bridge.lua"
    script.write_text(bridge.render_script(bridge_dir=bridge_dir,
                                          autoload_dir=tmp_path / "autoload",
                                          hotkey="Ctrl-Alt-M"),
                      encoding="utf-8")
    return {"bridge": bridge, "script": script, "dir": bridge_dir}


def load(script: Path, path: Path = FIXTURE) -> tuple[LuaEngine, AssDocument]:
    doc = AssDocument.load(str(path))
    engine = LuaEngine(doc)
    engine.load_file(script)
    return engine, doc


def event_state(doc: AssDocument) -> list[tuple[bool, int, int, str, str]]:
    """(is_comment, start_ms, end_ms, style, text) for every event line."""
    return [(e.is_comment, e.start_ms, e.end_ms, e.get("Style"), e.text)
            for e in doc.events()]


class TestRendering:
    def test_placeholders_are_substituted(self, installed):
        text = installed["script"].read_text(encoding="utf-8")
        assert "@BRIDGE_DIR@" not in text
        assert str(installed["dir"]) in text
        assert "local AUTO_PULL     = true" in text
        assert "krapau-bridge/1" in text

    def test_registers_three_macros(self, installed):
        engine, _doc = load(installed["script"])
        names = [item.name for item in engine.macros]
        assert names == [PULL, PUSH, "krapau-bridge: Bridge status"]
        assert all(item.description for item in engine.macros)

    def test_loads_without_a_document(self, installed):
        """Autoload scripts run at startup, when no subtitles are open."""
        script = installed["script"].read_text(encoding="utf-8")
        engine = LuaEngine(AssDocument.new(play_res=(1920, 1080)))
        engine.load_script(script, name=str(installed["script"]))
        assert len(engine.macros) == 3


class TestPull:
    def test_patch_path_updates_the_open_document(self, installed):
        source = AssDocument.load(str(FIXTURE))
        dialogue = [e for e in source.events() if not e.is_comment][0]
        dialogue.set("Text", "PUBLISHED by MCP")
        dialogue.set("Start", "0:00:02.50")
        dialogue.set("End", "0:00:07.00")
        published = installed["bridge"].publish(source, note="patch rev", doc_id="basic.ass")
        assert published["rev"] == 1

        engine, doc = load(installed["script"])
        result = engine.run_macro(PULL)

        assert result.ok, result.error
        assert "pull ok" in result.log
        assert result.undo_points == ["krapau-bridge: pull rev 1 -- patch rev"]
        assert event_state(doc) == [
            (False, 2500, 7000, "Default", "PUBLISHED by MCP"),
            (True, 5000, 6000, "Default", "Preserve me"),
        ]
        applied = parse_applied(installed["bridge"].applied_path)
        assert applied["rev"] == 1 and applied["result"] == "ok"
        assert applied["script_name"] == "krapau-bridge"
        assert applied["lines"] == 13

    def test_second_pull_is_a_no_op(self, installed):
        source = AssDocument.load(str(FIXTURE))
        installed["bridge"].publish(source, doc_id="basic.ass")
        engine, doc = load(installed["script"])
        assert engine.run_macro(PULL).ok
        before = doc.to_text()

        again = engine.run_macro(PULL)

        # Nothing is pending, so the validation callback reports false and the
        # engine refuses the run -- the same thing a greyed-out menu entry means.
        assert not again.ok
        assert "not applicable" in (again.error or "")
        assert doc.to_text() == before

    def test_append_path_adds_missing_lines(self, installed, tmp_path: Path):
        target = tmp_path / "ae.ass"
        doc = AssDocument.new(play_res=(1280, 720))
        doc.add_event(kind="Dialogue", start_ms=1000, end_ms=2000, text="first")
        doc.save(str(target))

        source = AssDocument.load(str(target))
        source.add_event(kind="Dialogue", start_ms=3000, end_ms=4000, text="second")
        installed["bridge"].publish(source, doc_id="ae.ass")

        engine, ae_doc = load(installed["script"], target)
        result = engine.run_macro(PULL)

        assert result.ok, result.error
        assert result.structure_changed
        assert [e.text for e in ae_doc.events()] == ["first", "second"]
        assert event_state(ae_doc)[-1][1:4] == (3000, 4000, "Default")

    def test_delete_path_drops_surplus_lines(self, installed, tmp_path: Path):
        target = tmp_path / "ae.ass"
        doc = AssDocument.new(play_res=(1280, 720))
        doc.add_event(kind="Dialogue", start_ms=1000, end_ms=2000, text="keep")
        doc.add_event(kind="Comment", start_ms=3000, end_ms=4000, text="drop me")
        doc.save(str(target))

        source = AssDocument.load(str(target))
        # Drop the Comment event by locating it in file-line space (all_lines is
        # what remove_lines indexes, and it counts heading/format lines too).
        drop = [index for index, (_section, entry) in enumerate(source.all_lines())
                if getattr(entry, "is_comment", False)][0]
        source.remove_lines([drop])
        assert [e.is_comment for e in source.events()] == [False]
        installed["bridge"].publish(source, doc_id="ae.ass")

        engine, ae_doc = load(installed["script"], target)
        result = engine.run_macro(PULL)

        assert result.ok, result.error
        assert result.structure_changed
        assert event_state(ae_doc) == [(False, 1000, 2000, "Default", "keep")]

    def test_pull_without_a_request_file_is_reported(self, installed):
        engine, doc = load(installed["script"])
        before = doc.to_text()
        result = engine.run_macro(PULL)
        assert not result.ok
        assert doc.to_text() == before

    def test_auto_pull_off_leaves_the_work_to_the_macro_body(self, installed):
        installed["bridge"].publish(AssDocument.load(str(FIXTURE)), doc_id="basic.ass")
        script = installed["script"]
        script.write_text(installed["bridge"].render_script(
            bridge_dir=installed["dir"], autoload_dir=installed["dir"] / "autoload",
            auto_pull=False), encoding="utf-8")
        engine, doc = load(script)

        result = engine.run_macro(PULL)

        # Without the validation hook the body still applies the revision, and
        # the log says so instead of "nothing pending".
        assert result.ok, result.error
        assert "rev 1 applied" in result.log
        assert parse_applied(installed["bridge"].applied_path)["rev"] == 1


class TestPush:
    def test_push_writes_snapshot_and_state(self, installed):
        engine, doc = load(installed["script"])
        doc.events()[0].set("Text", "edited in Aegisub")
        doc.mark_dirty()
        result = engine.run_macro(PUSH)
        assert result.ok, result.error

        bridge = installed["bridge"]
        snapshot = bridge.read_snapshot()
        assert "edited in Aegisub" in snapshot
        assert snapshot.splitlines()[0] == "[Script Info]"
        state = parse_state(bridge.state_path)
        assert state["ae_rev"] == 1
        assert state["event_count"] == 2
        assert state["trigger"] == "push"
        assert state["origin"] == "aegisub"
        assert state["changed"] == "1"
        assert state["hotkey"] == "Ctrl-Alt-M"
        assert state["aegisub_version"] == "4"
        assert state["format"] == "krapau-bridge/1"

    def test_push_is_idempotent_when_nothing_changed(self, installed):
        engine, _doc = load(installed["script"])
        engine.run_macro(PUSH)
        first = parse_state(installed["bridge"].state_path)
        engine.run_macro(PUSH)
        second = parse_state(installed["bridge"].state_path)
        assert second["ae_rev"] == first["ae_rev"]
        assert second["changed"] == "0"
        assert second["hash"] == first["hash"]

    def test_push_revision_bumps_after_an_edit(self, installed):
        engine, doc = load(installed["script"])
        engine.run_macro(PUSH)
        doc.events()[0].set("Text", "changed again")
        engine.run_macro(PUSH)
        assert parse_state(installed["bridge"].state_path)["ae_rev"] == 2

    def test_pushed_text_round_trips_through_the_python_parser(self, installed):
        engine, doc = load(installed["script"])
        engine.run_macro(PUSH)
        again = AssDocument.from_text(installed["bridge"].read_snapshot())
        assert event_state(again) == event_state(doc)
        assert [e.text for e in again.events()] == [e.text for e in doc.events()]

    def test_status_macro_logs_paths(self, installed):
        engine, _doc = load(installed["script"])
        result = engine.run_macro("krapau-bridge: Bridge status")
        assert result.ok
        assert str(installed["dir"]) in result.log
        assert "auto pull        : on" in result.log


class TestManualRoundTrip:
    def test_publish_then_pull_then_push_matches(self, installed):
        """MCP -> Aegisub -> MCP with a real document edit in between."""
        bridge: Bridge = installed["bridge"]
        source = AssDocument.load(str(FIXTURE))
        source.events()[0].set("Text", "from the MCP side")
        bridge.publish(source, note="round trip")

        engine, ae_doc = load(installed["script"])
        assert engine.run_macro(PULL).ok
        ae_doc.events()[0].set("Text", "typed in Aegisub afterwards")
        ae_doc.mark_dirty()
        assert engine.run_macro(PUSH).ok

        back = bridge.live_text("snapshot")
        assert "typed in Aegisub afterwards" in back["text"]
        imported = bridge.import_snapshot(doc_id="round-trip")
        assert imported["doc_id"] == "round-trip"
        assert imported["lines"] == len(ae_doc.all_lines())
        # Journal entries are flat JSON objects (BridgeEvent.to_dict).
        assert bridge.events(kinds=["mcp.publish"])[0]["note"] == "round trip"


class TestRobustness:
    def test_script_creates_a_missing_bridge_directory(self, tmp_path: Path):
        """Aegisub may write first, or the directory may have been cleaned up."""
        bridge_dir = tmp_path / "gone"
        script = tmp_path / "krapau-bridge.lua"
        script.write_text(Bridge(bridge_dir).render_script(bridge_dir=bridge_dir,
                                                          autoload_dir=tmp_path),
                          encoding="utf-8")
        assert not bridge_dir.exists()

        engine, _doc = load(script)
        result = engine.run_macro(PUSH)

        assert result.ok, result.error
        assert bridge_dir.is_dir()
        assert Bridge(bridge_dir).read_snapshot().startswith("[Script Info]")

    def test_tab_and_backslash_in_text_survive_the_wire(self, installed):
        tricky = "a\\tb\\nliteral \\\\ backslash \\N line break"
        source = AssDocument.load(str(FIXTURE))
        source.events()[0].set("Text", tricky)
        installed["bridge"].publish(source, doc_id="basic.ass")

        engine, doc = load(installed["script"])
        assert engine.run_macro(PULL).ok
        assert doc.events()[0].text == tricky


class TestJournalFromScript:
    def test_poll_sees_the_aegisub_side_after_a_push(self, installed):
        engine, _doc = load(installed["script"])
        engine.run_macro(PUSH)
        bridge: Bridge = installed["bridge"]
        events = bridge.poll()
        kinds = [event.kind for event in events]
        assert "aegisub.state" in kinds
        assert bridge.read_state()["ae_rev"] == 1
        # A second poll with nothing new must not invent events, and the poll
        # baseline must survive in its own file rather than as journal rows (otherwise
        # every watch iteration would look like a wake-up and the journal would grow
        # four rows a second).
        assert bridge.poll() == []
        assert bridge.last_seen()["ae_rev"] == 1
        assert bridge.seen_path.is_file()
        assert bridge.events(kinds=["bridge.fingerprint"]) == []
        assert bridge.seen_path.read_text(encoding="utf-8").strip().startswith("{")
        payload = json.loads(json.dumps([e.to_dict() for e in events]))
        assert all("seq" in item for item in payload)


class TestScriptDirectoryPublication:
    """The MCP side cannot guess which file Aegisub has open, so the script says.

    ``?script`` is the *directory* holding the open file.  Publishing it at load
    time is what lets ``ass_bridge_watch`` notice the user's own saves before the
    next pull happens.
    """

    def _target(self, tmp_path: Path) -> Path:
        folder = tmp_path / "subs"
        folder.mkdir()
        target = folder / "show.ass"
        target.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
        return target

    def test_loading_the_script_records_the_script_directory(self, installed, tmp_path):
        target = self._target(tmp_path)
        load(installed["script"], target)
        assert installed["bridge"].read_state()["script"] == str(target.parent)

    def test_loading_the_script_keeps_the_revision_bookkeeping(self, installed, tmp_path):
        # Re-writing the whole state at load would forget which revision was
        # already applied, so the next pull would apply it a second time.
        target = self._target(tmp_path)
        (installed["dir"] / "state.tsv").write_text(
            "format\tkrapau-bridge/1\nae_rev\t7\norigin\tpull-applied\n", encoding="utf-8")
        load(installed["script"], target)
        state = installed["bridge"].read_state()
        assert state["ae_rev"] == 7
        assert state["origin"] == "pull-applied"
        assert state["script"] == str(target.parent)
