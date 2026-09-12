"""The Python half of the Aegisub bridge, driven through the MCP tool surface.

``tests/test_bridge_lua.py`` covers the Aegisub-side script; this file covers the
MCP side: publishing a revision, seeing Aegisub's state and autosave as realtime
events, importing the live document back into the workspace, and installing the
script into an Aegisub user directory.  Where a test needs the other half it runs
the *real* rendered script through ``LuaEngine``, so both halves are exercised
against the same bridge directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegisub_mcp.asscore.document import AssDocument
from aegisub_mcp.bridge import (BRIDGE_ENV, DEFAULT_HOTKEY, PULL_COMMAND, SCRIPT_NAME,
                                Bridge, default_bridge_dir, user_dir)
from aegisub_mcp.lua.engine import LuaEngine
from aegisub_mcp.tools import base as base_tools
from aegisub_mcp.tools import bridge_tools as B
from aegisub_mcp.tools import lines as L
from aegisub_mcp.tools.base import ToolError

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "real" / "basic.ass"

PULL = "krapau-bridge: Pull changes from MCP"
PUSH = "krapau-bridge: Push my document to MCP"


@pytest.fixture(autouse=True)
def clean_workspace():
    """Every test starts from a pristine module-level workspace singleton."""
    base_tools.workspace.__init__()
    yield
    base_tools.workspace.__init__()


@pytest.fixture()
def bridge_dir(tmp_path: Path) -> Path:
    return tmp_path / "bridge"


@pytest.fixture()
def aegisub(installed_bridge):
    """Aegisub's half: the real script loaded into the real Automation 4 host."""

    def run(macro: str = PULL, doc_path: Path = FIXTURE) -> tuple:
        doc = AssDocument.load(str(doc_path))
        engine = LuaEngine(doc)
        engine.load_file(installed_bridge["script"])
        return engine.run_macro(macro), doc, engine

    return run


@pytest.fixture()
def installed_bridge(tmp_path: Path) -> dict:
    """A rendered Aegisub-side script whose bridge directory exists."""
    bridge = Bridge(tmp_path / "bridge")
    bridge.ensure()
    script = tmp_path / "krapau-bridge.lua"
    script.write_text(bridge.render_script(bridge_dir=bridge.dir,
                                          autoload_dir=tmp_path / "autoload",
                                          hotkey=DEFAULT_HOTKEY),
                      encoding="utf-8")
    return {"bridge": bridge, "script": script, "dir": bridge.dir}


def write_autosave(user: Path, name: str, stamp: str, text: str) -> Path:
    """Write an autosave copy the way Aegisub names them."""
    directory = user / "autosave"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{stamp}.AUTOSAVE.ass"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- paths


class TestPaths:
    def test_paths_reports_both_sides(self, bridge_dir: Path, tmp_path: Path) -> None:
        payload = B.ass_bridge_paths(bridge_dir=str(bridge_dir),
                                     user_dir_override=str(tmp_path / "home"))

        assert payload["bridge_dir"] == str(bridge_dir)
        assert payload["bridge_env_var"] == BRIDGE_ENV
        assert payload["user_dir"] == str(tmp_path / "home")
        assert payload["aegisub_config"] == str(tmp_path / "home" / "config.json")
        assert payload["autoload_dir"] == str(tmp_path / "home" / "automation" / "autoload")
        assert Path(payload["lua_asset"]).name == SCRIPT_NAME.replace(".lua", ".lua.in")

    def test_bridge_dir_honours_the_env_var(self, bridge_dir: Path, monkeypatch) -> None:
        monkeypatch.setenv(BRIDGE_ENV, str(bridge_dir))
        assert default_bridge_dir() == bridge_dir

    def test_user_dir_honours_home(self, tmp_path: Path, monkeypatch) -> None:
        # Aegisub ignores XDG_CONFIG_HOME: its user directory is $HOME/.aegisub.
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert user_dir() == tmp_path / "home" / ".aegisub"


# --------------------------------------------------------------------------- publish


class TestPublish:
    def test_publish_queues_a_revision(self, bridge_dir: Path) -> None:
        did = L.ass_new_document()["doc_id"]
        L.ass_add_line(0, 1000, "hello")

        payload = B.ass_bridge_publish(doc_id=did, note="from the test",
                                      bridge_dir=str(bridge_dir))

        assert payload["rev"] == 1
        assert payload["pending"]["pending"] is True
        assert payload["pending"]["mcp_rev"] == 1
        assert (bridge_dir / "to-aegisub.tsv").is_file()
        events = B.ass_bridge_events(bridge_dir=str(bridge_dir))
        assert [e["kind"] for e in events["events"]] == ["mcp.publish"]
        assert events["events"][0]["rev"] == 1

    def test_republish_bumps_the_revision(self, bridge_dir: Path) -> None:
        did = L.ass_new_document()["doc_id"]
        L.ass_add_line(0, 1000, "one")
        first = B.ass_bridge_publish(doc_id=did, bridge_dir=str(bridge_dir))
        L.ass_add_line(2000, 3000, "two")
        second = B.ass_bridge_publish(doc_id=did, bridge_dir=str(bridge_dir))

        assert (first["rev"], second["rev"]) == (1, 2)
        status = B.ass_bridge_status(bridge_dir=str(bridge_dir))
        assert status["mcp_revision"] == 2
        assert status["pending"]["mcp_rev"] == 2
        assert status["pending"]["pending"] is True
        assert status["pending"]["applied"] is False


# --------------------------------------------------------------------------- realtime


class TestRealtime:
    def test_aegisub_state_becomes_events(self, installed_bridge: dict) -> None:
        """Running Aegisub's Push macro shows up as bridge events."""
        bridge = installed_bridge["bridge"]
        doc = AssDocument.load(str(FIXTURE))
        engine = LuaEngine(doc)
        engine.load_file(installed_bridge["script"])
        engine.run_macro(PUSH)

        since = 0
        events = B.ass_bridge_events(bridge_dir=str(bridge.dir))
        kinds = [event["kind"] for event in events["events"]]
        assert "aegisub.state" in kinds
        assert "aegisub.snapshot" in kinds
        assert events["last_seq"] >= 2

        # ``since`` is the cursor an MCP client keeps: nothing new afterwards.
        again = B.ass_bridge_events(since=events["last_seq"], bridge_dir=str(bridge.dir))
        assert again["count"] == 0
        assert again["last_seq"] == events["last_seq"]

    def test_watch_returns_as_soon_as_aegisub_moves(self, installed_bridge: dict) -> None:
        """A watch started before Aegisub acts returns that event, not a timeout."""
        bridge = installed_bridge["bridge"]
        doc = AssDocument.load(str(FIXTURE))
        engine = LuaEngine(doc)
        engine.load_file(installed_bridge["script"])
        engine.run_macro(PUSH)

        payload = B.ass_bridge_watch(seconds=1.0, interval=0.05,
                                     bridge_dir=str(bridge.dir))
        assert payload["count"] >= 1
        assert payload["waited_s"] < 1.0
        assert "aegisub.state" in [event["kind"] for event in payload["events"]]

    def test_watch_is_quiet_when_aegisub_is(self, bridge_dir: Path) -> None:
        Bridge(bridge_dir).ensure()
        payload = B.ass_bridge_watch(seconds=0.05, interval=0.02,
                                     bridge_dir=str(bridge_dir))
        assert payload["count"] == 0
        assert payload["events"] == []


# --------------------------------------------------------------------------- round trip
class TestRoundTrip:
    def test_published_revision_reaches_aegisub_and_is_acknowledged(
            self, installed_bridge: dict) -> None:
        bridge = installed_bridge["bridge"]
        did = L.ass_new_document()["doc_id"]
        L.ass_add_line(0, 1500, "MCP wrote this")
        published = B.ass_bridge_publish(doc_id=did, note="round trip",
                                        bridge_dir=str(bridge.dir))
        assert published["rev"] == 1
        assert published["pending"]["pending"] is True

        doc = AssDocument.load(str(FIXTURE))
        engine = LuaEngine(doc)
        engine.load_file(installed_bridge["script"])
        result = engine.run_macro(PULL)
        assert result.ok, result.error
        assert [e.text for e in doc.events() if not e.is_comment] == ["MCP wrote this"]
        # The open document had two events, the published one has one, so the pull
        # is a structural change rather than a per-line patch.
        assert result.structure_changed or result.changed_lines

        status = B.ass_bridge_status(bridge_dir=str(bridge.dir))
        assert status["pending"]["applied_rev"] == 1
        assert status["pending"]["pending"] is False
        assert status["pending"]["applied"] is True
        assert status["pending"]["applied_result"] == "ok"

    def test_aegisub_edit_comes_back_into_the_workspace(self, installed_bridge: dict,
                                                        tmp_path: Path) -> None:
        """Aegisub's live document is readable and importable (the pull direction)."""
        bridge = installed_bridge["bridge"]
        doc = AssDocument.load(str(FIXTURE))
        engine = LuaEngine(doc)
        engine.load_file(installed_bridge["script"])
        for entry in doc.events():
            if not entry.is_comment:
                entry.set("Text", "typed in Aegisub")
        assert engine.run_macro(PUSH).ok

        live = B.ass_bridge_live(bridge_dir=str(bridge.dir))
        assert "typed in Aegisub" in live["text"]
        assert live["source"] == "snapshot"
        assert live["sha256"] and live["lines"] >= 4

        imported = B.ass_bridge_pull(bridge_dir=str(bridge.dir))
        assert imported["live_source"] == "snapshot"
        assert imported["doc_id"]
        imported_doc = base_tools.workspace.get(imported["doc_id"])
        assert [e.text for e in imported_doc.events() if not e.is_comment] \
            == ["typed in Aegisub"]

    def test_pull_without_any_aegisub_output_is_a_tool_error(self, bridge_dir: Path) -> None:
        Bridge(bridge_dir).ensure()
        with pytest.raises(ToolError):
            B.ass_bridge_live(bridge_dir=str(bridge_dir))
        with pytest.raises(ToolError):
            B.ass_bridge_pull(bridge_dir=str(bridge_dir))


# --------------------------------------------------------------------------- autosave
class TestAutosave:
    def test_newest_autosave_is_reported_and_readable(self, bridge_dir: Path,
                                                      tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir(parents=True)
        (home / "config.json").write_text(json.dumps({
            "App": {"Auto": {"Save": True, "Save Every Seconds": 60,
                             "Save on Every Change": True}},
            "Path": {"Auto": {"Save": "?user/autosave"}},
        }), encoding="utf-8")
        text = FIXTURE.read_text(encoding="utf-8")
        write_autosave(home, "basic.ass", "2026-09-13-02-00-00", text)

        payload = B.ass_bridge_autosave(bridge_dir=str(bridge_dir),
                                       user_dir_override=str(home))
        assert payload["settings"]["on_change"] is True
        assert payload["settings"]["enabled"] is True
        assert payload["autosave_dir"] == str(home / "autosave")
        assert payload["newest"]["path"].endswith("AUTOSAVE.ass")
        assert payload["newest"]["bytes"] == len(text.encode("utf-8"))

        live = B.ass_bridge_live(source="autosave", bridge_dir=str(bridge_dir),
                                 user_dir_override=str(home))
        assert live["source"] == "autosave"
        # Read back as utf-8-sig, exactly like Aegisub writes it.
        assert live["text"].lstrip("\ufeff") == text.lstrip("\ufeff")


# --------------------------------------------------------------------------- install
class TestInstall:
    def test_dry_run_touches_nothing(self, bridge_dir: Path, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()

        report = B.ass_bridge_install(bridge_dir=str(bridge_dir),
                                      user_dir_override=str(home))

        assert report["dry_run"] is True
        assert report["script"]["written"] if "written" in report["script"] else True
        assert not (home / "automation").exists()
        assert not (home / "config.json").exists()
        assert report["next"].startswith("restart Aegisub")

    def test_install_writes_script_config_and_hotkey(self, bridge_dir: Path,
                                                      tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir(parents=True)
        # Aegisub's real config.json nests option names under objects; the flat
        # "App/Auto/Save" keys below are what an older build of this module wrote
        # and are supposed to be cleaned up.
        (home / "config.json").write_text(json.dumps({
            "App": {"Auto": {"Save": False, "Save Every Seconds": 60}},
            "Path": {"Auto": {"Save": "?user/autosave"}},
            "App/Auto/Save": False,
        }), encoding="utf-8")

        report = B.ass_bridge_install(bridge_dir=str(bridge_dir),
                                      user_dir_override=str(home), dry_run=False)

        script = home / "automation" / "autoload" / SCRIPT_NAME
        assert script.is_file()
        assert str(bridge_dir) in script.read_text(encoding="utf-8")
        assert report["script"]["sha256"] == report["install_record"]["script_sha256"]

        config = json.loads((home / "config.json").read_text(encoding="utf-8"))
        assert config["App"]["Auto"]["Save"] is True
        # Interval must stay >= 1: 0 stops Aegisub's autosave timer outright
        # (autosave_timer_changed() in src/subs_controller.cpp), which would kill
        # the ?user/autosave stream the MCP side watches.
        assert config["App"]["Auto"]["Save Every Seconds"] == 5
        # Save-on-every-change rewrites the user's own file in place, so installing
        # must not turn it on behind the user's back.
        assert "Save on Every Change" not in config["App"]["Auto"]
        # The pre-existing key is preserved, and the flat junk key is gone.
        assert config["Path"]["Auto"]["Save"] == "?user/autosave"
        assert "App/Auto/Save" not in config
        assert report["config"]["repaired_flat_keys"] == ["App/Auto/Save"]
        assert report["config"]["keys_present"]["path"] == "?user/autosave"
        assert (home / "config.json.bak").is_file()

        hotkeys = json.loads((home / "hotkey.json").read_text(encoding="utf-8"))
        # The command id must match Aegisub's automation/lua/<stem>/<macro name>,
        # prefix included -- otherwise the keystroke is bound to nothing.
        assert hotkeys["Default"][PULL_COMMAND] == [DEFAULT_HOTKEY]
        assert PULL_COMMAND == "automation/lua/krapau-bridge/krapau-bridge: Pull changes from MCP"

        # Second install is a no-op for the script and the config.
        again = B.ass_bridge_install(bridge_dir=str(bridge_dir),
                                     user_dir_override=str(home), dry_run=False)
        assert again["script"]["up_to_date"] is True
        assert again["config"]["changes"] == {}
        assert again["hotkey"]["change"] is None

    def test_install_refuses_to_clobber_a_foreign_script(self, bridge_dir: Path,
                                                         tmp_path: Path) -> None:
        home = tmp_path / "home"
        autoload = home / "automation" / "autoload"
        autoload.mkdir(parents=True)
        (autoload / SCRIPT_NAME).write_text("-- someone else's script\n", encoding="utf-8")

        with pytest.raises(ToolError):
            B.ass_bridge_install(bridge_dir=str(bridge_dir),
                                 user_dir_override=str(home), dry_run=False)

        assert (autoload / SCRIPT_NAME).read_text(encoding="utf-8") \
            == "-- someone else's script\n"

    def test_status_reports_the_install(self, bridge_dir: Path, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        B.ass_bridge_install(bridge_dir=str(bridge_dir), user_dir_override=str(home),
                             dry_run=False)

        status = B.ass_bridge_status(bridge_dir=str(bridge_dir),
                                     user_dir_override=str(home))
        assert status["installed"]["script"].endswith(SCRIPT_NAME)
        assert status["installed"]["script_sha256"]
        assert status["installed"]["bridge_dir"] == str(bridge_dir)
        assert status["installed"]["hotkey"] == DEFAULT_HOTKEY
        assert status["installed"]["auto_pull"] is True
        assert status["installed"]["autosave"]["enable"] is True
        assert status["installed"]["autosave"]["interval"] == 5
        assert status["installed"]["autosave"]["save_on_every_change"] is False
