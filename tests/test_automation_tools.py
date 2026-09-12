"""The MCP tools that run *real* Aegisub Automation 4 scripts.

The whole point of these tools is that they do not re-implement Automation 4:
they drive the repo's :class:`~aegisub_mcp.lua.engine.LuaEngine`, which hosts
LuaJIT with Aegisub's own ``aegisub.*`` namespace, the ``subs`` subtitle-file
object and the vendored ``karaskel``/``utils`` libraries.  So the tests below
check the two halves that can break independently:

* the *engine* really executes the script -- mutations land in the workspace
  document, macro return values become the active line/selection, and
  ``register_macro``'s validation function is honoured;
* the *tools* wire arguments through honestly -- dry runs leave the document
  alone, load failures come back as ``ok: false`` instead of an exception,
  and bad arguments raise :class:`ToolError`.

One upstream behaviour worth stating because it looks like a bug and is not:
``line = subs[i]`` hands back a **copy** (upstream
``LuaAssFile::AssEntryToLua`` does ``lua_newtable`` and fills it), so editing
``line.text`` alone changes nothing -- a script must write the copy back with
``subs[i] = line``.  Aegisub's own ``automation/autoload/strip-tags.lua`` does
exactly that, and ``test_editing_the_copy_without_writing_it_back_is_a_noop``
pins the semantics so nobody "fixes" it into a divergence from Aegisub.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from aegisub_mcp.tools import automation_tools as A
from aegisub_mcp.tools import lines as L
from aegisub_mcp.tools.base import ToolError, workspace


@pytest.fixture(autouse=True)
def clean_workspace():
    """Every test starts from a pristine module-level workspace singleton."""
    workspace.__init__()
    yield
    workspace.__init__()


@pytest.fixture()
def doc() -> str:
    """A workspace document with two dialogue lines and one comment."""
    L.ass_new_document(doc_id="auto")
    L.ass_add_line(1000, 2000, "first")
    L.ass_add_line(3000, 4000, "second")
    L.ass_add_line(5000, 6000, "third", comment=True)
    return "auto"


def write_script(tmp_path: pathlib.Path, code: str, name: str = "script.lua") -> pathlib.Path:
    path = tmp_path / name
    path.write_text(code, encoding="utf-8")
    return path


def event_texts(doc_id: str = "auto") -> list[str]:
    """The text of every event line in the workspace document, in order."""
    return [line.text for line in workspace.get(doc_id).events()]


def lua_replace_text(macro: str, old: str, new: str) -> str:
    """Lua source for a macro that rewrites the one dialogue line matching ``old``.

    ``subs`` is indexed by *file line* (heading, ``Format:``, styles and events
    all count -- ``#subs`` is the total line count), so a script has to look at
    ``line.class``/``line.text`` instead of assuming the first dialogue is at
    ``subs[1]``.  This mirrors how real Automation 4 scripts are written.
    """
    return f"""
        aegisub.register_macro("{macro}", "d", function(subs, sel)
            for i = 1, #subs do
                local line = subs[i]
                if line.class == "dialogue" and line.text == "{old}" then
                    line.text = "{new}"
                    subs[i] = line
                end
            end
        end)
    """


# --------------------------------------------------------------------------- listing
# The discovery tools are read-only: they load a script into a scratch document
# just to see what it registers, and must never touch the workspace copy.


def test_dirs_reports_the_search_path_and_what_is_in_it(tmp_path: pathlib.Path) -> None:
    write_script(tmp_path, 'aegisub.register_macro("m", "d", function() end)\n')

    payload = A.ass_automation_dirs(directory=str(tmp_path))

    first = payload["directories"][0]
    assert first["path"] == str(tmp_path)
    assert first["exists"] is True
    assert first["scripts"] == 1
    assert first["sample"] == ["script.lua"]


def test_list_reports_macros_and_filters_separately(tmp_path: pathlib.Path) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Bump", "shift everything", function(subs, sel) end)
        aegisub.register_filter("Clean", "drop tags", function(subs, cfg) return subs end, 5)
    """)

    payload = A.ass_automation_list(directory=str(tmp_path))

    # The listing walks Aegisub's whole search path (the vendored bundle and the
    # user's ~/.aegisub/automation are on it too), so find *our* script by name.
    assert payload["count"] == len(payload["scripts"])
    entry = next(item for item in payload["scripts"] if item["name"] == "script.lua")
    assert entry["path"] == str(tmp_path / "script.lua")
    assert entry["loaded"] is True
    assert {item["name"] for item in entry["macros"]} == {"Bump"}
    assert {item["name"] for item in entry["filters"]} == {"Clean"}
    assert entry["macros"][0]["kind"] == "macro"
    assert entry["filters"][0]["kind"] == "filter"


def test_info_hashes_the_script_and_lists_its_registrations(tmp_path: pathlib.Path) -> None:
    path = write_script(tmp_path, 'aegisub.register_macro("Only", "d", function() end)\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    payload = A.ass_automation_info("script.lua", directory=str(tmp_path))

    assert payload["sha256"] == digest
    assert payload["path"] == str(path)
    assert [item["name"] for item in payload["macros"]] == ["Only"]
    assert payload["filters"] == []


# --------------------------------------------------------------------------- macros


def test_run_macro_changes_the_workspace_document(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Upper", "uppercase dialogue", function(subs, sel)
            for i = 1, #subs do
                local line = subs[i]
                -- ``class`` is "dialogue" for comment lines too; ``comment`` tells
                -- them apart, exactly as in Aegisub.
                if line.class == "dialogue" and not line.comment then
                    line.text = string.upper(line.text)
                    subs[i] = line        -- the read handed us a copy: write it back
                end
            end
            aegisub.set_undo_point("uppercase")
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is True
    assert result["error"] is None
    assert result["unchanged"] is False
    assert result["structure_changed"] is False
    assert event_texts(doc) == ["FIRST", "SECOND", "third"]
    assert result["undo_points"] == ["uppercase"]
    assert result["undo_depth"] >= 1
    assert result["undo_added"] >= 1
    assert result["redo_depth"] == 0
    assert result["changed_lines"]  # file-index space, as in Aegisub


def test_editing_the_copy_without_writing_it_back_is_a_noop(tmp_path: pathlib.Path, doc: str) -> None:
    """``line = subs[i]`` is a copy in Aegisub, so a bare field write is lost."""
    write_script(tmp_path, """
        aegisub.register_macro("Lost", "no write-back", function(subs, sel)
            for i = 1, #subs do subs[i].text = "clobbered" end
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is True
    assert result["unchanged"] is True
    assert event_texts(doc) == ["first", "second", "third"]


def test_run_macro_dry_run_leaves_the_document_untouched(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Upper", "uppercase", function(subs, sel)
            for i = 1, #subs do
                local line = subs[i]
                if line.class == "dialogue" then
                    line.text = string.upper(line.text)
                    subs[i] = line
                end
            end
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path),
                                        dry_run=True)

    assert result["dry_run"] is True
    assert result["unchanged"] is True
    assert event_texts(doc) == ["first", "second", "third"]


def test_run_macro_reports_the_active_line_and_selection_it_returns(tmp_path: pathlib.Path,
                                                                   doc: str) -> None:
    """A macro may return ``active_line, selection`` -- Aegisub moves the GUI."""
    write_script(tmp_path, """
        aegisub.register_macro("Move", "return a selection", function(subs, sel)
            return 2, { 2, 3 }
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is True
    assert result["active_line"] == 2
    assert result["selection"] == [2, 3]


def test_macro_sees_the_selection_and_active_line_it_was_given(tmp_path: pathlib.Path,
                                                               doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Report", "log the selection", function(subs, sel, active)
            aegisub.log("sel_count=%d active=%s\\n", #sel, tostring(active))
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path),
                                        selection=[2, 4])

    assert result["ok"] is True
    assert "sel_count=2" in result["log"]


def test_run_macro_from_inline_source(doc: str) -> None:
    result = A.ass_automation_run_macro(doc_id=doc,
                                        source=lua_replace_text("Inline", "first", "from inline"))

    assert result["ok"] is True
    assert event_texts(doc)[0] == "from inline"


def test_run_macro_save_to_writes_the_result(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, lua_replace_text("Touch", "first", "saved"))
    out = tmp_path / "out.ass"

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path),
                                        save_to=str(out))

    assert result["saved"]["path"] == str(out)
    assert result["saved"]["bytes"] > 0
    assert result["saved"]["bytes"] == out.stat().st_size
    assert "saved" in out.read_text(encoding="utf-8-sig")


def test_dialog_answers_reach_the_macro(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Ask", "d", function(subs)
            local button, results = aegisub.dialog.display(
                { { class = "edit", name = "input", value = "default",
                    x = 0, y = 0, width = 10, height = 1 } },
                { "OK", "Cancel" })
            aegisub.log("answer=" .. tostring(button) .. " input=" .. tostring(results.input) .. "\\n")
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path),
                                        dialog_answers={"display": {"button": "Cancel",
                                                                    "values": {"input": "typed"}}})

    # Aegisub returns the pressed button's *label* (auto4_lua_dialog.cpp
    # LuaReadBack pushes buttons[n].second), not an index.
    assert result["ok"] is True
    assert "answer=Cancel input=typed" in result["log"]
    record = result["dialogs"][0]
    assert record["kind"] == "display"
    assert record["buttons"] == ["OK", "Cancel"]
    assert record["button"] == 2
    # The control spec Aegisub hands the dialog is the first argument's array.
    assert record["controls"][0]["name"] == "input"


def test_dialog_closed_without_a_button_returns_false(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Ask", "d", function(subs)
            local button = aegisub.dialog.display(
                { { class = "label", label = "?", x = 0, y = 0, width = 5, height = 1 } },
                { "OK", "Cancel" })
            if not button then
                aegisub.log("cancelled\\n")
            end
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path),
                                        dialog_answers={"display": {"button": 99}})

    assert result["ok"] is True
    assert "cancelled" in result["log"]
    assert result["dialogs"][0]["button"] is None


def test_a_dialog_without_an_answer_is_reported_not_hung(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Ask", "d", function(subs)
            aegisub.dialog.display({ { class = "label", label = "?", x = 0, y = 0,
                                       width = 5, height = 1 } }, { "OK" })
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is False
    assert "dialog" in (result["error"] or "").lower()
    assert result["unchanged"] is True


# --------------------------------------------------------------------------- validation
# ``register_macro``'s optional validation function decides whether the macro is
# applicable at all.  Reading the wrong Lua slot here used to make *every*
# validate-gated macro refuse, so both branches are pinned.


def test_validate_returning_true_runs_the_macro(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Gated", "d", function(subs)
            for i = 1, #subs do
                local line = subs[i]
                if line.class == "dialogue" and line.text == "first" then
                    line.text = "ran"
                    subs[i] = line
                end
            end
        end, function(subs, sel, active)
            return true
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is True, result["error"]
    assert event_texts(doc)[0] == "ran"


def test_validate_returning_false_refuses_without_touching_the_document(tmp_path: pathlib.Path,
                                                                       doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Gated", "d", function(subs)
            local line = subs[1]
            line.text = "must not happen"
            subs[1] = line
        end, function(subs, sel, active)
            return false
        end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is False
    assert "not applicable" in (result["error"] or "")
    assert result["unchanged"] is True
    assert event_texts(doc) == ["first", "second", "third"]


def test_validate_raising_is_reported_as_refused(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_macro("Gated", "d", function(subs) end,
            function(subs, sel, active) error("cannot decide") end)
    """)

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is False
    assert "refused" in (result["error"] or "")
    assert "cannot decide" in (result["error"] or "")


# --------------------------------------------------------------------------- filters


def test_run_filter_edits_the_subtitles(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_filter("Prefix", "add a prefix", 100, function(subs, cfg)
            local prefix = (cfg and cfg.prefix) or "F:"
            for i = 1, #subs do
                local line = subs[i]
                if line.class == "dialogue" and not line.comment then
                    line.text = prefix .. line.text
                    subs[i] = line
                end
            end
        end)
    """)

    result = A.ass_automation_run_filter("script.lua", doc_id=doc, directory=str(tmp_path),
                                         config={"prefix": "F:"})

    assert result["ok"] is True
    assert result["script"] == str(tmp_path / "script.lua")
    assert result["script_sha256"]
    assert [item["name"] for item in result["registered"]] == ["Prefix"]
    assert event_texts(doc) == ["F:first", "F:second", "third"]


def test_run_filters_uses_aegisub_order_and_stops_at_the_first_failure(
        tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, """
        aegisub.register_filter("Second", "d", 20, function(subs)
            for i = 1, #subs do
                local line = subs[i]
                if line.class == "dialogue" and not line.comment then
                    line.text = line.text .. "+second"
                    subs[i] = line
                end
            end
        end)
        aegisub.register_filter("First", "d", 10, function(subs)
            for i = 1, #subs do
                local line = subs[i]
                if line.class == "dialogue" and not line.comment then
                    line.text = line.text .. "+first"
                    subs[i] = line
                end
            end
        end)
        aegisub.register_filter("Boom", "d", 30, function(subs) error("filter exploded") end)
        aegisub.register_filter("Never", "d", 40, function(subs)
            for i = 1, #subs do
                local line = subs[i]
                if line.class == "dialogue" then
                    line.text = "never runs"
                    subs[i] = line
                end
            end
        end)
    """)

    result = A.ass_automation_run_filters("script.lua", doc_id=doc, directory=str(tmp_path))

    names = [item["name"] for item in result["results"]]
    assert names[:3] == ["First", "Second", "Boom"]  # priority 10, 20, 30
    assert "Never" not in names  # the run stops at the first failure
    assert result["ok"] is False
    assert result["count"] == 3
    assert "filter exploded" in result["results"][-1]["error"]
    # First ran before Second, and "Never" never ran at all.
    assert event_texts(doc) == ["first+first+second", "second+first+second", "third"]


# --------------------------------------------------------------------------- failures


def test_syntax_error_is_reported_as_ok_false_with_the_lua_error(tmp_path: pathlib.Path,
                                                                 doc: str) -> None:
    write_script(tmp_path, 'aegisub.register_macro("Broken", "d", function(subs)\n')

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is False
    assert result["error"]
    assert result["doc_id"] == doc
    assert result["unchanged"] is True
    assert result["script"] == str(tmp_path / "script.lua")


def test_error_raised_inside_the_macro_is_reported_not_propagated(tmp_path: pathlib.Path,
                                                                 doc: str) -> None:
    write_script(tmp_path, 'aegisub.register_macro("Explode", "d", function(subs) error("boom") end)\n')

    result = A.ass_automation_run_macro("script.lua", doc_id=doc, directory=str(tmp_path))

    assert result["ok"] is False
    assert "boom" in (result["error"] or "")
    assert result["unchanged"] is True


def test_missing_script_raises_tool_error(doc: str, tmp_path: pathlib.Path) -> None:
    with pytest.raises(ToolError):
        A.ass_automation_run_macro("nope.lua", doc_id=doc, directory=str(tmp_path))


def test_unknown_macro_name_raises_tool_error(tmp_path: pathlib.Path, doc: str) -> None:
    write_script(tmp_path, 'aegisub.register_macro("Only", "d", function() end)\n')

    with pytest.raises(ToolError):
        A.ass_automation_run_macro("script.lua", macro="Absent", doc_id=doc,
                                   directory=str(tmp_path))


def test_script_and_source_together_raise_tool_error(doc: str) -> None:
    with pytest.raises(ToolError):
        A.ass_automation_run_macro(doc_id=doc, script="x.lua", source="return")

    with pytest.raises(ToolError):
        A.ass_automation_run_macro(doc_id=doc)


def test_from_source_rejects_an_unknown_kind(doc: str) -> None:
    with pytest.raises(ToolError):
        A.ass_automation_from_source("return", kind="banana", doc_id=doc)


def test_from_source_can_run_a_filter() -> None:
    L.ass_new_document(doc_id="src")
    L.ass_add_line(0, 1000, "hello")

    result = A.ass_automation_from_source("""
        aegisub.register_filter("Upper", "d", 1, function(subs)
            for i = 1, #subs do
                local line = subs[i]
                if line.class == "dialogue" then
                    line.text = string.upper(line.text)
                    subs[i] = line
                end
            end
        end)
    """, kind="filter", doc_id="src")

    assert result["ok"] is True
    assert event_texts("src") == ["HELLO"]


def test_scripts_register_with_the_mcp_server_under_ass_automation_names() -> None:
    names = [name for name in dir(A) if name.startswith("ass_automation_")]
    assert {"ass_automation_dirs", "ass_automation_list", "ass_automation_info",
            "ass_automation_run_macro", "ass_automation_run_filter",
            "ass_automation_run_filters", "ass_automation_from_source"} <= set(names)
