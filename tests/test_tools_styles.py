"""Tests for :mod:`aegisub_mcp.tools.styles`.

The module under test owns the *non-line* half of an ASS/SSA document: the
style library, ``[Script Info]``, attachments, extradata and structural
validation.  These tests drive it exactly the way FastMCP would — by calling
the module level ``ass_*`` functions — against the real world fixtures in
``tests/fixtures/real``.

Design notes
------------
* Every test opens a *copy* of a fixture (or writes a small crafted document)
  so that no two tests ever share a document id inside the workspace singleton.
* The ``ws`` fixture restores ``workspace.output_dir`` and closes every
  document it opened, so the module level singleton does not leak between
  tests.
* Assertions are made against observed, real output; anything environment
  dependent (fontconfig availability, a renderer for the bbox probe) is
  guarded with an explicit skip instead of a weaker assertion.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from aegisub_mcp.tools import base, styles as S
from aegisub_mcp.tools.base import ToolError

# --------------------------------------------------------------------------
# fixtures and helpers
# --------------------------------------------------------------------------

REAL = Path(__file__).resolve().parent / "fixtures" / "real"

FIXTURES = sorted(p.name for p in REAL.iterdir() if p.is_file())

STYLE_FIELDS = [
    "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour",
    "BackColour", "Bold", "Italic", "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing",
    "Angle", "BorderStyle", "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV",
    "Encoding",
]

STYLE_FORMAT_LINE = "Format: " + ", ".join(STYLE_FIELDS)
EVENT_FORMAT_LINE = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"

EXPECTED_TOOLS = {
    "ass_list_styles", "ass_get_style", "ass_add_style", "ass_update_style", "ass_remove_style",
    "ass_rename_style", "ass_copy_style", "ass_reorder_styles", "ass_style_usage",
    "ass_style_for_line", "ass_check_font_substitution", "ass_get_script_info",
    "ass_set_script_info", "ass_remove_script_info", "ass_set_play_res", "ass_set_wrap_style",
    "ass_set_scaled_border_and_shadow", "ass_set_timing_info", "ass_list_attachments",
    "ass_add_attachment", "ass_extract_attachment", "ass_remove_attachment",
    "ass_list_extradata", "ass_set_extradata", "ass_validate",
}

DEFAULT_STYLE_VALUES = {
    "Name": "Body", "Fontname": "Arial", "Fontsize": "40", "PrimaryColour": "&H00FFFFFF",
    "SecondaryColour": "&H000000FF", "OutlineColour": "&H00000000", "BackColour": "&H80000000",
    "Bold": "0", "Italic": "0", "Underline": "0", "StrikeOut": "0", "ScaleX": "100",
    "ScaleY": "100", "Spacing": "0", "Angle": "0", "BorderStyle": "1", "Outline": "2",
    "Shadow": "0", "Alignment": "2", "MarginL": "10", "MarginR": "10", "MarginV": "10",
    "Encoding": "1",
}


def style_record(**overrides: str) -> str:
    """Return one ``Style:`` line for the standard 23 column v4+ format."""
    values = dict(DEFAULT_STYLE_VALUES)
    values.update(overrides)
    return "Style: " + ",".join(values[name] for name in STYLE_FIELDS)


def ass_text(*, script_info=None, styles=(), events=(), extra: str = "",
             include_styles_section: bool = True) -> str:
    """Build a small v4+ document.  All arguments are plain strings/lists."""
    info = script_info if script_info is not None else [
        ("ScriptType", "v4.00+"), ("PlayResX", "640"), ("PlayResY", "480"),
    ]
    parts = ["[Script Info]\n", "".join(f"{key}: {value}\n" for key, value in info), "\n"]
    if include_styles_section:
        parts.append("[V4+ Styles]\n")
        parts.append(STYLE_FORMAT_LINE + "\n")
        parts.extend(f"{record}\n" for record in styles)
        parts.append("\n")
    parts.append("[Events]\n")
    parts.append(EVENT_FORMAT_LINE + "\n")
    parts.extend(f"{event}\n" for event in events)
    if extra:
        parts.append(extra)
    return "".join(parts)


def dialogue(start: str, end: str, style: str, text: str, *, layer: str = "0",
             name: str = "", marginl: str = "0", marginr: str = "0", marginv: str = "0",
             effect: str = "", kind: str = "Dialogue") -> str:
    return f"{kind}: {layer},{start},{end},{style},{name},{marginl},{marginr},{marginv},{effect},{text}"


@pytest.fixture()
def ws(tmp_path):
    """Isolated workspace: private output dir, every opened doc closed after."""
    workspace = base.workspace
    previous = workspace.output_dir
    workspace.set_output_dir(tmp_path / "out")
    yield workspace
    for doc_id in list(workspace.ids()):
        workspace.close(doc_id)
    workspace.set_output_dir(previous)


def open_fixture(workspace, tmp_path, name: str, *, as_name: str | None = None) -> str:
    """Copy a real fixture somewhere private and open the copy."""
    src = REAL / name
    dst = tmp_path / "docs" / (as_name or name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return workspace.open(str(dst))


def open_text(workspace, tmp_path, text: str | bytes, *, name: str = "crafted.ass",
              newline: str = "\n", bom: bool = False) -> str:
    """Write a crafted document and open it."""
    dst = tmp_path / "docs" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        dst.write_bytes(text)
    else:
        payload = text.encode("utf-8")
        if bom:
            payload = b"\xef\xbb\xbf" + payload
        if newline != "\n":
            payload = payload.replace(b"\n", newline.encode())
        dst.write_bytes(payload)
    return workspace.open(str(dst))


def as_json(result) -> str:
    """Serialize a tool result the way the MCP transport would."""
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------
# module contract / MCP registration
# --------------------------------------------------------------------------


def test_every_required_tool_exists_and_is_documented():
    missing = sorted(name for name in EXPECTED_TOOLS if not hasattr(S, name))
    assert missing == []

    for name in sorted(EXPECTED_TOOLS):
        func = getattr(S, name)
        assert callable(func), name
        assert (func.__doc__ or "").strip(), f"{name} has no docstring"
        assert func.__annotations__.get("return") is not None, f"{name} lacks a return annotation"
        assert func.__name__ == name


def test_module_exports_no_unexpected_ass_tools():
    exported = {name for name in vars(S) if name.startswith("ass_")}
    assert exported == EXPECTED_TOOLS


def test_register_installs_every_tool_and_returns_names():
    class FakeMCP:
        def __init__(self):
            self.registered: dict[str, object] = {}

        def tool(self):
            def decorate(func):
                self.registered[func.__name__] = func
                return func

            return decorate

    mcp = FakeMCP()
    names = S.register(mcp, base.workspace)

    assert names == sorted(EXPECTED_TOOLS)
    assert set(mcp.registered) == EXPECTED_TOOLS
    for name, func in mcp.registered.items():
        assert func is getattr(S, name)


def test_register_works_with_the_real_mcp_server():
    mcpserver = pytest.importorskip("mcp.server.mcpserver")
    server = mcpserver.MCPServer("aegisub-styles-test")
    names = S.register(server, base.workspace)
    assert names == sorted(EXPECTED_TOOLS)


# --------------------------------------------------------------------------
# byte exact round trip through the tool layer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", FIXTURES)
def test_read_only_tools_preserve_bytes_exactly(ws, tmp_path, fixture):
    doc_id = open_fixture(ws, tmp_path, fixture)
    original = (REAL / fixture).read_bytes()

    # Read-only surface only: nothing here may mark the document dirty.
    S.ass_list_styles(doc_id)
    S.ass_list_styles(doc_id, include_usage=True, include_bbox=True)
    S.ass_style_usage(doc_id)
    S.ass_style_usage(doc_id, by="actor")
    S.ass_get_script_info(doc_id)
    S.ass_list_attachments(doc_id)
    S.ass_list_extradata(doc_id)
    S.ass_validate(doc_id)
    S.ass_check_font_substitution(doc_id)

    styles_list = S.ass_list_styles(doc_id)["styles"]
    if styles_list:
        S.ass_get_style(styles_list[0]["Name"], doc_id)
        S.ass_get_style(styles_list[0]["Name"], doc_id, include_glyphs=True)
    events = ws.get(doc_id).events()
    if events:
        S.ass_style_for_line(0, doc_id)

    out = tmp_path / "out" / f"roundtrip-{fixture}"
    ws.save(doc_id, out)

    written = out.read_bytes()
    assert written == original, f"{fixture} did not round-trip byte exactly"
    assert ws.get(doc_id).dirty is False


def test_save_after_a_mutation_keeps_the_rest_of_the_file_bytes(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    before = ws.get(doc_id).to_text().splitlines()

    S.ass_set_script_info("Title", "changed", doc_id)

    after = ws.get(doc_id).to_text().splitlines()
    assert len(after) == len(before)
    assert after != before
    diff = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(diff) == 1
    assert after[diff[0]] == "Title: changed"
    assert before[diff[0]].startswith("Title:")


# --------------------------------------------------------------------------
# ass_list_styles / ass_get_style
# --------------------------------------------------------------------------


def test_list_styles_exposes_every_ass_field_with_document_spelling(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_list_styles(doc_id)

    assert result["doc_id"] == doc_id
    assert result["count"] == 1
    assert len(result["styles"]) == 1
    style = result["styles"][0]

    for field in STYLE_FIELDS:
        assert field in style, field
    assert style["extra_fields"] == {}
    assert style["section"] == "[V4+ Styles]"
    assert style["index"] == 0

    # The document stores &H00FFFFFF (no trailing &) — the value must come back
    # exactly as written, not normalised to another ASS spelling.
    assert style["PrimaryColour"] == "&H00FFFFFF"
    assert style["BackColour"] == "&H80000000"
    assert style["Fontsize"] == "48"
    assert style["Fontname"] == "Arial"
    assert as_json(result)


def test_list_styles_reports_usage_counts_and_used_flag(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    style = S.ass_list_styles(doc_id, include_usage=True)["styles"][0]

    assert style["used"] is True
    # basic.ass has one Dialogue and one Comment line; both use Default.
    assert style["usage"] == {"name": "Default", "total": 2, "dialogue": 1, "comment": 1}


def test_list_styles_without_usage_omits_the_usage_key(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    style = S.ass_list_styles(doc_id)["styles"][0]
    assert "usage" not in style
    assert "used" not in style


def test_list_styles_include_bbox_measures_the_sample_text(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    style = S.ass_list_styles(doc_id, include_bbox=True)["styles"][0]

    if style["bbox"] is None:
        pytest.skip(f"no renderer for bbox probe: {style['bbox_error']}")
    bbox = style["bbox"]
    assert bbox["width"] > 0
    assert bbox["height"] > 0
    assert bbox["x1"] > bbox["x"]
    assert style["bbox_error"] is None
    assert set(bbox) == {"x", "y", "x1", "y1", "width", "height"}


def test_get_style_returns_font_resolution_glyphs_and_usage(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_get_style("Default", doc_id, include_glyphs=True)

    assert result["name"] == "Default"
    assert result["used"] is True
    assert result["usage"]["total"] == 2

    font = result["font"]
    assert font["requested"] == "Arial"
    assert font["bold"] is False and font["italic"] is False
    assert font["available"] is True
    # Arial is a fontconfig alias, not an installed family, so what fontconfig
    # resolves it to is exactly what we must report.
    assert font["resolved"]
    assert font["substituted"] is (font["resolved"].lower() != "arial")
    if font["substituted"]:
        assert font["file"]

    glyphs = result["glyphs"]
    assert glyphs["font_requested"] == "Arial"
    assert glyphs["font_resolved"] == font["resolved"]
    assert glyphs["checked"] > 0
    assert set(glyphs["covered"]) <= set(result["glyph_sample_text"])
    assert glyphs["error"] is None
    # The sample is the document's own text: basic.ass has two lines.
    assert result["glyph_sample_text"] == "Hello world Preserve me"
    assert as_json(result)


def test_get_style_sample_text_can_be_supplied(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_get_style("Default", doc_id, include_glyphs=True, sample_text="AB")
    assert result["glyph_sample_text"] == "AB"
    assert set(result["glyphs"]["covered"]) <= {"A", "B"}
    assert "A" in result["glyphs"]["covered"]


def test_get_style_without_glyphs_omits_the_report(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_get_style("Default", doc_id)
    assert "glyphs" not in result
    assert "glyph_sample_text" not in result
    assert result["style"]["Name"] == "Default"


def test_get_style_unknown_name_raises_with_the_known_names(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError) as excinfo:
        S.ass_get_style("Nope", doc_id)
    assert "Nope" in str(excinfo.value)
    assert "Default" in str(excinfo.value)


def test_get_style_is_case_insensitive_but_reports_the_stored_name(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_get_style("default", doc_id)
    assert result["name"] == "Default"
    assert result["style"]["Name"] == "Default"


# --------------------------------------------------------------------------
# ass_add_style
# --------------------------------------------------------------------------


def test_add_style_inserts_one_native_line_after_the_existing_styles(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    before = ws.get(doc_id).to_text().splitlines()

    result = S.ass_add_style("Speaker", doc_id, font="Verdana", font_size=36.0)

    assert result["created"] is True
    assert result["name"] == "Speaker"
    assert result["style"]["Name"] == "Speaker"
    assert result["style"]["Fontname"] == "Verdana"
    assert result["style"]["Fontsize"] == "36"
    # v4+ has no RelativeTo column, so it is reported as ignored rather than
    # silently dropped.
    assert result["ignored_fields"] == {"RelativeTo": 2}

    after = ws.get(doc_id).to_text().splitlines()
    index = next(i for i, line in enumerate(after) if line.startswith("Style: Speaker"))
    assert after[index - 1].startswith("Style: Default")
    assert after[:index] == before[:index]
    assert after[index + 1:] == before[index:]

    # The new line must be laid out exactly like the section's Format line.
    fmt = next(line for line in after if line.startswith("Format:"))
    assert after[index].count(",") + 1 == fmt.count(",") + 1
    assert after[index].startswith("Style: ")
    assert as_json(result)


def test_add_style_writes_relative_to_only_when_the_format_has_it(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_add_style("NoRel", doc_id, relative_to=2)
    assert "RelativeTo" in result["ignored_fields"]
    assert "RelativeTo:" not in ws.get(doc_id).to_text()

    fields = STYLE_FIELDS + ["RelativeTo"]
    doc_id2 = open_text(ws, tmp_path, "\n".join([
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 640", "PlayResY: 480", "",
        "[V4+ Styles]", "Format: " + ", ".join(fields),
        "Style: " + ",".join(["Base"] + [DEFAULT_STYLE_VALUES[f] for f in STYLE_FIELDS[1:]] + ["2"]),
        "", "[Events]", EVENT_FORMAT_LINE,
    ]) + "\n", name="rel.ass")
    result2 = S.ass_add_style("WithRel", doc_id2, relative_to=1)
    assert result2["ignored_fields"] == {}
    assert result2["style"]["RelativeTo"] == "1"
    line = next(l for l in ws.get(doc_id2).to_text().splitlines() if l.startswith("Style: WithRel"))
    assert line.split(",")[-1] == "1"


def test_add_style_only_lays_out_columns_the_format_has(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ssa")
    result = S.ass_add_style("V4", doc_id, font="Times New Roman", font_size=30.0,
                             scale_x=120.0, underline=1)

    assert set(result["ignored_fields"]) == {
        "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "OutlineColour",
        "RelativeTo",
    }
    text = ws.get(doc_id).to_text()
    fmt = next(line for line in text.splitlines() if line.startswith("Format:"))
    style = next(line for line in text.splitlines() if line.startswith("Style: V4"))
    assert style.count(",") + 1 == fmt.count(",") + 1
    assert style.startswith("Style: V4,Times New Roman,30,")
    assert as_json(result)


def test_add_style_refuses_to_clobber_without_overwrite(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError) as excinfo:
        S.ass_add_style("Default", doc_id)
    assert "already exists" in str(excinfo.value)
    assert "overwrite" in str(excinfo.value)

    before = ws.get(doc_id).to_text()
    result = S.ass_add_style("Default", doc_id, font="Consolas", overwrite=True)
    assert result["created"] is False
    assert result["style"]["Fontname"] == "Consolas"
    after = ws.get(doc_id).to_text()
    assert len(after.splitlines()) == len(before.splitlines())
    assert sum(1 for l in after.splitlines() if l.startswith("Style: Default")) == 1


def test_add_style_creates_the_styles_section_when_missing(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "unknown-fields.ass")
    assert "V4+ Styles" not in ws.get(doc_id).to_text()

    S.ass_add_style("Fresh", doc_id)

    text = ws.get(doc_id).to_text()
    assert "[V4+ Styles]" in text
    assert "Style: Fresh" in text
    lines = text.replace("\r\n", "\n").splitlines()
    assert lines.index("[V4+ Styles]") < lines.index("[Events]")
    fmt = next(line for line in lines if line.startswith("Format:"))
    style = next(line for line in lines if line.startswith("Style: Fresh"))
    assert style.count(",") + 1 == fmt.count(",") + 1


def test_add_style_validates_numbers_and_alignment(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError):
        S.ass_add_style("Bad", doc_id, font_size="huge")
    with pytest.raises(ToolError):
        S.ass_add_style("Bad", doc_id, alignment=99)
    with pytest.raises(ToolError):
        S.ass_add_style("Bad", doc_id, primary_colour="#GGGGGG")
    assert "Style: Bad" not in ws.get(doc_id).to_text()


def test_add_style_accepts_colour_forms_and_matches_the_document(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_add_style("Col", doc_id, primary_colour="#00FF00",
                             secondary_colour=255, outline_colour="&HFF7F00")
    style = result["style"]
    # HTML and decimal input are converted, but following this document's
    # existing spelling (8 hex digits, no trailing &).
    assert style["PrimaryColour"] == "&H0000FF00"
    assert style["SecondaryColour"] == "&H000000FF"
    # An ASS colour is kept verbatim.
    assert style["OutlineColour"] == "&HFF7F00"


# --------------------------------------------------------------------------
# ass_update_style
# --------------------------------------------------------------------------


def test_update_style_changes_only_the_given_fields(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    before = S.ass_get_style("Default", doc_id)["style"]

    result = S.ass_update_style("Default", doc_id, font_size=60.0)

    assert result["changed"] == {"Fontsize": {"from": "48", "to": "60"}}
    after = result["style"]
    for field in STYLE_FIELDS:
        if field == "Fontsize":
            continue
        assert after[field] == before[field], field
    assert result["lines_updated"] == 0
    assert as_json(result)


def test_update_style_accepts_ass_column_spellings(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_update_style("Default", doc_id, MarginV=25, MarginL="30", Fontname="Tahoma")

    assert result["style"]["MarginV"] == "25"
    assert result["style"]["MarginL"] == "30"
    assert result["style"]["Fontname"] == "Tahoma"
    assert set(result["changed"]) == {"MarginV", "MarginL", "Fontname"}

    line = next(l for l in ws.get(doc_id).to_text().splitlines() if l.startswith("Style: Default"))
    fields = line.split(": ", 1)[1].split(",")
    assert len(fields) == len(STYLE_FIELDS)
    assert fields[STYLE_FIELDS.index("MarginV")] == "25"
    assert fields[STYLE_FIELDS.index("MarginL")] == "30"
    assert fields[STYLE_FIELDS.index("Fontname")] == "Tahoma"


def test_update_style_accepts_ssa_only_columns_when_present(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ssa")
    result = S.ass_update_style("Default", doc_id, TertiaryColour="&H00FF00FF", AlphaLevel=0)
    assert result["style"]["TertiaryColour"] == "&H00FF00FF"
    assert result["ignored_fields"] == {}


def test_update_style_reports_columns_the_format_cannot_store(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ssa")
    result = S.ass_update_style("Default", doc_id, scale_x=150.0)
    assert result["ignored_fields"] == {"ScaleX": 150.0}
    # SSA v4.00 cannot store ScaleX at all, so the field reads back as absent.
    assert result["style"]["ScaleX"] is None


def test_update_style_converts_html_and_decimal_following_the_document(ws, tmp_path):
    hexdoc = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_update_style("Default", hexdoc, primary_colour="#00FF00")
    assert result["style"]["PrimaryColour"] == "&H0000FF00"

    dec_text = ass_text(
        script_info=[("ScriptType", "v4.00+"), ("PlayResX", "640"), ("PlayResY", "480")],
        styles=[style_record(Name="Dec", PrimaryColour="16777215")],
        events=[dialogue("0:00:01.00", "0:00:02.00", "Dec", "hi")],
    )
    dec_id = open_text(ws, tmp_path, dec_text, name="decimal.ass")
    result2 = S.ass_update_style("Dec", dec_id, primary_colour="#00FF00")
    # The document spells colours as decimal integers; never invent hex here.
    assert result2["style"]["PrimaryColour"] == "65280"
    assert "&H" not in result2["style"]["PrimaryColour"]


def test_update_style_keeps_ass_colours_verbatim(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_update_style("Default", doc_id, primary_colour="&H7F00FF00&")
    assert result["style"]["PrimaryColour"] == "&H7F00FF00&"


def test_update_style_can_rename_and_rewrites_lines(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_update_style("Default", doc_id, Name="Renamed")

    assert result["previous_name"] == "Default"
    assert result["name"] == "Renamed"
    assert result["lines_updated"] == 2
    assert [event.get("Style") for event in ws.get(doc_id).events()] == ["Renamed"] * 2
    assert S.ass_get_style("Renamed", doc_id)["style"]["Name"] == "Renamed"


def test_update_style_unknown_field_lists_the_known_ones(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError) as excinfo:
        S.ass_update_style("Default", doc_id, Nonsense=1)
    message = str(excinfo.value)
    assert "Nonsense" in message
    assert "Fontsize" in message and "MarginV" in message


def test_update_style_unknown_style_raises(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError):
        S.ass_update_style("Ghost", doc_id, font="X")
    assert "Style: Default" in ws.get(doc_id).to_text()


# --------------------------------------------------------------------------
# ass_remove_style / rename / copy / reorder
# --------------------------------------------------------------------------


def test_remove_style_refuses_while_lines_use_it(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError) as excinfo:
        S.ass_remove_style("Default", doc_id)
    assert "2 line" in str(excinfo.value)
    assert "reassign_to" in str(excinfo.value)
    assert "Style: Default" in ws.get(doc_id).to_text()


def test_remove_style_with_reassign_to_repoints_lines_first(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    S.ass_add_style("Other", doc_id)

    result = S.ass_remove_style("Default", doc_id, reassign_to="Other")

    assert result["removed"] is True
    assert result["reassign_to"] == "Other"
    assert result["reassigned"] == 2
    text = ws.get(doc_id).to_text()
    assert "Style: Default" not in text
    assert [event.get("Style") for event in ws.get(doc_id).events()] == ["Other"] * 2
    assert [s["Name"] for s in S.ass_list_styles(doc_id)["styles"]] == ["Other"]


def test_remove_style_rejects_unknown_reassign_target(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError) as excinfo:
        S.ass_remove_style("Default", doc_id, reassign_to="Nope")
    assert "Nope" in str(excinfo.value)
    assert "Style: Default" in ws.get(doc_id).to_text()


def test_remove_style_works_for_an_unused_style(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    S.ass_add_style("Spare", doc_id)
    result = S.ass_remove_style("Spare", doc_id)
    assert result["removed"] is True
    assert result["reassigned"] == 0
    assert "Style: Spare" not in ws.get(doc_id).to_text()


def test_rename_style_rewrites_the_style_fields_of_its_lines(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_rename_style("Default", "Main", doc_id)

    assert result["old_name"] == "Default"
    assert result["new_name"] == "Main"
    assert result["lines_updated"] == 2
    text = ws.get(doc_id).to_text()
    assert "Style: Default" not in text
    assert "Style: Main" in text
    assert all(event.get("Style") == "Main" for event in ws.get(doc_id).events())


def test_rename_style_can_leave_lines_alone(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_rename_style("Default", "Main", doc_id, update_lines=False)

    assert result["lines_updated"] == 0
    assert all(event.get("Style") == "Default" for event in ws.get(doc_id).events())
    validate = S.ass_validate(doc_id)
    assert any(i["code"] == "missing_style" for i in validate["issues"])


def test_rename_style_rejects_a_name_that_is_taken(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    S.ass_add_style("Other", doc_id)
    with pytest.raises(ToolError) as excinfo:
        S.ass_rename_style("Default", "Other", doc_id)
    assert "Other" in str(excinfo.value)


def test_copy_style_duplicates_every_field(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    S.ass_update_style("Default", doc_id, font_size=52.0, margin_v=99)

    result = S.ass_copy_style("Default", "Loud", doc_id)

    assert result["source"] == "Default"
    assert result["created"] is True
    source = S.ass_get_style("Default", doc_id)["style"]
    copy = result["style"]
    for field in STYLE_FIELDS[1:]:  # Name is asserted separately below
        assert copy[field] == source[field], field
    assert copy["Name"] == "Loud"


def test_copy_style_samples_the_source_not_its_own_defaults(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        styles=[style_record(Name="Src", Fontname="Georgia", Fontsize="51", Bold="-1")],
        events=[dialogue("0:00:01.00", "0:00:02.00", "Src", "hi")],
    ))
    copy = S.ass_copy_style("Src", "Dst", doc_id)["style"]
    assert copy["Fontname"] == "Georgia"
    assert copy["Fontsize"] == "51"
    assert copy["Bold"] == "-1"


def test_copy_style_requires_overwrite_for_an_existing_name(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    S.ass_copy_style("Default", "Dup", doc_id)
    with pytest.raises(ToolError) as excinfo:
        S.ass_copy_style("Default", "Dup", doc_id)
    assert "overwrite" in str(excinfo.value)

    S.ass_update_style("Default", doc_id, font="Comic Sans MS")
    result = S.ass_copy_style("Default", "Dup", doc_id, overwrite=True)
    assert result["created"] is False
    assert result["style"]["Fontname"] == "Comic Sans MS"


def test_reorder_styles_reorders_the_section(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    S.ass_add_style("A", doc_id)
    S.ass_add_style("B", doc_id)

    result = S.ass_reorder_styles(["B", "Default", "A"], doc_id)

    assert result["order"] == ["B", "Default", "A"]
    assert result["count"] == 3
    assert [s["Name"] for s in S.ass_list_styles(doc_id)["styles"]] == ["B", "Default", "A"]
    lines = [l for l in ws.get(doc_id).to_text().splitlines() if l.startswith("Style: ")]
    assert [l.split(",")[0].split(": ", 1)[1] for l in lines] == ["B", "Default", "A"]


def test_reorder_styles_rejects_a_mismatched_set(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    S.ass_add_style("A", doc_id)

    with pytest.raises(ToolError) as excinfo:
        S.ass_reorder_styles(["Default", "Nope"], doc_id)
    message = str(excinfo.value)
    assert "Nope" in message and "A" in message
    assert [s["Name"] for s in S.ass_list_styles(doc_id)["styles"]] == ["Default", "A"]

    with pytest.raises(ToolError):
        S.ass_reorder_styles([], doc_id)
    with pytest.raises(ToolError):
        S.ass_reorder_styles(["Default"], doc_id)


def test_reorder_styles_can_use_a_single_style_document(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_reorder_styles(["Default"], doc_id)
    assert result["order"] == ["Default"]


# --------------------------------------------------------------------------
# ass_style_usage / ass_style_for_line / ass_check_font_substitution
# --------------------------------------------------------------------------


def test_style_usage_counts_styles_actors_and_lists_unused(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        styles=[style_record(Name="Talk"), style_record(Name="NoteOnly"),
                style_record(Name="Unused")],
        events=[
            dialogue("0:00:01.00", "0:00:02.00", "Talk", "one", name="Alice"),
            dialogue("0:00:02.00", "0:00:03.00", "Talk", "two", name="Bob"),
            dialogue("0:00:03.00", "0:00:04.00", "Talk", "three", name="Alice"),
            dialogue("0:00:04.00", "0:00:05.00", "NoteOnly", "comment", kind="Comment"),
            dialogue("0:00:05.00", "0:00:06.00", "Ghost", "dangling"),
        ],
    ))
    result = S.ass_style_usage(doc_id)

    assert result["by"] == "style"
    assert result["lines"] == 5
    assert result["by_style"]["Talk"] == {"name": "Talk", "total": 3, "dialogue": 3, "comment": 0}
    assert result["by_style"]["NoteOnly"] == {"name": "NoteOnly", "total": 1, "dialogue": 0, "comment": 1}
    assert result["unused_styles"] == ["Unused"]
    assert result["unknown_styles"] == {"Ghost": 1}
    assert result["by_actor"]["Alice"]["total"] == 2
    assert result["by_actor"]["Bob"]["total"] == 1
    assert as_json(result)


def test_style_usage_by_actor_puts_actors_in_counts(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        styles=[style_record(Name="Body")],
        events=[dialogue("0:00:01.00", "0:00:02.00", "Body", "x", name="Ann")],
    ))
    result = S.ass_style_usage(doc_id, by="actor")
    assert result["counts"]["Ann"]["total"] == 1
    assert result["by_actor"] == result["counts"]
    assert result["by_style"]["Body"]["total"] == 1


def test_style_usage_rejects_an_unknown_axis(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError) as excinfo:
        S.ass_style_usage(doc_id, by="colour")
    assert "style" in str(excinfo.value) and "actor" in str(excinfo.value)


def test_style_usage_counts_an_empty_document(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(styles=[], events=[]))
    result = S.ass_style_usage(doc_id)
    assert result["lines"] == 0
    assert result["by_style"] == {}
    assert result["unused_styles"] == []


def test_style_for_line_applies_inline_override_tags(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        styles=[style_record(Name="Base", Fontname="Arial", Fontsize="40")],
        events=[dialogue("0:00:01.00", "0:00:05.00", "Base",
                         r"{\fnImpact\fs60\c&HFF0000&\b1\i1}big red{\r} plain{\rAlt}other",
                         name="Speaker")],
    ))
    S.ass_add_style("Alt", doc_id, font="Verdana", font_size=20.0)

    result = S.ass_style_for_line(0, doc_id)

    assert result["style"] == "Base"
    assert result["style_found"] is True
    assert result["text"].startswith("{\\fnImpact")
    assert result["plain_text"] == "big red plainother"
    assert result["warnings"] == []

    first = result["runs"][0]
    assert first["values"]["font"] == "Impact"
    assert first["values"]["font_size"] == 60.0
    assert first["values"]["primary_colour"] == "&HFF0000&"
    assert bool(first["values"]["bold"]) is True
    assert bool(first["values"]["italic"]) is True
    assert set(first["sources"]) >= {"font", "font_size", "primary_colour", "bold", "italic"}

    resolved = result["resolved"]
    assert resolved["font"] == "Impact"
    assert resolved["font_size"] == 60.0
    assert set(result["overrides"]) >= {"font", "font_size", "primary_colour", "bold", "italic"}

    # {\r} resets to the line's style, {\rAlt} switches to the Alt style.
    last = result["runs"][-1]
    assert last["values"]["font"] == "Verdana"
    assert last["values"]["font_size"] == 20.0
    assert "reset" in result["overrides"]
    assert as_json(result)


def test_style_for_line_reports_transforms_it_cannot_resolve(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        styles=[style_record(Name="Base")],
        events=[dialogue("0:00:01.00", "0:00:02.00", "Base",
                         r"{\t(0,500,\fs80)}growing{\i1} italic")],
    ))
    result = S.ass_style_for_line(0, doc_id)

    assert result["transforms"] == [r"\t(0,500,\fs80)"]
    assert "font_size" not in result["overrides"]
    assert bool(result["runs"][-1]["values"]["italic"]) is True


def test_style_for_line_without_tags_returns_the_style_values(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_style_for_line(0, doc_id)

    assert result["resolved"]["font"] == "Arial"
    assert result["resolved"]["font_size"] == 48.0
    assert result["resolved"]["primary_colour"] == "&H00FFFFFF"
    assert result["runs"][0]["sources"] == {}
    assert result["style_definition"]["Name"] == "Default"
    assert result["plain_text"] == "Hello world"


def test_style_for_line_handles_comment_lines_and_missing_styles(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "unknown-fields.ass")
    result = S.ass_style_for_line(0, doc_id)

    assert result["style"] == "Default"
    assert result["style_found"] is False
    assert result["style_definition"] is None
    assert result["warnings"], "a dangling style reference must warn"


def test_style_for_line_rejects_out_of_range_indices(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError):
        S.ass_style_for_line(99, doc_id)


def test_style_for_line_accepts_a_selection(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    ws.selection = [0]
    result = S.ass_style_for_line(doc_id=doc_id)
    assert result["index"] == 0


def test_check_font_substitution_flags_substituted_families(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        styles=[style_record(Name="Missing", Fontname="Definitely Not A Font 12345"),
                style_record(Name="Mono", Fontname="DejaVu Sans Mono")],
        events=[dialogue("0:00:01.00", "0:00:02.00", "Missing", "x")],
    ))
    result = S.ass_check_font_substitution(doc_id)

    assert result["checked"] == 2
    if not result["fontconfig"]:
        pytest.skip("fontconfig is unavailable on this host")
    by_name = {entry["name"]: entry for entry in result["styles"]}
    assert by_name["Missing"]["substituted"] is True
    assert by_name["Missing"]["resolved"] != "Definitely Not A Font 12345"
    assert "Missing" in result["substitutions"]
    assert by_name["Mono"]["requested"] == "DejaVu Sans Mono"
    assert as_json(result)


def test_check_font_substitution_without_styles(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "unknown-fields.ass")
    result = S.ass_check_font_substitution(doc_id)
    assert result["checked"] == 0
    assert result["styles"] == []
    assert result["substitutions"] == []


# --------------------------------------------------------------------------
# script info
# --------------------------------------------------------------------------


def test_get_script_info_preserves_file_order_and_raw_text(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "unknown-fields.ass")
    result = S.ass_get_script_info(doc_id)

    assert [item["key"] for item in result["items"]] == ["Title", "FutureField"]
    assert result["ordered"] == [["Title", "spacing preserved"], ["FutureField", "value"]]
    # "Title : spacing preserved" keeps its odd spacing in raw form.
    assert result["items"][0]["raw"] == "Title : spacing preserved"
    assert result["duplicates"] == []
    assert as_json(result)


def test_get_script_info_flags_duplicate_keys(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        script_info=[("Title", "one"), ("Title", "two"), ("PlayResX", "640")],
    ))
    result = S.ass_get_script_info(doc_id)
    assert result["duplicates"] == ["Title"]
    assert [item["duplicate"] for item in result["items"]] == [False, True, False]


def test_set_script_info_updates_in_place_and_appends_new_keys(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")

    updated = S.ass_set_script_info("Title", "New title", doc_id)
    assert updated["created"] is False
    assert updated["index"] == 0
    assert ws.get(doc_id).to_text().splitlines()[1] == "Title: New title"

    created = S.ass_set_script_info("WrapStyle", "0", doc_id)
    assert created["created"] is True
    info = S.ass_get_script_info(doc_id)
    assert [item["key"] for item in info["items"]] == [
        "Title", "ScriptType", "PlayResX", "PlayResY", "WrapStyle",
    ]
    assert info["values"]["WrapStyle"] == "0"


def test_set_script_info_before_positions_a_new_key(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_set_script_info("Credits", "me", doc_id, before="PlayResY")

    assert result["created"] is True
    assert [item["key"] for item in S.ass_get_script_info(doc_id)["items"]] == [
        "Title", "ScriptType", "PlayResX", "Credits", "PlayResY",
    ]

    with pytest.raises(ToolError):
        S.ass_set_script_info("X", "1", doc_id, before="NoSuchKey")


def test_set_script_info_validates_its_arguments(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError):
        S.ass_set_script_info("", "value", doc_id)
    with pytest.raises(ToolError):
        S.ass_set_script_info("Key:with:colons", "value", doc_id)
    with pytest.raises(ToolError):
        S.ass_set_script_info("Key", "with\nnewline", doc_id)


def test_remove_script_info(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_remove_script_info("PlayResX", doc_id)

    assert result["removed"] is True
    assert [item["key"] for item in S.ass_get_script_info(doc_id)["items"]] == [
        "Title", "ScriptType", "PlayResY",
    ]
    assert S.ass_remove_script_info("PlayResX", doc_id)["removed"] is False


def test_set_play_res(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    assert ws.get(doc_id).play_res == (1280, 720)

    result = S.ass_set_play_res(1920, 1080, doc_id)

    assert (result["play_res_x"], result["play_res_y"]) == (1920, 1080)
    assert (result["previous_x"], result["previous_y"]) == (1280, 720)
    assert ws.get(doc_id).play_res == (1920, 1080)
    info = S.ass_get_script_info(doc_id)["values"]
    assert info["PlayResX"] == "1920" and info["PlayResY"] == "1080"

    with pytest.raises(ToolError):
        S.ass_set_play_res(0, 720, doc_id)
    with pytest.raises(ToolError):
        S.ass_set_play_res(1920, -5, doc_id)


@pytest.mark.parametrize("code,name", [
    (0, "smart"), (1, "end of line"), (2, "no wrapping"), (3, "bottom of line only"),
])
def test_set_wrap_style_numeric_codes(ws, tmp_path, code, name):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    result = S.ass_set_wrap_style(code, doc_id)
    assert result["wrap_style"] == code
    assert result["name"] == name
    assert f"WrapStyle: {code}" in ws.get(doc_id).to_text()
    assert ws.get(doc_id).wrapping == code


def test_set_wrap_style_accepts_names_and_rejects_garbage(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    assert S.ass_set_wrap_style("smart", doc_id)["wrap_style"] == 0
    assert S.ass_set_wrap_style("END OF LINE", doc_id)["wrap_style"] == 1
    assert S.ass_set_wrap_style("no wrapping", doc_id)["wrap_style"] == 2

    before = ws.get(doc_id).to_text()
    for bad in (9, -1, "sideways", True, None):
        with pytest.raises(ToolError):
            S.ass_set_wrap_style(bad, doc_id)
    assert ws.get(doc_id).to_text() == before


def test_set_scaled_border_and_shadow_round_trips_yes_no(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")

    result = S.ass_set_scaled_border_and_shadow(True, doc_id)
    assert result["enabled"] is True
    assert result["value"] == "yes"
    assert "ScaledBorderAndShadow: yes" in ws.get(doc_id).to_text()

    result2 = S.ass_set_scaled_border_and_shadow(False, doc_id)
    assert result2["enabled"] is False
    assert result2["value"] == "no"
    assert result2["previous"] == "yes"
    assert ws.get(doc_id).to_text().count("ScaledBorderAndShadow") == 1

    assert S.ass_set_scaled_border_and_shadow("yes", doc_id)["enabled"] is True
    with pytest.raises(ToolError):
        S.ass_set_scaled_border_and_shadow("maybe", doc_id)


def test_set_timing_info_sets_and_removes_keys(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")

    result = S.ass_set_timing_info(fps=23.976, video_file="video.mp4",
                                   timecodes_file="tc.txt", doc_id=doc_id)
    assert result["fps"] == "23.976"
    assert result["video_file"] == "video.mp4"
    assert result["timecodes_file"] == "tc.txt"
    assert set(result["changes"]) == {"FPS", "Video File", "Timecodes File"}
    values = S.ass_get_script_info(doc_id)["values"]
    assert values["Video File"] == "video.mp4"
    assert values["Timecodes File"] == "tc.txt"

    removed = S.ass_set_timing_info(video_file="", doc_id=doc_id)
    assert removed["video_file"] is None
    assert "Video File" not in S.ass_get_script_info(doc_id)["values"]
    assert S.ass_get_script_info(doc_id)["values"]["FPS"] == "23.976"

    noop = S.ass_set_timing_info(doc_id=doc_id)
    assert noop["changes"] == {}


def test_set_timing_info_validates_fps(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    for bad in (0, -24, "abc"):
        with pytest.raises(ToolError):
            S.ass_set_timing_info(fps=bad, doc_id=doc_id)


# --------------------------------------------------------------------------
# attachments
# --------------------------------------------------------------------------


def test_list_attachments_reports_kind_size_and_magic_sniff(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "legacy-attachments.ass")
    result = S.ass_list_attachments(doc_id)

    assert result["count"] == 2
    by_name = {item["name"]: item for item in result["attachments"]}
    assert set(by_name) == {"Sample_0.ttf", "card.png"}

    font = by_name["Sample_0.ttf"]
    assert font["kind"] == "font"
    assert font["section"] == "fonts"
    assert font["data_lines"] == 2
    assert font["encoded_chars"] > 0
    assert font["size"] > 0
    # The fixture data is not real base64; the tool must say so, not guess.
    assert font["decoded"] is False
    assert font["error"]
    assert font["sniff"] == "unknown"
    assert font["sha256"] is None

    image = by_name["card.png"]
    assert image["kind"] == "image"
    assert image["section"] == "graphics"
    assert image["line_indices"][0] == 0
    assert as_json(result)


def test_list_attachments_from_the_real_attachment_fixture(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "legacy-attachments-real.ass")
    result = S.ass_list_attachments(doc_id)

    names = {item["name"]: item for item in result["attachments"]}
    assert set(names) == {"Sample_0.ttf", "card.png"}
    assert len(names["Sample_0.ttf"]["line_indices"]) > 50
    assert names["Sample_0.ttf"]["kind"] == "font"


def test_list_attachments_kind_filter_and_errors(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "legacy-attachments-real.ass")

    fonts = S.ass_list_attachments(doc_id, kind="fonts")
    assert [item["kind"] for item in fonts["attachments"]] == ["font"]
    assert S.ass_list_attachments(doc_id, kind="font")["count"] == fonts["count"]

    images = S.ass_list_attachments(doc_id, kind="graphics")
    assert [item["name"] for item in images["attachments"]] == ["card.png"]
    assert S.ass_list_attachments(doc_id, kind="image")["count"] == 1

    with pytest.raises(ToolError) as excinfo:
        S.ass_list_attachments(doc_id, kind="video")
    assert "video" in str(excinfo.value)


def test_add_attachment_base64_encodes_and_sniffs(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    ttf = tmp_path / "MyFont.ttf"
    payload = b"\x00\x01\x00\x00" + bytes(range(256)) * 3
    ttf.write_bytes(payload)

    result = S.ass_add_attachment(ttf, doc_id)

    assert result["name"] == "MyFont.ttf"
    assert result["kind"] == "font"
    assert result["section"] == "[Fonts]"
    assert result["bytes"] == len(payload)
    assert result["sniff"] == "font"
    assert result["replaced"] is False
    assert result["base64_lines"] > 1

    text = ws.get(doc_id).to_text()
    assert "[Fonts]" in text
    assert "fontname: MyFont.ttf" in text
    encoded_lines = [l for l in text.splitlines() if l.startswith("fontname: MyFont.ttf")]
    assert len(encoded_lines) == 1
    listed = S.ass_list_attachments(doc_id)["attachments"][0]
    assert listed["size"] == len(payload)
    assert listed["decoded"] is True
    assert listed["sniff"] == "font"
    assert as_json(result)


def test_add_attachment_sniffs_images_and_respects_explicit_kind(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    png = tmp_path / "logo.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)

    sniffed = S.ass_add_attachment(png, doc_id)
    assert sniffed["kind"] == "image"
    assert sniffed["section"] == "[Graphics]"
    assert "filename: logo.png" in ws.get(doc_id).to_text()

    forced = S.ass_add_attachment(png, doc_id, name="odd.bin", kind="font")
    assert forced["name"] == "odd.bin"
    assert forced["kind"] == "font"
    assert forced["section"] == "[Fonts]"
    assert "fontname: odd.bin" in ws.get(doc_id).to_text()

    reused = S.ass_add_attachment(png, doc_id, kind="image")
    assert reused["replaced"] is True


def test_add_attachment_errors(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    missing = tmp_path / "nope.ttf"
    with pytest.raises(ToolError) as excinfo:
        S.ass_add_attachment(missing, doc_id)
    assert "not found" in str(excinfo.value)

    target = tmp_path / "dir"
    target.mkdir()
    with pytest.raises(ToolError):
        S.ass_add_attachment(target, doc_id)

    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError):
        S.ass_add_attachment(blob, doc_id, kind="video")
    with pytest.raises(ToolError):
        S.ass_add_attachment(blob, doc_id, name="bad/name.bin")
    assert "[Fonts]" not in ws.get(doc_id).to_text()


def test_extract_attachment_writes_bytes_and_sha256(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    ttf = tmp_path / "Round.ttf"
    payload = b"\x00\x01\x00\x00" + bytes(range(200)) * 2
    ttf.write_bytes(payload)
    S.ass_add_attachment(ttf, doc_id)

    result = S.ass_extract_attachment("Round.ttf", doc_id)

    assert result["name"] == "Round.ttf"
    assert result["bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    written = Path(result["path"])
    assert written.is_file()
    assert ws.output_dir in written.parents
    assert written.read_bytes() == payload

    listed = S.ass_list_attachments(doc_id)["attachments"][0]
    assert listed["sha256"] == result["sha256"]
    assert listed["decoded"] is True
    assert as_json(result)


def test_extract_attachment_honours_an_explicit_output_path(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    png = tmp_path / "pic.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x01" * 10)
    S.ass_add_attachment(png, doc_id)

    result = S.ass_extract_attachment("pic.png", doc_id, output_path="nested/pic-out.png")
    path = Path(result["path"])
    assert path.name == "pic-out.png"
    assert path.parent.name == "nested"
    assert ws.output_dir in path.parents
    assert path.read_bytes() == png.read_bytes()

    outside = tmp_path / "absolute.bin"
    result2 = S.ass_extract_attachment("pic.png", doc_id, output_path=str(outside))
    assert Path(result2["path"]).read_bytes() == png.read_bytes()


def test_extract_attachment_rejects_undecodable_data(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "legacy-attachments.ass")
    with pytest.raises(ToolError) as excinfo:
        S.ass_extract_attachment("Sample_0.ttf", doc_id)
    assert "base64" in str(excinfo.value)
    with pytest.raises(ToolError):
        S.ass_extract_attachment("Nothing.ttf", doc_id)


def test_remove_attachment(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "legacy-attachments.ass")
    result = S.ass_remove_attachment("card.png", doc_id)

    assert result["removed"] is True
    assert result["lines_removed"] >= 2
    text = ws.get(doc_id).to_text()
    assert "card.png" not in text
    assert [item["name"] for item in S.ass_list_attachments(doc_id)["attachments"]] == [
        "Sample_0.ttf",
    ]

    assert S.ass_remove_attachment("card.png", doc_id)["removed"] is False


def test_attachment_round_trip_survives_save_and_reload(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    data = b"\x00\x01\x00\x00binary-font-payload\xff\xfe"
    (tmp_path / "Saved.ttf").write_bytes(data)
    S.ass_add_attachment(tmp_path / "Saved.ttf", doc_id)

    out = tmp_path / "out" / "with-attachment.ass"
    ws.save(doc_id, out)
    reopened = ws.open(str(out), doc_id="reloaded")

    result = S.ass_extract_attachment("Saved.ttf", reopened, output_path="from-reload.bin")
    assert Path(result["path"]).read_bytes() == data
    assert result["sha256"] == hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# extradata
# --------------------------------------------------------------------------


def test_extradata_set_update_list_remove(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")

    created = S.ass_set_extradata("com.example.clip", "a,b,c", doc_id)
    assert created == {"doc_id": doc_id, "id": "com.example.clip", "value": "a,b,c",
                       "removed": False, "created": True}

    listing = S.ass_list_extradata(doc_id)
    assert listing["count"] == 1
    entry = listing["entries"][0]
    assert entry["id"] == "com.example.clip"
    assert entry["value"] == "a,b,c"
    assert entry["head"] == "Comment"
    assert "[Aegisub Extradata]" in ws.get(doc_id).to_text()

    updated = S.ass_set_extradata("com.example.clip", "updated", doc_id)
    assert updated["created"] is False
    assert S.ass_list_extradata(doc_id)["count"] == 1
    assert S.ass_list_extradata(doc_id)["entries"][0]["value"] == "updated"

    removed = S.ass_set_extradata("com.example.clip", "", doc_id, remove=True)
    assert removed["removed"] is True
    assert S.ass_list_extradata(doc_id)["count"] == 0
    assert as_json(created)


def test_extradata_values_with_commas_survive_a_reload(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    S.ass_set_extradata("com.example.clip", "a,b,c", doc_id)

    out = tmp_path / "out" / "extradata.ass"
    ws.save(doc_id, out)
    reopened = ws.open(str(out), doc_id="reloaded-extradata")

    entry = S.ass_list_extradata(reopened)["entries"][0]
    assert entry["id"] == "com.example.clip"
    assert entry["value"] == "a,b,c"


def test_extradata_validates_arguments(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError):
        S.ass_set_extradata("", "value", doc_id)
    with pytest.raises(ToolError):
        S.ass_set_extradata("has,comma", "value", doc_id)
    with pytest.raises(ToolError):
        S.ass_set_extradata("ok", "multi\nline", doc_id)
    with pytest.raises(ToolError):
        S.ass_set_extradata("missing", "", doc_id)


def test_list_extradata_on_a_document_without_extradata(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "legacy-attachments-real.ass")
    result = S.ass_list_extradata(doc_id)
    assert result["count"] == 0
    assert result["entries"] == []


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_validate_reports_every_issue_class(ws, tmp_path):
    text = ass_text(
        script_info=[("Title", "dup"), ("Title", "again")],  # duplicate key, no ScriptType
        styles=[style_record(Name="Talk"), style_record(Name="talk"),
                style_record(Name="NoteOnly"), style_record(Name="Unused")],
        events=[
            dialogue("0:00:01.00", "0:00:02.00", "Talk", "hello"),
            dialogue("0:00:05.00", "0:00:03.00", "Talk", "backwards"),
            dialogue("0:00:06.00", "0:00:06.00", "Talk", "zero"),
            dialogue("0:00:07.00", "0:00:08.00", "Missing", "dangling"),
            dialogue("0:00:08.00", "0:00:09.00", "NoteOnly", "commented", kind="Comment"),
            "Dialogue: 0,0:00:09.00,0:00:10.00,NoStyle,oops",
            "This line has no colon at all",
        ],
        extra="[Strange Section]\nArbitrary: keep me\n",
    )
    doc_id = open_text(ws, tmp_path, text)

    result = S.ass_validate(doc_id)
    codes = {issue["code"] for issue in result["issues"]}

    assert {
        "duplicate_style_name",       # Talk / talk
        "duplicate_script_info_key",  # Title twice
        "missing_script_type",        # no ScriptType line
        "missing_style",              # the line that names Missing / NoStyle
        "end_before_start",           # 0:00:05.00 -> 0:00:03.00
        "zero_duration",              # 0:00:06.00 -> 0:00:06.00
        "comments_only_style",        # NoteOnly used by a Comment only
        "unknown_section",            # [Strange Section]
        "malformed_raw_line",         # no colon at all
        "field_count_mismatch",       # missing Text field
    } <= codes
    # The comment only style is the one the validator has to single out.
    flagged = {i["message"] for i in result["issues"] if i["code"] == "comments_only_style"}
    assert any("NoteOnly" in message for message in flagged), flagged
    missing = {i["message"] for i in result["issues"] if i["code"] == "missing_style"}
    assert any("Missing" in message for message in missing), missing
    assert result["ok"] is False
    assert result["counts"]["error"] > 0
    assert as_json(result)


def test_validate_issue_shape_and_severities(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "malformed-unknown.ssa")
    result = S.ass_validate(doc_id)

    assert result["issues"], "the malformed fixture must produce issues"
    for issue in result["issues"]:
        assert set(issue) == {"code", "severity", "kind", "index", "message"}
        assert issue["severity"] in {"error", "warning"}
        assert isinstance(issue["message"], str) and issue["message"]
        assert issue["index"] is None or isinstance(issue["index"], int)
    assert result["counts"]["total"] == len(result["issues"])
    assert as_json(result)


@pytest.mark.parametrize("fixture,expected", [
    ("basic.ass", {"unknown_section"}),
    ("basic.ssa", set()),
    ("custom-format.ssa", set()),
    ("fallback-formats.ssa", set()),
    ("bom-crlf-no-final.ssa", set()),
    ("legacy-attachments-real.ass", set()),
    ("legacy-attachments.ass", {"missing_style"}),
    ("redefined-formats.ssa", {"comments_only_style"}),
    ("malformed-unknown.ssa", {"field_count_mismatch", "malformed_line", "missing_style"}),
    ("recoverable.ass", {"invalid_timestamp", "malformed_line", "missing_script_type"}),
    ("unknown-fields.ass", {"missing_script_type", "missing_style"}),
])
def test_validate_on_real_fixtures(ws, tmp_path, fixture, expected):
    doc_id = open_fixture(ws, tmp_path, fixture)
    result = S.ass_validate(doc_id)
    codes = {issue["code"] for issue in result["issues"]}
    assert expected <= codes, f"{fixture}: got {sorted(codes)}"
    if not expected:
        assert result["counts"]["error"] == 0
        assert result["ok"] is True


def test_validate_clean_document_is_ok(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        styles=[style_record(Name="Body")],
        events=[dialogue("0:00:01.00", "0:00:02.00", "Body", "hello")],
    ))
    result = S.ass_validate(doc_id)
    assert result["issues"] == []
    assert result["ok"] is True
    assert result["counts"] == {"error": 0, "warning": 0, "total": 0}


def test_validate_flags_field_count_mismatch_against_the_format(ws, tmp_path):
    doc_id = open_text(ws, tmp_path, ass_text(
        styles=[style_record(Name="Body")],
        events=[
            dialogue("0:00:01.00", "0:00:02.00", "Body", "good one"),
            "Dialogue: 0,0:00:02.00,0:00:03.00,Body",
        ],
    ))
    issues = S.ass_validate(doc_id)["issues"]
    mismatch = [i for i in issues if i["code"] == "field_count_mismatch"]
    assert mismatch, [i["code"] for i in issues]
    assert mismatch[0]["index"] == 1


# --------------------------------------------------------------------------
# cross cutting: undo, doc ids, JSON
# --------------------------------------------------------------------------


def test_mutations_can_be_undone_and_redone_through_the_tools(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    original = ws.get(doc_id).to_text()

    S.ass_add_style("Temp", doc_id)
    S.ass_set_script_info("Title", "changed", doc_id)
    assert "Style: Temp" in ws.get(doc_id).to_text()

    assert ws.undo(doc_id) is True
    assert "Title: changed" not in ws.get(doc_id).to_text()
    assert ws.undo(doc_id) is True
    assert ws.get(doc_id).to_text() == original

    assert ws.redo(doc_id) is True
    assert "Style: Temp" in ws.get(doc_id).to_text()


def test_every_tool_result_is_json_serialisable(ws, tmp_path):
    doc_id = open_fixture(ws, tmp_path, "basic.ass")
    att_id = open_fixture(ws, tmp_path, "legacy-attachments.ass", as_name="att.ass")
    blob = tmp_path / "blob.ttf"
    blob.write_bytes(b"\x00\x01\x00\x00payload")

    results = [
        S.ass_list_styles(doc_id),
        S.ass_list_styles(doc_id, include_usage=True, include_bbox=True),
        S.ass_get_style("Default", doc_id),
        S.ass_get_style("Default", doc_id, include_glyphs=True),
        S.ass_add_style("Json", doc_id),
        S.ass_update_style("Json", doc_id, font_size=22.0),
        S.ass_copy_style("Json", "Json2", doc_id),
        S.ass_reorder_styles(["Default", "Json", "Json2"], doc_id),
        S.ass_style_usage(doc_id),
        S.ass_style_for_line(0, doc_id),
        S.ass_check_font_substitution(doc_id),
        S.ass_get_script_info(doc_id),
        S.ass_set_script_info("WrapStyle", "0", doc_id),
        S.ass_set_play_res(800, 600, doc_id),
        S.ass_set_wrap_style(3, doc_id),
        S.ass_set_scaled_border_and_shadow(True, doc_id),
        S.ass_set_timing_info(fps=25.0, doc_id=doc_id),
        S.ass_list_attachments(att_id),
        S.ass_list_attachments(doc_id, kind="font"),
        S.ass_add_attachment(blob, doc_id),
        S.ass_extract_attachment("blob.ttf", doc_id),
        S.ass_list_extradata(doc_id),
        S.ass_set_extradata("com.example.json", "1", doc_id),
        S.ass_validate(doc_id),
        S.ass_rename_style("Json2", "Json3", doc_id),
        S.ass_remove_attachment("blob.ttf", doc_id),
        S.ass_remove_style("Json3", doc_id),
        S.ass_remove_script_info("WrapStyle", doc_id),
    ]

    for result in results:
        encoded = as_json(result)
        assert encoded.startswith("{")
        assert json.loads(encoded) == json.loads(as_json(result))


def test_tools_require_an_open_document(ws, tmp_path):
    for doc_id in list(ws.ids()):
        ws.close(doc_id)
    with pytest.raises(ToolError) as excinfo:
        S.ass_list_styles()
    assert "no document" in str(excinfo.value)


def test_tools_accept_an_explicit_doc_id_among_several(ws, tmp_path):
    first = open_fixture(ws, tmp_path, "basic.ass", as_name="one.ass")
    second = open_fixture(ws, tmp_path, "basic.ssa", as_name="two.ssa")

    styles_first = S.ass_list_styles(first)["styles"][0]
    styles_second = S.ass_list_styles(second)["styles"][0]
    assert styles_first["Fontsize"] == "48"
    assert styles_second["Fontsize"] == "24"
    assert S.ass_list_styles(doc_id=second)["styles"][0]["Fontsize"] == "24"


def test_unknown_document_id_raises(ws, tmp_path):
    open_fixture(ws, tmp_path, "basic.ass")
    with pytest.raises(ToolError) as excinfo:
        S.ass_list_styles("nope.ass@99")
    assert "nope.ass@99" in str(excinfo.value)


def test_fixtures_are_untouched_by_the_test_suite():
    """Guard: the suite must never mutate the checked in fixtures."""
    before = {name: (REAL / name).read_bytes() for name in FIXTURES}
    for name, payload in before.items():
        assert payload, name
    assert set(before) == set(FIXTURES)
