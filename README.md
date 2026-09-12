# aegisub-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server for Aegisub / ASS
subtitles. It exposes 121 tools that read and edit subtitle documents, plus a
libass-backed verification layer, so an MCP client (Hermes, Claude Desktop, Codex, …)
can do real subtitle work instead of text munging.

Work happens on **open documents** held by the server: call `ass_open` (path) or
`ass_new_document` first, then address the returned `doc_id` with the other tools.

## Requirements

- Python >= 3.10 (`requires-python` in `pyproject.toml`)
- `mcp >= 1.2` (MCP SDK) and `lupa >= 2.0` (Lua automations) — installed automatically
- Optional: `fonttools` + `uharfbuzz` for the font/glyph metrics tools (`pip install -e '.[metrics]'`)
- Optional: `pytest` + `pytest-timeout` for development (`pip install -e '.[dev]'`)

## Install

Either install it into a virtualenv:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

or run it straight from a checkout without installing (the package lives under `src/`):

```bash
PYTHONPATH=src .venv/bin/python -m aegisub_mcp
```

## Running

`aegisub-mcp` speaks MCP over **stdio**; stdout carries JSON-RPC framing and nothing
else, all diagnostics go to stderr.

```bash
aegisub-mcp                                  # console script, once installed
PYTHONPATH=src python -m aegisub_mcp         # from a checkout
```

A typical MCP client entry:

```json
{
  "mcpServers": {
    "aegisub": {
      "command": "aegisub-mcp",
      "args": []
    }
  }
}
```

For an uninstalled checkout use `"command": "python"` with
`"args": ["-m", "aegisub_mcp"]` and `"env": {"PYTHONPATH": "/path/to/aegisub-mcp/src"}`.

## Tool areas

Tools are registered from six modules under `src/aegisub_mcp/tools/`; every tool name
starts with the `ass_` prefix.

- **lines** (30) — document lifecycle and event lines: `ass_open`, `ass_new_document`,
  `ass_save`, `ass_list_lines`, `ass_get_line`, `ass_update_line(s)`, `ass_add_line(s)`,
  `ass_delete_lines`, `ass_move_lines`, `ass_duplicate_lines`, `ass_merge_lines`,
  `ass_split_line`, `ass_sort_lines`, `ass_set_comment`, `ass_find_replace`,
  `ass_export_text`, `ass_import_srt`, `ass_select`, `ass_undo`/`ass_redo`, `ass_stats`, …
- **styles** (25) — script info, styles and attachments: `ass_get_script_info`,
  `ass_set_script_info`, `ass_set_play_res`, `ass_list_styles`, `ass_add_style`,
  `ass_update_style`, `ass_rename_style`, `ass_copy_style`, `ass_style_usage`,
  `ass_style_for_line`, `ass_add_attachment`, `ass_extract_attachment`, `ass_set_extradata`,
  `ass_validate`, …
- **timing** (20) — timing and QC: `ass_set_times`, `ass_shift_times`, `ass_scale_times`,
  `ass_snap_to_frames`, `ass_snap_to_keyframes`, `ass_align_to_silence`, `ass_fix_timing`,
  `ass_cps`, `ass_qc`, `ass_check_overlaps`, `ass_read_timecodes`/`ass_write_timecodes`, …
- **karaoke_tools** (13) — karaoke: `ass_karaoke_generate`, `ass_karaoke_set_timings`,
  `ass_karaoke_auto_timings`, `ass_karaoke_retime`, `ass_karaoke_shift`, `ass_karaoke_scale`,
  `ass_karaoke_split`, `ass_karaoke_export`, `ass_karaoke_get`, …
- **tags_tools** (13) — override tags and typesetting: `ass_parse_text`, `ass_plain_text`,
  `ass_strip_tags`, `ass_set_tag`, `ass_remove_tag`, `ass_insert_tag_at`,
  `ass_apply_tag_to_block`, `ass_wrap_range`, `ass_swap_an_pos`, `ass_add_typesetting`,
  `ass_tag_summary`, …
- **drawing_tools** (20) — vector drawings, clipping and fonts: `ass_get_drawing`,
  `ass_set_drawing`, `ass_drawing_bbox`, `ass_drawing_to_svg`, `ass_svg_to_drawing`,
  `ass_scale_drawing`, `ass_join_drawings`, `ass_split_drawing`, `ass_get_clips`,
  `ass_set_clip`, `ass_remove_clip`, `ass_glyph_check`, `ass_list_fonts`, `ass_match_font`, …

## Tool results

Every tool returns a JSON object. Expected user errors are raised as `ToolError` inside
the tool layer and reach the client as `{"error": "<message>"}`; unexpected exceptions
produce the same payload with the traceback logged to stderr.

## File fidelity

The document core is built for byte-faithful round-trips, because subtitle files in the
wild are messy:

- Re-saving a document that was only opened changes nothing — file encoding, BOM, CRLF
  line endings and a missing final newline all survive a round-trip.
- Lines read from a file keep their original spelling; only the fields a tool is asked to
  change are rewritten.
- ASS writes `Layer` as the first event field, SSA (`ScriptType: v4.00`) writes `Marked`;
  a new document declares the format of the script type it was created with.
- Lines built from scratch adopt the document's own `Dialogue:` / `Comment:` separator
  (`Dialogue: 0,…` in Aegisub output, `Dialogue:0,…` in compact files) instead of forcing
  one spelling.
- Every mutation takes an undo snapshot, so `ass_undo` restores the previous bytes.

## Output files

Tools that write standalone files (text exports, drawings, fonts) put them in
`$AEGISUB_MCP_OUT` when set, otherwise in `./aegisub-mcp-out`. That directory is
build output, not source, and is git-ignored.

## Development

```bash
.venv/bin/python -m pytest            # whole suite
.venv/bin/python -m pytest tests/test_tools_lines.py -v
```

The suite drives the tool layer directly (`tests/test_tools_*.py`) as well as the stdio
server end to end (`tests/test_server_stdio.py`), and uses the frozen files in
`tests/fixtures/real/` as round-trip fixtures.

## License

MIT — see [LICENSE](LICENSE).
