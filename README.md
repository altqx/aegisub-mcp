# aegisub-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server for Aegisub / ASS
subtitles. It exposes 138 tools that read and edit subtitle documents, run real
Aegisub Automation 4 scripts (Lua) against them, and bridge a running Aegisub instance
to the MCP client over a shared directory — plus a libass-backed verification layer, so
an MCP client (Hermes, Claude Desktop, Codex, …) can do real subtitle work instead of
text munging.

Work happens on **open documents** held by the server: call `ass_open` (path) or
`ass_new_document` first, then address the returned `doc_id` with the other tools.

## Requirements

- Python >= 3.10 (`requires-python` in `pyproject.toml`)
- `mcp >= 2.2` (MCP SDK) and `lupa >= 2.0` (Lua automations) — installed automatically.
  The 2.x line is required: `mcp.server.mcpserver.MCPServer` (what `server.py` imports)
  does not exist in 1.x, which exposes only `FastMCP` / `Server`.
- Optional: `fonttools` + `uharfbuzz` for the font/glyph metrics tools (`pip install -e '.[metrics]'`)
- Optional: `pytest` + `pytest-timeout` for development (`pip install -e '.[dev]'`)
- For the live bridge: Aegisub itself (3.4.x tested) needs no Python at all — it runs the
  generated Lua. `xdotool`/`ydotool` is optional and only used by `ass_bridge_inject`.

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

Two entry points over one tool surface: `aegisub-mcp` (stdio) and `aegisub-mcp-http`
(streamable HTTP, the transport that serves protocol `2026-07-28`).

### stdio — the default

`aegisub-mcp` speaks MCP over **stdio**; stdout carries JSON-RPC framing and nothing
else, all diagnostics go to stderr.

Over stdio the server negotiates protocol revision **2025-11-25** — the newest revision
reachable through the `initialize` handshake. Revision `2026-07-28` is the *stateless*
per-request revision (no handshake, no session; carried by the `MCP-Protocol-Version`
header) and can only be served over HTTP, which this entrypoint does not carry: use
`aegisub-mcp-http` below. A client that asks stdio for `2026-07-28` is counter-offered
`2025-11-25`. Verify with any client, or by hand:

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2026-07-28","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  | PYTHONPATH=src .venv/bin/python -m aegisub_mcp | head -1
# -> "protocolVersion":"2025-11-25"
```

```bash
aegisub-mcp                                  # console script, once installed
PYTHONPATH=src python -m aegisub_mcp         # from a checkout
```

### Streamable HTTP — protocol 2026-07-28

`aegisub-mcp-http` serves the same tools at `POST /mcp`. Because `2026-07-28` is
stateless, each request is self-contained: the revision and the client capabilities
ride in the request's `_meta` envelope and its `MCP-Protocol-Version` / `Mcp-Method` /
`Mcp-Name` headers, there is no `initialize` and no session id. The handshake
replacement is `server/discover`.

Legacy clients that send no `MCP-Protocol-Version` header (or a handshake revision) are
served exactly as before on the same URL, so one endpoint answers both eras. Open
documents live in the server process, not in a session — a document opened by one POST
is still open for the next one.

```bash
aegisub-mcp-http --host 127.0.0.1 --port 8000                             # installed
PYTHONPATH=src .venv/bin/python -m aegisub_mcp.http_server --port 8000     # checkout
```

It binds to loopback by default. `--help` lists `--path`, `--json-response`,
`--stateless`, and `--allow-host` / `--allow-origin`: the MCP SDK's DNS-rebinding
protection is enabled automatically for loopback binds, and binding anywhere else turns
it on only when you name the allowed hosts (state both, or every request is refused with
`Invalid Host header`).

Verify the modern path by hand — one POST, no handshake:

```bash
curl -sS http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: server/discover' \
  -d '{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}'
# -> "supportedVersions":["2026-07-28"], "resultType":"complete", and no Mcp-Session-Id
```

A tool call is the same shape with `Mcp-Method: tools/call`, `Mcp-Name: <tool>`, and the
usual `params.name` / `params.arguments`:

```bash
curl -sS http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'MCP-Protocol-Version: 2026-07-28' \
  -H 'Mcp-Method: tools/call' -H 'Mcp-Name: ass_open' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ass_open","arguments":{"path":"/path/to/file.ass"},"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}'
```

A typical MCP client entry (stdio):

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

Tools are registered from eight modules under `src/aegisub_mcp/tools/` (138 in total);
every tool name starts with the `ass_` prefix.

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
- **automation_tools** (7) — run real Aegisub Automation 4 Lua: `ass_automation_dirs`,
  `ass_automation_list`, `ass_automation_info`, `ass_automation_run_macro`,
  `ass_automation_run_filter`, `ass_automation_run_filters`, `ass_automation_from_source`
- **bridge_tools** (10) — talk to a running Aegisub: `ass_bridge_publish`,
  `ass_bridge_pull`, `ass_bridge_status`, `ass_bridge_live`, `ass_bridge_events`,
  `ass_bridge_watch`, `ass_bridge_autosave`, `ass_bridge_install`, `ass_bridge_paths`,
  `ass_bridge_inject`

## Driving Aegisub itself

The document tools work on files; the last two modules work on *Aegisub*.

### Automation 4

`ass_automation_run_macro`, `ass_automation_run_filter(s)` and
`ass_automation_from_source` execute genuine Automation 4 Lua against the open document
with the API surface Aegisub provides (`subs`, `aegisub.dialog.display`,
`aegisub.set_undo_point`, progress, …) — no reimplementation of the subtitle model, the
same host Aegisub uses. `ass_automation_dirs` / `ass_automation_list` /
`ass_automation_info` show what is installed in `?user/automation`. A macro that raises
inside Lua is reported as a failure, never as a half-applied edit: the undo snapshot and
the apply path are shared with the document layer.

### The live bridge

Aegisub has **no MCP client, no socket API and no inbound trigger hook**, so the bridge
uses a shared directory of small TSV/ASS files. That is what makes it portable to Linux,
Windows and macOS, Wayland included: nothing depends on owning a window or injecting
keystrokes.

```bash
aegisub-mcp-bridge install              # write the Aegisub-side script + hotkey
aegisub-mcp-bridge install --dry-run    # show the config diff first
aegisub-mcp-bridge status               # bridge dir, install state, autosave settings
```

Then restart Aegisub: autoload scripts load at startup. The script adds two macros to
the Automation menu — `krapau-bridge: Pull changes from MCP` (default hotkey
`Ctrl-Alt-M`) and `krapau-bridge: Push my document to MCP`:

- **MCP → Aegisub**: `ass_bridge_publish` queues a revision; the pull macro applies it
  to the open document, then writes an acknowledgement with a line count.
- **Aegisub → MCP**: the macro writes state/snapshot files; polling them turns file
  changes into events, and `ass_bridge_live` reports which artifact is current.

The bridge directory is `$AEGISUB_MCP_BRIDGE` when set (run `aegisub-mcp-bridge paths`
for the resolved default).

`ass_bridge_watch` is the realtime view. It streams the bridge for `seconds` and reports
what changed **while watching** as `events`; whatever had already happened before the
watch began comes back separately as `caught_up`, because the first poll only
establishes the baseline — it never ends the watch early. `stop_after=1` returns on the
next change, `kinds=[...]` follows one channel. Event kinds:

- `aegisub.state` — the macro ran; the document revision moved
- `aegisub.applied` — a published revision was applied or refused, with a line count
- `aegisub.snapshot` — the snapshot file changed
- `aegisub.source` — Aegisub saved the file the user is editing, in place
- `aegisub.autosave` — a new autosave copy appeared

A macro whose `validate()` returns false is **disabled** by Aegisub, so the pull hotkey
does nothing while no revision is pending. That is the designed behaviour, not a broken
key binding.

### Realtime without a daemon

Aegisub writes files in exactly two situations, and `install` sets up both:

- `App/Auto/Save on Every Change` rewrites the user's own file on every change — always
  current, but it overwrites their file, so it stays **opt-in**: `--save-on-every-change`.
- `App/Auto/Save` + `Save Every Seconds` (install default: 5) writes
  `<name>.<timestamp>.AUTOSAVE.ass` into `?user/autosave`. An interval of `0` switches
  the timer off completely, so `install` never sets 0.

`--no-autosave` leaves Aegisub's own settings alone. Either way the resulting file change
surfaces through `ass_bridge_watch` as `aegisub.source` / `aegisub.autosave`.
`ass_bridge_inject` presses the pull hotkey with `xdotool`/`ydotool` where one exists —
a convenience for triggering a macro, never the data path (X11-only, so it is not used
by the bridge itself).

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

The suite drives the tool layer directly (`tests/test_tools_*.py`), the stdio server end to
end (`tests/test_server_stdio.py`), and the HTTP entry point end to end
(`tests/test_http_server.py`, which boots the real server and speaks `2026-07-28` over the
wire), and uses the frozen files in `tests/fixtures/real/` as round-trip fixtures.

The Aegisub-facing half is tested the same way: `tests/test_automation_tools.py` runs
Automation 4 scripts through the Lua host, `tests/test_bridge_lua.py` renders and executes
the *generated* bridge script (the same file `install` writes into `?user`), and
`tests/test_bridge_tools.py` drives the MCP half against a bridge directory — install,
publish, apply, poll, watch and the autosave channel. Nothing is mocked: the Lua side is
the real script and the Python side is the real tool layer.

## License

MIT — see [LICENSE](LICENSE).
