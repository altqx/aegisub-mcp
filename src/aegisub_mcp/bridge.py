"""File-based, cross-platform bridge between Aegisub and this MCP server.

Why a *file* bridge
-------------------
Aegisub ships no MCP client, no socket API, and no event loop that an
Automation 4 Lua script can hook: a macro or filter body runs **only** when the
user invokes it (menu entry, hotkey, "Apply filter"), and the only hooks
Automation 4 gives a script are the macro body and the ``register_macro``
validation callback.  What Aegisub *does* have, on every platform, is its own
per-user directory, and two of its built-in features write to it on their own:

* ``App/Auto/Save`` + ``App/Auto/Save on Every Change`` -- Aegisub writes a copy
  of the open document (unsaved in-memory edits included) into
  ``?user/autosave`` as ``<name>.<YYYY-MM-DD-HH-MM-SS>.AUTOSAVE.ass`` on every
  change.  That is a genuine realtime *Aegisub -> MCP* channel that needs no
  keystroke and works on Linux, Windows and macOS alike.
* ``?user/hotkey.json`` -- the hotkey table, so the MCP side can bind the
  bridge's pull macro for the user instead of asking them to click through
  preferences.

This module speaks a small, dependency-free TSV protocol in a directory both
sides know (the bridge directory): the Python side writes *requests* and reads
*state*, the Aegisub-side Lua script (``assets/krapau-bridge.lua.in``) writes
*state* and reads *requests*.  Nothing is binary, everything is UTF-8, and every
file is replaced atomically (write to ``*.tmp`` then ``os.replace``) so neither
side can ever read a half-written file.

Layout (in the bridge directory):

======================  ===============  ==================================================
``state.tsv``           Aegisub -> MCP   live document metadata + Aegisub's revision counter
``from-aegisub.ass``    Aegisub -> MCP   full snapshot of the live document (in-memory state)
``to-aegisub.tsv``      MCP -> Aegisub   pending event lines + this side's revision counter
``applied.tsv``         Aegisub -> MCP   ack of the last applied revision (ok/error)
``journal.jsonl``       MCP side         append-only log behind ``ass_bridge_events``
``mcp-revision.txt``    MCP side         last revision published by this side
``install.json``        MCP side         where the Lua script/config bits were installed
======================  ===============  ==================================================

Field names on the wire are **Aegisub's** Automation 4 names
(``layer``/``start_time``/``end_time``/``style``/``actor``/``margin_l``/
``margin_r``/``margin_t``/``effect``/``comment``/``text``) so the Lua side needs
no translation table and the protocol stays readable next to Aegisub's own docs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .asscore.document import AssDocument

BRIDGE_FORMAT = "krapau-bridge/1"
BRIDGE_ENV = "AEGISUB_MCP_BRIDGE"

STATE_NAME = "state.tsv"
SNAPSHOT_NAME = "from-aegisub.ass"
REQUEST_NAME = "to-aegisub.tsv"
APPLIED_NAME = "applied.tsv"
JOURNAL_NAME = "journal.jsonl"
REVISION_NAME = "mcp-revision.txt"
INSTALL_NAME = "install.json"
README_NAME = "README.txt"
# The poll fingerprint lives in its own file, not in the journal: ``poll()`` runs
# several times a second while watching, and four heartbeat rows a second would
# bury the real events and grow the journal without bound.
SEEN_NAME = "last-seen.json"

ASSET_NAME = "krapau-bridge.lua.in"
SCRIPT_NAME = "krapau-bridge.lua"
#: Aegisub builds a Lua macro's command id as
#: ``automation/lua/<script file stem>/<macro display name>``
#: (``src/auto4_lua.cpp``: ``cmd_name = "automation/lua/%s/%s"`` with the registry
#: ``filename`` stem and argument 1 of ``register_macro``).  The template names its
#: macros ``krapau-bridge: ...``, so the prefix is part of the command id -- bind
#: the hotkey without it and the keystroke does nothing at all.
SCRIPT_LABEL = Path(SCRIPT_NAME).stem
PULL_MACRO_NAME = f"{SCRIPT_LABEL}: Pull changes from MCP"
PUSH_MACRO_NAME = f"{SCRIPT_LABEL}: Push my document to MCP"
STATUS_MACRO_NAME = f"{SCRIPT_LABEL}: Bridge status"
PULL_COMMAND = f"automation/lua/{SCRIPT_LABEL}/{PULL_MACRO_NAME}"
DEFAULT_HOTKEY = "Ctrl-Alt-M"

#: Aegisub config keys the bridge cares about.  These are Aegisub *option names*
#: (``App/Auto/Save``); in ``config.json`` they are nested objects, i.e.
#: ``{"App": {"Auto": {"Save": ...}}}`` -- see :func:`get_option`/:func:`set_option`.
#: Verified against config.json written by Aegisub 3.4.2 on Linux; writing flat
#: ``"App/Auto/Save"`` keys instead leaves the real option untouched, so Aegisub
#: keeps autosaving on its 60 s timer and never on every change.
AUTOSAVE_KEYS = {
    "enabled": "App/Auto/Save",
    "interval": "App/Auto/Save Every Seconds",
    "on_change": "App/Auto/Save on Every Change",
    "path": "Path/Auto/Save",
}

#: Aegisub's automation search path (used by ``ass_bridge_paths``).
AUTOMATION_PATH_KEYS = {
    "base": "Path/Automation/Base",
    "include": "Path/Automation/Include",
    "autoload": "Path/Automation/Autoload",
}


def option_path(key: str) -> list[str]:
    """``App/Auto/Save`` (or ``App.Auto.Save``) -> ``["App", "Auto", "Save"]``."""
    return [part for part in key.replace(".", "/").split("/") if part]


def get_option(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read an Aegisub option out of a parsed ``config.json`` (nested form)."""
    node: Any = config
    for part in option_path(key):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def set_option(config: dict[str, Any], key: str, value: Any) -> None:
    """Write an Aegisub option into a parsed ``config.json`` (nested form)."""
    parts = option_path(key)
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def flat_remove(config: dict[str, Any], key: str) -> bool:
    """Delete a *flat* ``"App/Auto/Save"`` key (a bug this module used to write)."""
    if key in config:
        del config[key]
        return True
    return False

EVENT_FIELDS = ("layer", "start_time", "end_time", "style", "actor",
                "margin_l", "margin_r", "margin_t", "effect", "comment", "text")

# Keys of Bridge._fingerprint(); also the fields poll() reads back out of the
# last journal entry (journal entries are flat, so they cannot be nested).
FINGERPRINT_FIELDS = ("ae_rev", "applied_rev", "applied_result", "snapshot",
                      "autosave", "dirty", "event_count")

BRIDGE_README = """\
krapau-bridge -- file bridge between Aegisub and aegisub-mcp
===========================================================

Rewritten by both sides while they run; do not edit by hand.

  state.tsv          written by Aegisub   live document metadata + ae_rev
  from-aegisub.ass   written by Aegisub   full snapshot of the document in Aegisub
  to-aegisub.tsv     written by the MCP   event lines the MCP wants Aegisub to have
  applied.tsv        written by Aegisub   ack of the revision it applied last
  journal.jsonl      written by the MCP   append-only event log
  mcp-revision.txt   written by the MCP   last revision published from here

In Aegisub: Automation -> krapau-bridge -> "Pull changes from MCP" applies the
pending revision to the open document in place (undoable, one undo point), and
"Push my document to MCP" writes state.tsv + from-aegisub.ass by hand.
With App/Auto/Save on Every Change enabled, Aegisub also drops a copy of every
change into ?user/autosave, which the MCP side watches for realtime edits.
"""


# --------------------------------------------------------------------------- paths


def _home() -> Path:
    return Path.home()


def candidate_bridge_dirs() -> list[Path]:
    """Where a bridge directory may live, most specific first."""
    env = os.environ.get(BRIDGE_ENV)
    dirs: list[Path] = []
    if env:
        dirs.append(Path(env).expanduser())
    for base in candidate_user_dirs():
        dirs.append(base / "krapau-bridge")
    return dirs


def candidate_user_dirs() -> list[Path]:
    """Aegisub's ``?user`` directory on this platform, likeliest first.

    Aegisub keeps its per-user files in a legacy directory (``~/.aegisub`` on
    Linux, ``%APPDATA%\\Aegisub`` on Windows, ``~/Library/Application Support/
    Aegisub`` on macOS) and additionally honours ``~/.config/aegisub`` on Linux.
    Both are listed so an existing installation is found instead of guessed at.
    """
    home = _home()
    dirs = [home / ".aegisub", home / ".config" / "aegisub"]
    if sys.platform == "darwin":
        dirs.insert(0, home / "Library" / "Application Support" / "Aegisub")
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            dirs.insert(0, Path(appdata) / "Aegisub")
    return dirs


def user_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve Aegisub's ``?user`` directory (existing one wins)."""
    if explicit:
        return Path(explicit).expanduser()
    for base in candidate_user_dirs():
        if base.is_dir():
            return base
    return candidate_user_dirs()[0]


def default_bridge_dir() -> Path:
    """The bridge directory to use: an explicit ``$AEGISUB_MCP_BRIDGE`` wins.

    The environment variable is an instruction, not a hint, so it is honoured even
    before the directory exists (``Bridge.ensure()`` creates it).  Otherwise the
    first existing candidate on Aegisub's ``?user`` path is used, so an existing
    installation is found instead of guessing at a platform default.
    """
    env = os.environ.get(BRIDGE_ENV)
    if env:
        return Path(env).expanduser()
    for candidate in candidate_bridge_dirs():
        if candidate.is_dir():
            return candidate
    return candidate_bridge_dirs()[-1]


def find_bridge_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return default_bridge_dir()


def decode_aegisub_path(token: str, *, user: Path | None = None) -> Path:
    """Resolve Aegisub's ``?user`` / ``?data`` / ``?temp`` path tokens.

    Only the tokens the bridge needs are understood; anything else is returned
    unchanged so a caller can still see what Aegisub's config asked for.
    """
    user = user or user_dir()
    mapping = {
        "?user": user,
        "?data": user,
        "?temp": Path(os.environ.get("TEMP") or "/tmp"),
    }
    for prefix, base in mapping.items():
        if token.startswith(prefix):
            rest = token[len(prefix):].lstrip("/\\")
            return base / rest if rest else base
    return Path(token)


# --------------------------------------------------------------------------- tsv


def escape(value: Any) -> str:
    """Escape one TSV cell (backslash first, then TAB/CR/LF)."""
    text = "" if value is None else str(value)
    return (text.replace("\\", "\\\\").replace("\t", "\\t")
                .replace("\r", "\\r").replace("\n", "\\n"))


def unescape(value: str) -> str:
    out: list[str] = []
    i = 0
    table = {"\\": "\\", "t": "\t", "r": "\r", "n": "\n"}
    while i < len(value):
        char = value[i]
        if char == "\\" and i + 1 < len(value) and value[i + 1] in table:
            out.append(table[value[i + 1]])
            i += 2
            continue
        out.append(char)
        i += 1
    return "".join(out)


def dumps_rows(rows: Sequence[Sequence[Any]]) -> str:
    return "".join("\t".join(escape(cell) for cell in row) + "\n" for row in rows)


def atomic_write(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding, newline="")
    os.replace(tmp, path)
    return path


def read_rows(path: Path, *, encoding: str = "utf-8") -> list[list[str]]:
    if not path.is_file():
        return []
    rows: list[list[str]] = []
    for line in path.read_text(encoding=encoding, errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        rows.append([unescape(cell) for cell in line.split("\t")])
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def sha256_file(path: Path, *, limit: int | None = None) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            remaining = limit
            while True:
                chunk = handle.read(262144 if remaining is None else min(262144, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
        return digest.hexdigest()
    except OSError:
        return None


def now() -> float:
    return time.time()


def iso(timestamp: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp or now()))


# --------------------------------------------------------------------------- rows


def event_rows(doc: AssDocument) -> list[dict[str, Any]]:
    """Serialise a document's event lines with Aegisub's field names."""
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(doc.events(), start=1):
        rows.append({
            "index": index,
            "layer": entry.get("Layer", entry.get("Marked", "0")) or "0",
            "start_time": int(entry.start_ms),
            "end_time": int(entry.end_ms),
            "style": entry.get("Style", "Default"),
            "actor": entry.get("Name", ""),
            "margin_l": entry.get("MarginL", "0") or "0",
            "margin_r": entry.get("MarginR", "0") or "0",
            "margin_t": entry.get("MarginV", "0") or "0",
            "effect": entry.get("Effect", ""),
            "comment": 1 if entry.is_comment else 0,
            "text": entry.text,
        })
    return rows


def row_to_tsv(row: dict[str, Any]) -> list[str]:
    return ["D", row["index"], row["comment"], row["layer"], row["start_time"],
            row["end_time"], row["style"], row["actor"], row["margin_l"],
            row["margin_r"], row["margin_t"], row["effect"], row["text"]]


def tsv_to_row(cells: Sequence[str]) -> dict[str, Any]:
    if len(cells) < 13:
        raise ValueError(f"event row needs 13 cells, got {len(cells)}")
    keys = ("index", "comment", "layer", "start_time", "end_time", "style",
            "actor", "margin_l", "margin_r", "margin_t", "effect", "text")
    row: dict[str, Any] = {}
    for key, value in zip(keys, cells[1:]):
        if key in {"index", "comment", "start_time", "end_time"}:
            row[key] = int(float(value))
        else:
            row[key] = value
    return row


# --------------------------------------------------------------------------- payloads


def build_request(doc: AssDocument, *, rev: int, source: str = "aegisub-mcp",
                  note: str = "", script: str | None = None,
                  doc_id: str | None = None) -> str:
    """Render ``to-aegisub.tsv`` for the given document."""
    rows = event_rows(doc)
    head: list[list[Any]] = [
        ["# krapau-bridge request -- event lines the MCP wants Aegisub to have"],
        ["format", BRIDGE_FORMAT],
        ["rev", rev],
        ["time", f"{now():.3f}"],
        ["time_iso", iso()],
        ["source", source],
        ["doc_id", doc_id or ""],
        ["script", script or ""],
        ["note", note],
        ["events", len(rows)],
        ["columns", "index", "comment", "layer", "start_time", "end_time", "style",
         "actor", "margin_l", "margin_r", "margin_t", "effect", "text"],
    ]
    return dumps_rows(head) + dumps_rows([row_to_tsv(row) for row in rows])


def parse_kv(rows: Iterable[Sequence[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for cells in rows:
        if not cells:
            continue
        if cells[0] == "D":
            continue
        if len(cells) == 2:
            out[cells[0]] = cells[1]
        elif len(cells) > 2 and cells[0] == "columns":
            out["columns"] = "\t".join(cells[1:])
    return out


def parse_request(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    meta = parse_kv(rows)
    events = [tsv_to_row(cells) for cells in rows if cells and cells[0] == "D"]
    meta["events_rows"] = events
    return meta


def parse_state(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = parse_kv(read_rows(path))
    for key in ("ae_rev", "time", "line_count", "event_count", "active_line"):
        if key in meta:
            try:
                meta[key] = int(float(meta[key]))
            except ValueError:
                pass
    if "dirty" in meta:
        meta["dirty"] = meta["dirty"] not in {"0", "false", "False", ""}
    return meta


def parse_applied(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = parse_kv(read_rows(path))
    for key in ("rev", "ae_rev", "lines", "time"):
        if key in meta:
            try:
                meta[key] = int(float(meta[key])) if key != "time" else float(meta[key])
            except ValueError:
                pass
    return meta


# --------------------------------------------------------------------------- bridge


@dataclass
class BridgeEvent:
    """One observed change, as recorded in ``journal.jsonl``."""

    kind: str
    seq: int
    time: float
    rev: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "time": round(self.time, 3),
                "time_iso": iso(self.time), "rev": self.rev, **self.detail}


class Bridge:
    """Client for one bridge directory."""

    def __init__(self, directory: str | os.PathLike[str] | None = None):
        self.dir = find_bridge_dir(directory)

    # -- paths ---------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.dir / STATE_NAME

    @property
    def snapshot_path(self) -> Path:
        return self.dir / SNAPSHOT_NAME

    @property
    def request_path(self) -> Path:
        return self.dir / REQUEST_NAME

    @property
    def applied_path(self) -> Path:
        return self.dir / APPLIED_NAME

    @property
    def journal_path(self) -> Path:
        return self.dir / JOURNAL_NAME

    @property
    def revision_path(self) -> Path:
        return self.dir / REVISION_NAME

    @property
    def install_path(self) -> Path:
        return self.dir / INSTALL_NAME

    @property
    def seen_path(self) -> Path:
        return self.dir / SEEN_NAME

    def ensure(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        readme = self.dir / README_NAME
        if not readme.exists():
            atomic_write(readme, BRIDGE_README)
        return self.dir

    # -- reading -------------------------------------------------------------

    def read_state(self) -> dict[str, Any]:
        return parse_state(self.state_path)

    def read_applied(self) -> dict[str, Any]:
        return parse_applied(self.applied_path)

    def read_request(self) -> dict[str, Any]:
        return parse_request(self.request_path)

    def read_snapshot(self) -> str:
        try:
            return self.snapshot_path.read_text(encoding="utf-8-sig")
        except OSError:
            return ""

    def installed(self) -> dict[str, Any]:
        try:
            return json.loads(self.install_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def newest_autosave(self, *, user: Path | None = None) -> Path | None:
        """Newest ``*.AUTOSAVE.ass`` Aegisub wrote (its realtime channel)."""
        status = self.autosave_status(user=user)
        newest = status.get("newest")
        return Path(newest["path"]) if newest else None

    def live_text(self, source: str = "auto", *, user: Path | None = None) -> dict[str, Any]:
        """Return the freshest text Aegisub has for the open document.

        ``source`` is ``"snapshot"`` (the file the Push macro writes),
        ``"autosave"`` (Aegisub's own copy-on-change autosave) or ``"auto"``
        (whichever of the two is newer).  ``auto`` is the useful default because
        a user who never presses Push still gets realtime visibility through
        autosave, while a Push snapshot carries the *current* in-memory document
        even when autosave is off.
        """
        candidates: list[tuple[str, Path]] = []
        if source in {"auto", "snapshot"} and self.snapshot_path.is_file():
            candidates.append(("snapshot", self.snapshot_path))
        autosave_file = None
        if source in {"auto", "autosave"}:
            autosave_file = self.newest_autosave(user=user)
            if autosave_file is not None:
                candidates.append(("autosave", autosave_file))
        if not candidates:
            raise FileNotFoundError(
                "no live text from Aegisub: run the 'Push my document to MCP' macro, or "
                "enable autosave (ass_bridge_install does it) so Aegisub writes a copy "
                "of every change"
            )
        kind, path = max(candidates, key=lambda item: item[1].stat().st_mtime)
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        stat = path.stat()
        return {"source": kind, "path": str(path), "text": text,
                "mtime": stat.st_mtime, "age_s": round(now() - stat.st_mtime, 3),
                "bytes": stat.st_size, "sha256": sha256_text(text)}

    # -- revisions -----------------------------------------------------------

    def mcp_revision(self) -> int:
        try:
            return int(self.revision_path.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            return 0

    def next_revision(self) -> int:
        request = self.read_request()
        try:
            published = int(request.get("rev") or 0)
        except (TypeError, ValueError):
            published = 0
        return max(published, self.mcp_revision()) + 1

    def pending(self) -> dict[str, Any]:
        """Compare what this side published with what Aegisub acknowledged."""
        request = self.read_request()
        applied = self.read_applied()
        state = self.read_state()
        try:
            want = int(request.get("rev") or 0)
        except (TypeError, ValueError):
            want = 0
        try:
            done = int(applied.get("rev") or 0)
        except (TypeError, ValueError):
            done = 0
        return {
            "mcp_rev": want,
            "applied_rev": done,
            "ae_rev": state.get("ae_rev"),
            "pending": bool(want and want > done),
            "applied": bool(want and want <= done),
            "applied_result": applied.get("result"),
            "applied_error": applied.get("error"),
            "applied_time": applied.get("time_iso") or applied.get("time"),
            "last_origin": state.get("origin"),
        }

    # -- writing -------------------------------------------------------------

    def publish(self, doc: AssDocument, *, note: str = "",
                source: str = "ass_bridge_publish", script: str | None = None,
                doc_id: str | None = None) -> dict[str, Any]:
        """Write ``to-aegisub.tsv`` for a document and bump the revision."""
        self.ensure()
        rev = self.next_revision()
        payload = build_request(doc, rev=rev, source=source, note=note,
                                script=script, doc_id=doc_id)
        atomic_write(self.request_path, payload)
        atomic_write(self.revision_path, f"{rev}\n")
        rows = event_rows(doc)
        event = self.journal_append("mcp.publish", rev=rev, detail={
            "source": source, "note": note, "events": len(rows),
            "sha256": sha256_text(payload), "script": script or "",
        })
        return {"rev": rev, "path": str(self.request_path), "events": len(rows),
                "sha256": sha256_text(payload), "event": event.to_dict()}

    def write_state(self, **fields: Any) -> Path:
        """Write ``state.tsv`` -- normally Aegisub's job; used by tests/tools."""
        self.ensure()
        rows: list[list[Any]] = [["format", BRIDGE_FORMAT],
                                 ["time", f"{now():.3f}"],
                                 ["time_iso", iso()]]
        for key, value in fields.items():
            rows.append([key, "" if value is None else value])
        return atomic_write(self.state_path, dumps_rows(rows))

    def write_applied(self, rev: int, *, result: str = "ok", error: str = "",
                      lines: int | None = None, ae_rev: int | None = None) -> Path:
        self.ensure()
        rows: list[list[Any]] = [["format", BRIDGE_FORMAT], ["rev", rev],
                                 ["result", result], ["time", f"{now():.3f}"],
                                 ["time_iso", iso()], ["error", error]]
        if lines is not None:
            rows.append(["lines", lines])
        if ae_rev is not None:
            rows.append(["ae_rev", ae_rev])
        return atomic_write(self.applied_path, dumps_rows(rows))

    def write_snapshot(self, text: str) -> Path:
        self.ensure()
        return atomic_write(self.snapshot_path, text)

    # -- journal -------------------------------------------------------------

    def journal_append(self, kind: str, *, rev: int | None = None,
                       detail: dict[str, Any] | None = None) -> BridgeEvent:
        self.ensure()
        seq = len(self.events()) + 1
        event = BridgeEvent(kind=kind, seq=seq, time=now(), rev=rev,
                            detail=dict(detail or {}))
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def events(self, *, since: int = 0, limit: int | None = None,
               kinds: Sequence[str] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.journal_path.is_file():
            return out
        for line in self.journal_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if int(entry.get("seq") or 0) <= since:
                continue
            if kinds and entry.get("kind") not in kinds:
                continue
            out.append(entry)
        if limit is not None:
            out = out[-limit:]
        return out

    # -- observation ---------------------------------------------------------

    def _fingerprint(self) -> dict[str, Any]:
        state = self.read_state()
        applied = self.read_applied()
        snapshot = sha256_file(self.snapshot_path)
        newest = self._newest_autosave()
        if newest is None:
            autosave = ""
        else:
            stat = newest.stat()
            autosave = f"{newest}:{int(stat.st_mtime)}:{stat.st_size}"
        return {"ae_rev": state.get("ae_rev") or 0,
                "applied_rev": applied.get("rev") or 0,
                "applied_result": applied.get("result") or "",
                "snapshot": snapshot or "",
                "autosave": autosave,
                "dirty": bool(state.get("dirty")),
                "event_count": state.get("event_count") or 0}

    def autosave_dir(self, *, user: Path | None = None) -> Path:
        """Directory Aegisub autosaves into (``Path/Auto/Save`` in config.json)."""
        base = user or user_dir()
        config: dict[str, Any] = {}
        try:
            config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        raw_path = config.get(AUTOSAVE_KEYS["path"]) or "?user/autosave"
        return decode_aegisub_path(str(raw_path), user=base)

    def newest_autosave_light(self, *, user: Path | None = None) -> Path | None:
        """Newest ``*.ass`` in the autosave dir, without reading/hashing it.

        :meth:`autosave_status` hashes the file, which is far too expensive for
        :meth:`poll` (it runs several times a second while watching).
        """
        try:
            files = sorted(self.autosave_dir(user=user).glob("*.ass"),
                           key=lambda p: p.stat().st_mtime)
        except OSError:
            return None
        return files[-1] if files else None

    def _newest_autosave(self) -> Path | None:
        return self.newest_autosave_light()

    def last_seen(self) -> dict[str, Any]:
        """Fingerprint recorded by the most recent :meth:`poll`.

        Stored in :data:`SEEN_NAME` rather than as a journal row, so the journal
        holds only things that actually happened.  Bridge directories written by
        an older revision are still understood: they kept the fingerprint in the
        journal, so fall back to scanning for the newest such row.
        """
        try:
            payload = json.loads(self.seen_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            return {key: payload.get(key) for key in FINGERPRINT_FIELDS}
        for entry in reversed(self.events()):
            if entry.get("kind") == "bridge.fingerprint":
                return {key: entry.get(key) for key in FINGERPRINT_FIELDS}
        return {}

    def remember_seen(self, fingerprint: dict[str, Any]) -> None:
        """Persist the poll fingerprint (the baseline the next poll compares to)."""
        self.ensure()
        atomic_write(self.seen_path, json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")

    def poll(self) -> list[BridgeEvent]:
        """Record every change since the last poll; returns the new events."""
        self.ensure()
        current = self._fingerprint()
        previous = self.last_seen()
        events: list[BridgeEvent] = []
        if not previous or previous.get("ae_rev") != current["ae_rev"]:
            if self.state_path.is_file():
                events.append(self.journal_append("aegisub.state", rev=current["ae_rev"], detail={
                    "script": self.read_state().get("script", ""),
                    "dirty": current["dirty"],
                    "event_count": current["event_count"],
                    "origin": self.read_state().get("origin", ""),
                    "trigger": self.read_state().get("trigger", ""),
                }))
        if previous.get("applied_rev") != current["applied_rev"] and current["applied_rev"]:
            applied = self.read_applied()
            events.append(self.journal_append("aegisub.applied", rev=current["applied_rev"], detail={
                "result": applied.get("result", ""),
                "error": applied.get("error", ""),
                "lines": applied.get("lines"),
            }))
        if previous.get("snapshot") != current["snapshot"] and current["snapshot"]:
            events.append(self.journal_append("aegisub.snapshot", rev=current["ae_rev"], detail={
                "sha256": current["snapshot"],
                "bytes": self.snapshot_path.stat().st_size if self.snapshot_path.is_file() else 0,
            }))
        if previous.get("autosave") != current["autosave"] and current["autosave"]:
            newest = self._newest_autosave()
            detail: dict[str, Any] = {"fingerprint": current["autosave"]}
            if newest is not None:
                stat = newest.stat()
                detail.update({"path": str(newest), "bytes": stat.st_size,
                               "mtime": stat.st_mtime,
                               "age_s": round(now() - stat.st_mtime, 3)})
            events.append(self.journal_append("aegisub.autosave", rev=current["ae_rev"],
                                              detail=detail))
        self.remember_seen(current)
        return events

    def watch(self, *, seconds: float = 30.0, interval: float = 0.25,
              on_event: Callable[[dict[str, Any]], None] | None = None) -> list[BridgeEvent]:
        """Poll until ``seconds`` elapse; call ``on_event`` for each new event."""
        deadline = now() + max(0.0, seconds)
        collected: list[BridgeEvent] = []
        while True:
            for event in self.poll():
                collected.append(event)
                if on_event is not None:
                    on_event(event.to_dict())
            if now() >= deadline:
                return collected
            time.sleep(max(0.02, interval))

    # -- autosave / hotkeys --------------------------------------------------

    def autosave_status(self, *, user: Path | None = None) -> dict[str, Any]:
        """Report Aegisub's autosave settings and newest autosave file."""
        base = user or user_dir()
        config_path = base / "config.json"
        config: dict[str, Any] = {}
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        settings = {name: get_option(config, key) for name, key in AUTOSAVE_KEYS.items()}
        raw_path = get_option(config, AUTOSAVE_KEYS["path"]) or "?user/autosave"
        directory = decode_aegisub_path(str(raw_path), user=base)
        newest: dict[str, Any] | None = None
        try:
            files = sorted(directory.glob("*.ass"), key=lambda p: p.stat().st_mtime)
        except OSError:
            files = []
        if files:
            path = files[-1]
            stat = path.stat()
            newest = {"path": str(path), "bytes": stat.st_size,
                      "mtime": stat.st_mtime, "age_s": round(now() - stat.st_mtime, 3),
                      "sha256": sha256_file(path)}
        return {"config_path": str(config_path), "config_exists": config_path.is_file(),
                "settings": settings, "autosave_dir": str(directory),
                "autosave_dir_exists": directory.is_dir(),
                "newest": newest,
                "files": len(files)}

    def status(self, *, user: Path | None = None) -> dict[str, Any]:
        state = self.read_state()
        applied = self.read_applied()
        snapshot = self.snapshot_path
        snapshot_info: dict[str, Any] = {"exists": snapshot.is_file()}
        if snapshot.is_file():
            stat = snapshot.stat()
            snapshot_info.update({"bytes": stat.st_size, "mtime": stat.st_mtime,
                                  "age_s": round(now() - stat.st_mtime, 3),
                                  "sha256": sha256_file(snapshot)})
        installed = self.installed()
        return {
            "bridge_dir": str(self.dir),
            "exists": self.dir.is_dir(),
            "format": BRIDGE_FORMAT,
            "state": state,
            "state_age_s": (round(now() - self.state_path.stat().st_mtime, 3)
                            if self.state_path.is_file() else None),
            "snapshot": snapshot_info,
            "pending": self.pending(),
            "applied": applied,
            "mcp_revision": self.mcp_revision(),
            "installed": installed,
            "autosave": self.autosave_status(user=user),
            "events": len(self.events()),
        }

    # -- import --------------------------------------------------------------

    def import_snapshot(self, doc_id: str | None = None, *,
                        apply_to: str | None = None, source: str = "auto",
                        user: Path | None = None) -> dict[str, Any]:
        """Load Aegisub's live text into this workspace.

        ``apply_to="<doc_id>"`` merges it into an already-open document (so the
        workspace keeps its undo stack); otherwise it is opened as a new
        document.  ``source`` picks the channel -- see :meth:`live_text`.
        """
        info = self.live_text(source, user=user)
        text = info["text"]
        # Imported here (not at module scope) so ``aegisub_mcp.bridge`` stays
        # importable from the tool modules that already import it.
        from .tools.base import workspace

        where = {"live_source": info["source"], "live_path": info["path"],
                 "live_age_s": info["age_s"], "sha256": info["sha256"]}
        if apply_to:
            did = workspace.resolve_id(apply_to)
            doc = workspace.get(did)
            workspace.snapshot(did)
            doc.restore(text)
            return {"doc_id": did, "applied_to_open_document": True,
                    "lines": len(doc.all_lines()),
                    "bytes": len(text.encode("utf-8")),
                    "undo_depth": workspace.undo_depth(did), **where}
        did = workspace.register(AssDocument.from_text(text, path=str(self.snapshot_path)),
                                 doc_id=doc_id)
        return {"doc_id": did, "applied_to_open_document": False,
                "lines": len(workspace.get(did).all_lines()),
                "bytes": len(text.encode("utf-8")), **where}

    # -- install -------------------------------------------------------------

    def asset_path(self) -> Path:
        return Path(__file__).resolve().parent / "assets" / ASSET_NAME

    def render_script(self, *, bridge_dir: Path | None = None,
                      autoload_dir: Path | None = None,
                      hotkey: str = DEFAULT_HOTKEY,
                      auto_pull: bool = True) -> str:
        template = self.asset_path().read_text(encoding="utf-8")
        values = {
            "@BRIDGE_DIR@": str(bridge_dir or self.dir),
            "@AUTOLOAD_DIR@": str(autoload_dir or ""),
            "@HOTKEY@": hotkey,
            "@AUTO_PULL@": "true" if auto_pull else "false",
            "@FORMAT@": BRIDGE_FORMAT,
        }
        for key, value in values.items():
            template = template.replace(key, value)
        return template

    def install(self, *, user: Path | None = None, hotkey: str = DEFAULT_HOTKEY,
                enable_autosave: bool = True, autosave_interval: int = 5,
                save_on_every_change: bool = False,
                auto_pull: bool = True,
                dry_run: bool = False, force: bool = False) -> dict[str, Any]:
        """Write the Aegisub-side script and (optionally) patch Aegisub config.

        Touches three files outside this repository, each backed up first:

        1. ``?user/automation/autoload/krapau-bridge.lua`` -- the bridge script.
        2. ``?user/config.json`` -- autosave settings (so Aegisub writes a copy of
           every change, giving the MCP side a realtime view).
        3. ``?user/hotkey.json`` -- binds the pull macro's hotkey.

        With ``dry_run`` nothing is written and the report says what *would*
        change; ``force`` allows overwriting a script that is not ours.
        """
        base = user or user_dir()
        autoload = base / "automation" / "autoload"
        script_path = autoload / SCRIPT_NAME
        config_path = base / "config.json"
        hotkey_path = base / "hotkey.json"
        command = PULL_COMMAND

        warnings: list[str] = []
        # ``--user-dir`` / ``user_dir_override`` means Aegisub's ``?user`` directory
        # (``$HOME/.aegisub`` on Linux), not ``$HOME``.  Passing the home directory
        # is the easy mistake to make, and it silently installs where Aegisub will
        # never look, so say so instead of writing a file nothing reads.
        if not (base / "config.json").is_file() and (base / ".aegisub").is_dir():
            warnings.append(
                f"{base} looks like a home directory: Aegisub's user directory is "
                f"{base / '.aegisub'} on this platform -- the script is being written "
                "to the wrong place"
            )

        report: dict[str, Any] = {
            "user_dir": str(base),
            "bridge_dir": str(self.dir),
            "script": {"path": str(script_path), "exists": script_path.is_file()},
            "config": {"path": str(config_path), "exists": config_path.is_file()},
            "hotkey": {"path": str(hotkey_path), "exists": hotkey_path.is_file(),
                       "combo": hotkey, "command": command},
            "autosave": {"enable": enable_autosave, "interval": autosave_interval,
                         "save_on_every_change": save_on_every_change},
            "auto_pull": auto_pull,
            "dry_run": dry_run,
            "warnings": warnings,
        }
        existing = None
        if script_path.is_file():
            existing = script_path.read_text(encoding="utf-8")
            report["script"]["sha256"] = sha256_text(existing)
            if "krapau-bridge" not in existing and not force:
                raise FileExistsError(
                    f"{script_path} exists and does not look like the bridge script; "
                    "pass force=True to overwrite"
                )
        rendered = self.render_script(bridge_dir=self.dir, autoload_dir=autoload,
                                      hotkey=hotkey, auto_pull=auto_pull)
        report["script"]["rendered_sha256"] = sha256_text(rendered)
        report["script"]["up_to_date"] = existing == rendered if existing is not None else False
        report["script"]["bytes"] = len(rendered.encode("utf-8"))

        config_changes: dict[str, Any] = {}
        config: dict[str, Any] = {}
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except ValueError:
                report["warnings"].append(f"{config_path} is not valid JSON -- left untouched")
                config = {}
        if isinstance(config, dict):
            # Aegisub's config.json stores option names as nested objects, and older
            # revisions of this module wrote flat "App/Auto/Save" keys that Aegisub
            # ignores; drop those so the file has one source of truth.
            repaired = [key for key in AUTOSAVE_KEYS.values() if flat_remove(config, key)]
            if repaired:
                report["config"]["repaired_flat_keys"] = sorted(repaired)
            if enable_autosave:
                # ``App/Auto/Save`` + a *positive* interval keep Aegisub writing
                # ``<name>.<timestamp>.AUTOSAVE.ass`` copies into ``?user/autosave``;
                # that stream is the realtime view of the user's editing.
                # ``autosave_timer_changed()`` in src/subs_controller.cpp stops the
                # timer when the interval is 0, so 0 would kill the channel.
                # ``Save on Every Change`` is Aegisub *saving the user's own file*
                # in place on every commit -- real realtime, but it rewrites their
                # file, so it is opt-in.
                desired: dict[str, Any] = {
                    AUTOSAVE_KEYS["enabled"]: True,
                    AUTOSAVE_KEYS["interval"]: max(1, int(autosave_interval)),
                }
                if save_on_every_change:
                    desired[AUTOSAVE_KEYS["on_change"]] = True
            else:
                desired = {}
            for key, value in desired.items():
                if get_option(config, key) != value:
                    config_changes[key] = {"from": get_option(config, key), "to": value}
                    set_option(config, key, value)
        report["config"]["changes"] = config_changes
        report["config"]["keys_present"] = {
            name: get_option(config, key) for name, key in AUTOSAVE_KEYS.items()
        } if config else {}
        report["config"]["automation_paths"] = {
            name: get_option(config, key) for name, key in AUTOMATION_PATH_KEYS.items()
        } if config else {}

        hotkeys: dict[str, Any] = {}
        if hotkey_path.is_file():
            try:
                hotkeys = json.loads(hotkey_path.read_text(encoding="utf-8"))
            except ValueError:
                report["warnings"].append(f"{hotkey_path} is not valid JSON -- left untouched")
                hotkeys = {}
        hotkey_change: dict[str, Any] | None = None
        removed_commands: list[str] = []
        if isinstance(hotkeys, dict):
            context = hotkeys.setdefault("Default", {})
            if not isinstance(context, dict):
                report["warnings"].append("hotkey.json 'Default' context is not an object")
            else:
                # A revision of this module bound the pull macro without the
                # ``krapau-bridge: `` prefix.  Aegisub resolves the first matching
                # binding for a keystroke, so a stale entry wins over the correct
                # one and the keypress appears to do nothing -- drop any old bridge
                # command before adding the current one.
                for name in list(context):
                    if name.startswith(f"automation/lua/{SCRIPT_LABEL}/") and name != command:
                        context.pop(name)
                        removed_commands.append(name)
                current = context.get(command)
                if current != [hotkey]:
                    hotkey_change = {"from": current, "to": [hotkey]}
                    context[command] = [hotkey]
        report["hotkey"]["change"] = hotkey_change
        report["hotkey"]["removed_stale_commands"] = removed_commands

        if dry_run:
            return report

        self.ensure()
        autoload.mkdir(parents=True, exist_ok=True)
        if existing is not None:
            atomic_write(script_path.with_suffix(".lua.bak"), existing)
        atomic_write(script_path, rendered)
        report["script"]["written"] = True
        report["script"]["sha256"] = sha256_text(rendered)

        if config_changes and config_path.parent.is_dir():
            if config_path.is_file():
                atomic_write(config_path.with_name(config_path.name + ".bak"),
                             config_path.read_text(encoding="utf-8"))
            atomic_write(config_path, json.dumps(config, indent=4, ensure_ascii=False) + "\n")
            report["config"]["written"] = True
        if (hotkey_change or removed_commands) and hotkey_path.parent.is_dir():
            if hotkey_path.is_file():
                atomic_write(hotkey_path.with_name(hotkey_path.name + ".bak"),
                             hotkey_path.read_text(encoding="utf-8"))
            atomic_write(hotkey_path, json.dumps(hotkeys, indent=4, ensure_ascii=False) + "\n")
            report["hotkey"]["written"] = True

        install_record = {
            "installed_at": iso(),
            "bridge_dir": str(self.dir),
            "user_dir": str(base),
            "script": str(script_path),
            "script_sha256": report["script"]["sha256"],
            "hotkey": hotkey,
            "hotkey_command": command,
            "autosave": {"enable": enable_autosave, "interval": autosave_interval,
                         "save_on_every_change": save_on_every_change},
            "autosave_state": report["config"]["keys_present"],
            "auto_pull": auto_pull,
            "format": BRIDGE_FORMAT,
        }
        atomic_write(self.install_path, json.dumps(install_record, indent=2) + "\n")
        report["install_record"] = install_record
        return report


# --------------------------------------------------------------------------- injection


MACOS_KEYS = {"ctrl": "control down", "alt": "option down", "shift": "shift down",
              "cmd": "command down"}


def parse_combo(combo: str) -> tuple[list[str], str]:
    """Split an Aegisub hotkey such as ``Ctrl-Alt-M`` into modifiers + key."""
    parts = [part for part in combo.replace("+", "-").split("-") if part]
    modifiers: list[str] = []
    key = ""
    for part in parts:
        lowered = part.lower()
        if lowered in {"ctrl", "control"}:
            modifiers.append("ctrl")
        elif lowered in {"alt", "option"}:
            modifiers.append("alt")
        elif lowered in {"shift"}:
            modifiers.append("shift")
        elif lowered in {"cmd", "command", "super", "meta"}:
            modifiers.append("cmd")
        else:
            key = part
    return modifiers, key


def inject_hotkey(combo: str = DEFAULT_HOTKEY) -> dict[str, Any]:
    """Press ``combo`` in the running GUI, if this platform allows it.

    This is *optional*: the bridge works without it (the user presses the
    hotkey, or Aegisub applies the pull when it validates the macro).  Injection
    is what makes the MCP -> Aegisub direction hands-free, and its support is
    platform-specific:

    * Linux/X11 -- ``xdotool``; Linux/Wayland -- ``wtype`` (wlroots) or
      ``ydotool`` (needs the daemon/uinput).
    * macOS -- ``osascript`` (needs Accessibility permission).
    * Windows -- PowerShell ``SendKeys``.

    Returns ``{"ok": bool, "tool": ..., "command": [...], "error": ...}`` and
    never raises, so a caller can report why injection is unavailable.
    """
    mods, key = parse_combo(combo)
    if not key:
        return {"ok": False, "error": f"could not parse hotkey {combo!r}"}
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()

    def run(command: list[str]) -> dict[str, Any]:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "tool": command[0], "command": command, "error": str(exc)}
        if proc.returncode != 0:
            return {"ok": False, "tool": command[0], "command": command,
                    "error": (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"}
        return {"ok": True, "tool": command[0], "command": command}

    if sys.platform == "darwin":
        if not shutil.which("osascript"):
            return {"ok": False, "tool": "osascript", "error": "osascript not found"}
        using = ", ".join(MACOS_KEYS[m] for m in mods if m in MACOS_KEYS)
        script = f'tell application "System Events" to keystroke "{key.lower()}"' + (
            f" using {{{using}}}" if using else "")
        return run(["osascript", "-e", script])

    if os.name == "nt":
        if not shutil.which("powershell"):
            return {"ok": False, "tool": "powershell", "error": "powershell not found"}
        symbols = {"ctrl": "^", "alt": "%", "shift": "+"}
        keys = "".join(symbols[m] for m in mods if m in symbols) + key.lower()
        return run(["powershell", "-NoProfile", "-Command",
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    f"[System.Windows.Forms.SendKeys]::SendWait('{keys}')"])

    # Linux: prefer the tool that matches the session type.
    if session == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
        if shutil.which("wtype"):
            command = ["wtype"]
            for modifier in mods:
                command += ["-M", "alt" if modifier == "alt" else modifier]
            command += ["-k", key.lower(), "-m"]
            for modifier in reversed(mods):
                command += ["-m", "alt" if modifier == "alt" else modifier]
            return run(command)
        if shutil.which("ydotool"):
            return run(["ydotool", "key", "+".join(mods + [key.lower()])])
        return {"ok": False,
                "error": ("Wayland session without a supported injector: install wtype "
                          "(wlroots) or ydotool+ydotoold (uinput); otherwise press the "
                          "hotkey in Aegisub yourself")}
    if shutil.which("xdotool"):
        return run(["xdotool", "key", "--clearmodifiers", combo])
    return {"ok": False, "error": "xdotool not found (X11), wtype/ydotool (Wayland)"}


# --------------------------------------------------------------------------- cli


def _cmd_status(args: argparse.Namespace) -> int:
    bridge = Bridge(args.bridge_dir)
    print(json.dumps(bridge.status(user=Path(args.user_dir) if args.user_dir else None),
                     ensure_ascii=False, indent=2))
    return 0


def _cmd_paths(args: argparse.Namespace) -> int:
    print(json.dumps({
        "bridge_dir": str(find_bridge_dir(args.bridge_dir)),
        "candidates": [str(p) for p in candidate_bridge_dirs()],
        "user_dirs": [str(p) for p in candidate_user_dirs()],
        "asset": str(Bridge(args.bridge_dir).asset_path()),
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    from .tools.base import workspace

    bridge = Bridge(args.bridge_dir)
    if args.file:
        doc_id = workspace.open(args.file)
    else:
        doc_id = workspace.resolve_id(args.doc_id)
    doc = workspace.get(doc_id)
    result = bridge.publish(doc, note=args.note or "", source="aegisub-mcp-bridge publish",
                            script=str(workspace.path(doc_id) or "") or None, doc_id=doc_id)
    result["doc_id"] = doc_id
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    bridge = Bridge(args.bridge_dir)
    text = bridge.read_snapshot()
    if args.json:
        print(json.dumps({"bytes": len(text), "sha256": sha256_text(text),
                          "path": str(bridge.snapshot_path)}, indent=2))
    else:
        sys.stdout.write(text)
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    bridge = Bridge(args.bridge_dir)
    if args.follow:
        return _watch(bridge, args)
    print(json.dumps(bridge.events(since=args.since, limit=args.limit),
                     ensure_ascii=False, indent=2))
    return 0


def _watch(bridge: Bridge, args: argparse.Namespace) -> int:
    injected: set[int] = set()

    def on_event(event: dict[str, Any]) -> None:
        if args.json:
            print(json.dumps(event, ensure_ascii=False))
        else:
            print(f"[{event['time_iso']}] {event['kind']} rev={event.get('rev')} {event.get('detail', {})}")
        sys.stdout.flush()
        if args.inject and event["kind"] == "mcp.publish":
            outcome = inject_hotkey(args.hotkey)
            injected.add(outcome.get("ok", False))
            print(json.dumps({"inject": outcome}, ensure_ascii=False))

    print(json.dumps({"watch": str(bridge.dir), "seconds": args.seconds,
                      "interval": args.interval, "inject": bool(args.inject),
                      "hotkey": args.hotkey}, ensure_ascii=False))
    bridge.watch(seconds=args.seconds, interval=args.interval, on_event=on_event)
    return 0


def _cmd_install(args: argparse.Namespace) -> int:
    bridge = Bridge(args.bridge_dir)
    try:
        report = bridge.install(user=Path(args.user_dir) if args.user_dir else None,
                               hotkey=args.hotkey,
                               enable_autosave=not args.no_autosave,
                               autosave_interval=args.autosave_interval,
                               save_on_every_change=args.save_on_every_change,
                               auto_pull=not args.no_auto_pull,
                               dry_run=args.dry_run, force=args.force)
    except FileExistsError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"ok": True, **report}, ensure_ascii=False, indent=2))
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    bridge = Bridge(args.bridge_dir)
    print(json.dumps(bridge.import_snapshot(doc_id=args.doc_id, apply_to=args.apply_to),
                     ensure_ascii=False, indent=2))
    return 0


def _cmd_pull_state(args: argparse.Namespace) -> int:
    bridge = Bridge(args.bridge_dir)
    print(json.dumps({"pending": bridge.pending(),
                      "state": bridge.read_state(),
                      "applied": bridge.read_applied()},
                     ensure_ascii=False, indent=2))
    return 0


def _cmd_inject(args: argparse.Namespace) -> int:
    print(json.dumps(inject_hotkey(args.hotkey), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegisub-mcp-bridge",
        description="File bridge between Aegisub and aegisub-mcp (MCP <-> editor).",
    )
    parser.add_argument("--bridge-dir", help=f"bridge directory (default: ${BRIDGE_ENV} "
                                             "or <Aegisub user dir>/krapau-bridge)")
    parser.add_argument("--user-dir", help="Aegisub ?user directory override")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="report bridge + autosave state").set_defaults(func=_cmd_status)
    sub.add_parser("paths", help="show resolved paths").set_defaults(func=_cmd_paths)
    sub.add_parser("pending", help="show published vs applied revisions").set_defaults(func=_cmd_pull_state)

    publish = sub.add_parser("publish", help="queue the workspace document for Aegisub")
    publish.add_argument("--doc-id")
    publish.add_argument("--file", help="open this .ass file first")
    publish.add_argument("--note", help="free-form note stored in the request")
    publish.set_defaults(func=_cmd_publish)

    show = sub.add_parser("show", help="print Aegisub's snapshot")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=_cmd_show)

    events = sub.add_parser("events", help="print the journal")
    events.add_argument("--since", type=int, default=0)
    events.add_argument("--limit", type=int)
    events.add_argument("--follow", action="store_true", help="watch instead")
    events.add_argument("--seconds", type=float, default=60.0)
    events.add_argument("--interval", type=float, default=0.25)
    events.add_argument("--json", action="store_true")
    events.add_argument("--inject", action="store_true", help="press the pull hotkey on mcp.publish")
    events.add_argument("--hotkey", default=DEFAULT_HOTKEY)
    events.set_defaults(func=_cmd_events)

    watch = sub.add_parser("watch", help="watch the bridge directory")
    watch.add_argument("--seconds", type=float, default=60.0)
    watch.add_argument("--interval", type=float, default=0.25)
    watch.add_argument("--json", action="store_true")
    watch.add_argument("--inject", action="store_true")
    watch.add_argument("--hotkey", default=DEFAULT_HOTKEY)
    watch.set_defaults(func=lambda a: _watch(Bridge(a.bridge_dir), a))

    install = sub.add_parser("install", help="write the Aegisub-side script + config")
    install.add_argument("--hotkey", default=DEFAULT_HOTKEY)
    install.add_argument("--no-autosave", action="store_true",
                         help="leave Aegisub's autosave settings alone")
    install.add_argument("--autosave-interval", type=int, default=5,
                         help="App/Auto/Save Every Seconds; must be >= 1, since 0 "
                              "stops Aegisub's autosave timer (default: 5)")
    install.add_argument("--save-on-every-change", action="store_true",
                         help="also save the user's own file in place on every edit")
    install.add_argument("--no-auto-pull", action="store_true",
                         help="do not apply pending revisions from the macro validation hook")
    install.add_argument("--dry-run", action="store_true")
    install.add_argument("--force", action="store_true")
    install.set_defaults(func=_cmd_install)

    apply_cmd = sub.add_parser("apply", help="load Aegisub's snapshot into the workspace")
    apply_cmd.add_argument("--doc-id")
    apply_cmd.add_argument("--apply-to", help="merge into this open document instead of opening a new one")
    apply_cmd.set_defaults(func=_cmd_apply)

    inject = sub.add_parser("inject", help="press a hotkey in the running GUI (optional helper)")
    inject.add_argument("--hotkey", default=DEFAULT_HOTKEY)
    inject.set_defaults(func=_cmd_inject)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
