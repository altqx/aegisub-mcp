"""Tools for the Aegisub <-> MCP bridge (realtime edits in both directions).

The bridge itself lives in :mod:`aegisub_mcp.bridge`: a small TSV protocol in a
directory both sides know, plus the Aegisub-side Lua script that Aegisub loads
from its autoload directory.  These tools are the MCP-facing half:

* ``ass_bridge_publish`` -- hand a workspace document to Aegisub (it shows up in
  the open window as soon as Aegisub pulls, which the installed script does on
  its own while it validates the macro, or when the user presses the hotkey).
* ``ass_bridge_pull`` / ``ass_bridge_live`` -- read what Aegisub has *right now*,
  including unsaved in-memory edits, from the Push snapshot or from Aegisub's own
  autosave-on-every-change copies.
* ``ass_bridge_watch`` / ``ass_bridge_events`` -- bounded realtime observation: an
  MCP client waits for, or reads back, the changes Aegisub made.
* ``ass_bridge_install`` -- write the Autoload script and the Aegisub config bits
  (autosave + hotkey) with backups, or report exactly what would change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..bridge import (
    BRIDGE_ENV,
    DEFAULT_HOTKEY,
    Bridge,
    candidate_bridge_dirs,
    candidate_user_dirs,
    default_bridge_dir,
    inject_hotkey,
    user_dir,
)
from .base import ToolError, workspace

__all__ = [
    "ass_bridge_paths",
    "ass_bridge_status",
    "ass_bridge_publish",
    "ass_bridge_live",
    "ass_bridge_pull",
    "ass_bridge_events",
    "ass_bridge_watch",
    "ass_bridge_autosave",
    "ass_bridge_install",
    "ass_bridge_inject",
    "register",
]


def _bridge(bridge_dir: str | None = None) -> Bridge:
    return Bridge(bridge_dir)


# --------------------------------------------------------------------------- paths


def ass_bridge_paths(bridge_dir: str | None = None,
                     user_dir_override: str | None = None) -> dict[str, Any]:
    """Show every path the bridge uses, so setup problems are visible at once.

    Returns the bridge directory in use, the candidates it was chosen from,
    Aegisub's ``?user`` directory candidates on this platform, where the
    Aegisub-side script asset lives, and the config/hotkey files
    ``ass_bridge_install`` would touch.
    """
    resolved = default_bridge_dir() if bridge_dir is None else Path(bridge_dir).expanduser()
    base = Path(user_dir_override).expanduser() if user_dir_override else user_dir()
    return {
        "bridge_dir": str(resolved),
        "bridge_dir_exists": resolved.is_dir(),
        "bridge_env_var": BRIDGE_ENV,
        "bridge_candidates": [str(p) for p in candidate_bridge_dirs()],
        "user_dir": str(base),
        "user_dir_candidates": [str(p) for p in candidate_user_dirs()],
        "aegisub_config": str(base / "config.json"),
        "aegisub_hotkeys": str(base / "hotkey.json"),
        "autoload_dir": str(base / "automation" / "autoload"),
        "lua_asset": str(_bridge(bridge_dir).asset_path()),
        "default_hotkey": DEFAULT_HOTKEY,
    }


# --------------------------------------------------------------------------- status


def ass_bridge_status(bridge_dir: str | None = None,
                      user_dir_override: str | None = None) -> dict[str, Any]:
    """Report the bridge state: revisions, pending work, autosave, install info.

    ``pending`` compares the revision this side published with the one Aegisub
    acknowledged in ``applied.tsv`` -- that is the authoritative answer to "did
    my edit reach the editor?".  ``autosave`` reports Aegisub's autosave settings
    and the newest ``*.AUTOSAVE.ass`` copy together with its age and hash, which
    is the realtime record of what the user is doing in Aegisub.
    """
    bridge = _bridge(bridge_dir)
    status = bridge.status(user=Path(user_dir_override).expanduser() if user_dir_override else None)
    status["live"] = None
    try:
        info = bridge.live_text("auto", user=Path(user_dir_override).expanduser()
                                if user_dir_override else None)
    except FileNotFoundError:
        pass
    else:
        status["live"] = {"source": info["source"], "path": info["path"],
                          "age_s": info["age_s"], "bytes": info["bytes"],
                          "sha256": info["sha256"]}
    return status


# --------------------------------------------------------------------------- publish


def ass_bridge_publish(doc_id: str | None = None,
                       note: str = "",
                       bridge_dir: str | None = None,
                       save_first: bool = False) -> dict[str, Any]:
    """Queue a workspace document for the Aegisub open document.

    Writes ``to-aegisub.tsv`` with the document's event lines (Aegisub field
    names, times in ms) and bumps the bridge revision.  Aegisub applies it when
    it next pulls -- automatically with the installed script (auto-pull on
    validation), or when the user presses the bridge hotkey.  ``note`` ends up in
    Aegisub's undo point, so the editor shows why the document changed.

    ``save_first=True`` writes the file to disk first (``workspace.save``); only
    useful when Aegisub has the same file open and you want it on disk too.
    """
    bridge = _bridge(bridge_dir)
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    if save_first:
        workspace.save(did)
    path = workspace.path(did)
    result = bridge.publish(doc, note=note, source="ass_bridge_publish",
                            script=str(path) if path else None, doc_id=did)
    result["doc_id"] = did
    result["pending"] = bridge.pending()
    result["hint"] = ("Aegisub applies this on its next pull; check applied_rev in "
                      "ass_bridge_status to confirm")
    return result


# --------------------------------------------------------------------------- read back


def ass_bridge_live(source: str = "auto",
                    bridge_dir: str | None = None,
                    user_dir_override: str | None = None,
                    max_bytes: int = 400_000) -> dict[str, Any]:
    """Read what Aegisub has open *right now* (text + provenance).

    ``source`` is ``"auto"`` (newest of the Push snapshot and Aegisub's autosave
    copies), ``"snapshot"`` or ``"autosave"``.  The reply carries the source, the
    file path, its age and hash, and the document text (truncated past
    ``max_bytes``).
    """
    bridge = _bridge(bridge_dir)
    try:
        info = bridge.live_text(source, user=Path(user_dir_override).expanduser()
                                if user_dir_override else None)
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc
    text = info["text"]
    out = {k: v for k, v in info.items() if k != "text"}
    if len(text) > max_bytes:
        out["text"] = text[:max_bytes]
        out["truncated"] = True
    else:
        out["text"] = text
        out["truncated"] = False
    out["lines"] = text.count("\n") + (0 if text.endswith("\n") else 1)
    return out


def ass_bridge_pull(doc_id: str | None = None,
                    apply_to: str | None = None,
                    source: str = "auto",
                    bridge_dir: str | None = None,
                    user_dir_override: str | None = None) -> dict[str, Any]:
    """Import Aegisub's live document into this workspace.

    ``apply_to="<doc_id>"`` replaces an already-open document in place (undoable
    through ``ass_undo``); otherwise a new document is opened.  Use it to keep
    working from what the user has in the editor -- their in-memory edits, not
    just what is on disk.
    """
    bridge = _bridge(bridge_dir)
    try:
        return bridge.import_snapshot(doc_id=doc_id, apply_to=apply_to, source=source,
                                      user=Path(user_dir_override).expanduser()
                                      if user_dir_override else None)
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - parse failures must read as tool errors
        raise ToolError(f"could not import Aegisub's document: {exc}") from exc


# --------------------------------------------------------------------------- realtime


def ass_bridge_events(since: int = 0, limit: int = 50,
                      kinds: Sequence[str] | None = None,
                      bridge_dir: str | None = None) -> dict[str, Any]:
    """Read the bridge journal (what Aegisub did, in order).

    Event kinds: ``mcp.publish`` (this side queued a revision),
    ``aegisub.state`` (Aegisub pushed state), ``aegisub.snapshot`` (the document
    text changed), ``aegisub.applied`` (Aegisub acknowledged a revision) and
    ``aegisub.autosave`` (Aegisub wrote one of its copy-on-change files).
    ``bridge.fingerprint`` only appears in bridge directories written by an older
    revision -- the poll baseline now lives in ``last-seen.json`` instead, so the
    journal does not grow four rows a second while watching.  Pass ``since`` = the
    last ``seq`` you saw to get only new events.

    The bridge directory is polled first, so anything Aegisub has written since
    the last poll is journaled and appears in the reply -- this is the call to
    make when you want "what has the editor done lately?" without waiting.
    """
    bridge = _bridge(bridge_dir)
    bridge.poll()
    entries = bridge.events(since=since, limit=limit or None, kinds=kinds)
    return {"bridge_dir": str(bridge.dir), "since": since, "count": len(entries),
            "last_seq": entries[-1]["seq"] if entries else since,
            "events": entries}


def ass_bridge_watch(seconds: float = 5.0,
                     interval: float = 0.25,
                     kinds: Sequence[str] | None = None,
                     stop_after: int | None = None,
                     bridge_dir: str | None = None,
                     user_dir_override: str | None = None) -> dict[str, Any]:
    """Stream what Aegisub does for up to ``seconds`` and report every change.

    This is how an MCP client observes the editor in near realtime without a
    daemon: the call polls the bridge directory (default every 0.25 s), records
    new events in the journal and keeps going for the whole window, so a client
    can watch an editing session instead of taking one sample of it.  Pass
    ``stop_after`` to come back once N matching events have arrived (a "wake me
    on the next change"), or ``kinds`` to watch one channel only.

    ``events`` are the things that changed *while watching*.  The first poll
    establishes the baseline and anything it reports had already happened, so it
    comes back separately as ``caught_up`` and does not satisfy ``stop_after`` --
    otherwise a fresh bridge would return instantly with stale state and never
    see the edit the caller started watching for.  A client looping on this tool
    should treat ``caught_up`` and ``events`` alike as input.

    An empty ``events`` list means Aegisub was quiet for the whole window; it is
    not an error, and ``waited_s`` tells you how long that was.
    """
    bridge = _bridge(bridge_dir)
    base = Path(user_dir_override).expanduser() if user_dir_override else None
    seen_snapshot = bridge.last_seen()
    started = bridge.read_state().get("ae_rev")

    def _matching(events: Any) -> list[dict[str, Any]]:
        return [e.to_dict() for e in events if not kinds or e.kind in kinds]

    import time as _time

    started_at = _time.monotonic()
    deadline = started_at + max(0.0, seconds)

    catch_up = _matching(bridge.poll())
    collected: list[dict[str, Any]] = []
    while True:
        if stop_after is not None and len(collected) >= stop_after:
            break
        now = _time.monotonic()
        if now >= deadline:
            break
        _time.sleep(min(max(0.02, interval), deadline - now))
        collected.extend(_matching(bridge.poll()))

    out: dict[str, Any] = {
        "bridge_dir": str(bridge.dir),
        "waited_s": round(_time.monotonic() - started_at, 3),
        "count": len(collected),
        "events": collected,
        "caught_up": catch_up,
        "caught_up_count": len(catch_up),
        "stop_after": stop_after,
        "seen_before": seen_snapshot,
        "ae_rev_before": started,
        "ae_rev_after": bridge.read_state().get("ae_rev"),
    }
    try:
        info = bridge.live_text("auto", user=base)
    except FileNotFoundError:
        out["live"] = None
    else:
        out["live"] = {"source": info["source"], "path": info["path"],
                       "age_s": info["age_s"], "sha256": info["sha256"]}
    return out


def ass_bridge_autosave(bridge_dir: str | None = None,
                        user_dir_override: str | None = None) -> dict[str, Any]:
    """Report Aegisub's autosave channel: settings, directory, newest copy.

    Aegisub writes a copy of the open document into ``?user/autosave`` on every
    change when ``App/Auto/Save on Every Change`` is on.  That stream of
    ``<name>.<timestamp>.AUTOSAVE.ass`` files is the realtime view of the user's
    editing and needs nothing from the bridge script.
    """
    bridge = _bridge(bridge_dir)
    return bridge.autosave_status(user=Path(user_dir_override).expanduser()
                                  if user_dir_override else None)


# --------------------------------------------------------------------------- install


def ass_bridge_install(bridge_dir: str | None = None,
                       user_dir_override: str | None = None,
                       hotkey: str = DEFAULT_HOTKEY,
                       enable_autosave: bool = True,
                       autosave_interval: int = 5,
                       save_on_every_change: bool = False,
                       auto_pull: bool = True,
                       dry_run: bool = True,
                       force: bool = False) -> dict[str, Any]:
    """Install the Aegisub-side half of the bridge (dry run by default).

    Writes, with a ``.bak`` copy next to each file it replaces:

    * ``?user/automation/autoload/krapau-bridge.lua`` -- the Autoload script with
      the bridge directory baked in.
    * ``?user/config.json`` -- enables ``App/Auto/Save`` and sets a *positive*
      ``App/Auto/Save Every Seconds`` (default 5) so Aegisub keeps writing
      ``<name>.<timestamp>.AUTOSAVE.ass`` copies into ``?user/autosave``; that
      stream is the realtime view of the user's own editing.  An interval of 0
      would *stop* Aegisub's autosave timer (``autosave_timer_changed()`` in
      ``src/subs_controller.cpp``), so it is clamped to at least 1.
    * ``?user/hotkey.json`` -- binds the pull macro to ``hotkey`` (default
      Ctrl-Alt-M), under the Default context.

    ``save_on_every_change`` additionally enables ``App/Auto/Save on Every
    Change``, which makes Aegisub save the user's own file in place on every
    edit -- the fastest channel, but it rewrites their file, so it is opt-in.
    ``dry_run=True`` (the default) only reports what would change.  ``auto_pull``
    lets the script apply a pending revision from its macro-validation callback,
    which is what makes MCP edits appear without touching Aegisub.
    """
    bridge = _bridge(bridge_dir)
    try:
        report = bridge.install(user=Path(user_dir_override).expanduser()
                                if user_dir_override else None,
                                hotkey=hotkey, enable_autosave=enable_autosave,
                                autosave_interval=autosave_interval,
                                save_on_every_change=save_on_every_change,
                                auto_pull=auto_pull, dry_run=dry_run, force=force)
    except FileExistsError as exc:
        raise ToolError(str(exc)) from exc
    report["next"] = ("restart Aegisub (Autoload scripts load at startup), then open a "
                      "subtitle file; run the 'krapau-bridge: Bridge status' macro to "
                      "confirm the paths")
    return report


def ass_bridge_inject(hotkey: str = DEFAULT_HOTKEY) -> dict[str, Any]:
    """Press the pull hotkey in the running GUI (optional, platform specific).

    Only needed if you want MCP -> Aegisub to be hands-free *without* the
    script's auto-pull.  Uses ``xdotool`` (X11), ``wtype``/``ydotool`` (Wayland),
    ``osascript`` (macOS) or PowerShell ``SendKeys`` (Windows) -- and reports
    precisely which one is missing when it cannot.
    """
    return inject_hotkey(hotkey)


# --------------------------------------------------------------------------- register


def register(mcp, ws=None):
    """Register every ``ass_`` callable in this module with ``mcp``.

    ``ws`` is accepted for symmetry with the other tool modules but unused: the
    tools operate on the process-wide :data:`aegisub_mcp.tools.base.workspace`
    singleton.  Returns the sorted list of registered tool names.
    """
    names: list[str] = []
    for name in sorted(globals()):
        obj = globals()[name]
        if name.startswith("ass_") and callable(obj):
            mcp.tool()(obj)
            names.append(name)
    return names
