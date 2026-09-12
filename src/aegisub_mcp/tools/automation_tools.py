"""Tools that run *real* Aegisub Automation 4 scripts (macros and filters).

Aegisub's Automation 4 is a Lua scripting host: a script calls
``aegisub.register_macro`` / ``aegisub.register_filter`` and manipulates the
``subs`` object.  Those scripts live in the automation directories
(``~/.aegisub/automation``, ``?data/automation/include``, ...) and normally can
only run by clicking a menu item or pressing a hotkey inside the editor.

These tools load the *same, unmodified* ``.lua`` files into the pure-Python
Automation 4 host (:mod:`aegisub_mcp.lua.engine`) and run them against a
document in this workspace, so a macro can be driven head-lessly, scripted, and
verified by an MCP client.  ``ass_automation_run_macro`` mutates the workspace
document exactly like pressing the macro's hotkey inside Aegisub -- including
the in-memory document, undo stack, and ``subs`` semantics (``subs[0] = {...}``
appends, ``subs[-n] = {...}`` inserts, ``subs[n] = nil`` deletes).

Every tool returns a plain dict.  A script that raises is reported as
``{"ok": false, "error": ...}`` together with its ``log`` -- only bad *arguments*
(this module's own errors) raise.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

from ..lua.engine import LuaEngine, LuaError, MacroNotFound, MacroResult, include_dirs_default
from .base import ToolError, workspace

__all__ = [
    "ass_automation_dirs",
    "ass_automation_list",
    "ass_automation_info",
    "ass_automation_run_macro",
    "ass_automation_run_filter",
    "ass_automation_run_filters",
    "ass_automation_from_source",
    "register",
]

#: Scripts bigger than this are still runnable (pass the absolute path to a
#: run tool), but ``ass_automation_list`` will not execute them to enumerate
#: their registrations unless ``limit_bytes`` says otherwise.
DEFAULT_LIST_LIMIT_BYTES = 2 * 1024 * 1024


# --------------------------------------------------------------------------- paths


def _dedupe(paths: Sequence[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved)
        seen.setdefault(key, resolved)
    return list(seen.values())


def _candidate_dirs(directory: str | None = None,
                    include_dirs: Sequence[str] | None = None) -> list[Path]:
    """Search order: explicit ``directory``, workspace dirs, engine defaults."""
    ordered: list[Path] = []
    if directory:
        ordered.append(Path(directory))
    ordered.extend(Path(p) for p in workspace.automation_dirs or [])
    if include_dirs:
        ordered.extend(Path(p) for p in include_dirs)
    ordered.extend(Path(p) for p in include_dirs_default())
    return _dedupe(ordered)


def _resolve_script(name: str, *, directory: str | None = None,
                    include_dirs: Sequence[str] | None = None) -> Path:
    """Resolve a script reference to an existing file."""
    raw = Path(str(name)).expanduser()
    looks_like_path = raw.is_absolute() or raw.parent != Path(".")
    if looks_like_path:
        if raw.is_file():
            return raw
        raise ToolError(f"automation script not found: {raw}")
    tried: list[str] = []
    for base in _candidate_dirs(directory, include_dirs):
        for candidate in (base / raw, base / (str(raw) + ".lua")):
            tried.append(str(candidate))
            if candidate.is_file():
                return candidate
    raise ToolError(
        "automation script {0!r} not found; searched: {1}".format(
            name, ", ".join(tried) or "(no automation directories exist)"
        )
    )


def _script_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _engine(doc, *, selection: Sequence[int] | None = None,
            config: dict[str, Any] | None = None,
            include_dirs: Sequence[str] | None = None,
            project_path: str | None = None,
            video: str | None = None,
            audio: str | None = None,
            keyframes: Sequence[int] | None = None,
            fps: float | None = None,
            video_size: Sequence[int] | None = None,
            timecodes: Sequence[float] | None = None,
            interactive: bool = False,
            dialog_answers: dict[str, Any] | None = None) -> LuaEngine:
    dirs = [str(p) for p in _candidate_dirs(None, include_dirs)]
    return LuaEngine(
        doc,
        selection=selection,
        config=config,
        video=video,
        audio=audio,
        video_size=video_size,
        keyframes=keyframes,
        fps=fps,
        project_path=project_path,
        timecodes=timecodes,
        include_dirs=dirs,
        dialog_answers=dialog_answers,
        interactive=interactive,
        dialog_mode="interactive" if interactive else "raise",
    )


def _run_context(*, doc_id: str | None, selection: Sequence[int] | None,
                 config: dict[str, Any] | None, include_dirs: Sequence[str] | None,
                 project_path: str | None, video: str | None, audio: str | None,
                 fps: float | None, keyframes: Sequence[int] | None,
                 timecodes: Sequence[float] | None, interactive: bool,
                 dialog_answers: dict[str, Any] | None,
                 save_to: str | None, dry_run: bool, keep_undo: bool,
                 mutate: bool) -> dict[str, Any]:
    """Shared plumbing for the run tools: open the doc, build the engine."""
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    path = workspace.path(did)
    engine = _engine(
        doc,
        selection=selection,
        config=config,
        include_dirs=include_dirs,
        project_path=project_path or (str(path) if path else None),
        video=video,
        audio=audio,
        keyframes=keyframes,
        fps=fps,
        timecodes=timecodes,
        interactive=interactive,
        dialog_answers=dialog_answers,
    )
    # Read the counters *before* the pre-run snapshot so ``undo_added`` counts the
    # entry that lets ``ass_undo`` revert this macro.
    undo_before = workspace.undo_depth(did)
    if mutate and not dry_run and keep_undo:
        workspace.snapshot(did)
    before = doc.snapshot()
    return {"doc_id": did, "doc": doc, "engine": engine, "before": before,
            "undo_before": undo_before, "save_to": save_to,
            "dry_run": dry_run, "path": path}


def _finish(ctx: dict[str, Any], result) -> dict[str, Any]:
    """Turn one :class:`~aegisub_mcp.lua.engine.MacroResult` into tool output."""
    doc = ctx["doc"]
    undo_after, redo_after = workspace.undo_depth(ctx["doc_id"])
    undo_before = ctx["undo_before"][0]
    payload: dict[str, Any] = {
        "doc_id": ctx["doc_id"],
        "script": result.kind,
        "unchanged": doc.snapshot() == ctx["before"],
        # ``Workspace.undo_depth`` answers with two counters; flatten them so the
        # output stays JSON-shaped and callers can see what the run pushed.
        "undo_depth": undo_after,
        "undo_added": undo_after - undo_before,
        "redo_depth": redo_after,
    }
    payload.update(result.to_dict())
    if ctx["save_to"]:
        payload["saved"] = workspace.save(ctx["doc_id"], ctx["save_to"])
    return payload


def _failed_load(ctx: dict[str, Any], path: Path | None, kind: str,
                 exc: BaseException) -> dict[str, Any]:
    """Report a script that could not even be *loaded* as ``ok: false``.

    A syntax error or a ``error()`` at file scope is the script author's bug, not
    a bad tool argument, so it is reported like a failed run (with the Lua error
    and traceback) instead of raising -- the caller still gets ``doc_id`` and
    ``unchanged`` to reason about.
    """
    result = MacroResult(
        name=path.name if path is not None else "<inline>",
        kind=kind,
        ok=False,
        error=f"{type(exc).__name__}: {exc}",
    )
    payload = _finish(ctx, result)
    payload["script"] = str(path) if path is not None else "<inline>"
    return payload


# --------------------------------------------------------------------------- tools


def ass_automation_dirs(directory: str | None = None,
                        include_dirs: Sequence[str] | None = None) -> dict[str, Any]:
    """List the automation directories the runner searches, and what is in them.

    Mirrors Aegisub's own search order: the ``--automation-dir`` override (if
    given), the workspace's extra directories, then ``~/.aegisub/automation``,
    ``~/.config/aegisub/automation`` and the bundle vendored with this package.
    Returns each directory with its ``exists`` flag and ``scripts`` count, so a
    caller can tell "no scripts" apart from "wrong path".
    """
    entries: list[dict[str, Any]] = []
    for base in _candidate_dirs(directory, include_dirs):
        scripts = sorted(p.name for p in base.glob("*.lua")) if base.is_dir() else []
        entries.append({
            "path": str(base),
            "exists": base.is_dir(),
            "scripts": len(scripts),
            "sample": scripts[:10],
        })
    include = []
    for base in _dedupe([Path(p) for p in include_dirs_default()]):
        include.append({"path": str(base), "exists": base.is_dir()})
    return {"directories": entries, "include_dirs": include}


def ass_automation_list(directory: str | None = None,
                        include_dirs: Sequence[str] | None = None,
                        recursive: bool = False,
                        execute: bool = True,
                        limit: int = 200,
                        limit_bytes: int = DEFAULT_LIST_LIMIT_BYTES) -> dict[str, Any]:
    """Enumerate Automation 4 scripts and the macros/filters each one registers.

    Scripts are *executed* in a throwaway engine (against an empty scratch
    document) to read their registrations, because Automation 4 has no manifest
    -- ``aegisub.register_macro`` is the only source of truth.  A script that
    fails to load is reported with its ``error`` instead of aborting the listing.
    Pass ``execute=False`` to just list files without loading them.
    """
    files: list[Path] = []
    for base in _candidate_dirs(directory, include_dirs):
        if not base.is_dir():
            continue
        pattern = "**/*.lua" if recursive else "*.lua"
        files.extend(sorted(base.glob(pattern)))
    files = _dedupe([f for f in files if f.is_file()])[: max(0, int(limit))]

    items: list[dict[str, Any]] = []
    for path in files:
        entry: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "bytes": path.stat().st_size,
        }
        if not execute or entry["bytes"] > limit_bytes:
            entry["loaded"] = False
            if execute:
                entry["skipped"] = f"larger than limit_bytes={limit_bytes}"
            items.append(entry)
            continue
        try:
            engine = _engine(_scratch_document())
            engine.load_file(path)
        except Exception as exc:  # noqa: BLE001 - a broken script must not kill the listing
            entry.update({"loaded": False, "error": f"{type(exc).__name__}: {exc}"})
            items.append(entry)
            continue
        entry.update({
            "loaded": True,
            "macros": [i.to_dict() for i in engine.macros],
            "filters": [i.to_dict() for i in engine.filters],
            "unsupported": list(engine.unsupported),
        })
        items.append(entry)
    return {"count": len(items), "scripts": items}


def ass_automation_info(script: str,
                        directory: str | None = None,
                        include_dirs: Sequence[str] | None = None) -> dict[str, Any]:
    """Show one script's path, hash, and the macros/filters it registers.

    Loads the script against an empty scratch document.  ``sha256`` pins the
    exact bytes that were inspected, so a later run can be compared against it.
    """
    path = _resolve_script(script, directory=directory, include_dirs=include_dirs)
    engine = _engine(_scratch_document())
    try:
        engine.load_file(path)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"could not load {path}: {type(exc).__name__}: {exc}") from exc
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _script_sha256(path),
        "macros": [i.to_dict() for i in engine.macros],
        "filters": [i.to_dict() for i in engine.filters],
        "unsupported": list(engine.unsupported),
    }


def ass_automation_run_macro(script: str | None = None,
                             macro: str | None = None,
                             doc_id: str | None = None,
                             selection: Sequence[int] | None = None,
                             config: dict[str, Any] | None = None,
                             dry_run: bool = False,
                             save_to: str | None = None,
                             keep_undo: bool = True,
                             directory: str | None = None,
                             include_dirs: Sequence[str] | None = None,
                             project_path: str | None = None,
                             video: str | None = None,
                             audio: str | None = None,
                             fps: float | None = None,
                             keyframes: Sequence[int] | None = None,
                             timecodes: Sequence[float] | None = None,
                             interactive: bool = False,
                             dialog_answers: dict[str, Any] | None = None,
                             source: str | None = None) -> dict[str, Any]:
    """Run a real Automation 4 *macro* against a workspace document.

    ``script`` is a path or a bare file name resolved against the automation
    directories (``ass_automation_dirs`` lists them); ``source`` runs inline Lua
    instead of a file (handy for one-off macros).  ``macro`` picks one of the
    macros the script registers -- when omitted the first registered macro runs,
    which is what most single-macro scripts expect.

    The macro mutates the document **in the workspace**: subsequent tools see the
    change and ``ass_undo`` can revert it (``keep_undo=True`` snapshots first).
    ``selection`` is the Aegisub GUI selection (line numbers, 1-based in file
    index space) passed as ``selected_lines``/``sel``; several macros (e.g. the
    Karaoke Templater) require it.  ``dialog_answers`` answers
    ``aegisub.dialog`` calls for headless runs; with ``interactive=True`` those
    dialogs are recorded as unsupported instead of raising.

    Returns the engine's ``MacroResult`` fields -- ``ok``, ``error``, ``log``,
    ``changed_lines``, ``structure_changed``, ``undo_points`` -- plus
    ``unchanged`` (byte-level check that nothing moved) and ``saved`` when
    ``save_to`` was given.  A failing script is reported with ``ok: false``.
    """
    if script is None and source is None:
        raise ToolError("pass either script=<file> or source=<lua code>")
    if script is not None and source is not None:
        raise ToolError("pass only one of script/source")
    ctx = _run_context(doc_id=doc_id, selection=selection, config=config,
                       include_dirs=include_dirs, project_path=project_path,
                       video=video, audio=audio, fps=fps, keyframes=keyframes,
                       timecodes=timecodes, interactive=interactive,
                       dialog_answers=dialog_answers, save_to=save_to,
                       dry_run=dry_run, keep_undo=keep_undo, mutate=True)
    engine: LuaEngine = ctx["engine"]
    path: Path | None = None
    try:
        if source is not None:
            engine.load_script(source, name="<inline>")
        elif script is not None:
            path = _resolve_script(script, directory=directory, include_dirs=include_dirs)
            engine.load_file(path)
        result = engine.run_macro(macro or "", dry_run=dry_run)
    except MacroNotFound as exc:
        raise ToolError(str(exc)) from exc
    except (LuaError, OSError) as exc:
        return _failed_load(ctx, path, "macro", exc)
    payload = _finish(ctx, result)
    if path is not None:
        payload["script"] = str(path)
        payload["script_sha256"] = _script_sha256(path)
        payload["registered"] = [i.to_dict() for i in engine.macros]
    return payload


def ass_automation_run_filter(script: str | None = None,
                              filter: str | None = None,
                              doc_id: str | None = None,
                              selection: Sequence[int] | None = None,
                              config: dict[str, Any] | None = None,
                              dry_run: bool = False,
                              save_to: str | None = None,
                              keep_undo: bool = True,
                              directory: str | None = None,
                              include_dirs: Sequence[str] | None = None,
                              project_path: str | None = None,
                              video: str | None = None,
                              audio: str | None = None,
                              fps: float | None = None,
                              keyframes: Sequence[int] | None = None,
                              timecodes: Sequence[float] | None = None,
                              interactive: bool = False,
                              dialog_answers: dict[str, Any] | None = None,
                              source: str | None = None) -> dict[str, Any]:
    """Run one Automation 4 *filter* (the "Apply filter" path) on a document.

    Same contract as :func:`ass_automation_run_macro`; the only difference is
    that Automation 4 calls filters as ``fn(subtitles, config)`` and lets them
    return a modified ``subtitles`` table instead of mutating in place, which the
    engine handles for you.  ``config`` is the filter configuration table that
    the options window would normally produce.
    """
    if script is None and source is None:
        raise ToolError("pass either script=<file> or source=<lua code>")
    if script is not None and source is not None:
        raise ToolError("pass only one of script/source")
    ctx = _run_context(doc_id=doc_id, selection=selection, config=config,
                       include_dirs=include_dirs, project_path=project_path,
                       video=video, audio=audio, fps=fps, keyframes=keyframes,
                       timecodes=timecodes, interactive=interactive,
                       dialog_answers=dialog_answers, save_to=save_to,
                       dry_run=dry_run, keep_undo=keep_undo, mutate=True)
    engine: LuaEngine = ctx["engine"]
    path: Path | None = None
    try:
        if source is not None:
            engine.load_script(source, name="<inline>")
        elif script is not None:
            path = _resolve_script(script, directory=directory, include_dirs=include_dirs)
            engine.load_file(path)
        result = engine.run_filter(filter or "", dry_run=dry_run)
    except MacroNotFound as exc:
        raise ToolError(str(exc)) from exc
    except (LuaError, OSError) as exc:
        return _failed_load(ctx, path, "filter", exc)
    payload = _finish(ctx, result)
    if path is not None:
        payload["script"] = str(path)
        payload["script_sha256"] = _script_sha256(path)
        payload["registered"] = [i.to_dict() for i in engine.filters]
    return payload


def ass_automation_run_filters(script: str | None = None,
                               doc_id: str | None = None,
                               selection: Sequence[int] | None = None,
                               config: dict[str, Any] | None = None,
                               dry_run: bool = False,
                               save_to: str | None = None,
                               keep_undo: bool = True,
                               directory: str | None = None,
                               include_dirs: Sequence[str] | None = None,
                               source: str | None = None) -> dict[str, Any]:
    """Run *every* filter a script registers, in Aegisub's own order.

    Aegisub applies filters in ``(priority, registration order)`` -- the same
    ordering used here -- and stops at the first failing filter.  Use this to
    reproduce what "Apply all filters" does to a selection.
    """
    if script is None and source is None:
        raise ToolError("pass either script=<file> or source=<lua code>")
    if script is not None and source is not None:
        raise ToolError("pass only one of script/source")
    ctx = _run_context(doc_id=doc_id, selection=selection, config=config,
                       include_dirs=include_dirs, project_path=None,
                       video=None, audio=None, fps=None, keyframes=None,
                       timecodes=None, interactive=False, dialog_answers=None,
                       save_to=save_to, dry_run=dry_run, keep_undo=keep_undo,
                       mutate=True)
    engine: LuaEngine = ctx["engine"]
    try:
        if source is not None:
            engine.load_script(source, name="<inline>")
        elif script is not None:
            engine.load_file(_resolve_script(script, directory=directory,
                                             include_dirs=include_dirs))
        results = engine.run_filters(selection=selection, config=config, dry_run=dry_run)
    except (LuaError, OSError) as exc:
        return {"doc_id": ctx["doc_id"], "count": 0, "ok": False,
                "error": f"{type(exc).__name__}: {exc}", "results": [],
                "unchanged": ctx["doc"].snapshot() == ctx["before"], "saved": None}
    run = [_finish(ctx, r) for r in results]
    return {
        "doc_id": ctx["doc_id"],
        "count": len(run),
        "ok": all(r["ok"] for r in run),
        "unchanged": ctx["doc"].snapshot() == ctx["before"],
        "results": run,
        "saved": workspace.save(ctx["doc_id"], save_to) if save_to else None,
    }


def ass_automation_from_source(code: str,
                               macro: str | None = None,
                               kind: str = "macro",
                               doc_id: str | None = None,
                               selection: Sequence[int] | None = None,
                               config: dict[str, Any] | None = None,
                               dry_run: bool = False,
                               save_to: str | None = None,
                               include_dirs: Sequence[str] | None = None) -> dict[str, Any]:
    """Run inline Lua source as an Automation 4 script (no file needed).

    ``kind`` selects the registration the code is expected to use: ``"macro"``
    or ``"filter"``.  Everything else matches
    :func:`ass_automation_run_macro` / :func:`ass_automation_run_filter`.  This
    is the fastest way to try a snippet such as::

        aegisub.register_macro("bump", "shift +1s", function(subs, sel)
          for _, i in ipairs(sel) do
            subs[i].start_time = subs[i].start_time + 1000
            subs[i].end_time = subs[i].end_time + 1000
          end
        end)
    """
    kind = (kind or "macro").lower()
    if kind not in {"macro", "filter"}:
        raise ToolError("kind must be 'macro' or 'filter'")
    common: dict[str, Any] = {
        "doc_id": doc_id,
        "selection": selection,
        "config": config,
        "dry_run": dry_run,
        "save_to": save_to,
        "include_dirs": include_dirs,
        "source": code,
    }
    if kind == "filter":
        return ass_automation_run_filter(filter=macro, **common)
    return ass_automation_run_macro(macro=macro, **common)


def _scratch_document():
    """An empty document used only to load scripts for inspection."""
    from ..asscore.document import AssDocument

    return AssDocument.new(play_res=(1920, 1080))


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
