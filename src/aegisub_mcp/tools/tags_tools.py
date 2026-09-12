"""Override-tag and typesetting tools for the ASS workspace.

Every public function in this module is an MCP tool: a module-level ``ass_*``
callable with full type hints that returns a JSON-serialisable ``dict``.  The
module wraps :mod:`aegisub_mcp.asscore.tags` (the override-tag engine) and
:mod:`aegisub_mcp.asscore.measure` (libass ink measurement, used by
:func:`ass_swap_an_pos`).

Plain-character indices versus raw indices
------------------------------------------
Two different coordinate systems appear all over this module and they are
**never** mixed up:

* A **plain index** (also called a *visible* or *plain-text* index) counts only
  the characters the viewer actually sees.  Override blocks such as
  ``{\\an7}`` and the line breaks ``\\N``/``\\n`` containing escape sequences
  occupy **no** plain position.  In ``"ab{\\i1}cd"`` the visible characters are
  ``a``(0) ``b``(1) ``c``(2) ``d``(3), so ``plain_len`` is 4 and the plain range
  ``[1, 3)`` covers ``"bc"``.
* A **raw index** counts every code point of the stored ``Text`` field,
  braces included.  In the same string ``"ab{\\i1}cd"`` the raw positions of
  ``c``/``d`` are 7/8, and raw positions 2..6 fall *inside* the override block
  ``{\\i1}``.

Tool arguments whose name contains ``plain`` (``plain_index``) are plain
indices.  :func:`ass_wrap_range` is the only tool that accepts either system:
``scope="plain"`` (the default) reads ``start``/``end`` as plain indices while
``scope="raw"`` reads them as raw indices and refuses ranges whose boundary
falls inside an override block (splitting a block would corrupt the line).
Every returned dict states, per line, both index systems where relevant
(``start``/``end`` for raw, ``plain_start``/``plain_end`` for plain, plus the
``scope`` that was requested).

ASS escaping rules
------------------
Only a ``\\`` inside a ``{...}`` block starts a tag; braces always delimit an
override block; a visible ``\\`` needs ``\\\\`` (or ``\\h`` for a hard space).
Consequently every mutating tool here:

* validates that tag names and override payloads contain no ``{`` or ``}`` and
  no line breaks, so a caller cannot inject a brace and leave a block open;
* never writes a partially built block — tags are always inserted together with
  their opening and closing brace;
* re-parses its own output and refuses to return text that does not survive a
  byte-exact ``parse`` -> ``serialize`` round trip (see :func:`_sane`), which
  catches "override block without a closing brace" bugs automatically.

State and snapshots
-------------------
Indices are 0-based and follow ``AssDocument.events()`` order, comments
included.  Selections are resolved with ``base.resolve_indices``.  Every tool
that writes to a document calls ``workspace.snapshot(doc_id)`` *before* the
first write so :func:`ass_undo` can roll the change back.  Tools can also be
called on a raw string (``text="..."``); in that case nothing is written and the
new text is returned in ``text``.  Results always say which input was used
through the ``source`` field (``"text"``, ``"line"`` or ``"selection"``).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..asscore import assutil as U
from ..asscore import measure as M
from ..asscore import tags as T
from ..asscore.document import AssDocument, EventEntry
from .base import (
    ToolError,
    entry_at,
    ok,
    resolve_indices,
    workspace,
)

__all__ = ["register"]

#: ``source`` value used when the caller passed ``text=``.
SOURCE_TEXT = "text"
#: ``source`` value used when the caller passed ``index=`` (with ``doc_id=``).
SOURCE_LINE = "line"
#: ``source`` value used when the caller passed ``selection=``.
SOURCE_SELECTION = "selection"

_NAME_RE = re.compile(r"[1-4]?[A-Za-z]+")
#: Legacy SSA ``\a`` alignment codes -> ASS ``\an`` codes (Aegisub's own table).
SSA_ALIGNMENT_TO_AN = {1: 1, 2: 2, 3: 3, 5: 7, 6: 8, 7: 9, 9: 4, 10: 5, 11: 6}
#: Inverse of :data:`SSA_ALIGNMENT_TO_AN` (ASS ``\an`` -> legacy SSA ``\a``).
AN_TO_SSA_ALIGNMENT = {v: k for k, v in SSA_ALIGNMENT_TO_AN.items()}

_TAG_NAME_RE = re.compile(r"[1-4]?[A-Za-z]+")
_AN_VALUE_RE = re.compile(r"\s*(-?\d+)")
_COORD_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+")
#: A drawing path must open with a command: ``m``/``n`` (move / spline start)
#: followed later by ``l``/``b``/``s``/``p``/``c`` (line, bezier, b-spline,
#: extend, close).  Used to catch ``clip="..."`` typos before they reach libass.
_DRAWING_COMMAND_RE = re.compile(r"(?:^|\s)[mnlbspc](?=\s|$|[-+0-9.])", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _num(value: Any) -> str:
    """Format a number the way Aegisub does: no trailing ``.0``."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return U.round_coord(number, 2)


def _as_list(value: Any, what: str = "value") -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    raise ToolError(f"{what} must be a string or a list of strings")


def _split_names(value: Any, what: str = "value") -> tuple[list[str], list[str]]:
    """Split a name specification into ``(tag_names, group_names)``.

    Items may be separated by commas or semicolons when a string is given.  An
    item that matches a key of :data:`tags.TAG_GROUPS` becomes a group, anything
    else is treated as a tag name (a leading backslash is accepted and ignored).
    """
    if value is None:
        return [], []
    if isinstance(value, str):
        items: list[Any] = re.split(r"[,;]", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        raise ToolError(f"{what} must be a string or a list")
    names: list[str] = []
    groups: list[str] = []
    for item in items:
        token = str(item).strip()
        if not token:
            continue
        bare = token.lstrip("\\")
        if bare in T.TAG_GROUPS:
            groups.append(bare)
            continue
        names.append(_clean_name(token, what))
    return names, groups


def _clean_name(name: Any, what: str = "tag name") -> str:
    stripped = str(name or "").strip().lstrip("\\")
    if not stripped:
        raise ToolError(f"{what} must not be empty")
    if not _TAG_NAME_RE.fullmatch(stripped):
        raise ToolError(
            f"invalid {what} {str(name)!r}: expected something like 'pos', 'an', "
            f"'t' or '1c' (letters, optionally prefixed with 1-4)"
        )
    return stripped


def _group_names(value: Any, what: str = "group") -> list[str]:
    out: list[str] = []
    for item in _as_list(value, what):
        token = str(item).strip().lower()
        if not token:
            continue
        if token not in T.TAG_GROUPS:
            raise ToolError(
                f"unknown tag group {token!r}; known groups: {sorted(T.TAG_GROUPS)}"
            )
        out.append(token)
    return out


def _no_braces(payload: str, what: str) -> str:
    if "{" in payload or "}" in payload:
        raise ToolError(
            f"{what} must not contain '{{' or '}}': braces delimit override blocks, "
            f"and a stray brace would corrupt the line's escaping"
        )
    if "\n" in payload or "\r" in payload:
        raise ToolError(f"{what} must not contain line breaks")
    return payload


def _override_body(override: Any, what: str = "override") -> str:
    """Normalise an override payload to the *body* of a block.

    Surrounding braces are accepted and stripped, a missing leading backslash is
    added, and braces/line breaks inside the payload are rejected.
    """
    if override is None:
        raise ToolError(f"{what} is required")
    body = str(override).strip()
    if body.startswith("{"):
        body = body[1:]
    if body.endswith("}"):
        body = body[:-1]
    body = body.strip()
    if not body:
        raise ToolError(f"{what} is empty")
    _no_braces(body, what)
    if not body.startswith("\\"):
        body = "\\" + body
    return body


def _sane(text: str, what: str) -> str:
    """Guard: the produced text must survive a byte-exact parse/serialize cycle."""
    try:
        round_tripped = T.serialize(T.parse(text))
    except Exception as exc:  # pragma: no cover - defensive
        raise ToolError(f"internal error: {what} produced unparsable text: {exc}") from exc
    if round_tripped != text:
        raise ToolError(
            f"internal error: {what} produced malformed override text "
            f"(an override block without a closing brace?)"
        )
    return text


def _groups_of(name: str) -> list[str]:
    return sorted(g for g, members in T.TAG_GROUPS.items() if name in members)


def _group_histogram(names: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for group, members in T.TAG_GROUPS.items():
        total = sum(count for name, count in names.items() if name in members)
        if total:
            out[group] = total
    return out


@dataclass
class _Target:
    """The string a single-line tool operates on, plus where it came from."""

    text: str
    source: str
    index: int | None = None
    doc_id: str | None = None
    doc: AssDocument | None = None
    entry: EventEntry | None = None

    def where(self) -> dict[str, Any]:
        return {"source": self.source, "index": self.index, "doc_id": self.doc_id}


def _target(text: Any = None, index: Any = None, doc_id: str | None = None) -> _Target:
    """Resolve ``text=`` / ``index=``+``doc_id=`` into a :class:`_Target`."""
    if text is not None and index is not None:
        raise ToolError("pass either text= (a raw line) or index= (a line of doc_id=), not both")
    if index is not None:
        did = workspace.resolve_id(doc_id)
        doc = workspace.get(did)
        try:
            idx = int(index)
        except (TypeError, ValueError):
            raise ToolError(f"index must be an integer, got {index!r}") from None
        entry = entry_at(doc, idx)
        return _Target(entry.text, SOURCE_LINE, idx, did, doc, entry)
    if text is None:
        raise ToolError("provide text= (a raw line) or index= together with doc_id=")
    if not isinstance(text, str):
        raise ToolError("text must be a string")
    return _Target(text, SOURCE_TEXT, None, None, None, None)


def _commit(target: _Target, new_text: str, in_place: bool) -> bool:
    """Snapshot and write ``new_text`` back to the document; returns "written"."""
    if target.source != SOURCE_LINE or not in_place:
        return False
    if target.entry is None or target.doc is None:  # pragma: no cover - defensive
        raise ToolError("internal error: line target without a document")
    workspace.snapshot(target.doc_id)
    target.entry.set("Text", new_text)
    target.doc.dirty = True
    return True


def _doc_and_indices(
    selection: Any, doc_id: str | None
) -> tuple[str, AssDocument, list[int]]:
    did = workspace.resolve_id(doc_id)
    doc = workspace.get(did)
    indices = resolve_indices(selection, doc, selection=workspace.selection)
    return did, doc, indices


def _batch_write(
    doc: AssDocument,
    did: str | None,
    updates: Sequence[tuple[int, str]],
    *,
    fields: Sequence[tuple[int, str, str]] = (),
) -> int:
    """Snapshot once, then write every ``(index, text)`` pair.

    ``fields`` carries extra ``(index, field, value)`` writes such as the
    ``MarginV`` rewrite of ``ass_swap_an_pos`` in margin mode; they land under
    the same single snapshot.  Returns the number of *lines* written.
    """
    if not updates and not fields:
        return 0
    workspace.snapshot(did)
    events = doc.events()
    for index, text in updates:
        events[index].set("Text", text)
    for index, field, value in fields:
        events[index].set(field, str(value))
    doc.dirty = True
    return len({index for index, _ in updates} | {index for index, _, _ in fields})


def _override_pieces(spec: Any, what: str) -> list[str]:
    """Split an override string into its top-level tag pieces."""
    return [t.raw for t in T.parse_tags(spec)]


# --------------------------------------------------------------------------- #
# 1. parsing / inspection
# --------------------------------------------------------------------------- #


def ass_parse_text(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Tokenise an ASS line into its ordered segments and override tags.

    Args:
        text: a raw ASS ``Text`` field (tags included).  Use this to inspect a
            line that is not (yet) in a document.
        index: 0-based line index in ``doc_id``; the line's current ``Text``
            field is read from the document.
        doc_id: document holding ``index``; the current document when omitted.
            Passing both ``text`` and ``index`` is an error.

    Returns a dict with:

    ``source``/``index``/``doc_id``
        which input was used (``"text"`` or ``"line"``).
    ``raw``, ``raw_length``
        the line exactly as stored (raw index space).
    ``plain_text``, ``plain_length``, ``characters``
        the visible text and, for every visible character, its index in both
        spaces: ``{"plain_index": i, "char": c, "raw_index": r}``.
    ``segments``
        ordered alternating text runs and ``{...}`` blocks.  A text segment
        carries ``text`` plus ``start``/``end`` (raw indices) and
        ``plain_start``/``plain_end``.  A block segment carries ``raw``
        (including braces), ``inner``, ``block`` (0-based block number),
        ``is_override`` (``True`` when it holds at least one ``\\`` tag; a block
        without tags is an ASS comment block), the same four offset fields
        (``plain_start == plain_end`` for blocks) and ``tags``.
    ``tags``
        the flat list of every tag in line order, each with ``name`` (canonical,
        lowercase, e.g. ``kf`` for ``\\K``), ``argument`` (``""`` for
        valueless tags), ``raw`` (exact source text), ``paren`` (whether the
        argument was written in parentheses), ``is_override`` and ``block``.
        Every parsed tag has ``is_override`` ``True`` because only
        backslash-prefixed content is parsed as a tag; see the block-level flag
        for comment blocks.
    ``summary``
        per-line flags: ``blocks``, ``tags``, ``tag_names`` (histogram),
        ``tag_groups`` (histogram by tag family), ``has_drawing`` (drawing mode
        is still *active at the end* of the line, i.e. an unmatched ``\\p``),
        ``drawing_state``, ``has_karaoke``, ``has_transform``, ``has_clip`` and
        ``plain_length``.  Use ``tag_names["p"]``/``tag_groups["drawing"]`` to
        detect a drawing that is switched off again by ``\\p0``.

    Read-only; no snapshot.  ``start``/``end`` are raw indices, everything with
    ``plain`` in its name is a plain (visible character) index.
    """
    target = _target(text=text, index=index, doc_id=doc_id)
    parsed = T.parse(target.text)
    segments: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []
    raw_cursor = 0
    plain_cursor = 0
    block_no = 0
    for segment in parsed.segments:
        rendered = segment.render()
        if isinstance(segment, T.TagBlock):
            rows = []
            for tag in segment.tags:
                row = {
                    "name": tag.name,
                    "argument": tag.arg,
                    "raw": tag.raw,
                    "paren": bool(tag.paren),
                    "is_override": True,
                    "block": block_no,
                }
                rows.append(row)
                tag_rows.append(row)
            segments.append(
                {
                    "kind": "block",
                    "block": block_no,
                    "raw": rendered,
                    "inner": rendered[1:-1] if rendered.startswith("{") else rendered,
                    "start": raw_cursor,
                    "end": raw_cursor + len(rendered),
                    "plain_start": plain_cursor,
                    "plain_end": plain_cursor,
                    "is_override": bool(rows),
                    "tags": rows,
                }
            )
            block_no += 1
            raw_cursor += len(rendered)
        else:
            body = segment.text
            segments.append(
                {
                    "kind": "text",
                    "text": body,
                    "start": raw_cursor,
                    "end": raw_cursor + len(body),
                    "plain_start": plain_cursor,
                    "plain_end": plain_cursor + len(body),
                }
            )
            raw_cursor += len(rendered)
            plain_cursor += len(body)

    plain = parsed.plain_text()
    names = Counter(row["name"] for row in tag_rows)
    summary = {
        "blocks": block_no,
        "tags": len(tag_rows),
        "tag_names": dict(sorted(names.items())),
        "tag_groups": _group_histogram(dict(names)),
        "has_drawing": T.drawing_state(target.text) > 0,
        "drawing_state": T.drawing_state(target.text),
        "has_karaoke": any(row["name"] in T.KARAOKE_TAGS for row in tag_rows),
        "has_transform": any(row["name"] == "t" for row in tag_rows),
        "has_clip": any(row["name"] in T.CLIP_TAGS for row in tag_rows),
        "plain_length": len(plain),
    }
    characters = [
        {"plain_index": i, "char": char, "raw_index": raw}
        for i, (raw, char) in enumerate(parsed.char_positions())
    ]
    return ok(
        **target.where(),
        raw=target.text,
        raw_length=len(target.text),
        plain_text=plain,
        plain_length=len(plain),
        characters=characters,
        segments=segments,
        tags=tag_rows,
        summary=summary,
    )


def ass_plain_text(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
    keep: Any = "",
) -> dict[str, Any]:
    """Strip every override tag, optionally keeping some tags or tag groups.

    Args:
        text / index / doc_id: the line to inspect (raw string or a 0-based line
            index of ``doc_id``); exactly one source must be given.
        keep: ``""`` (or ``None``) removes everything.  Otherwise a comma/space
            separated string or a list whose items are either a tag name
            (``"pos"``, ``"\\an"``, ``"1c"``) or a tag group name.  Group names
            win over tag names and expand to the whole family, e.g. ``"clip"``
            keeps ``\\clip`` and ``\\iclip``, ``"fade"`` keeps ``\\fad`` and
            ``\\fade``.  Known groups: transform, fade, clip, drawing, karaoke,
            layout, color, style, reset, animation.

    Returns ``{"source", "index", "doc_id", "text", "plain_text", "keep_names",
    "keep_groups", "kept", "changed"}`` where ``text`` is the stripped line (it
    still contains the kept override blocks), ``plain_text`` is the fully
    tag-free visible text and ``kept`` lists the kept tags that were actually
    present.  Read-only; no snapshot, no indices.
    """
    target = _target(text=text, index=index, doc_id=doc_id)
    names, groups = _split_names(keep, "keep")
    stripped = T.strip_tags(
        target.text,
        keep=names or None,
        keep_groups=groups or None,
    )
    present = [t.name for t in T.parse(stripped).tags()]
    return ok(
        **target.where(),
        text=stripped,
        plain_text=T.plain_text(stripped),
        keep_names=names,
        keep_groups=groups,
        kept=present,
        changed=stripped != target.text,
    )


# --------------------------------------------------------------------------- #
# 2. stripping
# --------------------------------------------------------------------------- #


def ass_strip_tags(
    selection: Any = None,
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
    keep: Any = None,
    keep_groups: Any = None,
    remove: Any = None,
    remove_groups: Any = None,
    keep_drawing: bool = False,
    keep_karaoke: bool = False,
    keep_clip: bool = False,
    in_place: bool = False,
) -> dict[str, Any]:
    """Remove override tags from a line, a raw string or a whole selection.

    This is the bulk "clean up the tags" tool.  The keyword names mirror
    ``asscore.tags.strip_tags`` (``keep``, ``keep_groups``, ``remove_groups``,
    ``keep_drawing``, ``keep_karaoke``); ``remove`` and ``keep_clip`` are
    additions of this tool, where ``remove`` lists individual tag names to drop
    *before* stripping and ``keep_clip`` is shorthand for
    ``keep_groups=["clip"]``.

    Args:
        selection: a selection spelling (``None`` = every line, ``"0-4"``,
            ``{"style": "Default"}`` ...).  When given, every selected line is
            processed and ``text``/``index`` must be omitted.
        text: raw ASS line to process instead of a selection.
        index: 0-based index of the line to process inside ``doc_id``.
        doc_id: document used by ``selection``/``index``; the current one when
            omitted.
        keep: tag names (string or list) to keep, e.g. ``"pos,an"``.  Group
            names are accepted too and move into ``keep_groups``.
        keep_groups: tag group names to keep (transform, fade, clip, drawing,
            karaoke, layout, color, style, reset, animation).
        remove: tag names to delete outright before stripping (string or list).
        remove_groups: tag group names to delete outright.
        keep_drawing: keep ``\\p``/``\\pbo`` tags so drawings survive.
        keep_karaoke: keep ``\\k``-family tags so karaoke timings survive.
        keep_clip: keep ``\\clip``/``\\iclip``.
        in_place: write the result back to the document (line/selection mode
            only).  Snapshot-backed; the raw-string mode never writes.

    Returns, in single-line mode: ``{"source", "index", "doc_id", "text",
    "plain_text", "changed", "written", "keep_names", "keep_groups",
    "remove_names", "remove_groups", "keep_drawing", "keep_karaoke",
    "keep_clip"}``; in selection mode: ``{"source": "selection", "doc_id",
    "count", "changed", "written", "in_place", "lines": [{"index", "before",
    "after", "changed"}]}``.  Plain text is always tag-free; kept tags stay
    inside their override blocks.
    """
    keep_names, keep_gs = _split_names(keep, "keep")
    keep_gs.extend(_group_names(keep_groups, "keep_groups"))
    if keep_clip:
        keep_gs.append("clip")
    remove_names, remove_gs = _split_names(remove, "remove")
    remove_gs.extend(_group_names(remove_groups, "remove_groups"))
    keep_gs = sorted(set(keep_gs))
    remove_gs = sorted(set(remove_gs))
    remove_names = sorted(set(remove_names))

    def strip_one(line: str) -> str:
        work = T.remove_tags(line, remove_names) if remove_names else line
        return T.strip_tags(
            work,
            keep=keep_names or None,
            keep_groups=keep_gs or None,
            remove_groups=remove_gs or None,
            keep_drawing=bool(keep_drawing),
            keep_karaoke=bool(keep_karaoke),
        )

    if selection is not None:
        if text is not None or index is not None:
            raise ToolError("pass either selection= or text=/index=, not both")
        did, doc, indices = _doc_and_indices(selection, doc_id)
        events = doc.events()
        updates: list[tuple[int, str]] = []
        rows: list[dict[str, Any]] = []
        for i in indices:
            before = events[i].text
            after = strip_one(before)
            rows.append(
                {
                    "index": i,
                    "before": before,
                    "after": after,
                    "changed": after != before,
                }
            )
            if after != before:
                updates.append((i, after))
        written = _batch_write(doc, did, updates) if in_place else 0
        return ok(
            source=SOURCE_SELECTION,
            doc_id=did,
            count=len(indices),
            changed=len(updates),
            written=written,
            in_place=bool(in_place),
            keep_names=keep_names,
            keep_groups=keep_gs,
            remove_names=remove_names,
            remove_groups=remove_gs,
            lines=rows,
        )

    target = _target(text=text, index=index, doc_id=doc_id)
    after = strip_one(target.text)
    written = _commit(target, after, in_place) if after != target.text else False
    return ok(
        **target.where(),
        text=after,
        plain_text=T.plain_text(after),
        changed=after != target.text,
        written=written,
        keep_names=keep_names,
        keep_groups=keep_gs,
        remove_names=remove_names,
        remove_groups=remove_gs,
        keep_drawing=bool(keep_drawing),
        keep_karaoke=bool(keep_karaoke),
        keep_clip=bool(keep_clip),
    )


# --------------------------------------------------------------------------- #
# 3. single-tag edits
# --------------------------------------------------------------------------- #


#: ``\an``/``\a`` take a small integer: the numpad alignment runs 1..9 and the
#: legacy SSA alignment 1..11.
_ALIGN_RANGE = {"an": (1, 9), "a": (1, 11)}


def _check_alignment(clean_name: str, arg: str) -> None:
    """Reject an out-of-range ``\\an``/``\\a`` argument — a classic user error.

    Anything that is not an integer is refused too; passing ``"7,7"`` to ``\\an``
    silently produces a tag libass ignores, which is worth a ToolError.
    """
    bounds = _ALIGN_RANGE.get(clean_name.lower())
    if bounds is None or arg.strip() == "":
        return
    match = _AN_VALUE_RE.match(arg)
    if match is None or match.end() != len(arg.rstrip()):
        raise ToolError(f"\\{clean_name} takes an integer alignment, got {arg!r}")
    value = int(match.group(1))
    low, high = bounds
    if not low <= value <= high:
        raise ToolError(f"\\{clean_name} must be between {low} and {high}, got {value}")


def _tag_payload(body: str, name: str) -> tuple[str, bool]:
    """Split a rendered tag body (``\\name``, ``\\name(arg)``) into arg + paren."""
    if body.startswith("\\" + name + "(") and body.endswith(")"):
        return body[len(name) + 2 : -1], True
    if body == "\\" + name:
        return "", False
    return body[len(name) + 1 :], False


def _set_tag_once(line: str, body: str, name: str, where: str) -> tuple[str, list[str]]:
    """Apply ``body`` to ``line`` at the requested ``where`` position."""
    warnings: list[str] = []
    if where in ("prepend", "leading", "start"):
        return T.prepend_tags(line, body), warnings
    if where in ("after_first_block", "first_block", "first"):
        if T.get_tag(line, name) is not None:
            # the tag is already there: update it instead of stacking a duplicate
            payload, paren = _tag_payload(body, name)
            return T.set_tag(line, name, payload, paren=paren), warnings
        if line.startswith("{") and "}" in line:
            end = line.index("}")
            return "{" + line[1:end] + body + "}" + line[end + 1 :], warnings
        return "{" + body + "}" + line, warnings
    if where in ("prepend_block", "new_block"):
        return "{" + body + "}" + line, warnings
    if where in ("append", "end"):
        return T.append_tags(line, body), warnings
    if where in ("wrap", "wrap_line"):
        out, warn = T.wrap_range(line, 0, T.plain_len(line), body)
        return out, list(warn)
    if where in ("append_to_block", "last_block"):
        blocks = T.parse(line).blocks()
        if not blocks:
            return T.append_tags(line, body), ["line has no override block: appended a new one"]
        return _append_to_last_block(line, body), warnings
    raise ToolError(
        f"unknown where={where!r}; use 'prepend', 'after_first_block' (default), "
        f"'prepend_block', 'append' or 'wrap'"
    )


def _append_to_last_block(line: str, body: str) -> str:
    parsed = T.parse(line)
    blocks = parsed.blocks()
    if not blocks:
        return T.append_tags(line, body)
    last = blocks[-1]
    rendered = parsed.render()
    # Rebuild by walking the segments: the last block is the last occurrence of
    # the last block's rendered text.
    needle = last.render()
    at = rendered.rfind(needle)
    if at < 0:  # pragma: no cover - defensive
        raise ToolError("internal error: could not locate the last override block")
    return rendered[: at + len(needle) - 1] + body + rendered[at + len(needle) - 1 :]


def ass_set_tag(
    selection: Any = None,
    index: int | None = None,
    text: str | None = None,
    name: str = "",
    arg: Any = "",
    value: Any = None,
    doc_id: str | None = None,
    where: str = "after_first_block",
    only_if_missing: bool = False,
    in_place: bool = False,
) -> dict[str, Any]:
    """Insert (or update) a single override tag on a line or a selection.

    Args:
        selection: selection spelling; when given, ``text``/``index`` must be
            omitted and every selected line is processed.
        index / text / doc_id: single-line source (0-based index inside
            ``doc_id``, or a raw string).
        name: tag name without the backslash, e.g. ``"fad"``, ``"an"``,
            ``"pos"``, ``"fscx"`` or ``"1c"``.  A leading backslash is allowed.
        arg: the tag argument, e.g. ``"200,200"`` for ``\\fad(200,200)`` or
            ``"8"`` for ``\\an8``.  Parentheses are added automatically when the
            argument needs them.  Braces and line breaks are rejected, and so is
            an out-of-range alignment for ``\\an`` (1..9) or legacy ``\\a``
            (1..11) — those are ignored by libass when wrong, so they are caught
            here instead.
        value: alias for ``arg``; when not ``None`` it wins (handy for numeric
            callers, e.g. ``value=8``).
        doc_id: document for ``index``/``selection``.
        where: where to put the tag.  ``"prepend"`` merges the tag into a new
            leading block (the line's existing first block is reused so that
            repeated tags are updated rather than duplicated);
            ``"after_first_block"`` (default) appends the tag to the end of the
            line's first override block, creating that block when the line has
            none; ``"prepend_block"`` always inserts a brand new leading block;
            ``"append"`` adds a block at the very end of the line; ``"wrap"``
            wraps every visible character and restores the previous value
            afterwards (see :func:`ass_wrap_range`).
        only_if_missing: leave the line untouched when a tag with this name is
            already present anywhere in the line.
        in_place: write back to the document (single-line and selection mode).
            Snapshot-backed; the raw-string mode never writes.

    Indices: this tool edits whole override blocks, so it never takes character
    offsets.  Neither the plain-character index of the visible text nor the raw
    index of the line is used or reported — but every ``where`` mode leaves both
    maps of the visible characters unchanged (a tag is only ever inserted
    *between* characters).

    Returns ``{"source", "index", "doc_id", "name", "argument", "tag", "where",
    "text", "changed", "written", "skipped", "reason", "warnings"}`` for a
    single line (``tag`` is the exact tag text produced, e.g. ``\\fad(200,200)``)
    or ``{"source": "selection", "doc_id", "name", "tag", "where", "count",
    "changed", "written", "in_place", "lines": [...]}`` for a selection.
    ``duration``-style arguments are passed through verbatim.
    """
    clean = _clean_name(name, "tag name")
    raw_arg = arg if value is None else value
    arg_text = "" if raw_arg is None else str(raw_arg)
    # surrounding whitespace would make libass ignore the tag (\an( 8 ) is not
    # an alignment), so it is stripped before the tag body is built
    arg_text = arg_text.strip() if arg_text.strip() else arg_text
    _no_braces(arg_text, "tag argument")
    _check_alignment(clean, arg_text)
    body = T.tag(clean, arg_text)
    _sane("{" + body + "}", "ass_set_tag")

    if selection is not None:
        if text is not None or index is not None:
            raise ToolError("pass either selection= or text=/index=, not both")
        did, doc, indices = _doc_and_indices(selection, doc_id)
        events = doc.events()
        updates: list[tuple[int, str]] = []
        rows: list[dict[str, Any]] = []
        for i in indices:
            before = events[i].text
            if only_if_missing and T.get_tag(before, clean) is not None:
                rows.append(
                    {
                        "index": i,
                        "before": before,
                        "after": before,
                        "changed": False,
                        "skipped": True,
                        "reason": f"\\{clean} already present",
                    }
                )
                continue
            after, warnings = _set_tag_once(before, body, clean, where)
            _sane(after, "ass_set_tag")
            rows.append(
                {
                    "index": i,
                    "before": before,
                    "after": after,
                    "changed": after != before,
                    "skipped": False,
                    "warnings": warnings,
                }
            )
            if after != before:
                updates.append((i, after))
        written = _batch_write(doc, did, updates) if in_place else 0
        return ok(
            source=SOURCE_SELECTION,
            doc_id=did,
            name=clean,
            tag=body,
            where=where,
            count=len(indices),
            changed=len(updates),
            written=written,
            in_place=bool(in_place),
            lines=rows,
        )

    target = _target(text=text, index=index, doc_id=doc_id)
    if only_if_missing and T.get_tag(target.text, clean) is not None:
        return ok(
            **target.where(),
            name=clean,
            argument=arg_text,
            tag=body,
            where=where,
            text=target.text,
            changed=False,
            written=False,
            skipped=True,
            reason=f"\\{clean} already present",
            warnings=[],
        )
    after, warnings = _set_tag_once(target.text, body, clean, where)
    _sane(after, "ass_set_tag")
    written = _commit(target, after, in_place) if after != target.text else False
    return ok(
        **target.where(),
        name=clean,
        argument=arg_text,
        tag=body,
        where=where,
        text=after,
        changed=after != target.text,
        written=written,
        skipped=False,
        warnings=warnings,
    )


def ass_remove_tag(
    selection: Any = None,
    index: int | None = None,
    text: str | None = None,
    names: Any = None,
    doc_id: str | None = None,
    in_place: bool = False,
) -> dict[str, Any]:
    """Delete every occurrence of the named override tags.

    Args:
        selection / index / text / doc_id: the usual three input modes.
        names: a single tag name (``"pos"``) or a list (``["pos", "move"]``);
            a comma separated string works too.  Group names are accepted and
            expand to the whole family (``"clip"`` removes ``\\clip`` and
            ``\\iclip``).
        in_place: write back to the document (snapshot-backed).  The raw-string
            mode never writes.

    Returns, single-line: ``{"source", "index", "doc_id", "names", "removed",
    "removed_count", "text", "plain_text", "changed", "written"}`` where
    ``removed`` lists every deleted occurrence as ``{"name", "argument",
    "raw", "block"}``; selection mode returns
    ``{"source": "selection", "doc_id", "names", "count", "changed", "written",
    "in_place", "removed_count", "lines": [{"index", "before", "after",
    "removed", "changed"}]}``.  Removing a tag never touches the visible text
    and never leaves an unclosed block: emptied blocks are dropped entirely.
    """
    wanted_names, wanted_groups = _split_names(names, "names")
    wanted: set[str] = set(wanted_names)
    for group in wanted_groups:
        wanted.update(T.TAG_GROUPS[group])
    if not wanted:
        raise ToolError("names= is required (a tag name or a list of tag names)")
    ordered = sorted(wanted)

    def removed_rows(line: str) -> list[dict[str, Any]]:
        rows = []
        block = 0
        parsed = T.parse(line)
        for segment in parsed.segments:
            if isinstance(segment, T.TagBlock):
                for tag in segment.tags:
                    if tag.name in wanted:
                        rows.append(
                            {
                                "name": tag.name,
                                "argument": tag.arg,
                                "raw": tag.raw,
                                "block": block,
                            }
                        )
                block += 1
        return rows

    if selection is not None:
        if text is not None or index is not None:
            raise ToolError("pass either selection= or text=/index=, not both")
        did, doc, indices = _doc_and_indices(selection, doc_id)
        events = doc.events()
        updates: list[tuple[int, str]] = []
        rows: list[dict[str, Any]] = []
        total_removed = 0
        for i in indices:
            before = events[i].text
            gone = removed_rows(before)
            after = T.remove_tags(before, ordered) if gone else before
            total_removed += len(gone)
            _sane(after, "ass_remove_tag")
            rows.append(
                {
                    "index": i,
                    "before": before,
                    "after": after,
                    "removed": gone,
                    "changed": after != before,
                }
            )
            if after != before:
                updates.append((i, after))
        written = _batch_write(doc, did, updates) if in_place else 0
        return ok(
            source=SOURCE_SELECTION,
            doc_id=did,
            names=ordered,
            count=len(indices),
            changed=len(updates),
            written=written,
            in_place=bool(in_place),
            removed_count=total_removed,
            lines=rows,
        )

    target = _target(text=text, index=index, doc_id=doc_id)
    gone = removed_rows(target.text)
    after = T.remove_tags(target.text, ordered) if gone else target.text
    _sane(after, "ass_remove_tag")
    written = _commit(target, after, in_place) if after != target.text else False
    return ok(
        **target.where(),
        names=ordered,
        removed=gone,
        removed_count=len(gone),
        text=after,
        plain_text=T.plain_text(after),
        changed=after != target.text,
        written=written,
    )


# --------------------------------------------------------------------------- #
# 4. range and block operations
# --------------------------------------------------------------------------- #


def _raw_boundary_map(text: str) -> dict[int, int]:
    """Map every raw offset to its plain offset (blocks map to their edges).

    Only offsets that fall on a character boundary of the visible text are
    present: the start and end of every override block, and every position
    inside a text run.  An offset landing inside an override block is absent on
    purpose — a range boundary there would split a tag block.
    """
    mapping: dict[int, int] = {0: 0}
    raw_cursor = 0
    plain_cursor = 0
    for segment in T.parse(text).segments:
        rendered = segment.render()
        if isinstance(segment, T.TagBlock):
            mapping.setdefault(raw_cursor, plain_cursor)
            raw_cursor += len(rendered)
            mapping[raw_cursor] = plain_cursor
        else:
            for offset in range(len(rendered) + 1):
                mapping[raw_cursor + offset] = plain_cursor + offset
            raw_cursor += len(rendered)
            plain_cursor += len(rendered)
    return mapping


def ass_wrap_range(
    index: int | None = None,
    text: str | None = None,
    start: int = 0,
    end: int | None = None,
    override: str = "",
    doc_id: str | None = None,
    scope: str = "plain",
    in_place: bool = False,
) -> dict[str, Any]:
    """Wrap a character range in an override block, restoring the outer state.

    Args:
        index / text / doc_id: the line to edit (0-based index of ``doc_id`` or a
            raw string).
        start: range start.  A **plain index** when ``scope="plain"`` (the
            default), a **raw index** when ``scope="raw"``.
        end: range end, exclusive.  Same index system as ``start``; ``None``
            means "to the end of the line" (the visible end / the raw end).
        override: the tags to apply, with or without the surrounding braces and
            with or without the leading backslash (``r"\\fscx200"``,
            ``"fscx200"`` and ``r"{\\fscx200}"`` are all accepted).  Braces and
            line breaks inside the payload are rejected.
        doc_id: document holding ``index``.
        scope: ``"plain"`` (default) counts only visible characters — override
            blocks and the ``\\N``/``\\n`` line-break escapes are not counted, so
            the same ``start``/``end`` cover the same glyphs no matter how many
            tags precede them.  ``"raw"`` counts every stored code point
            including braces, which is what you want when you already have
            offsets into the stored ``Text`` field.  A raw boundary that would
            land inside an override block is refused with a ``ToolError``
            instead of silently splitting the block.
        in_place: write back to the document (snapshot-backed).  The raw-string
            mode never writes.

    The tags that were in effect before ``start`` are re-emitted after ``end``
    when they changed inside the range, so the override applies to exactly the
    requested characters.  ``warnings`` reports tags that had no previous value
    to restore (there is nothing to restore for e.g. ``\\an``).

    Returns ``{"source", "index", "doc_id", "scope", "start", "end",
    "raw_start", "raw_end", "plain_start", "plain_end", "override", "text",
    "plain_text", "changed", "written", "warnings"}``.  ``start``/``end`` are
    echoed in the requested index system; ``plain_start``/``plain_end`` are always
    **plain** (visible character) indices and ``raw_start``/``raw_end`` are always
    **raw** offsets into the stored line, so the two systems stay comparable no
    matter which one was passed in.
    """
    target = _target(text=text, index=index, doc_id=doc_id)
    body = _override_body(override)
    line = target.text
    plain_len = T.plain_len(line)
    if scope not in ("plain", "raw"):
        raise ToolError(f"scope must be 'plain' or 'raw', got {scope!r}")
    try:
        first = int(start)
    except (TypeError, ValueError):
        raise ToolError(f"start must be an integer, got {start!r}") from None
    if scope == "plain":
        limit = plain_len
        last = plain_len if end is None else int(end)
        plain_start, plain_end = first, last
        if not (0 <= plain_start <= plain_len):
            raise ToolError(
                f"start={plain_start} is outside the visible text "
                f"(the line has {plain_len} visible characters; plain indices)"
            )
        if not (0 <= plain_end <= plain_len):
            raise ToolError(
                f"end={plain_end} is outside the visible text "
                f"(the line has {plain_len} visible characters; plain indices)"
            )
    else:
        limit = len(line)
        last = len(line) if end is None else int(end)
        if not (0 <= first <= len(line)) or not (0 <= last <= len(line)):
            raise ToolError(
                f"raw range [{first}, {last}) is outside the stored text "
                f"(the line has {len(line)} characters)"
            )
        mapping = _raw_boundary_map(line)
        for raw_at, label in ((first, "start"), (last, "end")):
            if raw_at not in mapping:
                raise ToolError(
                    f"raw {label}={raw_at} falls inside an override block; use "
                    f"scope='plain' or pick a raw offset outside the braces "
                    f"(blocks: "
                    + ", ".join(
                        f"[{i}, {j})"
                        for i, j in _block_spans(line)
                    )
                    + ")"
                )
        plain_start, plain_end = mapping[first], mapping[last]
    if plain_end <= plain_start:
        raise ToolError(
            f"empty range: [{plain_start}, {plain_end}) covers no visible character "
            f"(scope={scope}, plain indices)"
        )
    out, warnings = T.wrap_range(line, plain_start, plain_end, body)
    _sane(out, "ass_wrap_range")
    written = _commit(target, out, in_place) if out != line else False
    parsed = T.parse(line)
    if scope == "raw":
        raw_start, raw_end = first, last
    else:
        raw_start = parsed.raw_index_of_char(plain_start)
        if raw_start is None:
            raw_start = len(line)
        raw_end = parsed.raw_index_after_char(plain_end - 1)
    return ok(
        **target.where(),
        scope=scope,
        start=first,
        end=last,
        raw_start=raw_start,
        raw_end=raw_end,
        plain_start=plain_start,
        plain_end=plain_end,
        override=body,
        text=out,
        plain_text=T.plain_text(out),
        changed=out != line,
        written=written,
        warnings=list(warnings),
    )


def _block_spans(text: str) -> list[tuple[int, int]]:
    """Raw ``[start, end)`` span of every override block, in order."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for segment in T.parse(text).segments:
        rendered = segment.render()
        if isinstance(segment, T.TagBlock):
            spans.append((cursor, cursor + len(rendered)))
        cursor += len(rendered)
    return spans


def ass_insert_tag_at(
    index: int | None = None,
    text: str | None = None,
    plain_index: int = 0,
    override: str = "",
    doc_id: str | None = None,
    after: bool = False,
    in_place: bool = False,
) -> dict[str, Any]:
    """Insert a bare override block at a plain character position.

    Args:
        index / text / doc_id: the line to edit (0-based index of ``doc_id`` or a
            raw string).
        plain_index: **plain (visible character) index** the block is inserted
            at; ``0`` puts it before the first visible character, ``plain_len``
            puts it at the end of the line.  Override blocks do not count, so
            for ``"ab{\\i1}cd"`` plain index 2 is between ``b`` and ``c``.
        override: the tags to insert (braces and the leading backslash are
            optional, braces inside the payload are rejected).
        after: insert *after* the character at ``plain_index`` instead of before
            it.  With ``after=True`` and ``plain_index == plain_len`` the block
            is appended at the very end.
        doc_id: document holding ``index``.
        in_place: write back to the document (snapshot-backed).  The raw-string
            mode never writes.

    Returns ``{"source", "index", "doc_id", "plain_index", "after", "override",
    "text", "plain_text", "changed", "written"}``.  ``plain_index`` is always a
    plain index; the raw offsets of the new block are not reported because the
    insertion shifts them.
    """
    target = _target(text=text, index=index, doc_id=doc_id)
    body = _override_body(override)
    line = target.text
    plain_len = T.plain_len(line)
    try:
        pos = int(plain_index)
    except (TypeError, ValueError):
        raise ToolError(f"plain_index must be an integer, got {plain_index!r}") from None
    if not (0 <= pos <= plain_len):
        raise ToolError(
            f"plain_index={pos} is outside the visible text: the line has "
            f"{plain_len} visible characters (plain indices, override blocks "
            f"do not count)"
        )
    if after:
        out = T.insert_override_after_char(line, pos, body)
    else:
        out = T.insert_override_at_char(line, pos, body)
    _sane(out, "ass_insert_tag_at")
    written = _commit(target, out, in_place) if out != line else False
    return ok(
        **target.where(),
        plain_index=pos,
        after=bool(after),
        override=body,
        text=out,
        plain_text=T.plain_text(out),
        changed=out != line,
        written=written,
    )


def ass_apply_tag_to_block(
    index: int | None = None,
    text: str | None = None,
    block: Any = 0,
    override: str = "",
    doc_id: str | None = None,
    in_place: bool = False,
) -> dict[str, Any]:
    """Append tags to one override block (or to every block) of a line.

    Args:
        index / text / doc_id: the line to edit (0-based index of ``doc_id`` or a
            raw string).
        block: 0-based override block number, or ``"all"`` to touch every block.
            Blocks are numbered in line order and, unlike plain indices, count
            comment blocks too.  When the line has no block at all, ``block=0``
            and ``"all"`` create one at the start of the line.
        override: the tags to append (braces and the leading backslash are
            optional; braces inside the payload are rejected).
        doc_id: document holding ``index``.
        in_place: write back to the document (snapshot-backed).  The raw-string
            mode never writes.

    Returns ``{"source", "index", "doc_id", "block", "blocks_total",
    "applied_blocks", "override", "text", "plain_text", "changed", "written"}``
    with ``applied_blocks`` listing the block numbers that actually received the
    tags.  The text inside every block is preserved verbatim and the closing
    brace is always kept.
    """
    target = _target(text=text, index=index, doc_id=doc_id)
    body = _override_body(override)
    line = target.text
    parsed = T.parse(line)
    rendered = parsed.render()
    spans: list[tuple[int, int]] = []
    cursor = 0
    for segment in rendered.__class__ and parsed.segments:
        text_of = segment.render()
        if isinstance(segment, T.TagBlock):
            spans.append((cursor, cursor + len(text_of)))
        cursor += len(text_of)

    def append_into(span: tuple[int, int], src: str) -> str:
        start, stop = span
        return src[: stop - 1] + body + src[stop - 1 :]

    if isinstance(block, str) and block.lower() == "all":
        if not spans:
            out = T.prepend_tags(line, body)
            applied = [0]
        else:
            out = line
            for span in reversed(spans):
                out = append_into(span, out)
            applied = list(range(len(spans)))
        total = max(len(spans), 1)
    else:
        try:
            number = int(block)
        except (TypeError, ValueError):
            raise ToolError(f"block must be an integer or 'all', got {block!r}") from None
        if number < 0:
            raise ToolError("block must be 0 or greater")
        if not spans:
            if number != 0:
                raise ToolError(
                    f"the line has no override block, so block={number} does not exist "
                    f"(block=0 would create one)"
                )
            out = T.prepend_tags(line, body)
            applied = [0]
            total = 1
        else:
            if number >= len(spans):
                raise ToolError(
                    f"block={number} does not exist: the line has {len(spans)} "
                    f"override block(s) (numbered 0..{len(spans) - 1})"
                )
            out = append_into(spans[number], line)
            applied = [number]
            total = len(spans)
    _sane(out, "ass_apply_tag_to_block")
    written = _commit(target, out, in_place) if out != line else False
    return ok(
        **target.where(),
        block=block,
        blocks_total=total,
        applied_blocks=applied,
        override=body,
        text=out,
        plain_text=T.plain_text(out),
        changed=out != line,
        written=written,
    )


# --------------------------------------------------------------------------- #
# 5. typesetting blocks
# --------------------------------------------------------------------------- #


def _pair(value: Any, what: str) -> tuple[str, str]:
    seq = _as_list(value, what)
    if len(seq) != 2:
        raise ToolError(f"{what} must be a pair of numbers, got {value!r}")
    return _num(seq[0]), _num(seq[1])


def _move_pieces(value: Any) -> list[str]:
    """``\\move`` arguments from a pair/pairs/four-tuple/6-tuple/dict."""
    times: list[Any] | None = None
    if isinstance(value, dict):
        try:
            coords = [value["x1"], value["y1"], value["x2"], value["y2"]]
        except KeyError as exc:
            raise ToolError(
                f"move dict needs x1, y1, x2 and y2 (missing {exc})"
            ) from None
        if any(k in value for k in ("t1", "t2")):
            times = [value.get("t1", 0), value.get("t2", 0)]
    else:
        seq = _as_list(value, "move")
        if len(seq) == 2 and all(isinstance(v, (list, tuple)) for v in seq):
            seq = list(seq[0]) + list(seq[1])
        if len(seq) == 2:
            coords = [seq[0], seq[1], seq[0], seq[1]]
        elif len(seq) == 4:
            coords = list(seq)
        elif len(seq) == 6:
            coords = list(seq[:4])
            times = list(seq[4:])
        else:
            raise ToolError(
                "move must be four numbers (x1, y1, x2, y2), two pairs, a "
                "four-tuple, or six numbers with t1/t2 in milliseconds"
            )
    if len(coords) != 4:
        raise ToolError("move needs exactly four coordinates")
    out = [_num(c) for c in coords]
    if times is not None:
        out.extend(str(int(_require_int(t, "move time"))) for t in times)
    return out


def _require_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ToolError(f"{what} must be an integer, got {value!r}") from None
        if number != int(number):
            raise ToolError(f"{what} must be a whole number, got {value!r}")
        return int(number)
    return value


def _fade_pieces(value: Any) -> str:
    """``\\fad``/``\\fade`` text from a pair, a 4-tuple or the full 7-tuple."""
    if isinstance(value, dict):
        try:
            alphas = [value[f"a{i}"] for i in (1, 2, 3)]
            times = [value[f"t{i}"] for i in (1, 2, 3, 4)]
        except KeyError as exc:
            raise ToolError(
                f"fade dict needs a1, a2, a3, t1, t2, t3 and t4 (missing {exc})"
            ) from None
        args = [_num(v) for v in alphas] + [str(_require_int(t, "fade time")) for t in times]
        return r"\fade(" + ",".join(args) + ")"
    seq = _as_list(value, "fade")
    if len(seq) == 2:
        return T.tag("fad", ",".join(_num(v) for v in seq))
    if len(seq) == 4:
        alphas = [_num(v) for v in seq[:3]]
        time = str(_require_int(seq[3], "fade time"))
        return r"\fade(" + ",".join(alphas + [time, time, time, time]) + ")"
    if len(seq) == 7:
        args = [_num(v) for v in seq[:3]] + [
            str(_require_int(t, "fade time")) for t in seq[3:]
        ]
        return r"\fade(" + ",".join(args) + ")"
    raise ToolError(
        "fade must be a pair of milliseconds (\\fad), a 4-tuple "
        "(a1, a2, a3, t) or the full \\fade 7-tuple"
    )


def _clip_pieces(value: Any) -> str:
    """``\\clip``/``\\iclip`` text from a rectangle, a drawing or a dict."""
    inverse = False
    scale: Any = None
    rect: Any = None
    drawing: Any = None
    if isinstance(value, dict):
        inverse = bool(value.get("inverse", value.get("iclip", False)))
        scale = value.get("scale")
        rect = value.get("rect")
        drawing = value.get("drawing", value.get("path", value.get("points")))
        if rect is None and drawing is None:
            raise ToolError("clip dict needs a 'rect' or a 'drawing'/'path'")
    elif isinstance(value, str):
        drawing = value
    else:
        seq = _as_list(value, "clip")
        if len(seq) == 4 and all(not isinstance(v, str) for v in seq):
            rect = seq
        elif len(seq) == 2 and isinstance(seq[1], str):
            scale, drawing = seq[0], seq[1]
        elif len(seq) == 1 and isinstance(seq[0], str):
            drawing = seq[0]
        elif len(seq) in (4, 5) and any(isinstance(v, str) for v in seq):
            strings = [v for v in seq if isinstance(v, str)]
            numbers = [v for v in seq if not isinstance(v, str)]
            drawing = strings[0]
            if numbers:
                scale = numbers[0]
        else:
            raise ToolError(
                "clip must be a rectangle [x0, y0, x1, y1], a drawing path "
                "(string), [scale, path], or a dict with 'rect'/'drawing' and "
                "an optional 'inverse' flag"
            )
    name = "iclip" if inverse else "clip"
    if rect is not None:
        seq = _as_list(rect, "clip rectangle")
        if len(seq) != 4:
            raise ToolError("clip rectangle needs exactly four numbers")
        return f"\\{name}(" + ",".join(_num(v) for v in seq) + ")"
    path = str(drawing).strip()
    if not path:
        raise ToolError("clip drawing is empty")
    _no_braces(path, "clip drawing")
    if not _DRAWING_COMMAND_RE.search(path):
        raise ToolError(
            f"clip drawing {path!r} has no drawing command (m, n, l, b, s, p or c); "
            "pass a rectangle [x0, y0, x1, y1] for a rectangular clip"
        )
    if scale is None:
        return f"\\{name}({path})"
    return f"\\{name}({_num(scale)},{path})"


def ass_add_typesetting(
    selection: Any,
    doc_id: str | None = None,
    pos: Any = None,
    an: Any = None,
    move: Any = None,
    fade: Any = None,
    clip: Any = None,
    org: Any = None,
    extra_tags: Any = None,
    reset_first: bool = False,
    in_place: bool = True,
) -> dict[str, Any]:
    """Build a leading override block out of named typesetting pieces.

    Args:
        selection: selection spelling; every selected line receives the block.
        doc_id: document to edit; the current one when omitted.
        pos: ``[x, y]`` -> ``\\pos(x,y)``.
        an: alignment, a plain integer 1..9 (``7`` -> ``\\an7``).  Booleans,
            floats and out-of-range values are rejected; ``\\q``-style
            alignments are not part of this helper.
        move: ``\\move`` animation.  Accepted shapes: six numbers
            ``[x1, y1, x2, y2, t1, t2]`` (times in **milliseconds**), four
            numbers, two pairs, a pair (expanded to a zero-length move), or a
            dict with ``x1``/``y1``/``x2``/``y2`` and optional ``t1``/``t2``.
        fade: ``\\fad``/``\\fade``.  A pair ``[in_ms, out_ms]`` produces
            ``\\fad(in,out)``; a 4-tuple ``[a1, a2, a3, t]`` produces
            ``\\fade(a1,a2,a3,t,t,t,t)``; the full 7-tuple
            ``[a1, a2, a3, t1, t2, t3, t4]`` is passed through verbatim; a dict
            with ``a1``/``a2``/``a3``/``t1``..``t4`` also works.
        clip: ``\\clip``/``\\iclip``.  A rectangle ``[x0, y0, x1, y1]``, a
            drawing path string (``"m 0 0 l 100 0 100 100"`` — it must contain a
            drawing command, otherwise the string is rejected instead of being
            handed to libass as a no-op), ``[scale, path]``,
            or a dict ``{"rect": [...]}`` / ``{"drawing": "...", "scale": n}``
            with ``"inverse": true`` to emit ``\\iclip``.
        org: ``[x, y]`` -> ``\\org(x,y)``.
        extra_tags: any additional override text (braces and the leading
            backslash are optional) appended to the block in the order given.
        reset_first: put ``\\r`` at the front of the block so the line starts
            from the style's values before the new tags are applied.
        in_place: write the block back to the document (snapshot-backed,
            ``True`` by default).  With ``False`` the planned lines are returned
            but the document is left alone.

    Returns ``{"doc_id", "tag_string", "override", "reset_first", "count",
    "changed", "written", "in_place", "lines": [{"index", "before", "text",
    "changed"}], "text"}``.  ``tag_string`` is the exact tag string produced
    (without the braces, e.g. ``\\an8\\pos(100,200)``); ``text`` is the new line
    text when the selection resolved to a single line, otherwise ``None`` (the
    per-line texts are in ``lines``).  The block is merged into the line's
    existing first block so duplicated tags are updated instead of stacking.
    """
    pieces: list[str] = []
    if reset_first:
        pieces.append(r"\r")
    if pos is not None:
        x, y = _pair(pos, "pos")
        pieces.append(f"\\pos({x},{y})")
    if an is not None:
        if isinstance(an, bool) or not isinstance(an, int):
            raise ToolError(
                f"an must be an integer between 1 and 9, got {an!r} "
                f"(\\an is an alignment, not a string)"
            )
        if not 1 <= an <= 9:
            raise ToolError(f"an must be between 1 and 9, got {an}")
        pieces.append(f"\\an{an}")
    if move is not None:
        pieces.append("\\move(" + ",".join(_move_pieces(move)) + ")")
    if fade is not None:
        pieces.append(_fade_pieces(fade))
    if clip is not None:
        pieces.append(_clip_pieces(clip))
    if org is not None:
        x, y = _pair(org, "org")
        pieces.append(f"\\org({x},{y})")
    if extra_tags is not None:
        pieces.append(_override_body(extra_tags, "extra_tags"))
    if not pieces:
        raise ToolError(
            "nothing to add: pass at least one of pos, an, move, fade, clip, "
            "org or extra_tags (or reset_first=True)"
        )
    body = "".join(pieces)
    _no_braces(body, "typesetting block")
    _sane("{" + body + "}", "ass_add_typesetting")

    did, doc, indices = _doc_and_indices(selection, doc_id)
    events = doc.events()
    updates: list[tuple[int, str]] = []
    rows: list[dict[str, Any]] = []
    for i in indices:
        before = events[i].text
        after = T.prepend_tags(before, body)
        _sane(after, "ass_add_typesetting")
        rows.append(
            {"index": i, "before": before, "text": after, "changed": after != before}
        )
        if after != before:
            updates.append((i, after))
    written = _batch_write(doc, did, updates) if in_place else 0
    return ok(
        doc_id=did,
        tag_string=body,
        override="{" + body + "}",
        reset_first=bool(reset_first),
        count=len(indices),
        changed=len(updates),
        written=written,
        in_place=bool(in_place),
        lines=rows,
        text=rows[0]["text"] if len(rows) == 1 else None,
    )


# --------------------------------------------------------------------------- #
# 6. the \an / \pos mirror
# --------------------------------------------------------------------------- #


def _mirror_an(an: int, *, horizontal: bool = False) -> tuple[int, int, int]:
    """Return ``(new_an, row_step, col_step)`` for the mirrored alignment.

    ``an`` is a 1..9 numpad alignment, i.e. ``divmod(an - 1, 3)`` gives
    ``(row, col)`` with row 2 = the top row (7/8/9), row 0 = the bottom row
    (1/2/3), and col 2 = the right column (3/6/9), col 0 = the left column
    (1/4/7).  The vertical mirror swaps the bottom and top rows, so 1 -> 7,
    2 -> 8, 3 -> 9 and vice versa, while the middle row (4, 5, 6) does not
    move.  The steps are the numpad distances (``±2`` for a top/bottom swap),
    which is what the caller turns into a pixel shift.
    """
    if isinstance(an, bool) or not isinstance(an, int) or not 1 <= an <= 9:
        raise ToolError(f"an must be an integer between 1 and 9, got {an!r}")
    row, col = divmod(an - 1, 3)
    new_row = 2 - row
    new_col = (2 - col) if horizontal else col
    return new_row * 3 + new_col + 1, new_row - row, new_col - col


def _style_spec(doc: AssDocument, entry: EventEntry) -> dict[str, Any] | None:
    name = (entry.get("Style") or "").strip()
    for style in doc.styles():
        if style.name == name:
            try:
                return dict(style.fields_dict())
            except Exception:  # pragma: no cover - defensive
                return None
    return None


def _measure_geometry(
    plain_text: str, style: dict[str, Any] | None, play_res: Sequence[int]
) -> dict[str, Any]:
    """Rendered size of a line in script pixels, taken from libass.

    Three probes are rendered with :func:`asscore.measure.measure_line_render`:

    * ``{\\an7\\pos(0,Y)}`` puts the top-left anchor at the origin column, so the
      ink box's left edge is the left side bearing and its ``y`` is the top-row
      anchor offset;
    * ``{\\an1\\pos(0,Y)}`` gives the bottom-row anchor offset.  The difference
      between the two ``y`` values is the **vertical anchor separation**, i.e.
      the line height in script pixels — exactly the amount ``\\pos`` has to
      move when ``\\an`` jumps between the bottom and the top row;
    * ``{\\an9\\pos(X,Y)}`` gives the right edge, so the advance width (the
      horizontal anchor separation used when the alignment jumps between the
      left and right columns) is ``left_bearing + ink_width + right_bearing``.

    The frame is padded to at least 2048x2048 because font sizes are script
    units: rendering at 1:1 keeps the pixel size identical while guaranteeing
    that no probe is clipped by the frame edge.  Raises ``ToolError`` when the
    measurement backend is unavailable or the line has no ink.
    """
    play_x = max(int(play_res[0]), 2048)
    play_y = max(int(play_res[1]), 2048)
    mid_y = play_y // 2
    mid_x = play_x // 2

    def probe(extra: str) -> dict[str, Any]:
        try:
            return M.measure_line_render(
                plain_text, style=style, play_res=(play_x, play_y), extra_tags=extra
            )
        except Exception as exc:
            raise ToolError(
                f"could not measure the line in script pixels: {exc} "
                f"(libass/ffmpeg measurement unavailable)"
            ) from exc

    top = probe("{\\an7\\pos(0,%d)}" % mid_y)
    bottom = probe("{\\an1\\pos(0,%d)}" % mid_y)
    right = probe("{\\an9\\pos(%d,%d)}" % (mid_x, mid_y))
    for label, result in (("\\an7", top), ("\\an1", bottom), ("\\an9", right)):
        if result.get("empty") or not result.get("rect"):
            raise ToolError(
                f"could not measure the line: the {label} probe rendered no ink "
                f"(is the line empty, or a drawing with no \\p state?)"
            )
    height = float(top["rect"]["y"]) - float(bottom["rect"]["y"])
    width = (
        float(top["rect"]["x"])
        + (mid_x - float(right["rect"]["x1"]))
        + float(top["rect"]["width"])
    )
    return {
        "width": round(width, 3),
        "height": round(height, 3),
        "ink_width": float(top["rect"]["width"]),
        "ink_height": float(top["rect"]["height"]),
        "left_bearing": float(top["rect"]["x"]),
        "source": "measure_line_render",
        "probe_play_res": [play_x, play_y],
    }


def _an_value(text: str) -> int | None:
    tag = T.get_tag(text, "an")
    if tag is None:
        return None
    match = _AN_VALUE_RE.match(tag.arg or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - defensive
        return None


def _pos_pair(text: str) -> tuple[float, float] | None:
    tag = T.get_tag(text, "pos")
    if tag is None:
        return None
    values = [float(m.group(0)) for m in _COORD_RE.finditer(tag.arg or "")]
    if len(values) < 2:
        return None
    return values[0], values[1]


def ass_swap_an_pos(
    selection: Any,
    doc_id: str | None = None,
    margin_mode: bool = False,
    dry_run: bool = False,
    horizontal: bool = False,
) -> dict[str, Any]:
    """Mirror ``\\an`` vertically and move ``\\pos`` so the text does not budge.

    The classic Aegisub helper: ``\\an1`` <-> ``\\an7``, ``\\an2`` <-> ``\\an8``,
    ``\\an3`` <-> ``\\an9`` (the middle row 4/5/6 has no vertical mirror and is
    left alone).  Because the anchor flips between the bottom and the top of the
    line, ``\\pos`` has to move by exactly the line height for the rendered text
    to stay where it was.

    The shift is computed from the **real rendered size** of the line, measured
    with :func:`asscore.measure.measure_line_render` in script pixels (see
    :func:`_measure_geometry`).  ``height`` is the vertical anchor separation
    (``\\an7`` ink top minus ``\\an1`` ink top, both anchored at the same
    ``y``), i.e. the line height, and ``width`` is the advance width (left side
    bearing + ink width + right side bearing, from the ``\\an7``/``\\an9``
    probes), i.e. the horizontal anchor separation.  One numpad row or column
    step moves the anchor by half of the corresponding separation, and the
    anchor must move the other way for the ink to stay put::

        row_step = new_row - row      # numpad grid, +-2 for a top/bottom swap
        col_step = new_col - col      # +-2 for a left/right swap
        dy = -(height / 2) * row_step # row 2 is the top of the screen
        dx = +(width  / 2) * col_step # col 2 is the right of the screen

    For ``\\an7 -> \\an1`` that is ``row_step = -2`` and ``dy = +height``: the
    anchor walks down by the line height so the text stays on its screen row.
    A purely vertical mirror never changes the column, so ``dx`` stays ``0``
    there; ``horizontal=True`` additionally mirrors the column
    (``\\an7 <-> \\an9``, ``\\an1 <-> \\an3``, ``\\an4 <-> \\an6``) and then
    ``dx = +width`` for ``\\an7 -> \\an3`` (numpad columns grow rightwards, so
    the new right-edge anchor has to sit one width further right), which is what
    makes the width term reachable.

    The measured text is the line with its own ``\\pos``/``\\move``/``\\an``
    removed, rendered with the line's style and the document's ``PlayRes``, so
    the height matches what libass does for the real line.

    Indices: ``\\an``/``\\pos`` are located inside the override blocks, so no
    plain-character index and no raw offset is taken as input.  ``selection``
    refers to event indices (``doc.events()`` order, 0-based), and the plain text
    of every line survives byte for byte, so both index maps are unchanged.

    Args:
        selection: selection spelling; the lines to rewrite.
        doc_id: document to edit; the current one when omitted.
        margin_mode: how to treat lines that have no ``\\pos``.  By default they
            are skipped (there is nothing to shift).  With ``margin_mode=True``
            the vertical margin is rewritten instead:
            ``MarginV = PlayResY - MarginV - height``, which keeps an
            alignment/margin-positioned line in place across the mirror.
        dry_run: compute and report the plan without writing (no snapshot).
        horizontal: also mirror the alignment column (180 degree mirror) and
            shift ``x`` by the advance width.  Off by default, matching the
            Aegisub helper.

    Returns ``{"doc_id", "dry_run", "margin_mode", "horizontal", "count",
    "changed", "written", "geometry_source", "lines": [...]}``; each line entry
    has ``index``, ``before``, ``after``, ``old_an``, ``new_an``, ``old_pos``,
    ``new_pos``, ``dx``, ``dy``, ``margin_v``, ``measured`` (``{"width",
    "height", "ink_width", "ink_height", "left_bearing"}``), ``changed``,
    ``skipped`` and ``reason``.  Retiming/render checks aside, the check to
    apply is that the ink bounding box before and after is identical — see the
    test suite, which asserts exactly that with
    :func:`asscore.measure.measure_render`.
    """
    did, doc, indices = _doc_and_indices(selection, doc_id)
    play_res = doc.play_res
    events = doc.events()
    updates: list[tuple[int, str]] = []
    margin_updates: list[tuple[int, str, str]] = []
    rows: list[dict[str, Any]] = []
    for i in indices:
        entry = events[i]
        line = entry.text
        row: dict[str, Any] = {
            "index": i,
            "before": line,
            "after": line,
            "old_an": None,
            "new_an": None,
            "old_pos": None,
            "new_pos": None,
            "dx": 0,
            "dy": 0,
            "margin_v": None,
            "margin_mode": bool(margin_mode),
            "measured": None,
            "changed": False,
            "skipped": False,
            "reason": None,
        }
        an_tag = T.get_tag(line, "an")
        if an_tag is None:
            row["skipped"] = True
            row["reason"] = (
                r"no \an tag (legacy SSA \a alignment is not rewritten)"
                if T.get_tag(line, "a") is not None
                else r"no \an tag"
            )
            rows.append(row)
            continue
        an_value = _an_value(line)
        if an_value is None or not 1 <= an_value <= 9:
            row["skipped"] = True
            row["reason"] = f"\\an value {an_tag.arg!r} is not a 1..9 alignment"
            rows.append(row)
            continue
        row["old_an"] = an_value
        new_an, row_step, col_step = _mirror_an(an_value, horizontal=horizontal)
        row["new_an"] = new_an
        if new_an == an_value:
            row["skipped"] = True
            row["reason"] = (
                r"\an4/\an5/\an6 sit in the middle row and do not mirror vertically"
            )
            rows.append(row)
            continue

        pos = _pos_pair(line)
        measured = None
        if pos is not None:
            plain = T.plain_text(T.remove_tags(line, ["pos", "move", "an", "a"]))
            measured = _measure_geometry(plain, _style_spec(doc, entry), play_res)
            width = measured["width"]
            height = measured["height"]
            # One numpad row/column step moves the anchor by half the measured
            # anchor separation, and the anchor has to move the other way for
            # the ink to stay put.  The row axis is flipped relative to screen
            # y (row 2 is the top of the screen), the column axis is not, hence
            # the opposite signs: dy = -(height/2) * row_step,
            # dx = +(width/2) * col_step.
            dy = -(height / 2.0) * row_step
            dx = (width / 2.0) * col_step
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            after = T.set_tag(line, "an", str(new_an))
            after = T.set_tag(after, "pos", f"{_num(new_x)},{_num(new_y)}")
            row.update(
                {
                    "old_pos": f"{_num(pos[0])},{_num(pos[1])}",
                    "new_pos": f"{_num(new_x)},{_num(new_y)}",
                    "dx": round(dx, 3),
                    "dy": round(dy, 3),
                    "measured": measured,
                    "after": after,
                    "changed": after != line,
                }
            )
        elif margin_mode:
            spec = _style_spec(doc, entry) or {}
            margin_v = int(float(entry.get("MarginV") or 0)) or int(
                float(spec.get("MarginV") or 0)
            )
            plain = T.plain_text(T.remove_tags(line, ["pos", "move", "an", "a"]))
            measured = _measure_geometry(plain, spec or None, play_res)
            height = measured["height"]
            new_margin = int(round(play_res[1] - margin_v - height))
            warning = None
            if new_margin < 0:
                new_margin = 0
                warning = (
                    "computed MarginV was negative and has been clamped to 0; "
                    "the line may still shift"
                )
            after = T.set_tag(line, "an", str(new_an))
            row.update(
                {
                    "margin_v": {"before": margin_v, "after": new_margin},
                    "measured": measured,
                    "after": after,
                    "changed": after != line,
                    "reason": warning,
                }
            )
            if not dry_run and after != line:
                updates.append((i, after))
                # the MarginV rewrite is queued too, so a dry run writes nothing
                # and both edits land under the same single snapshot
                margin_updates.append((i, "MarginV", str(new_margin)))
            rows.append(row)
            continue
        else:
            row["skipped"] = True
            row["reason"] = (
                r"no \pos to shift (pass margin_mode=True to move the vertical "
                r"margin instead)"
            )
            rows.append(row)
            continue

        if not dry_run and row["changed"]:
            updates.append((i, after))
        rows.append(row)

    written = 0
    if not dry_run:
        written = _batch_write(doc, did, updates, fields=margin_updates)
    return ok(
        doc_id=did,
        dry_run=bool(dry_run),
        margin_mode=bool(margin_mode),
        horizontal=bool(horizontal),
        count=len(indices),
        changed=sum(1 for row in rows if row.get("changed")),
        written=written,
        geometry_source="measure_line_render",
        lines=rows,
    )


# --------------------------------------------------------------------------- #
# 7. summaries
# --------------------------------------------------------------------------- #


def _summary_row(index: int, line: str) -> dict[str, Any]:
    parsed = T.parse(line)
    names = Counter(t.name for t in parsed.tags())
    blocks = parsed.blocks()
    first_names = {t.name for t in blocks[0].tags} if blocks else set()
    groups: dict[str, int] = {}
    for name, count in names.items():
        for group in _groups_of(name):
            groups[group] = groups.get(group, 0) + count
    return {
        "index": index,
        "raw": line,
        "plain_text": parsed.plain_text(),
        "visible_chars": T.plain_len(line),
        "blocks": len(blocks),
        "tags": sum(names.values()),
        "tag_names": dict(sorted(names.items())),
        "tag_groups": dict(sorted(groups.items())),
        "has_drawing": T.drawing_state(line) > 0,
        "drawing_state": T.drawing_state(line),
        "has_karaoke": any(name in T.KARAOKE_TAGS for name in names),
        "has_transform": any(name == "t" for name in names),
        "has_clip": any(name in T.CLIP_TAGS for name in names),
        "first_block_has_pos": "pos" in first_names,
        "first_block_has_move": "move" in first_names,
        "missing_position": ("pos" not in first_names and "move" not in first_names),
    }


def ass_tag_summary(
    selection: Any = None,
    doc_id: str | None = None,
    index: int | None = None,
    text: str | None = None,
) -> dict[str, Any]:
    """Per-line tag histogram and typesetting sanity check.

    Args:
        index / text / doc_id: inspect a single line (0-based index of ``doc_id``
            or a raw string).
        selection: inspect a set of lines (``None`` means every line when no
            ``index``/``text`` is given).  ``doc_id`` says which document.

    Returns ``{"source", "doc_id", "index", "count", "missing_position",
    "lines": [...], "totals": {...}}``.  Every line entry carries ``index``,
    ``raw``, ``plain_text``, ``visible_chars``, ``blocks`` (number of override
    blocks), ``tags`` (total), ``tag_names`` (per-name histogram), ``tag_groups``
    (the same count folded into tag families such as layout, color, karaoke,
    clip, transform), ``has_drawing`` (drawing mode still active at the end of
    the line) / ``drawing_state``,
    ``has_karaoke``/``has_transform``/``has_clip``,
    ``first_block_has_pos``/``first_block_has_move`` and ``missing_position``
    (``True`` when the line's first override block has neither ``\\pos`` nor
    ``\\move`` — the line is then placed by styles and margins only, which is the
    usual typesetting smell).  ``missing_position`` at the top level lists the
    indices of those lines; ``totals`` sums tags, blocks and flags over the
    selection.  Read-only; no snapshot.  All indices are 0-based line indices,
    while ``visible_chars`` counts plain (visible) characters.
    """
    if text is not None or index is not None:
        target = _target(text=text, index=index, doc_id=doc_id)
        row = _summary_row(target.index if target.index is not None else 0, target.text)
        return ok(
            **target.where(),
            count=1,
            missing_position=[row["index"]] if row["missing_position"] else [],
            lines=[row],
            totals={
                "lines": 1,
                "tags": row["tags"],
                "blocks": row["blocks"],
                "visible_chars": row["visible_chars"],
                "missing_position": 1 if row["missing_position"] else 0,
            },
        )
    did, doc, indices = _doc_and_indices(selection, doc_id)
    events = doc.events()
    rows = [_summary_row(i, events[i].text) for i in indices]
    missing = [row["index"] for row in rows if row["missing_position"]]
    totals: Counter = Counter()
    for row in rows:
        totals["lines"] += 1
        totals["tags"] += row["tags"]
        totals["blocks"] += row["blocks"]
        totals["visible_chars"] += row["visible_chars"]
        if row["missing_position"]:
            totals["missing_position"] += 1
        if row["has_drawing"]:
            totals["drawing"] += 1
        if row["has_karaoke"]:
            totals["karaoke"] += 1
        if row["has_clip"]:
            totals["clip"] += 1
    return ok(
        source=SOURCE_SELECTION,
        doc_id=did,
        index=None,
        count=len(indices),
        missing_position=missing,
        lines=rows,
        totals=dict(sorted(totals.items())),
    )


def ass_karaoke_tags_only(
    text: str | None = None,
    index: int | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """List only the karaoke tags of a line, with their arguments.

    Handy right before retiming: it shows the syllable tags in order together
    with the visible text each one covers.

    Args:
        text / index / doc_id: the line to inspect (raw string, or 0-based index
            inside ``doc_id``).

    Returns ``{"source", "index", "doc_id", "has_karaoke", "count", "tags",
    "plain_text", "syllable_text", "times_ms", "total_ms"}``.  Each entry of
    ``tags`` is ``{"name" (canonical: k, kf, ko or kt), "argument", "raw",
    "plain_index", "block", "text"}`` where ``plain_index`` is the **plain
    (visible character) index** at which the tag takes effect, ``block`` is the
    0-based override block it lives in and ``text`` is the visible text that
    follows it up to the next karaoke tag (the syllable it times).
    ``syllable_text`` concatenates those chunks and ``total_ms`` sums the
    integer arguments (``\\kt``/``\\ko`` are not durations and are counted in
    ``count`` but reported as ``None`` in ``times_ms``).  Read-only; no
    snapshot.
    """
    target = _target(text=text, index=index, doc_id=doc_id)
    parsed = T.parse(target.text)
    rows: list[dict[str, Any]] = []
    plain_cursor = 0
    block_no = 0
    pending: dict[str, Any] | None = None
    for segment in parsed.segments:
        if isinstance(segment, T.TagBlock):
            karaoke = [t for t in segment.tags if t.name in T.KARAOKE_TAGS]
            for tag in karaoke:
                row = {
                    "name": tag.name,
                    "argument": tag.arg,
                    "raw": tag.raw,
                    "plain_index": plain_cursor,
                    "block": block_no,
                    "text": "",
                }
                rows.append(row)
            if karaoke:
                pending = rows[-1]
            block_no += 1
        else:
            if pending is not None:
                pending["text"] += segment.text
            plain_cursor += len(segment.text)
    times: list[int | None] = []
    for row in rows:
        match = _AN_VALUE_RE.match(row["argument"] or "")
        if row["name"] in ("k", "kf") and match:
            times.append(int(match.group(1)))
        else:
            times.append(None)
    durations = [t for t in times if t is not None]
    return ok(
        **target.where(),
        has_karaoke=bool(rows),
        count=len(rows),
        karaoke_names=sorted({row["name"] for row in rows}),
        tags=rows,
        plain_text=parsed.plain_text(),
        syllable_text="".join(row["text"] for row in rows),
        times_ms=times,
        total_ms=sum(durations),
    )


# --------------------------------------------------------------------------- #
# 8. SSA <-> ASS tag normalisation
# --------------------------------------------------------------------------- #


_STYLE_CONVERSIONS = (("\\K", "\\kf"),)


def _convert_inner(inner: str, mode: str) -> tuple[str, list[dict[str, Any]]]:
    """Rewrite the inside of one override block; returns (new, replacements)."""
    replacements: list[dict[str, Any]] = []
    out = inner
    if mode == "to_ass":
        def align(match: re.Match[str]) -> str:
            code = int(match.group(1))
            target_code = SSA_ALIGNMENT_TO_AN.get(code)
            if target_code is None:
                return match.group(0)
            replacements.append(
                {
                    "from": match.group(0),
                    "to": f"\\an{target_code}",
                    "kind": "alignment",
                    "note": f"legacy SSA \\a{code} -> \\an{target_code}",
                }
            )
            return f"\\an{target_code}"

        out = re.sub(r"\\a(\d{1,2})", align, out)

        def karaoke(match: re.Match[str]) -> str:
            replacements.append(
                {
                    "from": match.group(0),
                    "to": "\\kf",
                    "kind": "karaoke",
                    "note": "legacy SSA \\K -> \\kf (fill karaoke)",
                }
            )
            return "\\kf"

        out = re.sub(r"\\K(?![A-Za-z])", karaoke, out)
    elif mode == "to_ssa":

        def align_back(match: re.Match[str]) -> str:
            code = int(match.group(1))
            target_code = AN_TO_SSA_ALIGNMENT.get(code)
            if target_code is None:
                return match.group(0)
            replacements.append(
                {
                    "from": match.group(0),
                    "to": f"\\a{target_code}",
                    "kind": "alignment",
                    "note": f"\\an{code} -> legacy SSA \\a{target_code}",
                }
            )
            return f"\\a{target_code}"

        out = re.sub(r"\\an(\d)", align_back, out)

        def karaoke_back(match: re.Match[str]) -> str:
            replacements.append(
                {
                    "from": match.group(0),
                    "to": "\\K",
                    "kind": "karaoke",
                    "note": "\\kf -> legacy SSA \\K",
                }
            )
            return "\\K"

        out = re.sub(r"\\kf(?![A-Za-z])", karaoke_back, out)
    else:  # pragma: no cover - guarded by the caller
        raise ToolError(f"unknown mode {mode!r}")
    return out, replacements


def ass_convert_tags(
    selection: Any,
    doc_id: str | None = None,
    mode: str = "to_ass",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Normalise legacy SSA override tags to their ASS spelling (or back).

    Only the inside of override blocks is touched; visible text is never
    rewritten, and blocks are rebuilt with their braces intact.

    Args:
        selection: selection spelling (``None`` = every line).
        doc_id: document to edit; the current one when omitted.
        mode: ``"to_ass"`` (default) rewrites the legacy spellings to ASS —
            ``\\a1``..``\\a11`` (SSA alignment) become the matching ``\\an1``..
            ``\\an9`` and ``\\K`` becomes ``\\kf``.  ``"to_ssa"`` does the
            opposite (``\\an`` back to ``\\a``, ``\\kf`` back to ``\\K``).
        dry_run: report the replacements without writing (no snapshot).

    Returns ``{"doc_id", "mode", "dry_run", "count", "changed", "written",
    "replacement_count", "lines": [{"index", "before", "after", "changed",
    "replacements": [{"from", "to", "kind", "note"}]}]}``; every replacement is
    listed individually so the caller can audit the conversion.  ``changed`` is
    the number of lines whose text differs (also in ``dry_run``), ``written`` the
    number actually stored.  Snapshot-backed unless ``dry_run`` is set.

    Indices: override blocks are addressed by position, so no plain-character
    index and no raw offset is taken as input.  Because only tag *names* inside
    the braces change, both index maps of the visible characters are preserved
    exactly: the plain-character index of every visible character and the raw
    index of every character in the line are the same before and after.
    """
    if mode not in ("to_ass", "to_ssa"):
        raise ToolError(f"mode must be 'to_ass' or 'to_ssa', got {mode!r}")
    did, doc, indices = _doc_and_indices(selection, doc_id)
    events = doc.events()
    updates: list[tuple[int, str]] = []
    rows: list[dict[str, Any]] = []
    total = 0
    for i in indices:
        before = events[i].text
        parsed = T.parse(before)
        chunks: list[str] = []
        replacements: list[dict[str, Any]] = []
        for segment in parsed.segments:
            if isinstance(segment, T.TagBlock):
                rendered = segment.render()
                inner = rendered[1:-1] if rendered.startswith("{") else rendered
                new_inner, found = _convert_inner(inner, mode)
                replacements.extend(found)
                chunks.append("{" + new_inner + "}")
            else:
                chunks.append(segment.render())
        after = "".join(chunks)
        _sane(after, "ass_convert_tags")
        total += len(replacements)
        rows.append(
            {
                "index": i,
                "before": before,
                "after": after,
                "changed": after != before,
                "replacements": replacements,
            }
        )
        if after != before:
            updates.append((i, after))
    written = _batch_write(doc, did, updates) if not dry_run else 0
    return ok(
        doc_id=did,
        mode=mode,
        dry_run=bool(dry_run),
        count=len(indices),
        changed=sum(1 for row in rows if row["changed"]),
        written=written,
        replacement_count=total,
        lines=rows,
    )


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def register(mcp: Any, ws: Any = None) -> list[str]:
    """Register every ``ass_*`` tool of this module with a FastMCP instance.

    ``ws`` is accepted for symmetry with the other tool modules; the tools use
    the ``workspace`` singleton from ``tools.base`` (when ``ws`` is passed it
    replaces that singleton for this module).  Returns the sorted list of
    registered tool names.
    """
    global workspace
    if ws is not None:
        workspace = ws
    names: list[str] = []
    for name, obj in sorted(globals().items()):
        if not name.startswith("ass_") or not callable(obj):
            continue
        if getattr(obj, "__module__", None) != __name__:
            continue
        mcp.tool()(obj)
        names.append(name)
    return sorted(names)
