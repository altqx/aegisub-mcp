"""Override-tag engine: parsing, editing, stripping and typesetting helpers.

Every function here works on *raw ASS override text* (the contents of a
Dialogue ``Text`` field, tags included) and is careful never to disturb parts
of the text it did not touch, so round-tripping a line through these helpers is
safe.

Tag syntax reference: ``{\\tag1\\tag2(value)}visible text{\\tag3}``.
Parentheses may nest (``\\t(0,500,\\clip(1,m 0 0 l 5 0 5 5))``), and tag
arguments can also be written bare (``\\b1``, ``\\fnArial``, ``\\c&HFFFFFF&``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

# ---------------------------------------------------------------------------
# Tag taxonomy
# ---------------------------------------------------------------------------

#: Tags that switch the leading drawing mode on/off (``\p``/``\pbo``).
DRAWING_TAGS = {"p", "pbo"}
#: Karaoke timing tags, in canonical spelling.
KARAOKE_TAGS = {"k", "kf", "ko", "kt"}
#: Clip tags (geometry handled in :mod:`aegisub_mcp.asscore.drawing`).
CLIP_TAGS = {"clip", "iclip"}
#: Animation tags.
ANIMATION_TAGS = {"t", "fad", "fade"}
#: Positioning/layout tags.
LAYOUT_TAGS = {
    "pos", "move", "org", "an", "a", "q", "fscx", "fscy", "fsc", "fr", "frz", "frx",
    "fry", "fax", "fay", "fsp", "fs", "fn", "fe",
}
#: Colour tags.
COLOR_TAGS = {
    "c", "1c", "2c", "3c", "4c", "alpha", "1a", "2a", "3a", "4a",
    "1c", "1a", "2c", "2a", "3c", "3a", "4c", "4a",
}
#: Bool-ish toggles plus the border/shadow/edge family.
STYLE_TAGS = {"b", "i", "u", "s", "bord", "shad", "be", "blur", "xbord", "ybord", "xshad", "yshad"}
#: Line-level reset/style reference.
RESET_TAGS = {"r"}

#: Convenience groups used by :func:`strip_tags`.
TAG_GROUPS = {
    "transform": {"t"},
    "fade": {"fad", "fade"},
    "clip": CLIP_TAGS,
    "drawing": DRAWING_TAGS,
    "karaoke": KARAOKE_TAGS,
    "layout": LAYOUT_TAGS,
    "color": COLOR_TAGS,
    "style": STYLE_TAGS,
    "reset": RESET_TAGS,
    "animation": ANIMATION_TAGS,
}

#: Tags whose absence means "use the style value"; used when restoring state
#: after wrapping a character range (see :func:`wrap_range`).
NEUTRAL_VALUES = {
    "b": "0", "i": "0", "u": "0", "s": "0", "be": "0", "blur": "0",
    "xbord": "0", "ybord": "0", "xshad": "0", "yshad": "0",
    "fscx": "100", "fscy": "100", "fsp": "0", "fr": "0", "frz": "0", "frx": "0", "fry": "0",
    "fax": "0", "fay": "0", "alpha": "&H00&", "1a": "&H00&", "2a": "&H00&", "3a": "&H00&", "4a": "&H00&",
    "p": "0", "pbo": "0",
}

_TAG_START_RE = re.compile(r"\\(?P<name>[1-4]?[A-Za-z]+)")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Tag:
    """A single override tag such as ``\\pos(10,20)`` or ``\\b1``."""

    name: str
    arg: str = ""
    paren: bool = False
    raw: str = ""
    dirty: bool = False

    def __post_init__(self) -> None:
        if not self.raw:
            self.raw = self.render()

    # -- rendering ----------------------------------------------------------
    def render(self) -> str:
        if not self.dirty and self.raw:
            return self.raw
        if self.paren:
            return f"\\{self.name}({self.arg})"
        return f"\\{self.name}{self.arg}"

    # -- typed accessors ----------------------------------------------------
    @property
    def value(self) -> str:
        return self.arg

    def float_arg(self, default: float = 0.0) -> float:
        try:
            return float(self.arg)
        except (TypeError, ValueError):
            return default

    def int_arg(self, default: int = 0) -> int:
        try:
            return int(float(self.arg))
        except (TypeError, ValueError):
            return default

    def args(self, count: int | None = None) -> list[str]:
        """Split a parenthesised argument list on top-level commas."""
        parts = split_args(self.arg)
        if count is not None:
            while len(parts) < count:
                parts.append("")
            parts = parts[:count]
        return parts

    def float_args(self, count: int | None = None, default: float = 0.0) -> list[float]:
        out = []
        for part in self.args(count):
            try:
                out.append(float(part.strip()))
            except (TypeError, ValueError):
                out.append(default)
        return out

    # -- specific tag shapes ------------------------------------------------
    def transform_parts(self) -> dict | None:
        """For ``\\t``: ``{'t1', 't2', 'accel', 'tags'}`` (times in ms)."""
        if self.name != "t":
            return None
        parts = [p.strip() for p in split_args(self.arg)]
        times: list[float] = []
        tags = ""
        for part in parts:
            if part.startswith("\\"):
                tags = part
                break
            try:
                times.append(float(part))
            except ValueError:
                tags = part
                break
        t1 = t2 = 0
        accel = 1.0
        if len(times) == 1:
            t1 = t2 = times[0]
        elif len(times) == 2:
            t1, t2 = times
        elif len(times) >= 3:
            t1, t2, accel = times[0], times[1], times[2]
        return {"t1": t1, "t2": t2, "accel": accel, "tags": tags}

    def fade_parts(self) -> list[float]:
        if self.name != "fad":
            return []
        return self.float_args()

    def clip_arg(self) -> str:
        """Raw argument of ``\\clip``/``\\iclip`` (geometry lives in drawing.py)."""
        return self.arg

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return self.render()


@dataclass
class TextSegment:
    text: str

    def render(self) -> str:
        return self.text


@dataclass
class TagBlock:
    """An override block ``{...}``."""

    tags: list[Tag] = field(default_factory=list)
    raw: str = ""  # inner text exactly as written (without the braces)
    dirty: bool = False

    def render(self) -> str:
        if not self.dirty and self.raw != "":
            return "{" + self.raw + "}"
        return "{" + "".join(tag.render() for tag in self.tags) + "}"

    def tag_names(self) -> list[str]:
        return [t.name for t in self.tags]


@dataclass
class ParsedText:
    segments: list[TextSegment | TagBlock]

    # -- convenience --------------------------------------------------------
    def render(self) -> str:
        return "".join(seg.render() for seg in self.segments)

    def blocks(self) -> list[TagBlock]:
        return [s for s in self.segments if isinstance(s, TagBlock)]

    def tags(self) -> list[Tag]:
        out: list[Tag] = []
        for block in self.blocks():
            out.extend(block.tags)
        return out

    def text_segments(self) -> list[TextSegment]:
        return [s for s in self.segments if isinstance(s, TextSegment)]

    def plain_text(self) -> str:
        return "".join(s.text for s in self.text_segments())

    # -- character mapping --------------------------------------------------
    def char_positions(self) -> list[tuple[int, str]]:
        """``(raw_index, char)`` for every visible character, in order.

        ``raw_index`` points at the character inside the rendered string, so it
        can be used to splice override blocks in without cutting existing ones.
        """
        positions: list[tuple[int, str]] = []
        cursor = 0
        for seg in self.segments:
            rendered = seg.render()
            if isinstance(seg, TextSegment):
                for i, ch in enumerate(seg.text):
                    positions.append((cursor + i, ch))
            cursor += len(rendered)
        return positions

    def raw_index_of_char(self, plain_index: int) -> int | None:
        """Raw-string index of the *start* of the given visible character."""
        positions = self.char_positions()
        if plain_index < 0:
            plain_index += len(positions)
        if not 0 <= plain_index < len(positions):
            return None
        return positions[plain_index][0]

    def raw_index_after_char(self, plain_index: int) -> int | None:
        """Raw-string index just after the given visible character."""
        positions = self.char_positions()
        if plain_index < 0:
            plain_index += len(positions)
        if not 0 <= plain_index < len(positions):
            return None
        raw_index, char = positions[plain_index]
        return raw_index + len(char)

    def state_at(self, plain_index: int) -> dict[str, Tag]:
        """Effective tag state (last-wins) at the given visible character."""
        state: dict[str, Tag] = {}
        plain_seen = 0
        for seg in self.segments:
            if isinstance(seg, TextSegment):
                if plain_seen >= plain_index:
                    break
                plain_seen += len(seg.text)
            else:
                for tag in seg.tags:
                    state[tag.name] = tag
        return state

    def drawing_flags(self) -> list[bool]:
        """For each segment, whether it is drawing data (``\\p`` mode active)."""
        active = False
        out: list[bool] = []
        for seg in self.segments:
            if isinstance(seg, TagBlock):
                for tag in seg.tags:
                    if tag.name == "p":
                        active = tag.int_arg(0) > 0
                    elif tag.name == "r":
                        active = False
            out.append(active)
        return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def split_args(arg: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    parts: list[str] = []
    depth = 0
    current = ""
    for ch in arg:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth = max(0, depth - 1)
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return parts


def _match_paren(text: str, start: int) -> int:
    """Index just past the closing paren matching ``text[start] == '('``."""
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def parse_block(body: str) -> list[Tag]:
    """Parse the inside of an override block (without braces) into tags."""
    tags: list[Tag] = []
    i = 0
    n = len(body)
    while i < n:
        if body[i] != "\\":
            i += 1
            continue
        m = _TAG_START_RE.match(body, i)
        if not m:
            i += 1
            continue
        name = m.group("name")
        name = _canon_name(name)
        j = m.end()
        start = i
        if j < n and body[j] == "(":
            end = _match_paren(body, j)
            inner = body[j + 1 : end - 1] if end > j + 1 else ""
            tags.append(Tag(name=name, arg=inner, paren=True, raw=body[start:end]))
            i = end
        else:
            k = j
            while k < n and body[k] != "\\":
                k += 1
            arg = body[j:k]
            tags.append(Tag(name=name, arg=arg, paren=False, raw=body[start:k]))
            i = k
    return tags


def _canon_name(name: str) -> str:
    """Canonicalise tag spelling (``\\K`` -> ``\\kf`` etc.)."""
    lowered = name
    if lowered == "K":
        return "kf"
    return lowered


def parse(text: str) -> ParsedText:
    """Split override text into text segments and tag blocks."""
    segments: list[TextSegment | TagBlock] = []
    i = 0
    n = len(text)
    buf = ""
    while i < n:
        ch = text[i]
        if ch == "{":
            # A block only starts when the brace is an override block; ASS has no
            # escaping, so treat every '{' as a block opening (Aegisub does too).
            end = _find_block_end(text, i)
            if end is None:
                buf += text[i:]
                i = n
                break
            if buf:
                segments.append(TextSegment(buf))
                buf = ""
            body = text[i + 1 : end - 1]
            segments.append(TagBlock(tags=parse_block(body), raw=body))
            i = end
        else:
            buf += ch
            i += 1
    if buf:
        segments.append(TextSegment(buf))
    return ParsedText(segments)


def _find_block_end(text: str, start: int) -> int | None:
    """Index just past the closing ``}`` of the override block at ``start``.

    ASS override blocks cannot nest braces, but ``\\t(...)`` arguments contain
    no braces either, so the first ``}`` terminates the block.
    """
    idx = text.find("}", start + 1)
    return None if idx < 0 else idx + 1


def parse_tags(text: str) -> list[Tag]:
    return parse(text).tags()


def tag_names(text: str) -> list[str]:
    return [t.name for t in parse_tags(text)]


def get_tag(text: str, name: str, *, index: int = -1) -> Tag | None:
    matches = [t for t in parse_tags(text) if t.name == name or _alt_name(t.name) == name]
    if not matches:
        return None
    try:
        return matches[index]
    except IndexError:
        return None


def _alt_name(name: str) -> str:
    return name


def get_tag_value(text: str, name: str, default: str | None = None) -> str | None:
    tag = get_tag(text, name)
    return tag.arg if tag else default


# ---------------------------------------------------------------------------
# Serialising / normalising
# ---------------------------------------------------------------------------


def serialize(parsed: ParsedText | str) -> str:
    if isinstance(parsed, str):
        parsed = parse(parsed)
    return parsed.render()


def plain_text(text: str) -> str:
    """Visible text with all override blocks removed (drawings are kept)."""
    return parse(text).plain_text()


def strip_tags(
    text: str,
    *,
    keep: Iterable[str] | None = None,
    keep_groups: Iterable[str] | None = None,
    remove_groups: Iterable[str] | None = None,
    keep_drawing: bool = False,
    keep_karaoke: bool = False,
) -> str:
    """Remove override tags.

    ``keep`` lists tag names to preserve. ``keep_groups`` keeps *only* the tags
    belonging to the named :data:`TAG_GROUPS` (``transform``, ``clip``,
    ``karaoke``, ...) and drops everything else. ``remove_groups`` removes only
    the listed groups; everything else is kept. Drawing commands (``\\p1`` +
    path data) are removed unless ``keep_drawing`` is set.
    """
    keep_set = {_canon_name(k) for k in (keep or [])}
    groups = {g: set(TAG_GROUPS.get(g, set())) for g in (remove_groups or [])}
    remove_only = bool(remove_groups)
    keep_only = bool(keep_groups)
    group_keep: set[str] = set()
    for g in keep_groups or []:
        group_keep |= set(TAG_GROUPS.get(g, set()))
    to_remove: set[str] = set()
    for names in groups.values():
        to_remove |= names
    parsed = parse(text)
    out = []
    drawing_active = False
    for seg in parsed.segments:
        if isinstance(seg, TagBlock):
            tags = []
            for tag in seg.tags:
                name = tag.name
                if name == "p":
                    drawing_active = tag.int_arg(0) > 0
                elif name == "r":
                    drawing_active = False
                if name in keep_set or name in group_keep:
                    tags.append(tag)
                    continue
                if keep_only:
                    # strict keep-list: only named groups/names survive, though an
                    # explicit keep_drawing / keep_karaoke still wins
                    if name in DRAWING_TAGS and keep_drawing:
                        tags.append(tag)
                    elif name in KARAOKE_TAGS and keep_karaoke:
                        tags.append(tag)
                    continue
                if remove_only:
                    if name in to_remove:
                        continue
                    tags.append(tag)
                    continue
                # default: not asked to keep -> everything goes, except drawings
                # and karaoke tags when those were explicitly requested
                if name in DRAWING_TAGS and keep_drawing:
                    tags.append(tag)
                    continue
                if name in KARAOKE_TAGS and keep_karaoke:
                    tags.append(tag)
                    continue
            if tags:
                block = TagBlock(tags=tags, dirty=True)
                out.append(block.render())
        else:
            if drawing_active and not keep_drawing:
                continue
            out.append(seg.text)
    return "".join(out)


def normalize(text: str, *, drop_redundant: bool = True) -> str:
    """Rewrite override blocks into canonical form (``\\tag(arg)`` order kept).

    Redundant tags (a value identical to the previous effective value and not
    inside a transform) are dropped when ``drop_redundant`` is set.
    """
    parsed = parse(text)
    out: list[str] = []
    state: dict[str, str] = {}
    for seg in parsed.segments:
        if isinstance(seg, TagBlock):
            kept: list[str] = []
            for tag in seg.tags:
                rendered = f"\\{tag.name}" + (f"({tag.arg})" if tag.paren else tag.arg)
                if drop_redundant and tag.name not in ("t", "r") and state.get(tag.name) == rendered:
                    continue
                state[tag.name] = rendered
                kept.append(rendered)
            if kept:
                out.append("{" + "".join(kept) + "}")
        else:
            out.append(seg.text)
    return "".join(out)


# ---------------------------------------------------------------------------
# Editing
# ---------------------------------------------------------------------------


def _tag_text(name: str, arg: str, *, paren: bool | None = None) -> str:
    if paren is None:
        paren = bool(arg) and not re.fullmatch(r"[0-9.\-]+", arg) and not arg.startswith("&H")
    return f"\\{name}({arg})" if paren else f"\\{name}{arg}"


def prepend_tags(text: str, tags: str) -> str:
    """Ensure an override block with ``tags`` at the very beginning."""
    body = tags if tags.startswith("\\") else "\\" + tags
    if text.startswith("{") and not _is_special_only(text):
        end = text.find("}")
        if end < 0:
            return text
        inner = text[1:end]
        merged = _merge_tag_strings(body, inner)
        return "{" + merged + "}" + text[end + 1 :]
    return "{" + body + "}" + text


def _is_special_only(text: str) -> bool:
    """True when a leading ``{...}`` block contains no real override tags."""
    return False


def append_tags(text: str, tags: str) -> str:
    body = tags if tags.startswith("\\") else "\\" + tags
    return text + "{" + body + "}"


def _merge_tag_strings(new_tags: str, existing: str) -> str:
    """Merge two tag strings, keeping existing order and updating duplicates."""
    new_parsed = parse_block(new_tags)
    existing_parsed = parse_block(existing)
    new_names = {t.name for t in new_parsed}
    kept = [t.render() for t in existing_parsed if t.name not in new_names]
    # drop redundant leading backslash duplication
    merged = "".join(t.render() for t in new_parsed) + "".join(kept)
    return merged


def set_tag(text: str, name: str, arg: str, *, paren: bool | None = None,
            create: bool = True, position: str = "start") -> str:
    """Set ``\\name`` to ``arg`` everywhere it appears (or create it once).

    Existing occurrences are updated in place (first occurrence wins for
    position, later duplicates are removed) which keeps animated ``\\t`` blocks
    intact when they contain the same tag.
    """
    parsed = parse(text)
    replacement = _tag_text(name, arg, paren=paren)
    updated = False
    new_segments: list[TextSegment | TagBlock] = []
    for seg in parsed.segments:
        if isinstance(seg, TagBlock):
            kept: list[str] = []
            for tag in seg.tags:
                if tag.name == name:
                    if not updated:
                        kept.append(replacement)
                        updated = True
                    # drop duplicates of the same tag in the same block
                    continue
                kept.append(tag.render())
            if kept:
                new_segments.append(TagBlock(tags=parse_block("".join(kept)), raw="".join(kept)))
        else:
            new_segments.append(seg)
    result = ParsedText(new_segments).render()
    if not updated and create:
        if position == "start":
            return prepend_tags(result, replacement)
        if position == "end":
            return append_tags(result, replacement)
        if isinstance(position, int) and position > 0:
            return _insert_block_at_plain_index(result, position, replacement)
    return result


def remove_tags(text: str, names: Iterable[str]) -> str:
    """Remove every occurrence of the given tag names."""
    wanted = {_canon_name(n) for n in names}
    parsed = parse(text)
    out: list[str] = []
    for seg in parsed.segments:
        if isinstance(seg, TagBlock):
            kept = [t.render() for t in seg.tags if t.name not in wanted]
            if kept:
                out.append("{" + "".join(kept) + "}")
        else:
            out.append(seg.text)
    return "".join(out)


def _insert_block_at_plain_index(text: str, plain_index: int, override: str,
                                 *, before: bool = True) -> str:
    parsed = parse(text)
    positions = parsed.char_positions()
    if not positions:
        return prepend_tags(text, override)
    if plain_index >= len(positions):
        return append_tags(text, override)
    raw_index = positions[plain_index][0] if before else positions[plain_index][0] + len(positions[plain_index][1])
    body = override if override.startswith("\\") else "\\" + override
    return text[:raw_index] + "{" + body + "}" + text[raw_index:]


def insert_override_at_char(text: str, plain_index: int, override: str) -> str:
    """Insert ``{override}`` immediately before the given visible character."""
    return _insert_block_at_plain_index(text, plain_index, override, before=True)


def insert_override_after_char(text: str, plain_index: int, override: str) -> str:
    return _insert_block_at_plain_index(text, plain_index, override, before=False)


def wrap_range(text: str, start: int, end: int, override: str, *,
               restore: bool = True, style_name: str | None = None,
               keep_previous: bool = True) -> tuple[str, list[str]]:
    """Wrap visible characters ``[start, end)`` in ``{override}``.

    Returns ``(new_text, warnings)``. When ``restore`` is set the original
    effective state is re-established after the range: values that existed
    before are re-emitted verbatim, and tags that did not exist before are
    reset to their neutral value (or, when a style is supplied, to nothing and
    reported as a warning, since only ``\\r`` can fully undo a style value).
    """
    parsed = parse(text)
    positions = parsed.char_positions()
    if not positions:
        return text, ["no visible characters to wrap"]
    warnings: list[str] = []
    start = max(0, start)
    end = min(end, len(positions))
    if end <= start:
        return text, ["empty character range"]

    before_state = parsed.state_at(start)
    inside_names = {t.name for t in parse_block(override if override.startswith("\\") else "\\" + override)}
    body = override if override.startswith("\\") else "\\" + override

    insert_at = positions[start][0]
    after_index = positions[end - 1][0] + len(positions[end - 1][1])

    result = text[:insert_at] + "{" + body + "}" + text[insert_at:after_index]
    if not restore:
        return result + text[after_index:], warnings

    restore_parts: list[str] = []
    for name in inside_names:
        if name in before_state:
            prev = before_state[name]
            restore_parts.append(f"\\{prev.name}" + (f"({prev.arg})" if prev.paren else prev.arg))
        elif name in NEUTRAL_VALUES:
            restore_parts.append(f"\\{name}{NEUTRAL_VALUES[name]}")
        else:
            if keep_previous and style_name:
                warnings.append(
                    f"\\{name} had no previous value; resetting the line with \\r is needed to fully undo it"
                )
            else:
                warnings.append(f"\\{name} had no previous value and cannot be fully undone without \\r")
    if restore_parts:
        result += "{" + "".join(restore_parts) + "}"
    return result + text[after_index:], warnings


def apply_tag_to_range(text: str, start: int, end: int, override: str, **kw):
    """Alias of :func:`wrap_range` with a friendlier name for tool callers."""
    return wrap_range(text, start, end, override, **kw)


def apply_tag_to_all_blocks(text: str, override: str, *, skip_transform: bool = True) -> str:
    """Add ``override`` to the first block of every line of text."""
    parts = re.split(r"(\\N|\\n)", text)
    out: list[str] = []
    for i, part in enumerate(parts):
        if part in (r"\N", r"\n"):
            out.append(part)
            continue
        out.append(prepend_tags(part, override))
    return "".join(out)


# ---------------------------------------------------------------------------
# Tag construction helpers
# ---------------------------------------------------------------------------


def tag(name: str, arg: str = "", *, paren: bool | None = None) -> str:
    return _tag_text(name, arg, paren=paren)


def tags(*items: str) -> str:
    out = ""
    for item in items:
        out += item if item.startswith("\\") else "\\" + item
    return out


def pos_tag(x: float, y: float, *, an: int | None = None) -> str:
    prefix = f"\\an{an}" if an else ""
    return f"{prefix}\\pos({_num(x)},{_num(y)})"


def move_tag(x1, y1, x2, y2, t1=None, t2=None, *, an: int | None = None) -> str:
    prefix = f"\\an{an}" if an else ""
    if t1 is None and t2 is None:
        return f"{prefix}\\move({_num(x1)},{_num(y1)},{_num(x2)},{_num(y2)})"
    return (
        f"{prefix}\\move({_num(x1)},{_num(y1)},{_num(x2)},{_num(y2)},"
        f"{_num(t1 or 0)},{_num(t2 or 0)})"
    )


def an_tag(value: int) -> str:
    return f"\\an{int(value)}"


def fade_tag(in_ms: int, out_ms: int) -> str:
    return f"\\fad({int(in_ms)},{int(out_ms)})"


def fade_complex_tag(a1: int, a2: int, a3: int, t1: int, t2: int, t3: int, t4: int) -> str:
    return f"\\fade({int(a1)},{int(a2)},{int(a3)},{int(t1)},{int(t2)},{int(t3)},{int(t4)})"


def transform_tag(t1: int, t2: int, inner: str, *, accel: float = 1.0) -> str:
    inner = inner if inner.startswith("\\") else "\\" + inner
    if abs(accel - 1.0) < 1e-9:
        return f"\\t({int(t1)},{int(t2)},{inner})"
    return f"\\t({int(t1)},{int(t2)},{_num(accel)},{inner})"


def alpha_tag(value: str, *, which: int | None = None, html: bool = False) -> str:
    """``\\alpha``/``\\1a`` tag; accepts ``&H80&`` or ``0-255``/``#RRGGBBAA``."""
    if html:
        from . import assutil

        r, g, b, a = assutil.parse_ass_color(value)
        arg = f"&H{a:02X}&"
    elif value.startswith("&"):
        arg = value
    else:
        arg = f"&H{int(float(value)) & 0xFF:02X}&"
    name = "alpha" if which is None else f"{which}a"
    return f"\\{name}{arg}"


def color_tag(value: str, *, which: int = 1, html: bool = False,
              include_alpha: bool = False) -> str:
    from . import assutil

    if html:
        r, g, b, a = assutil.parse_ass_color(value)
        if include_alpha:
            arg = f"&H{a:02X}{b:02X}{g:02X}{r:02X}&"
        else:
            arg = f"&H{b:02X}{g:02X}{r:02X}&"
        return f"\\{which}c{arg}"
    arg = value if value.startswith("&") else f"&H{int(value, 16):06X}&"
    return f"\\{which}c{arg}"


def _num(value) -> str:
    from .assutil import round_coord

    return round_coord(value, 2)


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def tag_summary(text: str) -> dict:
    """Counts and structure summary used by QC and inspection tools."""
    parsed = parse(text)
    blocks = parsed.blocks()
    names: dict[str, int] = {}
    for tag_ in parsed.tags():
        names[tag_.name] = names.get(tag_.name, 0) + 1
    return {
        "blocks": len(blocks),
        "tags": len(parsed.tags()),
        "tag_names": names,
        "visible_chars": len(parsed.plain_text()),
        "has_drawing": any(
            tag_.name == "p" and tag_.int_arg(0) > 0 for tag_ in parsed.tags()
        ),
        "karaoke": [t.name for t in parsed.tags() if t.name in KARAOKE_TAGS],
        "clip": [t.arg for t in parsed.tags() if t.name in CLIP_TAGS],
        "transforms": [t.arg for t in parsed.tags() if t.name == "t"],
    }


def drawing_parts(text: str) -> tuple[str, str, str]:
    """Split a drawing line into ``(prefix, drawing, suffix)``.

    ``prefix`` is everything up to and including the ``\\p1`` (or similar) tag
    block, ``drawing`` is the path data, ``suffix`` is whatever follows (usually
    a closing ``{\\p0}`` block).
    """
    parsed = parse(text)
    prefix_parts: list[str] = []
    drawing_parts_: list[str] = []
    suffix_parts: list[str] = []
    active = False
    consumed_prefix = False
    for seg in parsed.segments:
        rendered = seg.render()
        if isinstance(seg, TagBlock):
            for tag_ in seg.tags:
                if tag_.name == "p":
                    active = tag_.int_arg(0) > 0
            if active and not drawing_parts_:
                prefix_parts.append(rendered)
                consumed_prefix = True
                continue
            if not active and drawing_parts_:
                suffix_parts.append(rendered)
                continue
        if active:
            drawing_parts_.append(rendered)
            continue
        (drawing_parts_ and suffix_parts or prefix_parts).append(rendered)
    drawing = "".join(drawing_parts_)
    return "".join(prefix_parts), drawing.strip(), "".join(suffix_parts)


def drawing_state(text: str) -> int:
    """Value of ``\\p`` in effect at the end of ``text`` (0 = normal text)."""
    state = 0
    for seg in parse(text).segments:
        if isinstance(seg, TagBlock):
            for tag_ in seg.tags:
                if tag_.name == "p":
                    state = tag_.int_arg(0)
                elif tag_.name == "r":
                    state = 0
    return state


def parse_transform(text: str) -> dict | None:
    """Parse a ``\\t(...)`` argument (with or without the tag wrapper)."""
    body = text.strip()
    if body.startswith("\\t"):
        body = body[2:].strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    if not body:
        return None
    parsed = Tag(name="t", arg=body, paren=True).transform_parts()
    if parsed is None:  # pragma: no cover - defensive
        return None
    parsed["tag_list"] = parse_block(parsed["tags"]) if parsed["tags"] else []
    return parsed


def char_positions(text: str) -> list[tuple[int, str]]:
    """``(raw_index, char)`` for every visible character of ``text``."""
    return parse(text).char_positions()


def plain_len(text: str) -> int:
    """Number of visible characters in ``text``."""
    return len(plain_text(text))


def plain_to_raw_index(text: str, plain_index: int) -> int | None:
    """Raw index of the visible character at ``plain_index``."""
    return parse(text).raw_index_of_char(plain_index)


def raw_to_plain_index(text: str, raw_index: int) -> int | None:
    """Visible-character index of ``raw_index`` (``None`` inside a tag block)."""
    mapping: list[int | None] = []
    plain = 0
    for seg in parse(text).segments:
        if isinstance(seg, TagBlock):
            mapping.extend([None] * len(seg.render()))
        else:
            for _ in seg.text:
                mapping.append(plain)
                plain += 1
    if 0 <= raw_index < len(mapping):
        return mapping[raw_index]
    return None


def is_drawing(text: str) -> bool:
    parsed = parse(text)
    for seg in parsed.segments:
        if isinstance(seg, TagBlock):
            for tag_ in seg.tags:
                if tag_.name == "p" and tag_.int_arg(0) > 0:
                    return True
    return False


def iter_visible_chars(text: str) -> Iterator[tuple[int, str, dict[str, Tag]]]:
    """Yield ``(plain_index, char, state)`` for each visible character."""
    parsed = parse(text)
    state: dict[str, Tag] = {}
    plain_index = 0
    for seg in parsed.segments:
        if isinstance(seg, TagBlock):
            for tag_ in seg.tags:
                state[tag_.name] = tag_
        else:
            for ch in seg.text:
                yield plain_index, ch, dict(state)
                plain_index += 1
