"""Karaoke support: syllable handling, timing maths and ``\\k`` tag generation.

Covers the parts of karaoke work that do not need a Lua VM:

* splitting a line into syllables (marker, per-character, per-word or regex)
* reading existing ``\\k`` / ``\\kf`` / ``\\ko`` / ``\\kt`` timing
* generating ``\\k`` tags from explicit or distributed timings
* retiming, scaling, shifting, converting karaoke tag types
* a karaoke-aware view of a line used by the template engine
  (:mod:`aegisub_mcp.asscore.templates`)

The unit of ``\\k`` is the centisecond (``\\k20`` is 200 ms), while every
function here takes and returns milliseconds unless the name says otherwise.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import tags as T

KARAOKE_TAG_KINDS = ("k", "kf", "ko", "kt")
#: ``\K`` is an alias for ``\kf`` in VSFilter and Aegisub.
KARAOKE_ALIASES = {"K": "kf"}


@dataclass
class Syllable:
    """One timed karaoke syllable."""

    text: str
    start_ms: int = 0
    end_ms: int = 0
    kind: str = "k"
    index: int = 0
    prefix: str = ""  # override text that precedes the syllable in the source

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def duration_cs(self) -> int:
        return int(round(self.duration_ms / 10.0))

    @property
    def mid_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2

    @property
    def visible_text(self) -> str:
        return T.plain_text(self.text)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.visible_text,
            "raw": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "duration_cs": self.duration_cs,
            "kind": self.kind,
        }


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

_THAI_FOLLOW = (
    set(range(0x0E30, 0x0E34))  # sara a / mai han akat / sara aa / sara am
    | set(range(0x0E34, 0x0E3B))  # i, ii, ue, uee, u, uu, phinthu
    | set(range(0x0E47, 0x0E4F))  # tone marks, thanthakhat, nikhahit
)
#: Thai leading vowels (e, ae, o, ai, ao) attach to the *following* consonant.
_THAI_LEAD = set(range(0x0E40, 0x0E45))
_THAI_RANGE = (0x0E00, 0x0E7F)


def _is_combining(ch: str) -> bool:
    if unicodedata.combining(ch):
        return True
    code = ord(ch)
    if _THAI_RANGE[0] <= code <= _THAI_RANGE[1]:
        return code in _THAI_FOLLOW
    return False


def split_syllables(
    text: str,
    *,
    mode: str = "marker",
    marker: str = "|",
    pattern: str | None = None,
    keep_marker: bool = False,
) -> list[str]:
    """Split karaoke source text into (raw, tag-preserving) syllable strings.

    ``mode`` is one of:

    ``marker``
        split on ``marker`` (``|`` by default) — the usual typesetting workflow
    ``char``
        one visible character per syllable; combining marks stay attached to the
        preceding base character (safe for Thai/Latin accents)
    ``word``
        split on whitespace, keeping the space with the preceding syllable
    ``regex``
        split using ``pattern`` (a capture-free regex)

    Override blocks are kept and always travel with the syllable that follows
    them; empty syllables are dropped (unless ``keep_marker`` is set).
    """
    if mode == "char":
        return _split_by_char(text)
    if mode == "word":
        return _split_by_word(text)
    if mode == "regex":
        if not pattern:
            raise ValueError("mode='regex' needs a pattern")
        return _split_by_pattern(text, pattern, keep_marker=keep_marker)
    return _split_by_marker(text, marker, keep_marker=keep_marker)


def _segments_with_tags(text: str) -> list[tuple[str, str]]:
    """``(prefix_tags, visible_char)`` pairs for the whole line."""
    parsed = T.parse(text)
    pending = ""
    out: list[tuple[str, str]] = []
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            pending += seg.render()
        else:
            for ch in seg.text:
                out.append((pending, ch))
                pending = ""
    return out


def _split_by_char(text: str) -> list[str]:
    pieces: list[str] = []
    buf_prefix = ""
    buf = ""
    lead_pending = False
    for prefix, ch in _segments_with_tags(text):
        if not buf:
            buf_prefix = prefix
            buf = ch
            lead_pending = ch in _THAI_LEAD
            continue
        if _is_combining(ch) or lead_pending:
            # combining mark, or the consonant that a leading vowel belongs to
            buf += prefix + ch
            lead_pending = False
            continue
        pieces.append(buf_prefix + buf)
        buf_prefix = prefix
        buf = ch
        lead_pending = ch in _THAI_LEAD
    if buf:
        pieces.append(buf_prefix + buf)
    return pieces


def _split_by_word(text: str) -> list[str]:
    parsed = T.parse(text)
    pieces: list[str] = []
    buffer = ""
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            buffer += seg.render()
            continue
        parts = re.split(r"(?<=\s)", seg.text)
        for part in parts:
            if not part:
                continue
            buffer += part
            if part.endswith((" ", "\t")):
                pieces.append(buffer)
                buffer = ""
    if buffer:
        pieces.append(buffer)
    return [p for p in pieces if T.plain_text(p)]


def _split_by_marker(text: str, marker: str, *, keep_marker: bool) -> list[str]:
    if not marker:
        return [text]
    parsed = T.parse(text)
    pieces: list[str] = []
    buffer = ""
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            buffer += seg.render()
            continue
        chunks = seg.text.split(marker)
        for i, chunk in enumerate(chunks):
            buffer += chunk
            if i < len(chunks) - 1:
                pieces.append(buffer + (marker if keep_marker else ""))
                buffer = ""
    if buffer:
        pieces.append(buffer)
    return [p for p in pieces if T.plain_text(p) or (keep_marker and marker in p)]


def _split_by_pattern(text: str, pattern: str, *, keep_marker: bool) -> list[str]:
    parsed = T.parse(text)
    out: list[str] = []
    buffer = ""
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            buffer += seg.render()
            continue
        for chunk in re.split(pattern, seg.text):
            if chunk:
                out.append(buffer + chunk)
                buffer = ""
    if buffer:
        out.append(buffer)
    return [p for p in out if T.plain_text(p)]


# ---------------------------------------------------------------------------
# reading existing karaoke
# ---------------------------------------------------------------------------


def parse_karaoke(text: str, *, default_kind: str = "k") -> list[Syllable]:
    """Return the syllables of ``text``.

    If the line has ``\\k``-family tags their durations are used, otherwise the
    line is treated as a single untimed syllable. Times are relative to the
    line start (0 ms) unless :func:`with_absolute_times` is used.
    """
    parsed = T.parse(text)
    syllables: list[Syllable] = []
    pending = ""
    current: Syllable | None = None
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            pending += seg.render()
            for tag in seg.tags:
                if tag.name in KARAOKE_TAGS_ALL:
                    kind = KARAOKE_ALIASES.get(tag.name, tag.name)
                    if current is not None:
                        syllables.append(current)
                    current = Syllable(text="", kind=kind or default_kind,
                                       index=len(syllables), prefix=pending)
                    current.duration_cs_override = max(0, tag.int_arg(0))  # type: ignore[attr-defined]
                    pending = ""
            continue
        if current is None:
            current = Syllable(text="", kind=default_kind, index=len(syllables), prefix=pending)
            pending = ""
            current.duration_cs_override = 0  # type: ignore[attr-defined]
        current.text += seg.text
    if current is not None:
        syllables.append(current)

    t = 0
    for syl in syllables:
        dur_cs = getattr(syl, "duration_cs_override", 0)
        syl.start_ms = t
        syl.end_ms = t + dur_cs * 10
        t = syl.end_ms
    return syllables


KARAOKE_TAGS_ALL = {"k", "kf", "ko", "kt", "K"}


def karaoke_timings(text: str, line_start_ms: int = 0, line_end_ms: int | None = None) -> list[dict]:
    """Per-syllable timing table with absolute times where known."""
    syllables = parse_karaoke(text)
    total = sum(s.duration_cs for s in syllables)
    if line_end_ms is not None and line_start_ms is not None and total <= 0:
        # untimed line: distribute what we know
        pieces = [s for s in syllables]
        span = max(0, line_end_ms - line_start_ms)
        share = span // max(1, len(pieces))
        for i, syl in enumerate(pieces):
            syl.start_ms = line_start_ms + i * share
            syl.end_ms = line_start_ms + (i + 1) * share if i < len(pieces) - 1 else line_end_ms
        return [s.to_dict() for s in pieces]
    for syl in syllables:
        syl.start_ms += line_start_ms
        syl.end_ms += line_start_ms
    return [s.to_dict() for s in syllables]


def karaoke_total_ms(text: str) -> int:
    return sum(s.duration_cs * 10 for s in parse_karaoke(text))


def karaoke_tag_kinds(text: str) -> list[str]:
    return [s.kind for s in parse_karaoke(text) if s.duration_cs]


# ---------------------------------------------------------------------------
# generating karaoke
# ---------------------------------------------------------------------------


def _k_tag(kind: str, duration_cs: int) -> str:
    return f"\\{kind}{max(0, int(round(duration_cs)))}"


def distribute_by_length(syllables: Sequence[str], total_ms: int,
                         *, weights: Sequence[float] | None = None) -> list[int]:
    """Split ``total_ms`` across syllables proportionally to their visible length."""
    if not syllables:
        return []
    if weights is None:
        weights = [max(1.0, float(len(T.plain_text(s)) or 1)) for s in syllables]
    total_weight = sum(weights) or 1.0
    raw = [total_ms * (w / total_weight) for w in weights]
    # round to centiseconds, then push the rounding remainder into the longest
    cs = [int(round(value / 10.0)) for value in raw]
    remainder = int(round(total_ms / 10.0)) - sum(cs)
    if remainder:
        target = max(range(len(cs)), key=lambda i: cs[i])
        cs[target] = max(0, cs[target] + remainder)
    return cs


def generate_k(
    text: str,
    *,
    kind: str = "k",
    durations_cs: Sequence[int] | None = None,
    start_ms: int = 0,
    end_ms: int | None = None,
    mode: str = "marker",
    marker: str = "|",
    pattern: str | None = None,
    replace_existing: bool = True,
) -> str:
    """Return ``text`` rewritten with ``\\k`` tags in front of every syllable.

    Durations come from ``durations_cs`` when given, otherwise the line span
    ``start_ms`` .. ``end_ms`` is distributed proportionally to syllable length.
    """
    if replace_existing:
        # markers are source data for marker mode: only drop them for the others
        text = remove_karaoke(text, drop_markers=(mode != "marker"))
    pieces = [_strip_karaoke_tags(p) for p in split_syllables(
        text, mode=mode, marker=marker, pattern=pattern)]
    if not pieces:
        return text
    if durations_cs is None:
        span = 0 if end_ms is None else max(0, int(end_ms) - int(start_ms))
        durations_cs = distribute_by_length(pieces, span)
    if len(durations_cs) < len(pieces):
        durations_cs = list(durations_cs) + [0] * (len(pieces) - len(durations_cs))
    out: list[str] = []
    for i, piece in enumerate(pieces):
        tag = _k_tag(kind, durations_cs[i])
        if i == 0:
            out.append(T.prepend_tags(piece, tag))
        else:
            out.append("{" + tag + "}" + piece)
    return "".join(out)


def remove_karaoke(text: str, *, drop_markers: bool = False) -> str:
    """Strip ``\\k``/``\\kf``/``\\ko``/``\\kt`` tags.

    Syllable markers (``|``) are kept by default — they are source data used by
    :func:`generate_k`; pass ``drop_markers=True`` to remove them too.
    """
    stripped = T.remove_tags(text, sorted(KARAOKE_TAGS_ALL))
    if drop_markers:
        stripped = stripped.replace("|", "")
    return stripped


def set_karaoke_kind(text: str, kind: str) -> str:
    """Convert every karaoke tag in ``text`` to ``kind`` (``k``, ``kf``, ``ko``, ``kt``)."""
    kind = KARAOKE_ALIASES.get(kind, kind)
    parsed = T.parse(text)
    out: list[str] = []
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            new_tags = []
            for tag in seg.tags:
                if tag.name in KARAOKE_TAGS_ALL:
                    new_tags.append(T.Tag(name=kind, arg=tag.arg, paren=False, dirty=True))
                else:
                    new_tags.append(tag)
            if new_tags:
                out.append(T.TagBlock(tags=new_tags, dirty=True).render())
        else:
            out.append(seg.render())
    return "".join(out)


def shift_karaoke(text: str, delta_ms: int) -> str:
    """Shift every karaoke duration by ``delta_ms`` (negative trims)."""
    parsed = T.parse(text)
    out: list[str] = []
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            new_tags = []
            for tag in seg.tags:
                if tag.name in KARAOKE_TAGS_ALL:
                    cs = max(0, tag.int_arg(0) + int(round(delta_ms / 10.0)))
                    new_tags.append(T.Tag(name=tag.name, arg=str(cs), paren=False, dirty=True))
                else:
                    new_tags.append(tag)
            if new_tags:
                out.append(T.TagBlock(tags=new_tags, dirty=True).render())
        else:
            out.append(seg.render())
    return "".join(out)


def scale_karaoke(text: str, factor: float, *, min_cs: int = 0) -> str:
    """Multiply every karaoke duration by ``factor``."""
    parsed = T.parse(text)
    out: list[str] = []
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            new_tags = []
            for tag in seg.tags:
                if tag.name in KARAOKE_TAGS_ALL:
                    cs = max(min_cs, int(round(tag.int_arg(0) * factor)))
                    new_tags.append(T.Tag(name=tag.name, arg=str(cs), paren=False, dirty=True))
                else:
                    new_tags.append(tag)
            if new_tags:
                out.append(T.TagBlock(tags=new_tags, dirty=True).render())
        else:
            out.append(seg.render())
    return "".join(out)


def apply_timings(text: str, timings: Sequence[dict] | Sequence[int], *,
                  kind: str | None = None, mode: str = "marker", marker: str = "|",
                  pattern: str | None = None) -> str:
    """Rewrite ``text`` with explicit per-syllable timings.

    ``timings`` may be a list of millisecond durations, or dicts with either
    ``duration_ms`` / ``duration_cs`` or ``start_ms`` + ``end_ms``. Syllables are
    taken from the existing karaoke tags when present, otherwise from ``mode``.
    """
    if parse_karaoke_has_tags(text):
        pieces = _karaoke_piece_texts(text)
    else:
        pieces = split_syllables(text, mode=mode, marker=marker, pattern=pattern)
    pieces = [_strip_karaoke_tags(p) for p in pieces]
    values: list[int] = []
    for item in timings:
        if isinstance(item, dict):
            if "duration_cs" in item:
                values.append(int(item["duration_cs"]))
            elif "duration_ms" in item:
                values.append(int(round(int(item["duration_ms"]) / 10.0)))
            else:
                values.append(int(round((int(item.get("end_ms", 0)) - int(item.get("start_ms", 0))) / 10.0)))
        else:
            values.append(int(item))
    if not pieces:
        return text
    if len(values) < len(pieces):
        values = values + [0] * (len(pieces) - len(values))
    out: list[str] = []
    for i, piece in enumerate(pieces):
        kind_here = kind or _piece_kind(piece) or "k"
        tag = _k_tag(kind_here, values[i])
        out.append(T.prepend_tags(piece, tag) if i == 0 else "{" + tag + "}" + piece)
    return "".join(out)


def parse_karaoke_has_tags(text: str) -> bool:
    return any(tag.name in KARAOKE_TAGS_ALL for tag in T.parse_tags(text))


def _strip_karaoke_tags(piece: str) -> str:
    """Drop existing ``\\k`` tags from a syllable piece, keeping every other tag."""
    return T.remove_tags(piece, sorted(KARAOKE_TAGS_ALL))


def _karaoke_piece_texts(text: str) -> list[str]:
    """Split an existing karaoke line into ``prefix+text`` pieces, tags kept."""
    parsed = T.parse(text)
    pieces: list[str] = []
    buffer = ""
    started = False
    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            if any(t.name in KARAOKE_TAGS_ALL for t in seg.tags):
                if started:
                    pieces.append(buffer)
                buffer = seg.render()
                started = True
                continue
            buffer += seg.render()
            continue
        buffer += seg.render()
    if buffer:
        pieces.append(buffer)
    return [p for p in pieces if T.plain_text(p)]


def _piece_kind(piece: str) -> str | None:
    for tag in T.parse_tags(piece):
        if tag.name in KARAOKE_TAGS_ALL:
            return KARAOKE_ALIASES.get(tag.name, tag.name)
    return None


def retime_line(text: str, new_start_ms: int, new_end_ms: int, *, mode: str = "proportional") -> str:
    """Retime an existing karaoke line to a new span.

    ``proportional`` keeps the relative syllable lengths; ``even`` gives every
    syllable the same duration.
    """
    syllables = parse_karaoke(text)
    if not syllables:
        return text
    span = max(0, new_end_ms - new_start_ms)
    if mode == "even":
        share = span // len(syllables)
        durations = [int(round(share / 10.0))] * len(syllables)
    else:
        weights = [max(1, s.duration_cs) for s in syllables]
        durations = distribute_by_length([s.text for s in syllables], span, weights=weights)
    pieces = [_strip_karaoke_tags(p) for p in _karaoke_piece_texts(text)]
    if len(pieces) != len(syllables):
        pieces = [s.prefix + s.text for s in syllables]
    if len(durations) < len(pieces):
        durations = list(durations) + [0] * (len(pieces) - len(durations))
    out: list[str] = []
    for i, piece in enumerate(pieces):
        kind_here = syllables[i].kind or "k"
        tag = _k_tag(kind_here, durations[i])
        out.append(T.prepend_tags(piece, tag) if i == 0 else "{" + tag + "}" + piece)
    return "".join(out)


def syllable_weights_by_char_class(syllables: Sequence[str]) -> list[float]:
    """Weight syllables by character class (CJK counts 1, Latin words split)."""
    weights = []
    for syl in syllables:
        plain = T.plain_text(syl)
        w = 0.0
        for ch in plain:
            if ch.isspace():
                w += 0.35
            elif "CJK" in unicodedata.name(ch, ""):
                w += 1.0
            else:
                w += 0.55
        weights.append(max(0.5, w))
    return weights


def auto_timings(text: str, start_ms: int, end_ms: int, *, mode: str = "char",
                 marker: str = "|", weights: Sequence[float] | None = None) -> str:
    """Add ``\\k`` tags distributing ``start_ms``..``end_ms`` across syllables."""
    return generate_k(text, durations_cs=None, start_ms=start_ms, end_ms=end_ms,
                      mode=mode, marker=marker, replace_existing=True)
