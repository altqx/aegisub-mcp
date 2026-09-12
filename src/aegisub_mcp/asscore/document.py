"""Byte-faithful ASS/SSA document model.

Design goals (in priority order):

1. **Round-trip fidelity** – loading and saving an untouched file must return
   the *exact* same bytes: original encoding, BOM, line endings, trailing
   newline, unknown sections, comment lines, field padding and even malformed
   lines are preserved.
2. **Safe editing** – a line is rebuilt from its parsed fields only when one of
   those fields was actually modified, so unrelated formatting survives.
3. **Full coverage of the format** – ``[Script Info]``, ``[V4 Styles]``,
   ``[V4+ Styles]``, ``[Events]``, ``[Fonts]``, ``[Graphics]``,
   ``[Aegisub Extradata]`` and arbitrary vendor sections.
"""

from __future__ import annotations

import copy
import os
import re
import typing
import time
import uuid
from typing import Iterable, Iterator, Sequence

from . import assutil

# Canonical section kinds -----------------------------------------------------

KIND_INFO = "script_info"
KIND_STYLES = "styles"
KIND_EVENTS = "events"
KIND_FONTS = "fonts"
KIND_GRAPHICS = "graphics"
KIND_EXTRADATA = "extradata"
KIND_OTHER = "other"

_SECTION_ALIASES = {
    "script info": (KIND_INFO, "info"),
    "v4+ styles": (KIND_STYLES, "v4+"),
    "v4 styles": (KIND_STYLES, "v4"),
    "v4 styles+": (KIND_STYLES, "v4+"),
    "events": (KIND_EVENTS, None),
    "fonts": (KIND_FONTS, None),
    "graphics": (KIND_GRAPHICS, None),
    "aegisub extradata": (KIND_EXTRADATA, None),
    "aegisub project garbage": (KIND_INFO, None),
}

STYLE_FORMAT_V4P = [
    "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour",
    "BackColour", "Bold", "Italic", "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing",
    "Angle", "BorderStyle", "Outline", "Shadow", "Alignment", "MarginL", "MarginR",
    "MarginV", "Encoding",
]
STYLE_FORMAT_V4 = [
    "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "TertiaryColour",
    "BackColour", "Bold", "Italic", "BorderStyle", "Outline", "Shadow", "Alignment",
    "MarginL", "MarginR", "MarginV", "AlphaLevel", "Encoding",
]
EVENT_FORMAT_SSA = [
    "Marked", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text",
]
EVENT_FORMAT_ASS = [
    "Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text",
]

STYLE_ALIASES = {
    "primarycolour": "PrimaryColour",
    "secondarycolour": "SecondaryColour",
    "tertiarycolour": "TertiaryColour",
    "outlinecolour": "OutlineColour",
    "backcolour": "BackColour",
    "alphalevel": "AlphaLevel",
    "strikeout": "StrikeOut",
    "fontname": "Fontname",
    "fontsize": "Fontsize",
    "underline": "Underline",
    "bordstyle": "BorderStyle",
}


def canonical_style_key(name: str) -> str:
    stripped = (name or "").strip()
    return STYLE_ALIASES.get(stripped.lower().replace(" ", ""), stripped)


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


class Entry:
    """A single physical line inside a section."""

    __slots__ = ()

    def render(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def dirty(self) -> bool:
        return False


class RawEntry(Entry):
    """A line we keep verbatim (comments, attachments, unknown keys...)."""

    __slots__ = ("raw",)

    def __init__(self, raw: str = ""):
        self.raw = raw

    def render(self) -> str:
        return self.raw

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"RawEntry({self.raw[:60]!r})"


class DataEntry(Entry):
    """``Key: v1,v2,...`` line (Style, Dialogue, Comment, Format, ...).

    Unmodified fields are emitted exactly as they appeared in the source; only
    fields touched through :meth:`set` are re-rendered.
    """

    __slots__ = ("lead", "kind", "order", "raw_fields", "_over", "raw", "malformed")

    def __init__(
        self,
        lead: str,
        kind: str,
        order: Sequence[str],
        raw_fields: Sequence[str],
        raw: str = "",
        malformed: bool = False,
    ):
        self.lead = lead  # e.g. "Dialogue:" exactly as written
        self.kind = kind  # "Dialogue" / "Comment" / "Style" / "Format"
        self.order = list(order)
        self.raw_fields = list(raw_fields)
        self._over: dict[str, str] = {}
        self.raw = raw
        self.malformed = malformed

    # -- reading ------------------------------------------------------------
    def get(self, name: str, default: str = "") -> str:
        key = canonical_style_key(name)
        if key in self._over:
            return self._over[key]
        idx = self._index(key)
        if idx is None or idx >= len(self.raw_fields):
            return default
        value = self.raw_fields[idx]
        # Text is meaningful verbatim (leading spaces can be intentional);
        # every other field is whitespace padded in the wild.
        if key == "Text":
            return value
        return value.strip()

    def has(self, name: str) -> bool:
        return self._index(canonical_style_key(name)) is not None

    def _index(self, name: str) -> int | None:
        key = canonical_style_key(name)
        for i, field in enumerate(self.order):
            if canonical_style_key(field) == key:
                return i
        return None

    # -- writing ------------------------------------------------------------
    def set(self, name: str, value) -> None:
        key = canonical_style_key(name)
        if self._index(key) is None:
            self.order.append(key)
        self._over[key] = "" if value is None else str(value)

    def set_many(self, values: dict) -> None:
        for k, v in values.items():
            self.set(k, v)

    def unset(self, name: str) -> bool:
        key = canonical_style_key(name)
        idx = self._index(key)
        if idx is None:
            return False
        if key in self._over:
            del self._over[key]
        return True

    @property
    def dirty(self) -> bool:
        return bool(self._over) or self.malformed

    @property
    def is_comment(self) -> bool:
        return self.kind.lower().startswith("comment")

    def fields_dict(self) -> dict[str, str]:
        return {name: self.get(name) for name in self.order}

    # -- output -------------------------------------------------------------
    def render(self) -> str:
        if not self._over and self.raw:
            # untouched (including malformed lines): reproduce the source exactly
            return self.raw
        parts = []
        for i, name in enumerate(self.order):
            key = canonical_style_key(name)
            if key in self._over:
                parts.append(self._over[key])
            elif i < len(self.raw_fields):
                parts.append(self.raw_fields[i])
            else:
                parts.append("")
        return f"{self.lead}{','.join(parts)}"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"DataEntry({self.kind}:{self.get('Name') or self.get('Text')[:30]!r})"


class EventEntry(DataEntry):
    """Dialogue / Comment line with typed accessors."""

    __slots__ = ()

    @property
    def uid(self) -> str:
        return f"L{id(self):x}"

    @property
    def start_ms(self) -> int:
        return assutil.parse_time(self.get("Start", "0:00:00.00"))

    @property
    def end_ms(self) -> int:
        return assutil.parse_time(self.get("End", "0:00:00.00"))

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def text(self) -> str:
        return self.get("Text", "")

    def to_dict(self, *, full: bool = True) -> dict:
        data = {
            "kind": self.kind,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "start": self.get("Start"),
            "end": self.get("End"),
            "style": self.get("Style"),
            "actor": self.get("Name"),
            "effect": self.get("Effect"),
            "text": self.text,
            "comment": self.is_comment,
        }
        if full:
            data["layer"] = self.get("Layer", self.get("Marked", "0"))
            data["marked"] = self.get("Marked", self.get("Layer", "0"))
            data["margin_l"] = self.get("MarginL")
            data["margin_r"] = self.get("MarginR")
            data["margin_v"] = self.get("MarginV")
            data["malformed"] = self.malformed
        return data


class StyleEntry(DataEntry):
    __slots__ = ()

    @property
    def name(self) -> str:
        return self.get("Name")

    def to_dict(self) -> dict:
        return self.fields_dict()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class Section:
    __slots__ = ("kind", "header", "entries", "style_format", "version", "_name")

    def __init__(self, kind: str, header: str, entries: list[Entry] | None = None,
                 style_format: str | None = None, version: str | None = None,
                 name: str | None = None):
        self.kind = kind
        self.header = header  # "[Events]" exactly as written
        self.entries: list[Entry] = list(entries or [])
        self.style_format = style_format  # "v4+" / "v4" for style sections
        self.version = version
        self._name = name if name is not None else header.strip("[]").strip().lower()

    @property
    def name(self) -> str:
        """Lower-cased section name without brackets (e.g. ``v4+ styles``)."""
        return self._name


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

_NEWLINE_RE = re.compile(r"\r\n|\r|\n")


class AssDocument:
    """An editable ASS/SSA file."""

    def __init__(self, text: str = "", *, path: str | None = None,
                 encoding: str = "utf-8", newline: str = "\n",
                 trailing_newline: bool = True, has_bom: bool = False):
        self.path = path
        self.encoding = encoding
        self.newline = newline
        self.trailing_newline = trailing_newline
        self.has_bom = has_bom
        self.preamble: list[RawEntry] = []
        self.sections: list[Section] = []
        self.dirty = False
        self.opened_at = time.time()
        self._rev = 0
        if text:
            self._parse(text)

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "AssDocument":
        with open(path, "rb") as fh:
            data = fh.read()
        encoding, has_bom = _sniff_encoding(data)
        text = data.decode(encoding, errors="surrogateescape")
        if has_bom and text.startswith("\ufeff"):
            text = text[1:]
        crlf = text.count("\r\n")
        lf_only = text.count("\n") - crlf
        cr_only = text.count("\r") - crlf
        if crlf and crlf >= lf_only and crlf >= cr_only:
            newline = "\r\n"
        elif cr_only and cr_only > lf_only:
            newline = "\r"
        else:
            newline = "\n"
        trailing = text.endswith(("\n", "\r"))
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if trailing:
            normalized = normalized[:-1]
        return cls(normalized, path=os.path.abspath(path), encoding=encoding,
                   newline=newline, trailing_newline=trailing, has_bom=has_bom)

    @classmethod
    def from_text(cls, text: str, *, path: str | None = None, **kw) -> "AssDocument":
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        trailing = normalized.endswith("\n")
        if trailing:
            normalized = normalized[:-1]
        kw.setdefault("trailing_newline", trailing)
        return cls(normalized, path=path, **kw)

    @classmethod
    def from_bytes(cls, data: bytes, *, path: str | None = None) -> "AssDocument":
        """Load from raw bytes using the same encoding sniffing as :meth:`load`."""
        encoding, has_bom = _sniff_encoding(data)
        text = data.decode(encoding, errors="surrogateescape")
        if has_bom and text.startswith("\ufeff"):
            text = text[1:]
        crlf = text.count("\r\n")
        lf_only = text.count("\n") - crlf
        cr_only = text.count("\r") - crlf
        if crlf and crlf >= lf_only and crlf >= cr_only:
            newline = "\r\n"
        elif cr_only and cr_only > lf_only:
            newline = "\r"
        else:
            newline = "\n"
        trailing = text.endswith(("\n", "\r"))
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if trailing:
            normalized = normalized[:-1]
        return cls(normalized, path=path, encoding=encoding, newline=newline,
                   trailing_newline=trailing, has_bom=has_bom)

    @classmethod
    def new(cls, *, path: str | None = None, play_res: tuple[int, int] = (1920, 1080),
            title: str = "Untitled", styles: Sequence[dict] | None = None,
            script_type: str = "v4.00+") -> "AssDocument":
        doc = cls("", path=path)
        info = doc.ensure_section(KIND_INFO, header="[Script Info]")
        entries = [
            ("Title", title),
            ("ScriptType", script_type),
            ("WrapStyle", "0"),
            ("ScaledBorderAndShadow", "yes"),
            ("YCbCr Matrix", "TV.709"),
            ("PlayResX", str(play_res[0])),
            ("PlayResY", str(play_res[1])),
        ]
        for key, value in entries:
            info.entries.append(RawEntry(f"{key}: {value}"))
        styles_sec = doc.ensure_section(
            KIND_STYLES,
            header="[V4+ Styles]" if script_type.endswith("+") else "[V4 Styles]",
            style_format="v4+" if script_type.endswith("+") else "v4",
        )
        fmt = STYLE_FORMAT_V4P if script_type.endswith("+") else STYLE_FORMAT_V4
        styles_sec.entries.append(RawEntry("Format: " + ", ".join(fmt)))
        ev = doc.ensure_section(KIND_EVENTS, header="[Events]")
        # SSA (v4.00) spells the first event field "Marked", ASS spells it "Layer",
        # so a new document must declare the format of the script type it was asked
        # for.  Stored as a DataEntry — the shape the parser builds — because
        # format_order() reads the event format from it: a raw line would be invisible
        # and the lines added later would silently fall back to the ASS default.
        event_fmt = EVENT_FORMAT_ASS if script_type.endswith("+") else EVENT_FORMAT_SSA
        event_rest = " " + ", ".join(event_fmt)
        ev.entries.append(DataEntry("Format:", "Format", event_fmt, [event_rest],
                                    raw=f"Format:{event_rest}"))
        for style in styles or [default_style_spec()]:
            doc.add_style(style, touch=False)
        doc.dirty = False
        return doc

    # -- parsing ------------------------------------------------------------
    def _parse(self, text: str) -> None:
        lines = text.split("\n")
        current: Section | None = None
        current_format: list[str] | None = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]") and len(stripped) > 2:
                current = self._begin_section(stripped)
                current_format = None
                continue
            if current is None:
                self.preamble.append(RawEntry(line))
                continue
            if not stripped:
                current.entries.append(RawEntry(line))
                continue
            key, sep, rest = line.partition(":")
            head = key.strip()
            if not sep:
                current.entries.append(RawEntry(line))
                continue
            upper = head.lower()
            if upper == "format":
                fields = [f.strip() for f in rest.split(",")]
                current_format = fields
                current.entries.append(DataEntry(f"{head}:", head, fields, [rest], raw=line))
                continue
            if current.kind == KIND_STYLES and upper == "style":
                order = current_format or (STYLE_FORMAT_V4P if current.style_format == "v4+" else STYLE_FORMAT_V4)
                fields = rest.split(",", len(order) - 1) if order else [rest]
                entry = StyleEntry(f"{head}:", head, order, fields, raw=line,
                                   malformed=bool(order) and len(fields) != len(order))
                current.entries.append(entry)
                continue
            if current.kind == KIND_EVENTS:
                if upper in ("dialogue", "comment", "command", "picture", "sound", "movie", "text", "blank"):
                    order = current_format or (EVENT_FORMAT_ASS if head.lower() == "dialogue" else EVENT_FORMAT_SSA)
                    fields = rest.split(",", len(order) - 1) if order else [rest]
                    entry = EventEntry(f"{head}:", head, order, fields, raw=line,
                                       malformed=bool(order) and len(fields) != len(order))
                    current.entries.append(entry)
                    continue
            # anything else (Script Info keys, comments, attachment data...) stays raw
            current.entries.append(RawEntry(line))
        # document flags
        self.dirty = False

    def _begin_section(self, header: str) -> Section:
        name = header.strip("[]").strip().lower()
        kind, version = _SECTION_ALIASES.get(name, (KIND_OTHER, None))
        style_format = None
        if kind == KIND_STYLES:
            style_format = version or ("v4+" if "+" in name else "v4")
        section = Section(kind, header, style_format=style_format, version=version)
        section._name = name  # type: ignore[attr-defined]
        self.sections.append(section)
        return section

    # -- serialising --------------------------------------------------------
    def to_text(self) -> str:
        out: list[str] = []
        for entry in self.preamble:
            out.append(entry.render())
        for section in self.sections:
            out.append(section.header)
            for entry in section.entries:
                out.append(entry.render())
        text = "\n".join(out)
        if self.trailing_newline:
            text += "\n"
        return text

    def to_bytes(self, encoding: str | None = None) -> bytes:
        enc = encoding or self.encoding or "utf-8"
        text = self.to_text()
        if self.has_bom and not text.startswith("\ufeff") and enc.replace("-sig", "") == "utf-8":
            text = "\ufeff" + text
        if self.newline != "\n":
            text = text.replace("\n", self.newline)
        return text.encode(enc, errors="surrogateescape")

    def save(self, path: str | None = None, *, encoding: str | None = None,
             newline: str | None = None, create_backup: bool = False) -> str:
        target = path or self.path
        if not target:
            raise ValueError("no path given and document has no path")
        target = os.path.abspath(target)
        if create_backup and os.path.exists(target):
            backup = f"{target}.bak"
            with open(target, "rb") as src, open(backup, "wb") as dst:
                dst.write(src.read())
        if newline:
            self.newline = newline
        data = self.to_bytes(encoding)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(data)
        if path:
            self.path = target
        self.dirty = False
        self._rev += 1
        return target

    def snapshot(self) -> str:
        return self.to_text()

    def restore(self, text: str) -> None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        trailing = normalized.endswith("\n")
        if trailing:
            normalized = normalized[:-1]
        self.trailing_newline = trailing
        self.preamble = []
        self.sections = []
        self._parse(normalized)
        self.dirty = True

    def clone(self) -> "AssDocument":
        return copy.deepcopy(self)

    # -- structure helpers --------------------------------------------------
    def section(self, kind: str, *, header: str | None = None) -> Section | None:
        for sec in self.sections:
            if sec.kind == kind and (header is None or sec.header == header):
                return sec
        return None

    def sections_of(self, kind: str) -> list[Section]:
        return [s for s in self.sections if s.kind == kind]

    def ensure_section(self, kind: str, *, header: str | None = None,
                       style_format: str | None = None) -> Section:
        sec = self.section(kind, header=header)
        if sec is None:
            header = header or {
                KIND_INFO: "[Script Info]",
                KIND_STYLES: "[V4+ Styles]",
                KIND_EVENTS: "[Events]",
                KIND_FONTS: "[Fonts]",
                KIND_GRAPHICS: "[Graphics]",
                KIND_EXTRADATA: "[Aegisub Extradata]",
            }.get(kind, f"[{kind}]")
            sec = Section(kind, header, style_format=style_format)
            # keep canonical order: info, styles, events, then the rest
            order = {KIND_INFO: 0, KIND_STYLES: 1, KIND_EVENTS: 2}
            pos = len(self.sections)
            for i, existing in enumerate(self.sections):
                if order.get(existing.kind, 3) > order.get(kind, 3):
                    pos = i
                    break
            self.sections.insert(pos, sec)
            self.dirty = True
        return sec

    # -- Script Info --------------------------------------------------------
    def _info_section(self, create: bool = False) -> Section | None:
        sec = self.section(KIND_INFO, header="[Script Info]") or self.section(KIND_INFO)
        if sec is None and create:
            sec = self.ensure_section(KIND_INFO, header="[Script Info]")
        return sec

    def info_items(self) -> list[tuple[int, str, str]]:
        sec = self._info_section()
        items: list[tuple[int, str, str]] = []
        if not sec:
            return items
        for idx, entry in enumerate(sec.entries):
            if not isinstance(entry, RawEntry):
                continue
            text = entry.raw
            stripped = text.strip()
            if not stripped or stripped.startswith((";", "!")):
                continue
            key, sep, value = text.partition(":")
            if not sep:
                continue
            items.append((idx, key.strip(), value.strip()))
        return items

    def info_get(self, key: str, default: str | None = None) -> str | None:
        target = key.lower().replace(" ", "")
        for _, k, value in self.info_items():
            if k.lower().replace(" ", "") == target:
                return value
        return default

    def info_all(self) -> dict[str, str]:
        return {k: v for _, k, v in self.info_items()}

    def info_set(self, key: str, value) -> bool:
        """Set (or append) a Script Info key. Returns True if an existing key was updated."""
        sec = self.ensure_section(KIND_INFO, header="[Script Info]")
        target = key.lower().replace(" ", "")
        for idx, k, _ in self.info_items():
            if k.lower().replace(" ", "") == target:
                entry = sec.entries[idx]
                if isinstance(entry, RawEntry):
                    entry.raw = f"{k}: {value}"
                self.dirty = True
                return True
        sec.entries.append(RawEntry(f"{key}: {value}"))
        self.dirty = True
        return False

    def info_remove(self, key: str) -> bool:
        sec = self._info_section()
        if not sec:
            return False
        target = key.lower().replace(" ", "")
        for idx, k, _ in self.info_items():
            if k.lower().replace(" ", "") == target:
                del sec.entries[idx]
                self.dirty = True
                return True
        return False

    # -- styles -------------------------------------------------------------
    def style_section(self, *, create: bool = False) -> Section | None:
        sec = self.section(KIND_STYLES)
        if sec is None and create:
            sec = self.ensure_section(KIND_STYLES, header="[V4+ Styles]", style_format="v4+")
        return sec

    def styles(self) -> list[StyleEntry]:
        out: list[StyleEntry] = []
        for sec in self.sections_of(KIND_STYLES):
            for entry in sec.entries:
                if isinstance(entry, StyleEntry):
                    out.append(entry)
        return out

    def style_names(self) -> list[str]:
        return [s.name for s in self.styles()]

    def get_style(self, name: str) -> StyleEntry | None:
        target = (name or "").strip().lower()
        for style in self.styles():
            if style.name.lower() == target:
                return style
        return None

    def add_style(self, spec: dict, *, touch: bool = True) -> StyleEntry:
        sec = self.style_section(create=True)
        assert sec is not None
        order = None
        if sec.entries and isinstance(sec.entries[-1], DataEntry):
            order = sec.entries[-1].order
        if order is None:
            order = STYLE_FORMAT_V4P if (sec.style_format or "v4+") == "v4+" else STYLE_FORMAT_V4
        fields = {}
        for i, name in enumerate(order):
            fields[name] = ""
        entry = StyleEntry("Style:", "Style", order, ["" for _ in order])
        for key, value in spec.items():
            key = canonical_style_key(key)
            if key not in order:
                order.append(key)
                entry.raw_fields.append("")
                entry.order = order
                fields[key] = ""
            entry.set(key, value)
        sec.entries.append(entry)
        if touch:
            self.dirty = True
        return entry

    def remove_style(self, name: str) -> bool:
        for sec in self.sections_of(KIND_STYLES):
            for idx, entry in enumerate(sec.entries):
                if isinstance(entry, StyleEntry) and entry.name.lower() == name.strip().lower():
                    del sec.entries[idx]
                    self.dirty = True
                    return True
        return False

    # -- events -------------------------------------------------------------
    def event_sections(self, *, create: bool = False) -> list[Section]:
        secs = self.sections_of(KIND_EVENTS)
        if not secs and create:
            secs = [self.ensure_section(KIND_EVENTS, header="[Events]")]
        return secs

    def events(self, *, comments: bool = True) -> list[EventEntry]:
        out: list[EventEntry] = []
        for sec in self.event_sections():
            for entry in sec.entries:
                if isinstance(entry, EventEntry):
                    if entry.is_comment and not comments:
                        continue
                    out.append(entry)
        return out

    def event_index_map(self) -> list[EventEntry]:
        """All event entries in file order, comments included."""
        return self.events(comments=True)

    def format_order(self) -> list[str]:
        for sec in self.event_sections():
            for entry in sec.entries:
                if isinstance(entry, DataEntry) and entry.kind.lower() == "format":
                    return list(entry.order)
        return list(EVENT_FORMAT_ASS)

    def add_event(self, **fields) -> EventEntry:
        secs = self.event_sections(create=True)
        sec = secs[0]
        order = self.format_order()
        entry = EventEntry("Dialogue:", "Dialogue", order, ["" for _ in order])
        start = fields.pop("start_ms", None)
        if start is None:
            start = fields.pop("start", None)
        end = fields.pop("end_ms", None)
        if end is None:
            end = fields.pop("end", None)
        text_value = fields.pop("text", None)
        if isinstance(start, (int, float)):
            fields["Start"] = assutil.format_time(int(start))
        elif isinstance(start, str):
            fields["Start"] = start
        if isinstance(end, (int, float)):
            fields["End"] = assutil.format_time(int(end))
        elif isinstance(end, str):
            fields["End"] = end
        if text_value is not None:
            fields["Text"] = text_value
        # ASS spells the first event field "Layer", SSA spells it "Marked" and stores
        # its value as "Marked=N" — follow this document's own Format line.
        annotation = "Marked" if "Marked" in order else "Layer"
        defaults = {
            "Layer": 0, "Marked": "Marked=0", "Start": "0:00:00.00", "End": "0:00:00.00",
            "Style": "Default", "Name": "", "MarginL": 0, "MarginR": 0, "MarginV": 0,
            "Effect": "", "Text": "",
        }
        for key in order:
            if key in fields and fields[key] is not None:
                entry.set(key, fields[key])
            elif key in defaults:
                entry.set(key, defaults[key])
        kind = fields.get("kind") or fields.get("Kind") or "Dialogue"
        entry.kind = kind
        entry.lead = f"{kind}:"
        if kind.lower().startswith("comment"):
            entry.set(annotation, fields.get(annotation, defaults[annotation]))
        sec.entries.append(entry)
        self.dirty = True
        return entry

    def remove_events(self, indices: Iterable[int]) -> int:
        entries = self.events()
        doomed = {id(entries[i]) for i in indices if 0 <= i < len(entries)}
        removed = 0
        for sec in self.event_sections():
            keep: list[Entry] = []
            for entry in sec.entries:
                if isinstance(entry, EventEntry) and id(entry) in doomed:
                    removed += 1
                    continue
                keep.append(entry)
            sec.entries = keep
        if removed:
            self.dirty = True
        return removed

    def sorted_events(self, key="start", reverse: bool = False) -> list[EventEntry]:
        evs = self.events()
        if key == "start":
            evs.sort(key=lambda e: (e.start_ms, e.end_ms))
        elif key == "end":
            evs.sort(key=lambda e: (e.end_ms, e.start_ms))
        elif key == "style":
            evs.sort(key=lambda e: e.get("Style").lower())
        elif key == "text":
            evs.sort(key=lambda e: e.text.lower())
        elif key == "layer":
            evs.sort(key=lambda e: _int_or_zero(e.get("Layer")))
        if reverse:
            evs.reverse()
        return evs

    def reorder_events(self, ordered: Sequence[EventEntry]) -> None:
        """Reorder the event entries to match ``ordered`` (comments kept in place)."""
        pos = {id(e): i for i, e in enumerate(ordered)}
        for sec in self.event_sections():
            slots = [i for i, e in enumerate(sec.entries) if isinstance(e, EventEntry)]
            items = [sec.entries[i] for i in slots]
            items.sort(key=lambda e: pos.get(id(e), 10**9))
            for slot, item in zip(slots, items):
                sec.entries[slot] = item
        self.dirty = True

    # -- attachments --------------------------------------------------------
    def attachments(self, kind: str) -> list[str]:
        sec = self.section(kind)
        if not sec:
            return []
        return [e.render() for e in sec.entries if isinstance(e, RawEntry) and e.render().strip()]

    def extradata(self) -> list[dict]:
        """Aegisub Extradata entries (e.g. ``Comment: 0,<id>,key,value``)."""
        sec = self.section(KIND_EXTRADATA)
        out = []
        if not sec:
            return out
        for entry in sec.entries:
            if not isinstance(entry, DataEntry):
                continue
            parts = entry.raw.split(",", 3) if entry.raw else []
            if len(parts) >= 3:
                head, index, ident = parts[0], parts[1], parts[2]
                value = parts[3] if len(parts) > 3 else ""
                out.append({
                    "raw": entry.raw,
                    "section": head.split(":")[0].strip() or "Comment",
                    "index": index.strip(),
                    "id": ident.strip(),
                    "value": value,
                })
        return out

    def set_extradata(self, ident: str, value: str, *, index: int = 0,
                      is_comment: bool = True) -> None:
        sec = self.ensure_section(KIND_EXTRADATA, header="[Aegisub Extradata]")
        head = "Comment" if is_comment else "Dialogue"
        new_raw = f"{head}: {index},{ident},{value}"
        target = ident.strip().lower()
        for entry in sec.entries:
            if isinstance(entry, DataEntry):
                parts = entry.raw.split(",", 3)
                if len(parts) >= 3 and parts[2].strip().lower() == target:
                    entry.raw = new_raw
                    entry._over.clear()
                    self.dirty = True
                    return
        sec.entries.append(RawEntry(new_raw))
        self.dirty = True

    # -- convenience --------------------------------------------------------
    @property
    def play_res(self) -> tuple[int, int]:
        x = _int_or_zero(self.info_get("PlayResX", "") or "")
        y = _int_or_zero(self.info_get("PlayResY", "") or "")
        if x <= 0:
            x = 384
        if y <= 0:
            y = 288
        return (x, y)

    def set_play_res(self, x: int, y: int) -> None:
        self.info_set("PlayResX", int(x))
        self.info_set("PlayResY", int(y))

    @property
    def script_type(self) -> str:
        return self.info_get("ScriptType", "v4.00+") or "v4.00+"

    @property
    def wrapping(self) -> int:
        return _int_or_zero(self.info_get("WrapStyle", "0") or "0")

    def fps(self, default: float = 23.976) -> float:
        value = self.info_get("FPS", None)
        if value:
            try:
                return float(value)
            except ValueError:
                pass
        return default

    def stats(self) -> dict:
        evs = self.events()
        dialogues = [e for e in evs if not e.is_comment]
        styles = self.styles()
        total_ms = sum(max(0, e.duration_ms) for e in dialogues)
        chars = sum(len(assutil.text_to_plain(e.text)) for e in dialogues)
        x, y = self.play_res
        return {
            "path": self.path,
            "encoding": self.encoding,
            "newline": {"\n": "LF", "\r\n": "CRLF", "\r": "CR"}.get(self.newline, "LF"),
            "bom": self.has_bom,
            "script_type": self.script_type,
            "title": self.info_get("Title", "") or "",
            "play_res": [x, y],
            "sections": [
                {"header": s.header, "kind": s.kind, "entries": len(s.entries)} for s in self.sections
            ],
            "styles": len(styles),
            "style_names": [s.name for s in styles],
            "events_total": len(evs),
            "dialogue": len(dialogues),
            "comment": len(evs) - len(dialogues),
            "total_duration_ms": total_ms,
            "total_chars": chars,
            "attachments": {
                "fonts": len(self.attachments(KIND_FONTS)),
                "graphics": len(self.attachments(KIND_GRAPHICS)),
            },
            "extradata": len(self.extradata()),
            "dirty": self.dirty,
        }

    # -- whole-file line access (Automation 4 "subtitles" semantics) --------
    #
    # Aegisub's Lua API exposes *every* physical line of the file through one
    # flat, 1-based array, tagged with a `class` of "info" | "style" |
    # "dialogue" | "unknown".  These helpers give the Lua engine the same view
    # while leaving the typed accessors above untouched.

    def line_containers(self) -> list[tuple[Section | None, list[Entry]]]:
        """Containers that hold physical lines, in file order."""
        preamble = typing.cast("list[Entry]", self.preamble)
        out: list[tuple[Section | None, list[Entry]]] = [(None, preamble)]
        out.extend((section, section.entries) for section in self.sections)
        return out

    def all_lines(self) -> list[tuple[Section | None, Entry]]:
        """``[(section_or_None, entry), ...]`` for every physical line."""
        out: list[tuple[Section | None, Entry]] = []
        for section, entries in self.line_containers():
            out.extend((section, entry) for entry in entries)
        return out

    def line_total(self) -> int:
        return sum(len(entries) for _section, entries in self.line_containers())

    def line_index_of(self, entry: Entry) -> int:
        """0-based global index of ``entry`` (``-1`` when unknown)."""
        position = 0
        for _section, entries in self.line_containers():
            for item in entries:
                if item is entry:
                    return position
                position += 1
        return -1

    def _locate_line(self, index: int) -> tuple[list[Entry], int]:
        total = self.line_total()
        if index < 0:
            index += total
        if index < 0 or index >= total:
            raise IndexError(f"line index out of range (0..{total - 1})")
        remaining = index
        for _section, entries in self.line_containers():
            if remaining < len(entries):
                return entries, remaining
            remaining -= len(entries)
        raise IndexError(f"line index out of range (0..{total - 1})")  # pragma: no cover

    def get_line(self, index: int) -> Entry:
        """Physical line at 0-based global ``index``."""
        entries, offset = self._locate_line(index)
        return entries[offset]

    def insert_lines(self, index: int, lines: Sequence[Entry | str]) -> int:
        """Insert lines before 0-based global ``index``; returns how many."""
        total = self.line_total()
        if index < 0:
            index = max(total + index, 0)
        if index >= total:
            containers = self.line_containers()
            chosen = containers[-1][1] if containers else self.preamble
            target = typing.cast("list[Entry]", chosen)
            offset = len(target)
        else:
            target, offset = self._locate_line(index)
        payload = [RawEntry(line) if isinstance(line, str) else line for line in lines]
        target[offset:offset] = payload
        self.mark_dirty()
        return len(payload)

    def replace_line(self, index: int, value: Entry | str) -> None:
        """Replace the physical line at 0-based global ``index``."""
        entries, offset = self._locate_line(index)
        entries[offset] = RawEntry(value) if isinstance(value, str) else value
        self.mark_dirty()

    def remove_lines(self, indices: Sequence[int]) -> int:
        """Delete physical lines by 0-based global index (any order)."""
        total = self.line_total()
        normalized = sorted({i + total if i < 0 else i for i in indices}, reverse=True)
        removed = 0
        for index in normalized:
            entries, offset = self._locate_line(index)
            del entries[offset]
            removed += 1
        if removed:
            self.mark_dirty()
        return removed

    def mark_dirty(self) -> None:
        """Flag the document as modified (re-rendering touched lines only)."""
        self.dirty = True
        self._rev = getattr(self, "_rev", 0) + 1

    def script_info_lines(self) -> list[tuple[int, str, str]]:
        """``[(line_index, key, value), ...]`` for info-looking lines."""
        out: list[tuple[int, str, str]] = []
        for position, (section, entry) in enumerate(self.all_lines()):
            if section is not None and section.kind != KIND_INFO:
                continue
            if not isinstance(entry, RawEntry):
                continue
            key, sep, value = entry.raw.partition(":")
            key = key.strip()
            if not sep or not key or key.startswith((";", "!")):
                continue
            out.append((position, key, value.strip()))
        return out


def _int_or_zero(value) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def default_style_spec(name: str = "Default", **overrides) -> dict:
    spec = {
        "Name": name,
        "Fontname": "Arial",
        "Fontsize": 48,
        "PrimaryColour": "&H00FFFFFF&",
        "SecondaryColour": "&H000000FF&",
        "OutlineColour": "&H00000000&",
        "BackColour": "&H80000000&",
        "Bold": 0,
        "Italic": 0,
        "Underline": 0,
        "StrikeOut": 0,
        "ScaleX": 100,
        "ScaleY": 100,
        "Spacing": 0,
        "Angle": 0,
        "BorderStyle": 1,
        "Outline": 2,
        "Shadow": 1,
        "Alignment": 2,
        "MarginL": 20,
        "MarginR": 20,
        "MarginV": 20,
        "Encoding": 1,
    }
    spec.update(overrides)
    return spec


def _sniff_encoding(data: bytes) -> tuple[str, bool]:
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8", True
    if data.startswith(b"\xff\xfe") and not data.startswith(b"\xff\xfe\x00\x00"):
        return "utf-16-le", True
    if data.startswith(b"\xfe\xff"):
        return "utf-16-be", True
    try:
        data.decode("utf-8")
        return "utf-8", False
    except UnicodeDecodeError:
        pass
    try:
        data.decode("cp1252")
        return "cp1252", False
    except UnicodeDecodeError:  # pragma: no cover - cp1252 accepts almost anything
        return "latin-1", False


def read_text_with_encoding(path: str, encoding: str | None = None) -> str:
    with open(path, "rb") as fh:
        data = fh.read()
    enc = encoding or _sniff_encoding(data)[0]
    return data.decode(enc, errors="surrogateescape")
