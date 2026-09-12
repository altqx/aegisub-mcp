"""Styles, Script Info, attachments, extradata and structural validation tools.

This module owns everything about the *non-line* parts of an ASS/SSA file:

* the style library (``[V4+ Styles]`` / ``[V4 Styles]``) — listing, inspecting,
  creating, updating, renaming, copying, reordering, deleting, usage counts,
  per-line effective values and font-substitution diagnostics;
* the ``[Script Info]`` header, in file order, plus play resolution, wrapping,
  scaled-border-and-shadow and timing info;
* ``[Fonts]`` / ``[Graphics]`` attachments (list / add / extract / remove);
* the ``[Aegisub Extradata]`` section;
* a structural validator used by QC and by the agent before it trusts a file.

Ground rules that shape the code below
--------------------------------------
* Style field values are kept as **exact strings** as stored in the document.
  Colour fields in particular are never reformatted: ``basic.ass`` stores
  ``&H00FFFFFF`` (no trailing ``&``) and ``basic.ssa`` stores ``&H00FFFFFF&``;
  both round-trip verbatim.  When a caller passes an HTML colour (``#RRGGBB``)
  it is converted into the spelling the *document* already uses for that field.
* Field names are accepted in any spelling: the ASS column names
  (``Fontname``, ``Fontsize``, ``PrimaryColour``, ``MarginV``), snake_case
  (``font``, ``font_size``, ``primary_colour``, ``margin_v``) and
  case/space variants, all resolve to the same canonical field.
* Every mutating tool calls ``workspace.snapshot(doc_id)`` first, so
  ``ass_undo`` reverses exactly one tool call.
* A field that the document's ``Format`` line does not declare (for example
  ``ScaleX`` in a ``[V4 Styles]`` file, or ``RelativeTo`` everywhere in stock
  ASS) is reported in ``ignored_fields`` instead of being appended as an extra
  comma — appending would desynchronise the ``Format`` line from the
  ``Style`` line.

Known limitation inherited from :mod:`aegisub_mcp.asscore.document` (not
modified here): ``AssDocument.extradata()`` finds nothing (extradata lines are
kept as ``RawEntry``) and ``AssDocument.set_extradata()`` therefore appends a
duplicate on every call.  This module parses and edits the ``[Aegisub
Extradata]`` section itself; see ``_extradata_records()``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

from ..asscore import document as D
from ..asscore import measure as M
from ..asscore import tags as T
from .base import ToolError, ok, workspace

__all__ = [
    "ass_list_styles",
    "ass_get_style",
    "ass_add_style",
    "ass_update_style",
    "ass_remove_style",
    "ass_rename_style",
    "ass_copy_style",
    "ass_reorder_styles",
    "ass_style_usage",
    "ass_style_for_line",
    "ass_check_font_substitution",
    "ass_get_script_info",
    "ass_set_script_info",
    "ass_remove_script_info",
    "ass_set_play_res",
    "ass_set_wrap_style",
    "ass_set_scaled_border_and_shadow",
    "ass_set_timing_info",
    "ass_list_attachments",
    "ass_add_attachment",
    "ass_extract_attachment",
    "ass_remove_attachment",
    "ass_list_extradata",
    "ass_set_extradata",
    "ass_validate",
    "register",
]

# --------------------------------------------------------------------------
# Style field vocabulary
# --------------------------------------------------------------------------

STYLE_FIELDS_V4P: list[str] = list(D.STYLE_FORMAT_V4P)
STYLE_FIELDS_V4: list[str] = list(D.STYLE_FORMAT_V4)
#: Every canonical style field, V4+ first.
STYLE_FIELDS: list[str] = STYLE_FIELDS_V4P + [
    f for f in STYLE_FIELDS_V4 if f not in STYLE_FIELDS_V4P
]
#: Aegisub's style-manager-only extension; only written when the document's
#: Format line already declares the column.
EXTRA_FIELD = "RelativeTo"

COLOUR_FIELDS = {
    "PrimaryColour", "SecondaryColour", "TertiaryColour", "OutlineColour", "BackColour",
}
FLAG_FIELDS = {"Bold", "Italic", "Underline", "StrikeOut"}
FN, FS, PC, SC, OC, BC = "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour", "BackColour"

#: snake_case / alternative spelling -> canonical ASS column name.
_SNAKE: dict[str, str] = {
    "name": "Name",
    "font": FN,
    "fontname": FN,
    "font_name": FN,
    "fontsize": FS,
    "font_size": FS,
    "primary_colour": PC,
    "secondary_colour": SC,
    "tertiary_colour": "TertiaryColour",
    "outline_colour": OC,
    "back_colour": BC,
    "alpha_level": "AlphaLevel",
    "bold": "Bold",
    "italic": "Italic",
    "underline": "Underline",
    "strike_out": "StrikeOut",
    "strikeout": "StrikeOut",
    "strikethrough": "StrikeOut",
    "scale_x": "ScaleX",
    "scale_y": "ScaleY",
    "scalex": "ScaleX",
    "scaley": "ScaleY",
    "spacing": "Spacing",
    "angle": "Angle",
    "border_style": "BorderStyle",
    "bord_style": "BorderStyle",
    "bordstyle": "BorderStyle",
    "outline": "Outline",
    "shadow": "Shadow",
    "alignment": "Alignment",
    "align": "Alignment",
    "margin_l": "MarginL",
    "margin_r": "MarginR",
    "margin_v": "MarginV",
    "marginl": "MarginL",
    "marginr": "MarginR",
    "marginv": "MarginV",
    "encoding": "Encoding",
    "relative_to": EXTRA_FIELD,
    "relativeto": EXTRA_FIELD,
    "relative": EXTRA_FIELD,
}

_ALIAS_TABLE: dict[str, str] = {}


def _build_alias_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for field in STYLE_FIELDS + [EXTRA_FIELD]:
        variants = {
            field.lower(),
            field.lower().replace("colour", "color"),
            re.sub(r"(?<!^)(?=[A-Z])", "_", field).lower(),
            re.sub(r"(?<!^)(?=[A-Z])", "", field).lower(),
        }
        for variant in variants:
            table[variant] = field
            table[variant.replace("color", "colour")] = field
            table[variant.replace("colour", "color")] = field
            table[variant.replace("_", "")] = field
    table.update(_SNAKE)
    for key, value in list(_SNAKE.items()):
        table[key.replace("colour", "color")] = value
    return table


_ALIAS_TABLE = _build_alias_table()


def _canon_field(key: Any) -> str:
    """Canonical ASS column name for any accepted spelling of a style field."""
    raw = str(key).strip()
    if not raw:
        raise ToolError("style field name must not be empty")
    if raw in STYLE_FIELDS or raw == EXTRA_FIELD:
        return raw
    low = raw.lower()
    for candidate in (low, low.replace(" ", ""), low.replace(" ", "_").replace("-", "_")):
        found = _ALIAS_TABLE.get(candidate)
        if found:
            return found
    raise ToolError(
        f"unknown style field {key!r}; known fields: " + ", ".join(STYLE_FIELDS + [EXTRA_FIELD])
    )


# --------------------------------------------------------------------------
# Value formatting
# --------------------------------------------------------------------------

def _num_text(value: Any) -> str:
    if isinstance(value, bool):
        return "-1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value).strip()


def _html_value(value: str) -> int:
    """``#RRGGBB`` / ``#RRGGBBAA`` -> ASS ``AABBGGRR`` integer."""
    raw = value.strip().lstrip("#")
    if len(raw) not in (6, 8):
        raise ToolError(f"invalid HTML colour {value!r}; expected #RRGGBB or #RRGGBBAA")
    try:
        parts = [int(raw[i:i + 2], 16) for i in range(0, len(raw), 2)]
    except ValueError:
        raise ToolError(f"invalid HTML colour {value!r}") from None
    if len(parts) == 3:
        r, g, b = parts
        a = 0
    else:
        r, g, b, html_a = parts
        a = 255 - html_a
    return (a << 24) | (b << 16) | (g << 8) | r


def _format_colour_value(value: int, template: str | None) -> str:
    """Render an ASS colour integer using the document's own spelling."""
    tpl = (template or "").strip()
    if tpl and re.fullmatch(r"\d+", tpl):
        return str(int(value))
    digits = 8 if len(re.sub(r"[^0-9A-Fa-f]", "", tpl)) == 8 else 6
    body = f"{int(value) & 0xFFFFFFFF:08X}" if digits == 8 else f"{int(value) & 0xFFFFFF:06X}"
    trailing = "&" if tpl.endswith("&") else ""
    prefix = "&H" if (not tpl or tpl.upper().startswith("&H")) else "&"
    return f"{prefix}{body}{trailing}"


def _colour_text(value: Any, template: str | None) -> str:
    if isinstance(value, bool):
        raise ToolError(f"invalid colour value {value!r}")
    if isinstance(value, int):
        return _format_colour_value(value, template)
    raw = str(value).strip()
    if not raw:
        return ""
    if raw.startswith("#"):
        return _format_colour_value(_html_value(raw), template)
    if re.fullmatch(r"[0-9]+", raw):
        return raw
    if raw.startswith("&"):
        return raw
    return raw


def _field_text(field: str, value: Any, template: str | None = None) -> str:
    """Render a caller value as the exact string to store in the document."""
    if value is None:
        return ""
    if field in COLOUR_FIELDS:
        return _colour_text(value, template)
    if field in FLAG_FIELDS:
        if isinstance(value, bool):
            return "-1" if value else "0"
        text = str(value).strip().lower()
        if text in {"true", "yes", "on"}:
            return "-1"
        if text in {"false", "no", "off"}:
            return "0"
        return _num_text(value)
    return _num_text(value)


def _style_number(field: str, value: Any, *, minimum: float | None = None,
                  exclusive_min: bool = False, maximum: float | None = None) -> float:
    """Coerce a caller value to a finite float or raise :class:`ToolError`."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        raise ToolError(f"{field} must be a number, got {value!r}") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ToolError(f"{field} must be a finite number, got {value!r}")
    if minimum is not None:
        if exclusive_min and number <= minimum:
            raise ToolError(f"{field} must be greater than {minimum:g}, got {value!r}")
        if not exclusive_min and number < minimum:
            raise ToolError(f"{field} must be >= {minimum:g}, got {value!r}")
    if maximum is not None and number > maximum:
        raise ToolError(f"{field} must be <= {maximum:g}, got {value!r}")
    return number


def _style_integer(field: str, value: Any, *, minimum: int | None = None,
                   maximum: int | None = None,
                   allowed: Sequence[int] | None = None) -> int:
    """Coerce a caller value to an int (or a listed choice) or raise."""
    if isinstance(value, bool):
        raise ToolError(f"{field} must be a whole number, got {value!r}")
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ToolError(f"{field} must be a whole number, got {value!r}") from None
    if allowed is not None and number not in allowed:
        options = ", ".join(str(item) for item in allowed)
        raise ToolError(f"{field} must be one of {options}, got {value!r}")
    if minimum is not None and number < minimum:
        raise ToolError(f"{field} must be >= {minimum}, got {value!r}")
    if maximum is not None and number > maximum:
        raise ToolError(f"{field} must be <= {maximum}, got {value!r}")
    return number


#: Canonical style columns whose value must be a whole number.
INT_STYLE_FIELDS = {
    "BorderStyle", "Alignment", "MarginL", "MarginR", "MarginV", "Encoding", "RelativeTo",
    "AlphaLevel",
}
#: Canonical style columns whose value may be fractional.
FLOAT_STYLE_FIELDS = {"Fontsize", "ScaleX", "ScaleY", "Spacing", "Angle", "Outline", "Shadow"}


def _validate_style_values(requested: dict[str, Any]) -> None:
    """Reject style values that would corrupt a Style line, before any write.

    ``requested`` is keyed by canonical ASS column name (see
    :func:`_canon_field`).  Numbers must parse, ``Fontsize`` must be positive,
    ``Alignment`` must be 1-9, ``BorderStyle`` must be 1 or 3 and colours must
    be a spelling the document can store (ASS hex, decimal, ``#RRGGBB``).
    """
    for field, value in requested.items():
        if value is None:
            continue
        if field == "Alignment":
            _style_integer(field, value, allowed=tuple(range(1, 10)))
        elif field == "BorderStyle":
            _style_integer(field, value, allowed=(1, 3))
        elif field == "Fontsize":
            _style_number(field, value, minimum=0.0, exclusive_min=True)
        elif field in FLOAT_STYLE_FIELDS:
            _style_number(field, value)
        elif field in INT_STYLE_FIELDS:
            _style_integer(field, value)
        elif field in {"MarginL", "MarginR", "MarginV"}:
            _style_integer(field, value, minimum=0)
        elif field in FLAG_FIELDS:
            if isinstance(value, bool):
                continue
            text = str(value).strip().lower()
            if text in {"true", "false", "yes", "no", "on", "off"}:
                continue
            _style_integer(field, value)
        elif field in COLOUR_FIELDS:
            _colour_text(value, None)


def _is_int_text(value: str | None) -> bool:
    return bool(value) and bool(re.fullmatch(r"[+-]?\d+", str(value).strip()))


# --------------------------------------------------------------------------
# Generic document helpers
# --------------------------------------------------------------------------

def _doc(doc_id: str | None) -> tuple[str, D.AssDocument]:
    did = workspace.resolve_id(doc_id)
    return did, workspace.get(did)


def _blank(raw: str) -> bool:
    text = raw.strip()
    return not text or text.startswith((";", "!"))


def _norm_key(key: str) -> str:
    return str(key).strip().lower().replace(" ", "")


def _owning_section(doc: D.AssDocument, entry: D.DataEntry) -> D.Section:
    for sec in doc.sections_of(D.KIND_STYLES):
        for item in sec.entries:
            if item is entry:
                return sec
    sec = doc.style_section(create=False)
    if sec is None:
        raise ToolError("document has no style section")
    return sec


def _format_columns(raw: str) -> list[str] | None:
    """Column names declared by a verbatim ``Format:`` line, else ``None``."""
    match = re.match(r"^\s*format\s*:\s*(.*)$", raw or "", re.IGNORECASE)
    if match is None:
        return None
    names = [part.strip() for part in match.group(1).split(",")]
    names = [name for name in names if name]
    return names or None


def _section_field_names(sec: D.Section) -> list[str]:
    """Canonical field list a Style line in ``sec`` is expected to follow.

    The section's own ``Format:`` line decides: it is what every writer and
    ``ass_validate`` compare style lines against, and it survives a style entry
    whose column list drifted (a hand edited file, or a caller that set a field
    the format does not declare).  Only when the section has no ``Format`` line
    at all do we fall back to the last structured entry's order, and finally to
    this module's pristine copy of the stock column set — never the live
    ``asscore`` constants, which an earlier call may have extended in place.
    """
    for entry in sec.entries:
        if isinstance(entry, D.DataEntry):
            if str(getattr(entry, "kind", "")).lower() == "format":
                return [D.canonical_style_key(f) for f in entry.order]
            continue
        columns = _format_columns(getattr(entry, "raw", "") or "")
        if columns:
            return [D.canonical_style_key(f) for f in columns]
    for entry in reversed(sec.entries):
        if isinstance(entry, D.DataEntry):
            return [D.canonical_style_key(f) for f in entry.order]
    default = STYLE_FIELDS_V4 if (sec.style_format or "v4+").lower() == "v4" else STYLE_FIELDS_V4P
    return [D.canonical_style_key(f) for f in default]


def _is_style_line_start(item: Any) -> bool:
    """True for entries a new ``Style`` line may follow (Format line, styles)."""
    if isinstance(item, D.StyleEntry):
        return True
    if isinstance(item, D.DataEntry):
        return str(item.kind).lower() == "format"
    if isinstance(item, D.RawEntry):
        return _format_columns(getattr(item, "raw", "") or "") is not None
    return False


def _raw_padding(entry: Any, field: str) -> tuple[str, str]:
    """Leading/trailing whitespace the field carries in the source line.

    ``AssDocument`` renders a modified line as ``lead + ",".join(fields)`` and
    keeps the parsed fields verbatim, so the spacing a file uses after the
    colon lives in the *first field* (``" Default"``).  Carrying that padding
    over to the replacement value keeps ``Style: Default`` looking exactly like
    a line Aegisub wrote instead of ``Style:Default``.
    """
    order = list(getattr(entry, "order", []) or [])
    raw_fields = list(getattr(entry, "raw_fields", []) or [])
    key = D.canonical_style_key(field)
    for index, name in enumerate(order):
        if D.canonical_style_key(name) != key or index >= len(raw_fields):
            continue
        raw = raw_fields[index]
        stripped = raw.strip()
        if not stripped:
            return "", ""
        return raw[: len(raw) - len(raw.lstrip())], raw[len(raw.rstrip()):]
    return "", ""


def _padded(value: str, padding: tuple[str, str]) -> str:
    """Wrap ``value`` in the whitespace the field had in the source line."""
    return f"{padding[0]}{value}{padding[1]}"



def _ensure_style_section_format(sec: D.Section) -> None:
    """Give a freshly created style section the ``Format:`` line it lacks.

    ``AssDocument.ensure_section`` builds a bare section: appending a ``Style``
    line to it would leave the file with a style column layout that no
    ``Format`` line declares, so Aegisub (and this server's own parser) could
    no longer tell what the columns mean.  Mirrors what ``AssDocument.new``
    writes for a fresh document.
    """
    if any(isinstance(entry, D.DataEntry) for entry in sec.entries):
        return
    if any(_format_columns(getattr(entry, "raw", "") or "") for entry in sec.entries):
        return
    columns = (STYLE_FIELDS_V4 if (sec.style_format or "v4+").lower() == "v4"
               else STYLE_FIELDS_V4P)
    sec.entries.append(D.RawEntry("Format: " + ", ".join(columns)))
    if not sec.entries or (getattr(sec.entries[-1], "raw", None) or "").strip():
        sec.entries.append(D.RawEntry(""))


def _set_style_field(entry: Any, field: str, value: str) -> None:
    """Store ``value`` for ``field``, keeping the whitespace the source used.

    ``AssDocument`` renders a rewritten line as ``lead + ",".join(fields)`` and
    reads untouched fields verbatim from ``raw_fields`` while ``get()`` strips
    them.  Writing the *padded* text back into the raw field (instead of an
    override) therefore keeps ``Style: Default,...`` looking exactly like the
    file that came in, while callers still read ``"Default"`` back.
    """
    text = str(value).strip()
    order = list(getattr(entry, "order", []) or [])
    key = D.canonical_style_key(field)
    index = next((i for i, name in enumerate(order) if D.canonical_style_key(name) == key), None)
    raw_fields = getattr(entry, "raw_fields", None)
    if index is None or raw_fields is None:
        # ``DataEntry.set`` appends a field it does not know to ``order``, and
        # ``asscore`` hands freshly built entries the module level column list
        # itself, so copy before touching it: extending that object in place
        # would leak the new column into every later document in the process.
        entry.order = order
        entry.set(field, text)
        return
    padding = _raw_padding(entry, field)
    while len(raw_fields) <= index:
        raw_fields.append("")
    raw_fields[index] = _padded(text, padding)
    entry.unset(field)
    if getattr(entry, "raw", None):
        # ``render`` short-circuits to the untouched source when nothing is
        # overridden, which would hide the edit we just wrote.
        entry.raw = ""


def _entry_fields(entry: D.DataEntry) -> list[str]:
    return [D.canonical_style_key(f) for f in entry.order]


def _colour_convention(doc: D.AssDocument) -> dict[str, str]:
    """Example colour spellings taken from the document's first style."""
    out: dict[str, str] = {}
    for entry in doc.styles():
        for field in COLOUR_FIELDS:
            if field not in out:
                value = entry.get(field)
                if value:
                    out[field] = value
        if len(out) == len(COLOUR_FIELDS):
            break
    return out


def _style_dict(entry: D.DataEntry) -> dict[str, Any]:
    """All ASS style fields as stored, plus ``extra_fields`` (vendor columns).

    Columns the section's ``Format:`` line does not declare are reported as
    ``None`` instead of being left out, so a caller can read any standard field
    without guessing whether the document stores it at all (SSA ``v4.00`` has
    no ``ScaleX``/``Spacing``/``Angle``/``BorderStyle`` and no ``RelativeTo``).
    """
    fields: dict[str, Any] = {name: None for name in STYLE_FIELDS}
    fields.pop(EXTRA_FIELD, None)
    extras: dict[str, Any] = {}
    for name in entry.order:
        canon = D.canonical_style_key(name)
        value = entry.get(name)
        if canon in STYLE_FIELDS or canon == EXTRA_FIELD:
            fields[canon] = value
        else:
            extras[name] = value
    fields["extra_fields"] = extras
    return fields


def _flag(value: str | None) -> bool:
    return str(value or "").strip().lower() not in ("", "0", "false", "no", "off", "none")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _style_typed(entry: D.DataEntry | None) -> dict[str, Any]:
    """Typed view of a style (numbers/flags), colours kept verbatim."""
    if entry is None:
        return {
            "name": None, "font": "", "font_size": 0.0,
            "primary_colour": "", "secondary_colour": "", "tertiary_colour": "",
            "outline_colour": "", "back_colour": "",
            "bold": False, "italic": False, "underline": False, "strike_out": False,
            "scale_x": 100.0, "scale_y": 100.0, "spacing": 0.0, "angle": 0.0,
            "border_style": 1, "outline": 0.0, "shadow": 0.0, "alignment": 2,
            "margin_l": 0, "margin_r": 0, "margin_v": 0, "encoding": 1,
            "alpha_level": 0,
        }
    get = entry.get
    return {
        "name": entry.name,
        "font": get(FN),
        "font_size": _num(get(FS), 0.0),
        "primary_colour": get(PC),
        "secondary_colour": get(SC),
        "tertiary_colour": get("TertiaryColour"),
        "outline_colour": get(OC),
        "back_colour": get(BC),
        "bold": _flag(get("Bold")),
        "italic": _flag(get("Italic")),
        "underline": _flag(get("Underline")),
        "strike_out": _flag(get("StrikeOut")),
        "scale_x": _num(get("ScaleX"), 100.0),
        "scale_y": _num(get("ScaleY"), 100.0),
        "spacing": _num(get("Spacing"), 0.0),
        "angle": _num(get("Angle"), 0.0),
        "border_style": _int(get("BorderStyle"), 1),
        "outline": _num(get("Outline"), 0.0),
        "shadow": _num(get("Shadow"), 0.0),
        "alignment": _int(get("Alignment"), 2),
        "margin_l": _int(get("MarginL"), 0),
        "margin_r": _int(get("MarginR"), 0),
        "margin_v": _int(get("MarginV"), 0),
        "encoding": _int(get("Encoding"), 1),
        "alpha_level": _int(get("AlphaLevel"), 0),
    }


def _style_usage_counts(doc: D.AssDocument) -> dict[str, Any]:
    defined = {s.name.lower(): s.name for s in doc.styles()}
    by_style: dict[str, dict[str, Any]] = {}
    by_actor: dict[str, dict[str, Any]] = {}
    unknown: dict[str, int] = {}
    used_lower: set[str] = set()
    for entry in doc.events():
        key = (entry.get("Style") or "")
        lower = key.lower()
        used_lower.add(lower)
        display = defined.get(lower, key)
        bucket = by_style.setdefault(display, {"name": display, "total": 0, "dialogue": 0, "comment": 0})
        bucket["total"] += 1
        bucket["comment" if entry.is_comment else "dialogue"] += 1
        if lower and lower not in defined:
            unknown[display] = unknown.get(display, 0) + 1
        actor = entry.get("Name") or ""
        abucket = by_actor.setdefault(actor, {"name": actor, "total": 0, "dialogue": 0, "comment": 0})
        abucket["total"] += 1
        abucket["comment" if entry.is_comment else "dialogue"] += 1
    unused = [s.name for s in doc.styles() if s.name.lower() not in used_lower]
    return {
        "by_style": by_style,
        "by_actor": by_actor,
        "unused_styles": unused,
        "unknown_styles": unknown,
    }


# --------------------------------------------------------------------------
# Font probing
# --------------------------------------------------------------------------

def _fontconfig_available() -> bool:
    return bool(shutil.which("fc-match") and shutil.which("fc-list"))


def _resolve_font(values: dict[str, Any]) -> dict[str, Any]:
    requested = str(values.get("font") or "")
    bold = bool(values.get("bold"))
    italic = bool(values.get("italic"))
    result: dict[str, Any] = {
        "requested": requested,
        "bold": bold,
        "italic": italic,
        "available": False,
        "resolved": None,
        "resolved_style": None,
        "file": None,
        "index": None,
        "substituted": None,
        "candidates": [],
        "error": None,
    }
    if not requested:
        result["error"] = "style declares no Fontname"
        return result
    try:
        candidates = M.fonts_match(requested, bold=bold, italic=italic, limit=5)
    except M.MeasureError as exc:
        result["error"] = str(exc)
        return result
    except Exception as exc:  # noqa: BLE001 - subprocess surprises must not escape
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["available"] = True
    result["candidates"] = candidates
    if candidates:
        first = candidates[0]
        result["resolved"] = first.get("family")
        result["resolved_style"] = first.get("style")
        result["file"] = first.get("file")
        result["index"] = first.get("index")
        result["substituted"] = bool(first.get("substituted"))
    return result


# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------

BBOX_SAMPLE = "Ag"

def ass_list_styles(doc_id: str | None = None, include_usage: bool = False,
                    include_bbox: bool = False) -> dict[str, Any]:
    """List every style in the document.

    Args:
        doc_id: document id or ``None`` for the current document.
        include_usage: add ``used`` (bool) and ``usage``
            (``{total, dialogue, comment}``) to each style, counted over all
            event lines (comments included).
        include_bbox: render the sample text :data:`BBOX_SAMPLE` in each style
            with libass and add ``bbox`` — the ink box in script coordinates
            (``{x, y, x1, y1, width, height}``) — plus ``bbox_error`` when the
            measurement could not be produced (for example no ffmpeg/libass).

    Returns:
        ``{"doc_id", "count", "styles": [ ... ]}`` where each style dict holds
        every ASS field exactly as stored (``Name``, ``Fontname``,
        ``Fontsize`` ... ``Encoding``, colours verbatim), plus ``index``,
        ``section`` (its ``[V4+ Styles]``/``[V4 Styles]`` header) and
        ``extra_fields`` (vendor columns not in the ASS vocabulary).
    """
    did, doc = _doc(doc_id)
    usage = _style_usage_counts(doc) if include_usage else None
    out: list[dict[str, Any]] = []
    for index, entry in enumerate(doc.styles()):
        item = _style_dict(entry)
        item["index"] = index
        item["section"] = _owning_section(doc, entry).header
        if include_usage and usage is not None:
            counts = usage["by_style"].get(entry.name, {"name": entry.name, "total": 0, "dialogue": 0, "comment": 0})
            item["used"] = counts["total"] > 0
            item["usage"] = counts
        if include_bbox:
            box, error = _measure_style(doc, entry)
            item["bbox"] = box
            item["bbox_error"] = error
        out.append(item)
    return ok(doc_id=did, count=len(out), styles=out)


def _measure_style(doc: D.AssDocument, entry: D.DataEntry) -> tuple[dict[str, Any] | None, str | None]:
    fields = {k: v for k, v in _style_dict(entry).items() if k != "extra_fields"}
    try:
        result = M.measure_line_render(BBOX_SAMPLE, fields, doc.play_res)
    except Exception as exc:  # noqa: BLE001 - missing ffmpeg/libass must not raise
        return None, f"{type(exc).__name__}: {exc}"
    return result.get("rect"), None


def ass_get_style(name: str, doc_id: str | None = None, include_glyphs: bool = False,
                  sample_text: str | None = None) -> dict[str, Any]:
    """Inspect one style, including the font it really resolves to.

    Args:
        name: style name (case-insensitive).
        doc_id: document id or ``None`` for the current document.
        include_glyphs: add a ``glyphs`` report (approximate Aegisub "characters
            not in font" check via fontconfig) for ``sample_text``.
        sample_text: text used for the glyph report.  When omitted it defaults
            to the plain text of the lines that use the style (capped at 2000
            characters).

    Returns:
        ``{"doc_id", "name", "style", "values", "font", "used", "usage"}`` —
        ``style`` holds every ASS field as stored, ``values`` the typed view
        (numbers/flags, colours verbatim), ``font`` the fontconfig resolution:
        ``{requested, bold, italic, available, resolved, resolved_style, file,
        index, substituted, candidates, error}``.  With ``include_glyphs`` adds
        ``glyphs`` (``{font, missing, fallbacks, covered, checked,
        coverage_source}``) and ``glyph_sample_text``.
    """
    did, doc = _doc(doc_id)
    entry = doc.get_style(name)
    if entry is None:
        known = ", ".join(s.name for s in doc.styles()) or "(none)"
        raise ToolError(f"no style named {name!r}; document styles: {known}")
    values = _style_typed(entry)
    usage = _style_usage_counts(doc)["by_style"].get(entry.name)
    font_info = _resolve_font(values)
    result: dict[str, Any] = ok(
        doc_id=did,
        name=entry.name,
        style=_style_dict(entry),
        values=values,
        font=font_info,
        used=bool(usage and usage["total"] > 0),
        usage=usage or {"name": entry.name, "total": 0, "dialogue": 0, "comment": 0},
    )
    if include_glyphs:
        text = sample_text if sample_text is not None else _style_sample_text(doc, entry.name)
        # glyph coverage must be probed against the family fontconfig actually
        # resolves, otherwise a substituted family reports every character as
        # missing just because the requested name is an alias fontconfig hides.
        family = font_info.get("resolved") or values.get("font") or ""
        glyphs = _glyph_report(text, family)
        glyphs["font_requested"] = values.get("font") or ""
        glyphs["font_resolved"] = font_info.get("resolved")
        glyphs["substituted"] = font_info.get("substituted")
        result["glyph_sample_text"] = text
        result["glyphs"] = glyphs
    return result


def _glyph_report(text: str, family: str) -> dict[str, Any]:
    try:
        report = M.glyph_check(text, family or None, limit_fallbacks=5)
        report["available"] = True
        report["error"] = None
        return report
    except Exception as exc:  # noqa: BLE001 - fontconfig may be missing/broken
        return {"font": family, "missing": [], "fallbacks": {}, "covered": [],
                "checked": 0, "coverage_source": None, "available": False,
                "error": f"{type(exc).__name__}: {exc}"}


def _style_sample_text(doc: D.AssDocument, style_name: str, limit: int = 2000) -> str:
    chunks: list[str] = []
    total = 0
    for entry in doc.events():
        if (entry.get("Style") or "").lower() != style_name.lower():
            continue
        if T.drawing_state(entry.text) > 0:
            continue
        plain = T.plain_text(entry.text).replace("\\N", " ").strip()
        if not plain:
            continue
        chunks.append(plain)
        total += len(plain)
        if total >= limit:
            break
    return " ".join(chunks)[:limit]


def _style_values_from(**kwargs: Any) -> dict[str, str]:
    """Build a ``{canonical field: text}`` mapping from tool keyword arguments."""
    out: dict[str, str] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        out[_canon_field(key)] = value  # raw; formatted by the caller
    return out


def ass_add_style(
    name: str,
    doc_id: str | None = None,
    font: str = "Arial",
    font_size: float = 48.0,
    primary_colour: str = "&H00FFFFFF&",
    secondary_colour: str = "&H000000FF&",
    outline_colour: str = "&H00000000&",
    back_colour: str = "&H00000000&",
    bold: int = 0,
    italic: int = 0,
    underline: int = 0,
    strike_out: int = 0,
    scale_x: float = 100.0,
    scale_y: float = 100.0,
    spacing: float = 0.0,
    angle: float = 0.0,
    border_style: int = 1,
    outline: float = 2.0,
    shadow: float = 2.0,
    alignment: int = 2,
    margin_l: int = 10,
    margin_r: int = 10,
    margin_v: int = 10,
    encoding: int = 1,
    relative_to: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a style, or replace an existing one of the same name.

    Every value is stored verbatim (``/48.0`` becomes ``48``); colour values
    given as ``#RRGGBB`` are converted into the spelling the document already
    uses for that field.  Values are validated before anything is written:
    ``font_size`` must be positive, ``alignment`` 1-9, ``border_style`` 1 or 3,
    margins whole numbers and colours a spelling the document can store.

    ``overwrite=True`` *replaces* the style: every field is (re)written from
    the arguments, so fields left at their defaults go back to those defaults
    rather than keeping the old value — use ``ass_update_style`` to change a
    few fields and leave the rest alone.  ``relative_to`` is Aegisub's
    style-manager setting: it is only written when the document's ``Format``
    line already declares a ``RelativeTo`` column, otherwise it is listed in
    ``ignored_fields`` (adding the column would desynchronise the ``Format``
    line).

    Returns:
        ``{"doc_id", "name", "created", "style", "fields_set",
        "ignored_fields"}`` — ``created`` is False when ``overwrite=True``
        replaced an existing style.
    """
    did, doc = _doc(doc_id)
    style_name = _clean_style_name(name)
    existing = doc.get_style(style_name)
    if existing is not None and not overwrite:
        raise ToolError(
            f"a style named {style_name!r} already exists; pass overwrite=True to replace it"
        )
    requested = _style_values_from(
        name=style_name,
        font=font,
        font_size=font_size,
        primary_colour=primary_colour,
        secondary_colour=secondary_colour,
        outline_colour=outline_colour,
        back_colour=back_colour,
        bold=bold,
        italic=italic,
        underline=underline,
        strike_out=strike_out,
        scale_x=scale_x,
        scale_y=scale_y,
        spacing=spacing,
        angle=angle,
        border_style=border_style,
        outline=outline,
        shadow=shadow,
        alignment=alignment,
        margin_l=margin_l,
        margin_r=margin_r,
        margin_v=margin_v,
        encoding=encoding,
        relative_to=relative_to,
    )
    convention = _colour_convention(doc)
    _validate_style_values(requested)
    workspace.snapshot(did)
    created = existing is None
    if created:
        sec = doc.style_section(create=True)
        assert sec is not None
        _ensure_style_section_format(sec)
        available = _section_field_names(sec)
        spec: dict[str, str] = {}
        ignored: dict[str, Any] = {}
        for field, value in requested.items():
            if field not in available:
                ignored[field] = value
                continue
            spec[field] = _field_text(field, value, convention.get(field))
        entry = doc.add_style(spec)
        _finish_new_style_entry(sec, entry)
        _place_style_entry(sec, entry)
    else:
        assert existing is not None
        entry = existing
        changed, ignored = _apply_style_fields(doc, entry, requested, convention)
        spec = {k: v["to"] for k, v in changed.items()}
    doc.dirty = True
    return ok(
        doc_id=did,
        name=entry.name,
        created=created,
        style=_style_dict(entry),
        fields_set=spec,
        ignored_fields=ignored,
    )


def _clean_style_name(name: Any) -> str:
    text = str(name or "").strip()
    if not text:
        raise ToolError("style name must not be empty")
    if "," in text:
        raise ToolError(f"style name must not contain a comma: {text!r}")
    if "\n" in text or "\r" in text:
        raise ToolError("style name must not contain a line break")
    return text


def _finish_new_style_entry(sec: D.Section, entry: D.DataEntry) -> None:
    """Make a freshly created Style line look native and follow the Format line.

    ``AssDocument.add_style`` falls back to the stock V4+ column list when the
    style section ends with blank lines, which would emit more columns than the
    section's ``Format`` declares.  Re-align the entry with the section and give
    it the ``Style: `` prefix Aegisub itself writes.
    """
    entry.lead = "Style: "
    fields = _section_field_names(sec)
    if _entry_fields(entry) != fields:
        entry.order = list(fields)
        entry.raw_fields = ["" for _ in fields]


def _place_style_entry(sec: D.Section, entry: D.DataEntry) -> None:
    """Keep a freshly added Style line with the other styles, not after blanks."""
    if not sec.entries or sec.entries[-1] is not entry:
        return
    sec.entries.pop()
    target = len(sec.entries)
    for i, item in enumerate(sec.entries):
        if _is_style_line_start(item):
            target = i + 1
    sec.entries.insert(target, entry)


def _apply_style_fields(doc: D.AssDocument, entry: D.DataEntry, requested: dict[str, Any],
                        convention: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    available = set(_entry_fields(entry))
    convention = convention or _colour_convention(doc)
    changed: dict[str, Any] = {}
    ignored: dict[str, Any] = {}
    for field, value in requested.items():
        if field not in available:
            ignored[field] = value
            continue
        before = entry.get(field)
        _set_style_field(entry, field, _field_text(field, value, before or convention.get(field)))
        after = entry.get(field)
        if after != before:
            changed[field] = {"from": before, "to": after}
    if changed:
        doc.dirty = True
    return changed, ignored


def ass_update_style(
    name: str,
    doc_id: str | None = None,
    *,
    font: str | None = None,
    font_size: float | None = None,
    primary_colour: str | None = None,
    secondary_colour: str | None = None,
    tertiary_colour: str | None = None,
    outline_colour: str | None = None,
    back_colour: str | None = None,
    alpha_level: int | None = None,
    bold: int | None = None,
    italic: int | None = None,
    underline: int | None = None,
    strike_out: int | None = None,
    scale_x: float | None = None,
    scale_y: float | None = None,
    spacing: float | None = None,
    angle: float | None = None,
    border_style: int | None = None,
    outline: float | None = None,
    shadow: float | None = None,
    alignment: int | None = None,
    margin_l: int | None = None,
    margin_r: int | None = None,
    margin_v: int | None = None,
    encoding: int | None = None,
    relative_to: int | None = None,
    new_name: str | None = None,
    **aliases: Any,
) -> dict[str, Any]:
    """Change only the fields that are given, on an existing style.

    Field names may be written in any spelling: the snake_case parameters above,
    the ASS column names (``Fontname=``, ``Fontsize=``, ``PrimaryColour=``,
    ``MarginV=`` ...) or any case/space variant — all of them resolve to the
    same canonical field.  Passing ``new_name`` (or ``Name=``) renames the
    style and, when ``update_lines`` semantics apply, repoints the lines that
    used it (this always happens, exactly like ``ass_rename_style``).

    Args:
        name: the style to update (case-insensitive).
        doc_id: document id or ``None`` for the current document.
        aliases: any further ``field=value`` pairs, e.g. ``Fontsize=60``.

    Returns:
        ``{"doc_id", "name", "previous_name", "changed", "style",
        "lines_updated", "ignored_fields"}``.  ``changed`` maps each field to
        ``{"from", "to"}`` in stored form; ``ignored_fields`` lists values the
        document's ``Format`` line cannot hold (for example ``ScaleX`` in a
        ``[V4 Styles]`` file).
    """
    did, doc = _doc(doc_id)
    entry = doc.get_style(name)
    if entry is None:
        known = ", ".join(s.name for s in doc.styles()) or "(none)"
        raise ToolError(f"no style named {name!r}; document styles: {known}")

    requested: dict[str, Any] = {}
    for field, value in _style_values_from(
        font=font,
        font_size=font_size,
        primary_colour=primary_colour,
        secondary_colour=secondary_colour,
        tertiary_colour=tertiary_colour,
        outline_colour=outline_colour,
        back_colour=back_colour,
        alpha_level=alpha_level,
        bold=bold,
        italic=italic,
        underline=underline,
        strike_out=strike_out,
        scale_x=scale_x,
        scale_y=scale_y,
        spacing=spacing,
        angle=angle,
        border_style=border_style,
        outline=outline,
        shadow=shadow,
        alignment=alignment,
        margin_l=margin_l,
        margin_r=margin_r,
        margin_v=margin_v,
        encoding=encoding,
        relative_to=relative_to,
    ).items():
        requested[field] = value
    target_name = new_name
    for key, value in aliases.items():
        field = _canon_field(key)
        if field == "Name":
            target_name = value
        else:
            requested[field] = value

    previous = entry.name
    if target_name is not None:
        target_name = _clean_style_name(target_name)
        if target_name.lower() != previous.lower() and doc.get_style(target_name) is not None:
            raise ToolError(f"a style named {target_name!r} already exists")

    if not requested and target_name is None:
        raise ToolError("no style field given; pass at least one field to update")

    _validate_style_values(requested)
    workspace.snapshot(did)
    changed, ignored = _apply_style_fields(doc, entry, requested)
    lines_updated = 0
    if target_name is not None and target_name != previous:
        before = entry.get("Name")
        _set_style_field(entry, "Name", target_name)
        changed["Name"] = {"from": before, "to": entry.name}
        lines_updated = _repoint_lines(doc, previous, entry.name)
        doc.dirty = True
    return ok(
        doc_id=did,
        name=entry.name,
        previous_name=previous,
        changed=changed,
        style=_style_dict(entry),
        lines_updated=lines_updated,
        ignored_fields=ignored,
    )


def _repoint_lines(doc: D.AssDocument, old_name: str, new_name: str) -> int:
    count = 0
    for entry in doc.events():
        if (entry.get("Style") or "").lower() == old_name.lower():
            entry.set("Style", new_name)
            count += 1
    if count:
        doc.dirty = True
    return count


def ass_remove_style(name: str, doc_id: str | None = None,
                     reassign_to: str | None = None) -> dict[str, Any]:
    """Delete a style, refusing to orphan lines that still use it.

    Args:
        name: style to delete.
        doc_id: document id or ``None`` for the current document.
        reassign_to: when the style is still used, the name of the style the
            lines should be moved to before the style is deleted.  Without it a
            used style is an error.

    Returns:
        ``{"doc_id", "name", "removed", "reassigned", "reassign_to"}`` where
        ``reassigned`` is the number of lines repointed.
    """
    did, doc = _doc(doc_id)
    entry = doc.get_style(name)
    if entry is None:
        known = ", ".join(s.name for s in doc.styles()) or "(none)"
        raise ToolError(f"no style named {name!r}; document styles: {known}")
    target = None
    if reassign_to is not None:
        target = doc.get_style(reassign_to)
        if target is None:
            raise ToolError(f"cannot reassign to unknown style {reassign_to!r}")
        if target.name.lower() == entry.name.lower():
            raise ToolError("reassign_to must name a different style")
    used = [i for i, e in enumerate(doc.events())
            if (e.get("Style") or "").lower() == entry.name.lower()]
    if used and target is None:
        raise ToolError(
            f"style {entry.name!r} is used by {len(used)} line(s); "
            f"pass reassign_to=<style> to move them first"
        )
    workspace.snapshot(did)
    reassigned = 0
    if target is not None and used:
        reassigned = _repoint_lines(doc, entry.name, target.name)
    removed = doc.remove_style(entry.name)
    doc.dirty = True
    return ok(doc_id=did, name=entry.name, removed=removed,
              reassigned=reassigned, reassign_to=target.name if target else None)


def ass_rename_style(old_name: str, new_name: str, doc_id: str | None = None,
                     update_lines: bool = True) -> dict[str, Any]:
    """Rename a style and (by default) rewrite the lines that reference it.

    Args:
        old_name: existing style name.
        new_name: new name; must not collide with another style.
        doc_id: document id or ``None`` for the current document.
        update_lines: when true, every line whose ``Style`` field equals
            ``old_name`` is rewritten to ``new_name``.

    Returns:
        ``{"doc_id", "old_name", "new_name", "lines_updated", "style"}``.
    """
    did, doc = _doc(doc_id)
    entry = doc.get_style(old_name)
    if entry is None:
        known = ", ".join(s.name for s in doc.styles()) or "(none)"
        raise ToolError(f"no style named {old_name!r}; document styles: {known}")
    target = _clean_style_name(new_name)
    if target.lower() != entry.name.lower() and doc.get_style(target) is not None:
        raise ToolError(f"a style named {target!r} already exists")
    workspace.snapshot(did)
    previous = entry.name
    _set_style_field(entry, "Name", target)
    lines_updated = _repoint_lines(doc, previous, entry.name) if update_lines else 0
    doc.dirty = True
    return ok(doc_id=did, old_name=previous, new_name=entry.name,
              lines_updated=lines_updated, style=_style_dict(entry))


def ass_copy_style(source: str, new_name: str, doc_id: str | None = None,
                   overwrite: bool = False) -> dict[str, Any]:
    """Copy a style, field for field, under a new name.

    Args:
        source: style to copy.
        new_name: name for the copy.
        doc_id: document id or ``None`` for the current document.
        overwrite: replace an existing style of that name instead of failing.

    Returns:
        ``{"doc_id", "source", "name", "created", "style", "ignored_fields"}``.
    """
    did, doc = _doc(doc_id)
    entry = doc.get_style(source)
    if entry is None:
        known = ", ".join(s.name for s in doc.styles()) or "(none)"
        raise ToolError(f"no style named {source!r}; document styles: {known}")
    target_name = _clean_style_name(new_name)
    target = doc.get_style(target_name)
    if target is not None and not overwrite:
        raise ToolError(
            f"a style named {target_name!r} already exists; pass overwrite=True to replace it"
        )
    spec: dict[str, Any] = {}
    available = set(_entry_fields(entry))
    for name in entry.order:
        canon = D.canonical_style_key(name)
        if canon == "Name":
            spec[canon] = target_name
            continue
        if canon in available:
            spec[canon] = entry.get(name)
    workspace.snapshot(did)
    convention = _colour_convention(doc)
    if target is None:
        sec = doc.style_section(create=True)
        assert sec is not None
        sec_fields = set(_section_field_names(sec))
        cleaned = {k: v for k, v in spec.items() if k in sec_fields}
        ignored = {k: v for k, v in spec.items() if k not in sec_fields}
        new_entry = doc.add_style(cleaned)
        _finish_new_style_entry(sec, new_entry)
        _place_style_entry(sec, new_entry)
        entry_out = new_entry
        created = True
    else:
        changed, ignored = _apply_style_fields(doc, target, spec, convention)
        entry_out = target
        created = False
    doc.dirty = True
    return ok(doc_id=did, source=entry.name, name=entry_out.name, created=created,
              style=_style_dict(entry_out), ignored_fields=ignored)


def ass_reorder_styles(order: Sequence[str] | str, doc_id: str | None = None) -> dict[str, Any]:
    """Reorder the style section.

    Args:
        order: the complete list of style names in their new order (a
            comma-separated string is also accepted).  It must be a permutation
            of the styles already present — missing, unknown or duplicated names
            are reported in the error.
        doc_id: document id or ``None`` for the current document.

    Returns:
        ``{"doc_id", "order", "count"}`` with the resulting order.
    """
    did, doc = _doc(doc_id)
    entries = doc.styles()
    if not entries:
        raise ToolError("document has no styles to reorder")
    keys = [part.strip() for part in order.split(",")] if isinstance(order, str) else [str(x).strip() for x in order]
    if not keys:
        raise ToolError("order must list every style name")
    by_name: dict[str, D.StyleEntry] = {}
    for entry in entries:
        by_name.setdefault(entry.name.lower(), entry)
    unknown = [k for k in keys if k.lower() not in by_name]
    duplicate_keys = {k.lower() for k in keys if keys.count(k) == 1 and False}  # placeholder, see below
    seen: set[str] = set()
    duplicates: list[str] = []
    for key in keys:
        low = key.lower()
        if low in seen:
            duplicates.append(key)
        seen.add(low)
    missing = [e.name for e in entries if e.name.lower() not in seen]
    if unknown or missing or duplicates or len(keys) != len(entries):
        problems = []
        if unknown:
            problems.append("unknown style(s): " + ", ".join(unknown))
        if missing:
            problems.append("missing style(s): " + ", ".join(missing))
        if duplicates:
            problems.append("duplicated style(s): " + ", ".join(duplicates))
        if not problems and len(keys) != len(entries):
            problems.append(f"expected {len(entries)} names, got {len(keys)}")
        raise ToolError("order must list exactly the existing styles; " + "; ".join(problems))
    del duplicate_keys
    desired = [by_name[key.lower()] for key in keys]
    workspace.snapshot(did)
    slots: list[tuple[D.Section, int]] = []
    for sec in doc.sections_of(D.KIND_STYLES):
        for i, item in enumerate(sec.entries):
            if isinstance(item, D.StyleEntry):
                slots.append((sec, i))
    for (sec, index), new_entry in zip(slots, desired):
        sec.entries[index] = new_entry
    doc.dirty = True
    return ok(doc_id=did, order=[e.name for e in desired], count=len(desired))


def ass_style_usage(doc_id: str | None = None, by: str = "style") -> dict[str, Any]:
    """Count how styles and actors are used, and name the unused styles.

    Args:
        doc_id: document id or ``None`` for the current document.
        by: which grouping goes in ``counts``: ``"style"``, ``"actor"`` or
            ``"effect"``.

    Returns:
        ``{"doc_id", "by", "counts", "by_style", "by_actor", "unused_styles",
        "unknown_styles", "lines"}``.  Each count bucket is
        ``{"name", "total", "dialogue", "comment"}``; ``unknown_styles`` maps a
        style referenced by lines but absent from the style section to its line
        count.
    """
    did, doc = _doc(doc_id)
    key = str(by or "style").strip().lower()
    counts = _style_usage_counts(doc)
    if key in ("style", "styles"):
        primary = counts["by_style"]
    elif key in ("actor", "actors", "name", "names"):
        primary = counts["by_actor"]
    elif key in ("effect", "effects"):
        primary = {}
        for entry in doc.events():
            effect = entry.get("Effect") or ""
            bucket = primary.setdefault(effect, {"name": effect, "total": 0, "dialogue": 0, "comment": 0})
            bucket["total"] += 1
            bucket["comment" if entry.is_comment else "dialogue"] += 1
    else:
        raise ToolError(f"by must be 'style', 'actor' or 'effect', got {by!r}")
    return ok(
        doc_id=did,
        by=key,
        counts=primary,
        by_style=counts["by_style"],
        by_actor=counts["by_actor"],
        unused_styles=counts["unused_styles"],
        unknown_styles=counts["unknown_styles"],
        lines=len(doc.events()),
    )


# -- per-line effective values -------------------------------------------------

_TAG_FIELD_MAP = {
    "fn": "font",
    "fscx": "scale_x",
    "fscy": "scale_y",
    "fsp": "spacing",
    "fr": "angle",
    "frz": "angle",
    "bord": "outline",
    "shad": "shadow",
    "an": "alignment",
}
_COLOUR_TAG_MAP = {"c": "primary_colour", "1c": "primary_colour", "2c": "secondary_colour",
                   "3c": "outline_colour", "4c": "back_colour"}
_ALPHA_TAG_MAP = {"alpha": "alpha", "1a": "primary_alpha", "2a": "secondary_alpha",
                  "3a": "outline_alpha", "4a": "back_alpha"}

# ``\fnArial`` / ``\rAlt`` / ``\alpha&H80&``: asscore's tag tokenizer matches the
# name with ``[1-4]?[A-Za-z]+``, which swallows a *word* argument into the name
# (``\fnImpact`` -> name "fnImpact", arg "").  Tags whose argument is a bare
# word/colour are therefore split back apart here before they are interpreted.
# Numeric arguments are never affected (``\fs60`` -> name "fs", arg "60").
_WORD_ARG_TAGS = ("alpha", "1c", "1a", "2c", "2a", "3c", "3a", "4c", "4a", "fn", "c", "r")


def _split_tag(tag: T.Tag) -> tuple[str, str]:
    """Return ``(name, arg)`` for *tag*, undoing the parser's name/arg folding.

    ``\fnImpact`` comes out of :func:`aegisub_mcp.asscore.tags.parse_block` as
    ``Tag(name="fnImpact", arg="")``; this returns ``("fn", "Impact")``.
    Parenthesised tags and tags with numeric arguments are returned unchanged.
    """
    name = tag.name
    if tag.paren or name in _TAG_FIELD_MAP or name in _COLOUR_TAG_MAP or name in _ALPHA_TAG_MAP \
            or name in ("b", "i", "u", "s", "a", "t"):
        return name, tag.arg
    if tag.arg:
        return name, tag.arg
    for base in _WORD_ARG_TAGS:
        if name.startswith(base) and len(name) > len(base):
            return base, name[len(base):]
    return name, tag.arg


def _apply_tag(values: dict[str, Any], tag: T.Tag, sources: dict[str, str]) -> None:
    name, arg = _split_tag(tag)
    raw = tag.render()
    if name == "fs":
        text = arg.strip().lstrip("+")
        current = values.get("font_size") or 0.0
        delta = arg.strip().startswith(("+", "-"))
        size = _num(text, current) + (current if delta else 0.0) if delta else _num(text, current)
        values["font_size"] = size
        sources["font_size"] = raw
    elif name in _TAG_FIELD_MAP:
        field = _TAG_FIELD_MAP[name]
        if field in ("scale_x", "scale_y", "spacing", "angle", "outline", "shadow"):
            values[field] = _num(arg, values.get(field) or 0.0)
        elif field == "alignment":
            values[field] = _int(arg, values.get(field) or 2)
        else:
            values[field] = arg
        sources[field] = raw
    elif name in _COLOUR_TAG_MAP:
        field = _COLOUR_TAG_MAP[name]
        values[field] = arg
        sources[field] = raw
    elif name in _ALPHA_TAG_MAP:
        field = _ALPHA_TAG_MAP[name]
        values[field] = arg
        sources[field] = raw
    elif name in ("b", "i", "u", "s"):
        field = {"b": "bold", "i": "italic", "u": "underline", "s": "strike_out"}[name]
        values[field] = _int(arg, 0) != 0
        sources[field] = raw
    elif name == "a":
        values["alignment_legacy"] = _int(arg, 2)
        sources["alignment_legacy"] = raw


def ass_style_for_line(index: int | None = None, doc_id: str | None = None) -> dict[str, Any]:
    """Effective style values for one line, inline override tags included.

    The line's ``Style`` field is resolved to a style; on top of it the inline
    override tags that change font, size, colour, weight, scale, spacing,
    border, shadow or alignment are applied, so the caller sees what libass
    would actually use.  ``\\r``/``\\rStyle`` restarts from a (possibly other)
    style, exactly as the renderer does.

    Args:
        index: 0-based line index in ``doc.events()`` order (comments included).
            ``None`` uses the session selection (``ass_select``), falling back to
            its first index.
        doc_id: document id or ``None`` for the current document.

    Returns:
        ``{"doc_id", "index", "style", "style_found", "style_definition",
        "resolved", "runs", "overrides", "transforms", "warnings", "text",
        "plain_text"}``.
        ``resolved`` is the typed effective state at the first visible
        character; ``runs`` splits the visible text into
        ``{start, end, values, sources, style}`` spans with identical values;
        ``overrides`` lists the field names that inline tags modified;
        ``transforms`` lists the ``\\t(...)`` tags (animation is reported, not
        folded into the numbers); colours stay in the spelling the tag/style
        used.
    """
    did, doc = _doc(doc_id)
    from .base import entry_at, resolve_indices

    if index is None:
        selected = resolve_indices("selection", doc, selection=workspace.selection,
                                   default_all=False)
        if not selected:
            raise ToolError("no line index given and the workspace selection is empty")
        index = selected[0]
    entry = entry_at(doc, int(index))
    style_name = entry.get("Style") or ""
    style_entry = doc.get_style(style_name) if style_name else None
    base = _style_typed(style_entry)
    warnings: list[str] = []
    if style_entry is None:
        warnings.append(f"line uses undefined style {style_name!r}; showing document defaults")

    parsed = T.parse(entry.text)
    state: dict[str, T.Tag] = {}
    current = dict(base)
    runs: list[dict[str, Any]] = []
    overrides: set[str] = set()
    transforms: list[str] = []
    plain_index = 0

    def flush(values: dict[str, Any], sources: dict[str, str], start: int, end: int,
              style: str | None) -> None:
        if runs and runs[-1]["end"] == start and runs[-1]["values"] == values \
                and runs[-1]["style"] == style and runs[-1]["sources"] == sources:
            runs[-1]["end"] = end
            return
        runs.append({"start": start, "end": end, "style": style, "values": values,
                     "sources": sources})

    for seg in parsed.segments:
        if isinstance(seg, T.TagBlock):
            for tag in seg.tags:
                tag_name, tag_arg = _split_tag(tag)
                if tag_name == "t":
                    transforms.append(tag.render())
                    continue
                if tag_name == "r":
                    target = doc.get_style(tag_arg) if tag_arg.strip() else style_entry
                    if target is None:
                        warnings.append(f"line resets to unknown style {tag_arg!r}")
                        target = style_entry
                    current = _style_typed(target)
                    state.clear()
                    overrides.add("reset")
                    continue
                state[tag_name] = tag
            continue
        values = dict(current)
        sources: dict[str, str] = {}
        for tag in state.values():
            _apply_tag(values, tag, sources)
        overrides.update(sources)
        for _ in seg.text:
            flush(values, sources, plain_index, plain_index + 1,
                  current.get("name"))
            plain_index += 1

    resolved = runs[0]["values"] if runs else base
    return ok(
        doc_id=did,
        index=int(index),
        style=style_name,
        style_found=style_entry is not None,
        style_definition=_style_dict(style_entry) if style_entry is not None else None,
        resolved=resolved,
        runs=runs,
        overrides=sorted(overrides),
        transforms=transforms,
        warnings=warnings,
        text=entry.text,
        plain_text=parsed.plain_text(),
    )


def ass_check_font_substitution(doc_id: str | None = None) -> dict[str, Any]:
    """Compare every style's requested font with the family fontconfig resolves.

    Args:
        doc_id: document id or ``None`` for the current document.

    Returns:
        ``{"doc_id", "fontconfig", "checked", "substituted", "substitutions",
        "styles"}``.  Each entry of ``styles`` is
        ``{"name", "requested", "bold", "italic", "resolved", "resolved_style",
        "file", "substituted", "candidates", "available", "error"}``;
        ``substitutions`` lists the style names whose request is substituted,
        which is exactly the set of styles that will not render in the font
        they ask for.
    """
    did, doc = _doc(doc_id)
    cache: dict[tuple[str, bool, bool], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for entry in doc.styles():
        values = _style_typed(entry)
        family = str(values.get("font") or "")
        key = (family.lower(), bool(values.get("bold")), bool(values.get("italic")))
        if key not in cache:
            cache[key] = _resolve_font(values)
        info = cache[key]
        results.append({
            "name": entry.name,
            "requested": family,
            "bold": bool(values.get("bold")),
            "italic": bool(values.get("italic")),
            "resolved": info.get("resolved"),
            "resolved_style": info.get("resolved_style"),
            "file": info.get("file"),
            "substituted": info.get("substituted"),
            "candidates": info.get("candidates"),
            "available": info.get("available"),
            "error": info.get("error"),
        })
    subs = [r["name"] for r in results if r["substituted"]]
    return ok(
        doc_id=did,
        fontconfig=_fontconfig_available(),
        checked=len(results),
        substituted=len(subs),
        substitutions=subs,
        styles=results,
    )


# --------------------------------------------------------------------------
# Script Info
# --------------------------------------------------------------------------

def _info_sec(doc: D.AssDocument, *, create: bool = False) -> D.Section | None:
    sec = doc.section(D.KIND_INFO, header="[Script Info]") or doc.section(D.KIND_INFO)
    if sec is None and create:
        sec = doc.ensure_section(D.KIND_INFO, header="[Script Info]")
    return sec


def ass_get_script_info(doc_id: str | None = None) -> dict[str, Any]:
    """Read ``[Script Info]`` in file order.

    Returns:
        ``{"doc_id", "count", "items", "ordered", "values", "duplicates"}``.
        ``items`` is one ``{"index", "key", "value", "raw", "duplicate"}`` per
        key line, in the order they appear in the file (``raw`` is the line
        exactly as stored, so odd spacing such as ``Title : x`` survives);
        ``ordered`` is ``[[key, value], ...]``; ``values`` the last-wins map;
        ``duplicates`` lists keys that appear more than once.
    """
    did, doc = _doc(doc_id)
    sec = _info_sec(doc)
    items: list[dict[str, Any]] = []
    values: dict[str, str] = {}
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for pos, key, value in doc.info_items():
        raw = sec.entries[pos].render() if (sec and pos < len(sec.entries)) else f"{key}: {value}"
        norm = _norm_key(key)
        duplicate = norm in seen
        if duplicate and key not in duplicates:
            duplicates.append(key)
        seen[norm] = pos
        values[key] = value
        items.append({"index": pos, "key": key, "value": value, "raw": raw,
                      "duplicate": duplicate})
    return ok(
        doc_id=did,
        count=len(items),
        items=items,
        ordered=[[i["key"], i["value"]] for i in items],
        values=values,
        duplicates=duplicates,
    )


def _check_key(key: Any, what: str = "key") -> str:
    text = str(key or "").strip()
    if not text:
        raise ToolError(f"Script Info {what} must not be empty")
    if ":" in text:
        raise ToolError(f"Script Info {what} must not contain ':' : {text!r}")
    if "\n" in text or "\r" in text:
        raise ToolError(f"Script Info {what} must not contain a line break")
    return text


def _check_value(value: Any) -> str:
    text = "" if value is None else str(value)
    if "\n" in text or "\r" in text:
        raise ToolError("Script Info values must stay on one line")
    return text


def ass_set_script_info(key: str, value: Any, doc_id: str | None = None,
                        before: str | None = None) -> dict[str, Any]:
    """Set, insert or reposition-free-update a ``[Script Info]`` key.

    Args:
        key: the key, e.g. ``"Title"``.  Matching is case- and
            space-insensitive, and the spelling already in the file is kept.
        value: new value (converted with ``str``; must not contain a newline).
        doc_id: document id or ``None`` for the current document.
        before: when the key does not exist yet, insert it immediately before
            this existing key instead of appending at the end.

    Returns:
        ``{"doc_id", "key", "value", "created", "index"}`` where ``index`` is
        the entry position inside ``[Script Info]``.
    """
    did, doc = _doc(doc_id)
    name = _check_key(key)
    text = _check_value(value)
    sec = _info_sec(doc, create=True)
    assert sec is not None
    norm = _norm_key(name)
    for pos, k, _ in doc.info_items():
        if _norm_key(k) == norm:
            workspace.snapshot(did)
            entry = sec.entries[pos]
            if isinstance(entry, D.RawEntry):
                entry.raw = f"{k}: {text}"
            else:
                entry.set(k, text)
            doc.dirty = True
            return ok(doc_id=did, key=k, value=text, created=False, index=pos)
    target = doc.info_items()[0][0] if False else None  # keep flake-free structure
    del target
    if before is not None:
        wanted = _check_key(before, "before key")
        for pos, k, _ in doc.info_items():
            if _norm_key(k) == _norm_key(wanted):
                workspace.snapshot(did)
                sec.entries.insert(pos, D.RawEntry(f"{name}: {text}"))
                doc.dirty = True
                return ok(doc_id=did, key=name, value=text, created=True, index=pos)
        raise ToolError(f"cannot insert before unknown Script Info key {before!r}")
    workspace.snapshot(did)
    sec.entries.append(D.RawEntry(f"{name}: {text}"))
    doc.dirty = True
    return ok(doc_id=did, key=name, value=text, created=True, index=len(sec.entries) - 1)


def ass_remove_script_info(key: str, doc_id: str | None = None) -> dict[str, Any]:
    """Remove a ``[Script Info]`` key.

    Returns ``{"doc_id", "key", "removed"}`` (``removed`` is False when the key
    was not present).
    """
    did, doc = _doc(doc_id)
    sec = _info_sec(doc)
    norm = _norm_key(_check_key(key))
    existing = None
    if sec is not None:
        for pos, k, _ in doc.info_items():
            if _norm_key(k) == norm:
                existing = (pos, k)
                break
    if existing is None:
        return ok(doc_id=did, key=str(key), removed=False)
    workspace.snapshot(did)
    del sec.entries[existing[0]]
    doc.dirty = True
    return ok(doc_id=did, key=existing[1], removed=True)


def ass_set_play_res(x: int, y: int, doc_id: str | None = None) -> dict[str, Any]:
    """Set ``PlayResX``/``PlayResY``.

    Returns ``{"doc_id", "play_res_x", "play_res_y", "previous_x",
    "previous_y"}``.
    """
    did, doc = _doc(doc_id)
    width, height = _int(x, -1), _int(y, -1)
    if width <= 0 or height <= 0:
        raise ToolError(f"play resolution must be positive, got ({x!r}, {y!r})")
    previous = doc.play_res
    workspace.snapshot(did)
    doc.set_play_res(width, height)
    doc.dirty = True
    px, py = doc.play_res
    return ok(doc_id=did, play_res_x=px, play_res_y=py,
              previous_x=previous[0], previous_y=previous[1])


WRAP_STYLES: dict[int, str] = {
    0: "smart",
    1: "end of line",
    2: "no wrapping",
    3: "bottom of line only",
}


def _coerce_wrap_style(style: Any) -> int:
    if isinstance(style, bool):
        raise ToolError("wrap style must be 0-3 or a name, not a boolean")
    numeric: int | None = None
    if isinstance(style, int):
        numeric = int(style)
    elif isinstance(style, str) and re.fullmatch(r"-?\d+", style.strip()):
        numeric = int(style.strip())
    elif isinstance(style, float) and float(style).is_integer():
        numeric = int(style)
    if numeric is not None:
        if numeric not in WRAP_STYLES:
            raise ToolError(
                f"wrap style {style!r} is out of range; use 0 (smart), 1 (end of line), "
                "2 (no wrapping) or 3 (bottom of line only)"
            )
        return numeric
    token = str(style or "").strip().lower().replace("_", " ")
    aliases = {
        "smart": 0, "smart wrapping": 0,
        "end": 1, "end of line": 1, "eol": 1,
        "none": 2, "no": 2, "no wrapping": 2, "nowrap": 2,
        "bottom": 3, "bottom of line only": 3, "bottom only": 3,
    }
    if token in aliases:
        return aliases[token]
    raise ToolError(
        f"unknown wrap style {style!r}; use 0 (smart), 1 (end of line), "
        "2 (no wrapping) or 3 (bottom of line only)"
    )


def ass_set_wrap_style(style: int | str, doc_id: str | None = None) -> dict[str, Any]:
    """Set ``WrapStyle`` using Aegisub's numeric codes.

    Args:
        style: ``0`` smart, ``1`` end of line, ``2`` no wrapping, ``3`` bottom
            of line only.  The names are accepted too.
        doc_id: document id or ``None`` for the current document.

    Returns:
        ``{"doc_id", "wrap_style", "name", "previous"}``.
    """
    did, doc = _doc(doc_id)
    value = _coerce_wrap_style(style)
    previous = doc.wrapping
    workspace.snapshot(did)
    doc.info_set("WrapStyle", value)
    doc.dirty = True
    return ok(doc_id=did, wrap_style=value, name=WRAP_STYLES[value], previous=previous)


def _coerce_boolish(value: Any, what: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ToolError(f"{what} must be true/false (or yes/no), got {value!r}")
    token = str(value or "").strip().lower()
    if token in ("yes", "y", "true", "1", "on"):
        return True
    if token in ("no", "n", "false", "0", "off"):
        return False
    raise ToolError(f"{what} must be yes/no (or true/false), got {value!r}")


def ass_set_scaled_border_and_shadow(value: Any, doc_id: str | None = None) -> dict[str, Any]:
    """Set ``ScaledBorderAndShadow`` (``yes``/``no``).

    Returns ``{"doc_id", "value", "enabled", "previous"}``.
    """
    did, doc = _doc(doc_id)
    enabled = _coerce_boolish(value, "value")
    previous = doc.info_get("ScaledBorderAndShadow")
    workspace.snapshot(did)
    doc.info_set("ScaledBorderAndShadow", "yes" if enabled else "no")
    doc.dirty = True
    return ok(doc_id=did, value="yes" if enabled else "no", enabled=enabled,
              previous=previous)


def ass_set_timing_info(fps: float | None = None, video_file: str | None = None,
                        timecodes_file: str | None = None,
                        doc_id: str | None = None) -> dict[str, Any]:
    """Set the timing source keys Aegisub stores in ``[Script Info]``.

    ``fps`` writes ``FPS``, ``video_file`` writes ``Video File`` and
    ``timecodes_file`` writes ``Timecodes File``.  Passing an empty string for a
    path removes that key.  Calling the tool with no arguments at all changes
    nothing and just reports the three keys as they currently are (``changes``
    is then empty), which is the only way to read timing info.

    Returns:
        ``{"doc_id", "fps", "video_file", "timecodes_file", "changes"}`` with
        the resulting values (``None`` when a key is absent) and a
        ``changes`` map of ``key -> {"from", "to"}``.
    """
    did, doc = _doc(doc_id)
    if fps is None and video_file is None and timecodes_file is None:
        return ok(
            doc_id=did,
            fps=doc.info_get("FPS"),
            video_file=doc.info_get("Video File"),
            timecodes_file=doc.info_get("Timecodes File"),
            changes={},
        )
    workspace.snapshot(did)
    changes: dict[str, Any] = {}
    if fps is not None:
        value = _num(fps, -1.0)
        if value <= 0:
            raise ToolError(f"fps must be positive, got {fps!r}")
        text = f"{value:.6g}"
        before = doc.info_get("FPS")
        doc.info_set("FPS", text)
        changes["FPS"] = {"from": before, "to": text}
    for text_key, arg in (("Video File", video_file), ("Timecodes File", timecodes_file)):
        if arg is None:
            continue
        path = str(arg).strip()
        if "\n" in path or "\r" in path:
            raise ToolError(f"{text_key} must stay on one line")
        before = doc.info_get(text_key)
        if path == "":
            doc.info_remove(text_key)
            changes[text_key] = {"from": before, "to": None}
        else:
            doc.info_set(text_key, path)
            changes[text_key] = {"from": before, "to": path}
    doc.dirty = True
    return ok(
        doc_id=did,
        fps=doc.info_get("FPS"),
        video_file=doc.info_get("Video File"),
        timecodes_file=doc.info_get("Timecodes File"),
        changes=changes,
    )


# --------------------------------------------------------------------------
# Attachments
# --------------------------------------------------------------------------

ATTACHMENT_KINDS = {
    "font": D.KIND_FONTS,
    "fonts": D.KIND_FONTS,
    "graphic": D.KIND_GRAPHICS,
    "graphics": D.KIND_GRAPHICS,
    "image": D.KIND_GRAPHICS,
    "images": D.KIND_GRAPHICS,
    "picture": D.KIND_GRAPHICS,
}
_KIND_LABEL = {D.KIND_FONTS: "font", D.KIND_GRAPHICS: "image"}
_KIND_KEY = {D.KIND_FONTS: "fontname", D.KIND_GRAPHICS: "filename"}
_NAME_KEYS = {"fontname", "filename", "font", "image", "picture", "name", "file", "attachment"}

_FONT_MAGIC = (b"\x00\x01\x00\x00", b"OTTO", b"true", b"ttcf", b"wOFF", b"wOF2", b"ttf ", b"\x80\x01")
_IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM", b"\x00\x00\x01\x00", b"II*\x00", b"MM\x00*")

_EXT_KINDS = {
    ".ttf": "font", ".otf": "font", ".ttc": "font", ".otc": "font", ".ttf": "font",
    ".woff": "font", ".woff2": "font", ".pfb": "font", ".pfa": "font", ".dfont": "font",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".bmp": "image",
    ".webp": "image", ".ico": "image", ".tga": "image", ".tiff": "image",
}


def _sniff_bytes(data: bytes) -> str:
    if data.startswith(_FONT_MAGIC):
        return "font"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image"
    if data.startswith(_IMAGE_MAGIC):
        return "image"
    return "unknown"


def _kind_section(doc: D.AssDocument, kind: str, *, create: bool = False) -> D.Section | None:
    section_kind = ATTACHMENT_KINDS.get(str(kind).strip().lower())
    if section_kind is None:
        raise ToolError(
            f"unknown attachment kind {kind!r}; use 'font'/'fonts' or 'image'/'graphics'"
        )
    sec = doc.section(section_kind)
    if sec is None and create:
        header = "[Fonts]" if section_kind == D.KIND_FONTS else "[Graphics]"
        sec = doc.ensure_section(section_kind, header=header)
    return sec


def _attachment_records(sec: D.Section) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section_kind = sec.kind
    for pos, entry in enumerate(sec.entries):
        raw = entry.render()
        text = raw.strip()
        if not text:
            continue
        head, sep, rest = text.partition(":")
        key = head.strip().lower()
        if sep and key in _NAME_KEYS and not current_is_data_table(rest):
            declared = "font" if key in ("fontname", "font") else "image" if key in ("filename", "image", "picture") else _KIND_LABEL.get(section_kind)
            current = {
                "name": rest.strip(),
                "declared_kind": declared,
                "key": key,
                "name_line": pos,
                "line_indices": [pos],
                "data": [],
                "section_kind": section_kind,
            }
            records.append(current)
            continue
        if current is None:
            current = {
                "name": None,
                "declared_kind": _KIND_LABEL.get(section_kind),
                "key": None,
                "name_line": None,
                "line_indices": [],
                "data": [],
                "section_kind": section_kind,
            }
            records.append(current)
        current["line_indices"].append(pos)
        current["data"].append(raw.strip())
    return records


def current_is_data_table(rest: str) -> bool:
    """True when a ``key: value`` match is really a data line (e.g. ``font: x``).

    Base64 payloads never contain ``:``, so a name line must look like a bare
    filename: no further ``:`` and no ``=`` padding mixed into path-ish text.
    """
    value = rest.strip()
    return ":" in value or " " in value and "/" in value


def _attachment_info(record: dict[str, Any]) -> dict[str, Any]:
    data_text = "".join(record["data"])
    info: dict[str, Any] = {
        "name": record["name"],
        "kind": record["declared_kind"],
        "section": record["section_kind"],
        "key": record["key"],
        "data_lines": len(record["data"]),
        "encoded_chars": len(data_text),
        "line_indices": list(record["line_indices"]),
    }
    if not data_text:
        info.update({"decoded": True, "size": 0, "size_exact": True, "sha256": None,
                     "magic": None, "sniff": "unknown", "error": None})
        return info
    try:
        raw = base64.b64decode(data_text, validate=True)
    except (binascii.Error, ValueError) as exc:
        info.update({
            "decoded": False,
            "size": (len(data_text) * 3) // 4,
            "size_exact": False,
            "sha256": None,
            "magic": None,
            "sniff": "unknown",
            "error": f"{type(exc).__name__}: {exc}",
        })
        return info
    info.update({
        "decoded": True,
        "size": len(raw),
        "size_exact": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "magic": raw[:8].hex(),
        "sniff": _sniff_bytes(raw),
        "error": None,
    })
    return info


def ass_list_attachments(doc_id: str | None = None, kind: str | None = None) -> dict[str, Any]:
    """List ``[Fonts]``/``[Graphics]`` attachments with a magic-byte sniff.

    Args:
        doc_id: document id or ``None`` for the current document.
        kind: ``"font"``/``"fonts"`` or ``"image"``/``"graphics"`` to restrict
            the listing; ``None`` returns both sections.

    Returns:
        ``{"doc_id", "count", "attachments": [ ... ]}``.  Each attachment has
        ``name``, ``kind`` (the declared kind), ``section``, ``key``,
        ``data_lines``, ``encoded_chars``, ``decoded`` (is the payload valid
        base64), ``size`` (exact decoded size, or ``encoded*3/4`` when the data
        is not valid base64 — ``size_exact`` says which), ``sha256``,
        ``magic`` (first 8 decoded bytes, hex), ``sniff`` (``font``, ``image``
        or ``unknown`` from the magic bytes), ``line_indices`` and ``error``.
    """
    did, doc = _doc(doc_id)
    sections = (
        [(_kind_section(doc, kind) or None)]
        if kind is not None
        else [doc.section(D.KIND_FONTS), doc.section(D.KIND_GRAPHICS)]
    )
    out: list[dict[str, Any]] = []
    for sec in sections:
        if sec is None:
            continue
        for record in _attachment_records(sec):
            if record["name"] is None and not record["data"]:
                continue
            out.append(_attachment_info(record))
    return ok(doc_id=did, count=len(out), attachments=out)


def ass_add_attachment(path: str, doc_id: str | None = None, name: str | None = None,
                       kind: str | None = None) -> dict[str, Any]:
    """Attach a file to the document, base64-encoded, like Aegisub's attach menu.

    Args:
        path: file to attach (read from disk; the only filesystem input of this
            module besides the workspace itself).
        doc_id: document id or ``None`` for the current document.
        name: name stored in the document; defaults to the file's basename.
        kind: ``"font"`` or ``"image"``; inferred from the magic bytes, falling
            back to the file extension, when omitted.

    Returns:
        ``{"doc_id", "name", "kind", "section", "path", "bytes",
        "base64_lines", "sniff", "replaced"}``.  An attachment with the same
        name in the same section is replaced (``replaced`` is True).
    """
    did, doc = _doc(doc_id)
    source = Path(path).expanduser()
    if not source.is_file():
        raise ToolError(f"attachment file not found: {source}")
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise ToolError(f"cannot read {source}: {exc}") from exc
    stored_name = _check_value(name if name is not None else source.name).strip()
    if not stored_name:
        raise ToolError("attachment name must not be empty")
    sniffed = _sniff_bytes(data)
    resolved = (str(kind).strip().lower() if kind is not None else
                (sniffed if sniffed != "unknown" else _EXT_KINDS.get(source.suffix.lower(), "")))
    if resolved not in ATTACHMENT_KINDS:
        raise ToolError(
            f"cannot determine attachment kind for {source.name!r}; pass kind='font' or kind='image'"
        )
    section_kind = ATTACHMENT_KINDS[resolved]
    sec = _kind_section(doc, resolved, create=True)
    assert sec is not None
    payload = base64.b64encode(data).decode("ascii")
    lines = [payload[i:i + 80] for i in range(0, len(payload), 80)] or [""]
    workspace.snapshot(did)
    replaced = False
    for record in _attachment_records(sec):
        if record["name"] and record["name"].strip().lower() == stored_name.lower():
            _delete_attachment_lines(sec, record)
            replaced = True
    insert_at = len(sec.entries)
    while insert_at and not sec.entries[insert_at - 1].render().strip():
        insert_at -= 1
    block = [f"{_KIND_KEY[section_kind]}: {stored_name}"] + lines
    for offset, text in enumerate(block):
        sec.entries.insert(insert_at + offset, D.RawEntry(text))
    doc.dirty = True
    return ok(
        doc_id=did,
        name=stored_name,
        kind=_KIND_LABEL[section_kind],
        section=sec.header,
        path=str(source),
        bytes=len(data),
        base64_lines=len(lines),
        sniff=sniffed,
        replaced=replaced,
    )


def _delete_attachment_lines(sec: D.Section, record: dict[str, Any]) -> int:
    doomed = set(record["line_indices"])
    keep = [entry for i, entry in enumerate(sec.entries) if i not in doomed]
    removed = len(sec.entries) - len(keep)
    sec.entries = keep
    if removed:
        sec_owner_missing = False
        del sec_owner_missing
    return removed


def ass_extract_attachment(name: str, doc_id: str | None = None,
                           output_path: str | None = None) -> dict[str, Any]:
    """Decode an attachment and write it into the workspace output directory.

    Args:
        name: attachment name as stored in the document (case-insensitive).
        doc_id: document id or ``None`` for the current document.
        output_path: destination; a relative path is resolved inside
            ``workspace.output_dir``.

    Returns:
        ``{"doc_id", "name", "kind", "path", "bytes", "sha256", "sniff"}``.
    """
    did, doc = _doc(doc_id)
    wanted = str(name or "").strip().lower()
    if not wanted:
        raise ToolError("attachment name must not be empty")
    found: dict[str, Any] | None = None
    for sec in (doc.section(D.KIND_FONTS), doc.section(D.KIND_GRAPHICS)):
        if sec is None:
            continue
        for record in _attachment_records(sec):
            if record["name"] and record["name"].strip().lower() == wanted:
                found = record
                break
        if found:
            break
    if found is None:
        available = [a["name"] for a in _all_attachment_names(doc)]
        raise ToolError(
            f"no attachment named {name!r}; attachments: " + (", ".join(available) or "(none)")
        )
    info = _attachment_info(found)
    if not info["decoded"]:
        raise ToolError(
            f"attachment {found['name']!r} does not contain valid base64 data "
            f"({info['error']}); it cannot be extracted"
        )
    payload = base64.b64decode("".join(found["data"]), validate=True)
    out_dir = workspace.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    target = Path(output_path).expanduser() if output_path else Path(found["name"])
    if not target.is_absolute():
        target = out_dir / target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(payload)
    except OSError as exc:
        raise ToolError(f"cannot write {target}: {exc}") from exc
    return ok(
        doc_id=did,
        name=found["name"],
        kind=info["kind"],
        path=str(target),
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        sniff=info["sniff"],
    )


def _all_attachment_names(doc: D.AssDocument) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sec in (doc.section(D.KIND_FONTS), doc.section(D.KIND_GRAPHICS)):
        if sec is None:
            continue
        out.extend(_attachment_records(sec))
    return out


def ass_remove_attachment(name: str, doc_id: str | None = None) -> dict[str, Any]:
    """Remove an attachment (its name line and every payload line).

    Returns ``{"doc_id", "name", "removed", "lines_removed"}``.
    """
    did, doc = _doc(doc_id)
    wanted = str(name or "").strip().lower()
    if not wanted:
        raise ToolError("attachment name must not be empty")
    for sec in (doc.section(D.KIND_FONTS), doc.section(D.KIND_GRAPHICS)):
        if sec is None:
            continue
        for record in _attachment_records(sec):
            if record["name"] and record["name"].strip().lower() == wanted:
                workspace.snapshot(did)
                removed = _delete_attachment_lines(sec, record)
                doc.dirty = True
                return ok(doc_id=did, name=record["name"], removed=True,
                          lines_removed=removed)
    return ok(doc_id=did, name=str(name), removed=False, lines_removed=0)


# --------------------------------------------------------------------------
# Extradata
# --------------------------------------------------------------------------

def _extradata_records(doc: D.AssDocument) -> list[dict[str, Any]]:
    """Parse ``[Aegisub Extradata]`` lines without using ``AssDocument.extradata``.

    Two layouts are understood:

    ``Comment: <id>,<key>,<value>``            (what this module and
                                                ``AssDocument.set_extradata``
                                                write)
    ``Data: <kind>,<id>,<key>,<value>``        (integer kind + integer id)

    ``value`` is everything after the last structural comma, so values may
    contain commas.
    """
    sec = doc.section(D.KIND_EXTRADATA)
    out: list[dict[str, Any]] = []
    if sec is None:
        return out
    for pos, entry in enumerate(sec.entries):
        raw = entry.render()
        text = raw.strip()
        if _blank(text):
            continue
        head, sep, rest = text.partition(":")
        if not sep:
            continue
        parts = rest.split(",", 3)
        if len(parts) < 3:
            continue
        if _is_int_text(parts[0]) and _is_int_text(parts[1]) and len(parts) >= 4:
            layout = 4
            index, ident, value = parts[1].strip(), parts[2].strip(), parts[3]
        else:
            layout = 3
            index, ident = parts[0].strip(), parts[1].strip()
            value = ",".join(parts[2:])
        out.append({
            "index": pos,
            "head": head.strip(),
            "label": index,
            "id": ident,
            "value": value,
            "raw": raw,
            "layout": layout,
        })
    return out


def ass_list_extradata(doc_id: str | None = None) -> dict[str, Any]:
    """List ``[Aegisub Extradata]`` entries.

    Returns:
        ``{"doc_id", "count", "entries"}`` where each entry is
        ``{"index", "head", "label", "id", "value", "raw", "layout"}``
        (``index`` is the position inside the section, ``label`` the record's
        numeric label, ``id`` the stored identifier).
    """
    did, doc = _doc(doc_id)
    records = _extradata_records(doc)
    return ok(doc_id=did, count=len(records), entries=records)


def ass_set_extradata(ident: str, value: Any = None, doc_id: str | None = None,
                      remove: bool = False) -> dict[str, Any]:
    """Set or remove one ``[Aegisub Extradata]`` entry (idempotent).

    Args:
        ident: the entry identifier (matched case-insensitively).
        value: value to store; may be omitted when ``remove=True``.
        doc_id: document id or ``None`` for the current document.
        remove: delete every entry with this identifier instead of setting it.

    Returns:
        ``{"doc_id", "id", "value", "removed", "created"}``; setting an
        existing identifier updates it in place (never duplicates it).
    """
    did, doc = _doc(doc_id)
    key = _check_key(ident, "extradata id")
    if "," in key:
        # The value is the last comma separated field, so a comma in the id
        # would silently truncate it on the next read.
        raise ToolError(f"extradata id must not contain ',': {key!r}")
    sec = doc.section(D.KIND_EXTRADATA)
    matches = [r for r in _extradata_records(doc) if r["id"].lower() == key.lower()]
    if remove:
        if not matches:
            return ok(doc_id=did, id=key, value=None, removed=False, created=False)
        workspace.snapshot(did)
        doomed = {r["index"] for r in matches}
        assert sec is not None
        sec.entries = [e for i, e in enumerate(sec.entries) if i not in doomed]
        doc.dirty = True
        return ok(doc_id=did, id=key, value=None, removed=True, created=False)
    text = _check_value(value)
    if not text.strip():
        raise ToolError("extradata value must not be empty; pass remove=True to delete the entry")
    workspace.snapshot(did)
    if matches:
        assert sec is not None
        record = matches[0]
        entry = sec.entries[record["index"]]
        assert isinstance(entry, D.RawEntry)
        if record["layout"] == 4:
            parts = entry.raw.split(",", 3)
            entry.raw = f"{record['head']}: {parts[0].split(':', 1)[-1].strip()},{record['label']},{key},{text}"
            entry.raw = f"{record['head']}: {record['label']},{record['id']},{key},{text}"
        else:
            entry.raw = f"{record['head']}: {record['label']},{key},{text}"
        doc.dirty = True
        return ok(doc_id=did, id=key, value=text, removed=False, created=False)
    if sec is None:
        sec = doc.ensure_section(D.KIND_EXTRADATA, header="[Aegisub Extradata]")
    sec.entries.append(D.RawEntry(f"Comment: 0,{key},{text}"))
    doc.dirty = True
    return ok(doc_id=did, id=key, value=text, removed=False, created=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def ass_validate(doc_id: str | None = None) -> dict[str, Any]:
    """Structural validation of the whole document.

    Args:
        doc_id: document id or ``None`` for the current document.

    Returns:
        ``{"doc_id", "ok", "counts", "issues"}``.  ``ok`` is True when no issue
        has severity ``error``.  Every issue is
        ``{"code", "severity", "kind", "index", "message"}``; ``kind`` says what
        ``index`` refers to:

        ``script_info``  index into the ``[Script Info]`` entries
        ``style``        index into ``doc.styles()``
        ``line``         0-based line index in ``doc.events()`` order
        ``section``      index into ``doc.sections``
        ``document``     ``index`` is ``None``

        Codes: ``duplicate_style_name`` (error), ``duplicate_script_info_key``
        (warning), ``missing_script_type`` (warning), ``missing_style`` (error),
        ``end_before_start`` (error), ``zero_duration`` (warning),
        ``invalid_timestamp`` (error), ``comments_only_style`` (warning),
        ``unknown_section`` (warning), ``unknown_record`` (warning),
        ``malformed_line`` (warning), ``malformed_raw_line`` (warning) and
        ``field_count_mismatch`` (error).
    """
    did, doc = _doc(doc_id)
    issues: list[dict[str, Any]] = []

    def add(code: str, severity: str, kind: str, index: int | None, message: str) -> None:
        issues.append({"code": code, "severity": severity, "kind": kind,
                       "index": index, "message": message})

    # -- Script Info -------------------------------------------------------
    seen_keys: dict[str, int] = {}
    for pos, key, _ in doc.info_items():
        norm = _norm_key(key)
        if norm in seen_keys:
            add("duplicate_script_info_key", "warning", "script_info", pos,
                f"Script Info key {key!r} also appears at index {seen_keys[norm]}; "
                "readers disagree on which value wins")
        else:
            seen_keys[norm] = pos
    if doc.section(D.KIND_INFO) is None:
        add("missing_script_type", "warning", "document", None,
            "document has no [Script Info] section")
    elif "scripttype" not in seen_keys:
        add("missing_script_type", "warning", "document", None,
            "no ScriptType key; tools must guess between SSA v4.00 and ASS v4.00+")

    # -- styles ------------------------------------------------------------
    styles = doc.styles()
    by_name: dict[str, list[int]] = {}
    for index, entry in enumerate(styles):
        if not entry.name.strip():
            add("malformed_line", "warning", "style", index,
                "style line has an empty Name field")
        by_name.setdefault(entry.name.lower(), []).append(index)
        if entry.malformed:
            expected = len(entry.order)
            add("malformed_line", "warning", "style", index,
                f"style {entry.name!r} has {len(entry.raw_fields)} field(s) but "
                f"the Format line declares {expected}")
    for name, indices in by_name.items():
        if name and len(indices) > 1:
            add("duplicate_style_name", "error", "style", indices[1],
                f"style name {styles[indices[1]].name!r} is defined more than once "
                f"(indices {indices})")

    # -- sections ----------------------------------------------------------
    for index, sec in enumerate(doc.sections):
        if sec.kind == D.KIND_OTHER:
            add("unknown_section", "warning", "section", index,
                f"unknown section {sec.header!r} is preserved verbatim but ignored")
        for pos, entry in enumerate(sec.entries):
            if not isinstance(entry, D.RawEntry):
                continue
            text = entry.raw.strip()
            if _blank(text):
                continue
            if ":" not in text:
                if sec.kind in (D.KIND_INFO, D.KIND_STYLES, D.KIND_EVENTS, D.KIND_EXTRADATA):
                    # A line without a ':' in a ``key: value`` section cannot be
                    # parsed at all, so it is both a structurally malformed line
                    # and an unparseable raw record.
                    add("malformed_line", "warning", sec.kind, index,
                        f"{sec.header} line {pos} {text!r} has no ':' and cannot be parsed")
                    add("malformed_raw_line", "warning", "section", index,
                        f"{sec.header} line {pos} {text!r} has no ':' and cannot be parsed")
                continue
            head = text.partition(":")[0].strip().lower()
            if sec.kind == D.KIND_EVENTS and head not in (
                "dialogue", "comment", "command", "picture", "sound", "movie", "text",
                "blank", "format",
            ):
                add("unknown_record", "warning", "section", index,
                    f"{sec.header} record {head!r} is not a known event kind and is preserved verbatim")

    # -- lines -------------------------------------------------------------
    usage = _style_usage_counts(doc)
    dialogue_usage: dict[str, int] = {}
    comment_usage: dict[str, int] = {}
    comment_line_index: dict[str, int] = {}
    for index, entry in enumerate(doc.events()):
        raw_style = entry.get("Style")
        style_key = (raw_style or "").lower()
        if style_key and style_key not in by_name:
            add("missing_style", "error", "line", index,
                f"line uses style {raw_style!r} which is not defined in the style section")
        elif not style_key:
            add("missing_style", "error", "line", index,
                "line has an empty Style field")
        if entry.is_comment:
            comment_usage[style_key] = comment_usage.get(style_key, 0) + 1
            comment_line_index.setdefault(style_key, index)
        else:
            dialogue_usage[style_key] = dialogue_usage.get(style_key, 0) + 1

        if entry.malformed:
            add("field_count_mismatch", "error", "line", index,
                f"line has {len(entry.raw_fields)} field(s) but the Format line "
                f"declares {len(entry.order)}: {entry.kind}")
        start_raw = entry.get("Start")
        end_raw = entry.get("End")
        start = _try_time(start_raw)
        end = _try_time(end_raw)
        if start_raw and start is None:
            add("invalid_timestamp", "error", "line", index,
                f"Start {start_raw!r} is not a valid timestamp")
        if end_raw and end is None:
            add("invalid_timestamp", "error", "line", index,
                f"End {end_raw!r} is not a valid timestamp")
        if start is not None and end is not None:
            if end < start:
                add("end_before_start", "error", "line", index,
                    f"line ends ({end_raw}) before it starts ({start_raw})")
            elif end == start:
                add("zero_duration", "warning", "line", index,
                    "line has zero duration and will never be visible")
    for index, entry in enumerate(styles):
        key = entry.name.lower()
        if dialogue_usage.get(key, 0) == 0 and comment_usage.get(key, 0) > 0:
            add("comments_only_style", "warning", "style", index,
                f"style {entry.name!r} is only used by comment lines "
                f"({comment_usage[key]} comment line(s), 0 dialogue)")
    for key, count in comment_usage.items():
        # Same signal for a name that no style defines: nothing referencing it
        # is ever rendered either, and a typo in a comment line is exactly the
        # case a validator should surface.
        if not key or key in by_name or dialogue_usage.get(key, 0):
            continue
        add("comments_only_style", "warning", "line", comment_line_index.get(key),
            f"style {key!r} is used by {count} comment line(s) and no dialogue "
            f"line; nothing that uses it is ever rendered")
    del usage

    severity_rank = {"error": 0, "warning": 1}
    issues.sort(key=lambda i: (severity_rank.get(i["severity"], 2), i["kind"],
                               i["index"] if i["index"] is not None else -1))
    counts = {
        "error": sum(1 for i in issues if i["severity"] == "error"),
        "warning": sum(1 for i in issues if i["severity"] == "warning"),
        "total": len(issues),
    }
    return ok(doc_id=did, ok=counts["error"] == 0, counts=counts, issues=issues)


def _try_time(value: Any) -> int | None:
    from ..asscore import assutil as U

    return U.try_parse_time(value)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def register(mcp: Any, ws: Any = None) -> list[str]:
    """Register every ``ass_*`` function in this module with ``mcp``.

    ``ws`` is accepted for symmetry with the other tool modules (the singleton
    from :mod:`aegisub_mcp.tools.base` is what the functions actually use).
    Returns the sorted list of registered tool names.
    """
    del ws
    names: list[str] = []
    for name, obj in list(globals().items()):
        if not name.startswith("ass_") or not callable(obj):
            continue
        if not getattr(obj, "__module__", "").startswith(__name__):
            continue
        mcp.tool()(obj)
        names.append(name)
    return sorted(names)
