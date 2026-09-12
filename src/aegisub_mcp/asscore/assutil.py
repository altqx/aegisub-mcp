"""Shared helpers for ASS/SSA handling: time, colours, escapes, text utilities.

Kept dependency-free and byte-conscious: every helper here is used by the
document model, the tool layer and the Lua bridge, so behaviour must be
deterministic and documented.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

# ASS/SSA timestamps are h:mm:ss.cc (centiseconds). Some tools emit 3 decimal
# digits (milliseconds) and a few files use '.' or ':' as the fraction
# separator, so accept all of them.
_TIME_RE = re.compile(r"^\s*(?P<sign>-?)(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:[.,:](?P<frac>\d{1,3}))?\s*$")


class TimeParseError(ValueError):
    pass


def parse_time(value: str | int | float) -> int:
    """Parse an ASS timestamp into integer milliseconds.

    Accepts ``"0:00:01.23"``, ``"1:02:03.456"``, ``"0:00:01:23"`` and plain
    numbers (treated as milliseconds). Raises :class:`TimeParseError` otherwise.
    """
    if isinstance(value, bool):
        raise TimeParseError(f"cannot parse {value!r} as a timestamp")
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    if value is None:
        raise TimeParseError("cannot parse None as a timestamp")
    m = _TIME_RE.match(str(value))
    if not m:
        raise TimeParseError(f"invalid timestamp: {value!r}")
    hours = int(m.group("h"))
    minutes = int(m.group("m"))
    seconds = int(m.group("s"))
    frac_raw = m.group("frac") or "0"
    # 1 digit -> tenths, 2 -> centiseconds, 3 -> milliseconds
    frac_ms = int(frac_raw) * (10 ** (3 - len(frac_raw)))
    total = ((hours * 60 + minutes) * 60 + seconds) * 1000 + frac_ms
    if m.group("sign") == "-":
        total = -total
    return total


def try_parse_time(value: str | int | float | None, default: int | None = None) -> int | None:
    """Like :func:`parse_time` but returns ``default`` instead of raising."""
    try:
        return parse_time(value)  # type: ignore[arg-type]
    except (TimeParseError, TypeError, ValueError):
        return default


def format_time(ms: int | float, *, precision: int = 2) -> str:
    """Format milliseconds as an ASS timestamp (centiseconds by default)."""
    ms = int(round(ms))
    sign = "-" if ms < 0 else ""
    ms = abs(ms)
    total_seconds, millis = divmod(ms, 1000)
    if precision == 2:
        # ASS stores centiseconds; round half away from zero like Aegisub does.
        cs = (millis + 5) // 10
        if cs >= 100:
            cs = 0
            total_seconds += 1
        frac = f"{cs:02d}"
    elif precision == 3:
        frac = f"{millis:03d}"
    else:  # pragma: no cover - defensive
        raise ValueError("precision must be 2 or 3")
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{sign}{hours}:{minutes:02d}:{seconds:02d}.{frac}"


def ms_to_frames(ms: int | float, fps: float) -> int:
    return int(round((ms / 1000.0) * fps))


def frames_to_ms(frames: int | float, fps: float) -> int:
    return int(round((frames / fps) * 1000.0))


# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------

_ASS_COLOR_RE = re.compile(r"^&?H?([0-9A-Fa-f]{1,8})&?$")


def parse_ass_color(value: str, *, default_alpha: int = 0) -> tuple[int, int, int, int]:
    """Parse ``&HAABBGGRR`` (or decimal / ``#RRGGBB``) to ``(r, g, b, a)``.

    ``a`` is 0 = fully opaque (ASS convention), 255 = fully transparent.
    """
    if value is None or value == "":
        return (255, 255, 255, default_alpha)
    raw = str(value).strip()
    if raw.startswith("#"):
        hexpart = raw[1:]
        if len(hexpart) == 6:
            r, g, b = (int(hexpart[i : i + 2], 16) for i in (0, 2, 4))
            return (r, g, b, default_alpha)
        if len(hexpart) == 8:  # #RRGGBBAA (HTML-ish, alpha 00 = transparent)
            r, g, b, html_a = (int(hexpart[i : i + 2], 16) for i in (0, 2, 4, 6))
            return (r, g, b, 255 - html_a)
        raise ValueError(f"invalid hex colour: {value!r}")
    if raw.lower().startswith("&h"):
        hexpart = raw[2:].rstrip("&")
        val = int(hexpart, 16) if hexpart else 0
    elif raw.lower().startswith("&"):
        hexpart = raw[1:].rstrip("&")
        val = int(hexpart, 16) if hexpart else 0
    elif re.fullmatch(r"\d+", raw):
        val = int(raw, 10)
    elif _ASS_COLOR_RE.match(raw):
        val = int(raw, 16)
    else:
        raise ValueError(f"invalid ASS colour: {value!r}")
    a = (val >> 24) & 0xFF
    b = (val >> 16) & 0xFF
    g = (val >> 8) & 0xFF
    r = val & 0xFF
    return (r, g, b, a)


def format_ass_color(r: int, g: int, b: int, a: int = 0, *, use_decimal: bool = False) -> str:
    """Format ``(r, g, b, a)`` as an ASS colour string (alpha inverted)."""
    for name, v in (("r", r), ("g", g), ("b", b), ("a", a)):
        if not 0 <= int(v) <= 255:
            raise ValueError(f"colour component {name} out of range: {v}")
    val = (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r)
    if use_decimal:
        return f"{val:d}"
    if val <= 0xFFFFFF:
        return f"&H{val:06X}&"
    return f"&H{val:08X}&"


def ass_color_to_hex(value: str) -> str:
    """``&HAABBGGRR`` -> ``#RRGGBBAA`` (web order, alpha 00 = transparent)."""
    r, g, b, a = parse_ass_color(value)
    return f"#{r:02X}{g:02X}{b:02X}{255 - a:02X}"


def hex_to_ass_color(value: str) -> str:
    """``#RRGGBB`` / ``#RRGGBBAA`` -> ``&HAABBGGRR``."""
    return format_ass_color(*parse_ass_color(value))


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

# ``\N`` hard line break, ``\n`` soft break (renders as space in VSFilter,
# libass honours it as a break), ``\h`` non-breaking space.
BREAK_TOKENS = (r"\N", r"\n", r"\h")


def text_to_plain(text: str, *, keep_breaks: bool = False) -> str:
    """Strip override tags from ``text`` and turn \\N into a real newline."""
    from .tags import strip_tags  # local import to avoid cycles

    plain = strip_tags(text)
    plain = plain.replace(r"\h", "\u00a0")
    if keep_breaks:
        plain = plain.replace(r"\N", "\n").replace(r"\n", "\n")
    else:
        plain = plain.replace(r"\N", "\n").replace(r"\n", " ")
    return plain


def line_count_of(text: str) -> int:
    return text_to_plain(text, keep_breaks=True).count("\n") + 1


def escape_ass_text(text: str, *, hard_breaks: bool = True) -> str:
    """Escape raw newlines so the text stays on one ASS line."""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    if hard_breaks:
        out = out.replace("\n", r"\N")
    else:
        out = out.replace("\n", r"\n")
    return out


def unescape_ass_text(text: str) -> str:
    return text.replace(r"\N", "\n").replace(r"\n", "\n")


def reading_speed_cps(text: str, duration_ms: int) -> float:
    """Characters per second, using Aegisub's plain-text convention."""
    plain = text_to_plain(text)
    plain = plain.replace("\u00a0", "")
    dur = max(duration_ms, 0) / 1000.0
    if dur <= 0:
        return float("inf")
    return len(plain) / dur


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

def clamp(value, low, high):
    return max(low, min(high, value))


def round_coord(value: float, digits: int = 2) -> str:
    """Format a drawing/coordinate number the way ASS tools do (no trailing .0)."""
    rounded = round(float(value), digits)
    if abs(rounded - round(rounded)) < 10 ** (-(digits + 1)):
        return str(int(round(rounded)))
    s = f"{rounded:.{digits}f}".rstrip("0").rstrip(".")
    return s or "0"
