"""libass rendering of ASS scripts plus media probing helpers.

Everything in this module is built on two external programs only:

* ``ffmpeg``  -- renders ASS subtitles through its ``ass``/``subtitles`` video
  filter (which links libass/libharfbuzz/libfontconfig/libfreetype) and decodes
  audio/video for the probe helpers.
* ``ffprobe`` -- reports stream/frame metadata.

There is intentionally no dependency on Pillow, fontTools, numpy or any other
third party package: the only imports are the Python standard library plus
``subprocess``.  Bitmap inspection lives in :mod:`aegisub_mcp.asscore.measure`.

Rendering model
---------------
A frame is produced from a *synthetic* video source::

    -f lavfi -i color=c=black:s=<W>x<H>:r=25:d=0.04
    -vf "setpts=PTS+<seconds>/TB,ass=filename=<escaped>"

``setpts`` shifts the single generated frame to the requested timestamp *before*
the ``ass`` filter runs, so libass evaluates the script exactly as it would at
that point on a real timeline.  That makes ``render_frame`` O(1) in the
timestamp -- rendering at ``6:00:00.00`` costs the same as rendering at
``0:00:00.00``.  The source is always black so that background pixels never
contribute to ink measurement (see :func:`aegisub_mcp.asscore.measure.ink_bbox`).

Resolution / PlayRes handling
-----------------------------
``libass`` scales a script from its own ``PlayResX``/``PlayResY`` to whatever
frame size it is handed, so the *output* size is what the lavfi source is asked
for.  When the caller requests a size that differs from the script's PlayRes we
deliberately render the intermediate at PlayRes (1:1 script coordinates) and
then chain ``scale=<W>:<H>:flags=lanczos``.  That gives a deterministic,
inspectable pipeline: script units map 1:1 onto pixels in the intermediate
frame, and the reported ``width``/``height`` are exactly the requested ones.
The trade-off is a resample of already-rasterised glyphs instead of rasterising
at the final size; for measurement work (where the caller wants 1:1 script
coordinates) this is the correct behaviour, and callers who want maximum
crispness at a larger size can simply omit ``width``/``height`` and let libass
do the scaling itself.

Filter path escaping
--------------------
ffmpeg parses a ``-vf`` string once for the filtergraph syntax and once for the
option-value syntax, and both levels consume backslashes.  The escape counts
below were derived empirically on ffmpeg 8.0.1 by feeding candidate strings to
ffmpeg and reading back the value the parser actually produced (ffmpeg echoes
the unparseable path it tried to open, which makes a perfect oracle).  Naive
POSIX-style quoting does *not* survive; see :func:`safe_filter_path`.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from array import array
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "RenderError",
    "RenderUnavailable",
    "ffmpeg_path",
    "ffprobe_path",
    "ffmpeg_available",
    "ffprobe_available",
    "safe_filter_path",
    "render_frame",
    "render_frames",
    "render_video",
    "render_gray",
    "probe_video",
    "keyframes",
    "audio_peaks",
    "silencedetect",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "ENV_FFMPEG",
    "ENV_FFPROBE",
]

ENV_FFMPEG = "AEGISUB_MCP_FFMPEG"
ENV_FFPROBE = "AEGISUB_MCP_FFPROBE"

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

#: Value returned for ``rms_db``/``peak_db`` when an interval is digital
#: silence.  ``-inf`` is not representable in JSON, so a hard floor is used.
DB_FLOOR = -160.0

_AUDIO_RATE = 8000  # Hz, mono, used by audio_peaks
_SAMPLE_MAX = 32768.0  # full-scale magnitude of signed 16 bit audio

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Errors and executable discovery
# ---------------------------------------------------------------------------
class RenderError(RuntimeError):
    """Raised when a render/probe command fails or is given invalid input."""


class RenderUnavailable(RenderError):
    """Raised when ffmpeg/ffprobe (or a required encoder) cannot be found."""


def _resolve_executable(name: str, env_var: str, extra_candidates: Sequence[str]) -> str:
    override = os.environ.get(env_var)
    if override:
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return override
        raise RenderUnavailable(
            f"{env_var}={override!r} does not point to an executable file; "
            f"unset it or point it at a working {name} binary"
        )
    found = shutil.which(name)
    if found:
        return found
    for candidate in extra_candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RenderUnavailable(
        f"{name} not found on PATH (searched PATH and {', '.join(extra_candidates) or 'no fallbacks'}); "
        f"install ffmpeg or set {env_var} to an explicit binary path"
    )


def ffmpeg_path() -> str:
    """Return the ffmpeg binary to use.

    Honours the ``AEGISUB_MCP_FFMPEG`` environment variable first, then
    ``PATH``, then ``/usr/bin/ffmpeg`` / ``/usr/local/bin/ffmpeg``.

    Raises:
        RenderUnavailable: if no usable ffmpeg can be found.
    """
    return _resolve_executable("ffmpeg", ENV_FFMPEG, ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"))


def ffprobe_path() -> str:
    """Return the ffprobe binary to use (see :func:`ffmpeg_path`)."""
    return _resolve_executable(
        "ffprobe", ENV_FFPROBE, ("/usr/bin/ffprobe", "/usr/local/bin/ffprobe")
    )


def ffmpeg_available() -> bool:
    """True when :func:`ffmpeg_path` can resolve a binary.  Never raises."""
    try:
        ffmpeg_path()
    except RenderUnavailable:
        return False
    return True


def ffprobe_available() -> bool:
    """True when :func:`ffprobe_path` can resolve a binary.  Never raises."""
    try:
        ffprobe_path()
    except RenderUnavailable:
        return False
    return True


# ---------------------------------------------------------------------------
# Filter argument escaping
# ---------------------------------------------------------------------------
#: Number of backslashes to emit in front of a character so that ffmpeg's
#: filtergraph parser + option-value parser both hand the raw character to the
#: filter.  Empirically verified on ffmpeg 8.0.1 for an *unquoted* option value
#: such as ``ass=filename=VALUE``.
FILTER_ESCAPE_BACKSLASHES: Dict[str, int] = {
    "\\": 3,
    "'": 3,
    ":": 2,  # also covers the C:\ drive colon on Windows-style paths
    ",": 1,
    ";": 1,
    "[": 1,
    "]": 1,
}


def safe_filter_path(path: "str | os.PathLike[str]") -> str:
    """Escape a filesystem path for use as an ffmpeg *filter argument*.

    The returned string is meant to be dropped straight into a filtergraph,
    e.g. ``"ass=filename=" + safe_filter_path(p)``.

    ffmpeg parses that string twice -- once as filtergraph syntax (where
    ``\\``, ``,``, ``;``, ``[`` and ``]`` are special and single quotes group)
    and once as filter option values (where ``:`` separates options and ``\\``
    escapes).  A single backslash is therefore not enough: the table in
    :data:`FILTER_ESCAPE_BACKSLASHES` records how many are needed per character.
    Spaces do **not** need escaping in this context.

    Control characters (``c < 0x20``) cannot be represented reliably in a
    filtergraph and raise :class:`RenderError`; callers that must render next to
    such a path should copy the file somewhere safe first -- which is exactly
    what :func:`render_frame` and friends do automatically.

    Note:
        Verified against ffmpeg 8.0.1.  Older/other builds are believed to use
        the same rule (it follows from the two-level parser), but the rule is
        covered by round-trip tests that render through a directory whose name
        contains a space, a colon, a comma, a semicolon, brackets and an
        apostrophe.

    Args:
        path: filesystem path to escape.

    Returns:
        The escaped string, safe to concatenate after ``=`` in a filter option.
    """
    text = os.fspath(path)
    if isinstance(text, bytes):
        text = os.fsdecode(text)
    out: List[str] = []
    for ch in text:
        if ord(ch) < 0x20:
            raise RenderError(
                f"path {text!r} contains control character U+{ord(ch):04X}, which cannot be "
                "represented in an ffmpeg filtergraph; copy the file to a simple path first"
            )
        n = FILTER_ESCAPE_BACKSLASHES.get(ch)
        out.append("\\" * n + ch if n else ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _sanitize_color(value: str) -> str:
    """Validate a lavfi ``color=`` value so it cannot break the filtergraph."""
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9#@._,+-]{1,32}", text):
        raise RenderError(
            f"invalid background colour {value!r}: only letters, digits and #@._,+- are allowed"
        )
    return text


def _parse_play_res_text(text: str) -> Optional[Tuple[int, int]]:
    """Extract ``(PlayResX, PlayResY)`` from raw ASS text, or None."""
    xm = re.search(r"^[ \t]*PlayResX[ \t]*:[ \t]*(\d+)[ \t]*$", text, re.MULTILINE | re.IGNORECASE)
    ym = re.search(r"^[ \t]*PlayResY[ \t]*:[ \t]*(\d+)[ \t]*$", text, re.MULTILINE | re.IGNORECASE)
    if not xm or not ym:
        return None
    x, y = int(xm.group(1)), int(ym.group(1))
    if x <= 0 or y <= 0:
        return None
    return (x, y)


def _looks_like_ass_text(value: str) -> bool:
    """Heuristic: is this raw ASS content rather than a filesystem path?"""
    if "\n" in value or "\r" in value:
        return True
    return "[Script Info]" in value or "[Events]" in value or value.lstrip().startswith("[")


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 0x20 for ch in value)


@contextlib.contextmanager
def _ass_source(ass: "str | os.PathLike[str]") -> Iterator[Tuple[str, Optional[str]]]:
    """Yield ``(path, text)`` for an .ass file path or raw ASS text.

    Raw text is written to a private temporary directory that lives for the
    duration of the ``with`` block and is removed afterwards.  Files whose path
    cannot survive filtergraph escaping (control characters) are copied into
    that same temporary directory.
    """
    text: Optional[str] = None
    path: Optional[str] = None

    if isinstance(ass, (bytes, bytearray)):
        text = bytes(ass).decode("utf-8", "replace")
    elif isinstance(ass, os.PathLike):
        path = os.fspath(ass)
    elif isinstance(ass, str):
        if _looks_like_ass_text(ass):
            text = ass
        elif os.path.isfile(ass):
            path = ass
        else:
            raise RenderError(
                f"{ass!r} is neither an existing .ass file nor ASS text "
                "(raw text must contain a newline or a [Script Info]/[Events] section)"
            )
    else:
        raise RenderError(f"ass must be a path or ASS text, got {type(ass).__name__}")

    if path is not None and not _has_control_chars(path):
        if text is None:
            try:
                with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
                    text = fh.read()
            except OSError as exc:
                raise RenderError(f"cannot read .ass file {path!r}: {exc}") from exc
        yield path, text
        return

    # Raw text, or a path we must relocate: stage a private copy.
    tmpdir = tempfile.mkdtemp(prefix="aegisub-mcp-ass-")
    staged = os.path.join(tmpdir, "render.ass")
    try:
        if text is not None:
            with open(staged, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            assert path is not None
            try:
                shutil.copyfile(path, staged)
            except OSError as exc:
                raise RenderError(f"cannot stage .ass file {path!r}: {exc}") from exc
        yield staged, text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _effective_play_res(
    text: Optional[str], explicit: Optional[Sequence[int]]
) -> Optional[Tuple[int, int]]:
    """Explicit ``play_res`` argument wins, then the document's own PlayRes."""
    if explicit is not None:
        x, y = int(explicit[0]), int(explicit[1])
        if x <= 0 or y <= 0:
            raise RenderError(f"play_res must be positive, got {tuple(explicit)!r}")
        return (x, y)
    if text:
        return _parse_play_res_text(text)
    return None


def _lavfi_color(width: int, height: int, background: str, rate: int, duration: float) -> str:
    return (
        f"color=c={_sanitize_color(background)}:s={int(width)}x{int(height)}"
        f":r={int(rate)}:d={duration:.6f}"
    )


def _render_size(
    width: Optional[int], height: Optional[int], play_res: Optional[Tuple[int, int]]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return ``((src_w, src_h), (out_w, out_h))``."""
    src = play_res or (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    out = (int(width) if width else src[0], int(height) if height else src[1])
    if out[0] <= 0 or out[1] <= 0:
        raise RenderError(f"width/height must be positive, got {out!r}")
    return src, out


def _build_vf(
    ass_path: str,
    time_ms: float,
    src: Tuple[int, int],
    out: Tuple[int, int],
    extra_vf: Optional[str],
) -> Tuple[str, bool]:
    """Build the ``-vf`` chain; returns ``(vf, used_scale_filter)``."""
    parts = [
        f"setpts=PTS+{float(time_ms)}/1000/TB",
        f"ass=filename={safe_filter_path(ass_path)}",
    ]
    scaled = src != out
    if scaled:
        # See the module docstring: the intermediate frame is exactly PlayRes
        # sized (1:1 with script coordinates), then resampled to the request.
        parts.append(f"scale={out[0]}:{out[1]}:flags=lanczos")
    if extra_vf:
        parts.append(str(extra_vf))
    return ",".join(parts), scaled


def _run(cmd: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
    """Run a command capturing both streams as bytes.

    ``stdout`` is therefore always bytes, even for text output; use
    :func:`_stderr_text` or ``.decode()`` on it.
    """
    try:
        return subprocess.run(
            list(cmd),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - depends on load
        raise RenderError(f"command timed out after {timeout}s: {' '.join(cmd[:6])}...") from exc
    except OSError as exc:
        raise RenderUnavailable(f"cannot execute {cmd[0]!r}: {exc}") from exc


def _stderr_text(proc: subprocess.CompletedProcess) -> str:
    err = proc.stderr or b""
    if isinstance(err, str):
        return err
    return err.decode("utf-8", "replace")


def _check_ffmpeg(proc: subprocess.CompletedProcess, what: str) -> None:
    if proc.returncode != 0:
        tail = "\n".join(_stderr_text(proc).strip().splitlines()[-8:])
        raise RenderError(f"ffmpeg failed ({what}), exit {proc.returncode}:\n{tail}")


def _png_size(data: bytes) -> Optional[Tuple[int, int]]:
    """Read width/height from a PNG byte string (IHDR), stdlib only."""
    if len(data) < 24 or not data.startswith(_PNG_MAGIC) or data[12:16] != b"IHDR":
        return None
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    return (w, h) if w and h else None


_JPEG_SOF = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def _jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    """Read width/height from a JPEG byte string (SOFn marker), stdlib only."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in _JPEG_SOF:
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            return (w, h) if w and h else None
        i += 2 + seg_len
    return None


def _image_size(path: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    """Best-effort image dimensions read straight from the file header."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(65536)
    except OSError:
        return fallback
    for reader in (_png_size, _jpeg_size):
        size = reader(head)
        if size:
            return size
    return fallback


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_frame(
    ass: "str | os.PathLike[str]",
    time_ms: float,
    out_path: "str | os.PathLike[str]",
    width: Optional[int] = None,
    height: Optional[int] = None,
    play_res: Optional[Sequence[int]] = None,
    fmt: str = "png",
    background: str = "black",
    extra_vf: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    timeout: float = 60,
) -> Dict[str, Any]:
    """Render exactly one frame of an ASS script at ``time_ms``.

    Args:
        ass: either a path to an ``.ass`` file or the ASS document as raw text
            (raw text is written to a temporary file that is kept alive for the
            render and deleted afterwards).
        time_ms: timestamp in milliseconds at which libass should evaluate the
            script; implemented with ``setpts`` so cost is independent of it.
        out_path: destination image file.  Any format ffmpeg can write (``png``,
            ``jpg``, ``bmp``, ``webp``...).
        width, height: output size.  ``None`` means "use the document's PlayRes"
            (or 1920x1080 when the script declares none).  When a size is given
            that differs from PlayRes the frame is rendered at PlayRes and then
            rescaled with ``scale=W:H:flags=lanczos`` (see module docstring).
        play_res: force a script resolution instead of parsing PlayResX/PlayResY
            out of the document.  Useful for text snippets without a header.
        fmt: image format; inferred from ``out_path`` when not given explicitly
            (the ``-f`` muxer is chosen by ffmpeg from the extension, this
            argument is only used for the fallback size guess).
        background: lavfi colour for the backdrop.  Always black by default so
            the background contributes no ink (``ink_bbox`` uses a threshold).
        extra_vf: extra filter chain appended after ``ass`` (and after
            ``scale``).  Passed through verbatim -- treat it as trusted input.
        extra_args: extra ffmpeg arguments inserted just before the output path.
        timeout: per-invocation timeout in seconds.

    Returns:
        dict with ``out_path`` (absolute), ``width``, ``height``, ``bytes``,
        ``ffmpeg_cmd`` plus the informational keys ``time_ms``, ``play_res``,
        ``scaled`` and ``scale_filter``.

    Raises:
        RenderUnavailable: ffmpeg is missing.
        RenderError: bad input or a failing ffmpeg run.
    """
    ffmpeg = ffmpeg_path()
    out_abs = os.path.abspath(os.fspath(out_path))
    parent = os.path.dirname(out_abs)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    with _ass_source(ass) as (ass_path, text):
        src, out = _render_size(width, height, _effective_play_res(text, play_res))
        vf, scaled = _build_vf(ass_path, time_ms, src, out, extra_vf)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            _lavfi_color(src[0], src[1], background, 25, 0.04),
            "-vf",
            vf,
            "-frames:v",
            "1",
            *(str(a) for a in (extra_args or ())),
            out_abs,
        ]
        proc = _run(cmd, timeout)
        _check_ffmpeg(proc, f"render_frame at {time_ms} ms")

    if not os.path.isfile(out_abs):
        raise RenderError(f"ffmpeg reported success but {out_abs!r} was not created")
    return {
        "out_path": out_abs,
        "width": _image_size(out_abs, out)[0],
        "height": _image_size(out_abs, out)[1],
        "bytes": os.path.getsize(out_abs),
        "ffmpeg_cmd": cmd,
        "time_ms": float(time_ms),
        "play_res": list(src),
        "scaled": scaled,
    }


def _gcd(values: Iterable[int]) -> int:
    result = 0
    for value in values:
        result = math.gcd(result, int(value))
    return result


_RATE_DIVISORS = (1000, 500, 250, 200, 125, 100, 50, 40, 25, 20, 10, 8, 5, 4, 2, 1)


def render_frames(
    ass: "str | os.PathLike[str]",
    times_ms: Sequence[float],
    out_dir: "str | os.PathLike[str]",
    prefix: str = "frame",
    width: Optional[int] = None,
    height: Optional[int] = None,
    fmt: str = "png",
    **kw: Any,
) -> List[str]:
    """Render several frames of one script, in a single ffmpeg invocation.

    Implementation: **one ffmpeg call** using a lavfi ``color`` source whose
    frame rate is chosen so that every requested timestamp lands exactly on a
    frame boundary, plus a ``select='eq(n,i)+eq(n,j)+...'`` filter placed
    *before* the ``ass`` filter, so libass only rasterises the frames the caller
    asked for.  The source rate is ``1000 / g`` where ``g`` is the largest
    divisor of 1000 dividing every timestamp (subtitle times are normally
    multiples of 10 ms, so ``g`` is usually 10, 100 or 1000 and the number of
    source frames generated is tiny).  This keeps the output aligned 1:1 with
    ``times_ms`` while avoiding N process spawns.

    Args:
        ass: an .ass path or raw ASS text (see :func:`render_frame`).
        times_ms: timestamps to render; order is preserved in the result and
            duplicates map to the same file.
        out_dir: directory for the generated images (created if needed).
        prefix: filename prefix; files are ``<prefix>_%0Nd.<fmt>`` numbered from
            1 in ascending timestamp order.
        width, height, fmt: as in :func:`render_frame`.
        **kw: forwarded to the shared pipeline (``background``, ``extra_vf``,
            ``extra_args``, ``play_res``, ``timeout``).

    Returns:
        List of absolute file paths, one per entry of ``times_ms``.
    """
    times = [int(round(float(t))) for t in times_ms]
    if not times:
        return []
    ffmpeg = ffmpeg_path()
    out_dir_abs = os.path.abspath(os.fspath(out_dir))
    os.makedirs(out_dir_abs, exist_ok=True)

    uniq = sorted(set(times))
    step = _gcd(uniq) or 1
    step = next((d for d in _RATE_DIVISORS if step % d == 0), 1)
    rate = 1000 // step
    index = {t: t // step for t in uniq}
    # ffmpeg numbers the emitted image files sequentially from 1, so the file
    # name of a timestamp is its rank among the requested timestamps, while the
    # `select` expression uses the source frame index.
    rank = {t: i + 1 for i, t in enumerate(uniq)}
    duration = (max(index.values()) + 1) / rate

    pad = max(3, len(str(len(uniq))))
    pattern = os.path.join(out_dir_abs, f"{prefix}_%0{pad}d.{fmt}")
    select = "+".join(f"eq(n,{index[t]})" for t in uniq)

    extra_vf = kw.pop("extra_vf", None)
    background = kw.pop("background", "black")
    extra_args = kw.pop("extra_args", None)
    play_res = kw.pop("play_res", None)
    timeout = kw.pop("timeout", 120)
    if kw:
        raise RenderError(f"render_frames got unexpected keyword arguments: {sorted(kw)}")

    with _ass_source(ass) as (ass_path, text):
        src, out = _render_size(width, height, _effective_play_res(text, play_res))
        parts = [f"select='{select}'", f"ass=filename={safe_filter_path(ass_path)}"]
        if src != out:
            parts.append(f"scale={out[0]}:{out[1]}:flags=lanczos")
        if extra_vf:
            parts.append(str(extra_vf))
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            _lavfi_color(src[0], src[1], background, rate, duration),
            "-vf",
            ",".join(parts),
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(len(uniq)),
            *(str(a) for a in (extra_args or ())),
            pattern,
        ]
        proc = _run(cmd, timeout)
        _check_ffmpeg(proc, f"render_frames for {len(uniq)} timestamps")

    paths: Dict[int, str] = {}
    for t in uniq:
        p = os.path.join(out_dir_abs, f"{prefix}_{rank[t]:0{pad}d}.{fmt}")
        if not os.path.isfile(p):
            raise RenderError(f"expected frame {p!r} was not produced by ffmpeg")
        paths[t] = p
    return [paths[t] for t in times]


def _encoder_available(ffmpeg: str, name: str) -> bool:
    proc = _run([ffmpeg, "-hide_banner", "-v", "quiet", "-encoders"], 30)
    return name in (proc.stdout or b"").decode("utf-8", "replace")


def _document_end_ms(text: Optional[str], fallback: float) -> float:
    """Longest event end time in the document, in milliseconds."""
    if not text:
        return float(fallback)
    best = 0
    pattern = re.compile(
        r"^[ \t]*(?:Dialogue|Comment)[ \t]*:[ \t]*[^,]*,[ \t]*"
        r"\d+:\d{1,2}:\d{1,2}[.,]\d{1,3}[ \t]*,[ \t]*"  # start time (skipped)
        r"(\d+):(\d{1,2}):(\d{1,2})[.,](\d{1,3})",  # end time
        re.MULTILINE | re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        h, mi, s, cs = (int(g) for g in m.groups())
        best = max(best, ((h * 60 + mi) * 60 + s) * 1000 + cs * 10)
    return float(best) if best else float(fallback)


def render_video(
    ass: "str | os.PathLike[str]",
    out_path: "str | os.PathLike[str]",
    start_ms: float = 0,
    end_ms: Optional[float] = None,
    fps: float = 30,
    width: Optional[int] = None,
    height: Optional[int] = None,
    extra_vf: Optional[str] = None,
    background: str = "black",
    timeout: float = 600,
) -> Dict[str, Any]:
    """Render a subtitle video clip suitable as a user-facing preview.

    Uses ``libx264`` in an MP4 container when that encoder is available and
    falls back to ``mpeg4`` otherwise; both are muxed as ``yuv420p`` for broad
    player compatibility, which means the output dimensions are rounded up to
    even numbers when necessary.

    ``end_ms=None`` uses the longest ``Dialogue`` end time found in the document
    (falling back to ``start_ms + 5000`` when the document has no events).

    Returns:
        dict with ``out_path``, ``frames``, ``duration_ms``, ``ffmpeg_cmd`` plus
        ``width``, ``height``, ``codec``, ``bytes`` and ``fps``.
    """
    ffmpeg = ffmpeg_path()
    out_abs = os.path.abspath(os.fspath(out_path))
    parent = os.path.dirname(out_abs)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    fps = max(1.0, float(fps))

    with _ass_source(ass) as (ass_path, text):
        src, out = _render_size(width, height, _effective_play_res(text, None))
        out = (out[0] + out[0] % 2, out[1] + out[1] % 2)
        stop = float(end_ms) if end_ms is not None else _document_end_ms(text, float(start_ms) + 5000)
        duration_ms = max(1.0, stop - float(start_ms))
        duration_s = duration_ms / 1000.0

        if _encoder_available(ffmpeg, "libx264"):
            codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
            codec = "libx264"
        else:
            codec_args = ["-c:v", "mpeg4", "-q:v", "5"]
            codec = "mpeg4"

        parts = [
            # Shift the synthetic source to the clip's start so libass sees the
            # real timeline position of every frame.
            f"setpts=PTS+{float(start_ms)}/1000/TB",
            f"ass=filename={safe_filter_path(ass_path)}",
            # Re-zero the timestamps so the encoded clip starts at 0.
            "setpts=PTS-STARTPTS",
        ]
        if src != out:
            parts.append(f"scale={out[0]}:{out[1]}:flags=lanczos")
        if extra_vf:
            parts.append(str(extra_vf))
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            _lavfi_color(src[0], src[1], background, int(round(fps)), duration_s),
            "-vf",
            ",".join(parts),
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_abs,
        ]
        proc = _run(cmd, timeout)
        _check_ffmpeg(proc, f"render_video ({codec})")

    if not os.path.isfile(out_abs):
        raise RenderError(f"ffmpeg reported success but {out_abs!r} was not created")
    info: Dict[str, Any] = {}
    try:
        info = probe_video(out_abs)
    except RenderError:
        info = {}
    frames = info.get("nb_frames")
    if not frames:
        frames = int(round(duration_ms / 1000.0 * fps))
    return {
        "out_path": out_abs,
        "frames": int(frames),
        "duration_ms": float(duration_ms),
        "ffmpeg_cmd": cmd,
        "width": info.get("width", out[0]),
        "height": info.get("height", out[1]),
        "fps": info.get("fps") or fps,
        "codec": codec,
        "bytes": os.path.getsize(out_abs),
    }


def render_gray(
    ass: "str | os.PathLike[str]",
    time_ms: float,
    width: int,
    height: int,
    background: str = "black",
    timeout: float = 60,
) -> Tuple[bytes, int, int]:
    """Render one frame as 8-bit grayscale raw pixels.

    The pixels come back over stdout as ``-pix_fmt gray -f rawvideo pipe:1``,
    i.e. ``width * height`` bytes, row-major, top-left origin -- exactly what
    :func:`aegisub_mcp.asscore.measure.ink_bbox` consumes.

    ``width``/``height`` describe the raster to return.  As in
    :func:`render_frame`, a request that differs from the document's PlayRes is
    served by rendering at PlayRes and chaining ``scale=W:H:flags=lanczos``, so
    a caller wanting 1:1 script coordinates should pass the PlayRes values.

    Returns:
        ``(pixels, width, height)``.
    """
    ffmpeg = ffmpeg_path()
    with _ass_source(ass) as (ass_path, text):
        src, out = _render_size(width, height, _effective_play_res(text, None))
        vf, _ = _build_vf(ass_path, time_ms, src, out, None)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            _lavfi_color(src[0], src[1], background, 25, 0.04),
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        proc = _run(cmd, timeout)
        _check_ffmpeg(proc, f"render_gray at {time_ms} ms")
    data = proc.stdout or b""
    expected = out[0] * out[1]
    if len(data) != expected:
        raise RenderError(
            f"render_gray expected {expected} bytes ({out[0]}x{out[1]} gray) but got {len(data)}"
        )
    return data, out[0], out[1]


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def _probe_json(args: Sequence[str], timeout: float) -> Dict[str, Any]:
    cmd = [ffprobe_path(), "-v", "error", "-print_format", "json", *args]
    proc = _run(cmd, timeout)
    if proc.returncode != 0:
        tail = "\n".join(_stderr_text(proc).strip().splitlines()[-5:])
        raise RenderError(f"ffprobe failed, exit {proc.returncode}:\n{tail}")
    try:
        return json.loads((proc.stdout or b"").decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise RenderError(f"ffprobe produced invalid JSON: {exc}") from exc


def _parse_rate(value: Any) -> Optional[float]:
    """Parse an ffprobe rational such as ``30000/1001`` into a float."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("0/0", "N/A"):
        return None
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            num_f, den_f = float(num), float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        rate = num_f / den_f
    else:
        try:
            rate = float(text)
        except ValueError:
            return None
    return rate if rate > 0 else None


def _video_rotation(stream: Dict[str, Any]) -> Optional[float]:
    for entry in stream.get("side_data_list") or ():
        if isinstance(entry, dict) and "rotation" in entry:
            try:
                return float(entry["rotation"])
            except (TypeError, ValueError):
                continue
    tags = stream.get("tags") or {}
    if isinstance(tags, dict) and "rotate" in tags:
        try:
            return float(tags["rotate"])
        except (TypeError, ValueError):
            pass
    return None


def probe_video(path: "str | os.PathLike[str]") -> Dict[str, Any]:
    """Describe a media file with ``ffprobe -show_format -show_streams``.

    Returns:
        dict with ``width``, ``height``, ``fps`` (from ``r_frame_rate``,
        preferring it over ``avg_frame_rate``), ``duration_s``, ``nb_frames``,
        ``codec``, ``pix_fmt``, ``rotation`` (degrees from container side data,
        or None), ``audio`` (list of per-stream summaries) and ``format``
        (container format names), plus ``path``, ``size_bytes``, ``bit_rate``
        and ``stream_count``.
    """
    target = os.path.abspath(os.fspath(path))
    if not os.path.isfile(target):
        raise RenderError(f"cannot probe {target!r}: no such file")
    data = _probe_json(["-show_format", "-show_streams", target], 60)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    width = height = None
    fps = None
    nb_frames = None
    codec = pix_fmt = None
    rotation = None
    duration_s = None
    if video:
        width = video.get("width")
        height = video.get("height")
        fps = _parse_rate(video.get("r_frame_rate")) or _parse_rate(video.get("avg_frame_rate"))
        codec = video.get("codec_name")
        pix_fmt = video.get("pix_fmt")
        rotation = _video_rotation(video)
        if video.get("nb_frames") is not None:
            try:
                nb_frames = int(video["nb_frames"])
            except (TypeError, ValueError):
                nb_frames = None
        duration_s = _parse_rate(video.get("duration"))

    if duration_s is None:
        duration_s = _parse_rate(fmt.get("duration"))
    if duration_s is None and nb_frames and fps:
        duration_s = nb_frames / fps
    if nb_frames is None and duration_s and fps:
        nb_frames = int(round(duration_s * fps))

    return {
        "path": target,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_s": duration_s,
        "nb_frames": nb_frames,
        "codec": codec,
        "pix_fmt": pix_fmt,
        "rotation": rotation,
        "audio": [
            {
                "index": s.get("index"),
                "codec": s.get("codec_name"),
                "channels": s.get("channels"),
                "channel_layout": s.get("channel_layout"),
                "sample_rate": int(s["sample_rate"]) if str(s.get("sample_rate", "")).isdigit() else None,
                "bit_rate": int(s["bit_rate"]) if str(s.get("bit_rate", "")).isdigit() else None,
            }
            for s in audio
        ],
        "format": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
        "size_bytes": int(fmt["size"]) if str(fmt.get("size", "")).isdigit() else None,
        "bit_rate": int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        "stream_count": len(streams),
        "has_video": video is not None,
        "has_audio": bool(audio),
    }


_TIME_FIELDS = (
    "pts_time",
    "best_effort_timestamp_time",
    "pkt_pts_time",
    "pkt_dts_time",
    "pts",
)


def _frame_time(frame: Dict[str, Any]) -> Optional[float]:
    for field in _TIME_FIELDS:
        raw = frame.get(field)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _is_keyframe(frame: Dict[str, Any]) -> bool:
    if "key_frame" in frame:
        return bool(frame["key_frame"])
    if "keyframe" in frame:  # field-name variant
        return bool(frame["keyframe"])
    return True  # -skip_frame nokey already filtered


def _parse_keyframe_csv(text: str) -> List[float]:
    """Fallback parser for ``-of csv=p=0`` keyframe output."""
    times: List[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(",")]
        # csv output carries (key_frame, pts_time) in an unspecified order;
        # take the first field that looks like a fractional number.
        value = None
        for field in fields:
            if "." in field:
                try:
                    value = float(field)
                    break
                except ValueError:
                    continue
        if value is None:
            continue
        if fields and fields[0] in ("0", "1") and fields[0] == "0":
            continue  # non-keyframe flagged in the first column
        times.append(value)
    return times


def keyframes(
    path: "str | os.PathLike[str]", limit: int = 5000, timeout: float = 180
) -> List[float]:
    """Return keyframe presentation timestamps (seconds) of the video stream.

    Runs ``ffprobe -select_streams v -skip_frame nokey -show_entries
    frame=pts_time,key_frame`` and reads the JSON output, falling back to the
    CSV output when JSON is unavailable.  Timestamps are sorted ascending,
    deduplicated (1e-6 s granularity) and truncated to ``limit`` entries.
    """
    target = os.path.abspath(os.fspath(path))
    if not os.path.isfile(target):
        raise RenderError(f"cannot list keyframes of {target!r}: no such file")
    base = [
        "-select_streams",
        "v",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=pts_time,key_frame",
    ]
    try:
        data = _probe_json([*base, "-print_format", "json", target], timeout)
        times = []
        for frame in data.get("frames") or ():
            if not isinstance(frame, dict) or not _is_keyframe(frame):
                continue
            value = _frame_time(frame)
            if value is not None:
                times.append(value)
    except RenderError:
        cmd = [ffprobe_path(), "-v", "error", *base, "-of", "csv=p=0", target]
        proc = _run(cmd, timeout)
        if proc.returncode != 0:
            tail = "\n".join(_stderr_text(proc).strip().splitlines()[-5:])
            raise RenderError(f"ffprobe keyframe query failed, exit {proc.returncode}:\n{tail}")
        times = _parse_keyframe_csv((proc.stdout or b"").decode("utf-8", "replace"))

    unique = sorted({round(t, 6) for t in times})
    return unique[: max(0, int(limit))]


def audio_peaks(
    path: "str | os.PathLike[str]",
    interval_ms: int = 50,
    start_ms: float = 0,
    end_ms: Optional[float] = None,
    timeout: float = 300,
) -> List[Dict[str, Any]]:
    """Measure per-interval loudness of a media file's first audio stream.

    Audio is decoded to signed 16-bit mono at 8 kHz and read from stdout, then
    analysed in pure Python (``array`` + ``math.log10``).  ``audioop`` is *not*
    used: it was removed in Python 3.13.

    Args:
        path: media file.
        interval_ms: analysis window length in milliseconds.
        start_ms: skip this much audio (input-side ``-ss``).
        end_ms: stop at this timestamp (exclusive); ``None`` means to the end.
        timeout: subprocess timeout.

    Returns:
        One dict per window: ``t_ms`` (window start, absolute), ``rms_db`` and
        ``peak_db`` (dBFS, floored at :data:`DB_FLOOR` for digital silence) and
        ``samples`` (window length in decoded samples).
    """
    target = os.path.abspath(os.fspath(path))
    if not os.path.isfile(target):
        raise RenderError(f"cannot analyse audio of {target!r}: no such file")
    if interval_ms <= 0:
        raise RenderError(f"interval_ms must be positive, got {interval_ms!r}")
    ffmpeg = ffmpeg_path()

    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin"]
    if start_ms:
        cmd += ["-ss", f"{float(start_ms) / 1000.0:.6f}"]
    if end_ms is not None:
        cmd += ["-t", f"{max(0.0, (float(end_ms) - float(start_ms)) / 1000.0):.6f}"]
    cmd += [
        "-i",
        target,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_AUDIO_RATE),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    proc = _run(cmd, timeout)
    if proc.returncode != 0:
        tail = "\n".join(_stderr_text(proc).strip().splitlines()[-5:])
        raise RenderError(f"ffmpeg audio decode failed, exit {proc.returncode}:\n{tail}")

    raw = proc.stdout or b""
    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder != "little":  # pragma: no cover - only big-endian hosts
        samples.byteswap()

    per_interval = max(1, int(round(_AUDIO_RATE * float(interval_ms) / 1000.0)))
    results: List[Dict[str, Any]] = []
    for offset in range(0, len(samples), per_interval):
        window = samples[offset : offset + per_interval]
        if not window:
            break
        peak = 0
        total = 0
        for value in window:
            magnitude = value if value >= 0 else -value
            if magnitude > peak:
                peak = magnitude
            total += value * value
        rms = math.sqrt(total / len(window)) / _SAMPLE_MAX
        results.append(
            {
                "t_ms": float(start_ms) + (offset // per_interval) * float(interval_ms),
                "rms_db": _to_db(rms),
                "peak_db": _to_db(peak / _SAMPLE_MAX),
                "samples": len(window),
            }
        )
    return results


def _to_db(amplitude: float) -> float:
    if amplitude <= 0.0:
        return DB_FLOOR
    return max(DB_FLOOR, 20.0 * math.log10(amplitude))


_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)(?:\s*\|\s*silence_duration:\s*(-?[\d.]+))?")


def silencedetect(
    path: "str | os.PathLike[str]",
    noise_db: float = -45.0,
    min_duration_s: float = 0.2,
    timeout: float = 300,
) -> List[Dict[str, Any]]:
    """Find silent stretches using ffmpeg's ``silencedetect`` filter.

    Parses the filter's log lines (``silence_start: ...`` /
    ``silence_end: ... | silence_duration: ...``).  When the file ends inside a
    silence the final entry has ``end_ms`` and ``duration_ms`` set to ``None``.

    Args:
        path: media file.
        noise_db: silence threshold in dBFS (passed as ``noise=<v>dB``).
        min_duration_s: minimum silence length passed as ``d=<v>``.
        timeout: subprocess timeout.

    Returns:
        List of ``{"start_ms", "end_ms", "duration_ms"}`` dicts, in file order.
    """
    target = os.path.abspath(os.fspath(path))
    if not os.path.isfile(target):
        raise RenderError(f"cannot run silencedetect on {target!r}: no such file")
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-nostats",
        "-v",
        "info",
        "-nostdin",
        "-i",
        target,
        "-vn",
        "-af",
        f"silencedetect=noise={float(noise_db)}dB:d={float(min_duration_s)}",
        "-f",
        "null",
        "-",
    ]
    proc = _run(cmd, timeout)
    log = _stderr_text(proc)
    intervals = _parse_silence_log(log)
    if not intervals and proc.returncode != 0:
        return []
    return intervals


def _parse_silence_log(log: str) -> List[Dict[str, Any]]:
    """Parse ``silencedetect`` log lines into millisecond intervals.

    Split out of :func:`silencedetect` so the (otherwise unreachable in tests)
    unterminated-interval path can be exercised with a synthetic log.
    """
    intervals: List[Dict[str, Any]] = []
    open_start: Optional[float] = None
    for line in log.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            if open_start is not None:  # unterminated previous interval
                intervals.append({"start_ms": open_start, "end_ms": None, "duration_ms": None})
            open_start = float(m.group(1)) * 1000.0
            continue
        m = _SILENCE_END_RE.search(line)
        if m:
            end_s = float(m.group(1))
            dur_s = float(m.group(2)) if m.group(2) else None
            start_ms = (
                open_start
                if open_start is not None
                else max(0.0, (end_s - dur_s) * 1000.0) if dur_s else 0.0
            )
            intervals.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_s * 1000.0,
                    "duration_ms": (dur_s * 1000.0)
                    if dur_s is not None
                    else (end_s * 1000.0 - start_ms),
                }
            )
            open_start = None
    if open_start is not None:
        intervals.append({"start_ms": open_start, "end_ms": None, "duration_ms": None})
    return intervals
