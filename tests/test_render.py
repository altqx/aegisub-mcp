"""Tests for :mod:`aegisub_mcp.asscore.render` and ``...measure``.

The rendering/media tests require an ffmpeg+ffprobe pair with libass; they skip
cleanly when it is missing.  The font tests require fontconfig's ``fc-list`` and
``fc-match``.  Pure-unit tests (escaping, ``ink_bbox``, ``reading_speed``)
always run.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import struct
import subprocess

import pytest

from aegisub_mcp.asscore import measure as M
from aegisub_mcp.asscore import render as R

HAVE_FFMPEG = R.ffmpeg_available() and R.ffprobe_available()
HAVE_FONTS = bool(shutil.which("fc-list") and shutil.which("fc-match"))

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not available")
needs_fonts = pytest.mark.skipif(
    not HAVE_FONTS, reason="fontconfig fc-list/fc-match not available"
)

# PlayRes 640x480 so that one rendered pixel is one script unit.
RECT_INK = "m 100 50 l 300 50 300 150 100 150"
RECT_BOX = (100, 50, 300, 150)  # x0, y0, x1(exclusive), y1(exclusive)


def make_ass(
    text: str = "",
    play_res=None,
    tags: str = r"{\an7\pos(0,0)}",
    start: str = "0:00:00.00",
    end: str = "0:00:10.00",
    font: str = "DejaVu Sans",
    fontsize: int = 48,
    events=None,
) -> str:
    """Build a minimal ASS document for the tests."""
    if events is None:
        events = [(start, end, tags, text)]
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
    ]
    if play_res:
        header.append(f"PlayResX: {play_res[0]}")
        header.append(f"PlayResY: {play_res[1]}")
    body = [
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for ev_start, ev_end, ev_tags, ev_text in events:
        body.append(f"Dialogue: 0,{ev_start},{ev_end},Default,,0,0,0,,{ev_tags}{ev_text}")
    return "\n".join(
        header
        + [
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *body,
            "",
        ]
    )


RECT_ASS = make_ass(r"{\an7\pos(0,0)\p1}" + RECT_INK + r"{\p0}", play_res=(640, 480))

# Two non-overlapping events with different drawings, plus a leading blank:
#   0.0-1.0 s  blank | 1.0-2.0 s rect A | 2.5-3.5 s rect B
RECT_B_INK = "m 400 200 l 500 200 500 300 400 300"
MULTI_ASS = make_ass(
    play_res=(640, 480),
    events=[
        ("0:00:01.00", "0:00:02.00", r"{\an7\pos(0,0)\p1}", RECT_INK + r"{\p0}"),
        ("0:00:02.50", "0:00:03.50", r"{\an7\pos(0,0)\p1}", RECT_B_INK + r"{\p0}"),
    ],
)


def gray_bytes(width: int, height: int, points) -> bytes:
    """Build a black raster with the given (x, y, value) points set."""
    buf = bytearray(width * height)
    for x, y, value in points:
        buf[y * width + x] = value
    return bytes(buf)


def _decode_gray(path: pathlib.Path):
    """Decode a written PNG back to (gray_bytes, width, height) via ffmpeg."""
    data = pathlib.Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    proc = subprocess.run(
        [
            R.ffmpeg_path(), "-hide_banner", "-v", "error", "-i", str(path),
            "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
        ],
        capture_output=True,
        timeout=60,
        check=True,
    )
    assert len(proc.stdout) >= width * height
    return proc.stdout[: width * height], width, height


# ---------------------------------------------------------------------------
# escaping and executable discovery (no ffmpeg needed)
# ---------------------------------------------------------------------------
def test_safe_filter_path_escapes_specials() -> None:
    escaped = R.safe_filter_path("/a b/c:d,e;f[g]h'i\\j.ass")
    # Space must survive untouched (it is not special in a filtergraph).
    assert " " in escaped
    # Colon needs two backslashes, quote/backslash three, the rest one.
    assert "\\\\:" in escaped or escaped.count(":") == 1
    assert "c" in escaped and "d" in escaped
    for char in (",", ";", "[", "]"):
        assert "\\" + char in escaped
    assert escaped.endswith(".ass")
    # The raw path must not appear unescaped anywhere.
    assert "/a b/c:d,e;f[g]h'i\\j.ass" not in escaped


def test_safe_filter_path_leaves_simple_paths_alone() -> None:
    assert R.safe_filter_path("/tmp/plain.ass") == "/tmp/plain.ass"
    assert R.safe_filter_path("relative/file.ass") == "relative/file.ass"


def test_safe_filter_path_escapes_windows_drive_colon() -> None:
    escaped = R.safe_filter_path(r"C:\subs\a.ass")
    assert escaped.count(":") == 1
    assert escaped.index(":") > escaped.index("\\")  # the drive colon is escaped


def test_safe_filter_path_rejects_control_characters() -> None:
    with pytest.raises(R.RenderError):
        R.safe_filter_path("/tmp/bad\nname.ass")


def test_error_hierarchy() -> None:
    assert issubclass(R.RenderUnavailable, R.RenderError)
    assert issubclass(R.RenderError, RuntimeError)


def test_ffmpeg_path_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(R.ENV_FFMPEG, "/usr/bin/ffmpeg")
    assert R.ffmpeg_path() == "/usr/bin/ffmpeg"
    monkeypatch.setenv(R.ENV_FFPROBE, "/usr/bin/ffprobe")
    assert R.ffprobe_path() == "/usr/bin/ffprobe"


def test_missing_binary_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(R.ENV_FFMPEG, "/nonexistent/ffmpeg")
    with pytest.raises(R.RenderUnavailable) as excinfo:
        R.ffmpeg_path()
    assert R.ENV_FFMPEG in str(excinfo.value)
    assert R.ffmpeg_available() is False


# ---------------------------------------------------------------------------
# ink_bbox unit tests
# ---------------------------------------------------------------------------
def test_ink_bbox_empty_buffer_returns_none() -> None:
    assert M.ink_bbox(b"", 0, 0) is None
    assert M.ink_bbox(bytes(64), 8, 8) is None  # all background


def test_ink_bbox_single_pixel() -> None:
    buf = gray_bytes(10, 6, [(4, 2, 255)])
    box = M.ink_bbox(buf, 10, 6)
    assert box is not None
    assert (box["x0"], box["y0"], box["x1"], box["y1"]) == (4, 2, 5, 3)
    assert box["width"] == 1 and box["height"] == 1
    assert box["ink_pixels"] == 1
    assert box["coverage"] == pytest.approx(1 / 60)


def test_ink_bbox_corner_pixels() -> None:
    w, h = 8, 5
    buf = gray_bytes(w, h, [(0, 0, 200), (w - 1, 0, 200), (0, h - 1, 200), (w - 1, h - 1, 200)])
    box = M.ink_bbox(buf, w, h)
    assert box is not None
    assert (box["x0"], box["y0"], box["x1"], box["y1"]) == (0, 0, w, h)
    assert box["ink_pixels"] == 4


def test_ink_bbox_edges_are_exclusive_and_coverage_consistent() -> None:
    buf = gray_bytes(6, 4, [(x, y, 255) for x in range(1, 4) for y in range(1, 3)])
    box = M.ink_bbox(buf, 6, 4)
    assert box is not None
    assert (box["x0"], box["y0"], box["x1"], box["y1"]) == (1, 1, 4, 3)
    assert box["width"] == 3 and box["height"] == 2
    assert box["ink_pixels"] == 6
    assert box["coverage"] == pytest.approx(6 / 24)
    assert box["image_width"] == 6 and box["image_height"] == 4


def test_ink_bbox_threshold_is_inclusive_lower_bound() -> None:
    buf = gray_bytes(4, 1, [(0, 0, 15), (1, 0, 16), (2, 0, 200)])
    assert M.ink_bbox(buf, 4, 1, threshold=16)["x0"] == 1
    assert M.ink_bbox(buf, 4, 1, threshold=200)["x0"] == 2
    assert M.ink_bbox(buf, 4, 1, threshold=255) is None
    assert M.ink_bbox(buf, 4, 1, threshold=1)["x0"] == 0


def test_ink_bbox_short_buffer_raises() -> None:
    with pytest.raises(ValueError):
        M.ink_bbox(bytes(5), 4, 4)


def test_ink_bbox_ignores_trailing_bytes() -> None:
    buf = gray_bytes(4, 2, [(1, 1, 255)]) + b"\xff\xff"
    box = M.ink_bbox(buf, 4, 2)
    assert box is not None and box["ink_pixels"] == 1


# ---------------------------------------------------------------------------
# measure helpers that do not need ffmpeg
# ---------------------------------------------------------------------------
def test_reading_speed() -> None:
    assert M.reading_speed(40, 2000) == pytest.approx(20.0)
    assert M.reading_speed(0, 1000) == pytest.approx(0.0)
    assert M.reading_speed(10, 0) == float("inf")
    assert M.reading_speed(10, -5) == float("inf")


def test_build_line_ass_contains_play_res_and_tags() -> None:
    text = M.build_line_ass("Hello, world!", {"Fontname": "Noto Sans", "Fontsize": 64}, (1280, 720))
    assert "PlayResX: 1280" in text and "PlayResY: 720" in text
    assert "Noto Sans" in text and "64" in text
    assert M.DEFAULT_TAGS + "Hello, world!" in text
    assert "0:00:00.00,0:00:10.00" in text


def test_build_line_ass_escapes_newlines_and_rejects_bad_play_res() -> None:
    text = M.build_line_ass("a\nb", {}, (640, 480))
    assert "a\\Nb" in text
    with pytest.raises(M.MeasureError):
        M.build_line_ass("x", {}, (0, 0))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
@needs_ffmpeg
def test_render_frame_returns_metadata(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "one.png"
    result = R.render_frame(RECT_ASS, 1500, out)
    assert os.path.isabs(result["out_path"])
    assert result["out_path"] == str(out)
    assert (result["width"], result["height"]) == (640, 480)
    assert result["bytes"] == out.stat().st_size > 0
    assert result["play_res"] == [640, 480]
    assert result["scaled"] is False
    assert isinstance(result["ffmpeg_cmd"], list)
    assert "ass=filename=" in " ".join(result["ffmpeg_cmd"])
    assert out.read_bytes().startswith(b"\x89PNG")


@needs_ffmpeg
def test_render_frame_supports_jpeg_and_reads_real_size(tmp_path: pathlib.Path) -> None:
    result = R.render_frame(RECT_ASS, 1500, tmp_path / "one.jpg")
    assert (result["width"], result["height"]) == (640, 480)


@needs_ffmpeg
def test_render_frame_accepts_raw_text_and_stages_a_temp_file(tmp_path: pathlib.Path) -> None:
    result = R.render_frame(RECT_ASS, 1500, tmp_path / "raw.png")
    assert result["play_res"] == [640, 480]
    gray, w, h = R.render_gray(RECT_ASS, 1500, 640, 480)
    assert M.ink_bbox(gray, w, h)["width"] == 200


@needs_ffmpeg
def test_render_frame_is_time_aware() -> None:
    """0.0-10.0 s has the drawing, so 11.0 s must render nothing at all."""
    inside = R.render_gray(RECT_ASS, 1500, 640, 480)
    also_inside = R.render_gray(RECT_ASS, 9500, 640, 480)
    after_end = R.render_gray(RECT_ASS, 11000, 640, 480)
    assert M.ink_bbox(*inside) is not None
    assert M.ink_bbox(*also_inside) is not None
    assert M.ink_bbox(*after_end) is None
    # ...and the frame really is blank, not merely dark.
    assert set(after_end[0]) == {0}


@needs_ffmpeg
def test_render_frame_uses_play_res_when_size_not_given(tmp_path: pathlib.Path) -> None:
    result = R.render_frame(RECT_ASS, 1500, tmp_path / "pr.png")
    assert (result["width"], result["height"]) == (640, 480)


@needs_ffmpeg
def test_render_frame_defaults_to_1920x1080_without_play_res(tmp_path: pathlib.Path) -> None:
    ass = make_ass("hi", play_res=None)
    result = R.render_frame(ass, 1500, tmp_path / "def.png")
    assert (result["width"], result["height"]) == (R.DEFAULT_WIDTH, R.DEFAULT_HEIGHT)


@needs_ffmpeg
def test_render_frame_chains_scale_filter_when_size_differs(tmp_path: pathlib.Path) -> None:
    result = R.render_frame(RECT_ASS, 1500, tmp_path / "big.png", width=1280, height=960)
    assert result["scaled"] is True
    assert (result["width"], result["height"]) == (1280, 960)
    assert "scale=1280:960" in " ".join(result["ffmpeg_cmd"])


@needs_ffmpeg
def test_p1_rectangle_ink_bbox_matches_expected_pixels() -> None:
    """A filled p1 rectangle from (100,50) to (300,150) must measure as itself."""
    gray, w, h = R.render_gray(RECT_ASS, 1500, 640, 480)
    box = M.ink_bbox(gray, w, h)
    assert box is not None
    assert abs(box["x0"] - RECT_BOX[0]) <= 4
    assert abs(box["y0"] - RECT_BOX[1]) <= 4
    assert abs(box["x1"] - RECT_BOX[2]) <= 4
    assert abs(box["y1"] - RECT_BOX[3]) <= 4
    # 200 x 100 filled pixels.
    assert box["ink_pixels"] == pytest.approx(200 * 100, rel=0.05)


@needs_ffmpeg
def test_render_succeeds_in_directory_with_space_and_colon(tmp_path: pathlib.Path) -> None:
    """The escaping proof: a temp dir whose name has a space *and* a colon."""
    weird = tmp_path / "dir with space: and colon"
    weird.mkdir()
    ass_file = weird / "rect.ass"
    ass_file.write_text(RECT_ASS, encoding="utf-8")

    assert "\\\\:" in R.safe_filter_path(str(ass_file))
    result = R.render_frame(str(ass_file), 1500, weird / "frame.png")
    assert result["out_path"] == str(weird / "frame.png")
    assert pathlib.Path(result["out_path"]).is_file()

    gray, w, h = R.render_gray(str(ass_file), 1500, 640, 480)
    box = M.ink_bbox(gray, w, h)
    assert box is not None
    assert abs(box["x0"] - RECT_BOX[0]) <= 4 and abs(box["y1"] - RECT_BOX[3]) <= 4

    # Re-render *through* that same path to prove nothing silently fell back.
    assert M.measure_render(str(ass_file), 1500)["rect"]["width"] == pytest.approx(200, abs=4)


@needs_ffmpeg
def test_render_succeeds_in_directory_with_every_escaped_character(
    tmp_path: pathlib.Path,
) -> None:
    weird = tmp_path / "a,b;c[d]e's f:g"
    weird.mkdir()
    ass_file = weird / "stress.ass"
    ass_file.write_text(RECT_ASS, encoding="utf-8")
    gray, w, h = R.render_gray(str(ass_file), 1500, 640, 480)
    box = M.ink_bbox(gray, w, h)
    assert box is not None
    assert abs(box["x0"] - RECT_BOX[0]) <= 4
    assert abs(box["x1"] - RECT_BOX[2]) <= 4


@needs_ffmpeg
def test_render_gray_shape() -> None:
    gray, w, h = R.render_gray(RECT_ASS, 1500, 640, 480)
    assert (w, h) == (640, 480)
    assert len(gray) == 640 * 480
    assert isinstance(gray, bytes)
    # Corners are background, the middle of the rectangle is solid white.
    assert gray[0] == 0 and gray[-1] == 0
    assert gray[100 * 640 + 200] > 200


@needs_ffmpeg
def test_render_frames_single_invocation_matches_render_frame(tmp_path: pathlib.Path) -> None:
    """The one-pass select renderer must produce byte-identical frames."""
    times = [500, 1500, 3000]
    paths = R.render_frames(MULTI_ASS, times, tmp_path / "frames")
    assert len(paths) == 3
    assert all(os.path.isabs(p) and os.path.isfile(p) for p in paths)
    for time_ms, path in zip(times, paths):
        single = R.render_frame(MULTI_ASS, time_ms, tmp_path / f"single_{time_ms}.png")
        assert pathlib.Path(path).read_bytes() == pathlib.Path(single["out_path"]).read_bytes(), (
            f"frame at {time_ms} ms differs from the single-frame render"
        )


@needs_ffmpeg
def test_render_frames_maps_each_time_to_the_right_picture(tmp_path: pathlib.Path) -> None:
    """Guards the select() index mapping: each time must show its own event."""
    paths = R.render_frames(MULTI_ASS, [500, 1500, 3000], tmp_path / "mapped")
    boxes = []
    for path in paths:
        gray, w, h = _decode_gray(path)
        boxes.append(M.ink_bbox(gray, w, h))

    assert boxes[0] is None  # 0.5 s: nothing on screen yet
    assert boxes[1] is not None and abs(boxes[1]["x0"] - 100) <= 4  # rect A
    assert boxes[2] is not None and abs(boxes[2]["x0"] - 400) <= 4  # rect B
    assert boxes[1]["x0"] != boxes[2]["x0"]


@needs_ffmpeg
def test_render_frames_preserves_order_and_duplicates(tmp_path: pathlib.Path) -> None:
    paths = R.render_frames(MULTI_ASS, [3000, 1500, 1500], tmp_path / "ordered")
    assert len(paths) == 3
    assert paths[1] == paths[2]
    assert paths[0] != paths[1]
    first, second = (_decode_gray(pathlib.Path(p)) for p in (paths[0], paths[1]))
    assert M.ink_bbox(*second)["x0"] < M.ink_bbox(*first)["x0"]


@needs_ffmpeg
def test_render_frames_empty_input(tmp_path: pathlib.Path) -> None:
    assert R.render_frames(RECT_ASS, [], tmp_path / "none") == []


@needs_ffmpeg
def test_render_video_produces_a_probeable_preview(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "preview.mp4"
    result = R.render_video(RECT_ASS, out, fps=15)
    assert os.path.isfile(result["out_path"])
    assert result["codec"] == "libx264"
    assert result["duration_ms"] == pytest.approx(10000.0, abs=100)
    assert result["frames"] == pytest.approx(150, abs=5)
    assert (result["width"], result["height"]) == (640, 480)
    info = R.probe_video(out)
    assert info["has_video"] and info["codec"] == "h264"
    assert info["duration_s"] == pytest.approx(10.0, abs=0.2)


@needs_ffmpeg
def test_render_video_respects_explicit_range(tmp_path: pathlib.Path) -> None:
    result = R.render_video(RECT_ASS, tmp_path / "clip.mp4", start_ms=1000, end_ms=3000, fps=10)
    assert result["duration_ms"] == pytest.approx(2000.0, abs=1)
    assert result["frames"] == pytest.approx(20, abs=2)


# ---------------------------------------------------------------------------
# media probes
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> str:
    """1 s 320x240 @30fps h264 clip, keyframe every 0.5 s, tone from 0.2-0.4 s."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
        "-f", "lavfi", "-i", r"aevalsrc=0.6*sin(2*PI*440*t)*between(t\,0.2\,0.4):d=1:s=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "15",
        "-c:a", "aac", "-shortest", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


@needs_ffmpeg
def test_probe_video_fields(sample_video: str) -> None:
    info = R.probe_video(sample_video)
    assert (info["width"], info["height"]) == (320, 240)
    assert info["fps"] == pytest.approx(30.0, abs=0.01)
    assert info["duration_s"] == pytest.approx(1.0, abs=0.05)
    assert info["nb_frames"] == pytest.approx(30, abs=2)
    assert info["codec"] == "h264"
    assert info["pix_fmt"] == "yuv420p"
    assert "mp4" in (info["format"] or "")
    assert info["has_audio"] is True and info["has_video"] is True
    assert info["audio"][0]["codec"] == "aac"
    assert info["audio"][0]["channels"] == 1
    assert info["audio"][0]["sample_rate"] == 44100
    assert info["rotation"] is None


@needs_ffmpeg
def test_probe_video_reads_rotation_from_side_data(tmp_path: pathlib.Path) -> None:
    flat = tmp_path / "flat.mp4"
    subprocess.run(
        [
            R.ffmpeg_path(), "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(flat),
        ],
        check=True, capture_output=True,
    )
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            R.ffmpeg_path(), "-hide_banner", "-v", "error", "-y",
            "-display_rotation", "90", "-i", str(flat), "-c", "copy", str(rotated),
        ],
        check=True, capture_output=True,
    )
    assert R.probe_video(rotated)["rotation"] == pytest.approx(90.0)


@needs_ffmpeg
def test_keyframes_are_sane(sample_video: str) -> None:
    frames = R.keyframes(sample_video)
    assert frames, "expected at least one keyframe"
    assert frames == sorted(frames)
    assert len(frames) == len(set(frames))
    assert frames[0] == pytest.approx(0.0, abs=0.02)
    assert len(frames) >= 2, f"expected keyframes every 0.5 s, got {frames}"
    assert (frames[1] - frames[0]) == pytest.approx(0.5, abs=0.05)
    assert all(t >= 0 for t in frames)
    assert len(R.keyframes(sample_video, limit=1)) == 1


@needs_ffmpeg
def test_audio_peaks_finds_the_tone_and_the_silence(sample_video: str) -> None:
    peaks = R.audio_peaks(sample_video, interval_ms=50)
    assert peaks, "expected audio intervals"
    assert all(p["t_ms"] >= 0 for p in peaks)
    assert [p["t_ms"] for p in peaks] == sorted(p["t_ms"] for p in peaks)
    assert peaks[-1]["t_ms"] <= 1050
    assert all(p["samples"] > 0 for p in peaks)

    tone = [p for p in peaks if 200 <= p["t_ms"] < 400]
    assert tone, "expected intervals inside the 0.2-0.4 s tone gate"
    assert max(p["rms_db"] for p in tone) > -20.0
    assert max(p["peak_db"] for p in tone) > -15.0
    assert all(p["rms_db"] <= p["peak_db"] for p in peaks)

    silent = [p for p in peaks if p["t_ms"] < 150 or p["t_ms"] >= 450]
    assert silent
    assert max(p["rms_db"] for p in silent) < -40.0


@needs_ffmpeg
def test_audio_peaks_window_range(sample_video: str) -> None:
    peaks = R.audio_peaks(sample_video, interval_ms=100, start_ms=200, end_ms=400)
    assert 1 <= len(peaks) <= 4
    assert peaks[0]["t_ms"] == pytest.approx(200.0)
    assert max(p["rms_db"] for p in peaks) > -20.0


# ---------------------------------------------------------------------------
# silencedetect log parsing (pure unit, no ffmpeg)
# ---------------------------------------------------------------------------
def test_parse_silence_log_handles_open_interval() -> None:
    log = (
        "[silencedetect @ 0x55] silence_start: 0.401361\n"
        "[silencedetect @ 0x55] silence_end: 1.021678 | silence_duration: 0.620317\n"
        "[silencedetect @ 0x55] silence_start: 2.5\n"
    )
    assert R._parse_silence_log(log) == [
        {"start_ms": pytest.approx(401.361), "end_ms": pytest.approx(1021.678),
         "duration_ms": pytest.approx(620.317)},
        {"start_ms": pytest.approx(2500.0), "end_ms": None, "duration_ms": None},
    ]


def test_parse_silence_log_handles_missing_duration_and_garbage() -> None:
    log = "noise\n[silencedetect @ 0x55] silence_end: 3.0\n"
    parsed = R._parse_silence_log(log)
    assert len(parsed) == 1
    assert parsed[0]["end_ms"] == pytest.approx(3000.0)
    assert parsed[0]["duration_ms"] == pytest.approx(3000.0)
    assert R._parse_silence_log("") == []


@needs_ffmpeg
def test_silencedetect_reports_the_gap(sample_video: str) -> None:
    intervals = R.silencedetect(sample_video, noise_db=-45.0, min_duration_s=0.2)
    assert intervals, "expected at least one silence interval"
    first = intervals[0]
    assert 300 <= first["start_ms"] <= 520
    assert first["end_ms"] is not None
    assert first["duration_ms"] > 400
    assert first["duration_ms"] == pytest.approx(first["end_ms"] - first["start_ms"], abs=5)
    assert all(i["end_ms"] is None or i["end_ms"] >= i["start_ms"] for i in intervals)


# ---------------------------------------------------------------------------
# text measurement
# ---------------------------------------------------------------------------
@needs_ffmpeg
def test_measure_line_render_text_box_is_plausible() -> None:
    result = M.measure_line_render("Aj", {"Fontname": "DejaVu Sans", "Fontsize": 48}, (640, 480))
    assert result["empty"] is False
    assert result["bbox"] is not None
    rect = result["rect"]
    assert result["scale"] == [1.0, 1.0]
    assert rect["x"] >= 0 and rect["y"] >= 0
    # 48 px text is roughly 30-40 px tall including the 2 px outline.
    assert 20 <= rect["height"] <= 60
    assert 5 <= rect["width"] <= 60
    assert rect["x1"] > rect["x"]
    assert result["ink_pixels"] > 50
    assert "PlayResX: 640" in result["ass_text"]


@needs_ffmpeg
def test_measure_line_render_longer_text_is_wider() -> None:
    style = {"Fontname": "DejaVu Sans", "Fontsize": 48}
    short = M.measure_line_render("A", style, (640, 480))
    long = M.measure_line_render("AVERYLONGWORD", style, (640, 480))
    assert short["rect"]["width"] > 0 and long["rect"]["width"] > 0
    assert long["rect"]["width"] > short["rect"]["width"] * 2
    # Single line, so the height must stay in the same ballpark.
    assert abs(long["rect"]["height"] - short["rect"]["height"]) <= 8


@needs_ffmpeg
def test_measure_line_render_empty_text_has_no_ink() -> None:
    result = M.measure_line_render("", {"Fontname": "DejaVu Sans"}, (640, 480))
    assert result["empty"] is True
    assert result["bbox"] is None and result["rect"] is None


@needs_ffmpeg
def test_measure_render_full_document() -> None:
    result = M.measure_render(RECT_ASS, 1500)
    assert result["play_res"] == [640, 480]
    assert (result["render_width"], result["render_height"]) == (640, 480)
    assert result["scale"] == [1.0, 1.0]
    rect = result["rect"]
    assert abs(rect["x"] - RECT_BOX[0]) <= 4
    assert abs(rect["y"] - RECT_BOX[1]) <= 4
    assert abs(rect["x1"] - RECT_BOX[2]) <= 4
    assert abs(rect["y1"] - RECT_BOX[3]) <= 4
    assert 0.0 < result["coverage"] < 1.0


@needs_ffmpeg
def test_measure_render_scaled_maps_back_to_script_coordinates() -> None:
    result = M.measure_render(RECT_ASS, 1500, width=1280, height=960)
    assert (result["render_width"], result["render_height"]) == (1280, 960)
    assert result["scale"] == [0.5, 0.5]
    rect = result["rect"]
    # The rectangle is at the same script coordinates regardless of render size.
    assert abs(rect["x"] - RECT_BOX[0]) <= 5
    assert abs(rect["width"] - 200) <= 6


# ---------------------------------------------------------------------------
# font probing
# ---------------------------------------------------------------------------
@needs_fonts
def test_fonts_list_returns_entries() -> None:
    fonts = M.fonts_list()
    assert len(fonts) > 0
    sample = fonts[0]
    assert set(sample) >= {"family", "style", "file", "index", "family_raw"}
    assert sample["family"]
    assert os.path.exists(sample["file"])


@needs_fonts
def test_fonts_list_pattern_filters_case_insensitively() -> None:
    everything = M.fonts_list()
    match = M.fonts_list("dejavu")
    assert match, "expected the DejaVu family to be installed"
    assert all(
        "dejavu" in (f["family"] + f["style"] + f["file"]).casefold() for f in match
    )
    assert len(match) <= len(everything)
    assert M.fonts_list("zzzz-no-such-font-zzzz") == []
    assert len(M.fonts_list(limit=1)) == 1


@needs_fonts
def test_fonts_with_char_finds_thai_kokai() -> None:
    families = M.fonts_with_char("\u0e01")  # THAI CHARACTER KO KAI
    assert families, "expected at least one Thai-capable family on this host"
    assert all(isinstance(name, str) and name for name in families)
    assert families == sorted(families, key=str.casefold)
    # A plain ASCII letter is covered far more widely than Thai.
    assert len(M.fonts_with_char("A")) >= 1
    with pytest.raises(M.MeasureError):
        M.fonts_with_char("ab")


@needs_fonts
def test_fonts_match_detects_substitution() -> None:
    matches = M.fonts_match("DejaVu Sans", limit=3)
    assert matches
    assert matches[0]["family"] == "DejaVu Sans"
    assert matches[0]["substituted"] is False
    assert os.path.exists(matches[0]["file"])

    bogus = M.fonts_match("NoSuchFontFamily12345", limit=2)
    assert bogus
    assert bogus[0]["substituted"] is True
    assert bogus[0]["family"] != "NoSuchFontFamily12345"


@needs_fonts
def test_glyph_check_reports_missing_thai_characters() -> None:
    result = M.glyph_check("A\u0e01\u0e02", "DejaVu Sans", limit_fallbacks=3)
    assert result["font"] == "DejaVu Sans"
    assert result["checked"] == 3
    assert "A" in result["covered"]
    assert result["missing"] == ["\u0e01", "\u0e02"]
    for char in result["missing"]:
        assert 1 <= len(result["fallbacks"][char]) <= 3
    assert M.glyph_check("ABC", "DejaVu Sans")["missing"] == []


@needs_fonts
def test_glyph_check_uses_default_sans_when_no_font_given() -> None:
    result = M.glyph_check("A")
    assert result["font"]
    assert result["missing"] == []
    assert result["coverage_source"] in ("charset", "query")


@needs_fonts
def test_glyph_check_ignores_spaces() -> None:
    result = M.glyph_check("a b\tc", "DejaVu Sans")
    assert result["checked"] == 3
    assert result["missing"] == []


@needs_ffmpeg
def test_encoder_detection_drives_the_mpeg4_fallback() -> None:
    """render_video picks h264 when this ffmpeg has it, mpeg4 otherwise."""
    assert R._encoder_available(R.ffmpeg_path(), "libx264") is True
    assert R._encoder_available(R.ffmpeg_path(), "definitely-not-an-encoder") is False
