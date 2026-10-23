"""Tests for the pieces added in this fork: whisper token merging, word-edge
refinement, EDL validation/snapping, and caption cue grouping."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))

import captions  # noqa: E402
import transcribe  # noqa: E402
import validate_edl  # noqa: E402


def tok(text, a, b, dtw=-1, p=0.99):
    return {"text": text, "offsets": {"from": a, "to": b}, "t_dtw": dtw, "p": p}


def test_tokens_merge_into_words_and_punct_has_no_timing():
    wj = {"transcription": [{"tokens": [
        tok("[_BEG_]", 0, 0), tok(" Um", 0, 150, 14), tok(",", 150, 380, 38),
        tok(" pipe", 430, 600, 55), tok("line", 600, 800, 80), tok(".", 800, 990, 99),
    ]}]}
    words = transcribe.tokens_to_words(wj)
    assert [w["text"] for w in words] == ["Um,", "pipeline."]
    assert words[0]["end"] == 0.15          # comma's 0.38 not taken
    assert words[1]["anchor"] == 0.80       # DTW of the last real token


def test_dtw_preset_names():
    assert transcribe.dtw_preset(Path("ggml-large-v3-turbo.bin")) == "large.v3.turbo"
    assert transcribe.dtw_preset(Path("ggml-medium.en-q5_0.bin")) == "medium.en"
    assert transcribe.dtw_preset(Path("ggml-custom.bin")) is None


def test_refine_moves_sentence_start_out_of_silence():
    # speech 0-1.0s, silence 1.0-1.5s, speech 1.5-2.5s (10ms frames)
    env = np.full(250, -20.0)
    env[100:150] = -70.0
    thr = -45.0
    words = [
        {"text": "over.", "start": 0.5, "end": 0.9, "anchor": 0.95},
        {"text": "This", "start": 0.92, "end": 1.6, "anchor": 1.7},  # whisper put it in the gap
    ]
    transcribe.refine_words(words, env, thr)
    assert words[0]["end"] == 1.0
    assert words[1]["start"] == 1.5


def words(*spec):
    return [{"type": "word", "text": t, "start": a, "end": b} for t, a, b in spec]


W = words(("one", 0.0, 0.4), ("two", 0.6, 1.0), ("three", 1.3, 1.8))


def test_snap_start_inside_word():
    assert validate_edl.snap_start(W, 0.65, 0.06) == 0.54   # keep "two", pad 60ms
    assert validate_edl.snap_start(W, 0.95, 0.06) == 1.0    # mostly past "two": drop it, floor at its end


def test_snap_end_inside_word_and_never_into_next():
    assert validate_edl.snap_end(W, 0.95, 0.10, None) == 1.1
    assert validate_edl.snap_end(W, 1.0, 0.5, None) == 1.3  # pad capped at next word


def test_validate_flags_bad_edl(tmp_path):
    edl = {"sources": {"a": str(tmp_path / "missing.mp4")},
           "ranges": [{"source": "b", "start": 0, "end": 1},
                      {"source": "a", "start": 2, "end": 1}]}
    errors, _ = validate_edl.validate(edl, tmp_path, fix=False, pad_in=0.06, pad_out=0.1)
    text = "\n".join(errors)
    assert "file not found" in text
    assert "'b' not in sources" in text
    assert "end 1.00 <= start 2.00" in text


def test_caption_cues_never_cross_a_cut():
    ws = [
        {"text": "Hello", "start": 0.0, "end": 0.3, "seg_end": 0.5},
        {"text": "there", "start": 0.3, "end": 0.45, "seg_end": 0.5},
        {"text": "new", "start": 0.5, "end": 0.7, "seg_end": 2.0},
        {"text": "segment", "start": 0.7, "end": 1.0, "seg_end": 2.0},
    ]
    cues = captions.group_words(ws, per_cue=3, max_words=3)
    assert [[w["text"] for w in c["words"]] for c in cues] == [["Hello", "there"], ["new", "segment"]]
    assert cues[0]["end"] <= 0.5


def test_closure_dip_inside_word_does_not_delay_its_start():
    # "Risk": speech 1.0-1.2, a 30ms /k/ closure 1.12-1.15, then "free" 1.25-1.6.
    env = np.full(200, -70.0)
    env[100:160] = -20.0
    env[112:115] = -70.0
    env[125:160] = -20.0
    env[120:125] = -70.0
    words = [
        {"text": "Risk", "start": 1.1, "end": 1.2, "anchor": 1.18},
        {"text": "free.", "start": 1.2, "end": 1.6, "anchor": 1.55},
    ]
    transcribe.refine_words(words, env, -45.0)
    assert words[0]["start"] == 1.0          # not 1.15 (after the closure)
    assert words[0]["end"] == 1.2
    assert words[1]["start"] == 1.25


def test_segment_frames_rounds_to_whole_frames():
    import render
    assert render.segment_frames(1.61, "30/1") == 48
    assert render.segment_frames(1.0, "30000/1001") == 30


def test_render_timeline_equals_edl_arithmetic(tmp_path):
    """Many short cuts must not drift: rendered length == sum of whole-frame ranges."""
    import json
    import subprocess
    from fractions import Fraction

    import render

    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
         "-f", "lavfi", "-i", "sine=f=300:d=6", "-t", "6", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    ranges = [{"source": "src", "start": 0.11 + i * 0.5, "end": 0.11 + i * 0.5 + 0.37} for i in range(10)]
    edl = {"sources": {"src": str(src)}, "ranges": ranges}
    edl_path = tmp_path / "edl.json"
    edl_path.write_text(json.dumps(edl))
    out = tmp_path / "out.mp4"
    subprocess.run([sys.executable, str(Path(render.__file__)), str(edl_path), "-o", str(out),
                    "--draft", "--no-loudnorm"], check=True, capture_output=True)
    expected = sum(render.segment_frames(r["end"] - r["start"], "30/1") for r in ranges) / 30
    base = tmp_path / "base_draft.mov"
    for path in (base, out):
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True).stdout)
        assert abs(dur - expected) <= float(Fraction(1, 30)) + 0.03, (path.name, dur, expected)


def test_pause_before_word_is_not_swallowed_by_release_burst():
    # "it," speech 0.5-0.8 with a closure 0.75-0.77 and a release burst 0.77-0.80,
    # then a 900ms pause, then "I" 1.70-1.85.
    env = np.full(250, -70.0)
    env[50:75] = -20.0
    env[77:80] = -30.0
    env[170:185] = -20.0
    words = [
        {"text": "it,", "start": 0.5, "end": 0.75, "anchor": 0.74, "p": 0.9},
        {"text": "I", "start": 0.8, "end": 1.85, "anchor": 1.80, "p": 0.9},
    ]
    transcribe.refine_words(words, env, -45.0)
    assert words[1]["start"] == 1.7
    qc = transcribe.transcript_qc(words, transcribe.find_silences(env, -45.0))
    assert qc["spans_pause"] == []


def test_quantize_never_rounds_an_edge_into_the_next_word(tmp_path):
    import subprocess
    src = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=30",
                    "-f", "lavfi", "-i", "sine=d=8", "-t", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(src)], check=True)
    tr = tmp_path / "transcripts"
    tr.mkdir()
    import json
    tr.joinpath("s.json").write_text(json.dumps({"words": words(
        ("a", 4.0, 4.5), ("hundred", 4.82, 5.04))}))
    edl = {"sources": {"s": str(src)}, "ranges": [{"source": "s", "start": 3.9, "end": 4.9}]}  # first half of "hundred"
    errors, warns = validate_edl.validate(edl, tmp_path, fix=True, pad_in=0.06, pad_out=0.1, fps="30/1")
    end = edl["ranges"][0]["end"]
    assert end <= 4.82, end


def test_reframe_crop_offsets_centre_the_face():
    import reframe
    # 1920x1080 → 1080x1920: scaled to 3413x1920, 2333px of horizontal slack.
    cx, cy = reframe.crop_offsets(0.23, 0.54, 1920, 1080, 1080, 1920, 0.4)
    assert cy == 0.5                       # no vertical slack
    face_px = 0.23 * 1920 * (1920 / 1080)  # face x in the scaled frame
    left = (1920 * 1920 / 1080 - 1080) * cx
    assert abs((face_px - left) - 540) < 2  # lands in the middle of the crop
    assert reframe.crop_offsets(0.01, 0.5, 1920, 1080, 1080, 1920, 0.4)[0] == 0.0  # clamped
    assert reframe.crop_offsets(0.5, 0.5, 1920, 1080, 1920, 1080, 0.4) == (0.5, 0.5)  # same aspect
