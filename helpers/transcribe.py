"""Transcribe a video locally with whisper.cpp (Metal GPU on Apple Silicon).

Extracts mono 16kHz audio via ffmpeg, runs `whisper-cli` with full JSON
output, merges tokens into words, then tightens each word's edges against
the audio energy envelope (whisper token times drift 50-150ms). Writes a
Scribe-compatible transcript to <edit_dir>/transcripts/<video_stem>.json:

    {"engine": "whisper.cpp", "words": [
        {"type": "word", "text": "Um,", "start": 0.0, "end": 0.19, "p": 0.32},
        {"type": "spacing", "text": " ", "start": 0.19, "end": 0.43},
        ...],
     "silences": [[0.19, 0.43], ...], "duration": 11.0, "source_sig": "..."}

Whisper normalizes fillers away by default. An initial prompt full of
disfluencies ("Umm, uh, ...") makes it transcribe them verbatim, which the
editor needs to find bad takes. No diarization: `speaker_id` is absent.

Cached: skipped if the transcript exists and the source file is unchanged.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --model ~/models/ggml-large-v3.bin

Env:
    VIDEO_EDIT_WHISPER_MODEL   path to a ggml model (default: large-v3-turbo)
    VIDEO_EDIT_WHISPER_BIN     whisper.cpp CLI (default: whisper-cli on PATH)
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np


MODEL_CANDIDATES = [
    Path.home() / ".claude-video-vision/models/ggml-large-v3-turbo.bin",
    Path("/opt/homebrew/share/whisper-cpp/models/ggml-large-v3-turbo.bin"),
    Path.home() / ".cache/whisper/ggml-large-v3-turbo.bin",
]

# Disfluent prompt: keeps "um", "uh", false starts and repeats in the output.
# Prompt wording matters: longer run-on prompts made whisper drop the opening
# "Um" and merge sentences. This one keeps fillers and sentence punctuation.
FILLER_PROMPT = "Um, uh, er, ah. Like, you know, I mean... So, um, yeah."

ENVELOPE_HOP_S = 0.01       # 10ms RMS frames
SNAP_END_WINDOW_S = 0.25    # how far past whisper's end to look for the true end
SNAP_START_WINDOW_S = 0.15  # how far before whisper's start to look for the onset
MIN_GAP_S = 0.04            # quiet shorter than this is a closure inside a word, not a gap


def find_model(explicit: Path | None) -> Path:
    if explicit:
        p = explicit.expanduser()
        if not p.exists():
            sys.exit(f"whisper model not found: {p}")
        return p
    env = os.environ.get("VIDEO_EDIT_WHISPER_MODEL")
    if env:
        return find_model(Path(env))
    for c in MODEL_CANDIDATES:
        if c.exists():
            return c
    sys.exit(
        "no whisper.cpp model found. Download one, e.g.\n"
        "  curl -L -o ~/.cache/whisper/ggml-large-v3-turbo.bin "
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin\n"
        "or set VIDEO_EDIT_WHISPER_MODEL."
    )


def find_whisper_bin() -> str:
    b = os.environ.get("VIDEO_EDIT_WHISPER_BIN") or shutil.which("whisper-cli")
    if not b:
        sys.exit("whisper-cli not found. Install with: brew install whisper-cpp")
    return b


def source_signature(video: Path) -> str:
    st = video.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# -------- whisper.cpp ---------------------------------------------------------


DTW_PRESETS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large.v1", "large.v2", "large.v3", "large.v3.turbo",
}


def dtw_preset(model: Path) -> str | None:
    """whisper.cpp DTW alignment-head preset for a model file, e.g.
    ggml-large-v3-turbo.bin -> large.v3.turbo. None for unknown/quantized names."""
    name = model.name.removeprefix("ggml-").removesuffix(".bin").replace("-", ".")
    for suffix in (".q5_0", ".q5_1", ".q8_0", ".q4_0"):
        name = name.removesuffix(suffix)
    return name if name in DTW_PRESETS else None


def run_whisper(audio: Path, model: Path, language: str | None, out_base: Path) -> dict:
    cmd = [
        find_whisper_bin(), "-np",
        "-m", str(model),
        "-f", str(audio),
        "-l", language or "auto",
        "-ojf", "-of", str(out_base),
        "--prompt", FILLER_PROMPT,
    ]
    preset = dtw_preset(model)
    if preset:
        # DTW token timing needs flash attention off.
        cmd += ["-nfa", "-dtw", preset]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out_json = out_base.with_suffix(".json")
    if res.returncode != 0 or not out_json.exists():
        raise RuntimeError(f"whisper-cli failed ({res.returncode}): {res.stderr[-800:]}")
    return json.loads(out_json.read_text())


def tokens_to_words(whisper_json: dict) -> list[dict]:
    """Merge whisper.cpp tokens into words.

    A token starting with a space opens a new word. Punctuation-only tokens
    are appended to the word's text but carry no timing.

    Each word gets `start`/`end` from the token offsets and, when DTW ran,
    an `anchor`: the DTW time of its last token, which lands near the end of
    the spoken word and drifts far less than the offsets do.
    """
    words: list[dict] = []
    for seg in whisper_json.get("transcription", []):
        for tok in seg.get("tokens", []):
            text = tok.get("text", "")
            if not text or text.startswith("[_") or text.startswith("<|"):
                continue
            t0 = tok["offsets"]["from"] / 1000.0
            t1 = tok["offsets"]["to"] / 1000.0
            dtw = tok.get("t_dtw", -1)
            anchor = dtw / 100.0 if isinstance(dtw, (int, float)) and dtw >= 0 else None
            p = float(tok.get("p", 1.0))
            is_punct = not any(ch.isalnum() for ch in text)
            if words and (is_punct or not text.startswith(" ")):
                w = words[-1]
                w["text"] += text.strip() if is_punct else text
                if not is_punct:
                    w["end"] = max(w["end"], t1)
                    w["p"] = min(w["p"], p)
                    if anchor is not None:
                        w["anchor"] = anchor
                continue
            if is_punct:
                continue
            words.append({"type": "word", "text": text.strip(), "start": t0,
                          "end": max(t1, t0), "p": p, "anchor": anchor})
    return words


# -------- energy envelope -----------------------------------------------------


def rms_envelope_db(wav_path: Path) -> np.ndarray:
    """10ms-hop RMS envelope in dBFS."""
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * ENVELOPE_HOP_S)
    n = len(data) // hop
    if n == 0:
        return np.array([-120.0])
    frames = data[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def silence_threshold_db(env: np.ndarray) -> float:
    """Adaptive threshold: halfway (in dB) between the noise floor and speech level."""
    floor = float(np.percentile(env, 10))
    speech = float(np.percentile(env, 90))
    return max(floor + 6.0, (floor + speech) / 2.0 - 6.0)


def find_silences(env: np.ndarray, thr: float, min_len_s: float = 0.12) -> list[list[float]]:
    quiet = env < thr
    out: list[list[float]] = []
    i = 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * ENVELOPE_HOP_S >= min_len_s:
                out.append([round(i * ENVELOPE_HOP_S, 3), round(j * ENVELOPE_HOP_S, 3)])
            i = j
        else:
            i += 1
    return out


def quiet_run_at(env: np.ndarray, k: int, thr: float, min_frames: int) -> bool:
    """True if a quiet run of at least `min_frames` starts at frame k."""
    end = k + min_frames
    return end <= len(env) and bool(np.all(env[k:end] < thr))


def refine_words(words: list[dict], env: np.ndarray, thr: float) -> None:
    """Place word edges on the audio.

    With DTW anchors (each near the end of its word), words are placed in
    order. A word starts at the beginning of the contiguous speech run that
    contains its anchor, found by walking back from the anchor: quiet runs
    shorter than MIN_GAP_S (a stop closure, the /t/ in "nights") do not break
    the run, a real pause does, and it never goes before the previous word's
    end. So neither a closure nor a pause before the word can misplace it. It ends at the first quiet run of at least
    MIN_GAP_S at/after its anchor, before the next anchor and within
    SNAP_END_WINDOW_S; with no such gap (connected speech) it ends at its
    anchor. An anchor that fell into trailing silence is pulled back to the
    speech offset.

    Without anchors: fall back to whisper's token offsets, nudged onto the
    nearest speech onset/offset.
    """
    n = len(env)
    total = n * ENVELOPE_HOP_S
    gap_frames = max(1, int(round(MIN_GAP_S / ENVELOPE_HOP_S)))

    def idx(t: float) -> int:
        return min(max(int(round(t / ENVELOPE_HOP_S)), 0), n - 1)

    have_anchors = words and all(w.get("anchor") is not None for w in words)
    if have_anchors:
        for i in range(1, len(words)):  # anchors must be non-decreasing
            words[i]["anchor"] = max(words[i]["anchor"], words[i - 1]["anchor"])
        prev_end = 0
        for i, w in enumerate(words):
            hi = max(idx(w["anchor"]), prev_end)
            nxt = idx(words[i + 1]["anchor"]) if i + 1 < len(words) else n
            limit = min(idx(min(w["anchor"] + SNAP_END_WINDOW_S, total)), nxt)
            # start: walk back from the anchor to the start of its speech run
            k = hi
            while k > prev_end and env[k] < thr:  # anchor in trailing quiet
                k -= 1
            if env[k] < thr:  # nothing spoken between prev word and anchor:
                k = hi        # anchor landed before the onset, look ahead
                while k < limit and env[k] < thr:
                    k += 1
            else:
                while k > prev_end:
                    if env[k - 1] >= thr:
                        k -= 1
                        continue
                    q = 0  # length of the quiet run ending at k-1
                    while k - 1 - q >= prev_end and env[k - 1 - q] < thr:
                        q += 1
                    if q >= gap_frames or k - q <= prev_end:
                        break  # a real pause (or the previous word): stop
                    k -= q     # a closure inside the word: step over it
            start = k
            # end
            e = max(hi, start)
            if e > start and env[e - 1] < thr:  # anchor sits in trailing silence
                while e > start + 1 and env[e - 1] < thr:
                    e -= 1
            else:
                f = e
                while f < limit and not quiet_run_at(env, f, thr, gap_frames):
                    f += 1
                e = f if f < limit else e
            e = max(e, start + 1)
            w["start"] = start * ENVELOPE_HOP_S
            w["end"] = e * ENVELOPE_HOP_S
            prev_end = e
    else:
        for i, w in enumerate(words):
            prev_end = words[i - 1]["end"] if i > 0 else 0.0
            next_start = words[i + 1]["start"] if i + 1 < len(words) else total
            e = idx(w["end"])
            limit = idx(min(w["end"] + SNAP_END_WINDOW_S, next_start))
            while e < limit and env[e] >= thr:
                e += 1
            w["end"] = max(w["end"], e * ENVELOPE_HOP_S)
            s = idx(w["start"])
            floor_i = idx(max(w["start"] - SNAP_START_WINDOW_S, prev_end))
            while s > floor_i and env[s - 1] >= thr:
                s -= 1
            w["start"] = min(w["start"], s * ENVELOPE_HOP_S)

    # Monotonic, non-overlapping, rounded
    for i, w in enumerate(words):
        if i > 0 and w["start"] < words[i - 1]["end"]:
            words[i - 1]["end"] = max(words[i - 1]["start"], w["start"])
        w["start"] = round(w["start"], 3)
        w["end"] = round(max(w["end"], w["start"]), 3)
        w.pop("anchor", None)


def transcript_qc(words: list[dict], silences: list[list[float]]) -> dict:
    """Flag timing the editor should not trust: words squeezed to <= 20ms,
    words whose start sits inside a detected silence, and words that span a
    pause of >= 150ms (a swallowed gap: a cut there would chop the word)."""
    tiny = [w["text"] for w in words if w["end"] - w["start"] < 0.025]  # float-safe "<= 20ms"
    in_silence = [w["text"] for w in words
                  if any(a + 0.02 < w["start"] < b - 0.02 for a, b in silences)]
    spans_pause = [w["text"] for w in words
                   if any(b - a >= 0.15 and w["start"] < a and b < w["end"] for a, b in silences)]
    return {"tiny_words": tiny, "starts_in_silence": in_silence, "spans_pause": spans_pause,
            "low_confidence": [w["text"] for w in words if w["p"] < 0.4]}


def with_spacing(words: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, w in enumerate(words):
        if i > 0 and w["start"] > words[i - 1]["end"]:
            out.append({"type": "spacing", "text": " ",
                        "start": words[i - 1]["end"], "end": w["start"]})
        out.append(w)
    return out


# -------- public API ----------------------------------------------------------


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands. Track 0 keeps the plain name."""
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def is_cached(out_path: Path, video: Path) -> bool:
    if not out_path.exists():
        return False
    try:
        return json.loads(out_path.read_text()).get("source_sig") == source_signature(video)
    except Exception:
        return False


def transcribe_one(
    video: Path,
    edit_dir: Path,
    language: str | None = None,
    model: Path | None = None,
    verbose: bool = True,
    audio_track: int = 0,
    **_ignored,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON."""
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)

    if is_cached(out_path, video):
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    model_path = find_model(model)
    n_tracks = count_audio_tracks(video)
    if n_tracks == 0:
        raise RuntimeError(f"{video.name} has no audio track")
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio.wav"
        extract_audio(video, audio, audio_track)

        peak = peak_dbfs(audio)
        if peak < -60.0:
            raise RuntimeError(
                f"track {audio_track + 1} of {video.name} is silent (peak {peak:.1f} dBFS). "
                + (f"The file has {n_tracks} audio tracks; try --audio-track." if n_tracks > 1
                   else "Check the source audio.")
            )

        if verbose:
            print(f"  whisper.cpp ({model_path.name}) on {video.name}", flush=True)
        wj = run_whisper(audio, model_path, language, Path(tmp) / "out")
        env = rms_envelope_db(audio)

    thr = silence_threshold_db(env)
    words = tokens_to_words(wj)
    refine_words(words, env, thr)
    payload = {
        "engine": "whisper.cpp",
        "model": model_path.name,
        "language": (wj.get("result") or {}).get("language", language),
        "duration": round(len(env) * ENVELOPE_HOP_S, 3),
        "source": str(video),
        "source_sig": source_signature(video),
        "silence_threshold_db": round(thr, 1),
        "text": " ".join(w["text"] for w in words),
        "words": with_spacing(words),
        "silences": find_silences(env, thr),
    }
    payload["qc"] = transcript_qc(words, payload["silences"])
    out_path.write_text(json.dumps(payload, indent=1))

    if verbose:
        qc = payload["qc"]
        print(f"  saved: {out_path.name} — {len(words)} words, "
              f"{len(qc['low_confidence'])} low-confidence, {time.time() - t0:.1f}s")
        bad = len(qc["tiny_words"]) + len(qc["starts_in_silence"]) + len(qc["spans_pause"])
        if bad:
            print(f"  QC: {len(qc['tiny_words'])} words <=20ms, "
                  f"{len(qc['starts_in_silence'])} starting inside silence, "
                  f"{len(qc['spans_pause'])} spanning a pause — "
                  f"check those cut points with timeline_view before trusting them")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video locally with whisper.cpp")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit output directory (default: <video_parent>/edit)")
    ap.add_argument("--language", type=str, default=None,
                    help="ISO language code (e.g. 'en'). Omit to auto-detect.")
    ap.add_argument("--model", type=Path, default=None, help="Path to a ggml whisper model")
    ap.add_argument("--audio-track", type=int, default=0,
                    help="Zero-based audio track to transcribe (OBS: 0 = game, 1 = mic).")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    transcribe_one(video, edit_dir, language=args.language, model=args.model,
                   audio_track=args.audio_track)


if __name__ == "__main__":
    main()
