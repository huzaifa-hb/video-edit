"""Detect shot changes in a video with OpenCV — the visual counterpart of the
transcript, for footage where speech doesn't tell you the structure: b-roll,
product shots, montages, screen recordings, multicam exports.

Compares HSV colour histograms of consecutive frames (at reduced size); a
sharp drop in similarity is a cut. Also flags near-black frames (fades,
lens cap) and very blurry shots (low Laplacian variance), which are usually
unusable in an ad.

Writes <edit>/shots/<video_stem>.json and prints one line per shot:

    [000.00-003.40]  3.40s  bright 0.52  sharp 412
    [003.40-007.93]  4.53s  bright 0.08  sharp  35   DARK BLURRY

Usage:
    python helpers/scenes.py <video>
    python helpers/scenes.py <video> --threshold 0.45 --min-shot 0.5
    python helpers/scenes.py <video> --edit-dir /custom/edit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SMALL_W = 160
DARK = 0.06          # mean luma below this (0-1) → DARK
BLURRY = 60.0        # Laplacian variance below this → BLURRY (at SMALL_W*2 width)


def histogram(frame_small: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def detect(video: Path, threshold: float, min_shot: float) -> tuple[list[dict], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cuts = [0.0]
    stats: list[tuple[float, float, float]] = []  # (t, luma, sharpness)
    prev = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (SMALL_W, max(1, int(h * SMALL_W / w))))
        cur = histogram(small)
        if prev is not None:
            diff = 1.0 - cv2.compareHist(prev, cur, cv2.HISTCMP_CORREL)
            if diff > threshold and t - cuts[-1] >= min_shot:
                cuts.append(t)
        prev = cur
        if i % max(1, int(round(fps / 5))) == 0:  # ~5 quality samples per second
            gray = cv2.cvtColor(cv2.resize(frame, (SMALL_W * 2, max(1, int(h * SMALL_W * 2 / w)))),
                                cv2.COLOR_BGR2GRAY)
            stats.append((t, float(gray.mean()) / 255.0, float(cv2.Laplacian(gray, cv2.CV_64F).var())))
        i += 1
    cap.release()
    duration = i / fps
    cuts.append(duration)

    shots: list[dict] = []
    for a, b in zip(cuts, cuts[1:]):
        inside = [s for s in stats if a <= s[0] < b] or [min(stats, key=lambda s: abs(s[0] - a))] if stats else []
        luma = float(np.median([s[1] for s in inside])) if inside else 0.0
        sharp = float(np.median([s[2] for s in inside])) if inside else 0.0
        flags = [f for f, bad in (("DARK", luma < DARK), ("BLURRY", sharp < BLURRY)) if bad]
        shots.append({"start": round(a, 3), "end": round(b, 3), "brightness": round(luma, 3),
                      "sharpness": round(sharp, 1), "flags": flags})
    return shots, duration


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect shot changes with OpenCV")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None,
                    help="Edit directory (default: <video_parent>/edit)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Histogram dissimilarity that counts as a cut, 0-2 (default 0.5; lower = more cuts)")
    ap.add_argument("--min-shot", type=float, default=0.4, help="Shortest shot in seconds (default 0.4)")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    edit_dir = (args.edit_dir or video.parent / "edit").resolve()
    shots, duration = detect(video, args.threshold, args.min_shot)

    out = edit_dir / "shots" / f"{video.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": str(video), "duration": round(duration, 3), "shots": shots}, indent=1))

    print(f"## {video.stem}  ({duration:.1f}s, {len(shots)} shots) → {out}")
    for s in shots:
        print(f"  [{s['start']:06.2f}-{s['end']:06.2f}] {s['end'] - s['start']:5.2f}s  "
              f"bright {s['brightness']:.2f}  sharp {s['sharpness']:5.0f}   {' '.join(s['flags'])}")


if __name__ == "__main__":
    main()
