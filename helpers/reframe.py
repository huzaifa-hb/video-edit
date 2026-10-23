"""Face-aware reframing: set each range's crop_x / crop_y so the speaker stays
in frame when 16:9 footage is cut to 9:16, 4:5 or 1:1.

Samples frames across every EDL range, finds the largest face with OpenCV's
YuNet detector (models/face_detection_yunet_2023mar.onnx, MIT), and converts
the median face position into the crop offsets render.py uses: face centred
horizontally, eyes-line at --headroom of the frame height from the top.

Needs an EDL `canvas`. Ranges with "crop_lock": true are left alone.

Usage:
    python helpers/reframe.py <edl.json>            # report only
    python helpers/reframe.py <edl.json> --apply    # write crop_x/crop_y (keeps .bak)
    python helpers/reframe.py <edl.json> --apply --headroom 0.35 --samples 12

Report flags:
    no face     nothing detected; crop left as is (product shot, b-roll, back of head)
    moves       the face travels more than ~1/4 of the crop window inside the
                range: a single static crop will lose it, split the range
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide a harmless DNN backend warning
import cv2  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 640          # frames are downscaled to this width for detection
MIN_SCORE = 0.7


def resolve(maybe: str, base: Path) -> Path:
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def crop_offsets(cx: float, cy: float, src_w: int, src_h: int,
                 canvas_w: int, canvas_h: int, headroom: float) -> tuple[float, float]:
    """Face centre (0-1 of the source frame) → render.py crop_x / crop_y (0-1).

    render.py scales the source to cover the canvas, then crops at
    ((scaled_w - canvas_w) * crop_x, (scaled_h - canvas_h) * crop_y).
    """
    scale = max(canvas_w / src_w, canvas_h / src_h)
    sw, sh = src_w * scale, src_h * scale

    def solve(pos: float, scaled: float, window: float, target: float) -> float:
        slack = scaled - window
        if slack <= 1:
            return 0.5
        return round(float(min(max((pos * scaled - window * target) / slack, 0.0), 1.0)), 3)

    return solve(cx, sw, canvas_w, 0.5), solve(cy, sh, canvas_h, headroom)


def detect_faces(video: Path, start: float, end: float, samples: int,
                 detector) -> tuple[list[tuple[float, float]], int, int]:
    """Centres (0-1) of the largest confident face in each sampled frame."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video}")
    centres: list[tuple[float, float]] = []
    w = h = 0
    n = max(1, samples)
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        s = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, max(1, int(round(h * s)))))
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = detector.detect(small)
        if faces is None or len(faces) == 0:
            continue
        faces = [f for f in faces if f[-1] >= MIN_SCORE]
        if not faces:
            continue
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
        centres.append((float(x + fw / 2) / small.shape[1], float(y + fh / 2) / small.shape[0]))
    cap.release()
    return centres, w, h


def main() -> None:
    ap = argparse.ArgumentParser(description="Face-aware crop offsets for an EDL")
    ap.add_argument("edl", type=Path)
    ap.add_argument("--apply", action="store_true", help="Write crop_x/crop_y into the EDL")
    ap.add_argument("--samples", type=int, default=8, help="Frames sampled per range (default 8)")
    ap.add_argument("--headroom", type=float, default=0.40,
                    help="Where the face centre sits, as a fraction from the top (default 0.40)")
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    edit_dir = edl_path.parent
    edl = json.loads(edl_path.read_text())
    canvas = edl.get("canvas")
    if not canvas:
        sys.exit("EDL has no `canvas` — reframing only applies when cutting to a new aspect ratio")
    cw, ch = int(canvas["width"]), int(canvas["height"])
    if not MODEL.exists():
        sys.exit(f"face model missing: {MODEL} (re-run setup.sh)")
    detector = cv2.FaceDetectorYN.create(str(MODEL), "", (DETECT_WIDTH, DETECT_WIDTH), MIN_SCORE)

    changed = 0
    for i, r in enumerate(edl["ranges"]):
        tag = f"range[{i}] {r.get('beat', '')}".rstrip()
        if r.get("crop_lock"):
            print(f"{tag}: crop_lock, skipped")
            continue
        src = resolve(edl["sources"][r["source"]], edit_dir)
        centres, w, h = detect_faces(src, float(r["start"]), float(r["end"]), args.samples, detector)
        if not centres:
            print(f"{tag}: no face — crop left at x={r.get('crop_x', 0.5)} y={r.get('crop_y', 0.5)}")
            continue
        cx = statistics.median(c[0] for c in centres)
        cy = statistics.median(c[1] for c in centres)
        crop_x, crop_y = crop_offsets(cx, cy, w, h, cw, ch, args.headroom)
        # How far does the face wander, relative to the visible window?
        scale = max(cw / w, ch / h)
        spread = (max(c[0] for c in centres) - min(c[0] for c in centres)) * w * scale / cw
        note = "  MOVES — split this range or check with timeline_view" if spread > 0.25 else ""
        print(f"{tag}: face in {len(centres)}/{args.samples} frames at ({cx:.2f},{cy:.2f}) "
              f"→ crop_x={crop_x} crop_y={crop_y}{note}")
        if args.apply:
            r["crop_x"], r["crop_y"] = crop_x, crop_y
            changed += 1

    if args.apply:
        shutil.copy2(edl_path, edl_path.with_suffix(".json.bak"))
        edl_path.write_text(json.dumps(edl, indent=2))
        print(f"updated {changed} range(s) in {edl_path.name} (backup: {edl_path.name}.bak)")


if __name__ == "__main__":
    main()
