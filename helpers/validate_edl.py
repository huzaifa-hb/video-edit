"""Check an EDL before rendering, and optionally fix what can be fixed.

LLM-written EDLs fail in predictable ways: invented source names, times past
the end of the file, cuts that land inside a word, edges with no breathing
room, the same line used twice, overlays that point at files that were never
rendered. This catches all of them before ffmpeg spends minutes on a render.

Checks (E = error, blocks render; W = warning):
  E  source id missing from `sources`, or file missing on disk
  E  start >= end, start < 0, end > source duration
  E  overlay / captions / subtitles file missing, overlay past the output end
  E  canvas not positive even numbers
  E  a range's source has no transcript while the EDL wants `captions`
  W  cut edge inside a word            (--fix snaps it to the word gap)
  W  cut edge with < 30ms of silence on either side (kept word clipped by
     the fade, or the dropped neighbour bleeding in) — also after --fix,
     because words that touch leave no gap to pad into
  W  range length not a whole number of frames (--fix rounds the end)
  W  segment shorter than 0.4s
  W  two ranges reuse the same source time
  W  stated total_duration_s disagrees with the ranges (--fix rewrites it)

Snapping uses the word-level transcript in <edit>/transcripts/<source>.json.
A start inside a word moves to that word's start if the cut is in its first
half (keep the word), else to its end (drop it); ends mirror this. Then the
edge is padded into the silence by --pad-in / --pad-out, never past the
neighbouring word.

Usage:
    python helpers/validate_edl.py <edl.json>            # report only
    python helpers/validate_edl.py <edl.json> --fix      # rewrite edl.json (keeps edl.json.bak)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from render import resolve_render_fps, segment_frames  # same directory

MIN_PAD = 0.03
MAX_PAD = 0.20
MIN_SEGMENT = 0.4


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def resolve(maybe: str, base: Path, also_cwd: bool = False) -> Path:
    """Resolve exactly as render.py does: relative to the EDL's directory
    (sources, overlays), plus the current directory for captions/subtitles."""
    p = Path(maybe)
    if p.is_absolute():
        return p
    if also_cwd and not (base / p).exists() and (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()
    return (base / p).resolve()


def load_words(edit_dir: Path, src: str) -> list[dict] | None:
    tr = edit_dir / "transcripts" / f"{src}.json"
    if not tr.exists():
        return None
    data = json.loads(tr.read_text())
    return [w for w in data.get("words", []) if w.get("type") == "word"
            and w.get("start") is not None and w.get("end") is not None]


def word_at(words: list[dict], t: float) -> int | None:
    for i, w in enumerate(words):
        if w["start"] < t < w["end"]:
            return i
    return None


def snap_start(words: list[dict], t: float, pad: float) -> float:
    i = word_at(words, t)
    if i is not None:
        w = words[i]
        t = w["start"] if t - w["start"] < w["end"] - t else w["end"]
    nxt = next((w for w in words if w["start"] >= t - 1e-6), None)
    prev = next((w for w in reversed(words) if w["end"] <= t + 1e-6), None)
    if nxt is None:
        return t
    floor = prev["end"] if prev else 0.0
    # aim `pad` before the first kept word, never into the previous word
    return round(max(floor, min(t, nxt["start"] - pad)), 3)


def snap_end(words: list[dict], t: float, pad: float, src_dur: float | None) -> float:
    i = word_at(words, t)
    if i is not None:
        w = words[i]
        t = w["end"] if w["end"] - t < t - w["start"] else w["start"]
    prev = next((w for w in reversed(words) if w["end"] <= t + 1e-6), None)
    nxt = next((w for w in words if w["start"] >= t - 1e-6), None)
    if prev is None:
        return t
    ceil = nxt["start"] if nxt else (src_dur if src_dur else prev["end"] + pad)
    return round(min(ceil, max(t, prev["end"] + pad)), 3)


def edge_gaps(words: list[dict], start: float, end: float) -> tuple[float | None, ...]:
    """(silence before start, lead into first kept word, tail after last kept
    word, silence after end). None where there is no neighbouring word."""
    inside = [w for w in words if w["end"] > start and w["start"] < end]
    prev = next((w for w in reversed(words) if w["end"] <= start + 1e-6), None)
    nxt = next((w for w in words if w["start"] >= end - 1e-6), None)
    lead = inside[0]["start"] - start if inside else None
    tail = end - inside[-1]["end"] if inside else None
    before = start - prev["end"] if prev else None
    after = nxt["start"] - end if nxt else None
    return before, lead, tail, after


def validate(edl: dict, edit_dir: Path, fix: bool, pad_in: float, pad_out: float,
             fps: str | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []
    sources = edl.get("sources") or {}
    ranges = edl.get("ranges") or []
    if not ranges:
        errors.append("EDL has no ranges")

    canvas = edl.get("canvas")
    if canvas:
        try:
            w, h = int(canvas["width"]), int(canvas["height"])
            if w <= 0 or h <= 0 or w % 2 or h % 2:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"canvas must be {{width, height}} positive even ints, got {canvas}")

    durations: dict[str, float | None] = {}
    for sid, path in sources.items():
        p = resolve(path, edit_dir)
        if not p.exists():
            errors.append(f"source {sid}: file not found: {p}")
            durations[sid] = None
        else:
            durations[sid] = probe_duration(p)

    used: dict[str, list[tuple[float, float, int]]] = {}
    for i, r in enumerate(ranges):
        tag = f"range[{i}] {r.get('beat', '')}".rstrip()
        sid = r.get("source")
        if sid not in sources:
            errors.append(f"{tag}: source '{sid}' not in sources {list(sources)}")
            continue
        try:
            start, end = float(r["start"]), float(r["end"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{tag}: start/end missing or not numbers")
            continue
        dur = durations.get(sid)
        # Judge the requested end before --fix clamps anything to the file's end.
        if dur is not None and end > dur + 0.01:
            errors.append(f"{tag}: end {end:.2f} is past the end of {sid} ({dur:.2f}s) — wrong source?")
            continue

        def quantize() -> None:
            # Whole output frames, so EDL arithmetic == rendered timeline.
            nonlocal end
            if not fps or end <= start:
                return
            n = segment_frames(end - start, fps)
            exact_end = round(start + float(n / Fraction(fps)), 3)
            if words and exact_end > end and n > 1:
                # Rounding up must not carry the edge into the next word.
                nxt = next((w for w in words if w["start"] >= end - 1e-6), None)
                if word_at(words, exact_end) is not None or (nxt and exact_end > nxt["start"]):
                    n -= 1
                    exact_end = round(start + float(n / Fraction(fps)), 3)
            if abs(exact_end - end) > 0.0015:
                if fix:
                    r["end"] = end = exact_end
                else:
                    warns.append(f"{tag}: length is not whole frames at {fps} fps "
                                 f"(render will use {n} frames)")

        words = load_words(edit_dir, sid)
        if not words:
            quantize()
        if words:
            if word_at(words, start) is not None:
                warns.append(f"{tag}: start {start:.2f} is inside the word "
                             f"'{words[word_at(words, start)]['text']}'")
            if word_at(words, end) is not None:
                warns.append(f"{tag}: end {end:.2f} is inside the word "
                             f"'{words[word_at(words, end)]['text']}'")
            if fix:
                new_start = snap_start(words, start, pad_in)
                new_end = snap_end(words, end, pad_out, dur)
                if (new_start, new_end) != (start, end):
                    warns.append(f"{tag}: fixed {start:.2f}-{end:.2f} → {new_start:.2f}-{new_end:.2f}")
                    r["start"], r["end"] = new_start, new_end
                    start, end = new_start, new_end
                quantize()
            else:
                quantize()
            before, lead, tail, after = edge_gaps(words, start, end)
            if lead is not None and lead < MIN_PAD:
                warns.append(f"{tag}: only {lead * 1000:.0f}ms before the first kept word — "
                             "its onset sits in the fade")
            if before is not None and before < MIN_PAD:
                warns.append(f"{tag}: start is {before * 1000:.0f}ms after the previous word — "
                             "it may bleed in; prefer a cut at a pause")
            if tail is not None and tail < MIN_PAD:
                warns.append(f"{tag}: only {tail * 1000:.0f}ms after the last kept word — "
                             "its tail sits in the fade")
            if after is not None and after < MIN_PAD:
                warns.append(f"{tag}: end is {after * 1000:.0f}ms before the next word — "
                             "it may bleed in; prefer a cut at a pause")
        elif words is None:
            msg = (f"{tag}: no transcript for source id '{sid}' (expected "
                   f"transcripts/{sid}.json — ids must be the file stem)")
            if edl.get("captions"):
                errors.append(msg + "; captions would skip this range")
            else:
                warns.append(msg + "; word boundaries unchecked")

        if start < 0:
            errors.append(f"{tag}: start {start:.2f} < 0")
        if end <= start:
            errors.append(f"{tag}: end {end:.2f} <= start {start:.2f}")
        if 0 < end - start < MIN_SEGMENT:
            warns.append(f"{tag}: only {end - start:.2f}s long — reads as a flash")
        for (a, b, j) in used.get(sid, []):
            overlap = min(b, end) - max(a, start)
            if overlap > 0.25:
                warns.append(f"{tag}: reuses {overlap:.1f}s of {sid} already in range[{j}]")
        used.setdefault(sid, []).append((start, end, i))

    def seg_len(r: dict) -> float:
        try:
            d = max(0.0, float(r["end"]) - float(r["start"]))
        except (KeyError, TypeError, ValueError):
            return 0.0
        return float(segment_frames(d, fps) / Fraction(fps)) if fps and d > 0 else d

    total = round(sum(seg_len(r) for r in ranges), 3)
    stated = edl.get("total_duration_s")
    try:
        stated_ok = stated is not None and abs(float(stated) - total) <= 0.05
    except (TypeError, ValueError):
        stated_ok = False
    if not stated_ok:
        warns.append(f"total_duration_s {stated} → actual {total:.2f}s")
        if fix:
            edl["total_duration_s"] = total

    for k, ov in enumerate(edl.get("overlays") or []):
        p = resolve(ov.get("file", ""), edit_dir)
        if not p.exists():
            errors.append(f"overlay[{k}]: file not found: {p}")
        try:
            t = float(ov.get("start_in_output", 0))
            d = float(ov.get("duration", 0))
        except (TypeError, ValueError):
            errors.append(f"overlay[{k}]: start_in_output/duration must be numbers")
            continue
        if t < 0 or d <= 0:
            errors.append(f"overlay[{k}]: needs start_in_output >= 0 and duration > 0")
        elif t + d > total + 0.05:
            errors.append(f"overlay[{k}]: ends at {t + d:.2f}s but the cut is {total:.2f}s")

    for key in ("captions", "subtitles"):
        if edl.get(key) and not resolve(edl[key], edit_dir, also_cwd=True).exists():
            errors.append(f"{key}: file not found: {edl[key]} (render it, or drop the key)")

    return errors, warns


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate (and optionally fix) an EDL")
    ap.add_argument("edl", type=Path)
    ap.add_argument("--fix", action="store_true", help="Snap/pad edges and rewrite the EDL")
    ap.add_argument("--pad-in", type=float, default=0.06,
                    help="Seconds of air before the first kept word (default 0.06)")
    ap.add_argument("--pad-out", type=float, default=0.10,
                    help="Seconds of air after the last kept word (default 0.10)")
    ap.add_argument("--fps", default=None,
                    help="Output frame rate; must match render.py --fps. Default: first source's rate.")
    args = ap.parse_args()

    for name, v in (("--pad-in", args.pad_in), ("--pad-out", args.pad_out)):
        if not MIN_PAD <= v <= MAX_PAD:
            sys.exit(f"{name} must be within {MIN_PAD}-{MAX_PAD}s")

    edl_path = args.edl.resolve()
    edl = json.loads(edl_path.read_text())
    try:
        fps = resolve_render_fps(edl, edl_path.parent, args.fps)
    except Exception:
        fps = None  # missing source etc.; reported as an error below
    errors, warns = validate(edl, edl_path.parent, args.fix, args.pad_in, args.pad_out, fps)

    for w in warns:
        print(f"W  {w}")
    for e in errors:
        print(f"E  {e}")

    if args.fix:
        shutil.copy2(edl_path, edl_path.with_suffix(".json.bak"))
        edl_path.write_text(json.dumps(edl, indent=2))
        print(f"rewrote {edl_path.name} (backup: {edl_path.name}.bak)")

    total = edl.get("total_duration_s")
    print(f"{len(errors)} error(s), {len(warns)} warning(s); "
          f"{len(edl.get('ranges') or [])} ranges, {total}s")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
