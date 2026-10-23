"""Render animated word-by-word captions for an EDL with Hyperframes.

Maps every kept word onto the OUTPUT timeline (Rule 5:
output_time = word.start - segment_start + segment_offset), groups words
into short cues, writes a Hyperframes composition to <edit>/captions_<edl stem>/, renders
it to a transparent WebM, and points the EDL's `captions` key at it. render.py
then lays it over everything else (captions are the top layer, Rule 1).

Replaces SRT burn-in, which needs an ffmpeg built with libass (Homebrew's
is not), and looks better: the active word lights up as it is spoken.

Usage:
    python helpers/captions.py <edl.json>
    python helpers/captions.py <edl.json> --accent "#FFD400" --top 62 --size 84
    python helpers/captions.py <edl.json> --case none --no-pop --words 3
    python helpers/captions.py <edl.json> --no-render     # write the composition only

Canvas size comes from the EDL `canvas`, else the first source's display size
scaled the way render.py scales it (1920 on the long edge).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from render import resolve_render_fps, segment_frames  # same directory

HYPERFRAMES = "hyperframes@0.8.188"
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "captions"
PUNCT_BREAK = set(".,!?;:")


def resolve(maybe: str, base: Path) -> Path:
    p = Path(maybe)
    return p if p.is_absolute() else (base / p).resolve()


def display_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:stream_side_data=rotation",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    st = json.loads(out.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    for sd in st.get("side_data_list") or []:
        if sd.get("rotation") is not None and int(round(float(sd["rotation"]))) % 360 in (90, 270):
            w, h = h, w
    # render.py scales the long edge to 1920 (keeping aspect, even sizes)
    if h > w:
        return (round(w * 1920 / h / 2) * 2, 1920)
    return (1920, round(h * 1920 / w / 2) * 2)


def output_words(edl: dict, edit_dir: Path, fps: str) -> tuple[list[dict], float]:
    """Kept words with output-timeline times, plus the cut's total length.

    Each range lasts exactly segment_frames(duration, fps) / fps in the render,
    so offsets advance by that, not by the raw EDL difference.
    """
    words: list[dict] = []
    offset = 0.0
    missing: list[str] = []
    for r in edl["ranges"]:
        start = float(r["start"])
        seg_len = float(segment_frames(float(r["end"]) - start, fps) / Fraction(fps))
        end = start + seg_len
        tr = edit_dir / "transcripts" / f"{r['source']}.json"
        if tr.exists():
            for w in json.loads(tr.read_text()).get("words", []):
                if w.get("type") != "word" or w["end"] <= start or w["start"] >= end:
                    continue
                text = (w.get("text") or "").strip()
                if not text:
                    continue
                words.append({
                    "text": text,
                    "start": round(max(start, w["start"]) - start + offset, 3),
                    "end": round(min(end, w["end"]) - start + offset, 3),
                    "seg_end": round(end - start + offset, 3),
                })
        else:
            missing.append(r["source"])
        offset += seg_len
    if missing:
        sys.exit(
            f"no transcript for source id(s) {sorted(set(missing))}: captions would silently skip "
            "those ranges. Source ids must match transcripts/<id>.json (the file stem, or "
            "<stem>.track<N> for --audio-track N)."
        )
    return words, round(offset, 3)


def group_words(words: list[dict], per_cue: int, max_words: int,
                min_s: float = 0.35, pause_s: float = 0.3) -> list[dict]:
    """Same cue rules as render.py's SRT chunker: break on punctuation, on a
    pause, at a segment boundary, or at `per_cue` words (growing to `max_words`
    rather than flashing a cue shorter than `min_s`)."""
    groups: list[dict] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 0.0
        dur = w["end"] - cur[0]["start"]
        crosses_cut = nxt is not None and nxt["seg_end"] != w["seg_end"]
        if (nxt is None or crosses_cut or w["text"][-1] in PUNCT_BREAK or gap >= pause_s
                or len(cur) >= max_words or (len(cur) >= per_cue and dur >= min_s)):
            groups.append(cur)
            cur = []

    cues: list[dict] = []
    for gi, g in enumerate(groups):
        nxt_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else None
        # hold the cue until the next one (max +0.4s), never past its segment
        end = g[-1]["end"] + 0.4
        if nxt_start is not None:
            end = min(end, nxt_start)
        end = max(end, g[0]["start"] + 0.2)
        end = min(end, g[-1]["seg_end"])  # never crosses a cut, even for a tiny segment
        cues.append({
            "start": g[0]["start"],
            "end": round(end, 3),
            "words": [{"text": re.sub(r"[,;:]+$", "", w["text"]), "start": w["start"]} for w in g],
        })
    return cues


def hf_fps(rate: str) -> str:
    """ffmpeg rational → Hyperframes --fps (integer, or rational like 30000/1001)."""
    r = Fraction(rate)
    return str(r.numerator) if r.denominator == 1 else f"{r.numerator}/{r.denominator}"


def fill_template(cues: list[dict], width: int, height: int, duration: float, a) -> str:
    html = (TEMPLATE_DIR / "index.html").read_text()
    stroke = max(2, round(a.size * 0.07))
    values = {
        "__GROUPS__": json.dumps(cues),
        "__WIDTH__": str(width),
        "__HEIGHT__": str(height),
        "__DURATION__": f"{duration:.3f}",
        "__TOP__": str(a.top),
        "__SIZE__": str(a.size),
        "__FONT__": a.font,
        "__CASE__": "uppercase" if a.case == "upper" else "none",
        "__COLOR__": a.color,
        "__ACCENT__": a.accent,
        "__STROKE__": str(stroke),
        "__SHADOW__": str(stroke),
        "__SHADOWBLUR__": str(stroke * 3),
        "__POP__": "true" if a.pop else "false",
    }
    for k, v in values.items():
        html = html.replace(k, v)
    leftover = re.findall(r"__[A-Z]+__", html)
    if leftover:
        sys.exit(f"template placeholders not filled: {sorted(set(leftover))}")
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Hyperframes captions for an EDL")
    ap.add_argument("edl", type=Path)
    ap.add_argument("--words", type=int, default=2, help="Target words per cue (default 2)")
    ap.add_argument("--max-words", type=int, default=3, help="Max words per cue (default 3)")
    ap.add_argument("--top", type=float, default=None,
                    help="Vertical centre of the captions, %% from top. Default 62 for vertical "
                         "(clear of TikTok/Reels UI), 80 for landscape.")
    ap.add_argument("--size", type=int, default=None, help="Font px. Default ~4.4%% of canvas height.")
    ap.add_argument("--font", default='"Helvetica Neue", Helvetica, Arial, sans-serif',
                    help="CSS font-family. Use fonts installed on this Mac.")
    ap.add_argument("--color", default="#FFFFFF", help="Base text colour")
    ap.add_argument("--accent", default="#FFD400", help="Colour of the word being spoken")
    ap.add_argument("--case", choices=["upper", "none"], default="upper")
    ap.add_argument("--no-pop", dest="pop", action="store_false", help="Disable the scale pop per cue")
    ap.add_argument("--fps", default=None,
                    help="Output frame rate; must match render.py --fps. Default: first source's rate.")
    ap.add_argument("--no-render", action="store_true", help="Write the composition, skip rendering")
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    edit_dir = edl_path.parent
    edl = json.loads(edl_path.read_text())

    if edl.get("canvas"):
        width, height = int(edl["canvas"]["width"]), int(edl["canvas"]["height"])
    else:
        first = edl["ranges"][0]["source"]
        width, height = display_size(resolve(edl["sources"][first], edit_dir))
    vertical = height > width
    if args.top is None:
        args.top = 62 if vertical else 80
    if args.size is None:
        args.size = round(height * (0.044 if vertical else 0.06))

    fps = resolve_render_fps(edl, edit_dir, args.fps)
    words, duration = output_words(edl, edit_dir, fps)
    if not words:
        sys.exit("no transcript words fall inside the EDL ranges — nothing to caption")
    cues = group_words(words, args.words, max(args.words, args.max_words))

    # One captions dir per EDL, so 9:16 / 1:1 / 16:9 variants never overwrite each other.
    out_dir = edit_dir / f"captions_{edl_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(fill_template(cues, width, height, duration, args))
    shutil.copy2(TEMPLATE_DIR / "gsap.min.js", out_dir / "gsap.min.js")
    (out_dir / "cues.json").write_text(json.dumps(cues, indent=1))
    print(f"captions composition → {out_dir}  ({len(cues)} cues, {width}x{height}, {duration:.2f}s)")

    if args.no_render:
        return

    # Gate: lint + runtime + layout. A `first_frame_empty` craft warning is
    # expected for a caption layer and does not fail the check.
    chk = subprocess.run(["npx", "--yes", HYPERFRAMES, "check"], cwd=out_dir,
                         capture_output=True, text=True)
    if chk.returncode != 0:
        sys.exit(f"hyperframes check failed:\n{(chk.stdout + chk.stderr)[-2000:]}")

    webm = out_dir / "captions.webm"
    cmd = ["npx", "--yes", HYPERFRAMES, "render", "--format", "webm",
           "--fps", hf_fps(fps), "--output", str(webm)]
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    if res.returncode != 0 or not webm.exists():
        sys.exit(f"hyperframes render failed:\n{(res.stdout + res.stderr)[-2000:]}")

    edl["captions"] = f"{out_dir.name}/captions.webm"
    edl_path.write_text(json.dumps(edl, indent=2))
    print(f"  rendered {webm.name}; EDL `captions` → {edl['captions']}")


if __name__ == "__main__":
    main()
