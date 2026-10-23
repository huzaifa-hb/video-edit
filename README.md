# video-edit

A Claude Code skill that edits real footage by conversation, entirely on this Mac.
Fork of [browser-use/video-use](https://github.com/browser-use/video-use) (MIT).

```
footage ─► whisper.cpp (word-level, fillers kept, edges snapped to audio)
        ─► takes_packed.md ─► Claude picks takes ─► edl.json
        ─► validate_edl.py --fix (word-gap snapping, sanity gate)
        ─► reframe.py (OpenCV face tracking → crop for 9:16 / 4:5 / 1:1)
        ─► captions.py (Hyperframes, transparent WebM)
        ─► render.py (reframe, grade, fades, overlays, captions, -14 LUFS)
        ─► self-check frames + loudness ─► preview.mp4 / final.mp4
```

## What changed from upstream

| Upstream | Here | Why |
|---|---|---|
| ElevenLabs Scribe (cloud, paid) | whisper.cpp large-v3-turbo on Metal | Local, free, no key. A disfluent prompt keeps "um/uh"; DTW timing + audio-envelope snapping fixes whisper's drift at sentence starts. |
| EDL rendered as written | `validate_edl.py` gate, `--fix` snaps edges to word gaps | LLM edit lists fail on invented ids, out-of-range times and mid-word cuts. |
| libass SRT burn-in | Hyperframes word-highlight captions as a top alpha layer | Homebrew ffmpeg has no libass; animated captions are what social ads use anyway. |
| 1080p, source aspect only | EDL `canvas` + per-range `crop_x/crop_y` | 9:16 / 4:5 / 1:1 ad variants from 16:9 footage. |
| Centre crop only | `reframe.py`: OpenCV YuNet face detection aims each crop; `scenes.py` finds shot cuts and dark/blurry shots | Speakers are rarely centred in 16:9 UGC; b-roll has no transcript to cut on. |
| WebM overlays decoded opaque | libvpx-vp9 decoder for `.webm` overlays | Keeps Hyperframes transparency. |

Not ported: speaker diarization and audio-event tags (laughs, applause). Whisper doesn't produce them.

## Install

One command installs the skill into `~/.claude/skills/video-edit` together with everything it needs:

```bash
curl -fsSL https://raw.githubusercontent.com/huzaifa-hb/video-edit/main/install.sh | bash
```

It clones this repo (or updates an existing clone), then `setup.sh`:

- `brew install`s whatever is missing of **ffmpeg**, **whisper-cpp** and **Node 22+** (Homebrew must already be installed; on Linux install those yourself first)
- creates `.venv` with **numpy, Pillow, OpenCV (headless)** and pytest
- downloads the **whisper large-v3-turbo** model (~1.6 GB) to `~/.cache/whisper/` unless one is already present (`VIDEO_EDIT_MODEL_NAME=base.en` for a ~150 MB model, or point `VIDEO_EDIT_WHISPER_MODEL` at your own)
- caches **Hyperframes 0.8.188** and its headless Chrome
- verifies each piece and prints `ready.` (re-run any time; `./setup.sh --check` verifies without installing)

Re-run the same command to update. Install somewhere else with `VIDEO_EDIT_DIR=/path bash install.sh`.

## Use

Open Claude Code in a folder of footage and ask, e.g. "cut these takes into a 30s vertical ad with captions". Claude loads the skill from `SKILL.md`, which holds the full workflow.

Tests: `.venv/bin/python -m pytest -q tests`

Licences: this repo MIT (see `LICENSE`). Hyperframes Apache-2.0 (fetched by npx). GSAP (`templates/captions/gsap.min.js`) under GreenSock's standard no-charge licence. YuNet face model (`models/face_detection_yunet_2023mar.onnx`) MIT, from opencv_zoo.
