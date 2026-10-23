---
name: video-edit
description: Edit real footage by conversation, fully local. Transcribe with whisper.cpp, pick takes from the transcript, write a JSON edit list, validate it, render with ffmpeg, add animated Hyperframes captions and overlays, reframe to 9:16 / 1:1 / 16:9, then self-check the render. Use for talking heads, UGC and testimonial ads, founder videos, call recordings, interviews, tutorials — whenever someone wants footage cut, trimmed, captioned, reframed or turned into ad variants ("cut this into a 30s ad", "remove the ums", "make a vertical version", "edit these takes").
---

# Video Edit

Fork of [browser-use/video-use](https://github.com/browser-use/video-use) (MIT): local whisper.cpp instead of ElevenLabs, a validation gate before every render, Hyperframes captions instead of libass subtitles, and canvas reframing for social ad variants.

## Principle

1. **LLM reasons from raw transcript + on-demand visuals.** The only derived artifact that earns its keep is a packed phrase-level transcript (`takes_packed.md`). Everything else — filler tagging, retake detection, shot classification, emphasis scoring — you derive at decision time.
2. **Audio is primary, visuals follow.** Cut candidates come from speech boundaries and silence gaps. Drill into visuals only at decision points.
3. **Ask → confirm → execute → iterate → persist.** Never touch the cut until the user has confirmed the strategy in plain English.
4. **Generalize.** Do not assume what kind of video this is. Look at the material, ask the user, then edit.
5. **Artistic freedom is the default.** Every specific value, preset, font, color, duration, pitch structure, and technique in this document is a *worked example* from one proven video — not a mandate. Read them to understand what's possible and why each worked. Then make your own taste calls based on what the material actually is and what the user actually wants. **The only things you MUST do are in the Hard Rules section below.** Everything else is yours.
6. **Invent freely.** If the material calls for a technique not described here — split-screen, picture-in-picture, lower-third identity cards, reaction cuts, speed ramps, freeze frames, crossfades, match cuts, L-cuts, J-cuts, speed ramps over breath, whatever — build it. The helpers are ffmpeg and PIL. They can do anything the format supports. Do not wait for permission.
7. **Verify your own output before showing it to the user.** If you wouldn't ship it, don't present it.

## Hard Rules (production correctness — non-negotiable)

These are the things where deviation produces silent failures or broken output. They are not taste, they are correctness. Memorize them.

1. **Subtitles are applied LAST in the filter chain**, after every overlay. Otherwise overlays hide captions. Silent failure.
2. **Per-segment extract → lossless `-c copy` concat**, not single-pass filtergraph. Otherwise you double-encode every segment when overlays are added.
3. **30ms audio fades at every segment boundary** (`afade=t=in:st=0:d=0.03,afade=t=out:st={dur-0.03}:d=0.03`). Otherwise audible pops at every cut.
4. **Overlays use `setpts=PTS-STARTPTS+T/TB`** to shift the overlay's frame 0 to its window start. Otherwise you see the middle of the animation during the overlay window.
5. **Captions use output-timeline offsets**: `output_time = word.start - segment_start + segment_offset`, where each segment lasts its whole-frame length (`captions.py` does this). Otherwise captions misalign after segment concat.
6. **Never cut inside a word.** Snap every cut edge to a word boundary from the transcript. `validate_edl.py --fix` does this mechanically.
7. **Pad every cut edge.** Working window: 30–200ms. Word edges are already snapped to the audio envelope, but padding absorbs what's left. Tighter for fast-paced, looser for cinematic.
8. **Word-level verbatim ASR only.** Never SRT/phrase mode (loses sub-second gap data). Never normalized fillers (loses editorial signal). `transcribe.py` prompts whisper to keep "um"/"uh" — don't remove the prompt.
9. **Cache transcripts per source.** Never re-transcribe unless the source file itself changed.
10. **Parallel sub-agents for multiple animations.** Never sequential. Spawn N at once via the `Agent` tool; total wall time ≈ slowest one.
11. **Strategy confirmation before execution.** Never touch the cut until the user has approved the plain-English plan.
12. **All session outputs in `<videos_dir>/edit/`.** Never write inside the skill directory.
13. **Validate before every render.** `validate_edl.py <edl> --fix`, then render only on exit code 0. It catches invented source ids, times past the end of a file, mid-word cuts, missing overlay files and overlays that outrun the cut — the failure modes LLM-written edit lists actually have.
14. **Captions are a Hyperframes layer, not burned subtitles.** Homebrew's ffmpeg has no libass, so `subtitles` in the EDL fails. Use `captions.py`; `render.py` composites the captions WebM above every overlay.
15. **Source ids are file stems.** In `sources`, the key must be the video's file name without extension (`take_B` for `take_B.mp4`), or `<stem>.track<N>` when transcribed with `--audio-track N` — that's how transcripts are found. The validator errors on a mismatch when captions are on.
16. **One frame rate per render.** If you pass `--fps` to `render.py`, pass the same `--fps` to `validate_edl.py` and `captions.py`. Ranges are rounded to whole frames so captions never drift.

Everything else in this document is a worked example. Deviate whenever the material calls for it.

## Directory layout

The skill lives in its own directory (helpers + templates next to this file). User footage lives wherever they put it. All session outputs go into `<videos_dir>/edit/`.

```
<videos_dir>/
├── <source files, untouched>
└── edit/
    ├── project.md                 ← memory; appended every session
    ├── takes_packed.md            ← phrase-level transcripts, the LLM's primary reading view
    ├── edl.json                   ← cut decisions (edl_9x16.json, edl_1x1.json … for variants)
    ├── transcripts/<name>.json    ← cached whisper.cpp transcripts (word-level + silences)
    ├── captions_<edl stem>/       ← Hyperframes caption composition + captions.webm
    ├── animations/slot_<id>/      ← per-animation source + render + reasoning
    ├── clips_graded/              ← per-segment extracts with grade + fades
    ├── downloads/                 ← yt-dlp outputs
    ├── verify/                    ← debug frames / timeline PNGs
    ├── preview.mp4
    └── final.mp4
```

## Setup

`install.sh` / `setup.sh` install everything (Homebrew packages, `.venv`, whisper model, Hyperframes + Chrome). On cold start just verify — and if `.venv` is missing or a helper fails on an import or a missing tool, run `<this skill dir>/setup.sh` (it is idempotent) before anything else:

- `PY=<this skill dir>/.venv/bin/python` exists. Run every helper with `$PY`, never the system python (it lacks numpy/Pillow/OpenCV).
- `ffmpeg`, `ffprobe`, `whisper-cli` on PATH, and a ggml whisper model. Default: `~/.claude-video-vision/models/ggml-large-v3-turbo.bin`; override with `VIDEO_EDIT_WHISPER_MODEL`.
- Node.js 22+ for Hyperframes. Always call the pinned version: `npx --yes hyperframes@0.8.188 …`.
- `yt-dlp` and Manim only on first use. This skill vendors `skills/manim-video/`; read its SKILL.md when building a Manim slot.
- No API keys. Nothing leaves the machine except the one-time npm download of Hyperframes.

Helpers live in `helpers/` next to this SKILL.md. Resolve their paths relative to the directory containing this file.

## Helpers

All take `$PY helpers/<name>.py …`.

- **`transcribe.py <video>`** — local whisper.cpp, word-level, fillers kept, word edges snapped to the audio envelope, silences listed. Cached per source (re-runs only if the file changed). `--language en` skips detection; `--audio-track N` for multi-track recordings.
- **`transcribe_batch.py <videos_dir>`** — every video in a folder, one at a time (whisper already saturates the GPU).
- **`pack_transcripts.py --edit-dir <dir>`** — `transcripts/*.json` → `takes_packed.md` (phrase-level, break on silence ≥ 0.5s; lower `--silence-threshold` to 0.3 for fast talkers).
- **`timeline_view.py <video> <start> <end>`** — filmstrip + waveform PNG. On-demand visual drill-down. **Not a scan tool** — use it at decision points, not constantly.
- **`validate_edl.py <edl.json> [--fix]`** — the gate before every render (Hard Rule 13). `--fix` snaps edges to word gaps, pads them (`--pad-in`, `--pad-out`), rounds ranges to whole frames, recomputes `total_duration_s`, keeps `edl.json.bak`. Warnings about edges with < 30ms of silence survive `--fix`: those cuts sit where words touch — move them to a real pause.
- **`captions.py <edl.json>`** — output-timeline word captions as a Hyperframes composition → transparent `captions.webm`, sets the EDL's `captions` key. Style flags: `--accent`, `--color`, `--top`, `--size`, `--font`, `--case`, `--words`, `--no-pop`.
- **`render.py <edl.json> -o <out>`** — per-segment extract (frame-exact, HDR→SDR, canvas reframe, grade, fades) → lossless concat → overlays (PTS-shifted) → captions → loudnorm. `--preview` for QC, `--draft` for cut checks only.
- **`reframe.py <edl.json> [--apply]`** — OpenCV face detection (YuNet) across each range → `crop_x`/`crop_y` that keep the speaker centred with headroom on the EDL `canvas`. Run it after `validate_edl.py --fix` whenever a `canvas` changes the aspect ratio. It flags ranges where the face `MOVES` too much for one static crop (split them) and ranges with `no face` (pick the crop yourself). `"crop_lock": true` on a range protects a manual crop.
- **`scenes.py <video>`** — OpenCV shot-change detection plus `DARK` / `BLURRY` flags per shot → `edit/shots/<stem>.json`. Use it for footage the transcript can't structure: b-roll, product shots, montages, multicam exports, silent clips. Shot boundaries are clean visual cut points; flagged shots are rarely usable in an ad.
- **`grade.py <in> -o <out>`** — ffmpeg filter chain grade. Presets + `--filter '<raw>'` for custom.

For animations, create `<edit>/animations/slot_<id>/` with `Bash` and spawn a sub-agent via the `Agent` tool.

## The process

1. **Inventory.** `ffprobe` every source. `transcribe_batch.py` on the directory. For sources with little or no speech (b-roll, product shots), run `scenes.py` instead to get their shots. Read the `QC:` lines it prints (also under `qc` in each transcript): words squeezed to ≤20ms or starting inside a silence have untrustworthy times — check any cut near them with `timeline_view` before using it.
   - **Multiple speakers** (calls, interviews): there are no speaker labels. Infer turns from content (questions vs answers), then confirm each phrase you plan to keep with `timeline_view` (on video calls the active-speaker tile shows who talks). If the call was recorded with one person per audio track, transcribe each track with `--audio-track N` instead. `pack_transcripts.py` to produce `takes_packed.md`. Sample one or two `timeline_view`s for a visual first impression.
2. **Pre-scan for problems.** One pass over `takes_packed.md` to note verbal slips, obvious mis-speaks, or phrasings to avoid. Plain list, feed into the editor brief.
3. **Converse.** Describe what you see in plain English. Ask questions *shaped by the material*. Collect: content type, target length/aspect, aesthetic/brand direction, pacing feel, must-preserve moments, must-cut moments, animation and grade preferences, subtitle needs. Do not use a fixed checklist — the right questions are different every time.
4. **Propose strategy.** 4–8 sentences: shape, take choices, cut direction, animation plan, grade direction, subtitle style, length estimate. **Wait for confirmation.**
5. **Execute.** Produce `edl.json` via the editor sub-agent brief. Drill into `timeline_view` at ambiguous moments. Build animations in parallel sub-agents. Run `validate_edl.py --fix` until it exits 0. Then `captions.py` if captions were asked for (captions are built from the final ranges, so re-run it after any range change).
6. **Preview.** `validate_edl.py` (no `--fix`) → `render.py --preview`.
7. **Self-eval (before showing the user).** Run `timeline_view` on the **rendered output** (not the sources) at every cut boundary (±1.5s window). Check each image for:
   - Visual discontinuity / flash / jump at the cut
   - Waveform spike at the boundary (audio pop that slipped past the 30ms fade)
   - Subtitle hidden behind an overlay (Rule 1 violation)
   - Overlay misaligned or showing wrong frames (Rule 4 violation)

   Also sample: first 2s, last 2s, and 2–3 mid-points — check grade consistency, subtitle readability, overall coherence. Run `ffprobe` on the output to verify duration matches the EDL expectation.

   Measure the audio, don't assume it: `ffmpeg -i out.mp4 -af ebur128=peak=true -f null -` for integrated loudness and true peak, plus RMS per section (dialogue, music-only, end card). An end card 15 dB under the dialogue, or effects louder than speech, is a bug. You cannot listen: say so, and report the numbers.

   For anything the user will publish (launch, promo, ad), also spawn one **critic sub-agent** with the rendered file, the EDL, and any reference videos the user gave. Brief it to roast, not to praise: a verdict, ranked problems with timecodes and evidence (frames, levels), and the 5 fixes to do first. Fresh eyes catch what the author stopped seeing — cut-off payoff lines, 0.5s memes, unreadable 28px text at phone size.

   If anything fails: fix → re-render → re-eval. **Cap at 3 self-eval passes** — if issues remain after 3, flag them to the user rather than looping forever. Only present the preview once the self-eval passes.
8. **Iterate + persist.** Natural-language feedback, re-plan, re-render. Never re-transcribe. Final render on confirmation. Append to `project.md`.

## Cut craft (techniques)

- **Audio-first.** Candidate cuts from word boundaries and silence gaps.
- **Preserve peaks.** Laughs, punchlines, emphasis beats. Extend past punchlines to include reactions — the laugh IS the beat.
- **Speaker handoffs** benefit from air between utterances. Common values: 400–600ms. Less for fast-paced, more for cinematic. Taste call.
- **Laughs and reactions** are not tagged (whisper doesn't emit audio events). Find them as speech-free but loud stretches in `timeline_view` waveforms; extend past them — the laugh IS the beat.
- **Silence gaps are cut candidates.** Silences ≥400ms are usually the cleanest. 150–400ms phrase boundaries are usable with a visual check. <150ms is unsafe (mid-phrase).
- **Example cut padding** (the launch video shipped with this): 50ms before the first kept word, 80ms after the last. Tighter for montage energy, looser for documentary. Stay in the 30–200ms working window (Hard Rule 7).
- **Never reason audio and video independently.** Every cut must work on both tracks.

## The packed transcript (primary reading view)

`pack_transcripts.py` reads all `transcripts/*.json` and produces one markdown file where each take is a list of phrase-level lines, each prefixed with its `[start-end]` time range. Phrases break on any silence ≥ 0.5s or at the end of a sentence (whisper gives no speaker labels). This is the artifact the editor sub-agent reads to pick cuts — it gives word-boundary precision from text alone at 1/10 the tokens of raw JSON.

Example line:
```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] We fixed this.
```

## Editor sub-agent brief (for multi-take selection)

When the task is "pick the best take of each beat across many clips," spawn a dedicated sub-agent with a brief shaped like this. The structure is load-bearing; the pitch-shape example is not.

```
You are editing a <type> video. Pick the best take of each beat and 
assemble them chronologically by beat, not by source clip order.

INPUTS:
  - takes_packed.md (time-annotated phrase-level transcripts of all takes)
  - Product/narrative context: <2 sentences from the user>
  - Speaker(s): <name, role, delivery style note>
  - Expected structure: <pick an archetype or invent one>
  - Verbal slips to avoid: <list from the pre-scan pass>
  - Target runtime: <seconds>

Common structural archetypes (pick, adapt, or invent):
  - Tech launch / demo:   HOOK → PROBLEM → SOLUTION → BENEFIT → EXAMPLE → CTA
  - Tutorial:             INTRO → SETUP → STEPS → GOTCHAS → RECAP
  - Interview:            (QUESTION → ANSWER → FOLLOWUP) repeat
  - Travel / event:       ARRIVAL → HIGHLIGHTS → QUIET MOMENTS → DEPARTURE
  - Documentary:          THESIS → EVIDENCE → COUNTERPOINT → CONCLUSION
  - Music / performance:  INTRO → VERSE → CHORUS → BRIDGE → OUTRO
  - Or invent your own.

RULES:
  - Start/end times must fall on word boundaries from the transcript.
  - Pad cut boundaries (working window 30–200ms).
  - Prefer silences ≥ 400ms as cut targets.
  - Unavoidable slips are kept if no better take exists. Note them in "reason".
  - If over budget, revise: drop a beat or trim tails. Report total and self-correct.

OUTPUT (JSON array, no prose):
  [{"source": "C0103", "start": 2.42, "end": 6.85, "beat": "HOOK",
    "quote": "...", "reason": "..."}, ...]

Return the final EDL and a one-line total runtime check.
```

## Color grade (when requested)

Your job is to **reason about the image**, not apply a preset. Look at a frame (via `timeline_view`), decide what's wrong, adjust one thing, look again.

Mental model is ASC CDL. Per channel: `out = (in * slope + offset) ** power`, then global saturation. `slope` → highlights, `offset` → shadows, `power` → midtones.

**Example filter chains** (`grade.py` has `--list-presets`; use them as starting points or mix your own):

- **`warm_cinematic`** — retro/technical, subtle teal/orange split, desaturated. Shipped in a real launch video. Safe for talking heads.
- **`neutral_punch`** — minimal corrective: contrast bump + gentle S-curve. No hue shifts.
- **`none`** — straight copy. Default when the user hasn't asked.

For anything else — portraiture, nature, product, music video, documentary — invent your own chain. `grade.py --filter '<raw ffmpeg>'` accepts any filter string.

Hard rules: apply **per-segment during extraction** (not post-concat, which re-encodes twice). Never go aggressive without testing skin tones.

## Captions (when requested)

Captions have three dimensions worth reasoning about: **chunking** (1/2/3/sentence per cue), **case** (UPPER/natural), and **placement** (vertical position). The right combo depends on content.

`captions.py` renders them with Hyperframes from `templates/captions/index.html`: each cue pops in, and the word being spoken switches to the accent colour on its own timestamp. Times use output-timeline offsets (Rule 5) and a cue never crosses a cut.

**Worked styles** — pick, adapt, or invent:

- **Social ad (default).** `--words 2 --case upper`, white + `#FFD400` accent, centred at `--top 62` on 9:16 so TikTok/Reels/Shorts UI (bottom ~25–30%) never covers it. Font size defaults to ~4.4% of canvas height (≈84px at 1920).
- **Natural sentence.** Narrative, testimonial, education: `--words 5 --max-words 7 --case none --no-pop`, `--top 80` on 16:9.
- **Brand.** Pass the brand's colours with `--color`/`--accent`, and `--font` with a font installed on this Mac. For anything beyond these flags (boxes, emoji, per-word animation), edit the composition in `<edit>/captions_<edl>/index.html`, run `npx --yes hyperframes@0.8.188 check` there, then render with `npx --yes hyperframes@0.8.188 render --format webm --output captions.webm`.

`hyperframes check` flags `first_frame_empty` on a caption layer. Expected, ignore it. Captions must stay the top layer (Rule 1); `render.py` handles that when the EDL has `captions`.

## Animations (when requested)

Animations match the content and the brand. **Get the palette, font, and visual language from the conversation** — never assume a default. If the user hasn't told you, propose a palette in the strategy phase and wait for confirmation before building anything.

**Tool options:**

Pick the engine per animation slot. Do not default to Remotion just because the animation is web-adjacent.

Default to **HyperFrames** (Apache-2.0, already used for captions). Remotion needs a paid company licence for teams over 3 people — use it only when the user has one or asks for it.

- **HyperFrames** — Browser-native HTML/CSS/GSAP video compositions: product UI motion, website-to-video or mockup-to-video captures, kinetic typography, landing-page/storyboard promos, data-driven UI states, transparent WebM overlays, and clips that need deterministic frame capture plus HyperFrames lint/validate/render checks. Best when the animation should be authored and verified like a web composition instead of a React component tree.
- **Remotion** — React/CSS compositions with component state, reusable React primitives, or an existing Remotion brand system. Best when the user specifically asks for React/Remotion or when React composition is the simpler authoring model.
- **Manim** — formal diagrams, state machines, equation derivations, graph morphs. Read `skills/manim-video/SKILL.md` and its references for depth.
- **PIL + PNG sequence + ffmpeg** — simple overlay cards: counters, typewriter text, single bar reveals, progressive draws. Fast to iterate, any aesthetic you want. The launch video used this.

For HyperFrames slots, scaffold the slot inside `edit/animations/slot_<id>/` with `npx --yes hyperframes@0.8.188 init . --example blank --non-interactive --skip-skills`, build the HTML composition there, run the HyperFrames checks that fit the slot (`lint`, `validate`, and a draft render when practical), then produce the final overlay video with `npx --yes hyperframes@0.8.188 render --output render.mp4`, or `--format webm --output render.webm` when it must be transparent over footage. `render.py` decodes WebM overlays with alpha. Point the EDL overlay `file` at the actual rendered path.

For Remotion slots, keep the Remotion project isolated inside the same slot directory, scaffold with `npx create-video@latest` or install Remotion locally there, render the composition to `render.mp4` with the project-local `remotion render` command, and verify duration and dimensions with `ffprobe`.

None is mandatory. Invent hybrids if useful (e.g., PIL background with a HyperFrames or Remotion layer on top).

**Duration rules of thumb, context-dependent:**

- **Sync-to-narration explanations.** A viewer needs to parse the content at 1×. Rough floor 3s, typical 5–7s for simple cards, 8–14s for complex diagrams. The launch video shipped at 5–7s per simple card.
- **Beat-synced accents** (music video, fast montage). 0.5–2s is fine — they're visual accents, not information. The "readable at 1×" rule becomes *"recognizable at 1×"*, not *"fully parseable."*
- **Hold the final frame ≥ 1s** before the cut (universal).
- **Over voiceover:** total duration ≥ `narration_length + 1s` (universal).
- **Never parallel-reveal independent elements** — the eye can't track two new things at once. One thing, pause, next thing.

**Animation payoff timing (rule for sync-to-narration):** get the payoff word's timestamp. Start the overlay `reveal_duration` seconds earlier so the landing frame coincides with the spoken payoff word. Without this sync the animation feels disconnected.

**Easing** (universal — never `linear`, it looks robotic):

```python
def ease_out_cubic(t):    return 1 - (1 - t) ** 3
def ease_in_out_cubic(t):
    if t < 0.5: return 4 * t ** 3
    return 1 - (-2 * t + 2) ** 3 / 2
```

`ease_out_cubic` for single reveals (slow landing). `ease_in_out_cubic` for continuous draws.

**Typing text anchor trick:** center on the FULL string's width, not the partial-string width — otherwise text slides left during reveal.

**Example palette** (the launch video — one aesthetic among infinite):
- Background `(10, 10, 10)` near-black
- Accent `#FF5A00` / `(255, 90, 0)` orange
- Labels `(110, 110, 110)` dim gray
- Font: Menlo Bold at `/System/Library/Fonts/Menlo.ttc` (index 1)
- ≤ 2 accent colors, ~40% empty space, minimal chrome
- Result: terminal / retro tech feel

This is one style. If the brand is warm and serif, use that. If it's colorful and playful, use that. If the user handed you a style guide, follow it. If they didn't, propose one and confirm.

**Fonts fail silently.** A web font that didn't load renders in a fallback face with no error — the video ships in "almost Arial". In HyperFrames/Remotion, await the font load and then assert it: `if (!document.fonts.check('700 76px "Inter"')) throw new Error(...)`. In PIL, pass an explicit font path; never rely on the default.

**Worked example — "show the edit" hook** (a launch video for this tool). Instead of a title card, the first 3s visualize the editing itself: each transcript word pops in on its transcript timestamp, with bars under it drawn from the real audio envelope; a filler ("ummm") grows letter by letter while it is spoken, turns orange and is cut out on screen at the same frame the audio cuts, and the next line lands immediately. It works because the picture is *driven by the same data as the sound* — one composition (Remotion) reads frame-exact word/envelope JSON produced in Python, so nothing can drift. Use the idea whenever the story is "we removed something": make the removal visible.

**Parallel sub-agent brief** — each animation is one sub-agent spawned via the `Agent` tool. Each prompt is self-contained (sub-agents have no parent context). Include:

1. One-sentence goal: *"Build ONE animation: [spec]. Nothing else."*
2. Absolute output path (`<edit>/animations/slot_<id>/render.mp4`)
3. Exact technical spec: resolution, fps, codec, pix_fmt, CRF, duration
4. Style palette as concrete values (RGB tuples, hex, or reference to a design system)
5. Font path with index
6. Frame-by-frame timeline (what happens when, with easing)
7. Anti-list ("no chrome, no extras, no titles unless specified")
8. Code pattern reference (copy helpers inline, don't import across slots)
9. Deliverable checklist (script, render, verify duration via ffprobe, report)
10. **"Do not ask questions. If anything is ambiguous, pick the most obvious interpretation and proceed."**

One sub-agent = one file (unique filenames, parallel agents don't overwrite each other).

## Music and sound effects (when requested)

Sound is where generated videos sound cheap. Worked rules from launch edits:

- **Fewer effects.** Every effect is tied to something visible (a cut, a landing, a click). ~20 stock whooshes/risers/impacts in 18s reads as generic; ~8 reads as designed.
- **Hit on the frame.** Most effects have an attack (silence or a build before the transient). Measure it (first sample above ~-30 dBFS of the peak) and start the file `attack` seconds *before* the visible contact frame.
- **Duck music under speech** (roughly -12 to -15 dB relative to its music-only level), and ramp it out before a stinger or end card instead of letting its own tail decay under your CTA.
- **Master once:** mix to PCM, then two-pass loudnorm (-14 LUFS, true peak ≤ -1 dBTP) on the final mix. Then measure per section (see Self-eval).
- **Music taste is the user's call.** Generated music defaults to "hype"; offer two contrasting beds and let the user listen. Don't claim a mix sounds good — you can only measure it.

## Output spec and ad variants

Match the source unless the user asked for something specific. Without a `canvas`, `render.py` scales the long edge to 1920 and keeps the source aspect (all sources must share an orientation, or the concat breaks — set a `canvas` when mixing).

For social ads, set `canvas`, run `reframe.py --apply` to aim the crop at the speaker's face, and reframe: `{"width": 1080, "height": 1920}` (9:16), `1080×1350` (4:5 feed), `1080×1080` (1:1), `1920×1080` (16:9). Every segment is scaled to cover the canvas and centre-cropped; per-range `crop_x` / `crop_y` (0–1) move the crop when the subject isn't centred. Check the crop with `timeline_view` on the render — a cropped-off face is the commonest reframe bug.

**Variants:** one EDL per variant in the same `edit/` (`edl_9x16.json`, `edl_1x1.json`, `edl_hookB.json`). Each gets its own captions dir. Render them one after another (the segment cache dir is shared).

**From an editor brief:** when the user already has a cut list with timecodes (e.g. from the `call-footage-ad-briefs` skill), turn each ad into its own EDL directly from those in/out points, then let `validate_edl.py --fix` snap them to word gaps. Skip the strategy questions that the brief already answers.

## EDL format

Paths are absolute, or relative to the EDL's own folder (`edit/`).

```json
{
  "version": 1,
  "sources": {"C0103": "/abs/path/C0103.MP4", "C0108": "/abs/path/C0108.MP4"},
  "canvas": {"width": 1080, "height": 1920},
  "ranges": [
    {"source": "C0103", "start": 2.42, "end": 6.85,
     "beat": "HOOK", "quote": "...", "reason": "Cleanest delivery, stops before slip at 38.46."},
    {"source": "C0108", "start": 14.30, "end": 28.90, "crop_x": 0.35,
     "beat": "SOLUTION", "quote": "...", "reason": "Only take without the false start."}
  ],
  "grade": "warm_cinematic",
  "overlays": [
    {"file": "animations/slot_1/render.webm", "start_in_output": 0.0, "duration": 5.0}
  ],
  "captions": "captions_edl/captions.webm",
  "total_duration_s": 19.03
}
```

`canvas` is optional (reframe target). `crop_x`/`crop_y` optional per range. `grade` is a preset name or raw ffmpeg filter. `overlays` are rendered animation clips (MP4, or WebM with alpha). `captions` is written by `captions.py` and composited above all overlays. `subtitles` (SRT burn-in) only works with an ffmpeg that has libass.

## Memory — `project.md`

Append one section per session at `<edit>/project.md`:

```markdown
## Session N — YYYY-MM-DD

**Strategy:** one paragraph describing the approach
**Decisions:** take choices, cuts, grades, animations + why
**Reasoning log:** one-line rationale for non-obvious decisions
**Outstanding:** deferred items
```

On startup, read `project.md` if it exists and summarize the last session in one sentence before asking whether to continue.

## Anti-patterns

Things that consistently fail regardless of style:

- **Hierarchical pre-computed codec formats** with USABILITY / tone tags / shot layers. Over-engineering. Derive from the transcript at decision time.
- **Hand-tuned moment-scoring functions.** The LLM picks better than any heuristic you'll write.
- **Whisper SRT / phrase-level output.** Loses sub-second gap data. Always word-level verbatim.
- **Trusting raw whisper word times.** They drift at sentence starts. `transcribe.py` corrects them with DTW + the audio envelope; `validate_edl.py --fix` snaps anyway.
- **Rendering an unvalidated EDL.** Minutes of ffmpeg to discover an invented source id. Always validate first.
- **Burning subtitles into base before compositing overlays.** Overlays hide them. (Hard Rule 1.)
- **Single-pass filtergraph when you have overlays.** Double re-encodes. Use per-segment extract → concat.
- **Linear animation easing.** Looks robotic. Always cubic.
- **Unverified web fonts.** A failed load silently falls back to a system face. Assert the font loaded before rendering.
- **Stock SFX on every transition.** Tie each effect to a visible event; cap the count.
- **Hard audio cuts at segment boundaries.** Audible pops. (Hard Rule 3.)
- **Typing text centered on the partial string.** Text slides left as it grows.
- **Sequential sub-agents for multiple animations.** Always parallel.
- **Editing before confirming the strategy.** Never.
- **Re-transcribing cached sources.** Immutable outputs of immutable inputs.
- **Assuming what kind of video it is.** Look first, ask second, edit last.
