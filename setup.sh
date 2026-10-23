#!/usr/bin/env bash
# Install and verify everything the video-edit skill needs. Safe to re-run.
#
#   ./setup.sh           install what's missing, then verify
#   ./setup.sh --check   verify only, install nothing
#
# Env:
#   VIDEO_EDIT_WHISPER_MODEL   use this ggml model file instead of downloading one
#   VIDEO_EDIT_MODEL_NAME      model to download if none is found (default large-v3-turbo,
#                              ~1.6 GB; base.en is ~150 MB but less accurate)
set -euo pipefail
cd "$(dirname "$0")"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1
HYPERFRAMES="hyperframes@0.8.188"
MODEL_NAME="${VIDEO_EDIT_MODEL_NAME:-large-v3-turbo}"
MODEL_DIR="$HOME/.cache/whisper"

ok()   { printf "  ok    %s\n" "$1"; }
miss() { printf "  MISS  %s\n" "$1"; MISSING=1; }
step() { printf "  ...   %s\n" "$1"; }
MISSING=0

echo "video-edit setup$([ $CHECK_ONLY = 1 ] && echo ' (check only)')"

# ---- system packages ---------------------------------------------------------
need_brew=()
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || need_brew+=(ffmpeg)
command -v whisper-cli >/dev/null || need_brew+=(whisper-cpp)
if command -v node >/dev/null; then
  [ "$(node -p 'process.versions.node.split(".")[0]')" -ge 22 ] || need_brew+=(node)
else
  need_brew+=(node)
fi

if [ ${#need_brew[@]} -gt 0 ]; then
  if [ $CHECK_ONLY = 0 ] && command -v brew >/dev/null; then
    step "brew install ${need_brew[*]}"
    brew install "${need_brew[@]}" || brew upgrade "${need_brew[@]}" || true
  elif ! command -v brew >/dev/null; then
    miss "${need_brew[*]} — install Homebrew first (https://brew.sh), then re-run. On Linux install ffmpeg, whisper.cpp and Node 22+ with your package manager."
  fi
fi

for b in ffmpeg ffprobe whisper-cli node npx; do
  command -v "$b" >/dev/null && ok "$b" || miss "$b"
done
if command -v node >/dev/null; then
  major=$(node -p 'process.versions.node.split(".")[0]')
  [ "$major" -ge 22 ] && ok "node $major (Hyperframes needs 22+)" || miss "node 22+ (have $major): brew upgrade node"
fi

if command -v ffmpeg >/dev/null; then
  decoders=$(ffmpeg -hide_banner -decoders 2>/dev/null || true)
  [[ "$decoders" == *libvpx-vp9* ]] && ok "ffmpeg libvpx-vp9 (alpha WebM overlays)" \
    || miss "ffmpeg without libvpx-vp9: transparent captions/overlays will render opaque"
  filters=$(ffmpeg -hide_banner -filters 2>/dev/null || true)
  scale_help=$(ffmpeg -hide_banner -h filter=scale 2>/dev/null || true)
  if [[ "$filters" == *" zscale "* ]]; then
    ok "ffmpeg zscale (HDR tone-mapping)"
  elif [[ "$scale_help" == *in_transfer* ]]; then
    ok "ffmpeg colour-managed scale (HDR fallback; no zscale in this build)"
  else
    miss "HDR conversion: ffmpeg has neither zscale nor colour-managed scale — iPhone HLG footage will fail"
  fi
fi

# ---- python ------------------------------------------------------------------
if ! command -v python3 >/dev/null; then
  miss "python3 (3.10+)"
else
  pyok=$(python3 -c 'import sys; print(int(sys.version_info >= (3, 10)))')
  if [ "$pyok" != 1 ]; then
    miss "python 3.10+ (have $(python3 -V))"
  else
    if [ ! -x .venv/bin/python ]; then
      [ $CHECK_ONLY = 0 ] && python3 -m venv .venv || miss "python venv (.venv) — run ./setup.sh"
    fi
    if [ -x .venv/bin/python ]; then
      [ $CHECK_ONLY = 0 ] && .venv/bin/pip install -q --disable-pip-version-check numpy pillow opencv-python-headless pytest
      .venv/bin/python -c 'import numpy, PIL' 2>/dev/null && ok "python venv (.venv) with numpy + pillow" \
        || miss "numpy/pillow in .venv — run ./setup.sh"
      cv=$(OPENCV_LOG_LEVEL=ERROR .venv/bin/python -c 'import cv2; cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320)); print(cv2.__version__)' 2>/dev/null || true)
      [ -n "$cv" ] && ok "opencv $cv + YuNet face model" || miss "opencv-python-headless in .venv — run ./setup.sh"
    fi
  fi
fi

# ---- whisper model -----------------------------------------------------------
find_model() {
  [ -x .venv/bin/python ] || return 1
  .venv/bin/python - <<'EOF' 2>/dev/null
import sys
sys.path.insert(0, "helpers")
import transcribe
print(transcribe.find_model(None))
EOF
}
model=$(find_model || true)
if [ -z "$model" ] && [ $CHECK_ONLY = 0 ]; then
  mkdir -p "$MODEL_DIR"
  dest="$MODEL_DIR/ggml-$MODEL_NAME.bin"
  step "downloading whisper model $MODEL_NAME to $dest"
  curl -fL --progress-bar -o "$dest.part" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$MODEL_NAME.bin"
  mv "$dest.part" "$dest"
  if [ "$MODEL_NAME" != "large-v3-turbo" ]; then
    echo "  note: add  export VIDEO_EDIT_WHISPER_MODEL=$dest  to your shell profile"
    export VIDEO_EDIT_WHISPER_MODEL="$dest"
  fi
  model=$(find_model || true)
fi
[ -n "$model" ] && ok "whisper model $model" || miss "whisper model — run ./setup.sh (downloads ~1.6 GB) or set VIDEO_EDIT_WHISPER_MODEL"

# ---- hyperframes -------------------------------------------------------------
if command -v npx >/dev/null; then
  npx --yes "$HYPERFRAMES" --version >/dev/null 2>&1 && ok "$HYPERFRAMES (cached by npx)" \
    || miss "$HYPERFRAMES via npx (needs network once)"
  if [ $CHECK_ONLY = 0 ]; then
    npx --yes "$HYPERFRAMES" browser ensure >/dev/null 2>&1 || true
  fi
  npx --yes "$HYPERFRAMES" browser path >/dev/null 2>&1 && ok "hyperframes Chrome for rendering" \
    || miss "hyperframes Chrome — needs network once: run ./setup.sh"
fi

command -v yt-dlp >/dev/null && ok "yt-dlp (optional, for online sources)" \
  || echo "  info  yt-dlp not installed (optional: brew install yt-dlp)"

if [ "$MISSING" = 0 ]; then
  echo "ready."
else
  echo "fix the MISS lines above, then re-run ./setup.sh"
  exit 1
fi
