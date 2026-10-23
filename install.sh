#!/usr/bin/env bash
# Install (or update) the video-edit skill for Claude Code, with all dependencies.
#
#   curl -fsSL https://raw.githubusercontent.com/huzaifa-hb/video-edit/main/install.sh | bash
#
# Env:
#   VIDEO_EDIT_DIR   where to install (default ~/.claude/skills/video-edit)
set -euo pipefail

REPO="https://github.com/huzaifa-hb/video-edit.git"
DEST="${VIDEO_EDIT_DIR:-$HOME/.claude/skills/video-edit}"

command -v git >/dev/null || { echo "git is required (macOS: xcode-select --install)"; exit 1; }

if [ -d "$DEST/.git" ]; then
  echo "updating $DEST"
  git -C "$DEST" pull --ff-only
else
  if [ -e "$DEST" ]; then
    backup="$DEST.bak.$(date +%Y%m%d%H%M%S)"
    echo "moving existing $DEST to $backup"
    mv "$DEST" "$backup"
  fi
  mkdir -p "$(dirname "$DEST")"
  echo "cloning into $DEST"
  git clone --depth 1 "$REPO" "$DEST"
fi

"$DEST/setup.sh"

echo
echo "video-edit is installed at $DEST"
echo "Open Claude Code in a folder of footage and ask, e.g.:"
echo '  "cut these takes into a 30s vertical ad with captions"'
