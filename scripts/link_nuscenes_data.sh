#!/bin/bash
# Symlink repo-relative data/nuscenes -> your nuScenes root (mini or full).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="${1:-$HOME/reservoir/datasets/autonomy/nuscenes}"
mkdir -p "$REPO_ROOT/data"
rm -f "$REPO_ROOT/data/nuscenes"
ln -sfn "$TARGET" "$REPO_ROOT/data/nuscenes"
echo "Linked $REPO_ROOT/data/nuscenes -> $TARGET"
