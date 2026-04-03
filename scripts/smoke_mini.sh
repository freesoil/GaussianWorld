#!/bin/bash
# In-container or local: build mini PKLs + one dataloader batch (dummy SurroundOcc).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export GAUSSIANWORLD_DUMMY_SURROUNDOCC=1
: "${NUSCENES_DATA_ROOT:?Set NUSCENES_DATA_ROOT to nuScenes root (contains v1.0-mini/)}"
python tools/gen_nuscenes_mini_temporal_pkl.py --dataroot "$NUSCENES_DATA_ROOT" --out-dir data
python tools/smoke_mini_dataset.py
