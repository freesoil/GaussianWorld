#!/bin/bash
# Build legacy FlashOCC / MambaOcc images and run import smoke tests.
# Requires: Docker (user in `docker` group, or run with sudo bash …).
# Optional GPU: add --gpus all to docker run if you test CUDA ops inside the container.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DOCKER=${DOCKER:-docker}

if ! $DOCKER version >/dev/null 2>&1; then
  echo "Docker not usable (try: sudo usermod -aG docker \$USER && newgrp docker)"
  exit 1
fi

echo "=== Build FlashOCC image ==="
$DOCKER build -f docker/Dockerfile.legacy-occ --build-arg PROJECT=FlashOCC -t flashocc-legacy "$ROOT"

echo "=== Build MambaOcc image ==="
$DOCKER build -f docker/Dockerfile.legacy-occ --build-arg PROJECT=MambaOcc -t mambaocc-legacy "$ROOT"

echo "=== FlashOCC: Python import smoke ==="
# train.py imports mmcv before argparse — use inline imports only.
$DOCKER run --rm flashocc-legacy python3.8 -c "
import mmcv
import mmcv._ext
from mmcv import Config
import mmdet
import mmdet3d
import mmdet3d_plugin
print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmdet3d', mmdet3d.__version__)
"

echo "=== MambaOcc: Python import smoke ==="
$DOCKER run --rm mambaocc-legacy python3.8 -c "
import mmcv
import mmcv._ext
from mmcv import Config
import mmdet
import mmdet3d
import mmdet3d_plugin
print('mmcv', mmcv.__version__, 'mmdet', mmdet.__version__, 'mmdet3d', mmdet3d.__version__)
"

echo "=== All docker smoke checks passed ==="
