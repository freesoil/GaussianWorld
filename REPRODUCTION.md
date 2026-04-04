# WorldOcc: Unified Occupancy Model Comparison

This guide provides the steps to reproduce the evaluation for **GaussianWorld**, **FlashOCC**, and **MambaOcc** using the shared nuScenes dataset.

## 1. Data Staging (Host Side)
Data is expected under `../data` (relative to this repo root). Ensure your dataset is organized as follows:
```bash
../data/
├── nuscenes/
│   ├── samples/
│   ├── sweeps/
│   ├── maps/
│   ├── v1.0-mini/   # Or v1.0-trainval for full dataset
│   └── ...
├── surroundocc/
│   ├── samples/
│   └── ...
├── nuscenes_temporal_infos_train.pkl
├── nuscenes_temporal_infos_val.pkl
└── (optional) v1mini/  # if you keep the mini split elsewhere
```

## 2. Environment Persistence & Setup
The Docker environments for FlashOCC and MambaOcc require specific CUDA extensions. To avoid recompilation, use the provided setup commands inside the containers.

### FlashOCC Setup
```bash
# Build image from repo root
docker build -f docker/Dockerfile.legacy-occ --build-arg PROJECT=FlashOCC -t flashocc-legacy .

docker run -it --gpus all \
  -v "$(pwd)/FlashOCC:/workspace/FlashOCC" \
  -v "$(pwd)/../data:/workspace/data" \
  flashocc-legacy bash

# Inside the container:
pip install ninja
cd /workspace/mmdetection3d && pip install -e .
export PYTHONPATH=/workspace
```

### MambaOcc Setup
```bash
# Build image from repo root
docker build -f docker/Dockerfile.legacy-occ --build-arg PROJECT=MambaOcc -t mambaocc-legacy .

docker run -it --gpus all \
  -v "$(pwd)/MambaOcc:/workspace/MambaOcc" \
  -v "$(pwd)/../data:/workspace/data" \
  mambaocc-legacy bash

# Inside the container:
pip install ninja
cd /workspace/mmdetection3d && pip install -e .
cd /workspace/projects && pip install -e .
cd /workspace/ops_dcnv3 && python setup.py install
cd /workspace/VMamba/kernels/selective_scan && pip install .
export PYTHONPATH=/workspace
```

## 3. Running Evaluation (nuScenes-mini)

### FlashOCC
```bash
# Inside FlashOCC container
python tools/test.py \
  projects/configs/flashocc/flashocc-r50.py \
  ckpts/flashocc-r50-256x704.pth \
  --eval map
```

### MambaOcc
```bash
# Inside MambaOcc container
python tools/test.py \
  projects/configs/mambaocc/mambaocc_tiny.py \
  pretrained_model/vssm_tiny_0230_ckpt_epoch_262.pth \
  --eval map
```

### GaussianWorld
```bash
# Build image from repo root
docker build -t gaussianworld GaussianWorld

docker run --gpus all -it \
  -v "$(pwd)/GaussianWorld:/workspace" \
  -v "$(pwd)/../data:/workspace/data" \
  --shm-size=100G \
  gaussianworld bash

# Inside the container:
cd /workspace
./scripts/eval_mini.sh
```

## 4. Key Implementation Notes
- **ExFAT Support:** Both `mmdetection3d` installations have been patched in their `setup.py` to use `copy` instead of `symlink` for `.mim` extensions, enabling compatibility with ExFAT drives.
- **Data Infos:** Custom `tools/create_data_bevdet_mini.py` scripts were created to generate the `.pkl` metadata specifically for the `v1.0-mini` dataset split.
- **FlashOCC dvr Module:** Requires `ninja` for JIT compilation during the first run.

## 5. Experiment Log

### Dataset Used
- nuScenes mini payload from Kaggle, staged at `/mnt/backwater/bev/data/nuscenes/kaggle`
- Wired into the shared dataset root as:
  - `/mnt/backwater/bev/data/nuscenes/samples -> kaggle/samples`
  - `/mnt/backwater/bev/data/nuscenes/sweeps -> kaggle/sweeps`
  - `/mnt/backwater/bev/data/nuscenes/maps -> kaggle/maps`
  - `/mnt/backwater/bev/data/nuscenes/v1.0-mini -> kaggle/v1.0-mini`
- Occ3D nuScenes mini ground-truth extracted to `/mnt/backwater/bev/data/nuscenes/gts`
- BEVDet info files used by FlashOCC and MambaOcc:
  - `/mnt/backwater/bev/data/nuscenes/bevdetv2-nuscenes_infos_train.pkl`
  - `/mnt/backwater/bev/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl`

### Weights Used
- FlashOCC: `FlashOCC/ckpts/flashocc-r50-256x704.pth`
- MambaOcc: `MambaOcc/pretrained_model/vssm_tiny_0230_ckpt_epoch_262.pth`

### Reproduction Commands

FlashOCC:
```bash
docker run --rm --gpus all   -v "$(pwd)/FlashOCC:/workspace/FlashOCC"   -v "$(pwd)/../data:/workspace/data"   flashocc-legacy bash -lc "cd /workspace/FlashOCC && PYTHONPATH=/workspace/FlashOCC python -u tools/test.py projects/configs/flashocc/flashocc-r50.py ckpts/flashocc-r50-256x704.pth --eval map"
```

MambaOcc:
```bash
docker run --rm --gpus all   -v "$(pwd)/MambaOcc:/workspace/MambaOcc"   -v "$(pwd)/../data:/workspace/data"   mambaocc-legacy bash -lc "cd /workspace/MambaOcc && PYTHONPATH=/workspace/MambaOcc python -u tools/test.py projects/configs/mambaocc/mambaocc_tiny.py pretrained_model/vssm_tiny_0230_ckpt_epoch_262.pth --eval map"
```

### Notes From Evaluation
- `FlashOCC/projects/configs/flashocc/flashocc-r50.py` was set to `workers_per_gpu=0` to avoid shared-memory worker crashes.
- `MambaOcc/projects/configs/mambaocc/mambaocc_tiny.py` was also set to `workers_per_gpu=0`.
- `MambaOcc/VMamba/classification/models/vmamba.py` was patched so the legacy VMamba checkpoint loads without a missing-key crash.
- `FlashOCC` currently reaches model execution but fails in `bev_pool_v2` with `RuntimeError: Tensors of type TensorImpl do not have sizes`.
- `MambaOcc` completes evaluation on the mini split and reports `mIoU of 81 samples: 0.87`.
