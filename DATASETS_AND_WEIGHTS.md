# Datasets and Large Assets

This repo intentionally excludes datasets, checkpoints, and other large binaries. Use the links in each model's README and the summary below to fetch what you need after cloning.

If you want a shared dataset location across repos, you can keep data outside the repo (e.g., `../data`) and bind-mount it in Docker. The `REPRODUCTION.md` examples assume that layout. If you prefer an in-repo layout, use `data/` at the repo root and adjust paths accordingly.

## Common Dataset: nuScenes
Required by all three models.
- Download nuScenes v1.0 (mini or trainval) from the official site.
- Expected structure (relative to your chosen dataset root):
  - `data/nuscenes/samples`
  - `data/nuscenes/sweeps`
  - `data/nuscenes/maps`
  - `data/nuscenes/v1.0-mini` or `data/nuscenes/v1.0-trainval`
- See `REPRODUCTION.md` and each model README for details.

## GaussianWorld
Large assets (not tracked):
- Image backbone pretrain: `GaussianWorld/pretrain/r101_dcn_fcos3d_pretrain.pth`
- Optional eval checkpoints (if you want their provided results): e.g. `GaussianWorld/out/ckpt_base.pth`, `GaussianWorld/out/ckpt_stream.pth`

See `GaussianWorld/README.md` for download links and usage.

## FlashOCC
Large assets (not tracked):
- Official checkpoint: `FlashOCC/ckpts/flashocc-r50-256x704.pth`

See `FlashOCC/README.md` and `REPRODUCTION.md` for links and invocation.

## MambaOcc
Large assets (not tracked):
- Occ3D nuScenes occupancy GT: `data/nuscenes/gts`
- VMamba pretrained weights: `MambaOcc/pretrained_model/vssm_*_ckpt_epoch_*.pth`

See `MambaOcc/README.md` for the specific links (Google Drive / VMamba) and usage.

## Notes
- This file is a quick index; it does not replace the per-model setup docs.
- If you add new assets, keep them outside the repo and update this file with the expected paths and source links.
