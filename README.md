Best-effort plan (optimize for: run + compare, then attempt light training)
Large datasets/checkpoints are intentionally excluded. See `DATASETS_AND_WEIGHTS.md` and `REPRODUCTION.md` for what to fetch and where to place it.
0. Pick the benchmark target (1 decision)
To compare these models “fairly”, we need the same dataset + voxel grid + class mapping.

Default (best match to your current repo):

SurroundOcc validation, using your existing grid sizing and mIoU/geo-IoU logic.
Fallback (if other repos don’t support SurroundOcc):

Use each model’s official Occ3D-nuScenes evaluation and report their official mIoU, then do a second pass later if we can export/convert predictions into your SurroundOcc grid.
1. Environment approach using cvml (and contingency)
Try to install dependencies for FlashOCC and MambaOcc into the existing ~/.venvs/cvml (same env as you requested).
If compiled deps (mmcv-full, mmdet3d, CUDA extensions like dcnv3/VMamba kernels) fail to build or are incompatible with PyTorch 2.10 / Python 3.12:
Stop short of heavy work,
Ask you whether you’re okay with creating temporary per-model conda envs (still on the same machine) for successful training/testing.
(Reason: these repos strongly assume specific torch/mmcv versions, and “same venv” can become a hard blocker.)

2. Set up a common “benchmark harness” (so outputs are comparable)
Even if each model uses its own codebase, we’ll create a single harness that records:

Semantic mIoU and geometry IoU (preferably computed with your GaussianWorld/utils/iou_eval.py logic)
Peak GPU memory
Throughput (FPS) on the same machine
Optionally runtime broken down by forward pass only vs full preprocessing
Concretely:

Write experiments/benchmark/run_occupancy_eval.py that:
Loads a model wrapper (one wrapper per model)
Runs evaluation on a fixed number of val samples (start with small N like 20–50 to iterate quickly)
Converts model outputs to the voxel grid format needed by the IoU calculator
Writes results/occupancy_bench.json
This gives you apples-to-apples comparison even if training strategies differ.

3. Model-by-model setup strategy
A. FlashOcc (highest-accuracy / TensorRT-friendly)
Clone Yzichen/FlashOCC.
Choose a smallest config first (e.g., M0 with R50/256x704 as in their README).
Get it running in inference mode:
Use their provided checkpoints if available (fastest way to validate the pipeline).
Compute evaluation numbers with the harness.
Training attempt (only after inference is proven):
Do short fine-tuning (e.g., head-only or very small number of iterations) because RTX 2070 + occupancy backbones is usually too slow/heavy for full training.
Notes:

FlashOcc’s repo explicitly mentions TensorRT for FPS numbers, but we can still measure FPS in PyTorch for your machine.
B. MambaOcc (state-space model occupancy)
Clone Hub-Tian/MambaOcc.
Follow their installation steps (includes VMamba + kernels + mmcv/mmdet stack).
Same workflow:
Smoke-test inference on a small val subset
Plug outputs into the benchmark harness
Only then attempt very short fine-tuning if dependencies install cleanly
Risk:

Their repo installation steps are very environment-specific (torch/mmcv + CUDA kernels). This is likely where cvml incompatibility will show up first.
C. FastOcc (fast voxel decoder)
This one needs clarification before I can guarantee a working plan:

“FastOcc” appears in web results with name collisions (unrelated packages exist), and I did not yet confirm the correct public occupancy implementation/code link for the ICRA 2024 method from the arXiv page.
Without the correct repo/checkpoints, I can’t safely write training/testing commands.
Plan:

I’ll search further for the actual occupancy FastOcc implementation + configs + checkpoints.
If the code still can’t be located, I’ll pause and ask you for the repo link (or checkpoint link) you intended.
If you provide it, we’ll integrate it the same way as FlashOcc/MambaOcc.
D. Mobile-GS (“Sort-Free Gaussians”)
Mobile-GS (as I found in web sources) is Gaussian Splatting / rendering, not directly semantic occupancy with voxel mIoU. So there are two possible interpretations:

If you want occupancy mIoU comparison:
We must define how to convert Gaussian splatting outputs into a voxel occupancy grid with semantic labels (likely non-trivial and may not be supported by the method as-is).
If you want speed/memory comparison only:
We can run Mobile-GS rendering benchmarks and compare FPS/memory, but it won’t be the same metric as mIoU.
Plan:

I’ll ask you which definition you want before proceeding.
4. Training/testing schedule (realistic for RTX 2070)
To satisfy “ideally train and test”, but also keep it feasible:

Phase 1 (must-do): Inference-only evaluation for all models where code is available and installable.
Phase 2 (optional if stable): short fine-tuning:
Start from provided pretrained weights
Train for a small number of iterations/epochs on a reduced subset
Re-evaluate with the harness
If install/training is blocked by environment incompatibilities:
Still do Phase 1 and report “training not achievable in current env”; we’ll fix env or adjust scope.
5. Deliverables you’ll get at the end
A single results/occupancy_bench.json containing per-model:
semantic mIoU
geometry IoU
peak memory
FPS/runtime
A README section describing:
which dataset split/grid was used
what preprocessing mapping was applied to make outputs comparable
what failed (if any) and why (dependency incompatibility vs missing code)
Open questions (so I don’t build the wrong pipeline). This is just for my memo, I will have to own the problem and figure it out
FastOcc: can you provide the exact GitHub link (or checkpoint link) for the occupancy FastOcc you mean? The name is ambiguous online.
Mobile-GS: do you want it evaluated with occupancy mIoU like the others, or is rendering FPS/memory acceptable?
Dataset: should all models be evaluated on SurroundOcc (your repo’s format), or is it acceptable to use each model’s official Occ3D-nuScenes evaluation if SurroundOcc integration is too hard?
If you answer those 3, I can proceed with the actual setup starting with FlashOcc + MambaOcc smoke tests (highest likelihood of “works first”), and only then move to the uncertain ones.
