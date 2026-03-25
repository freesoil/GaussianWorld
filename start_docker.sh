docker run  --network host \
  --gpus all -it --name my-gaussian \
  -e DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --privileged \
  -v $(pwd):/workspace \
  --shm-size=100G \
  -w /workspace \
  gaussianworld \
  bash
