FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip python3-distutils \
    build-essential git wget curl ca-certificates \
    libjpeg-dev libpng-dev libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libevent-dev \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
 && python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN python -m pip install openmim
RUN python -m mim install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
RUN python -m pip install mmdet==2.25.1 mmsegmentation==0.25.0 mmdet3d==1.0.0rc4

RUN python -m pip install \
    pycuda \
    lyft-dataset-sdk \
    nuscenes-devkit \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh==2.35.39 \
    setuptools==59.5.0 \
    yapf==0.40.1 \
    networkx==2.2 \
    numba==0.53.0 \
    numpy==1.23.5 \
    git+https://github.com/NVIDIA/gpu_affinity

WORKDIR /workspace
COPY . /workspace

CMD ["/bin/bash"]
