FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

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

RUN python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN python -m pip install -U openmim
RUN python -m mim install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.1/index.html
RUN python -m pip install mmdet==2.25.1 mmsegmentation==0.25.0 mmengine==0.10.1

RUN python -m pip install \
    pycuda \
    lyft-dataset-sdk \
    nuscenes-devkit \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh==2.35.39 \
    networkx==2.2 \
    numba==0.53.0 \
    numpy==1.23.5 \
    setuptools==59.5.0 \
    yapf==0.40.1 \
    einops \
    triton \
    timm==0.4.12 \
    chardet \
    yacs \
    submitit \
    tensorboardX \
    fvcore \
    seaborn \
    ipdb \
    spconv-cu113 \
    git+https://github.com/NVIDIA/gpu_affinity

WORKDIR /workspace
COPY . /workspace

CMD ["/bin/bash"]
