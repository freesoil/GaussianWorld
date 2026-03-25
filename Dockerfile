# Base image: Ubuntu 20.04 with CUDA 11.8 (devel = includes nvcc)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.9" 

# Install system dependencies and Python 3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3.8-distutils \
    build-essential git wget curl ca-certificates \
    libjpeg-dev libpng-dev libgl1 \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Install pip for Python 3.8
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

# Install PyTorch 2.0.0 (compiled for CUDA 11.8)
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install MMlab ecosystem
RUN pip install openmim \
 && mim install mmcv==2.0.1 \
 && mim install mmdet==3.0.0 \
 && mim install mmsegmentation==1.0.0 \
 && mim install mmdet3d==1.1.1

# Install additional required packages
RUN pip install spconv-cu117 \
 && pip install timm \
 && pip install git+https://github.com/NVIDIA/gpu_affinity

# Optional visualization tools
# RUN pip install pyvirtualdisplay matplotlib==3.7.2 PyQt5 \
#     vtk==9.0.1 mayavi==4.7.3 configobj numpy==1.23.5

# Step 1: Install VTK first
RUN pip install vtk==9.0.1

# Step 2: Then install Mayavi (and rest)
RUN pip install pyvirtualdisplay matplotlib==3.5.3 PyQt5  configobj numpy==1.23.5

RUN pip install mayavi==4.7.3

# Set working directory and copy code
WORKDIR /workspace
COPY . /workspace

RUN nvcc --version && ls /usr/local/cuda/include/cuda_runtime_api.h


# Build custom CUDA ops
RUN CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    CFLAGS="-I/usr/local/cuda/include" \
    CXXFLAGS="-I/usr/local/cuda/include" \
    pip install -e /workspace/model/encoder/gaussian_encoder/ops

# RUN cd /workspace/model/encoder/gaussian_encoder/ops && pip install -e . 

RUN cd /workspace/model/head/localagg && pip install -e . 

# Default container entry
CMD ["/bin/bash"]
# CMD ["python", "-c", "import torch; print(f'CUDA Version: {torch.version.cuda}'); print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"]

# CMD ["bash", "scripts/eval_base.sh", "config/nusc_surroundocc_base_eval.py", "out/ckpt_base.pth", "out/xxxx"]
