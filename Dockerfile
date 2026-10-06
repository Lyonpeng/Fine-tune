# Acacia Model Fine-Tuning — Node/Express web app + Detectron2 (PyTorch/CUDA) training.
#
# Detectron2 requirements (https://detectron2.readthedocs.io/en/latest/tutorials/install.html):
#   Linux · Python >= 3.7 · PyTorch >= 1.8 with matching torchvision · gcc & g++ >= 5.4
#   and a CUDA toolkit (nvcc) matching the CUDA version PyTorch was built with.
# The -devel PyTorch image provides all of these: Python 3.11, PyTorch 2.5.0 built for
# CUDA 12.1, and the CUDA 12.1 toolkit — the same versions as the local detectron_env.
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel

# Detectron2 commit to install (same as the local D:\Project\ModelTraining\detectron2 checkout)
ARG DETECTRON2_REF=02b5c4e295e990042a714712c21dc79b731e8833
# No GPU is visible during `docker build`, so list the GPU architectures to compile
# for: 7.0 V100 · 7.5 T4/RTX 20xx · 8.0 A100 · 8.6 A10/RTX 30xx · 8.9 L4/RTX 40xx · 9.0 H100
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    PORT=5000 \
    PYTHON_BIN=python \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FVCORE_CACHE=/opt/fvcore-cache

# System deps:
#  - git                     : pip fetches detectron2 straight from GitHub
#  - build-essential / ninja : gcc/g++ to compile detectron2's C++/CUDA ops
#  - libgl1 / libglib2.0-0   : runtime libs for OpenCV (pulled in by detectron2 deps)
#  - nodejs                  : Node 20 (via NodeSource) to run the Express server
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Detectron2, built from source in one line (per the install guide) ---
# --no-build-isolation lets its setup.py see the torch already in the image.
RUN python -m pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_REF}"

# --- Remaining Python deps of train_net.py ---
COPY training/requirements.txt ./training/requirements.txt
RUN python -m pip install --no-cache-dir -r training/requirements.txt

# --- ImageNet-pretrained backbone used when "Transfer learning" is Yes ---
# Baked in so the server can train without internet access. The path mirrors where
# detectron2 caches "detectron2://ImageNetPretrained/MSRA/R-50.pkl" under FVCORE_CACHE.
RUN curl -fsSL --create-dirs \
        -o ${FVCORE_CACHE}/detectron2/ImageNetPretrained/MSRA/R-50.pkl \
        https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl

# --- Node deps (production only) ---
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# --- Application source (see .dockerignore for what is excluded) ---
COPY . .

EXPOSE 5000
CMD ["node", "index.js"]
