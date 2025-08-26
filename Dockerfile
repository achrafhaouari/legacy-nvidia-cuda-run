FROM ubuntu:18.04

# Prevent interactive tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gnupg \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA 10.2 repo + key
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin \
    && mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb \
    && apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub \
    && apt-get update \
    && apt-get -y install cuda

# Set CUDA environment
ENV PATH=/usr/local/cuda-10.2/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

# Install a CuPy version compatible with CUDA 10.2
RUN pip3 install cupy-cuda102==9.6.0
