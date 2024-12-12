# Base image with CUDA and cuDNN for PyTorch
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install Python 3.7, pip, and development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.7 \
        python3.7-dev \
        python3.7-distutils \
        curl \
        build-essential \
        cmake \
        git && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install pip for Python 3.7
RUN curl -sS https://bootstrap.pypa.io/pip/3.7/get-pip.py | python3.7 && \
    # Set Python 3.7 as the default Python 3
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    # Upgrade pip, setuptools, and wheel
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set the working directory
WORKDIR /app

# Install PyTorch and torchvision (CUDA 11.7)
RUN python3 -m pip install --no-cache-dir torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install LIME and other dependencies
# The presence of build-essential and cmake should allow hnswlib to compile successfully
RUN python3 -m pip install --no-cache-dir \
    lime==0.2.0.1 \
    numpy \
    scipy \
    sympy \
    scikit-learn \
    matplotlib \
    tqdm \
    pygod \
    networkx \
    dgl \
    cloudpickle \
    pickleshare \
    pygsp \
    pandas \
    argparse \
    ruamel.yaml \
    hnswlib \
    transformers \
    datasets 

# Copy your project files into the container
COPY . .
