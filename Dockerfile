FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install Python 3.7 and development tools
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Install pip3 (using the script for Python 3.7)
RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py \
    && python3.7 get-pip.py \
    && rm get-pip.py

# Set the working directory
WORKDIR /app

# Install the specified versions of PyTorch and torchvision (for CUDA 11.7)
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install the LIME interpreter and other dependencies
RUN pip3 install lime==0.2.0.1

# Install other Python packages
RUN pip3 install --no-cache-dir \
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
    ruamel.yaml

# Copy all project files into the container
COPY . .
