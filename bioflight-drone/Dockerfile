# Use NVIDIA's official CUDA base image with cuDNN and Ubuntu
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /bioflight-drone

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget unzip nano libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy your project into the container
COPY . /bioflight-drone

# Install remaining project dependencies
RUN pip3 install -r requirements.txt

# Default command
CMD ["bash"]
