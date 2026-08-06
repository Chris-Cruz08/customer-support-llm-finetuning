# Base image with CUDA 12.1 support to match the training environment
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install PyTorch with CUDA 12.1 support first (special index required)
RUN pip3 install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code and the trained LoRA adapter
COPY src/ ./src/
COPY qwen_qlora_adapter/ ./qwen_qlora_adapter/

# Gradio's default port
EXPOSE 7860

# Environment variables for HuggingFace cache inside the container
ENV HF_HOME=/app/hf_cache
ENV HF_DATASETS_CACHE=/app/hf_cache/datasets
ENV HF_HUB_CACHE=/app/hf_cache/hub

CMD ["python3", "src/app.py"]