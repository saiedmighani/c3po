#!/bin/bash

set -e

echo "Checking for NVIDIA GPU inside Docker..."
if ! command -v nvidia-smi &> /dev/null
then
    echo "ERROR: NVIDIA GPU not detected inside the container!"
    exit 1
fi

echo "NVIDIA GPU detected. Installing CUDA and NVIDIA Container Toolkit..."

# Get Ubuntu version
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# Add NVIDIA package repository (without sudo)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list and install CUDA tools & NVIDIA Container Toolkit
apt-get update && apt-get install -y \
    nvidia-container-toolkit-base \
    cuda-command-line-tools-12-1 \
    && rm -rf /var/lib/apt/lists/*

echo "Verifying CUDA installation..."
nvidia-smi

echo "Starting Ollama with GPU support..."
ollama serve &  # Run Ollama in background

# Wait for Ollama to be ready before running Llama3.2
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/generate > /dev/null; do
    sleep 2
    echo "Still waiting for Ollama..."
done

# Run Llama3.2 as a persistent process
echo "Running deepseek-r1:1.5b..."
ollama run deepseek-r1:1.5b &

# Keep the container running indefinitely
tail -f /dev/null
