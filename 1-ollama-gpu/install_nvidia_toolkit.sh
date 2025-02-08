#!/bin/bash

set -e  # Exit on error

echo "Installing dependencies..."
apt-get update && apt-get install -y curl gnupg

echo "Installing NVIDIA Container Toolkit..."

# Add NVIDIA repository key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA container toolkit repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package lists and install NVIDIA Container Toolkit
apt-get update && apt-get install -y nvidia-container-toolkit

echo "NVIDIA Container Toolkit installed successfully."

# Verify if GPU is available
if ! command -v nvidia-smi &> /dev/null
then
    echo "NVIDIA driver not installed or not working, exiting..."
    exit 1
fi

# Start Ollama with GPU support
echo "Starting Ollama with GPU..."
ollama serve --gpu &  # Ensure the command to start Ollama has a flag to use GPU, if applicable

# Wait a few seconds to ensure Ollama is running
sleep 5

# Run Llama3.2 as a persistent process with GPU support
echo "Running Llama3.2 on GPU..."
ollama run --model llama3.2 --gpu &  # Ensure that there is a flag to use GPU

# Keep the container running indefinitely
tail -f /dev/null
