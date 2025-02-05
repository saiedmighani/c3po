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

# Start Ollama in the background
echo "Starting Ollama..."
ollama serve &

# Wait a few seconds to ensure Ollama is running
sleep 5

# Run Llama3.2 as a persistent process
echo "Running Llama3.2..."
ollama run llama3.2 &

# Keep the container running indefinitely
tail -f /dev/null
