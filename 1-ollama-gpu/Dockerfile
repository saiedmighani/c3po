# Use an NVIDIA CUDA base image to ensure the Ollama image supports GPU
# Ensure to match the CUDA version with your GPU and software requirements
FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install curl and gnupg before running the script
RUN apt-get update && apt-get install -y curl gnupg && rm -rf /var/lib/apt/lists/*

# Copy the installation script into the container
COPY install_nvidia_toolkit.sh /app/install_nvidia_toolkit.sh

# Grant execute permissions
RUN chmod +x /app/install_nvidia_toolkit.sh

# Ensure NVIDIA's runtime is used
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run the script, start Ollama, and keep the container running
ENTRYPOINT ["/bin/bash", "/app/install_nvidia_toolkit.sh"]
