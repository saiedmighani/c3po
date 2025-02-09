# Use NVIDIA's official CUDA image for GPU support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install Python, dependencies, and debugging tools
RUN apt-get update && apt-get install -y python3 python3-pip curl net-tools && \
    rm -rf /var/lib/apt/lists/*

# Install FAISS GPU for CUDA 12, Flask, and Gunicorn
RUN pip3 install --no-cache-dir "numpy<2" faiss-gpu-cu12 flask gunicorn

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Expose API port
EXPOSE 5000

# Start FAISS server using Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--chdir", "/app", "faiss_server:app"]
