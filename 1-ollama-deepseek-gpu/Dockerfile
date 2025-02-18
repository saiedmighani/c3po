# Use the official Ollama image
FROM ollama/ollama:latest

# Set working directory
WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y curl gnupg ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the installation script
COPY install_nvidia_toolkit.sh /app/install_nvidia_toolkit.sh

# Grant execute permissions
RUN chmod +x /app/install_nvidia_toolkit.sh

# Run the script and start Ollama
ENTRYPOINT ["/bin/bash", "/app/install_nvidia_toolkit.sh"]
