# Use the official Ollama image
FROM ollama/ollama:latest

# Set working directory
WORKDIR /app

# Install curl and gnupg before running the script
RUN apt-get update && apt-get install -y curl gnupg && rm -rf /var/lib/apt/lists/*

# Copy the installation script into the container
COPY install_nvidia_toolkit.sh /app/install_nvidia_toolkit.sh

# Grant execute permissions
RUN chmod +x /app/install_nvidia_toolkit.sh

# Run the script, start Ollama, and keep the container running
ENTRYPOINT ["/bin/bash", "/app/install_nvidia_toolkit.sh"]
