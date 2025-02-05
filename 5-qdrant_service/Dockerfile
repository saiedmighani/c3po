# Use the official Qdrant image
FROM qdrant/qdrant:latest

# Set working directory
WORKDIR /app

# Install Python & Pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirement packages
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy application files
COPY . .

# Expose Qdrant API port
EXPOSE 6333

# Run the Python script, then start Qdrant as the main process
CMD python3 main.py && exec /qdrant/qdrant
