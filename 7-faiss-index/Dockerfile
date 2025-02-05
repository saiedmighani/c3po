# Use Python base image
FROM python:3.10

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN pip install --no-cache-dir faiss-cpu flask numpy

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Expose API port
EXPOSE 5000

# Start FAISS server
CMD ["python", "faiss_server.py"]
