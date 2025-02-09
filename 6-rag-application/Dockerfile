# Use Python 3.10 base image
FROM python:3.10

WORKDIR /app

# Install required packages
RUN pip install --no-cache-dir numpy requests sentence-transformers torch

# Copy the RAG application script into the container
COPY rag_application.py .

# Use ENTRYPOINT so command-line arguments are passed to the Python script
ENTRYPOINT ["python", "rag_application.py"]
