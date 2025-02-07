# Base image with Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy FAISS index and source code
COPY faiss_index/ /app/faiss_index/
COPY movie_recommender.py /app/movie_recommender.py

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "movie_recommender:app", "--host", "0.0.0.0", "--port", "8000"]
