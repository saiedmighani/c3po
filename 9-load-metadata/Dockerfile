# Use a Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install required packages:
# - pandas, numpy for CSV handling
# - sentence-transformers, torch for encoding
# - scikit-learn for PCA
# - requests for HTTP calls
RUN pip install --no-cache-dir pandas numpy sentence-transformers torch scikit-learn requests

# Copy the CSV file and the helper script into the image
COPY movies_shows_dataset.csv .
COPY add_csv_vectors.py .

# Run the helper script when the container starts
CMD ["python", "add_csv_vectors.py"]
