FROM daangn/faiss

# Set the working directory
WORKDIR /app

# Copy the script into the container
COPY test.py /app/test.py

# Default command
CMD ["python", "test.py"]
