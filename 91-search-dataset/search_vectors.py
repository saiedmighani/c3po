import requests
from sentence_transformers import SentenceTransformer
import numpy as np

# FAISS API URL (adjust if needed)
FAISS_API_URL = "http://localhost:5000/search"

# Read query from user input
query_text = input("Enter your query: ")

# Load the Sentence Transformer model (produces 384-d vectors)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode the query text into a 384-dimensional vector
query_embedding = model.encode(query_text, convert_to_numpy=True)
query_vector = query_embedding.tolist()  # Convert to list for JSON serialization

# Prepare payload: specify top_k=10 to retrieve the top 10 similar items
payload = {
    "vector": query_vector,
    "top_k": 10
}

# Send search request to the FAISS server
response = requests.post(FAISS_API_URL, json=payload)

# Process and print the response
if response.status_code == 200:
    result = response.json()
    print("✅ FAISS Query Successful!")
    print("Similarity Scores (Distances):")
    print(result.get("distances"))
    print("\nIndices:")
    print(result.get("indices"))
    print("\nMetadata:")
    print(result.get("metadata"))
else:
    print("❌ FAISS Query Failed!")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
