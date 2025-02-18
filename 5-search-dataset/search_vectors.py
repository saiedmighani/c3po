import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import time  # ✅ Import time module for latency measurement

# FAISS API URL (adjust if needed)
FAISS_API_URL = "http://localhost:5000/search"

# ✅ Function to normalize vectors (for cosine similarity)
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + 1e-10)  # Avoid division by zero

# Read query from user input
query_text = input("Enter your query: ")

# Load the Sentence Transformer model (produces 384-d vectors)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Measure latency: Start time before request
start_time = time.time()

# Encode the query text into a 384-dimensional vector
query_embedding = model.encode(query_text, convert_to_numpy=True)

# ✅ Normalize the query embedding
query_embedding = normalize(query_embedding)

query_vector = query_embedding.tolist()  # Convert to list for JSON serialization

# Prepare payload: specify top_k=10 to retrieve the top 10 similar items
payload = {
    "vector": query_vector,
    "top_k": 10
}

# Send search request to the FAISS server
response = requests.post(FAISS_API_URL, json=payload)

# ✅ Measure latency: End time after response
end_time = time.time()
latency_ms = (end_time - start_time) * 1000  # Convert seconds to milliseconds

# Process and print the response
if response.status_code == 200:
    result = response.json()
    print("✅ FAISS Query Successful!")
    print(f"⚡ Query Latency: {latency_ms:.2f} ms\n")

    print("🔍 **Retrieved Items (Filtered by Distance > 0.4):**")

    for i, (distances, metadata_list) in enumerate(zip(result.get("distances", []), result.get("metadata", []))):
        for j, (distance, metadata) in enumerate(zip(distances, metadata_list)):
            if distance > 0.4:  # ✅ Filter out weak matches
                print(f"\n📍 **Result {i+1}-{j+1}:** (Similarity Score: {distance:.2f})")
                print(f"  - 🆔 ID: {metadata.get('id', 'N/A')}")
                print(f"  - 📌 Name: {metadata.get('name', 'N/A')}")
                print(f"  - 🎭 Type: {metadata.get('type', 'N/A')}")
                print(f"  - 📝 Description: {metadata.get('description', 'N/A')}")
                print("-" * 40)  # Separator for readability

else:
    print("❌ FAISS Query Failed!")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
