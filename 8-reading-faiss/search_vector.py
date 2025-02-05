import requests
import numpy as np

# FAISS API URL
FAISS_API_URL = "http://localhost:5000/search"

# Generate a random query vector (128-dim)
query_vector = np.random.random(128).astype('float32').tolist()

# Send search request
response = requests.post(FAISS_API_URL, json={"vector": query_vector, "top_k": 5})

# Print response
if response.status_code == 200:
    print("✅ FAISS Query Successful!")
    print(response.json())
else:
    print("❌ FAISS Query Failed!")
    print(response.text)
