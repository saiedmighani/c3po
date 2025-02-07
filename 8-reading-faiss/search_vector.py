import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# FAISS API URL
FAISS_API_URL = "http://localhost:5000/search"


model = SentenceTransformer("all-MiniLM-L6-v2")
query_embedding = model.encode('pirate movies', convert_to_numpy=True)
query_vector = query_embedding.tolist()

# Send search request
response = requests.post(FAISS_API_URL, json={"vector": query_vector, "top_k": 5})

# Print response
if response.status_code == 200:
    print("✅ FAISS Query Successful!")
    print(response.json())
else:
    print("❌ FAISS Query Failed!")
    print(response.text)
