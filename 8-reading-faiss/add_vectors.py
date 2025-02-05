import requests
import numpy as np

# FAISS API URL
FAISS_API_URL = "http://localhost:5000/add"

# Generate new vectors (5 new 128-dim vectors)
new_vectors = np.random.random((5, 128)).astype('float32').tolist()

# Send the request to add vectors
response = requests.post(FAISS_API_URL, json={"vectors": new_vectors})

# Print response
if response.status_code == 200:
    print("✅ Successfully added new vectors!")
    print(response.json())
else:
    print("❌ Failed to add vectors!")
    print(response.text)
