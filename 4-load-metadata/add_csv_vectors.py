import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import sys
import torch

FAISS_API_URL = "http://faiss-container:5000/add"
FAISS_API_URL = "http://localhost:5000/add"

print("✅ PyTorch is using GPU:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✅ Loading model on {device}...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("✅ Model loaded successfully on:", device)

# ✅ Function to normalize vectors (for cosine similarity)
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)  # Avoid division by zero

# -------------------------
for i in range(1000):
    csv_file = f"./batch_files/batch_{i+1}.csv"
    df = pd.read_csv(csv_file, low_memory=False, dtype={"id": str})  # ✅ Force `id` as string
    df.columns = [col.strip() for col in df.columns]

    df["combined_text"] = ("content title is: " + df["name"].fillna('') + "\n" +
                           "content type is: " + df["type"].fillna('') + "\n" +
                           "content summary is: " + df["description"].fillna(''))

    metadata = df.apply(lambda row: {
        "id": str(row["id"]),  # ✅ Force all IDs to be strings (UUID-safe)
        "name": row["name"] if pd.notna(row["name"]) else "",
        "type": row["type"] if pd.notna(row["type"]) else "",
        "description": row["description"] if pd.notna(row["description"]) else ""
    }, axis=1).tolist()

    df["combined_text"] = df["combined_text"].astype(str)

    batch = df["combined_text"].tolist()
    embeddings_list = model.encode(batch, convert_to_numpy=True)

    # ✅ Convert to NumPy array
    embeddings = np.vstack(embeddings_list).astype(np.float32)

    # ✅ Normalize embeddings (important for cosine similarity)
    normalized_embeddings = normalize(embeddings)

    batch_vectors = normalized_embeddings.tolist()
    batch_metadata = metadata

    payload = {
        "vectors": batch_vectors,
        "metadata": batch_metadata
    }

    try:
        response = requests.post(FAISS_API_URL, json=payload)
        if response.status_code == 200:
            print(f"✅ Successfully added batch {i} vectors to FAISS.")
        else:
            print(f"❌ Failed to add vectors (Batch {i}). Status Code:", response.status_code)
            print("Response:", response.text)
    except Exception as e:
        print(f"❌ Exception occurred while sending batch {i}:", str(e))
