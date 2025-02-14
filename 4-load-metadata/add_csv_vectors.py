import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import sys
import torch
print("✅ PyTorch is using GPU:", torch.cuda.is_available())

from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✅ Loading model on {device}...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print("✅ Model loaded successfully on:", device)
# sys.exit()
# -------------------------
# 1. Load the CSV file
# -------------------------
csv_file = "cleaned_dataset_no_duplicates.csv"

# ✅ Use low_memory=False to avoid mixed type warnings
df = pd.read_csv(csv_file, low_memory=False, dtype={"id": str})  # ✅ Force `id` as string

# Clean up column names (strip any extra whitespace)
df.columns = [col.strip() for col in df.columns]

# -------------------------
# 2. Combine entity-description and handle missing values
# -------------------------
df["combined_text"] = df["name"].fillna('') + "\n" + df["description"].fillna('')

# -------------------------
# 3. Create metadata (Keep id as string)
# -------------------------
metadata = df.apply(lambda row: {
    "id": str(row["id"]),  # ✅ Force all IDs to be strings (UUID-safe)
    "name": row["name"] if pd.notna(row["name"]) else "",
    "description": row["description"] if pd.notna(row["description"]) else ""
}, axis=1).tolist()

# -------------------------
# 4. Load Sentence Transformer and encode texts
# -------------------------
# ✅ Use GPU for fast embedding generation
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Ensure all text fields are **strings**
df["combined_text"] = df["combined_text"].astype(str)

# ✅ Batch processing for efficiency
batch_size = 5000
embeddings_list = []
for i in range(0, len(df), batch_size):
    batch = df["combined_text"].iloc[i : i + batch_size].tolist()
    batch_embeddings = model.encode(batch, convert_to_numpy=True)
    embeddings_list.append(batch_embeddings)

# ✅ Convert embeddings into a single NumPy array (optimized for FAISS)
embeddings = np.vstack(embeddings_list).astype(np.float32)

# -------------------------
# 5. Send Data to FAISS in Batches
# -------------------------
FAISS_API_URL = "http://faiss-container:5000/add"

batch_size_faiss = 5000

for i in range(0, len(embeddings), batch_size_faiss):
    batch_vectors = embeddings[i : i + batch_size_faiss].tolist()
    batch_metadata = metadata[i : i + batch_size_faiss]

    payload = {
        "vectors": batch_vectors,
        "metadata": batch_metadata
    }

    try:
        response = requests.post(FAISS_API_URL, json=payload)
        if response.status_code == 200:
            print(f"✅ Successfully added {len(batch_vectors)} vectors to FAISS.")
        else:
            print(f"❌ Failed to add vectors (Batch {i}). Status Code:", response.status_code)
            print("Response:", response.text)
    except Exception as e:
        print(f"❌ Exception occurred while sending batch {i}:", str(e))

print("✅ All vectors added successfully!")
