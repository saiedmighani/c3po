import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# -------------------------
# 1. Load the CSV file
# -------------------------
csv_file = "movies_shows_dataset.csv"
df = pd.read_csv(csv_file)

# Clean up column names (strip any extra whitespace)
df.columns = [col.strip() for col in df.columns]

# -------------------------
# 2. Combine entity-description and 1-sentence-summary into a single text field
# -------------------------
df["combined_text"] = df["entity-description"] + " " + df["1-sentence-summary"]

# -------------------------
# 3. Create metadata from id and entity-title
# -------------------------
# Each metadata record is a dictionary with 'id' and 'entity-title'
metadata = df.apply(lambda row: {"id": row["id"], "entity-title": row["entity-title"]}, axis=1).tolist()

# -------------------------
# 4. Load the Sentence Transformer model and encode texts
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
# Encode all combined texts; the output shape will be (N, 384)
embeddings = model.encode(df["combined_text"].tolist(), convert_to_numpy=True)

# Convert the embeddings into a list of vectors
vectors = embeddings.tolist()

# -------------------------
# 5. Prepare the payload and send it to the FAISS server
# -------------------------
payload = {
    "vectors": vectors,
    "metadata": metadata
}

# The FAISS server is assumed to be accessible via hostname "faiss-container" on port 5000.
FAISS_API_URL = "http://faiss-container:5000/add"

try:
    response = requests.post(FAISS_API_URL, json=payload)
    if response.status_code == 200:
        print("✅ Successfully added vectors to FAISS.")
        print("Response:", response.json())
    else:
        print("❌ Failed to add vectors to FAISS.")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
except Exception as e:
    print("❌ Exception occurred while sending vectors:", str(e))
