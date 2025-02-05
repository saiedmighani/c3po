import time
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests

# Get Qdrant URL (supports Docker containers)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
qdrant_url = f"http://{QDRANT_HOST}:6333/healthz"

timeout = 60  # Wait up to 60 seconds
start_time = time.time()

while True:
    try:
        response = requests.get(qdrant_url)
        if response.status_code == 200:
            print("✅ Qdrant is ready!")
            break
    except requests.exceptions.ConnectionError:
        pass  # Qdrant is still starting

    if time.time() - start_time > timeout:
        print("❌ Qdrant failed to start in time! Check if it's running.")
        exit(1)  # Exit with an error code

    print("⏳ Waiting for Qdrant to start...")
    time.sleep(3)  # Check every 3 seconds

# Connect to Qdrant
client = QdrantClient(f"http://{QDRANT_HOST}:6333")

# Wine descriptions
wines_with_descriptions = [
    "Chateau Margaux a prestigious Bordeaux wine with rich aromas of blackcurrant, tobacco, and licorice.",
    "Penfolds Grange an iconic Australian Shiraz known for its full-bodied structure and deep, complex flavors.",
]

# Collection name
collection_name = "wines5"

# **🔹 Fix: Properly check and create the collection**
if client.collection_exists(collection_name):
    print(f"🗑 Deleting existing collection: {collection_name}")
    client.delete_collection(collection_name)

print(f"🆕 Creating collection: {collection_name}")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=128, distance=Distance.COSINE)
)

# **🔹 Fix: Proper way to add vectors**
print("🚀 Adding vectors to Qdrant...")
vector_data = [
    [0.1] * 128,  # Dummy vector for wine 1
    [0.2] * 128,  # Dummy vector for wine 2
]

points = [
    PointStruct(id=i, vector=vector_data[i], payload={"description": wines_with_descriptions[i]})
    for i in range(len(wines_with_descriptions))
]

client.upload_points(collection_name=collection_name, points=points)

print("✅ Database initialized and documents loaded.")
