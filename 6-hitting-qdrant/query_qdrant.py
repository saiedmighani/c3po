import requests

# Qdrant API URL
QDRANT_URL = "http://localhost:6333"

# Check Qdrant health
def check_qdrant_health():
    health_endpoint = f"{QDRANT_URL}/healthz"
    try:
        response = requests.get(health_endpoint)
        if response.status_code == 200:
            print("✅ Qdrant is running:", response.json())
        else:
            print("⚠️ Qdrant health check failed:", response.status_code, response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Qdrant. Is the container running?")

# Fetch list of collections
def get_collections():
    collections_endpoint = f"{QDRANT_URL}/collections"
    try:
        response = requests.get(collections_endpoint)
        if response.status_code == 200:
            print("📂 Collections in Qdrant:", response.json())
        else:
            print("⚠️ Failed to retrieve collections:", response.status_code, response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Qdrant.")

# Query a collection
def search_collection(collection_name):
    search_endpoint = f"{QDRANT_URL}/collections/{collection_name}/points/search"

    query_payload = {
        "vector": [0.1] * 128,  # Dummy query vector
        "top": 2  # Get top 2 closest results
    }

    try:
        response = requests.post(search_endpoint, json=query_payload)
        if response.status_code == 200:
            print(f"🔎 Search results from {collection_name}:", response.json())
        else:
            print(f"⚠️ Search failed for {collection_name}:", response.status_code, response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Qdrant.")

# Run the tests
if __name__ == "__main__":
    check_qdrant_health()  # Check if Qdrant is running
    get_collections()  # List available collections
    search_collection("wines5")  # Search inside the "wines5" collection
