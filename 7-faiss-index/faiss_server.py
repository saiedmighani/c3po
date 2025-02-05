import faiss
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# FAISS index settings
d = 384  # Vector dimension

# Initialize FAISS index (CPU version) with an empty index
index = faiss.IndexFlatL2(d)
print("✅ FAISS (CPU) vector database initialized.")

# Global metadata store to hold metadata for each vector in the index.
metadata_store = []

# API: Health Check
@app.route('/healthz', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# API: Search FAISS
@app.route('/search', methods=['POST'])
def search():
    # Expecting a query vector in the request JSON
    query_vector = np.array(request.json['vector'], dtype='float32').reshape(1, -1)
    k = request.json.get('top_k', 5)  # Number of nearest neighbors

    # Search in the FAISS index
    distances, indices = index.search(query_vector, k)

    # Retrieve metadata for the found indices
    results_metadata = []
    for idx in indices[0]:
        if idx < len(metadata_store):
            results_metadata.append(metadata_store[idx])
        else:
            results_metadata.append(None)

    return jsonify({
        "distances": distances.tolist(),
        "indices": indices.tolist(),
        "metadata": results_metadata
    })

# API: Add New Vectors with Metadata
@app.route('/add', methods=['POST'])
def add_vectors():
    try:
        # Expect "vectors": a list of vectors and "metadata": a list of metadata objects.
        new_vectors = np.array(request.json['vectors'], dtype='float32')
        new_metadata = request.json.get('metadata', None)

        # Ensure that metadata is provided and its length matches the number of vectors
        if new_metadata is None or len(new_metadata) != new_vectors.shape[0]:
            return jsonify({
                "status": "error",
                "message": "A metadata list matching the number of vectors must be provided."
            }), 400

        # Add new vectors to FAISS index
        index.add(new_vectors)
        # Extend the global metadata store with the new metadata entries
        metadata_store.extend(new_metadata)

        return jsonify({
            "status": "success",
            "message": f"{new_vectors.shape[0]} vectors added!"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Start Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
