import faiss
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# FAISS index settings
d = 384  # Vector dimension

# ✅ Check FAISS GPU availability
num_gpus = faiss.get_num_gpus()
if num_gpus > 0:
    print(f"✅ FAISS detected {num_gpus} GPU(s), using GPU mode.")
    gpu_resources = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(d)  # CPU index
    index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)  # Move to GPU
else:
    print("⚠️ FAISS GPU not available, falling back to CPU mode.")
    index = faiss.IndexFlatL2(d)

print("✅ FAISS vector database initialized.")

# Global metadata store
metadata_store = []

# API: Health Check
@app.route('/healthz', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# API: Search FAISS
@app.route('/search', methods=['POST'])
def search():
    try:
        query_vector = np.array(request.json['vector'], dtype='float32').reshape(1, -1)
        k = request.json.get('top_k', 5)

        if index.ntotal == 0:
            return jsonify({"status": "error", "message": "FAISS index is empty!"}), 400

        distances, indices = index.search(query_vector, k)
        results_metadata = [metadata_store[idx] if idx < len(metadata_store) else None for idx in indices[0]]

        return jsonify({
            "distances": distances.tolist(),
            "indices": indices.tolist(),
            "metadata": results_metadata
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# API: Add New Vectors with Metadata
@app.route('/add', methods=['POST'])
def add_vectors():
    try:
        new_vectors = np.array(request.json['vectors'], dtype='float32')
        new_metadata = request.json.get('metadata', [])

        if len(new_metadata) != new_vectors.shape[0]:
            return jsonify({"status": "error", "message": "Metadata length mismatch!"}), 400

        index.add(new_vectors)
        metadata_store.extend(new_metadata)

        return jsonify({"status": "success", "message": f"{new_vectors.shape[0]} vectors added!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Start Flask API using Gunicorn
if __name__ == "__main__":
    from os import system
    system("gunicorn -w 2 -b 0.0.0.0:5000 faiss_server:app")
