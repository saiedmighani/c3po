import faiss
import numpy as np
from flask import Flask, request, jsonify


# Initialize Flask app
app = Flask(__name__)

# FAISS index settings
d = 384  # Vector dimension
nlist = 100  # Number of clusters for IVF (tunable)
nprobe = 10  # Number of clusters to search (tunable)

# ✅ Check FAISS GPU availability
num_gpus = faiss.get_num_gpus()
if num_gpus > 0:
    print(f"✅ FAISS detected {num_gpus} GPU(s), using GPU mode.")
    gpu_resources = faiss.StandardGpuResources()

    # Use IVFFlat for much faster searches (10x speedup)
    quantizer = faiss.IndexFlatL2(d)  # Base quantizer
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(np.random.random((1000, d)).astype("float32"))  # Pre-train with random data

    index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)  # Move to GPU
    index.nprobe = nprobe  # Set search efficiency
    index_device = "gpu"


else:
    print("⚠️ FAISS GPU not available, falling back to CPU mode.")
    index = faiss.IndexFlatL2(d)  # Slower but still functional
    index_device = "cpu"

print("✅ FAISS vector database initialized.")

# Global metadata store
metadata_store = []

# API: Health Check
@app.route('/healthz', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# API: Search FAISS (Supports Batch Queries)
@app.route('/search', methods=['POST'])
def search():
    try:
        query_vectors = np.array(request.json['vector'], dtype='float32')
        if len(query_vectors.shape) == 1:  # If single query, reshape
            query_vectors = query_vectors.reshape(1, -1)

        k = request.json.get('top_k', 5)

        if index.ntotal == 0:
            return jsonify({"status": "error", "message": "FAISS index is empty!"}), 400

        distances, indices = index.search(query_vectors, k)  # Batch search

        results_metadata = []
        for row in indices:
            row_metadata = [metadata_store[idx] if idx < len(metadata_store) else None for idx in row]
            results_metadata.append(row_metadata)

        return jsonify({
            "distances": distances.tolist(),
            "indices": indices.tolist(),
            "metadata": results_metadata
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# API: Add Vectors (Batch Insertion)
@app.route('/add', methods=['POST'])
def add_vectors():
    try:
        new_vectors = np.array(request.json['vectors'], dtype='float32')
        new_metadata = request.json.get('metadata', [])

        if len(new_metadata) != new_vectors.shape[0]:
            return jsonify({"status": "error", "message": "Metadata length mismatch!"}), 400

        index.add(new_vectors)  # Batch insertion
        metadata_store.extend(new_metadata)

        return jsonify({"status": "success", "message": f"{new_vectors.shape[0]} vectors added!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# API: Save FAISS Index to Disk
@app.route('/save', methods=['POST'])
def save_index():
    try:
        faiss.write_index(faiss.index_gpu_to_cpu(index), "faiss_index.bin")
        return jsonify({"status": "success", "message": "FAISS index saved to disk."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# API: Load FAISS Index from Disk
@app.route('/load', methods=['POST'])
def load_index():
    try:
        global index
        index = faiss.read_index("faiss_index.bin")
        print("✅ FAISS index loaded from disk.")
        return jsonify({"status": "success", "message": "FAISS index loaded from disk."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# API: Get FAISS Index Count
@app.route('/index_count', methods=['GET'])
def get_index_count():
    """Returns the number of vectors in the FAISS index."""
    return jsonify({"status": "success", "index_count": index.ntotal})

# API: Get FAISS Vector Dimension
@app.route('/vector_dimension', methods=['GET'])
def get_vector_dimension():
    """Returns the dimension of vectors in the FAISS index."""
    return jsonify({"status": "success", "vector_dimension": index.d})

# API: Check if FAISS is on GPU or CPU
@app.route('/index_device', methods=['GET'])
def get_index_device():
    """Returns whether the FAISS index is running on GPU or CPU."""
    return jsonify({"status": "success", "device": index_device})


# Start Flask API using Gunicorn with async workers
if __name__ == "__main__":
    from os import system
    system("gunicorn -w 4 -k gevent -b 0.0.0.0:5000 faiss_server:app")
