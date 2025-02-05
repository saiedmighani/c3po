import os
import sys
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# FAISS API endpoint (assuming FAISS container is running on your user-defined network)
FAISS_API_URL = "http://faiss-container:5000/search"

# Ollama API endpoint (Ollama runs on port 11434 by default)
OLLAMA_API_URL = "http://ollama-container:11434/api/generate"

# --- Functions ---

def query_faiss(query_text, top_k=10):
    """
    Encode the query using Sentence Transformer and query FAISS.
    """
    # Encode query text into a 384-dimensional vector using Sentence Transformer.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query_text, convert_to_numpy=True)
    query_vector = query_embedding.tolist()

    payload = {
        "vector": query_vector,
        "top_k": top_k
    }

    try:
        response = requests.post(FAISS_API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("FAISS query error:", e)
        return None

def query_ollama(model_name, prompt):
    """
    Query the Ollama API with a given model and prompt.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # Single response
    }
    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received.")
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def main():
    # Determine the user query:
    # 1. If a command-line argument is provided, use that.
    # 2. Otherwise, check for an environment variable USER_QUERY.
    # 3. Otherwise, fall back to interactive input.
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = os.environ.get("USER_QUERY")
        if not user_query:
            try:
                user_query = input("Enter your query: ")
            except EOFError:
                print("No input provided and no query found in USER_QUERY; exiting.")
                sys.exit(1)

    # 2. Query FAISS for the top 10 similar items
    faiss_results = query_faiss(user_query, top_k=10)
    if not faiss_results:
        print("No results from FAISS.")
        return

    # 3. Extract metadata from FAISS results
    # (Assumes FAISS response includes "metadata" key with list of metadata dictionaries.)
    metadata_list = faiss_results.get("metadata", [])
    context_parts = []
    for meta in metadata_list:
        if meta is not None:
            # Using "entity-title" as context; adjust as needed.
            title = meta.get("entity-title", "")
            context_parts.append(title)
    context_text = "\n".join(context_parts)

    # 4. Build a prompt for Ollama using the user query and context from FAISS
    prompt = f"User Query: {user_query}\n\nRelevant Documents:\n{context_text}\n\nBased on the above context, please provide an answer."

    # 5. Query Ollama with the prompt
    model_name = "llama3.2"  # Replace with your desired model name
    ollama_response = query_ollama(model_name, prompt)

    # 6. Output the result
    print("\nOllama Response:")
    print(ollama_response)

if __name__ == "__main__":
    main()
