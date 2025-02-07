import os
import sys
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
FAISS_API_URL = "http://localhost:5000/search"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# --- Functions ---

def is_movie_related(query):
    """
    Determines if a query is related to movies using keyword matching.
    """
    movie_keywords = [
        "movie", "film", "cinema", "Hollywood", "Bollywood", "Netflix", "Hulu",
        "Amazon Prime", "director", "actor", "actress", "recommend", "suggest",
        "watch", "TV show", "series", "box office", "theater", "trailer"
    ]

    query_lower = query.lower()

    return any(keyword in query_lower for keyword in movie_keywords)

def query_faiss(query_text, top_k=10, distance_threshold=2.0):
    """
    Encode the query using Sentence Transformer and query FAISS.
    Filters out results where distance > distance_threshold.
    """
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
        results = response.json()

        # Filter results based on distance threshold
        filtered_metadata = []
        for i, meta in enumerate(results.get("metadata", [])):
            distance = results.get("distances", [])[0][i]
            if distance <= distance_threshold:
                filtered_metadata.append(meta)

        return filtered_metadata

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
    # Get user query
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

    # **Guardrail: Reject Non-Movie Queries**
    if not is_movie_related(user_query):
        print("\nDarling, can I offer you any movie recommendation? 🎬🍿")
        return

    # Query FAISS for the top 10 similar items and apply thresholding
    filtered_metadata_list = query_faiss(user_query, top_k=10, distance_threshold=2.0)

    if not filtered_metadata_list:
        print("No suitable results from FAISS.")
        return

    # Extract movie titles from metadata
    context_parts = []
    for meta in filtered_metadata_list:
        if meta is not None:
            title = meta.get("1-sentence summary", "")
            context_parts.append(title)
    context_text = "\n".join(context_parts)

    # **Strict Prompt for Ollama with Fun Intro & No Extra Text After Recommendations**
    prompt = f"""User Query: {user_query}

Relevant Documents:
{context_text}

You are a fun, engaging, and concise movie recommendation assistant. 🎬🍿 
Begin with a SHORT and engaging introduction that matches the user's query. 
Then, list up to 10 movie recommendations ranked by relevance.

STRICT RULES:
- **Start with 1-2 engaging sentences** to set the tone.
- **Then, ONLY list the recommendations** (each on a new line).
- **DO NOT include any extra text after the recommendations.** No summaries, no closing statements, no additional commentary.

If a question is rude, inappropriate, or unrelated to movie recommendations, respond with:
"Darling, can I offer you any movie recommendation?" and NOTHING ELSE.

Format Example:
"You're in for a cinematic ride! Here are some top picks just for you:"
1. Movie Title 1
2. Movie Title 2
3. Movie Title 3
...
"""

    # Query Ollama
    model_name = "llama3.2"  # Replace with your desired model name
    ollama_response = query_ollama(model_name, prompt)

    # **Enhanced Output with Fun Presentation**
    print("\n🎬 Hulu-eyah! Here are my top picks for you: 🍿\n")
    print(ollama_response)  # Directly print the LLM response (it should only be recommendations)

if __name__ == "__main__":
    main()
