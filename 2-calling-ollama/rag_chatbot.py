from flask import Flask, request, jsonify
import redis
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import requests

app = Flask(__name__)

# Redis connection
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# API endpoints
OLLAMA_API_URL = "http://localhost:11434/api/generate"
FAISS_API_URL = "http://localhost:5000/search"

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # FAISS search
intent_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")  # Intent detection

MAX_HISTORY = 20  # Limit chat history length

def get_greeting():
    """Return an appropriate greeting based on the current time."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning! ☀️ I'm your movie and series recommendation bot. What are you in the mood for?"
    elif hour < 18:
        return "Good afternoon! 🎬 Looking for something to watch? I can help you find movies and series."
    else:
        return "Good evening! 🍿 Tell me what you're in the mood to watch, and I'll find something for you."

def normalize(vectors):
    """Normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + 1e-10)

def item_retrieval(query_text):
    """Retrieve similar items from FAISS."""
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True)
    query_embedding = normalize(query_embedding)
    query_vector = query_embedding.tolist()

    payload = {"vector": query_vector, "top_k": 10}
    response = requests.post(FAISS_API_URL, json=payload)

    retrieved_items = []
    if response.status_code == 200:
        result = response.json()
        for distances, metadata_list in zip(result.get("distances", []), result.get("metadata", [])):
            for distance, metadata in zip(distances, metadata_list):
                if distance > 0.3:  # Filter weak matches
                    retrieved_items.append(
                        f"📍 **{metadata.get('name', 'N/A')}** (Score: {distance:.2f})\n"
                        f"  - 🎭 Type: {metadata.get('type', 'N/A')}\n"
                        f"  - 📝 Description: {metadata.get('description', 'N/A')}"
                    )
        return retrieved_items[:3] if retrieved_items else None
    return None

def query_ollama(model_name, prompt, retrieved_titles=None, max_words=20):
    """Query Ollama API while ensuring responses are engaging and relevant."""
    headers = {"Content-Type": "application/json"}

    # Prepare movie-based fun commentary
    if retrieved_titles:
        formatted_titles = ", ".join(retrieved_titles)
        extra_context = (
            f"You're in luck! I found some interesting picks: {formatted_titles}. "
            "Let me know if one catches your eye! 🎬"
        )
    else:
        extra_context = ""

    instruction = (
        "You are a witty and engaging movie and series recommendation assistant. "
        "Keep responses short, fun, and conversational.\n"
        "- When asked about movies or series, introduce interesting titles creatively.\n"
        "- If retrieved content exists, weave it into the response naturally.\n"
        "- Do not invent or recommend movies yourself. Use only what is retrieved.\n"
        "- Keep responses under {max_words} words."
    )

    payload = {
        "model": model_name,
        "prompt": f"{instruction}\nUser: {prompt}\nSearchBot: {extra_context}",
        "stream": False,
        "temperature": 0.1
    }

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json().get("response", "No response received.").strip()
    except requests.exceptions.RequestException as e:
        return f"Error occurred: {str(e)}"

def detect_intent(user_input):
    """Detect if the user wants a recommendation or general conversation."""
    candidate_labels = ["movie recommendation", "general conversation"]
    result = intent_classifier(user_input, candidate_labels)
    return "movie_recommendation" if result["labels"][0] == "movie recommendation" and result["scores"][0] > 0.5 else "general_conversation"

def save_to_redis(profile_id, user_input, bot_response):
    """Save chat history to Redis."""
    history_key = f"history:{profile_id}"
    redis_client.rpush(history_key, json.dumps({"user": user_input, "bot": bot_response}))
    redis_client.ltrim(history_key, -MAX_HISTORY, -1)

def get_history_from_redis(profile_id):
    """Retrieve the last MAX_HISTORY messages."""
    history_key = f"history:{profile_id}"
    history = redis_client.lrange(history_key, -MAX_HISTORY, -1)
    return [json.loads(msg) for msg in history] if history else []

@app.route("/call_rag", methods=["POST"])
def call_rag():
    data = request.json
    user_input = data.get("user_input", "")
    profile_id = data.get("profile_id", "")

    if not user_input or not profile_id:
        return jsonify({"error": "Missing user_input or profile_id"}), 400

    intent = detect_intent(user_input)
    response_data = {"response": "", "faiss_results": [], "intent": intent, "profile_id": profile_id}

    if intent == "movie_recommendation":
        faiss_results = item_retrieval(user_input)

        if faiss_results:
            retrieved_titles = [result.split("**")[1] for result in faiss_results]  # Extract movie names
            bot_response = query_ollama("llama3.1:8b-instruct-q2_K", user_input, retrieved_titles, max_words=20)
            response_data.update({"response": bot_response, "faiss_results": faiss_results})
        else:
            bot_response = "I couldn't find anything for that. Maybe try a different genre or title? 🎬"
            response_data["response"] = bot_response
    else:
        history = get_history_from_redis(profile_id)
        full_prompt = "\n".join([f"User: {h['user']}\nSearchBot: {h['bot']}" for h in history] + [f"User: {user_input}"]) + "\nSearchBot:"
        bot_response = query_ollama("llama3.1:8b-instruct-q2_K", full_prompt)
        response_data["response"] = bot_response

    save_to_redis(profile_id, user_input, bot_response)
    response_data["history"] = get_history_from_redis(profile_id)

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
