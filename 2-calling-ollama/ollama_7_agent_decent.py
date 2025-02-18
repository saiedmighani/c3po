import requests
import json
import time
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

# API endpoints
OLLAMA_API_URL = "http://localhost:11434/api/generate"
FAISS_API_URL = "http://localhost:5000/search"

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # FAISS search

MAX_HISTORY = 15

def get_greeting():
    """Return an appropriate greeting based on the current time."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning! ☀️ Hulue-yah! What are you in the mood to watch today? 🎬"
    elif hour < 18:
        return "Good afternoon! 🌞 Hulue-yah! Looking for something fun to watch? 🎬"
    else:
        return "Good evening! 🌙 Hulue-yah! Let's find the perfect movie or show for you. 🎬"

def normalize(vectors):
    """Normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + 1e-10)  # Avoid division by zero

def item_retrieval(query_text):
    """Retrieve items similar to the query using FAISS."""
    start_time = time.time()
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True)
    query_embedding = normalize(query_embedding)
    query_vector = query_embedding.tolist()

    payload = {"vector": query_vector, "top_k": 10}
    response = requests.post(FAISS_API_URL, json=payload)

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    retrieved_items = []
    if response.status_code == 200:
        result = response.json()

        for distances, metadata_list in zip(result.get("distances", []), result.get("metadata", [])):
            for distance, metadata in zip(distances, metadata_list):
                if distance > 0.4:  # Filter weak matches
                    retrieved_items.append(
                        f"📍 **{metadata.get('name', 'N/A')}** (Score: {distance:.2f})\n"
                        f"  - 🎭 Type: {metadata.get('type', 'N/A')}\n"
                        f"  - 📝 Description: {metadata.get('description', 'N/A')}"
                    )

        if retrieved_items:
            return f":\n{''.join(retrieved_items[:3])}", latency_ms  # Return top 3 matches
        else:
            return None, latency_ms  # No strong matches found
    else:
        return None, 0

def query_ollama(model_name, prompt, max_words=20):
    """Send a query to Ollama's local API."""
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.6
    }

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip().lower()
    except requests.exceptions.RequestException:
        return "error"

# ✅ **Proper Intent Detection - Understands Small Talk vs. Movie Queries**
def detect_intent(user_input):
    """Uses DeepSeek (via Ollama) to classify intent properly."""
    system_prompt = (
        "Classify the user's intent into one of these categories:\n"
        "- 'movie_recommendation': If they are asking for movie or TV show suggestions.\n"
        "- 'general_conversation': If they are making small talk (e.g., 'how are you?', 'hello').\n"
        "- 'reject': Only if the message is COMPLETELY unrelated to entertainment (math, politics, science, random life events, etc.).\n"
        "Return ONLY the category name, nothing else."
    )

    input_text = f"{system_prompt}\nUser: {user_input}\nIntent:"
    result = query_ollama("deepseek-r1:1.5b", input_text, max_words=3)

    # ✅ Ensuring intent classification is valid
    valid_intents = {"movie_recommendation", "general_conversation", "reject"}
    if result in valid_intents:
        return result
    return "general_conversation"  # Default to conversation if uncertain

# ✅ **Fix General Chat to Handle Small Talk Normally**
def general_chat(user_input):
    """Handles small talk and casual entertainment-related discussions."""
    system_prompt = (
        "You are a friendly chatbot that ONLY talks about movies, TV shows, and streaming. "
        "If the user makes small talk (e.g., 'how are you?'), respond naturally. "
        "If the user brings up a completely off-topic subject (math, history, politics), steer them back to entertainment."
    )

    input_text = f"{system_prompt}\nUser: {user_input}\nAssistant:"
    return query_ollama("deepseek-r1:1.5b", input_text, max_words=20)

# ✅ **FAISS is the Only Source of Recommendations**
def content_retrieval(user_input):
    """Retrieves FAISS results ONLY for content-related queries."""
    faiss_results, retrieval_latency = item_retrieval(user_input)

    if faiss_results:
        print(f"SearchBot: Here are some great picks for you!")
        print(f"{faiss_results}")  # Print FAISS results
        print(f"⚡ FAISS Latency: {retrieval_latency:.2f} ms\n")
        return faiss_results
    else:
        return "SearchBot: Hmm, I couldn't find anything. Maybe try a different genre?"

# ✅ Main Chatbot Execution
if __name__ == "__main__":
    greeting = get_greeting()
    print(f"SearchBot: {greeting}")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "bye", "quit"]:
            print("SearchBot: Alright, see you next time! 🍿🎥")
            break

        # ✅ **Detect intent correctly**
        intent = detect_intent(user_input)

        if intent == "movie_recommendation":
            # ✅ Call FAISS for content recommendations
            response = content_retrieval(user_input)
            print(f"SearchBot: {response}")
        elif intent == "general_conversation":
            # ✅ Properly handles small talk
            response = general_chat(user_input)
            print(f"SearchBot: {response}")
        elif intent == "reject":
            # ✅ More natural response instead of looping rejections
            print("SearchBot: Hmm, that’s not something I can help with. But if you're looking for a good movie, I'm your bot! 🎬😊")
        else:
            # ✅ Catch-all response
            print("SearchBot: That sounds interesting! What kind of movies do you like? 🎬")

    print("SearchBot: Take care! 🎬👋")
