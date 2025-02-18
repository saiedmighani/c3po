import requests
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
from guardrails import Guard

# API endpoints
OLLAMA_API_URL = "http://localhost:11434/api/generate"
FAISS_API_URL = "http://localhost:5000/search"

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # FAISS search
intent_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")  # Intent detection

MAX_HISTORY = 15

def get_greeting():
    """Return an appropriate greeting based on the current time."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning! ☀️ Hulue-yah! What kind of content are you looking for? 🎬"
    elif hour < 18:
        return "Good afternoon! 🌞 Hulue-yah! Tell me what type of content you're in the mood for. 🎬"
    else:
        return "Good evening! 🌙 Hulue-yah! Let me know what you feel like watching. 🎬"

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

# ✅ Define Guardrails AI Validation Properly
guardrails_schema = """
<rail version="0.1">
  <output>
    <string name="response"/>
  </output>
  <prompt>
    You are a simple assistant. Do NOT suggest movies or make recommendations.
    If FAISS content recommendations are available, only format and display them.
    If no FAISS results exist, inform the user politely and encourage a refined query.
    DO NOT create recommendations on your own.
  </prompt>
  <validations>
    <assert that="not ('suggest' in response.lower() or 'recommend' in response.lower())" 
    message="Do not make up movie recommendations." />
  </validations>
</rail>
"""

guard = Guard.from_rail_string(guardrails_schema)

def query_ollama(model_name, prompt, max_words=15):
    """Query Ollama while ensuring it follows strict guardrails."""
    headers = {"Content-Type": "application/json"}

    instruction = (
        "You are a simple assistant. "
        "Do NOT suggest movies or make recommendations. "
        "Encourage users to ask about specific content instead. "
        "If FAISS content recommendations are available, only format them. "
        "If no FAISS results exist, inform the user politely and encourage a refined query. "
        "DO NOT create recommendations on your own."
    )

    prompt = f"{instruction}\nUser asked: {prompt}\nKeep it conversational."

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.1  # Lower temperature for stability
    }

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "No response received.")

        # ✅ Validate response with Guardrails - FIXED OUTPUT EXTRACTION
        validation_outcome = guard.validate(
            output={"response": response_text}, llm_output=response_text
        )

        # ✅ Extract and return the validated response correctly
        return validation_outcome.output["response"]

    except requests.exceptions.RequestException as e:
        return "Error occurred."

def detect_intent_huggingface(user_input):
    """Detect if the user wants a recommendation or general conversation."""
    candidate_labels = ["movie recommendation", "general conversation"]
    result = intent_classifier(user_input, candidate_labels)

    intent_label = result["labels"][0]  # Top classified intent
    confidence = result["scores"][0]  # Confidence score

    if intent_label == "movie recommendation" and confidence > 0.5:
        return "movie_recommendation"
    return "general_conversation"

if __name__ == "__main__":
    model = "llama3.2"
    greeting = get_greeting()
    history = [f"SearchBot: {greeting}"]

    print(history[0])

    while len(history) < MAX_HISTORY * 2:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "bye", "quit"]:
            print("SearchBot: Bye! 🎥🍿")
            break

        history.append(f"User: {user_input}")

        # Detect intent
        intent = detect_intent_huggingface(user_input)

        if intent == "movie_recommendation":
            # Get FAISS results
            faiss_results, retrieval_latency = item_retrieval(user_input)

            if faiss_results:
                # If FAISS found results, format them
                start_time = time.time()
                bot_response = query_ollama(model, user_input, max_words=15)
                end_time = time.time()
                print(f"SearchBot: {bot_response}")
                print(f"{faiss_results}")  # Print FAISS results after ":"
                print(f"⚡ FAISS Latency: {retrieval_latency:.2f} ms\n")
            else:
                # If FAISS found nothing, encourage user to refine their query
                print("SearchBot: I couldn't find anything for that. Maybe try a different genre or title? 🎬")

            print(f"⚡ Query Latency: {(end_time - start_time) * 1000:.2f} ms\n")

        else:
            # General chat response from Ollama
            full_prompt = "\n".join(history) + "\nSearchBot:"
            start_time = time.time()
            bot_response = query_ollama(model, full_prompt)
            end_time = time.time()

            history.append(f"SearchBot: {bot_response}")
            print(f"SearchBot: {bot_response}")
            print(f"⚡ Query Latency: {(end_time - start_time) * 1000:.2f} ms\n")

        if len(history) > MAX_HISTORY * 2:
            history = history[-MAX_HISTORY * 2:]

    print("SearchBot: Bye! 🎬👋")
