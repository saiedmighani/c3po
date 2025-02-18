import requests
import json
import time
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

# API endpoints
OLLAMA_API_URL = "http://localhost:11434/api/generate"
FAISS_API_URL = "http://localhost:5000/search"

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # FAISS search
intent_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")  # Intent detection
llm = Ollama(model="llama3.2")

MAX_HISTORY = 15

# Initialize memory for tracking conversation history
memory = ConversationBufferMemory(max_history=MAX_HISTORY, return_messages=True)

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
                        f"  - 📝 Description: {metadata.get('description', 'N/A')}\n"
                    )

        if retrieved_items:
            return f":\n{''.join(retrieved_items[:3])}", latency_ms  # Return top 3 matches
        else:
            return None, latency_ms  # No strong matches found
    else:
        return None, 0

# Define LLM prompt for structured response generation
response_prompt = PromptTemplate(
    input_variables=["history", "query"],
    template="""
        You are a helpful assistant specialized in streaming content recommendations.
        Your job is to guide the user towards relevant content, NOT make up recommendations.
        If there are FAISS recommendations available, format them. If none exist, ask the user to refine their search.
        Keep responses concise and engaging.

        Conversation History:
        {history}

        User: {query}
        Assistant:
    """
)

llm_chain = LLMChain(llm=llm, prompt=response_prompt, memory=memory)

def detect_intent(user_input):
    """Detect if the user wants a recommendation or general conversation."""
    candidate_labels = ["movie recommendation", "general conversation"]
    result = intent_classifier(user_input, candidate_labels)

    intent_label = result["labels"][0]  # Top classified intent
    confidence = result["scores"][0]  # Confidence score

    if intent_label == "movie recommendation" and confidence > 0.5:
        return "movie_recommendation"
    return "general_conversation"

if __name__ == "__main__":
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
        intent = detect_intent(user_input)

        if intent == "movie_recommendation":
            # Get FAISS results
            faiss_results, retrieval_latency = item_retrieval(user_input)

            if faiss_results:
                # If FAISS found results, format them
                start_time = time.time()
                bot_response = llm_chain.run(history="\n".join(history), query=user_input)
                end_time = time.time()
                print(f"SearchBot: {bot_response}")
                print(f"{faiss_results}")  # Print FAISS results after ":"
                print(f"⚡ FAISS Latency: {retrieval_latency:.2f} ms\n")
            else:
                # If FAISS found nothing, encourage user to refine their query
                print("SearchBot: I couldn't find anything for that. Maybe try a different genre or title? 🎬")

            print(f"⚡ Query Latency: {(end_time - start_time) * 1000:.2f} ms\n")

        else:
            # General chat response from Ollama using LangChain
            start_time = time.time()
            bot_response = llm_chain.run(history="\n".join(history), query=user_input)
            end_time = time.time()

            history.append(f"SearchBot: {bot_response}")
            print(f"SearchBot: {bot_response}")
            print(f"⚡ Query Latency: {(end_time - start_time) * 1000:.2f} ms\n")

        if len(history) > MAX_HISTORY * 2:
            history = history[-MAX_HISTORY * 2:]

    print("SearchBot: Bye! 🎬👋")
