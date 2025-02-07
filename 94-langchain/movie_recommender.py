import os
import sys
import json
import requests
import numpy as np
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
FAISS_INDEX_PATH = "./faiss_index"
OLLAMA_API_URL = "http://ollama-container:11434/api/generate"

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index for movie retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local(FAISS_INDEX_PATH, embeddings).as_retriever(search_kwargs={"k": 10})

# Initialize memory (keeps last 3 queries)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)

# --- Precomputed Embedding Categories ---
rude_queries = [
    "You are stupid", "Shut up", "You are an idiot", "Dumb bot", "I hate you", "Go away", "You're useless"
]
non_movie_queries = [
    "Who won the last World Cup?", "What is the capital of France?", "Tell me a joke", "How’s the weather today?"
]
movie_queries = [
    "Recommend a movie", "What is a good sci-fi movie?", "Suggest a horror film", "Best action movie of all time?"
]

# Compute embeddings for predefined categories
rude_embeddings = embedding_model.encode(rude_queries, convert_to_numpy=True)
non_movie_embeddings = embedding_model.encode(non_movie_queries, convert_to_numpy=True)
movie_embeddings = embedding_model.encode(movie_queries, convert_to_numpy=True)

# --- Helper Functions ---
def get_embedding(text):
    """Encode text using the Sentence Transformer model."""
    return embedding_model.encode(text, convert_to_numpy=True).reshape(1, -1)

def is_rude(query):
    """Check if a query is rude using cosine similarity."""
    query_embedding = get_embedding(query)
    similarity_scores = cosine_similarity(query_embedding, rude_embeddings)
    return np.max(similarity_scores) > 0.7  # Threshold for rudeness detection

def is_movie_related(query):
    """Check if a query is movie-related using cosine similarity."""
    query_embedding = get_embedding(query)
    similarity_scores = cosine_similarity(query_embedding, movie_embeddings)
    return np.max(similarity_scores) > 0.7  # Threshold for movie relevance

def query_ollama(prompt):
    """Query the Ollama LLM API."""
    headers = {"Content-Type": "application/json"}
    payload = {"model": "llama3.2", "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json().get("response", "No response received.")
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Define LangChain Prompt Template
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question", "retrieved_movies"],
    template="""You are a friendly and concise movie recommendation assistant.
You always match the mood of the user's question and provide short, sweet, and relevant suggestions.
Strictly recommend only movies retrieved from FAISS. If a question is rude, refuse to answer politely.
If unrelated to movies, politely redirect the user.

Conversation History:
{chat_history}

User Query: {question}

Relevant Movies from FAISS:
{retrieved_movies}

Your response should be **1-2 sentences**, followed by a **maximum of 10 movie recommendations from FAISS only**.
"""
)

# Create Conversational Retrieval Chain
llm_chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="gpt-4"),
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
)

# --- FastAPI Setup ---
app = FastAPI()

@app.get("/recommend")
def recommend(query: str = Query(..., title="User Query")):
    """API endpoint to get movie recommendations based on user query."""

    # Check for rudeness
    if is_rude(query):
        return {"response": "I’m here to spread positivity! Let's talk about movies instead. What movie can I recommend?"}

    # Check if query is movie-related
    if not is_movie_related(query):
        return {"response": "I'm a movie recommendation assistant! Let's talk films. What movie can I recommend?"}

    # Query FAISS for movie recommendations
    faiss_results = retriever.get_relevant_documents(query)

    if not faiss_results:
        return {"response": "No relevant movies found."}

    # Extract strictly retrieved movie titles
    retrieved_movies = "\n".join([doc.page_content for doc in faiss_results[:10]])

    # Generate response with LangChain
    response = llm_chain.invoke({"question": query, "retrieved_movies": retrieved_movies})

    return {"response": response["answer"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
