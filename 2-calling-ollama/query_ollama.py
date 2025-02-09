import requests
import json

# Define the Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama runs on port 11434 by default

def query_ollama(model_name, prompt):
    """Send a query to the Ollama container and get a response."""
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,  # Example: "llama3.2"
        "prompt": prompt,
        "stream": False  # Set to False for a single response
    }

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an error for bad responses

        # Extract and return response text
        result = response.json()
        return result.get("response", "No response received.")

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    model = "llama3.2"
    question = "give me the top 5 movies from 2000's"

    print("Querying Ollama...")
    result = query_ollama(model, question)
    print("\nOllama Response:")
    print(result)
