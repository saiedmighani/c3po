import requests
import json
import time

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MAX_HISTORY = 15

def query_ollama(model_name, prompt):
    headers = {"Content-Type": "application/json"}
    payload = {"model": model_name, "prompt": prompt, "stream": False, "max_tokens": 15}

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received.")

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def format_history(history):
    return "\n".join(history)

if __name__ == "__main__":
    model = "llama3.2"
    history = []

    print("SearchBot: Hulue-yah! Need a movie? 🎬")

    while len(history) < MAX_HISTORY:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "bye", "quit"]:
            print("SearchBot: Bye! 🎥🍿")
            break

        history.append(f"User: {user_input}")
        full_prompt = format_history(history) + "\nSearchBot:"

        start_time = time.time()
        bot_response = query_ollama(model, full_prompt)
        end_time = time.time()

        history.append(f"SearchBot: {bot_response}")

        if len(history) > MAX_HISTORY * 2:
            history = history[-MAX_HISTORY * 2:]

        print(f"SearchBot: {bot_response}")
        print(f"⚡ Latency: {(end_time - start_time) * 1000:.2f} ms\n")

    print("SearchBot: Bye! 🎬👋")
