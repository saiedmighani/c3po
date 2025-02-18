import requests
import json
import time
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MAX_HISTORY = 15

def get_greeting():
    """Return an appropriate greeting based on the current time."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning! ☀️ Hulue-yah! Need a movie? 🎬"
    elif hour < 18:
        return "Good afternoon! 🌞 Hulue-yah! Need a movie? 🎬"
    else:
        return "Good evening! 🌙 Hulue-yah! Need a movie? 🎬"

def query_ollama(model_name, prompt):
    headers = {"Content-Type": "application/json"}
    instruction = "Respond in a friendly manner. Keep responses concise, maximum 15 words."

    payload = {
        "model": model_name,
        "prompt": f"{instruction}\n{prompt}",
        "stream": False,
        "temperature": 0.7
    }

    try:
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received.")

    except requests.exceptions.RequestException as e:
        return "Error occurred."

def format_history(history):
    return "\n".join(history)

if __name__ == "__main__":
    model = "llama3.2"
    greeting = get_greeting()
    history = [f"SearchBot: {greeting}"]

    print(history[0])

    while len(history) < MAX_HISTORY * 2:  # Each exchange is two entries
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
