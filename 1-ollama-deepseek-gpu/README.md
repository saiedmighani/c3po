# Ollama
c3po

docker build --no-cache -t ollama-gpu . <br>
docker run -d --gpus all -it --rm --name ollama-container --network my-network -p 11434:11434 ollama-gpu

# Confirm oolama is getting served on GPU:

docker exec -it caa5f6ffc9a7 bash

ps aux | grep ollama
nvidia-smi

ollama run llama3.2


# Running from remote:
Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body '{"model": "llama3.1:8b-instruct-q2_K", "prompt": "Hello"}' -ContentType "application/json"
