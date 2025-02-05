# Ollama
c3po

docker pull saiedmighani/ollama:latest <br>
docker run --gpus all -it --rm -p 11434:11434 saiedmighani/ollama:latest

docker run --gpus all -it --rm --network my-network -p 11434:11434 saiedmighani/ollama:latest
docker run --gpus all -it --rm --network my-network --name ollama-container -p 11434:11434 saiedmighani/ollama:latest
