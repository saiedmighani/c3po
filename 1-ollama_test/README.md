# Ollama
c3po

docker build -t ollama-gpu .<br>
docker run --gpus all -it --rm -p 11434:11434 ollama-gpu
