# Ollama
c3po

docker network create my-network <b>


docker build --no-cache -t faiss-gpu-server . <br>
docker run -d --name faiss-container --network my-network --gpus all -p 5000:5000 faiss-gpu-server

curl http://localhost:5000/healthz   

curl http://localhost:5000/index_device

curl http://localhost:5000/vector_dimension

curl http://localhost:5000/index_count



