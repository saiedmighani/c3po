# Ollama
c3po

docker build -t faiss-cpu-app . <br>
docker network create my-network <b>
docker run -d --name faiss-container --network my-network -p 5000:5000 faiss-cpu-app

curl http://localhost:5000/healthz   


