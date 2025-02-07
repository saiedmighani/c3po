# Step 1: Running Ollama on separate port

docker run --gpus all -it --rm --network my-network --name ollama-container -p 11434:11434 saiedmighani/ollama:latest



# Step 2: Running FAISS Index

run commands in this path: 7-faiss-index


docker build -t faiss-cpu-app . <br>
docker network create my-network <b>
docker run -d --name faiss-container --network my-network -p 5000:5000 faiss-cpu-app

# Step 3: Load csv fie into vector DB

Run following command in this path: 9-load-metadata:

docker build --no-cache -t sentence-transformer-helper .
docker run --rm --network=my-network sentence-transformer-helper

# Step 4: Testing search vector db

Run following command in 91-search-dataset:

search_vectors.py


# Step 5: Running RAG application

docker build --no-cache -t rag-helper . <br>
docker run --rm --network my-network rag-helper "give me some good movies about vampires?"

# Step 6: local run:

python rag_application_local.py

