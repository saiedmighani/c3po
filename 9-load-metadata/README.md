# Ollama

docker build --no-cache -t sentence-transformer-helper .
docker run --rm --network=my-network sentence-transformer-helper
docker push saiedmighani/sentence-transformer-helper:latest
