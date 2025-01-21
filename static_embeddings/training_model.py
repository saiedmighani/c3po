from sentence_transformers import SentenceTransformer, util

# Load the model
model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")

# List of sentences to compare against
sentences = [
    "Back to the Future",
    "Batman Begins",
    "Black Panther",
    "Braveheart",
    "Beauty and the Beast",
    "Borat",
    "Blade Runner",
    "Birdman",
    "Bohemian Rhapsody",
    "Blazing Saddles",
    "Beverly Hills Cop",
    "Brokeback Mountain",
    "Big",
    "Boyhood",
    "Bridesmaids",
    "Basic Instinct",
    "Breakfast at Tiffany's",
    "Bonnie and Clyde",
    "Bridge of Spies",
    "Blue Valentine",
    "Forrest Gump",
    "Fight Club",
    "Frozen",
    "Fast & Furious",
    "Fantastic Beasts and Where to Find Them",
    "Finding Nemo",
    "Fargo",
    "Ferris Bueller's Day Off",
    "Fifty Shades of Grey",
    "Field of Dreams"
]

# Encode the sentences into embeddings
sentence_embeddings = model.encode(sentences)

# Example query
query = "back"
query_embedding = model.encode(query)

# Compute cosine similarities between the query and sentences
similarities = util.cos_sim(query_embedding, sentence_embeddings)

# Prepare a list of sentences with their similarity scores
results = [(sentences[i], similarities[0, i].item()) for i in range(len(sentences))]

# Sort the results by similarity scores in descending order
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

# Print the sorted list of sentences with scores
print(f"Query: {query}\n")
print("Similar sentences ranked by similarity:")
for rank, (sentence, score) in enumerate(sorted_results, start=1):
    print(f"{rank}. {sentence} (Score: {score:.4f})")
