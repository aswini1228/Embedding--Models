from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Input sentences
sentences = [
    "I enjoy coding in Python.",
    "I love programming in Python.",
    "Python is my favorite programming language.",
    "The weather is very hot today.",
    "It is raining heavily outside.",
    "I went to college this morning.",
    "My college has many computer science students."
]

# Generate sentence embeddings
embeddings = model.encode(sentences)

# Display information
print("Total number of sentences:", len(sentences))
print("Embedding dimension:", embeddings.shape[1])

print("\n--- Similarity between sentences ---")

# Calculate cosine similarity
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):

        similarity = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        # Display highly similar pairs
        if similarity > 0.7:
            print(
                f"\n'{sentences[i]}'\n"
                f"'{sentences[j]}'\n"
                f"Similarity: {similarity:.4f}"
            )
