import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Sample sentences
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

# Display full embedding values
np.set_printoptions(threshold=np.inf)

print("Total number of sentences:", len(sentences))
print("Embedding dimension:", len(embeddings[0]))

print("\n--- Embeddings ---")

for i, sentence in enumerate(sentences):
    print("\nSentence:", sentence)
    print("Embedding:", embeddings[i])

# Calculate cosine similarity
similarity = cosine_similarity(embeddings)

print("\n--- Similarity between sentences ---")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        if similarity[i][j] > 0.7:
            print(
                f"\n'{sentences[i]}'"
                f"\n'{sentences[j]}'"
                f"\nSimilarity: {similarity[i][j]:.4f}"
            )
