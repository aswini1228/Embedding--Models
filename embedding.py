from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Sample sentences
sentences = [
    "I love learning Artificial Intelligence.",
    "I enjoy studying AI.",
    "Python is a programming language.",
    "I like coding in Python.",
    "The weather is very hot today.",
    "It is a sunny and warm day.",
    "Machine learning is a part of Artificial Intelligence."
]

# Generate embeddings
embeddings = model.encode(sentences)

# Display basic information
print("Total number of sentences:", len(sentences))
print("Embedding dimension:", embeddings.shape[1])

print("\n--- Similarity between sentence pairs ---")

# Calculate cosine similarity
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):

        similarity = cosine_similarity(
            [embeddings[i]],
            [embeddings[j]]
        )[0][0]

        # Display only highly similar pairs
        if similarity >= 0.7:
            print("\nSentence 1:", sentences[i])
            print("Sentence 2:", sentences[j])
            print("Similarity:", round(similarity, 4))
