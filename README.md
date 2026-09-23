#  Sentence Embedding & Semantic Similarity

A Python-based Natural Language Processing (NLP) project that converts sentences into high-dimensional numerical embeddings and measures their semantic similarity using cosine similarity.

##  Overview

This project uses the pre-trained **Sentence Transformer `all-mpnet-base-v2`** model to transform sentences into **768-dimensional vector representations**.

Cosine similarity is then applied to compare these vectors and identify sentences with similar meanings.

##  Features

- Generate sentence embeddings
- Create 768-dimensional vector representations
- Calculate cosine similarity
- Identify semantically similar sentences
- Display complete embedding vectors
- Use a pre-trained Sentence Transformer model

##  Workflow
```

Input Sentences
      ↓
Sentence Transformer
      ↓
768-Dimensional Embeddings
      ↓
Cosine Similarity
      ↓
Similarity Score
      ↓
Similar Sentence Pairs
Tech Stack
Technology
Purpose
Python
Core programming
Sentence Transformers
Generate sentence embeddings
Scikit-learn
Calculate cosine similarity
NumPy
Numerical operations
 Model
Model: all-mpnet-base-v2
The model converts each sentence into a 768-dimensional embedding vector.
Example:
Sentence: I enjoy coding in Python.

Embedding:
[ 1.70083828e-02  5.86691052e-02 -7.10436329e-02 ... ]````
```

## Installation
- Clone the repository:
- git clone https://github.com/aswini1228/Embedding- Model.git
cd Embedding-Model
- pip install -r requirements.txt
 Run the Project
- python embedding.py
- 
## Applications
- Semantic Search
- Text Similarity
- Document Matching
- Question Answering
- Recommendation Systems
- Natural Language Processing (NLP)

## Future Enhancements
- Add custom user input
- Build an interactive Streamlit interface
- Add similarity visualization
- Implement semantic search
- Support document-level similarity
 ## Author
- Aswini.S
- B.Sc Computer Science with Artificial Intelligence
