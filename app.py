import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Sentence Embedding & Similarity")

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = st.text_area("Enter Sentence 1")
sentence2 = st.text_area("Enter Sentence 2")

if st.button("Compare"):
    if sentence1 and sentence2:
        embedding1 = model.encode(sentence1)
        embedding2 = model.encode(sentence2)

        score = cosine_similarity(
            [embedding1],
            [embedding2]
        )[0][0]

        st.subheader("Similarity Result")
        st.write(f"Similarity Score: {score:.4f}")
        st.write(f"Similarity Percentage: {score * 100:.2f}%")

        if score >= 0.7:
            st.success("Highly Similar")
        elif score >= 0.4:
            st.warning("Moderately Similar")
        else:
            st.error("Not Similar")
    else:
        st.warning("Please enter both sentences.")
