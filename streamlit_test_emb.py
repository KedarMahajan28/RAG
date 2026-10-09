import streamlit as st
print(">>> BEFORE HuggingFaceEmbeddings import", flush=True)
from langchain_huggingface import HuggingFaceEmbeddings
print(">>> AFTER HuggingFaceEmbeddings import", flush=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print(">>> AFTER HuggingFaceEmbeddings init", flush=True)
st.write("Loaded successfully!")
