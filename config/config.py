import os

try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY", ""))
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_key_here")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "your_tavily_key_here")

GROQ_MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 10
