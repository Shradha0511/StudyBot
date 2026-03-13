import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL_NAME


def get_embedding_model():
    """
    Initialize and return the HuggingFace embedding model.
    
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")