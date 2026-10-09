"""Hugging Face embedding model used to build and query the FAISS index.

The model runs LOCALLY with FastEmbed (ONNX Runtime): it is downloaded from
Hugging Face once, cached on disk, and then needs no API key, no credits and
no internet connection.
"""

from langchain_community.embeddings import FastEmbedEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> FastEmbedEmbeddings:
    """Return the local all-MiniLM-L6-v2 embedding model (384 dimensions)."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
