"""Build (offline) and load (online) the BM25 keyword retriever."""

import pickle
import re
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ingestion.document_loader import PROJECT_ROOT

BM25_PATH = PROJECT_ROOT / "data" / "bm25.pkl"
KEYWORD_TOP_K = 5


def tokenize_text(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics, so "P038?" matches "P038" and "python" matches "Python"."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_keyword_retriever(chunks: list[Document], path: Path = BM25_PATH) -> BM25Retriever:
    """Build the BM25 retriever over all chunks and pickle it."""
    retriever = BM25Retriever.from_documents(chunks, k=KEYWORD_TOP_K, preprocess_func=tokenize_text)
    with open(path, "wb") as file:
        pickle.dump(retriever, file)
    return retriever


def load_keyword_retriever(path: Path = BM25_PATH) -> BM25Retriever:
    """Load the pickled BM25 retriever. Never rebuilds it."""
    if not path.exists():
        raise FileNotFoundError(f"BM25 index not found at {path}. Run: python ingestion/build_index.py")
    # The pickle is a local file created by build_index.py, so loading it is safe.
    with open(path, "rb") as file:
        return pickle.load(file)
