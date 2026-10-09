"""Build (offline) and load (online) the FAISS vector store."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ingestion.document_loader import PROJECT_ROOT
from ingestion.embeddings import get_embedding_model

FAISS_INDEX_DIR = PROJECT_ROOT / "data" / "faiss_index"
EMBEDDING_BATCH_SIZE = 32


def build_vector_store(chunks: list[Document], embeddings: Embeddings, index_dir: Path = FAISS_INDEX_DIR) -> FAISS:
    """Embed chunks in small batches (keeps API requests small) and save the index."""
    vector_store = FAISS.from_documents(chunks[:EMBEDDING_BATCH_SIZE], embeddings)
    for start in range(EMBEDDING_BATCH_SIZE, len(chunks), EMBEDDING_BATCH_SIZE):
        vector_store.add_documents(chunks[start : start + EMBEDDING_BATCH_SIZE])

    vector_store.save_local(str(index_dir))
    return vector_store


def load_vector_store(index_dir: Path = FAISS_INDEX_DIR) -> FAISS:
    """Load the persisted FAISS index. Never rebuilds it."""
    if not index_dir.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_dir}. Run: python ingestion/build_index.py")
    # The index is a local file created by build_index.py, so deserializing it is safe.
    return FAISS.load_local(str(index_dir), get_embedding_model(), allow_dangerous_deserialization=True)
