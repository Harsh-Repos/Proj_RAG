"""State shared by all LangGraph nodes."""

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str                       # original user question (never modified)
    search_query: str                   # retrieval-friendly query (refined on retry)
    metadata_filters: dict              # structured conditions found in the question

    semantic_documents: list[Document]  # FAISS results
    keyword_documents: list[Document]   # BM25 results
    documents: list[Document]           # merged, de-duplicated context
    metadata_matches: list[dict]        # projects satisfying metadata_filters (from SQLite)

    answer: str

    is_valid: bool
    verification_feedback: str

    retry_count: int
