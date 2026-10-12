"""Routing decisions: after the cache lookup and after answer verification."""

from graph.state import RAGState

MAX_RETRIES = 1


def check_cache_hit(state: RAGState) -> str:
    """Return "hit" to answer from the cache, otherwise "miss" to run the full pipeline."""
    return "hit" if state["cache_hit"] else "miss"


def check_verification(state: RAGState) -> str:
    """Return "end" if the answer is valid or the single retry is used up, otherwise "retry"."""
    if state["is_valid"]:
        return "end"
    if state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "end"
