"""Routing decision after answer verification."""

from graph.state import RAGState

MAX_RETRIES = 1


def check_verification(state: RAGState) -> str:
    """Return "end" if the answer is valid or the single retry is used up, otherwise "retry"."""
    if state["is_valid"]:
        return "end"
    if state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "end"
