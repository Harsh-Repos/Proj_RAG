"""Groq chat model initialization."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Groq's free tier limits tokens per minute (prompt + reserved output tokens).
# A bounded output keeps every request under that limit; on HTTP 429 the client
# waits and retries.
MAX_OUTPUT_TOKENS = 2048
MAX_API_RETRIES = 5


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Return the Groq chat model (created once per process).

    Reads GROQ_API_KEY and, optionally, GROQ_MODEL from .env.
    """
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file.")

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    # gpt-oss models "think" before answering; low effort is enough for grounded RAG
    # and keeps latency and token usage down.
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    return ChatGroq(model=model, temperature=0, max_tokens=MAX_OUTPUT_TOKENS, max_retries=MAX_API_RETRIES, **extra)
