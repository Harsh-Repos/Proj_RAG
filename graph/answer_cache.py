"""Answer cache: remembers verified answers so a repeated question is answered instantly.

Two levels of caching exist in this project:
  1. Resource cache (@lru_cache in nodes.py, @st.cache_resource in the UI): FAISS, BM25,
     the project metadata and the LLM client are loaded from disk once per process.
  2. Answer cache (this file): the full result of a question that PASSED verification is
     stored in data/answer_cache.db. Asking the same question again skips every LLM call.

The cache key is the normalized question: its meaningful words in order, in one standard form
(see normalize_question). Rewordings ("used" / "uses", "gen ai" / "genai", "tell me about" /
"give details of") hit the cache; questions that differ in any meaningful word do not.
There is deliberately no "similar meaning" (embedding) cache. Measured with our embedding
model: "project P098" vs "P089" = 0.996 similar, "team size more than 5" vs "less than 5" =
0.991, "Python and Snowflake" vs "Python or Snowflake" = 0.993, while two true rewordings
scored only 0.77-0.83. A meaning-based cache would return wrong answers.
The cache is cleared by `python ingestion/build_index.py --rebuild` and by the
"Clear answer cache" button in the UI, so it never outlives the indexes it was built from.
"""

import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime

from langchain_core.documents import Document

from ingestion.document_loader import PROJECT_ROOT

ANSWER_CACHE_PATH = PROJECT_ROOT / "data" / "answer_cache.db"

# State fields that make up a finished answer (everything the UI shows).
CACHED_FIELDS = (
    "search_query",
    "metadata_filters",
    "metadata_matches",
    "documents",
    "answer",
    "is_valid",
    "verification_feedback",
    "retry_count",
)

# Words that never change the answer ("tell me about project P098" = "P098"). A word is only
# listed here if removing it can never turn one question into a different one.
FILLER_WORDS = {
    "a", "an", "the", "of", "about", "regarding", "tell", "me", "please", "kindly", "can", "could",
    "would", "you", "give", "show", "share", "list", "info", "information", "detail", "what",
    "which", "is", "are", "was", "were", "do", "does", "did", "i", "want", "to", "know", "that",
    "have", "has", "had", "with", "project", "repository",
}
# Different spellings of the same thing, unified before comparing.
PHRASE_VARIANTS = {"generative ai": "genai", "gen ai": "genai", "then": "than"}
WORD_VARIANTS = {"used": "use", "uses": "use", "using": "use"}

CREATE_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS answer_cache (
    question_key TEXT PRIMARY KEY,
    question     TEXT NOT NULL,
    result_json  TEXT NOT NULL,
    created_at   TEXT NOT NULL
)
"""


def normalize_word(word: str) -> str:
    """One form per word: "used" / "uses" / "using" -> "use", "projects" -> "project"."""
    word = WORD_VARIANTS.get(word, word)
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]  # plural -> singular ("months" -> "month"); "less" is kept
    return word


def normalize_question(question: str) -> str:
    """Cache key: the question's meaningful words, in order, in one standard form.

    "Tell me about the project that used python and gen ai" and "tell about the project
    that uses python and genai" share one key ("use python and genai"). Words that change
    the answer are kept: project IDs, numbers, technologies, field names ("team size",
    "duration", "solution approach"), "more" / "less", "not", "and" / "or", "how many"
    and the comparison signs < > =.
    """
    text = question.lower()
    for phrase, standard in PHRASE_VARIANTS.items():
        text = re.sub(rf"\b{phrase}\b", standard, text)
    # Filler is checked before ("does") and after ("projects" -> "project") normalizing a word.
    words = (normalize_word(word) for word in re.findall(r"[a-z0-9]+|[<>=]+", text) if word not in FILLER_WORDS)
    return " ".join(word for word in words if word not in FILLER_WORDS)


def load_cached_answer(question: str) -> dict | None:
    """Return the cached result for this question, or None if it was never cached."""
    if not ANSWER_CACHE_PATH.exists():
        return None
    # closing(): the file must not stay open, or Windows cannot delete it when the cache is cleared.
    with closing(sqlite3.connect(ANSWER_CACHE_PATH)) as connection:
        connection.execute(CREATE_CACHE_TABLE)
        row = connection.execute(
            "SELECT result_json FROM answer_cache WHERE question_key = ?", (normalize_question(question),)
        ).fetchone()
    if row is None:
        return None

    result = json.loads(row[0])
    result["documents"] = [Document(**document) for document in result["documents"]]
    return result


def save_answer_to_cache(question: str, result: dict) -> None:
    """Store the finished answer (only the fields in CACHED_FIELDS) as JSON."""
    record = {field: result[field] for field in CACHED_FIELDS}
    record["documents"] = [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in result["documents"]
    ]
    ANSWER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(ANSWER_CACHE_PATH)) as connection, connection:
        connection.execute(CREATE_CACHE_TABLE)
        connection.execute(
            "INSERT OR REPLACE INTO answer_cache (question_key, question, result_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (normalize_question(question), question, json.dumps(record), datetime.now().isoformat(timespec="seconds")),
        )


def clear_answer_cache() -> None:
    """Delete every cached answer (used on --rebuild and by the UI button)."""
    ANSWER_CACHE_PATH.unlink(missing_ok=True)
