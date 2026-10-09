"""LangGraph nodes for the controlled Hybrid RAG workflow.

LLM calls happen only in: analyze_query, generate_answer, verify_answer, refine_query.
The retrieval and merge nodes are deterministic.
"""

import json
from collections import Counter
from functools import lru_cache

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.llm import get_llm
from graph.metadata_filters import (
    PROJECT_ID_PATTERN,
    QueryAnalysis,
    build_metadata_filters,
    describe_filters,
    filter_projects_by_metadata,
)
from graph.prompts import (
    ANSWER_PROMPT,
    JSON_FALLBACK_INSTRUCTION,
    QUERY_ANALYSIS_PROMPT,
    REFINE_QUERY_PROMPT,
    VERIFICATION_PROMPT,
)
from graph.state import RAGState
from ingestion.database import load_project_metadata
from ingestion.keyword_retriever import load_keyword_retriever
from ingestion.vector_store import load_vector_store

SEMANTIC_TOP_K = 5
# For questions with conditions, retrieval is restricted to the summary chunks of the
# matching projects (one chunk per project, holding every field). Up to this many
# projects' text is sent to the LLM; the full match list is always included as well.
MAX_FILTERED_PROJECTS = 8


# ---------------------------------------------------------------------------
# Persisted resources: loaded from disk once per process, never rebuilt.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_vector_store() -> FAISS:
    return load_vector_store()


@lru_cache(maxsize=1)
def get_keyword_retriever() -> BM25Retriever:
    return load_keyword_retriever()


@lru_cache(maxsize=1)
def get_project_metadata() -> list[dict]:
    return load_project_metadata()


@lru_cache(maxsize=1)
def get_technology_names() -> list[str]:
    return sorted({tech for project in get_project_metadata() for tech in project["technology_stack"]})


def get_matching_project_ids(filters: dict) -> set[str] | None:
    """IDs of projects satisfying the question's conditions (exact Python comparison).

    Returns None when the question has no conditions, meaning "search everything".
    """
    if not filters:
        return None
    return {project["project_id"] for project in filter_projects_by_metadata(filters, get_project_metadata())}


def is_matching_summary_chunk(metadata: dict, allowed_ids: set[str]) -> bool:
    """True for the summary chunk of a project that satisfies the conditions."""
    return metadata["project_id"] in allowed_ids and metadata.get("section") == "summary"


class VerificationResult(BaseModel):
    """Structured output of the verification step."""

    is_valid: bool = Field(description="True only if the answer passes every check.")
    feedback: str = Field(description="Short explanation of problems found, or 'All checks passed.'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def invoke_structured(prompt: ChatPromptTemplate, schema: type[BaseModel], inputs: dict) -> BaseModel:
    """Call the LLM and get back a filled Pydantic object.

    First try: Groq strict JSON-schema mode. gpt-oss sometimes writes its reasoning
    as the reply instead of JSON, and Groq then rejects it with HTTP 400
    (output_parse_failed). In that case the call is repeated once in JSON mode,
    with the schema spelled out in the prompt.
    """
    llm = get_llm()
    strict_chain = prompt | llm.with_structured_output(schema, method="json_schema", strict=True)
    json_mode_chain = (prompt + [("human", JSON_FALLBACK_INSTRUCTION)]) | llm.with_structured_output(
        schema, method="json_mode"
    )
    fallback_inputs = {**inputs, "json_schema": json.dumps(schema.model_json_schema())}
    try:
        return strict_chain.invoke(inputs)
    except Exception:
        return json_mode_chain.invoke(fallback_inputs)


def keep_project_ids(query: str, question: str) -> str:
    """Guarantee that project IDs from the question survive LLM rewriting."""
    missing = [pid.upper() for pid in PROJECT_ID_PATTERN.findall(question) if pid.upper() not in query.upper()]
    return " ".join([query, *missing]).strip()


def clean_single_line(text: str) -> str:
    """Take the first non-empty line of an LLM reply and strip quotes."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[0].strip("\"'` ") if lines else ""


def format_metadata_matches(filters: dict, matches: list[dict]) -> str:
    """Render the deterministic filter results as a compact list for the prompt."""
    header = f"PROJECTS MATCHING THE METADATA FILTERS ({describe_filters(filters)}):"
    if not matches:
        return f"{header}\nNo project in the repository satisfies these conditions."
    rows = [
        f"- {m['project_id']} | {m['project_name']} | Type: {m['type']} | Domain: {m['domain']} | "
        f"Team Size: {m['team_size']} people | Duration: {m['duration']} | "
        f"Technologies: {', '.join(m['technology_stack'])}"
        for m in matches
    ]
    # Exact technology tally across the matching projects, so the LLM never has to count.
    counts = Counter(tech for m in matches for tech in m["technology_stack"])
    tally = ", ".join(f"{tech} ({count})" for tech, count in counts.most_common())
    return "\n".join([header, *rows, "", f"TECHNOLOGIES ACROSS THESE PROJECTS (number of projects): {tally}"])


def format_context(state: RAGState) -> str:
    """Build the prompt context: filter results (if any) + retrieved chunks."""
    sections = []
    if state.get("metadata_filters"):
        sections.append(format_metadata_matches(state["metadata_filters"], state["metadata_matches"]))

    chunks = [
        f"[Source {i}] {doc.metadata['project_id']} — {doc.metadata['project_name']} "
        f"(page {doc.metadata['page']})\n{doc.page_content}"
        for i, doc in enumerate(state["documents"], start=1)
    ]
    sections.append("RETRIEVED PROJECT TEXT:\n\n" + ("\n\n".join(chunks) if chunks else "(none)"))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def analyze_query(state: RAGState) -> dict:
    """One LLM call that (a) rewrites the question for retrieval and (b) extracts its
    conditions into a fixed schema. The LLM only understands the wording; the numeric
    comparison happens later in Python (merge_documents)."""
    question = state["question"]
    # Tool-calling mode failed with HTTP 400 on off-topic questions, so JSON modes are used.
    analysis = invoke_structured(QUERY_ANALYSIS_PROMPT, QueryAnalysis, {"question": question})
    search_query = clean_single_line(analysis.search_query) or question

    return {
        "search_query": keep_project_ids(search_query, question),
        "metadata_filters": build_metadata_filters(analysis, question, get_technology_names()),
        "retry_count": 0,
    }


def semantic_retrieve(state: RAGState) -> dict:
    """Semantic retrieval with FAISS: finds chunks with similar meaning.

    If the question has conditions, only the summary chunks of the matching
    projects are searched.
    """
    vector_store = get_vector_store()
    allowed_ids = get_matching_project_ids(state["metadata_filters"])
    if allowed_ids is None:
        documents = vector_store.similarity_search(state["search_query"], k=SEMANTIC_TOP_K)
    else:
        documents = vector_store.similarity_search(
            state["search_query"],
            k=MAX_FILTERED_PROJECTS,
            filter=lambda metadata: is_matching_summary_chunk(metadata, allowed_ids),
            fetch_k=vector_store.index.ntotal,  # consider every chunk before filtering
        )
    return {"semantic_documents": documents}


def keyword_retrieve(state: RAGState) -> dict:
    """Keyword retrieval with BM25: finds chunks sharing exact terms (P038, Snowflake, ...).

    If the question has conditions, only the summary chunks of the matching
    projects are ranked.
    """
    retriever = get_keyword_retriever()
    allowed_ids = get_matching_project_ids(state["metadata_filters"])
    if allowed_ids is None:
        documents = retriever.invoke(state["search_query"])
    else:
        scores = retriever.vectorizer.get_scores(retriever.preprocess_func(state["search_query"]))
        ranked = sorted(zip(scores, retriever.docs), key=lambda pair: pair[0], reverse=True)
        documents = [
            doc for _, doc in ranked if is_matching_summary_chunk(doc.metadata, allowed_ids)
        ][:MAX_FILTERED_PROJECTS]
    return {"keyword_documents": documents}


def merge_documents(state: RAGState) -> dict:
    """Combine FAISS and BM25 results and remove duplicates.

    - Each chunk appears once, tagged retrieval_method = semantic / keyword / both.
    - Chunks found by both retrievers are listed first.
    - If the question has structured conditions, the full list of matching projects
      is kept for the prompt (the retrievers already searched only those projects).
    """
    merged: dict[str, Document] = {}
    for method, documents in (("semantic", state["semantic_documents"]), ("keyword", state["keyword_documents"])):
        for doc in documents:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id in merged:
                merged[chunk_id].metadata["retrieval_method"] = "both"
            else:
                # Copy so the cached retriever's documents are never modified.
                merged[chunk_id] = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "retrieval_method": method},
                )

    documents = sorted(merged.values(), key=lambda doc: doc.metadata["retrieval_method"] != "both")

    filters = state["metadata_filters"]
    matches: list[dict] = []
    if filters:
        matches = filter_projects_by_metadata(filters, get_project_metadata())

    return {"documents": documents, "metadata_matches": matches}


def generate_answer(state: RAGState) -> dict:
    """Generate an answer with Groq using only the merged context."""
    feedback_note = ""
    if state["retry_count"] > 0:
        feedback_note = f"A previous answer was rejected for this reason: {state['verification_feedback']}\n\n"

    chain = ANSWER_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({
        "context": format_context(state),
        "question": state["question"],
        "feedback_note": feedback_note,
    })
    return {"answer": answer.strip()}


def verify_answer(state: RAGState) -> dict:
    """Check the answer against the context. Returns a verdict, not a new answer."""
    result = invoke_structured(VERIFICATION_PROMPT, VerificationResult, {
        "context": format_context(state),
        "question": state["question"],
        "answer": state["answer"],
    })
    return {"is_valid": result.is_valid, "verification_feedback": result.feedback}


def refine_query(state: RAGState) -> dict:
    """Write a better search query using the verifier feedback (runs at most once)."""
    chain = REFINE_QUERY_PROMPT | get_llm() | StrOutputParser()
    refined = clean_single_line(chain.invoke({
        "question": state["question"],
        "search_query": state["search_query"],
        "feedback": state["verification_feedback"],
    })) or state["search_query"]

    return {
        "search_query": keep_project_ids(refined, state["question"]),
        "retry_count": state["retry_count"] + 1,
    }
