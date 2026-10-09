"""ONLINE QUERYING: Streamlit UI for the Hybrid RAG system.

Run with:  streamlit run ui/app.py

This app only LOADS the persisted FAISS index, BM25 retriever and SQLite database.
It never parses the PDF or builds indexes; run ingestion/build_index.py once first.
"""

import sys
from pathlib import Path

import streamlit as st

# Allow "streamlit run ui/app.py" to import the project packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.metadata_filters import describe_filters  # noqa: E402
from graph.nodes import (  # noqa: E402
    get_keyword_retriever,
    get_project_metadata,
    get_vector_store,
)
from graph.prompts import OUT_OF_SCOPE_ANSWER  # noqa: E402
from graph.workflow import build_graph  # noqa: E402
from ingestion.database import DATABASE_PATH  # noqa: E402
from ingestion.keyword_retriever import BM25_PATH  # noqa: E402
from ingestion.vector_store import FAISS_INDEX_DIR  # noqa: E402

RUN_NAME = "Project Repository RAG"


@st.cache_resource(show_spinner="Loading indexes...")
def load_rag_graph():
    """Load persisted resources and compile the graph once per Streamlit server."""
    get_vector_store()
    get_keyword_retriever()
    get_project_metadata()
    return build_graph()


def escape_markdown(text: str) -> str:
    """Escape "$" so amounts like "$2.8M" are not rendered as LaTeX."""
    return text.replace("$", r"\$")


def summarize_sources(documents: list) -> list[dict]:
    """One row per project from the retrieved chunks (no raw chunk text)."""
    sources: dict[str, dict] = {}
    for doc in documents:
        meta = doc.metadata
        source = sources.setdefault(meta["project_id"], {
            "Project": f"{meta['project_id']} — {meta['project_name']}",
            "Type": meta["type"],
            "Domain": meta["domain"],
            "Pages": set(),
            "Retrieved by": set(),
        })
        source["Pages"].add(meta["page"])
        source["Retrieved by"].add(meta["retrieval_method"])

    for source in sources.values():
        source["Pages"] = ", ".join(str(page) for page in sorted(source["Pages"]))
        methods = source["Retrieved by"]
        source["Retrieved by"] = "both" if "both" in methods or len(methods) > 1 else methods.pop()
    return list(sources.values())


def show_result(result: dict) -> None:
    st.subheader("Answer")
    if result["answer"].strip() == OUT_OF_SCOPE_ANSWER:
        # Off-topic question: the retrieved chunks are irrelevant, so don't show them.
        st.info(result["answer"])
        return
    st.markdown(escape_markdown(result["answer"]))

    retry_note = " after 1 retry" if result["retry_count"] else ""
    if result["is_valid"]:
        st.caption(f"✅ Verified{retry_note}: {result['verification_feedback']}")
    else:
        st.warning(f"Verification did not pass{retry_note}: {result['verification_feedback']}")

    st.subheader("Sources")
    sources = summarize_sources(result["documents"])
    if sources:
        st.dataframe(sources, hide_index=True, width="stretch")
    else:
        st.write("No retrieved chunks were used.")

    if result["metadata_filters"]:
        matches = result["metadata_matches"]
        with st.expander(f"Projects matching metadata filters ({len(matches)}) — {describe_filters(result['metadata_filters'])}"):
            st.dataframe(
                [
                    {
                        "Project": f"{m['project_id']} — {m['project_name']}",
                        "Type": m["type"],
                        "Team Size": m["team_size"],
                        "Duration": m["duration"],
                        "Technologies": ", ".join(m["technology_stack"]),
                    }
                    for m in matches
                ],
                hide_index=True,
                width="stretch",
            )

    with st.expander("Retrieval details"):
        st.write(f"**Search query:** {result['search_query']}")
        st.write(f"**Retries used:** {result['retry_count']}")


def main() -> None:
    st.set_page_config(page_title="Project Repository RAG", page_icon="📚")
    st.title("Project Repository RAG")
    st.caption("Hybrid retrieval (FAISS + BM25) · Groq · LangGraph with answer verification")

    missing = [path.name for path in (FAISS_INDEX_DIR, BM25_PATH, DATABASE_PATH) if not path.exists()]
    if missing:
        st.error(f"Missing indexes: {', '.join(missing)}. Run `python ingestion/build_index.py` first.")
        st.stop()

    graph = load_rag_graph()

    question = st.text_input("Ask a question about the project repository")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving, answering and verifying..."):
            try:
                result = graph.invoke({"question": question.strip()}, config={"run_name": RUN_NAME})
            except Exception as error:  # show API errors (e.g. rate limits) instead of a stack trace
                st.error(f"Something went wrong: {error}")
                st.stop()
        show_result(result)


main()
