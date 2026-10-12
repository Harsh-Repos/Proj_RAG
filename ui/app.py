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

from graph.answer_cache import clear_answer_cache  # noqa: E402
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

    if result.get("cache_hit"):
        st.caption("⚡ Answered from the answer cache (this question was verified earlier).")

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
        st.write(f"**Rewritten search query:** {result['search_query']}")
        st.write(f"**Retries used:** {result['retry_count']}")
        st.write(f"**From answer cache:** {'yes' if result.get('cache_hit') else 'no'}")


def start_answering() -> None:
    """Ask-button callback: remember the question and lock the input until it is answered.

    Without the lock, clicking Ask again while an answer is being generated starts a second
    question in parallel; both then compete for Groq's tokens-per-minute limit (HTTP 429).
    """
    question = st.session_state.question_input.strip()
    if question:
        st.session_state.pending_question = question
        st.session_state.is_answering = True


def main() -> None:
    st.set_page_config(page_title="Project Repository RAG", page_icon="📚")
    st.title("Project Repository RAG")
    st.caption("Hybrid retrieval (FAISS + BM25) · Groq · LangGraph with answer verification")

    missing = [path.name for path in (FAISS_INDEX_DIR, BM25_PATH, DATABASE_PATH) if not path.exists()]
    if missing:
        st.error(f"Missing indexes: {', '.join(missing)}. Run `python ingestion/build_index.py` first.")
        st.stop()

    graph = load_rag_graph()

    st.session_state.setdefault("is_answering", False)
    is_answering = st.session_state.is_answering

    if st.sidebar.button("Clear answer cache", disabled=is_answering):
        clear_answer_cache()
        st.sidebar.success("Answer cache cleared.")

    st.text_input("Ask a question about the project repository", key="question_input", disabled=is_answering)
    st.button("Ask", type="primary", on_click=start_answering, disabled=is_answering)

    if is_answering:
        with st.spinner("Retrieving, answering and verifying... (Ask is locked until this answer is ready)"):
            try:
                st.session_state.last_result = graph.invoke(
                    {"question": st.session_state.pending_question}, config={"run_name": RUN_NAME}
                )
                st.session_state.last_error = None
            except Exception as error:  # show API errors (e.g. rate limits) instead of a stack trace
                st.session_state.last_result = None
                st.session_state.last_error = str(error)
        st.session_state.is_answering = False
        st.rerun()  # redraw with the input unlocked and the new answer

    if st.session_state.get("last_error"):
        st.error(f"Something went wrong: {st.session_state.last_error}")
    elif st.session_state.get("last_result"):
        show_result(st.session_state.last_result)


main()
