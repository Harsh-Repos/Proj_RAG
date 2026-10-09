"""Assemble the controlled Hybrid RAG LangGraph.

    START -> analyze_query -> (semantic_retrieve || keyword_retrieve) -> merge_documents
          -> generate_answer -> verify_answer -> END
                                     | invalid and retry_count < 1
                                     v
                               refine_query -> (semantic_retrieve || keyword_retrieve) -> ...
"""

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    analyze_query,
    generate_answer,
    keyword_retrieve,
    merge_documents,
    refine_query,
    semantic_retrieve,
    verify_answer,
)
from graph.routing import check_verification
from graph.state import RAGState


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(RAGState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("semantic_retrieve", semantic_retrieve)
    graph.add_node("keyword_retrieve", keyword_retrieve)
    graph.add_node("merge_documents", merge_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_answer", verify_answer)
    graph.add_node("refine_query", refine_query)

    graph.add_edge(START, "analyze_query")

    # Fan out: FAISS and BM25 run in parallel.
    graph.add_edge("analyze_query", "semantic_retrieve")
    graph.add_edge("analyze_query", "keyword_retrieve")

    # Fan in: merge waits for both retrievers.
    graph.add_edge(["semantic_retrieve", "keyword_retrieve"], "merge_documents")

    graph.add_edge("merge_documents", "generate_answer")
    graph.add_edge("generate_answer", "verify_answer")

    graph.add_conditional_edges(
        "verify_answer",
        check_verification,
        {"retry": "refine_query", "end": END},
    )

    # The single retry re-runs both retrievers with the refined query.
    graph.add_edge("refine_query", "semantic_retrieve")
    graph.add_edge("refine_query", "keyword_retrieve")

    return graph.compile()
