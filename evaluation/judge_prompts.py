"""LLM judge prompts used by evaluation/metrics.py (faithfulness and correctness)."""

from langchain_core.prompts import ChatPromptTemplate

# Faithfulness: is every claim in the answer supported by the retrieved context?
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are evaluating a RAG system. Measure FAITHFULNESS: whether the answer is supported "
     "by the retrieved context.\n"
     "Steps:\n"
     "1. Split the ANSWER into its individual factual claims (one fact each, e.g. "
     "'P038 has 10 people', 'P038 uses Kafka'). For a list of projects, each listed project "
     "with its stated details is one claim. If the answer does not repeat the project, its "
     "claims are about the project(s) named in the QUESTION.\n"
     "2. For each claim, decide if the CONTEXT supports it. A claim is supported only if the "
     "context states it; outside knowledge does not count.\n"
     "Return total_claims, supported_claims and the unsupported claims (or 'none')."),
    ("human",
     "QUESTION: {question}\n\n"
     "CONTEXT:\n{context}\n\n"
     "ANSWER:\n{answer}"),
])

# Correctness: does the answer match the reference (golden) answer?
CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are evaluating a RAG system. Measure CORRECTNESS: whether the ANSWER gives the same "
     "facts as the REFERENCE ANSWER for the QUESTION.\n"
     "- correct: all key facts of the reference are present and nothing contradicts it.\n"
     "- partially_correct: some key facts are present, but some are missing or wrong.\n"
     "- incorrect: the key facts are missing or contradict the reference.\n"
     "Different wording, order or formatting is fine. Judge the facts only."),
    ("human",
     "QUESTION: {question}\n\n"
     "REFERENCE ANSWER:\n{reference_answer}\n\n"
     "ANSWER:\n{answer}"),
])
