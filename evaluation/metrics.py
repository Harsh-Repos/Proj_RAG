"""Evaluation metrics: precision, recall, F1, faithfulness and correctness.

Precision / Recall / F1 (deterministic, no LLM)
    Compare the project IDs the answer mentions with the expected project IDs.
      precision = correct IDs in the answer / all IDs in the answer   ("did it mention wrong projects?")
      recall    = correct IDs in the answer / all expected IDs        ("did it miss projects?")
      F1        = 2 * precision * recall / (precision + recall)       (one number for both)
    They apply to questions whose answer IS a set of projects (PROJECT_LIST_CATEGORIES).
    A question that already names its project ("team size of P057?") is answered without
    repeating the ID, so for those, correctness and faithfulness are the right measures.
    The same formula on the retrieved context gives context precision / recall (retrieval quality).

Faithfulness (LLM judge)
    supported claims / total claims in the answer, checked against the retrieved context.
    1.0 = nothing made up. A fixed reply ("I don't know...", out-of-scope) has no claims -> 1.0.

Correctness (LLM judge, or exact match for fixed replies)
    The answer is compared with the golden reference answer:
      correct = 1.0, partially_correct = 0.5, incorrect = 0.0
"""

from typing import Literal

from pydantic import BaseModel, Field

from evaluation.judge_prompts import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT
from graph.metadata_filters import PROJECT_ID_PATTERN
from graph.nodes import invoke_structured
from graph.prompts import NO_ANSWER, OUT_OF_SCOPE_ANSWER

FIXED_REPLIES = (NO_ANSWER, OUT_OF_SCOPE_ANSWER)
FIXED_REPLY_CATEGORIES = ("out_of_scope", "not_in_documents")
# Categories where the answer is a set of projects (or, for fixed replies, no project at all).
PROJECT_LIST_CATEGORIES = ("semantic_search", "metadata_filter", "typo_filter", *FIXED_REPLY_CATEGORIES)
CORRECTNESS_SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


class FaithfulnessJudgement(BaseModel):
    total_claims: int = Field(description="Number of factual claims in the answer.")
    supported_claims: int = Field(description="Number of those claims the context supports.")
    unsupported_claims: str = Field(description="The unsupported claims, or 'none'.")


class CorrectnessJudgement(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    reason: str = Field(description="One short sentence explaining the verdict.")


def extract_project_ids(text: str) -> set[str]:
    """All project IDs mentioned in a text, e.g. {"P038", "P057"}."""
    return {project_id.upper() for project_id in PROJECT_ID_PATTERN.findall(text)}


def precision_recall_f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    """Set-based precision, recall and F1.

    When nothing is expected (off-topic or unanswerable question), mentioning no project
    is a perfect score and mentioning any project is a precision of 0.
    """
    if not expected:
        score = 1.0 if not predicted else 0.0
        return score, score, score
    if not predicted:
        return 0.0, 0.0, 0.0
    correct = len(predicted & expected)
    precision = correct / len(predicted)
    recall = correct / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if correct else 0.0
    return precision, recall, f1


def score_faithfulness(question: str, answer: str, context: str) -> tuple[float, str]:
    """Share of the answer's claims that the retrieved context supports (0.0 to 1.0)."""
    if answer.strip() in FIXED_REPLIES:
        return 1.0, "Fixed reply, no claims to check."
    judgement = invoke_structured(FAITHFULNESS_PROMPT, FaithfulnessJudgement, {
        "question": question,
        "context": context,
        "answer": answer,
    })
    if judgement.total_claims <= 0:
        return 1.0, "No factual claims."
    score = min(judgement.supported_claims, judgement.total_claims) / judgement.total_claims
    return score, f"Unsupported: {judgement.unsupported_claims}"


def score_correctness(question: str, answer: str, reference_answer: str, category: str) -> tuple[float, str]:
    """How well the answer matches the golden reference answer (1.0, 0.5 or 0.0)."""
    if category in FIXED_REPLY_CATEGORIES:
        matches = answer.strip() == reference_answer.strip()
        return (1.0, "Expected fixed reply given.") if matches else (0.0, "Expected fixed reply not given.")
    judgement = invoke_structured(CORRECTNESS_PROMPT, CorrectnessJudgement, {
        "question": question,
        "reference_answer": reference_answer,
        "answer": answer,
    })
    return CORRECTNESS_SCORES[judgement.verdict], f"{judgement.verdict}: {judgement.reason}"
