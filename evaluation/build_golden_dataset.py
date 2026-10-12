"""Build the golden dataset: questions with known correct answers.

Usage:
    python evaluation/build_golden_dataset.py

Every correct answer is taken straight from data/project.db with plain SQL, never from
the RAG pipeline, so the dataset is an independent answer key. The result is written to
evaluation/golden_dataset.json, which can be read and edited by hand.

Each item:
    id                    "Q01"
    category              fact_lookup | technology_stack | section_lookup | semantic_search |
                          metadata_filter | typo_filter | out_of_scope | not_in_documents
    question              what the user asks
    expected_project_ids  projects a correct answer must mention ([] when none)
    reference_answer      the correct answer, used by the correctness judge
"""

import json
import sqlite3
import sys
from pathlib import Path

# Allow "python evaluation/build_golden_dataset.py" to import the project packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.prompts import NO_ANSWER, OUT_OF_SCOPE_ANSWER  # noqa: E402
from ingestion.database import DATABASE_PATH  # noqa: E402
from ingestion.metadata_extractor import get_section  # noqa: E402

GOLDEN_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"

FACT_LOOKUP_PROJECTS = ["P057", "P012", "P051", "P090", "P033"]
TECHNOLOGY_STACK_PROJECTS = ["P038", "P005", "P025"]
SECTION_LOOKUPS = [  # (project, section heading, next heading, wording in the question)
    ("P090", "SOLUTION APPROACH", "PROJECT DETAILS", "solution approach"),
    ("P001", "PROBLEM STATEMENT", "SOLUTION APPROACH", "problem statement"),
    ("P045", "SOLUTION APPROACH", "PROJECT DETAILS", "solution approach"),
]
SEMANTIC_SEARCHES = [  # questions that describe a project without naming it
    ("Which project predicts which customers are likely to stop using a service?", "P001"),
    ("Which project built a chatbot that answers employee questions about HR policies?", "P011"),
    ("Which project lets business users ask questions in plain English and turns them into SQL?", "P061"),
]
METADATA_FILTERS = [  # (question, SQL WHERE clause that defines the correct answer)
    ("Which projects have a team size less than 5 and a duration less than 3 months?",
     "team_size < 5 AND duration_weeks < 13"),
    ("Which projects use Python and Snowflake?",
     "technology_stack LIKE '%Python%' AND technology_stack LIKE '%Snowflake%'"),
    ("Which NLP / GenAI projects have a team size of at least 5?",
     "type = 'NLP / GenAI' AND team_size >= 5"),
    ("Which Data Engineering projects last longer than 6 months?",
     "type = 'Data Engineering' AND duration_weeks > 26"),
    ("List the projects with a team size of more than 6 people.",
     "team_size > 6"),
]
TYPO_FILTERS = [  # same conditions as above, written with typos (tests the query rewriter)
    ("which projcts use pythn and snowflak",
     "technology_stack LIKE '%Python%' AND technology_stack LIKE '%Snowflake%'"),
]
OUT_OF_SCOPE_QUESTIONS = [
    "What is the weather in Delhi today?",
    "What is Apache Kafka in general?",
]
NOT_IN_DOCUMENTS_QUESTIONS = [  # about a real project, but the PDF does not contain the answer
    "Who was the project manager of P038?",
]


def load_projects() -> dict[str, dict]:
    """Every project row from SQLite, keyed by project ID."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM projects").fetchall()
    return {row["project_id"]: dict(row) for row in rows}


def find_project_ids(where_clause: str) -> list[str]:
    """Project IDs matching a SQL condition (the independent answer key for filter questions)."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(f"SELECT project_id FROM projects WHERE {where_clause} ORDER BY project_id")
        return [row[0] for row in rows]


def build_fact_questions(projects: dict) -> list[dict]:
    return [
        {
            "category": "fact_lookup",
            "question": f"What is the team size and duration of {pid}?",
            "expected_project_ids": [pid],
            "reference_answer": f"{pid} has a team size of {projects[pid]['team_size']} people "
                                f"and a duration of {projects[pid]['duration']}.",
        }
        for pid in FACT_LOOKUP_PROJECTS
    ]


def build_technology_questions(projects: dict) -> list[dict]:
    return [
        {
            "category": "technology_stack",
            "question": f"What technologies are used in {pid}?",
            "expected_project_ids": [pid],
            "reference_answer": f"{pid} uses {projects[pid]['technology_stack']}.",
        }
        for pid in TECHNOLOGY_STACK_PROJECTS
    ]


def build_section_questions(projects: dict) -> list[dict]:
    items = []
    for pid, heading, next_heading, wording in SECTION_LOOKUPS:
        section_text = get_section(projects[pid]["content"], heading, next_heading)
        items.append({
            "category": "section_lookup",
            "question": f"What is the {wording} of {pid}?",
            "expected_project_ids": [pid],
            "reference_answer": " ".join(section_text.split()),
        })
    return items


def build_semantic_questions(projects: dict) -> list[dict]:
    return [
        {
            "category": "semantic_search",
            "question": question,
            "expected_project_ids": [pid],
            "reference_answer": f"{pid} - {projects[pid]['project_name']}.",
        }
        for question, pid in SEMANTIC_SEARCHES
    ]


def build_filter_questions(projects: dict, filters: list[tuple[str, str]], category: str) -> list[dict]:
    items = []
    for question, where_clause in filters:
        ids = find_project_ids(where_clause)
        listing = ", ".join(f"{pid} ({projects[pid]['project_name']})" for pid in ids)
        items.append({
            "category": category,
            "question": question,
            "expected_project_ids": ids,
            "reference_answer": f"{len(ids)} projects match: {listing}." if ids else NO_ANSWER,
        })
    return items


def build_special_questions() -> list[dict]:
    out_of_scope = [
        {"category": "out_of_scope", "question": q, "expected_project_ids": [], "reference_answer": OUT_OF_SCOPE_ANSWER}
        for q in OUT_OF_SCOPE_QUESTIONS
    ]
    not_in_documents = [
        {"category": "not_in_documents", "question": q, "expected_project_ids": [], "reference_answer": NO_ANSWER}
        for q in NOT_IN_DOCUMENTS_QUESTIONS
    ]
    return out_of_scope + not_in_documents


def main() -> None:
    if not DATABASE_PATH.exists():
        raise SystemExit("data/project.db not found. Run: python ingestion/build_index.py")

    projects = load_projects()
    items = (
        build_fact_questions(projects)
        + build_technology_questions(projects)
        + build_section_questions(projects)
        + build_semantic_questions(projects)
        + build_filter_questions(projects, METADATA_FILTERS, "metadata_filter")
        + build_filter_questions(projects, TYPO_FILTERS, "typo_filter")
        + build_special_questions()
    )
    dataset = [{"id": f"Q{number:02d}", **item} for number, item in enumerate(items, start=1)]

    GOLDEN_DATASET_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(dataset)} questions to {GOLDEN_DATASET_PATH}")


if __name__ == "__main__":
    main()
