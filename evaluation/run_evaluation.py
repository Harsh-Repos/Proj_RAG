"""Run the golden dataset through the RAG pipeline and score every answer.

Usage:
    python evaluation/run_evaluation.py                 # all questions
    python evaluation/run_evaluation.py --limit 5       # first 5 questions
    python evaluation/run_evaluation.py --ids Q01,Q15   # selected questions (other results are kept)
    python evaluation/run_evaluation.py --rerun-errors  # only questions that failed last time
                                                        # (e.g. Groq rate limit), merged into the results

The answer cache is switched off (use_cache=False), so every question really runs.
Results are written to evaluation/results/:
    evaluation_results.csv    one row per question (answer, scores, judge reasons), for reading
    evaluation_results.json   the same rows, used by --rerun-errors
    evaluation_summary.json   average scores, overall and per category
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# Allow "python evaluation/run_evaluation.py" to import the project packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.build_golden_dataset import GOLDEN_DATASET_PATH  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    PROJECT_LIST_CATEGORIES,
    extract_project_ids,
    precision_recall_f1,
    score_correctness,
    score_faithfulness,
)
from graph.nodes import format_context  # noqa: E402
from graph.workflow import build_graph  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV_PATH = RESULTS_DIR / "evaluation_results.csv"
RESULTS_JSON_PATH = RESULTS_DIR / "evaluation_results.json"
SUMMARY_JSON_PATH = RESULTS_DIR / "evaluation_summary.json"

SCORE_COLUMNS = [
    "faithfulness", "correctness", "precision", "recall", "f1",
    "context_precision", "context_recall", "latency_seconds",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden dataset.")
    parser.add_argument("--limit", type=int, help="only evaluate the first N questions")
    parser.add_argument("--ids", help="comma-separated question IDs, e.g. Q01,Q15")
    parser.add_argument("--rerun-errors", action="store_true",
                        help="re-run only the questions that failed in the last run and merge the results")
    return parser.parse_args()


def load_previous_rows() -> list[dict]:
    """Rows saved by the last run (needed by --rerun-errors)."""
    if not RESULTS_JSON_PATH.exists():
        raise SystemExit("No previous results found. Run the evaluation once without --rerun-errors.")
    return json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))


def load_golden_dataset(args: argparse.Namespace) -> list[dict]:
    if not GOLDEN_DATASET_PATH.exists():
        raise SystemExit("Golden dataset not found. Run: python evaluation/build_golden_dataset.py")
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if args.ids:
        wanted = {qid.strip().upper() for qid in args.ids.split(",")}
        dataset = [item for item in dataset if item["id"] in wanted]
    return dataset[: args.limit] if args.limit else dataset


def context_project_ids(result: dict) -> set[str]:
    """Projects the LLM saw: retrieved chunks plus the metadata-filter match list."""
    ids = {doc.metadata["project_id"] for doc in result["documents"]}
    ids |= {match["project_id"] for match in result["metadata_matches"]}
    return ids


def evaluate_question(graph, item: dict) -> dict:
    """Run one golden question through the graph and compute all metrics."""
    start = time.perf_counter()
    result = graph.invoke(
        {"question": item["question"], "use_cache": False},
        config={"run_name": f"Evaluation {item['id']}"},
    )
    latency = time.perf_counter() - start

    answer = result["answer"]
    expected = set(item["expected_project_ids"])
    precision = recall = f1 = None
    if item["category"] in PROJECT_LIST_CATEGORIES:
        precision, recall, f1 = precision_recall_f1(extract_project_ids(answer), expected)
    # Retrieval always returns some chunks, so context scores only make sense when projects are expected.
    context_precision = context_recall = None
    if expected:
        context_precision, context_recall, _ = precision_recall_f1(context_project_ids(result), expected)
    faithfulness, faithfulness_reason = score_faithfulness(item["question"], answer, format_context(result))
    correctness, correctness_reason = score_correctness(
        item["question"], answer, item["reference_answer"], item["category"]
    )

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer,
        "faithfulness": faithfulness,
        "correctness": correctness,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "latency_seconds": latency,
        "verified": result["is_valid"],
        "retries": result["retry_count"],
        "faithfulness_reason": faithfulness_reason,
        "correctness_reason": correctness_reason,
        "error": "",
    }


def average(rows: list[dict], column: str) -> float | None:
    """Mean of a score column, skipping questions where the score does not apply (None)."""
    values = [row[column] for row in rows if row[column] is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(rows: list[dict]) -> dict:
    """Average every score overall and per category (questions that errored are excluded)."""
    scored = [row for row in rows if not row["error"]]
    summary = {
        "questions": len(rows),
        "errors": len(rows) - len(scored),
        "verified_rate": round(sum(row["verified"] for row in scored) / len(scored), 3) if scored else None,
        "retry_rate": round(sum(row["retries"] > 0 for row in scored) / len(scored), 3) if scored else None,
        "overall": {column: average(scored, column) for column in SCORE_COLUMNS} if scored else {},
        "by_category": {},
    }
    for category in dict.fromkeys(row["category"] for row in scored):
        in_category = [row for row in scored if row["category"] == category]
        summary["by_category"][category] = {
            "questions": len(in_category),
            **{column: average(in_category, column) for column in SCORE_COLUMNS},
        }
    return summary


def save_results(rows: list[dict], summary: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    RESULTS_JSON_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def print_summary(summary: dict) -> None:
    print("\n=== Evaluation summary ===")
    print(f"Questions: {summary['questions']}  Errors: {summary['errors']}  "
          f"Verified: {summary['verified_rate']}  Retried: {summary['retry_rate']}")
    print(f"{'category':<18}{'n':>3}" + "".join(f"{column[:12]:>13}" for column in SCORE_COLUMNS))
    rows = [("OVERALL", summary["questions"] - summary["errors"], summary["overall"])]
    rows += [(category, scores["questions"], scores) for category, scores in summary["by_category"].items()]
    for name, count, scores in rows:
        cells = "".join(f"{'-' if scores.get(column) is None else scores[column]:>13}" for column in SCORE_COLUMNS)
        print(f"{name:<18}{count:>3}{cells}")


def main() -> None:
    args = parse_arguments()
    dataset = load_golden_dataset(args)
    kept_rows: dict[str, dict] = {}
    if args.rerun_errors:
        kept_rows = {row["id"]: row for row in load_previous_rows() if not row["error"]}
        dataset = [item for item in dataset if item["id"] not in kept_rows]
    elif args.ids and RESULTS_JSON_PATH.exists():
        # Re-running selected questions: keep every other question's previous result.
        selected = {item["id"] for item in dataset}
        kept_rows = {row["id"]: row for row in load_previous_rows() if row["id"] not in selected}
    if kept_rows:
        print(f"Keeping {len(kept_rows)} previous results, running {len(dataset)}.")
    graph = build_graph()

    rows = []
    for number, item in enumerate(dataset, start=1):
        print(f"[{number}/{len(dataset)}] {item['id']} {item['question']}", flush=True)
        try:
            row = evaluate_question(graph, item)
            f1_text = "-" if row["f1"] is None else f"{row['f1']:.2f}"
            print(f"    faithfulness {row['faithfulness']:.2f}  correctness {row['correctness']:.2f}  "
                  f"F1 {f1_text}  ({row['latency_seconds']:.1f}s)", flush=True)
        except Exception as error:  # keep going: one API failure should not lose the whole run
            print(f"    ERROR: {error}", flush=True)
            row = {"id": item["id"], "category": item["category"], "question": item["question"], "answer": "",
                   **{column: 0.0 for column in SCORE_COLUMNS}, "verified": False, "retries": 0,
                   "faithfulness_reason": "", "correctness_reason": "", "error": str(error)}
        rows.append(row)

    if kept_rows:
        rows = sorted([*kept_rows.values(), *rows], key=lambda row: row["id"])
    summary = summarize(rows)
    save_results(rows, summary)
    print_summary(summary)
    print(f"\nSaved {RESULTS_CSV_PATH} and {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
