"""OFFLINE INGESTION: the only place where indexes are built.

Usage:
    python ingestion/build_index.py            # build once (stops if indexes exist)
    python ingestion/build_index.py --rebuild  # delete and rebuild everything
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Allow "python ingestion/build_index.py" to import the ingestion package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.answer_cache import clear_answer_cache  # noqa: E402
from ingestion.database import DATABASE_PATH, create_database, insert_projects  # noqa: E402
from ingestion.document_loader import load_pdf, group_pages_by_project  # noqa: E402
from ingestion.embeddings import get_embedding_model  # noqa: E402
from ingestion.keyword_retriever import BM25_PATH, build_keyword_retriever  # noqa: E402
from ingestion.metadata_extractor import build_project_text, extract_project_metadata  # noqa: E402
from ingestion.text_splitter import split_project_into_chunks  # noqa: E402
from ingestion.vector_store import FAISS_INDEX_DIR, build_vector_store  # noqa: E402

INDEX_PATHS = [FAISS_INDEX_DIR, BM25_PATH, DATABASE_PATH]

logger = logging.getLogger("build_index")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FAISS, BM25 and SQLite indexes from the PDF.")
    parser.add_argument("--rebuild", action="store_true", help="delete existing indexes and rebuild them")
    return parser.parse_args()


def remove_existing_indexes() -> None:
    """Delete previously built indexes and the cached answers built from them (only with --rebuild)."""
    for path in INDEX_PATHS:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    clear_answer_cache()


def parse_projects() -> tuple[list[dict], list]:
    """PDF -> projects -> metadata -> chunks. Returns (project records, chunks)."""
    pages = load_pdf()
    projects = group_pages_by_project(pages)
    logger.info("Loaded %d pages, found %d projects.", len(pages), len(projects))

    project_records, chunks, problems = [], [], []
    for project in projects:
        try:
            metadata = extract_project_metadata(project)
        except ValueError as error:
            problems.append(str(error))
            continue
        project_records.append({**metadata, "content": build_project_text(project)})
        chunks.extend(split_project_into_chunks(project, metadata))

    project_ids = [record["project_id"] for record in project_records]
    duplicates = sorted({pid for pid in project_ids if project_ids.count(pid) > 1})
    if duplicates:
        problems.append(f"Duplicate project IDs: {', '.join(duplicates)}")

    if problems:
        logger.warning("Metadata problems (these projects were NOT indexed):")
        for problem in problems:
            logger.warning("  - %s", problem)
    if not project_records or duplicates:
        raise SystemExit("Ingestion aborted: fix the problems above and run again.")
    return project_records, chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy_logger in ("httpx", "faiss.loader"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    args = parse_arguments()

    existing = [path for path in INDEX_PATHS if path.exists()]
    if existing and not args.rebuild:
        logger.info("Indexes already exist: %s", ", ".join(path.name for path in existing))
        logger.info("Nothing to do. Use --rebuild to rebuild them.")
        return
    if args.rebuild:
        logger.info("--rebuild: removing existing indexes.")
        remove_existing_indexes()
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    project_records, chunks = parse_projects()
    logger.info("Created %d chunks from %d projects.", len(chunks), len(project_records))

    logger.info("Building FAISS index (Hugging Face embeddings)...")
    build_vector_store(chunks, get_embedding_model())

    logger.info("Building BM25 index...")
    build_keyword_retriever(chunks)

    logger.info("Creating SQLite database...")
    create_database()
    insert_projects(project_records)

    logger.info("Done. Saved %s, %s and %s.", FAISS_INDEX_DIR, BM25_PATH, DATABASE_PATH)


if __name__ == "__main__":
    main()
