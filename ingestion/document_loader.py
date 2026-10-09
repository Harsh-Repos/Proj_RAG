"""Load the project repository PDF and group its pages into projects."""

import logging
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "Document" / "Dataset_project_repository.pdf"

# Every project page starts with its ID, e.g. "P038 Data Mesh Implementation ..."
# (summary page) or "P038  ›  Data Mesh Implementation ... — Technical Deep Dive".
PROJECT_HEADER_PATTERN = re.compile(r"^(P\d{3})\b")

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: Path = PDF_PATH) -> list[Document]:
    """Load the PDF with PyPDFLoader, one Document per page."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return PyPDFLoader(str(pdf_path)).load()


def get_page_number(page: Document) -> int:
    """Return the 1-based page number (PyPDFLoader stores a 0-based index)."""
    return page.metadata.get("page", 0) + 1


def group_pages_by_project(pages: list[Document]) -> list[dict]:
    """Group consecutive pages that belong to the same project.

    A page whose first line starts with a new project ID begins a new project.
    A page without an ID continues the current project. Pages before the first
    project (the cover page) are skipped.

    Returns a list of {"project_id": "P038", "pages": [Document, ...]}.
    """
    projects: list[dict] = []
    for page in pages:
        first_line = page.page_content.strip().split("\n", 1)[0]
        match = PROJECT_HEADER_PATTERN.match(first_line)

        if match and (not projects or projects[-1]["project_id"] != match.group(1)):
            projects.append({"project_id": match.group(1), "pages": [page]})
        elif projects:
            projects[-1]["pages"].append(page)
        else:
            logger.info("Skipping page %d: no project header (cover page).", get_page_number(page))
    return projects
