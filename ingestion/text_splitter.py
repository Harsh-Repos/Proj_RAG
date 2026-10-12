"""Split one project into chunks that all inherit the project's metadata.

Chunking strategy: structure-based (one chunk per project page)
  1. Pages are first grouped by project (document_loader.py), so a chunk never mixes projects.
  2. Each page is split on its own. Page 1 = "summary" section, page 2 = "technical_deep_dive".
  3. RecursiveCharacterTextSplitter (1000 characters, 100 overlap) cuts at paragraph/line
     boundaries. Every page in this PDF is 418-764 characters, so each page becomes exactly
     ONE chunk (200 chunks for 100 projects); the size limit is only a safety net for longer pages.
  4. A short header (ID, name, type, domain, team size, duration) is added to every chunk, so
     each chunk names its project for both FAISS and BM25.
  5. Every chunk inherits the project's metadata plus page, section and chunk_id.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.document_loader import get_page_number

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def build_chunk_header(metadata: dict) -> str:
    """Short project header prepended to every chunk.

    A later chunk of P038 may not mention "P038" or its name at all. The header
    keeps the project identity inside the text, so both FAISS and BM25 can match it.
    """
    return (
        f"Project {metadata['project_id']}: {metadata['project_name']}\n"
        f"Type: {metadata['type']} | Domain: {metadata['domain']} | "
        f"Team Size: {metadata['team_size']} people | Duration: {metadata['duration']}\n\n"
    )


def split_project_into_chunks(project: dict, metadata: dict) -> list[Document]:
    """Split each page of a project separately so every chunk keeps its exact page number."""
    header = build_chunk_header(metadata)
    chunk_metadata = {key: value for key, value in metadata.items() if key != "pages"}

    chunks = []
    for page_index, page in enumerate(project["pages"]):
        page_number = get_page_number(page)
        # The first page is the project summary (problem, solution approach, details,
        # tech stack, outcomes); the second is the "Technical Deep Dive", which repeats parts of it.
        section = "summary" if page_index == 0 else "technical_deep_dive"
        for index, text in enumerate(text_splitter.split_text(page.page_content)):
            chunks.append(
                Document(
                    page_content=header + text,
                    metadata={
                        **chunk_metadata,
                        "page": page_number,
                        "section": section,
                        "chunk_id": f"{metadata['project_id']}-p{page_number}-c{index}",
                    },
                )
            )
    return chunks
