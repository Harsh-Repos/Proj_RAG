"""Extract structured project metadata from a project's summary page.

Each project's first page has a fixed layout:

    P038 Data Mesh Implementation for Telco Data Engineering
    PROBLEM STATEMENT ...
    SOLUTION APPROACH ...
    PROJECT DETAILS
    Domain Telecom / Data Platform
    Team Size 10 people
    Duration 9 months
    Type Data Engineering
    TECHNOLOGY STACK
    GCP Dataplex  ·  BigQuery  ·  dbt  ·  Apache Kafka  ·  Terraform
    ·  Python
    EXPECTED OUTCOMES ...

Metadata is read only from this page. The "Technical Deep Dive" page drops units
(e.g. "9 / Duration") and sometimes splits words, so it is not used for metadata.

Why these fields: every project's PROJECT DETAILS block has exactly the same fields, and they
are what users filter and ask by.
  project_id, project_name   identify and cite the project; BM25 matches exact IDs
  type, domain               filter by category ("Data Engineering projects")
  team_size                  numeric filter ("team size at least 5"), stored as int
  duration                   original text, shown to the user ("3 months")
  duration_weeks             duration in one unit, so "10 weeks" and "3 months" compare as numbers
  technology_stack           technology filter ("uses Python and Snowflake"), stored as a list
  pages                      source pages, for citations
"""

import re

from ingestion.document_loader import get_page_number

WEEKS_PER_MONTH = 52 / 12
TECHNOLOGY_SEPARATOR = "·"

DETAIL_PATTERNS = {
    "domain": re.compile(r"^Domain\s+(.+)$", re.MULTILINE),
    "team_size": re.compile(r"^Team Size\s+(\d+)\s+(?:people|person)", re.MULTILINE),
    "duration": re.compile(r"^Duration\s+(\d+\s+(?:weeks?|months?))\s*$", re.MULTILINE),
    "type": re.compile(r"^Type\s+(.+)$", re.MULTILINE),
}


def get_section(text: str, start_heading: str, end_heading: str) -> str:
    """Return the text between two headings, or "" if either is missing."""
    match = re.search(rf"{start_heading}\s*\n(.*?)\n\s*{end_heading}", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_technology_stack(section: str) -> list[str]:
    """Split the "A · B · C" stack, re-joining names that wrapped across lines.

    PDF text extraction inserts a space before hyphens ("GPT -4", "Scikit -learn"),
    so that artifact is removed to restore the names shown in the PDF.
    """
    joined = re.sub(r"\s+-", "-", " ".join(section.split()))
    return [tech.strip() for tech in joined.split(TECHNOLOGY_SEPARATOR) if tech.strip()]


def normalize_duration_to_weeks(duration: str) -> float:
    """Convert "5 weeks" / "3 months" into weeks so durations can be compared."""
    amount, unit = duration.split()
    weeks = float(amount) * (WEEKS_PER_MONTH if unit.startswith("month") else 1)
    return round(weeks, 2)


def extract_project_name(header_line: str, project_id: str, project_type: str) -> str:
    """Header is "<ID> <Project Name> <Type>"; strip the ID prefix and type suffix."""
    name = header_line.removeprefix(project_id).strip()
    if project_type and name.endswith(project_type):
        name = name[: -len(project_type)].strip()
    return name


def extract_project_metadata(project: dict) -> dict:
    """Extract metadata for one project. Raises ValueError if a field is missing."""
    project_id = project["project_id"]
    summary_text = project["pages"][0].page_content
    details = get_section(summary_text, "PROJECT DETAILS", "TECHNOLOGY STACK")

    fields = {}
    for field, pattern in DETAIL_PATTERNS.items():
        match = pattern.search(details)
        fields[field] = match.group(1).strip() if match else None

    header_line = summary_text.strip().split("\n", 1)[0]
    metadata = {
        "project_id": project_id,
        "project_name": extract_project_name(header_line, project_id, fields["type"] or ""),
        "domain": fields["domain"],
        "type": fields["type"],
        "team_size": int(fields["team_size"]) if fields["team_size"] else None,
        "duration": fields["duration"],
        "duration_weeks": normalize_duration_to_weeks(fields["duration"]) if fields["duration"] else None,
        "technology_stack": parse_technology_stack(
            get_section(summary_text, "TECHNOLOGY STACK", "EXPECTED OUTCOMES")
        ),
        "pages": [get_page_number(page) for page in project["pages"]],
    }

    missing = [key for key, value in metadata.items() if value in (None, "", [])]
    if missing:
        raise ValueError(f"{project_id}: could not extract {', '.join(missing)}")
    return metadata


def build_project_text(project: dict) -> str:
    """Join all pages of a project into its full text."""
    return "\n\n".join(page.page_content.strip() for page in project["pages"])
