"""SQLite store for structured, project-level metadata.

SQLite is NOT a retriever. It holds one clean row per project so that:
  1. the repository can be inspected (e.g. with the SQLite VS Code extension), and
  2. numeric/categorical conditions (team size, duration, type, technology)
     can be checked deterministically in Python.
"""

import sqlite3
from pathlib import Path

from ingestion.document_loader import PROJECT_ROOT

DATABASE_PATH = PROJECT_ROOT / "data" / "project.db"
TECHNOLOGY_DELIMITER = ", "

CREATE_PROJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS projects (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id       TEXT NOT NULL UNIQUE,
    project_name     TEXT NOT NULL,
    domain           TEXT NOT NULL,
    type             TEXT NOT NULL,
    team_size        INTEGER NOT NULL,
    duration         TEXT NOT NULL,
    duration_weeks   REAL NOT NULL,
    technology_stack TEXT NOT NULL,
    pages            TEXT NOT NULL,
    content          TEXT NOT NULL
)
"""


def create_database(path: Path = DATABASE_PATH) -> None:
    """Create the projects table."""
    with sqlite3.connect(path) as connection:
        connection.execute(CREATE_PROJECTS_TABLE)


def insert_projects(projects: list[dict], path: Path = DATABASE_PATH) -> None:
    """Insert one row per project. Each item holds the metadata fields plus "content"."""
    rows = [
        (
            project["project_id"],
            project["project_name"],
            project["domain"],
            project["type"],
            project["team_size"],
            project["duration"],
            project["duration_weeks"],
            TECHNOLOGY_DELIMITER.join(project["technology_stack"]),
            TECHNOLOGY_DELIMITER.join(str(page) for page in project["pages"]),
            project["content"],
        )
        for project in projects
    ]
    with sqlite3.connect(path) as connection:
        connection.executemany(
            """INSERT INTO projects (project_id, project_name, domain, type, team_size,
                                     duration, duration_weeks, technology_stack, pages, content)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def load_project_metadata(path: Path = DATABASE_PATH) -> list[dict]:
    """Load every project's structured metadata (without the full content)."""
    if not path.exists():
        raise FileNotFoundError(f"Database not found at {path}. Run: python ingestion/build_index.py")
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT project_id, project_name, domain, type, team_size,
                      duration, duration_weeks, technology_stack
               FROM projects ORDER BY project_id"""
        ).fetchall()

    projects = [dict(row) for row in rows]
    for project in projects:
        project["technology_stack"] = project["technology_stack"].split(TECHNOLOGY_DELIMITER)
    return projects
