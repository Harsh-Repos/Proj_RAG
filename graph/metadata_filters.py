"""Structured filtering (team size, duration, type, technology).

Questions like "team size of atleast 5 and duration more then 3 months" need two
different skills, so the work is split:

  1. UNDERSTANDING the question -> the LLM (in analyze_query) fills the
     QueryAnalysis schema below. It handles typos, slang and any wording
     ("atleast", "more then", "5+ people", "a quarter or longer").
  2. COMPARING the numbers -> plain Python. build_metadata_filters() validates
     the LLM output, and filter_projects_by_metadata() compares it against the
     exact values stored in SQLite. The LLM never decides which projects match.

This is not a third retriever: it only decides which projects satisfy the
stated conditions. FAISS + BM25 still provide the text context.
"""

import difflib
import operator
import re
from typing import Literal

from pydantic import BaseModel, Field

from ingestion.metadata_extractor import WEEKS_PER_MONTH

OPERATORS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "=": operator.eq}
PROJECT_TYPES = ("ML/AI", "NLP / GenAI", "Data Engineering")
PROJECT_ID_PATTERN = re.compile(r"\bP\d{3}\b", re.IGNORECASE)
TECHNOLOGY_MATCH_CUTOFF = 0.8  # how similar a misspelled name must be ("snowflak" -> "Snowflake")

Operator = Literal["<", "<=", ">", ">=", "="]


# ---------------------------------------------------------------------------
# Schema the LLM fills in analyze_query
# ---------------------------------------------------------------------------

class TeamSizeCondition(BaseModel):
    operator: Operator = Field(description="'at least 5' -> '>=', 'more than 5' -> '>', 'under 5' -> '<'")
    value: int = Field(description="Number of people")


class DurationCondition(BaseModel):
    operator: Operator = Field(description="'more than 3 months' -> '>', 'up to 6 weeks' -> '<='")
    value: float = Field(description="Amount of time, e.g. 3")
    unit: Literal["weeks", "months"]


class QueryAnalysis(BaseModel):
    """The explicit conditions stated in the question.

    Every field is required (null when not applicable) because Groq's strict
    JSON-schema mode needs that, and strict mode guarantees a valid reply even
    for odd questions (the default tool-calling mode can fail with HTTP 400).
    """

    team_size: TeamSizeCondition | None = Field(description="Only if the question states a team size condition")
    duration: DurationCondition | None = Field(description="Only if the question states a duration condition")
    project_types: list[Literal["ML/AI", "NLP / GenAI", "Data Engineering"]] | None = Field(
        description="Only if the question restricts the project type/category")
    technologies: list[str] | None = Field(
        description="Specific named tools/languages/platforms the projects must use")
    technology_match: Literal["all", "any"] | None = Field(
        description="'all' if every technology is required ('X and Y'), 'any' for 'X or Y'")


# ---------------------------------------------------------------------------
# Deterministic validation and comparison
# ---------------------------------------------------------------------------

def contains_term(text: str, term: str) -> bool:
    """Case-insensitive whole-word check ("spark" must not match "sparkle")."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", text.lower()) is not None


def match_technology_name(term: str, technology_names: list[str]) -> str:
    """Map a technology mentioned by the user onto the names used in the PDF.

    - exact (any case):              "python"    -> "Python"
    - part of a longer name:         "Kafka"     -> "Kafka" (also matches "Apache Kafka")
    - misspelled:                    "snowflak"  -> "Snowflake"
    - unknown (not in the PDF):      kept as is, so it honestly matches no project
    """
    by_lower = {name.lower(): name for name in technology_names}
    if term.lower() in by_lower:
        return by_lower[term.lower()]
    if any(contains_term(name, term) for name in technology_names):
        return term
    close = difflib.get_close_matches(term.lower(), by_lower, n=1, cutoff=TECHNOLOGY_MATCH_CUTOFF)
    return by_lower[close[0]] if close else term


def choose_technology_match(llm_choice: str | None, question: str) -> str:
    """'any' only if the LLM says so AND the question really says "or" / "either" / "any of".

    Otherwise every technology is required ("Python and Snowflake" -> all). This guards
    against the LLM reading "and" as "or", which turns 17 matches into 97.
    """
    says_any = re.search(r"\b(or|either|any of)\b", question, re.IGNORECASE) is not None
    return "any" if llm_choice == "any" and says_any else "all"


def build_metadata_filters(analysis: QueryAnalysis, question: str, technology_names: list[str]) -> dict:
    """Convert the LLM's QueryAnalysis into validated filters for Python comparison.

    Example: "team size atleast 5 and duration more then 3 months" ->
        {"team_size": [">=", 5], "duration_weeks": [">", 13.0], "duration_text": "3 months"}

    Returns {} when the question names specific projects (e.g. "P038"),
    because then it is about those projects, not a search over all of them.
    """
    if PROJECT_ID_PATTERN.search(question):
        return {}

    filters: dict = {}
    if analysis.team_size:
        filters["team_size"] = [analysis.team_size.operator, analysis.team_size.value]

    if analysis.duration:
        amount, unit = analysis.duration.value, analysis.duration.unit
        weeks = amount * WEEKS_PER_MONTH if unit == "months" else amount
        filters["duration_weeks"] = [analysis.duration.operator, round(weeks, 2)]
        filters["duration_text"] = f"{amount:g} {unit}"

    if analysis.project_types:
        filters["types"] = sorted(set(analysis.project_types))

    technologies = sorted({
        match_technology_name(tech, technology_names) for tech in analysis.technologies or [] if tech.strip()
    })
    if technologies:
        filters["technologies"] = technologies
        filters["technology_match"] = choose_technology_match(analysis.technology_match, question)

    return filters


def project_matches_filters(project: dict, filters: dict) -> bool:
    """Check one project's structured metadata against every filter."""
    for field in ("team_size", "duration_weeks"):
        if field in filters:
            op, value = filters[field]
            if not OPERATORS[op](project[field], value):
                return False

    if "types" in filters and project["type"] not in filters["types"]:
        return False

    if "technologies" in filters:
        stack = " | ".join(project["technology_stack"])
        hits = [contains_term(stack, tech) for tech in filters["technologies"]]
        if not (any(hits) if filters["technology_match"] == "any" else all(hits)):
            return False
    return True


def filter_projects_by_metadata(filters: dict, projects: list[dict]) -> list[dict]:
    """Return every project whose structured metadata satisfies all filters."""
    return [project for project in projects if project_matches_filters(project, filters)]


def describe_filters(filters: dict) -> str:
    """Human-readable summary, e.g. "team size >= 5; duration > 3 months"."""
    parts = []
    if "team_size" in filters:
        parts.append(f"team size {filters['team_size'][0]} {filters['team_size'][1]}")
    if "duration_weeks" in filters:
        parts.append(f"duration {filters['duration_weeks'][0]} {filters['duration_text']}")
    if "types" in filters:
        parts.append("type is " + " or ".join(filters["types"]))
    if "technologies" in filters:
        joiner = " or " if filters["technology_match"] == "any" else " and "
        parts.append("uses " + joiner.join(filters["technologies"]))
    return "; ".join(parts)
