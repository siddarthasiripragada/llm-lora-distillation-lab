from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


REFERENCE_DATE = date(2026, 7, 10)

DOCUMENT_TERMS = {"doc", "docs", "document", "documents", "pdf", "slide", "slides", "spreadsheet", "report"}
EMAIL_TERMS = {"email", "emails", "inbox", "mail", "thread", "sender"}
GITHUB_TERMS = {"github", "repo", "repository", "issue", "issues", "pull request", "pr", "commit", "branch"}
STATUS_TERMS = {
    "open": "open",
    "closed": "closed",
    "merged": "merged",
    "draft": "draft",
    "unread": "unread",
    "read": "read",
}
FILE_TYPES = ["pdf", "docx", "xlsx", "csv", "pptx", "md", "py", "ipynb", "txt"]
STOPWORDS = {
    "find", "show", "search", "look", "for", "about", "me", "the", "a", "an", "in", "from",
    "to", "with", "and", "or", "latest", "recent", "all", "any", "where", "is", "are",
    "today", "yesterday", "week", "month", "last", "it", "items",
}


@dataclass(frozen=True)
class Scenario:
    input: str
    category: str = "general"


def empty_filters() -> dict[str, str | None]:
    return {"start_date": None, "end_date": None, "owner": None, "file_type": None, "status": None}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_quoted(text: str) -> list[str]:
    return re.findall(r'"([^"]+)"', text)


def infer_source(text: str) -> tuple[str, str, list[str]]:
    lower = text.lower()
    hits = {
        "documents": any(term in lower for term in DOCUMENT_TERMS),
        "email": any(term in lower for term in EMAIL_TERMS),
        "github": any(term in lower for term in GITHUB_TERMS),
    }
    active = [source for source, matched in hits.items() if matched]
    if len(active) > 1 or "across everything" in lower or "all sources" in lower:
        return "combined_search", "all", ["SS-012"]
    if active == ["documents"]:
        return "document_search", "documents", ["SS-004"]
    if active == ["email"]:
        return "email_search", "email", ["SS-013"]
    if active == ["github"]:
        return "repository_search", "github", ["SS-012"]
    return "clarification_required", "unknown", ["SS-005"]


def infer_dates(text: str, filters: dict[str, str | None]) -> list[str]:
    lower = text.lower()
    rules: list[str] = []
    iso_range = re.search(r"(\d{4}-\d{2}-\d{2})\s+(?:to|through|-)\s+(\d{4}-\d{2}-\d{2})", lower)
    if iso_range:
        filters["start_date"], filters["end_date"] = iso_range.group(1), iso_range.group(2)
        return ["SS-003"]
    year = re.search(r"\b(20\d{2})\b", lower)
    if year:
        filters["start_date"] = f"{year.group(1)}-01-01"
        filters["end_date"] = f"{year.group(1)}-12-31"
        rules.append("SS-003")
    if "today" in lower:
        filters["start_date"] = filters["end_date"] = REFERENCE_DATE.isoformat()
        rules.append("SS-002")
    elif "yesterday" in lower:
        value = (REFERENCE_DATE - timedelta(days=1)).isoformat()
        filters["start_date"] = filters["end_date"] = value
        rules.append("SS-002")
    elif "last week" in lower:
        filters["start_date"] = (REFERENCE_DATE - timedelta(days=7)).isoformat()
        filters["end_date"] = REFERENCE_DATE.isoformat()
        rules.append("SS-002")
    elif "last month" in lower:
        filters["start_date"] = "2026-06-01"
        filters["end_date"] = "2026-06-30"
        rules.append("SS-002")
    return rules


def infer_filters(text: str, filters: dict[str, str | None]) -> list[str]:
    lower = text.lower()
    rules: list[str] = []
    owner = re.search(r"(?:owner|author|sender|from)\s+([A-Za-z][A-Za-z0-9_-]+)", text)
    if owner:
        filters["owner"] = owner.group(1)
        rules.append("SS-001")
    for file_type in FILE_TYPES:
        if re.search(rf"\b{re.escape(file_type)}\b", lower):
            filters["file_type"] = file_type
            rules.append("SS-017")
            break
    for term, status in STATUS_TERMS.items():
        if re.search(rf"\b{term}\b", lower):
            filters["status"] = status
            rules.append("SS-018")
            break
    if any(token in lower for token in ["exclude", "except", "without", "not "]):
        rules.append("SS-010")
    return rules


def extract_entities(text: str) -> list[str]:
    quoted = extract_quoted(text)
    if quoted:
        return quoted
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    entities = [token for token in tokens if token.lower() not in STOPWORDS and token.lower() not in DOCUMENT_TERMS | EMAIL_TERMS | GITHUB_TERMS]
    return entities[:4]


def rewrite_query(text: str, entities: list[str]) -> str:
    base = " ".join(entities) if entities else normalize(text)
    expansions = {"pr": "pull request", "docs": "documents", "repo": "repository"}
    words = [expansions.get(word.lower(), word) for word in base.split()]
    return normalize(" ".join(words))


def apply_rules(input_text: str) -> dict[str, Any]:
    text = normalize(input_text)
    lower = text.lower()
    filters = empty_filters()
    intent, source, rules = infer_source(text)
    rules.extend(infer_dates(text, filters))
    rules.extend(infer_filters(text, filters))
    entities = extract_entities(text)
    if not entities:
        intent = "clarification_required"
        source = "unknown" if source == "unknown" else source
        rules.append("SS-006")
    if "maybe" in lower or "not sure" in lower:
        rules.append("SS-016")
    if '"' in text:
        rules.append("SS-008")
    if "latest" in lower or "recent" in lower:
        rules.append("SS-009")
    if any(short in lower.split() for short in ["pr", "docs", "repo"]):
        rules.append("SS-007")
    clarification_required = intent == "clarification_required"
    confidence = "low" if clarification_required else ("medium" if "SS-016" in rules or source == "all" else "high")
    if clarification_required:
        question = "Which source and entity should SidSearch use?"
    else:
        question = None
    rules.extend(["SS-015", "SS-019", "SS-020"])
    unique_rules = sorted(set(rules), key=lambda item: int(item.split("-")[1]))
    return {
        "intent": intent,
        "entities": entities,
        "source": source,
        "filters": filters,
        "rewritten_query": rewrite_query(text, entities),
        "confidence": confidence,
        "clarification_required": clarification_required,
        "clarification_question": question,
        "applied_rules": unique_rules,
    }
