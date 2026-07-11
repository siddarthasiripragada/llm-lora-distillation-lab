from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from sidsearch_lora_lab.config import DATASET_VERSION, PROTOCOL_VERSION, SEED
from sidsearch_lora_lab.protocol.rule_engine import apply_rules


CATEGORIES = [
    "exact_entity", "relative_date", "explicit_date_range", "ambiguous_source", "missing_entity",
    "exclusion_filter", "repository_search", "email_search", "document_search", "combined_search",
    "abbreviation_expansion", "confidence_assignment", "clarification_required", "conflicting_rules",
    "boundary_case", "insufficient_information",
]

TEMPLATES = {
    "exact_entity": ['Find documents about "{entity}"', 'Search email for "{entity}" from {owner}'],
    "relative_date": ["Find recent docs about {entity} from last week", "Show email about {entity} yesterday"],
    "explicit_date_range": ["Search repo issues about {entity} from 2026-01-01 to 2026-03-31", "Find pdf reports about {entity} in 2025"],
    "ambiguous_source": ["Find {entity}", "Search for {entity}"],
    "missing_entity": [
        "Show latest {file_type} documents",
        "Search {status} email",
        "Find documents from {relative_time}",
        "Show github issues {relative_time}",
    ],
    "exclusion_filter": ["Find documents about {entity} excluding drafts", "Search github issues about {entity} without closed"],
    "repository_search": ["Find open repo issues about {entity}", "Search github pull request for {entity} owner {owner}"],
    "email_search": ["Search unread email from {owner} about {entity}", "Find inbox thread about {entity}"],
    "document_search": ["Find pptx documents about {entity}", "Search docs for {entity} owner {owner}"],
    "combined_search": ["Search email and github for {entity}", "Find {entity} across everything"],
    "abbreviation_expansion": ["Find pr about {entity}", "Search docs about {entity}"],
    "confidence_assignment": ["Maybe search email for {entity}", "I am not sure, find github issues about {entity}"],
    "clarification_required": ["Look for {entity}", "Find anything related to {entity}"],
    "conflicting_rules": ["Find email and repo updates about {entity} except closed", "Search documents and github for {entity} last month"],
    "boundary_case": ['Find "{entity} v2" in md docs', "Search repo branch about {entity} today"],
    "insufficient_information": [
        "Find it {relative_time}",
        "Search latest {file_type}",
        "Look for it in {status} items",
        "Show recent {source_word}",
    ],
}
ENTITIES = [
    "NorthstarCRM",
    "InvoicePilot",
    "CloudLedger",
    "PolicyVault",
    "SupportPulse",
    "ReleaseRadar",
    "ContractHub",
    "MetricsBridge",
]
OWNERS = ["Alex", "Morgan", "Taylor", "Jordan", "Casey"]
DIFFICULTIES = ["easy", "medium", "hard"]
FILE_TYPES = ["pdf", "docx", "xlsx", "csv", "pptx", "md", "py", "ipynb", "txt"]
RELATIVE_TIMES = ["today", "yesterday", "last week", "last month"]
SOURCE_WORDS = ["documents", "email", "github", "docs"]


def build_examples(count: int = 325, seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    category_index = 0
    duplicate_attempts = 0
    while len(examples) < count:
        category = CATEGORIES[category_index % len(CATEGORIES)]
        template = rng.choice(TEMPLATES[category])
        text = template.format(
            entity=rng.choice(ENTITIES),
            owner=rng.choice(OWNERS),
            file_type=rng.choice(FILE_TYPES),
            status=rng.choice(["open", "closed", "draft", "unread"]),
            relative_time=rng.choice(RELATIVE_TIMES),
            source_word=rng.choice(SOURCE_WORDS),
        )
        if text in seen_inputs:
            duplicate_attempts += 1
            if duplicate_attempts > 200:
                category_index += 1
                duplicate_attempts = 0
            continue
        duplicate_attempts = 0
        seen_inputs.add(text)
        expected = apply_rules(text)
        examples.append(
            {
                "example_id": f"SS-{DATASET_VERSION}-{len(examples)+1:04d}",
                "dataset_version": DATASET_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "category": category,
                "difficulty": rng.choice(DIFFICULTIES),
                "input": text,
                "expected_output": expected,
                "applied_rules": expected["applied_rules"],
                "source": expected["source"],
                "human_review_status": "synthetic_rule_engine_reviewed",
            }
        )
        category_index += 1
    return examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
