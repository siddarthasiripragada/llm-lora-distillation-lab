from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


RULE_PATTERN = re.compile(r"^(SS-\d{3})\s+\|\s+([^|]+)\|\s+priority=(\d+)\s+\|\s+(.+)$")


@dataclass(frozen=True)
class ProtocolRule:
    rule_id: str
    category: str
    priority: int
    text: str


def parse_protocol(path: Path) -> list[ProtocolRule]:
    rules: list[ProtocolRule] = []
    in_rules = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "## Rules":
            in_rules = True
            continue
        if in_rules and line.startswith("## "):
            break
        if not in_rules or not line.startswith("SS-"):
            continue
        match = RULE_PATTERN.match(line)
        if not match:
            raise ValueError(f"Malformed protocol rule line: {line}")
        rule_id, category, priority, text = match.groups()
        rules.append(ProtocolRule(rule_id=rule_id, category=category.strip(), priority=int(priority), text=text.strip()))
    if not rules:
        raise ValueError(f"No protocol rules found in {path}")
    ids = [rule.rule_id for rule in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate protocol rule IDs found")
    return rules


def known_rule_ids(path: Path) -> set[str]:
    return {rule.rule_id for rule in parse_protocol(path)}

