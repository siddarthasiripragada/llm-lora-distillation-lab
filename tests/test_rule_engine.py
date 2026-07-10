from sidsearch_lora_lab.protocol.parser import parse_protocol
from sidsearch_lora_lab.protocol.rule_engine import apply_rules


def test_protocol_has_twenty_rules():
    rules = parse_protocol(__import__("pathlib").Path("data/protocol.md"))
    assert len(rules) == 20


def test_document_source_routing():
    output = apply_rules("Find pptx documents about AtlasSync")
    assert output["intent"] == "document_search"
    assert output["source"] == "documents"
    assert output["filters"]["file_type"] == "pptx"


def test_email_source_routing():
    output = apply_rules("Search unread email from Mira about BeaconPlan")
    assert output["intent"] == "email_search"
    assert output["source"] == "email"
    assert output["filters"]["owner"] == "Mira"
    assert output["filters"]["status"] == "unread"


def test_repository_source_routing():
    output = apply_rules("Find open repo issues about FluxRouter")
    assert output["intent"] == "repository_search"
    assert output["source"] == "github"
    assert output["filters"]["status"] == "open"


def test_combined_source_routing():
    output = apply_rules("Search email and github for DeltaIndex")
    assert output["intent"] == "combined_search"
    assert output["source"] == "all"


def test_relative_date_resolution():
    output = apply_rules("Show email about EchoBudget yesterday")
    assert output["filters"]["start_date"] == "2026-07-09"
    assert output["filters"]["end_date"] == "2026-07-09"


def test_explicit_year_resolution():
    output = apply_rules("Find pdf reports about HelioDraft in 2025")
    assert output["filters"]["start_date"] == "2025-01-01"
    assert output["filters"]["end_date"] == "2025-12-31"


def test_quoted_phrase_is_single_entity():
    output = apply_rules('Find documents about "AtlasSync v2"')
    assert output["entities"] == ["AtlasSync v2"]

