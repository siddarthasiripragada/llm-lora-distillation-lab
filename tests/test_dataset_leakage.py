from sidsearch_lora_lab.data.generator import build_examples
from sidsearch_lora_lab.data.leakage import duplicate_inputs, suspicious_overlap
from sidsearch_lora_lab.data.splitter import deterministic_split
from sidsearch_lora_lab.data.validator import validate_dataset_rows


def test_generated_examples_have_no_duplicate_inputs():
    assert duplicate_inputs(build_examples()) == []


def test_deterministic_split_sizes():
    splits = deterministic_split(build_examples())
    assert len(splits["train"]) == 150
    assert len(splits["validation"]) == 25
    assert len(splits["heldout"]) == 50


def test_leakage_heuristic_flags_identical_text():
    rows = build_examples(20)
    assert suspicious_overlap([rows[0]], [rows[0]], threshold=1.0)


def test_generated_expected_outputs_validate():
    errors = validate_dataset_rows(build_examples(40), __import__("pathlib").Path("data/protocol.md"))
    assert errors == []
