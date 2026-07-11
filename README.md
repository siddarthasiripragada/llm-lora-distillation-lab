# SidSearch LoRA Distillation Lab

A reproducible small-model fine-tuning experiment: freeze a base language model, train only LoRA adapter weights on a private structured task, and measure before/after performance on a held-out benchmark.

The task is intentionally narrow. SidSearch turns natural-language internal search requests into strict JSON execution plans across documents, email, and GitHub. The project is designed to answer one question cleanly:

> Can a 0.5B parameter model learn a private query-planning protocol through sequence-level distillation and supervised LoRA fine-tuning?

## Result

Evaluation uses the same 50 held-out SidSearch examples for the untouched base model and the LoRA-adapted model.

| System | Protocol In Prompt | Adapter | Composite Score | JSON Validity | Intent Accuracy | Source Accuracy | Entity F1 | Full Record Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base closed-book | No | No | 0.2542 | 0.9600 | 0.0000 | 0.0000 | 0.0600 | 0.0000 |
| Base open-book | Yes | No | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| LoRA closed-book | No | Yes | 0.3053 | 0.8000 | 0.1600 | 0.2200 | 0.2408 | 0.0000 |

The LoRA-adapted model improved the closed-book composite score from `0.2542` to `0.3053`, an absolute gain of `0.0511` on the held-out benchmark.

The result is modest but real: the adapter improved several task-specific fields while still failing strict schema compliance and exact full-record match. This is evidence of task specialization, not broad model improvement.

## What Was Trained

Base model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

LoRA configuration:

```text
rank: 8
alpha: 16
dropout: 0.05
target modules: q_proj, k_proj, v_proj, o_proj
```

Parameter accounting:

```text
total parameters:     495,114,112
trainable parameters:   1,081,344
trainable fraction:          0.2184%
```

Gradient verification:

```text
LoRA gradients non-zero:        true
unexpected trainable base params: 0
frozen base gradients:           0
```

The original model weights remained frozen. Backpropagation updated only the LoRA adapter matrices.

## Task

SidSearch is a private query-planning protocol. The model receives a natural-language request:

```text
Search unread email from Morgan about InvoicePilot
```

It must emit one JSON object:

```json
{
  "intent": "email_search",
  "entities": ["InvoicePilot"],
  "source": "email",
  "filters": {
    "start_date": null,
    "end_date": null,
    "owner": "Morgan",
    "file_type": null,
    "status": "unread"
  },
  "rewritten_query": "InvoicePilot",
  "confidence": "high",
  "clarification_required": false,
  "clarification_question": null,
  "applied_rules": ["SS-001", "SS-013", "SS-015", "SS-018", "SS-019", "SS-020"]
}
```

The protocol defines source routing, relative date handling, explicit date filters, ambiguity handling, file-type inference, status inference, confidence assignment, and schema enforcement.

## Experimental Design

```text
SidSearch protocol
        |
        v
Synthetic scenarios
        |
        v
Teacher model via Ollama
        |
        v
Validated sequence-level distillation data
        |
        v
Frozen Qwen2.5-0.5B base model + trainable LoRA adapters
        |
        v
Held-out benchmark
        |
        v
Base vs LoRA comparison
```

Dataset:

```text
training examples:   150
validation examples:  25
held-out benchmark:   50
accepted teacher rows: 204
rejected teacher rows:  71
```

The benchmark is held out from teacher-data generation and model training. Duplicate and overlap checks run before writing train, validation, and benchmark files.

## Distillation

This project uses sequence-level distillation. A stronger teacher model receives the SidSearch protocol and a scenario, then produces a target JSON sequence. The target is parsed, schema-checked, compared against deterministic rule-engine fields, and either accepted or rejected.

This is not logit distillation. No teacher token probabilities are used.

Teacher runtime:

```text
Ollama qwen2.5:3b
temperature: 0
```

Validation rejects teacher outputs with malformed JSON, wrong schema shape, unsupported rule IDs, or deterministic-field mismatches.

## Metrics

The benchmark scorer records raw generations before parsing. Metrics include:

- JSON validity
- schema compliance
- exact intent accuracy
- exact source accuracy
- exact confidence accuracy
- clarification accuracy
- entity F1
- applied-rule F1
- filter-field accuracy
- exact full-record match
- composite score

The composite score weights intent, source, filters, clarification, rule F1, JSON validity, schema compliance, and entity extraction.

## Reproduce

Install:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Validate protocol and data:

```powershell
python scripts/01_create_protocol.py
python scripts/02_generate_seed_scenarios.py
python scripts/04_validate_distilled_data.py
python scripts/05_freeze_benchmark.py
python -m pytest -v
```

Verify LoRA mechanics:

```powershell
python scripts/07_run_cpu_smoke_test.py
```

Run base benchmark:

```powershell
python scripts/06_run_base_benchmark.py
```

Train full LoRA adapter on GPU with:

```text
notebooks/train_lora_colab.ipynb
```

Run adapter benchmark:

```powershell
python scripts/09_run_lora_benchmark.py
python scripts/10_compare_results.py
python scripts/11_generate_report.py
```

## Repository Map

```text
data/protocol.md                         private SidSearch protocol
data/train.jsonl                         supervised training split
data/validation.jsonl                    validation split
benchmarks/heldout.jsonl                 held-out benchmark
src/sidsearch_lora_lab/protocol/         parser and deterministic rule engine
src/sidsearch_lora_lab/distillation/     Ollama teacher client and validation pipeline
src/sidsearch_lora_lab/training/         formatting, LoRA injection, gradient verification
src/sidsearch_lora_lab/evaluation/       benchmark runner and metrics
scripts/                                 executable experiment stages
notebooks/train_lora_colab.ipynb         GPU training notebook
results/comparison.json                  final benchmark comparison
results/report.md                        generated experiment report
index.html                               readable project page
```

## Interpretation

The adapter improved the narrow benchmark composite score, but the model remains far from a production parser. Exact full-record match and schema compliance are still zero in the current run. The experiment demonstrates a real fine-tuning pipeline and measurable task specialization, while also showing that small-model structured generation still needs better formatting constraints, decoding, or post-processing for robust deployment.

## Claim Boundary

Supported claim:

> On 50 held-out SidSearch examples, the LoRA-adapted 0.5B model improved composite task score from `0.2542` to `0.3053` while training only `0.2184%` of parameters.

Unsupported claims:

- The model is generally better.
- The model is production ready.
- The adapter improves tasks outside SidSearch.
- Training-set performance demonstrates generalization.

