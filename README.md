# SidSearch LoRA Distillation Lab

> A reproducible fine-tuning experiment that freezes a small base language model, trains only LoRA adapter weights on a private structured task, and compares the model before and after training on the same held-out benchmark.

SidSearch is a deliberately narrow research task. It asks a model to translate natural-language search requests into strict JSON execution plans for documents, email, GitHub, or multi-source search.

The core question:

> Can a 0.5B parameter model learn a private query-planning protocol through sequence-level distillation and supervised LoRA fine-tuning?

This repository is built as an end-to-end research artifact: protocol design, synthetic data generation, teacher distillation, validation gates, LoRA training, frozen-weight verification, held-out benchmarking, and final reporting.

## Table Of Contents

- [Headline Result](#headline-result)
- [What This Demonstrates](#what-this-demonstrates)
- [Task Definition](#task-definition)
- [System Design](#system-design)
- [Dataset](#dataset)
- [Distillation Method](#distillation-method)
- [LoRA Training Evidence](#lora-training-evidence)
- [Evaluation Metrics](#evaluation-metrics)
- [Reproduction Guide](#reproduction-guide)
- [Repository Structure](#repository-structure)
- [Interpretation](#interpretation)
- [Claim Boundary](#claim-boundary)

## Headline Result

The final evaluation uses the same 50 held-out SidSearch examples for the untouched base model and the LoRA-adapted model.

| System | Protocol In Prompt | Adapter | Composite Score | JSON Validity | Intent Accuracy | Source Accuracy | Entity F1 | Full Record Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base closed-book | No | No | 0.2542 | 0.9600 | 0.0000 | 0.0000 | 0.0600 | 0.0000 |
| Base open-book | Yes | No | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| LoRA closed-book | No | Yes | 0.3053 | 0.8000 | 0.1600 | 0.2200 | 0.2408 | 0.0000 |

Summary:

```text
base closed-book composite: 0.2542
LoRA closed-book composite: 0.3053
absolute gain:              0.0511
relative gain:              ~20.1%
held-out examples:          50
```

The LoRA-adapted model improved the narrow task composite score. The result is not a broad model-quality claim. Exact full-record match and schema compliance remain weak, which is part of the finding.

## What This Demonstrates

This project demonstrates the mechanics of real parameter-efficient fine-tuning:

- A private protocol is defined before training.
- A held-out benchmark is frozen before final evaluation.
- A stronger teacher model generates sequence-level supervision.
- Teacher outputs are validated before use.
- The base model weights are frozen.
- Only LoRA adapter parameters are trainable.
- Backpropagation produces non-zero LoRA gradients.
- The same benchmark is used for before/after comparison.

It is intentionally closer to a small research lab than a product demo.

## Task Definition

SidSearch converts a user search request into one JSON execution plan.

Example input:

```text
Search unread email from Morgan about InvoicePilot
```

Expected output:

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

The protocol covers:

- exact entity preservation
- relative date resolution
- explicit date filters
- source routing
- ambiguous source handling
- missing entity handling
- query expansion
- quoted phrase preservation
- freshness handling
- negation and exclusion filters
- repository, email, document, and multi-source routing
- confidence assignment
- clarification policy
- file-type and status inference
- output schema enforcement

The protocol itself lives in:

```text
data/protocol.md
```

## System Design

```text
SidSearch protocol
        |
        v
Synthetic scenarios
        |
        v
Teacher model through Ollama
        |
        v
Validated sequence-level distillation data
        |
        v
Frozen Qwen2.5-0.5B base model
        +
Trainable LoRA adapters
        |
        v
Held-out SidSearch benchmark
        |
        v
Base vs LoRA comparison
```

The experiment separates four concerns:

| Layer | Purpose |
|---|---|
| Protocol | Defines the target behavior. |
| Data pipeline | Creates, validates, and splits examples. |
| Training pipeline | Injects LoRA, freezes base weights, verifies gradients, trains adapters. |
| Evaluation pipeline | Preserves raw outputs, parses JSON, computes metrics, reports failures. |

## Dataset

Final split:

```text
training examples:      150
validation examples:     25
held-out benchmark:      50
accepted teacher rows:  204
rejected teacher rows:   71
```

The held-out benchmark is not used for training. Leakage checks verify that benchmark examples do not duplicate or strongly overlap with training examples.

Important files:

```text
data/seed_scenarios.jsonl
data/distilled_raw.jsonl
data/distilled_accepted.jsonl
data/rejected_examples.jsonl
data/train.jsonl
data/validation.jsonl
benchmarks/heldout.jsonl
```

The benchmark should be treated as immutable once final training begins.

## Distillation Method

This project uses **sequence-level distillation**.

A teacher model receives:

```text
SidSearch protocol + one scenario
```

and emits:

```text
one JSON target answer
```

Those generated answers become supervised training labels only if they pass validation.

Teacher runtime:

```text
provider: Ollama
model:    qwen2.5:3b
temperature: 0
```

Validation rejects rows when:

- JSON is malformed
- schema keys are missing or extra
- values use unsupported enums
- filters have the wrong shape
- rule IDs are invalid
- deterministic fields disagree with the SidSearch rule engine

This is not logit distillation. No teacher probability distributions or hidden states are transferred.

## LoRA Training Evidence

Base model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

LoRA configuration:

```text
rank:           8
alpha:          16
dropout:        0.05
target modules: q_proj, k_proj, v_proj, o_proj
```

Parameter accounting:

```text
total parameters:        495,114,112
trainable parameters:      1,081,344
trainable percentage:          0.2184%
```

Gradient verification:

```text
LoRA gradients non-zero:           true
unexpected trainable base params:  0
frozen base gradients:             0
```

The base model remains frozen. Backpropagation updates only LoRA matrices.

Colab training summary:

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 2.039536 | 1.175299 |
| 2 | 0.686261 | 0.495133 |
| 3 | 0.327262 | 0.299387 |

The decreasing validation loss shows the adapter learned the supervised SidSearch format. The held-out benchmark determines whether that learning generalizes.

## Evaluation Metrics

The benchmark runner stores raw model output before parsing. Metrics include:

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
- median latency
- composite score

The composite score combines several task-specific metrics. It is useful as a single summary number, but the component metrics are more informative.

Result interpretation:

- LoRA improved composite score, intent accuracy, source accuracy, entity F1, and clarification accuracy.
- LoRA reduced JSON validity compared with the base closed-book run.
- Exact full-record match stayed at zero.
- Schema compliance stayed at zero.

That mixture is important: fine-tuning improved some semantic task behavior, but did not solve robust structured output.

## Reproduction Guide

For a full step-by-step walkthrough, including local setup, Ollama teacher generation, Google Colab training, adapter download, and final benchmarking, see:

```text
docs/setup_guide.md
```

### 1. Install

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Validate The Repo

```powershell
python scripts/01_create_protocol.py
python -m pytest -v
```

### 3. Generate Or Validate Distillation Data

```powershell
ollama pull qwen2.5:3b
python scripts/03_generate_teacher_outputs.py
python scripts/04_validate_distilled_data.py
python scripts/05_freeze_benchmark.py
```

### 4. Verify LoRA Mechanics Locally

```powershell
python scripts/07_run_cpu_smoke_test.py
```

This writes evidence files such as:

```text
results/trainable_parameters.json
results/gradient_verification.json
```

### 5. Run Base Benchmark

```powershell
python scripts/06_run_base_benchmark.py
```

### 6. Train Full Adapter On GPU

Use:

```text
notebooks/train_lora_colab.ipynb
```

Expected adapter files:

```text
adapters/sidsearch-lora/adapter_config.json
adapters/sidsearch-lora/adapter_model.safetensors
```

Adapter weights are not committed to the repository.

### 7. Run LoRA Benchmark And Report

```powershell
python scripts/09_run_lora_benchmark.py
python scripts/10_compare_results.py
python scripts/11_generate_report.py
```

## Repository Structure

```text
configs/                                  model, training, benchmark, distillation configs
data/protocol.md                          SidSearch source-of-truth protocol
data/train.jsonl                          supervised training data
data/validation.jsonl                     validation data
benchmarks/heldout.jsonl                  held-out benchmark
src/sidsearch_lora_lab/config.py          paths, manifests, environment helpers
src/sidsearch_lora_lab/schemas.py         JSON schema validation
src/sidsearch_lora_lab/protocol/          parser and deterministic rule engine
src/sidsearch_lora_lab/data/              generation, validation, leakage checks, splitting
src/sidsearch_lora_lab/distillation/      Ollama client and teacher-output validation
src/sidsearch_lora_lab/training/          formatting, LoRA injection, gradient verification, training
src/sidsearch_lora_lab/inference/         generation helper
src/sidsearch_lora_lab/evaluation/        metrics, benchmark runner, bootstrap, error analysis
src/sidsearch_lora_lab/reporting/         report and chart scaffolding
scripts/                                  executable experiment stages
notebooks/train_lora_colab.ipynb          GPU training notebook
docs/interview_prep.md                    explanation and interview notes
docs/presentation_examples.md             clean fictional examples for demos
docs/setup_guide.md                       full local + Colab reproduction guide
article/                                  article and social post drafts
index.html                                readable static project page
```

## Interpretation

The experiment succeeded in demonstrating a real before/after LoRA fine-tuning loop:

```text
frozen base model
validated synthetic supervision
LoRA-only trainable parameters
non-zero adapter gradients
held-out benchmark comparison
measured composite-score gain
```

It also exposed the hard part of structured generation. The adapter learned some SidSearch semantics, but it did not reliably emit perfect schema-valid JSON. A stronger next iteration would likely add constrained decoding, JSON repair, stricter assistant-token masking, more accepted teacher rows, or a smaller schema-specific post-processor.

## Claim Boundary

Supported claim:

> On 50 held-out SidSearch examples, the LoRA-adapted 0.5B model improved composite task score from `0.2542` to `0.3053` while training only `0.2184%` of parameters.

Unsupported claims:

- The model is generally better.
- The model is better than larger general-purpose models.
- The model is production ready.
- The adapter improves tasks outside SidSearch.
- Training-set performance demonstrates generalization.
