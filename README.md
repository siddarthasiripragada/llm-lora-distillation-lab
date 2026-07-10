# SidSearch LoRA Distillation Lab

This repository is a reproducible lab for testing whether a small LoRA-specialized model can outperform its untouched base model on a narrow, private, held-out SidSearch benchmark. It does not claim general model superiority or production readiness.

## Experiment Question

Can `Qwen/Qwen2.5-0.5B-Instruct`, with frozen base weights and trainable LoRA adapter matrices, learn a private SidSearch query-planning protocol well enough to improve held-out structured-generation performance?

## Architecture

```text
SidSearch Protocol
        |
        v
Scenario Generator
        |
        v
Teacher Model through Ollama
        |
        v
Validated Distillation Dataset
        |
        v
Frozen Qwen2.5-0.5B Base Model
        +
Trainable LoRA Matrices
        |
        v
Backpropagation Updates Adapter Only
        |
        v
Held-Out Benchmark
        |
        v
Base vs LoRA Evaluation Report
```

## What Is Real Training

The local CPU smoke test loads the student model, injects PEFT LoRA adapters, verifies that only adapter parameters require gradients, runs a backward pass, and saves/reloads adapter artifacts. The full experiment should be run in Colab with a GPU.

## Where Distillation Occurs

This project uses sequence-level knowledge distillation: a stronger teacher receives the SidSearch protocol and a scenario, then emits a JSON target answer. Validated teacher outputs become supervised examples for the student. This is not logit distillation.

## Where Backpropagation Occurs

Backpropagation occurs during supervised LoRA fine-tuning in `src/sidsearch_lora_lab/training/train.py`. Gradients are expected only on LoRA adapter parameters.

## What Is Frozen

The base Qwen model weights are frozen by PEFT LoRA setup. `results/trainable_parameters.json` records total parameters, trainable parameters, trainable percentage, and all trainable parameter names.

## What LoRA Updates

LoRA updates low-rank matrices attached to selected attention projections: `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

## Dataset Design

The SidSearch protocol is defined in `data/protocol.md`. Generated examples include metadata, deterministic expected outputs, categories, difficulty, and human review status. The benchmark is frozen into `benchmarks/heldout.jsonl` before model training.

## Leakage Prevention

`scripts/05_freeze_benchmark.py` performs deterministic splitting, duplicate checks, and a conservative text-overlap leakage heuristic before writing train, validation, and held-out files.

## Local Windows Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python --version
python -c "import torch, transformers, peft, datasets, trl; print('imports ok')"
pip check
```

## CPU Smoke Test

```powershell
python scripts/01_create_protocol.py
python scripts/02_generate_seed_scenarios.py
python scripts/05_freeze_benchmark.py
pytest -v
python scripts/07_run_cpu_smoke_test.py
```

The smoke test is the first local proof of real LoRA backpropagation. It writes `results/trainable_parameters.json` and `results/gradient_verification.json`. The important fields are `trainable_parameters > 0`, `trainable_percentage` well below full fine-tuning, `lora_gradient_nonzero: true`, `unexpected_base_trainable_parameters: 0`, and `frozen_base_gradients: 0`.

## Teacher Data

Start Ollama and pull the practical CPU teacher first:

```powershell
ollama pull qwen2.5:3b
Invoke-RestMethod -Uri http://localhost:11434/api/tags
python scripts/03_generate_teacher_outputs.py
python scripts/04_validate_distilled_data.py
python scripts/05_freeze_benchmark.py
```

`03_generate_teacher_outputs.py` never uses held-out benchmark prompts. `04_validate_distilled_data.py` writes accepted rows to `data/distilled_accepted.jsonl` and rejected rows to `data/rejected_examples.jsonl`. Do not train blindly on raw teacher responses.

## Colab Full Training

Open `notebooks/train_lora_colab.ipynb` and run cells top to bottom:

1. Install pinned dependencies.
2. Clone the GitHub repository.
3. Verify GPU availability.
4. Generate/freeze the dataset.
5. Load `Qwen/Qwen2.5-0.5B-Instruct`.
6. Inject LoRA adapters.
7. Verify frozen/trainable parameters and gradients.
8. Train for the configured epochs.
9. Save and zip the best adapter plus metrics.
10. Download the zip and copy the adapter into `adapters/sidsearch-lora` for local benchmarking.

## Benchmark Methodology

All systems must run against the exact same `benchmarks/heldout.jsonl` file. Raw outputs, parse failures, latency, and metric records are stored in `results/`. Do not edit expected outputs after seeing predictions.

Run the base benchmark before training:

```powershell
python scripts/06_run_base_benchmark.py
```

After the Colab adapter is copied into `adapters/sidsearch-lora`, run:

```powershell
python scripts/09_run_lora_benchmark.py
python scripts/10_compare_results.py
python scripts/11_generate_report.py
```

## Actual Results

No real base-vs-LoRA benchmark results are committed yet. The deterministic rule-engine benchmark is only a pipeline validation baseline and must not be reported as model performance.

Do not publish training-set performance as evidence of improvement. A valid public claim must come from the 50 held-out SidSearch examples.

## Failure Analysis

Failure analysis is generated after benchmark files exist. Reports must include regressions and parsing/schema failures.

## Limitations

The protocol is synthetic and narrow. Local CPU training is a smoke test only. Full training requires external runtime time, preferably a free Colab GPU. Any public claim must be limited to measured SidSearch benchmark results.

## Reproduction Commands

```powershell
python scripts/01_create_protocol.py
python scripts/02_generate_seed_scenarios.py
python scripts/05_freeze_benchmark.py
pytest -v
python scripts/06_run_base_benchmark.py
python scripts/07_run_cpu_smoke_test.py
python scripts/10_compare_results.py
python scripts/11_generate_report.py
```

## Repository Structure

The repository contains configs, source package, protocol/data scripts, benchmark tools, tests, docs, article drafts, Colab notebook, adapter output folder, and result output folder matching the requested lab structure.
