# Setup Guide: Reproduce The SidSearch LoRA Experiment

This guide walks through the full experiment from an empty machine to a before/after LoRA benchmark.

The goal is to reproduce this claim:

> On 50 held-out SidSearch examples, the base model scored `0.2542` composite. The same model with a LoRA adapter scored `0.3053` composite while training only `0.2184%` of parameters.

## 0. What You Are Building

The experiment teaches a small model a private structured protocol called SidSearch.

Example request:

```text
Find open GitHub issues about ReleaseRadar from last month
```

Expected behavior:

```json
{
  "intent": "repository_search",
  "entities": ["ReleaseRadar"],
  "source": "github",
  "filters": {
    "start_date": "2026-06-01",
    "end_date": "2026-06-30",
    "owner": null,
    "file_type": null,
    "status": "open"
  },
  "confidence": "high",
  "clarification_required": false
}
```

This is not a chatbot demo. It is a measurable structured-generation experiment.

## 1. Clone And Install Locally

```powershell
git clone https://github.com/siddarthasiripragada/llm-lora-distillation-lab.git
cd llm-lora-distillation-lab
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify imports:

```powershell
python -c "import torch, transformers, peft, datasets, trl; print('imports ok')"
```

## 2. Validate The Codebase

```powershell
python scripts/01_create_protocol.py
python -m pytest -v
```

Expected:

```text
protocol ok: 20 rules
28 passed
```

## 3. Generate Or Refresh Seed Scenarios

```powershell
python scripts/02_generate_seed_scenarios.py
```

This writes:

```text
data/seed_scenarios.jsonl
```

## 4. Generate Teacher Outputs With Ollama

Install Ollama, then pull the teacher model:

```powershell
ollama pull qwen2.5:3b
ollama list
```

Generate teacher outputs:

```powershell
python scripts/03_generate_teacher_outputs.py
```

This writes:

```text
data/distilled_raw.jsonl
```

Validate teacher outputs:

```powershell
python scripts/04_validate_distilled_data.py
```

Expected shape:

```text
accepted=<some number>
rejected=<some number>
```

For the completed run:

```text
accepted=204
rejected=71
```

## 5. Freeze Train, Validation, And Benchmark Splits

```powershell
python scripts/05_freeze_benchmark.py
```

Expected:

```text
train: 150
validation: 25
heldout: 50
duplicates: []
suspicious_train_benchmark_overlap: []
```

Files:

```text
data/train.jsonl
data/validation.jsonl
benchmarks/heldout.jsonl
```

Do not edit `benchmarks/heldout.jsonl` after this step.

## 6. Run CPU Smoke Test

```powershell
python scripts/07_run_cpu_smoke_test.py
```

This proves real LoRA training mechanics:

- base model loads
- LoRA adapters are inserted
- base weights are frozen
- LoRA weights are trainable
- loss is computed
- `loss.backward()` runs
- LoRA gradients are non-zero
- frozen base gradients remain absent

Important files:

```text
results/trainable_parameters.json
results/gradient_verification.json
```

Expected evidence:

```text
trainable_parameters: 1,081,344
trainable_percentage: 0.2184%
lora_gradient_nonzero: true
unexpected_base_trainable_parameters: 0
frozen_base_gradients: 0
```

## 7. Run The Base Benchmark

Run this before full LoRA training:

```powershell
python scripts/06_run_base_benchmark.py
```

This creates the before-training benchmark:

```text
results/base_closed_book_results.json
results/base_open_book_results.json
```

Completed run:

```text
base_closed_book composite_score: 0.2542
base_open_book composite_score: 0.0
```

## 8. Full LoRA Training In Google Colab

Local CPU is enough for validation, but full training should run on a GPU.

Open:

```text
notebooks/train_lora_colab.ipynb
```

In Google Colab:

1. Upload the notebook.
2. Select `Runtime -> Change runtime type`.
3. Choose `GPU`.
4. Confirm the runtime shows a GPU such as Tesla T4.
5. Upload or clone the project files.
6. Install dependencies.
7. Run protocol/data validation cells.
8. Load `Qwen/Qwen2.5-0.5B-Instruct`.
9. Inject LoRA adapters.
10. Verify trainable parameters and gradients.
11. Train for 3 epochs.
12. Save and download the adapter zip.

The completed training run produced:

| Epoch | Training Loss | Validation Loss |
|---:|---:|---:|
| 1 | 2.039536 | 1.175299 |
| 2 | 0.686261 | 0.495133 |
| 3 | 0.327262 | 0.299387 |

The downloaded adapter should contain:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
```

## 9. Copy Adapter Back Locally

Extract the trained adapter into:

```text
adapters/sidsearch-lora
```

Confirm:

```powershell
Get-ChildItem adapters/sidsearch-lora
```

Required files:

```text
adapter_config.json
adapter_model.safetensors
```

## 10. Run The LoRA Benchmark

```powershell
python scripts/09_run_lora_benchmark.py
```

This creates:

```text
results/lora_closed_book_results.json
```

Completed run:

```text
lora_closed_book composite_score: 0.3053
```

## 11. Compare And Report

```powershell
python scripts/10_compare_results.py
python scripts/11_generate_report.py
```

Outputs:

```text
results/comparison.json
results/report.md
```

Final measured result:

```text
base closed-book composite: 0.2542
LoRA closed-book composite: 0.3053
absolute gain: 0.0511
```

## 12. What To Claim

Correct:

> On 50 held-out SidSearch examples, the LoRA-adapted 0.5B model improved composite score from `0.2542` to `0.3053` while training only `0.2184%` of parameters.

Incorrect:

- The model is generally better.
- The model is production ready.
- The model is better than GPT.
- Training-set performance proves generalization.

