# Interview Prep: LoRA Fine-Tuning and Distillation Lab

## 1. One-Minute Project Summary

This project demonstrates a complete fine-tuning experiment for a small language model.

The goal is to take a base model, test it before training, fine-tune it on a narrow private task using LoRA, then test it again on unseen benchmark examples to measure whether performance improved.

The private task is called SidSearch. SidSearch converts natural-language search requests into strict JSON search plans.

Example input:

```text
Find open GitHub issues about AtlasSync from last week
```

Expected output:

```json
{
  "intent": "repository_search",
  "entities": ["AtlasSync"],
  "source": "github",
  "filters": {
    "start_date": "2026-07-03",
    "end_date": "2026-07-10",
    "owner": null,
    "file_type": null,
    "status": "open"
  },
  "rewritten_query": "AtlasSync",
  "confidence": "high",
  "clarification_required": false,
  "clarification_question": null,
  "applied_rules": ["SS-002", "SS-012", "SS-018", "SS-015", "SS-019", "SS-020"]
}
```

The important claim is narrow:

> A LoRA-specialized version of a small model may perform better than the untouched base model on the private SidSearch benchmark.

We are not claiming the model is generally smarter, production ready, or better than GPT.

## 2. What Problem Are We Solving?

General language models do not automatically know a company's private protocols.

For example, a company may have internal rules for:

- how to route search queries
- how to choose between email, documents, and GitHub
- how to apply date filters
- how to handle unclear user requests
- how to return machine-readable JSON

The experiment asks:

> Can a small model learn this private structured task through LoRA fine-tuning?

This is a good interview project because it shows more than prompt engineering. It shows dataset design, training mechanics, parameter freezing, backpropagation, evaluation discipline, and honest benchmark reporting.

## 3. Why SidSearch Is a Good Task

SidSearch is a structured-generation task.

That means the model is not just writing free-form text. It must produce a strict JSON object.

This makes evaluation much easier and more objective.

We can score:

- Did it produce valid JSON?
- Did it choose the correct intent?
- Did it choose the correct source?
- Did it extract the right entities?
- Did it apply the right filters?
- Did it request clarification when needed?
- Did it cite valid protocol rules?

This is stronger than saying, "The answer looks good to me."

## 4. Repository Structure

Important files and folders:

```text
data/protocol.md
```

Defines the SidSearch rules. This is the private protocol the model needs to learn.

```text
data/train.jsonl
data/validation.jsonl
benchmarks/heldout.jsonl
```

The dataset split:

- training examples: used for fine-tuning
- validation examples: used during training
- held-out benchmark examples: used only for final evaluation

```text
src/sidsearch_lora_lab/
```

Main Python package.

```text
src/sidsearch_lora_lab/protocol/rule_engine.py
```

Deterministic rule engine that creates expected outputs from SidSearch rules.

```text
src/sidsearch_lora_lab/training/lora.py
```

LoRA injection and trainable-parameter checks.

```text
src/sidsearch_lora_lab/training/verify_gradients.py
```

Checks whether LoRA parameters receive gradients and frozen base-model parameters do not.

```text
src/sidsearch_lora_lab/evaluation/metrics.py
```

Metric calculations for benchmark scoring.

```text
scripts/
```

Runnable pipeline steps.

```text
notebooks/train_lora_colab.ipynb
```

GPU training notebook for the full LoRA run.

## 5. Step-by-Step Pipeline

### Step 1: Define the Private Protocol

Command:

```powershell
python scripts\01_create_protocol.py
```

What it does:

- reads `data/protocol.md`
- parses all SidSearch rules
- confirms the protocol is valid

Why it matters:

The protocol is the private knowledge. If the protocol is unclear, the model cannot learn the task reliably.

Interview explanation:

> I first defined the target behavior formally. The protocol acts as the source of truth for the task. This prevents vague evaluation and lets me generate objective labels.

### Step 2: Generate Seed Scenarios

Command:

```powershell
python scripts\02_generate_seed_scenarios.py
```

What it does:

- creates natural-language SidSearch requests
- assigns each example a category
- uses the rule engine to create expected JSON outputs
- writes `data/seed_scenarios.jsonl`

Why it matters:

Fine-tuning needs examples. Each example needs an input and a correct target output.

Interview explanation:

> I generated examples covering different rule categories such as email search, repository search, date handling, ambiguous requests, and clarification cases. This gives the model a diverse set of task patterns.

### Step 3: Freeze Train, Validation, and Held-Out Benchmark Splits

Command:

```powershell
python scripts\05_freeze_benchmark.py
```

What it does:

- creates `data/train.jsonl`
- creates `data/validation.jsonl`
- creates `benchmarks/heldout.jsonl`
- checks for duplicates
- checks for suspicious overlap between train and benchmark

Expected split:

```text
train: 150
validation: 25
heldout benchmark: 50
```

Why it matters:

The held-out benchmark is the fair test. The model must never train on those examples.

Interview explanation:

> I froze the benchmark before training to avoid data leakage. The final claim must come from examples the model never saw during training.

### Step 4: Run Unit Tests

Command:

```powershell
python -m pytest -v
```

What it does:

- tests schema validation
- tests rule engine behavior
- tests leakage detection
- tests metric calculations
- tests LoRA freezing logic
- tests gradient behavior on a tiny model

Why it matters:

This verifies the experiment code before spending time on model training.

Interview explanation:

> I tested the infrastructure separately from the model. This helps ensure that benchmark scores are not caused by broken evaluation code.

### Step 5: Run CPU LoRA Smoke Test

Command:

```powershell
python scripts\07_run_cpu_smoke_test.py
```

What it does:

- downloads `Qwen/Qwen2.5-0.5B-Instruct`
- loads the tokenizer
- loads the base model
- injects LoRA adapters
- freezes base-model parameters
- confirms only LoRA parameters are trainable
- runs a forward pass
- calculates cross-entropy loss
- runs `loss.backward()`
- confirms LoRA parameters have non-zero gradients
- confirms frozen base parameters do not receive gradients
- runs 3 tiny CPU training steps
- saves a smoke-test adapter

Why it matters:

This proves real training mechanics are happening.

It does not prove final model improvement. It proves the pipeline can train LoRA parameters correctly.

Interview explanation:

> The smoke test verifies the mechanics of fine-tuning. The base model remains frozen, LoRA parameters are trainable, backpropagation runs, and gradients flow into the adapter weights.

What the screenshot shows:

Your screenshot shows:

```text
Loading weights: 100%
Map: 100%
loss: 2.814
grad_norm: 4.78
learning_rate: 0.0002
1/3 training steps
```

That means:

- the model loaded successfully
- data tokenization finished
- training has started
- the trainer completed at least one optimization/logging step
- CPU training is slow, but the process is working

### Step 6: Run Base Model Benchmark

Command:

```powershell
python scripts\06_run_base_benchmark.py
```

What it does:

- evaluates the untouched base model
- uses the 50 held-out benchmark examples
- records raw model outputs
- scores the model objectively

Why it matters:

This is the "before" score.

Interview explanation:

> Before fine-tuning, I benchmarked the original model on the held-out set. This creates the baseline that the LoRA model must beat.

Important:

The base model benchmark should happen before final LoRA training.

### Step 7: Generate Teacher Outputs

Command:

```powershell
python scripts\03_generate_teacher_outputs.py
```

What it does:

- sends SidSearch protocol + scenario to an Ollama teacher model
- receives a structured JSON target answer
- writes raw teacher outputs to `data/distilled_raw.jsonl`

Why it matters:

This is sequence-level distillation.

The teacher model is not transferring logits. It is generating target answers as text/JSON.

Interview explanation:

> I used sequence-level distillation, where a stronger teacher model creates target JSON responses. Those validated responses become supervised fine-tuning examples for the smaller student model.

### Step 8: Validate Teacher Outputs

Command:

```powershell
python scripts\04_validate_distilled_data.py
```

What it does:

- parses teacher JSON
- validates schema
- checks rule IDs
- checks deterministic fields against the rule engine
- writes accepted examples
- writes rejected examples

Why it matters:

Teacher models can make mistakes. We should not blindly train on invalid outputs.

Interview explanation:

> I added a validation gate between teacher generation and student training. This prevents malformed or rule-violating teacher outputs from contaminating the training dataset.

### Step 9: Full LoRA Training in Google Colab

File:

```text
notebooks/train_lora_colab.ipynb
```

What it does:

- uses a GPU runtime
- loads the base model
- injects LoRA adapters
- trains on SidSearch examples
- evaluates on validation examples
- saves adapter files
- exports metrics

Why Colab is needed:

CPU training is very slow. Colab provides a GPU, usually a T4, which makes the full training practical.

Interview explanation:

> I used Colab for the full training run because the local machine is CPU-only. The local machine validates correctness, while Colab provides acceleration for the full experiment.

### Step 10: Copy the Trained Adapter Back Locally

Expected adapter path:

```text
adapters/sidsearch-lora
```

Expected files:

```text
adapter_config.json
adapter_model.safetensors
```

Why it matters:

The adapter is the fine-tuned part. The base model stays the same; the adapter contains the learned task specialization.

Interview explanation:

> The trained LoRA adapter is portable. I can attach it back onto the same base model to run inference and benchmarking.

### Step 11: Run LoRA Benchmark

Command:

```powershell
python scripts\09_run_lora_benchmark.py
```

What it does:

- loads the same base model
- attaches the trained LoRA adapter
- evaluates on the same 50 held-out benchmark examples
- records raw outputs and scores

Why it matters:

This is the "after" score.

Interview explanation:

> I evaluated the fine-tuned adapter model on the exact same held-out benchmark. That makes the before/after comparison fair.

### Step 12: Compare Results and Generate Report

Commands:

```powershell
python scripts\10_compare_results.py
python scripts\11_generate_report.py
```

What it does:

- compares base model results and LoRA model results
- writes score summaries
- creates a report from actual files

Why it matters:

This produces the final evidence.

Interview explanation:

> I only report metrics generated from result files. This avoids invented numbers and keeps the claim tied to reproducible outputs.

## 6. What Is LoRA Technically?

LoRA stands for Low-Rank Adaptation.

In normal fine-tuning, we may update many or all model weights.

With LoRA:

- the original model weights are frozen
- small trainable matrices are inserted into selected layers
- only those small matrices are updated

In this project, LoRA targets attention projection modules:

```text
q_proj
k_proj
v_proj
o_proj
```

The configuration is:

```text
rank: 8
alpha: 16
dropout: 0.05
```

Interview explanation:

> LoRA is parameter-efficient fine-tuning. Instead of updating the entire model, I freeze the base model and train small low-rank adapter matrices. This reduces memory and storage cost while still allowing task specialization.

## 7. What Is Backpropagation Doing Here?

During training:

1. The model receives a prompt.
2. The model predicts output tokens.
3. The training code compares predicted tokens to the target answer.
4. It computes cross-entropy loss.
5. `loss.backward()` computes gradients.
6. The optimizer updates only LoRA parameters.
7. Base model parameters remain unchanged.

Interview explanation:

> Backpropagation calculates which trainable weights contributed to the error. Because the base weights are frozen, only LoRA adapter weights receive optimizer updates.

## 8. What Is Sequence-Level Distillation?

Sequence-level distillation means:

- a teacher model generates complete target answers
- those answers are used as supervised labels
- the student model learns to imitate the teacher's final output

This is different from logit distillation.

Logit distillation would require the teacher's probability distribution over tokens.

This project does not do that.

Interview explanation:

> This is sequence-level distillation, not logit distillation. The teacher provides target JSON sequences, and the student is trained with supervised fine-tuning on those sequences.

## 9. What Counts as Final Success?

Final success requires these real evidence files:

```text
results/base_closed_book_results.json
results/base_open_book_results.json
results/lora_closed_book_results.json
results/comparison.json
results/report.md
results/trainable_parameters.json
results/gradient_verification.json
```

Important evidence:

- base benchmark score
- LoRA benchmark score
- held-out benchmark size
- trainable parameter percentage
- LoRA gradients non-zero
- frozen base gradients absent
- failure cases
- category-level performance

## 10. What We Can Say Before Colab

Before Colab full training, we can say:

> The dataset, benchmark, evaluation code, and LoRA training mechanics are validated.

We cannot yet say:

> The LoRA model improved benchmark performance.

Why?

Because the final trained adapter does not exist yet.

## 11. What We Can Say After Colab and Benchmarking

After full training and benchmarking, we can say:

> On 50 held-out SidSearch examples, the base model scored X. The same base model with a LoRA adapter scored Y.

If Y is higher than X:

> The LoRA-specialized model improved on the narrow SidSearch benchmark.

If Y is not higher:

> The LoRA training did not improve the benchmark, and we report that honestly.

Both outcomes are valid scientifically.

## 12. Interview Talking Points

### If Asked: Why Not Just Prompt the Model?

Answer:

Prompting can help if the protocol fits in context and the model follows it reliably. Fine-tuning is useful when we want the behavior encoded into the model/adapters so it can perform the task without always providing the full protocol.

### If Asked: How Did You Prevent Data Leakage?

Answer:

I froze a separate held-out benchmark before training and ran duplicate/overlap checks between training and benchmark examples.

### If Asked: How Do You Know Training Was Real?

Answer:

The smoke test verifies that LoRA parameters require gradients, base parameters are frozen, `loss.backward()` runs, LoRA gradients are non-zero, and frozen parameters do not receive gradients.

### If Asked: What Exactly Was Fine-Tuned?

Answer:

Only the LoRA adapter matrices attached to selected attention projection layers. The base model parameters remained frozen.

### If Asked: What Is the Main Metric?

Answer:

The final result is based on the 50 held-out benchmark examples. Metrics include JSON validity, schema compliance, exact intent accuracy, source accuracy, filter accuracy, entity F1, applied-rule F1, and full-record match.

### If Asked: What Is the Risk?

Answer:

The task is narrow and synthetic. Improvement on SidSearch does not imply general model improvement. Also, synthetic teacher data may contain errors, so validation is important.

### If Asked: Why Use a Small Model?

Answer:

A small model makes the experiment cheaper and easier to inspect. The goal is not to build the best possible assistant; the goal is to demonstrate a reproducible fine-tuning pipeline.

### If Asked: Why Use Colab?

Answer:

Local CPU is enough for validation but too slow for full training. Colab provides a GPU for practical LoRA training.

## 13. Clean Final Presentation Flow

Use this structure:

1. Problem: base model does not know private SidSearch protocol.
2. Task: convert natural-language search requests into strict JSON plans.
3. Dataset: 150 train, 25 validation, 50 held-out benchmark.
4. Baseline: test untouched base model first.
5. Training: freeze base model, train LoRA adapters only.
6. Distillation: teacher model creates validated target JSON examples.
7. Evaluation: test base and LoRA model on same held-out benchmark.
8. Results: show before/after scores.
9. Evidence: show trainable parameter percentage and gradient verification.
10. Limitations: narrow benchmark only, no broad superiority claim.

## 14. Best Short Answer for an Interview

> I built a reproducible LoRA fine-tuning experiment around a private structured-generation task called SidSearch. First I defined a protocol and generated train, validation, and held-out benchmark splits. Then I verified the evaluation code and LoRA mechanics, including frozen base weights and non-zero adapter gradients. The full training uses sequence-level distillation from a stronger teacher model and LoRA fine-tuning of Qwen2.5-0.5B. The final claim is based only on comparing the untouched base model and the LoRA-adapted model on the same 50 held-out benchmark examples.

