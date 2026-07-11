# I froze 99.78% of a language model—and improved it on a private task

Only **1,081,344 of 495,114,112 parameters** were allowed to change. The remaining **99.78% stayed frozen**.

The experiment used `Qwen/Qwen2.5-0.5B-Instruct` on **SidSearch**, a private task that converts natural-language search requests into structured execution plans across documents, email, GitHub, and multi-source search.

## The setup

- 150 training examples
- 25 validation examples
- 50 held-out benchmark examples
- LoRA rank: 8
- Trainable parameters: 0.2184%

The untouched base model and the LoRA-adapted model were evaluated on the same held-out benchmark.

## The result

| Metric | Base | LoRA | Change |
|---|---:|---:|---:|
| Composite score | 0.2542 | 0.3053 | **+20.1% relative** |
| Intent accuracy | 0.00 | 0.16 | **+0.16** |
| Source accuracy | 0.00 | 0.22 | **+0.22** |
| Entity F1 | 0.06 | 0.24 | **+0.18** |

By training only **0.22% of the model**, the adapter improved the model’s ability to understand the SidSearch task—especially intent detection, source selection, and entity extraction.

## Why this matters

LoRA makes specialization practical. Instead of retraining an entire model, a small adapter can learn a focused behavior while the original model remains unchanged.

> A compact model does not need every parameter retrained to become better at one clearly defined job.

The result is intentionally specific to SidSearch, but it demonstrates the value of parameter-efficient fine-tuning for private, domain-focused tasks.

**Repository:** https://github.com/siddarthasiripragada/llm-lora-distillation-lab

**Interactive project page:** https://siddarthasiripragada.github.io/llm-lora-distillation-lab/release.html
