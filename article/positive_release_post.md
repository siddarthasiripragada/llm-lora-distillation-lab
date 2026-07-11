# I froze 99.78% of a language model—and it still learned the task

Only **1,081,344 of 495,114,112 parameters** were trained. The remaining **99.78% stayed frozen**.

The experiment used `Qwen/Qwen2.5-0.5B-Instruct` on **SidSearch**, a private task that converts natural-language search requests into JSON execution plans across documents, email, GitHub, and multi-source search.

## The experiment

- 150 training examples
- 25 validation examples
- 50 held-out benchmark examples
- LoRA rank: 8
- Trainable parameters: 0.2184%

Both the untouched base model and the LoRA-adapted model were evaluated on the same held-out benchmark.

## The result

| Metric | Base | LoRA | Improvement |
|---|---:|---:|---:|
| Composite score | 0.2542 | 0.3053 | **+20.1% relative** |
| Intent accuracy | 0.00 | 0.16 | **+0.16** |
| Source accuracy | 0.00 | 0.22 | **+0.22** |
| Entity F1 | 0.06 | 0.24 | **+0.18** |

The adapter improved the model's ability to understand the private task while changing less than one quarter of one percent of its parameters.

## Why this matters

LoRA makes specialization practical. Instead of retraining the entire model, a small adapter learns the new behavior while the original model remains unchanged.

This experiment demonstrates a simple idea:

> A small model does not need every parameter retrained to become more capable at one clearly defined job.

The result is intentionally narrow: it applies to the 50-example SidSearch benchmark, not general model intelligence. But it shows that parameter-efficient fine-tuning can move a compact model meaningfully toward a private task.

**Repository:** https://github.com/siddarthasiripragada/llm-lora-distillation-lab

**Interactive project page:** https://siddarthasiripragada.github.io/llm-lora-distillation-lab/release.html
