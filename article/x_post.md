I froze 99.78% of a language model and trained only the remaining 0.22%.

Held-out SidSearch benchmark, n=50:
Base: 0.2542 composite
LoRA: 0.3053 composite

But the caveat matters:
JSON validity fell 0.96 -> 0.80.
Full-record match stayed 0.00.

Real finding: LoRA improved partial task semantics, not robust structured output.

Repo: https://github.com/siddarthasiripragada/llm-lora-distillation-lab
