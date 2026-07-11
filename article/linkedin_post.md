I froze almost the entire model and trained only a small set of low-rank matrices.

The experiment used Qwen2.5-0.5B-Instruct on a private structured task called SidSearch: converting search requests into strict JSON execution plans.

The base model stayed frozen. LoRA trained 1,081,344 parameters out of 495,114,112 total, or 0.2184%.

On 50 held-out SidSearch examples:

Base closed-book composite: 0.2542
LoRA closed-book composite: 0.3053

The improvement is narrow and task-specific. Exact full-record match is still 0.0, so this is not production-ready structured parsing. But it does show a complete before/after fine-tuning loop: frozen base weights, adapter gradients, validated distillation data, and held-out evaluation.
