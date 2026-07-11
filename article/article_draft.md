# I Froze 99%+ of a Language Model and Trained the Rest

This experiment tested whether a small model could learn a private structured search protocol through sequence-level distillation and LoRA fine-tuning.

The base model was `Qwen/Qwen2.5-0.5B-Instruct`. The task was SidSearch: convert natural-language internal search requests into strict JSON plans for documents, email, GitHub, or combined search.

The important engineering constraint was that the base model weights stayed frozen. LoRA adapters were attached to attention projection layers, and only those adapter weights were trainable.

Parameter counts:

```text
total parameters:     495,114,112
trainable parameters:   1,081,344
trainable fraction:          0.2184%
```

Gradient verification passed:

```text
LoRA gradients non-zero:          true
unexpected trainable base params: 0
frozen base gradients:            0
```

Training data came from sequence-level distillation. A stronger Ollama teacher model generated JSON target answers from the SidSearch protocol and scenario. The pipeline accepted 204 teacher rows and rejected 71 malformed or rule-breaking rows.

The final benchmark used 50 held-out examples that were not used for training.

Result:

```text
base closed-book composite: 0.2542
LoRA closed-book composite: 0.3053
absolute gain:              0.0511
paired bootstrap 95% CI:    [-0.0146, 0.1207]
```

The result is useful but limited. The LoRA model improved task-specific fields such as intent, source, entity extraction, and clarification handling, but JSON validity fell from 0.9600 to 0.8000. It did not achieve exact full-record match, and schema compliance remained weak. This is evidence of narrow task movement, not production-ready structured parsing.
