I froze 99.78% of a language model and trained only the remaining 0.22%.

The experiment used Qwen2.5-0.5B-Instruct on a private structured task called SidSearch: converting messy search requests into strict JSON execution plans.

The base model stayed frozen. LoRA trained 1,081,344 parameters out of 495,114,112 total, or 0.2184%.

On 50 held-out SidSearch examples:

Base closed-book composite: 0.2542
LoRA closed-book composite: 0.3053

That is the shiny number.

The more interesting part is the caveat.

LoRA improved semantic partial-credit metrics like intent, source, and entity extraction, but it reduced raw JSON validity from 0.96 to 0.80. Exact full-record match stayed at 0.00. The paired bootstrap interval on the composite delta is also wide on n=50.

So the honest finding is not "small model solved structured output."

The honest finding is:

LoRA moved a tiny frozen model toward the private task, but robust structured generation still needs a schema layer, constrained decoding, repair, or stronger supervision.

That is the kind of failure mode I wanted to expose. The repo includes the setup guide, Google Colab training flow, benchmark scripts, and static project page.

GitHub: https://github.com/siddarthasiripragada/llm-lora-distillation-lab
Project page: https://siddarthasiripragada.github.io/llm-lora-distillation-lab/
Setup guide: https://github.com/siddarthasiripragada/llm-lora-distillation-lab/blob/main/docs/setup_guide.md
