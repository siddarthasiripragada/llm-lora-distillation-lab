from __future__ import annotations

from pathlib import Path

from sidsearch_lora_lab.config import Paths, environment_manifest, write_json
from sidsearch_lora_lab.data.validator import read_jsonl
from sidsearch_lora_lab.training.formatting import format_plain_text
from sidsearch_lora_lab.training.lora import assert_only_lora_trainable, inject_lora
from sidsearch_lora_lab.training.verify_gradients import verify_gradient_flow


def run_cpu_smoke_test(paths: Paths = Paths(), max_steps: int = 3) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "CPU smoke test requires torch, transformers, datasets, peft, and their dependencies. "
            "Install them with `py -3.11 -m pip install -r requirements.txt` or run the Colab notebook."
        ) from exc

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
    model = inject_lora(base)
    parameter_info = assert_only_lora_trainable(model)
    gradient_info = verify_gradient_flow(model, tokenizer, paths.results / "gradient_verification.json")

    rows = read_jsonl(paths.data / "train.jsonl")[:10]
    texts = [format_plain_text(row) for row in rows]

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        encoded = tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")
        encoded["labels"] = [item[:] for item in encoded["input_ids"]]
        return encoded

    dataset = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    output_dir = paths.adapters / "sidsearch-lora-smoke"
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=max_steps,
        report_to=[],
        no_cuda=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    train_result = trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {key: float(value) if isinstance(value, (int, float)) else value for key, value in train_result.metrics.items()}
    write_json(paths.results / "trainable_parameters.json", parameter_info)
    write_json(paths.results / "train_metrics.json", metrics)
    write_json(paths.results / "environment_manifest.json", environment_manifest({"model_id": model_id}))
    return {"parameter_info": parameter_info, "gradient_info": gradient_info, "train_metrics": metrics, "adapter_dir": str(output_dir)}
