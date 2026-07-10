from __future__ import annotations

from typing import Any


LORA_NAME_MARKERS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def is_lora_parameter(name: str) -> bool:
    return any(marker in name for marker in LORA_NAME_MARKERS)


def parameter_report(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    unexpected_trainable: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
            if not is_lora_parameter(name):
                unexpected_trainable.append(name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": (trainable / total * 100.0) if total else 0.0,
        "trainable_parameter_names": trainable_names,
        "unexpected_trainable_parameters": unexpected_trainable,
    }


def assert_only_lora_trainable(model: Any) -> dict[str, Any]:
    report = parameter_report(model)
    if report["unexpected_trainable_parameters"]:
        raise RuntimeError(f"Unexpected non-LoRA trainable parameters: {report['unexpected_trainable_parameters']}")
    if report["trainable_parameters"] <= 0:
        raise RuntimeError("No trainable LoRA parameters found.")
    return report


def inject_lora(model: Any, rank: int = 8, alpha: int = 16, dropout: float = 0.05, target_modules: list[str] | None = None) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, config)

