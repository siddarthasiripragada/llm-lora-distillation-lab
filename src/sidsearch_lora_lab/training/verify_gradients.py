from __future__ import annotations

from pathlib import Path
from typing import Any

from sidsearch_lora_lab.config import write_json
from sidsearch_lora_lab.training.lora import is_lora_parameter, parameter_report


def verify_gradient_flow(model: Any, tokenizer: Any, output_path: Path, text: str = "USER: Find docs about AtlasSync\nASSISTANT:") -> dict[str, Any]:
    import torch

    model.train()
    encoded = tokenizer(text, return_tensors="pt")
    labels = encoded["input_ids"].clone()
    output = model(**encoded, labels=labels)
    loss = output.loss
    loss.backward()
    lora_nonzero: list[str] = []
    frozen_with_grad: list[str] = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if parameter.requires_grad and is_lora_parameter(name) and grad is not None and torch.any(grad.detach() != 0):
            lora_nonzero.append(name)
        if not parameter.requires_grad and grad is not None:
            frozen_with_grad.append(name)
    report = parameter_report(model)
    result = {
        **report,
        "loss": float(loss.detach().cpu()),
        "lora_parameters_with_nonzero_gradient": lora_nonzero,
        "frozen_parameters_with_gradient": frozen_with_grad,
        "passed": bool(lora_nonzero) and not frozen_with_grad and not report["unexpected_trainable_parameters"],
    }
    write_json(output_path, result)
    if not result["passed"]:
        raise RuntimeError(f"Gradient verification failed: {result}")
    return result
