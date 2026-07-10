from __future__ import annotations

from typing import Any


def generate_text(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 256) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)

