from __future__ import annotations

import requests


class OllamaClient:
    def __init__(self, api_url: str = "http://localhost:11434/api/generate", timeout_seconds: int = 120) -> None:
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def generate(self, model: str, prompt: str, options: dict[str, object] | None = None) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options
        response = requests.post(self.api_url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise RuntimeError(f"Ollama response missing 'response': {data}")
        return str(data["response"])

