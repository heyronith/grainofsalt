"""OpenRouter client with cost tracking, provider pinning, and secret safety."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from grainofsalt.logging_utils import ROOT


@dataclass
class CostTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)
    cost_cap_usd: float = 10.0

    def record(
        self,
        *,
        batch_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        price_prompt: float,
        price_completion: float,
        model: str,
        provider: str | None,
    ) -> None:
        cost = prompt_tokens * price_prompt + completion_tokens * price_completion
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost
        self.calls.append(
            {
                "batch_id": batch_id,
                "model": model,
                "provider": provider,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
            }
        )
        if self.cost_usd > self.cost_cap_usd:
            raise RuntimeError(
                f"OpenRouter spend ${self.cost_usd:.4f} exceeded cap ${self.cost_cap_usd:.2f}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "n_calls": len(self.calls),
            "calls": self.calls,
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.summary(), indent=2))


class OpenRouterClient:
    def __init__(
        self,
        *,
        model: str,
        provider: str,
        temperature: float = 0.9,
        price_prompt: float = 0.000000036,
        price_completion: float = 0.0000005,
        cost_cap_usd: float = 10.0,
        reasoning: bool = False,
    ) -> None:
        load_dotenv(ROOT / ".env")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY missing from .env")
        # Never store/print raw key in logs; keep only for Authorization header.
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/heyronith/grainofsalt",
            "X-Title": "grainofsalt-brief1",
        }
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.price_prompt = price_prompt
        self.price_completion = price_completion
        self.reasoning = reasoning
        self.tracker = CostTracker(cost_cap_usd=cost_cap_usd)
        self._client = httpx.Client(timeout=120.0)

    def chat_json(
        self,
        *,
        messages: list[dict[str, str]],
        batch_id: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        seed: int | None = None,
        retries: int = 3,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (parsed_json, meta). Pins provider; reasoning off."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
            "response_format": {"type": "json_object"},
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "grainofsalt",
                    "strict": True,
                    "schema": schema,
                },
            }
        if seed is not None:
            body["seed"] = seed
        if not self.reasoning:
            body["reasoning"] = {"effort": "none"}

        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self._client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=self._headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                prompt_tok = int(usage.get("prompt_tokens") or 0)
                completion_tok = int(usage.get("completion_tokens") or 0)
                # Prefer response-reported provider if present
                provider_used = (
                    (data.get("provider") if isinstance(data.get("provider"), str) else None)
                    or self.provider
                )
                self.tracker.record(
                    batch_id=batch_id,
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                    price_prompt=self.price_prompt,
                    price_completion=self.price_completion,
                    model=self.model,
                    provider=provider_used,
                )
                content = data["choices"][0]["message"]["content"]
                # Strip optional markdown fences
                text = content.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1]
                    if text.endswith("```"):
                        text = text[: -3]
                    text = text.strip()
                parsed = json.loads(text)
                meta = {
                    "provider": provider_used,
                    "model": data.get("model", self.model),
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": completion_tok,
                    "id": data.get("id"),
                }
                return parsed, meta
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")

    def close(self) -> None:
        self._client.close()
