"""Minimal client for an OpenAI-compatible /chat/completions endpoint (vLLM, LiteLLM, etc.).

Only what the golden-set runner needs: one blocking request, and a parsed view of the
first choice that keeps finish_reason and any separated reasoning text, because both
matter when you score reasoning models (see README, "empty content").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class EndpointError(RuntimeError):
    """The request did not produce a usable chat completion (transport error, HTTP error,
    or a body that is not a chat completion)."""


@dataclass
class Completion:
    content: str
    finish_reason: str | None
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty_truncation(self) -> bool:
        """True when the token budget ran out before ANY answer text was emitted.

        Reasoning parsers (e.g. vLLM's reasoning-parser for thinking models) buffer the
        visible answer until the reasoning block closes. If max_tokens is exhausted first,
        the server returns finish_reason="length" with empty content. That is a measurement
        failure (budget too small), not a refusal and not a blank answer.
        """
        return self.finish_reason == "length" and not self.content.strip() and not self.tool_calls


def parse_completion(body: dict[str, Any], latency_s: float = 0.0) -> Completion:
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise EndpointError(f"response has no choices: {str(body)[:200]}") from exc
    message = choice.get("message") or {}
    return Completion(
        content=(message.get("content") or "").strip(),
        finish_reason=choice.get("finish_reason"),
        # vLLM uses reasoning_content; some servers/gateways use reasoning.
        reasoning=(message.get("reasoning_content") or message.get("reasoning") or ""),
        tool_calls=list(message.get("tool_calls") or []),
        latency_s=latency_s,
        usage=dict(body.get("usage") or {}),
    )


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        **extra: Any,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **extra,
        }
        t0 = time.monotonic()
        try:
            resp = self._http.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise EndpointError(f"{type(exc).__name__}: {exc}") from exc
        latency = time.monotonic() - t0
        if resp.status_code != 200:
            raise EndpointError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise EndpointError(f"non-JSON response: {resp.text[:200]}") from exc
        return parse_completion(data, latency)
