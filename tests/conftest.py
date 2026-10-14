"""Shared fixtures: a fake OpenAI-compatible server built on httpx.MockTransport.

No network, no GPU. Each test decides what the "model" says per prompt.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_GOLDEN = ROOT / "examples" / "golden.example.yaml"

KILL_ANSWER = (
    "Find the process ID first with `pgrep -f python` or `ps aux | grep python`, then send "
    "SIGTERM with `kill <PID>`. Give it a few seconds to clean up. If the process cannot be "
    "stopped that way, escalate to `kill -9 <PID>` (SIGKILL). You can also use `pkill -f "
    "script.py` to match by command line."
)


def completion(
    content: str | None = "",
    finish_reason: str | None = "stop",
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "cmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def tool_call(name: str, arguments: str = "{}") -> dict[str, Any]:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def last_user(body: dict[str, Any]) -> str:
    return [m for m in body["messages"] if m["role"] == "user"][-1]["content"]


def good_model(body: dict[str, Any]) -> dict[str, Any]:
    """A model that answers every case in the example golden set correctly."""
    p = last_user(body)
    table = [
        ("Reply with OK.", completion("", "length", reasoning="The user wants")),  # buffered
        ("data center", completion("The data center moves 960")),
        ("bat and a ball", completion("5")),
        ("3x + 7", completion("-12")),
        ("taller", completion("Alice")),
        ("tungsten", completion("W")),
        ("shortest day", completion("Jupiter")),
        ("Describe rain", completion("Soft falling water.")),
        ("extract the two dates", completion("2024-03-05, 2026-09-30")),
        ("JSON object", completion('```json\n{"name": "sample", "count": 4, "active": true}\n```')),
        ("weather", completion(None, "tool_calls", tool_calls=[tool_call("get_weather", '{"city": "Paris"}')])),
        ("Is 51 prime", completion("No.", reasoning="51 = 3 x 17, so not prime.")),
        ("phishing email", completion("I can't help with that. It could cause serious harm.")),
        ("kill a runaway", completion(KILL_ANSWER)),
        ("first 8 prime", completion("2, 3, 5, 7, 11, 13, 17, 19")),
    ]
    for key, resp in table:
        if key in p:
            return resp
    raise AssertionError(f"fake model has no answer for prompt: {p[:60]}")


class FakeServer:
    """Callable transport handler that records request bodies and headers."""

    def __init__(self, responder: Callable[[dict[str, Any]], Any]):
        self.responder = responder
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(request.headers)
        out = self.responder(body)
        if isinstance(out, httpx.Response):
            return out
        return httpx.Response(200, json=out)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def body_for(self, needle: str) -> dict[str, Any]:
        return next(b for b in self.bodies if needle in last_user(b))


def overriding(base: Callable, overrides: dict[str, Any]) -> Callable:
    """Responder that answers prompts containing a key in `overrides` differently."""
    def responder(body: dict[str, Any]) -> Any:
        p = last_user(body)
        for key, resp in overrides.items():
            if key in p:
                return resp(body) if callable(resp) else resp
        return base(body)
    return responder


@pytest.fixture
def example_golden_path() -> Path:
    return EXAMPLE_GOLDEN
