"""Load a YAML golden set, run it against an endpoint, and produce a JSON-able result.

Each case ends in exactly one status:

  pass             every check passed
  fail             the model answered and at least one check failed
  empty_truncated  finish_reason == "length" with EMPTY content: the token budget ran out
                   before any answer was emitted (reasoning parsers buffer output). This
                   is a budget problem in the eval, not a refusal or a blank answer, so the
                   checks are not run and the report tells you to raise max_tokens.
  error            the request failed, or finish_reason was "abort" / "error"

Only "pass" counts towards the score. A gate therefore cannot pass on cases it could not
measure, but the report keeps "the model got it wrong" separate from "we did not get a
usable measurement".

vLLM does not expose a request-failure counter in every build, so errors are derived from
finish_reason (stop | length | abort | error), which is the field the server always sets.
"""
from __future__ import annotations

import concurrent.futures
import datetime as _dt
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .checks import CheckResult, CheckSpecError, ScoringConfig, run_check, validate_check
from .client import ChatClient, Completion, EndpointError

ERROR_FINISH_REASONS = {"abort", "error"}
DEFAULT_MAX_TOKENS = 1024
PREVIEW_CHARS = 160


class GoldenError(ValueError):
    pass


@dataclass
class Case:
    id: str
    messages: list[dict[str, Any]]
    checks: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    critical: bool = False
    deterministic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)  # tools, tool_choice, params


@dataclass
class Golden:
    name: str
    path: str
    sha256: str
    cases: list[Case]
    scoring: ScoringConfig
    gate: dict[str, Any]


def load_golden(path: str | Path) -> Golden:
    p = Path(path)
    raw_bytes = p.read_bytes()
    try:
        doc = yaml.safe_load(raw_bytes) or {}
    except yaml.YAMLError as exc:
        raise GoldenError(f"{p}: invalid YAML: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("cases"), list) or not doc["cases"]:
        raise GoldenError(f"{p}: expected a mapping with a non-empty 'cases' list")

    defaults = doc.get("defaults") or {}
    scoring_raw = doc.get("scoring") or {}
    scoring = ScoringConfig(
        refusal_markers=tuple(scoring_raw.get("refusal_markers", ScoringConfig.refusal_markers)),
        refusal_scan_chars=int(scoring_raw.get("refusal_scan_chars", ScoringConfig.refusal_scan_chars)),
        min_answer_chars=int(scoring_raw.get("min_answer_chars", ScoringConfig.min_answer_chars)),
    )
    gate = dict(doc.get("gate") or {})
    critical_ids = set(gate.get("critical") or [])

    cases: list[Case] = []
    seen: set[str] = set()
    for i, c in enumerate(doc["cases"]):
        if not isinstance(c, dict) or not c.get("id"):
            raise GoldenError(f"{p}: case #{i} needs an 'id'")
        cid = str(c["id"])
        if cid in seen:
            raise GoldenError(f"{p}: duplicate case id {cid!r}")
        seen.add(cid)

        if "messages" in c:
            messages = list(c["messages"])
        elif "prompt" in c:
            messages = [{"role": "user", "content": str(c["prompt"])}]
        else:
            raise GoldenError(f"{p}: case {cid!r} needs 'prompt' or 'messages'")
        if c.get("system"):
            messages = [{"role": "system", "content": str(c["system"])}] + messages

        checks = c.get("checks") or []
        if not checks and not c.get("deterministic"):
            raise GoldenError(f"{p}: case {cid!r} has no checks")
        for spec in checks:
            try:
                validate_check(spec)
            except CheckSpecError as exc:
                raise GoldenError(f"{p}: case {cid!r}: {exc}") from exc

        extra: dict[str, Any] = dict(c.get("params") or {})
        for key in ("tools", "tool_choice"):
            if key in c:
                extra[key] = c[key]

        cases.append(Case(
            id=cid,
            messages=messages,
            checks=list(checks),
            max_tokens=int(c.get("max_tokens", defaults.get("max_tokens", DEFAULT_MAX_TOKENS))),
            temperature=float(c.get("temperature", defaults.get("temperature", 0.0))),
            critical=bool(c.get("critical", False)) or cid in critical_ids,
            deterministic=bool(c.get("deterministic", False)),
            extra=extra,
        ))

    unknown_critical = critical_ids - seen
    if unknown_critical:
        raise GoldenError(f"{p}: gate.critical names unknown case ids {sorted(unknown_critical)}")
    gate["critical"] = [c.id for c in cases if c.critical]

    return Golden(
        name=str(doc.get("name") or p.stem),
        path=str(p),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        cases=cases,
        scoring=scoring,
        gate=gate,
    )


def _preview(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 3] + "..."


def run_case(case: Case, client: ChatClient, scoring: ScoringConfig) -> tuple[dict[str, Any], list[str]]:
    """Run one case. Returns (case_result, finish_reasons_seen)."""
    out: dict[str, Any] = {
        "id": case.id,
        "critical": case.critical,
        "max_tokens": case.max_tokens,
        "status": "error",
        "passed": False,
        "checks": [],
        "warnings": [],
    }
    reasons: list[str] = []
    calls = 2 if case.deterministic else 1
    completions: list[Completion] = []
    for _ in range(calls):
        try:
            comp = client.chat(case.messages, max_tokens=case.max_tokens,
                               temperature=case.temperature, **case.extra)
        except EndpointError as exc:
            reasons.append("transport_error")
            out["detail"] = str(exc)
            return out, reasons
        reasons.append(comp.finish_reason or "none")
        completions.append(comp)

    first = completions[0]
    out.update({
        "finish_reason": first.finish_reason,
        "content_chars": len(first.content),
        "reasoning_chars": len(first.reasoning),
        "content_preview": _preview(first.content),
        "latency_s": round(sum(c.latency_s for c in completions), 3),
    })

    bad = [c.finish_reason for c in completions if c.finish_reason in ERROR_FINISH_REASONS]
    if bad:
        out["detail"] = f"server finished with finish_reason={bad[0]}"
        return out, reasons

    if any(c.is_empty_truncation for c in completions):
        out["status"] = "empty_truncated"
        out["detail"] = (
            f"finish_reason=length with empty content at max_tokens={case.max_tokens}: the "
            "budget ran out before the model emitted an answer (reasoning output is buffered). "
            "Raise max_tokens for this case; this is not a refusal."
        )
        return out, reasons

    if first.finish_reason == "length":
        out["warnings"].append(
            f"truncated at max_tokens={case.max_tokens}; scored on partial content"
        )

    results: list[CheckResult] = [run_check(spec, first, scoring) for spec in case.checks]
    if case.deterministic:
        digests = [hashlib.sha256(c.content.encode()).hexdigest()[:12] for c in completions]
        same = len(set(digests)) == 1
        results.append(CheckResult(
            "deterministic", same, "==".join(digests) if same else " != ".join(digests) + " DRIFT"
        ))

    out["checks"] = [r.as_dict() for r in results]
    out["passed"] = all(r.passed for r in results)
    out["status"] = "pass" if out["passed"] else "fail"
    failed = [f"{r.name}: {r.detail}" for r in results if not r.passed]
    if failed:
        out["detail"] = "; ".join(failed)
    return out, reasons


def run_golden(golden: Golden, client: ChatClient, concurrency: int = 2) -> dict[str, Any]:
    started = _dt.datetime.now(_dt.timezone.utc)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        outcomes = list(pool.map(lambda c: run_case(c, client, golden.scoring), golden.cases))
    cases = [o for o, _ in outcomes]
    finish = Counter(r for _, rs in outcomes for r in rs)
    statuses = Counter(c["status"] for c in cases)
    passed = statuses.get("pass", 0)
    total = len(cases)
    return {
        "schema": 1,
        "tool": f"evalgate {__version__}",
        "started_at": started.isoformat(timespec="seconds"),
        "duration_s": round((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(), 1),
        "golden": {"name": golden.name, "sha256": golden.sha256, "cases": total},
        "endpoint": {"base_url": client.base_url, "model": client.model},
        "gate_config": golden.gate,
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "status_counts": dict(statuses),
            "finish_reasons": dict(finish),
            "errors": finish.get("abort", 0) + finish.get("error", 0) + finish.get("transport_error", 0),
        },
        "cases": cases,
    }


def preflight(client: ChatClient) -> str | None:
    """One cheap request before the suite. Returns an error message, or None if usable.

    Running a gate against a sick server produces scores that look like a regression but
    are really infrastructure noise. Abort instead. Note: a reasoning model may answer this
    with empty content and finish_reason=length; that is fine here, we only need a
    well-formed completion back.
    """
    try:
        comp = client.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=16)
    except EndpointError as exc:
        return str(exc)
    if comp.finish_reason in ERROR_FINISH_REASONS:
        return f"preflight finished with finish_reason={comp.finish_reason}"
    return None
