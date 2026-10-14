"""The promotion gate: absolute thresholds plus regression against a stored baseline.

The gate fails (blocks promotion) when any of these hold:
  * pass rate is below ``min_pass_rate`` (from the golden set's ``gate:`` block or the CLI);
  * any case marked critical did not pass;
  * fewer cases pass than in the baseline, beyond ``tolerance``. Only cases present in
    both runs are compared, so editing the golden set does not fake a regression.

Typical flow: run the suite on the model in production and save the result as the
baseline; run it on the candidate with ``--gate --baseline``; promote only on exit 0;
after promoting, the candidate's result becomes the new baseline. The gate never
rewrites the baseline itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateReport:
    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    newly_failing: list[str] = field(default_factory=list)
    newly_passing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failures": self.failures,
            "warnings": self.warnings,
            "newly_failing": self.newly_failing,
            "newly_passing": self.newly_passing,
        }


def _validate(result: dict[str, Any], label: str) -> None:
    if not isinstance(result, dict) or not isinstance(result.get("cases"), list):
        raise ValueError(f"{label} is not an evalgate result (no 'cases' list)")


def _status_by_id(result: dict[str, Any]) -> dict[str, str]:
    return {c["id"]: c.get("status", "error") for c in result["cases"]}


def check_thresholds(result: dict[str, Any], min_pass_rate: float | None = None) -> list[str]:
    _validate(result, "result")
    cfg = result.get("gate_config") or {}
    failures: list[str] = []
    required = min_pass_rate if min_pass_rate is not None else cfg.get("min_pass_rate")
    summary = result.get("summary") or {}
    if required is not None:
        rate = float(summary.get("pass_rate", 0.0))
        if rate + 1e-9 < float(required):
            failures.append(
                f"pass rate {summary.get('passed')}/{summary.get('total')} ({rate:.0%}) "
                f"is below the required {float(required):.0%}"
            )
    for case in result["cases"]:
        if case.get("critical") and case.get("status") != "pass":
            failures.append(
                f"critical case {case['id']!r} is {case.get('status')}: {case.get('detail', '')}".rstrip(": ")
            )
    return failures


def compare_to_baseline(
    result: dict[str, Any], baseline: dict[str, Any], tolerance: int = 0
) -> GateReport:
    _validate(result, "result")
    _validate(baseline, "baseline")
    report = GateReport(ok=True)

    if (result.get("golden") or {}).get("sha256") != (baseline.get("golden") or {}).get("sha256"):
        report.warnings.append(
            "golden set differs from the baseline's; comparing only cases present in both"
        )

    cur, base = _status_by_id(result), _status_by_id(baseline)
    common = sorted(set(cur) & set(base))
    if not common:
        report.ok = False
        report.failures.append("no cases in common with the baseline; nothing to compare")
        return report

    base_pass = {cid for cid in common if base[cid] == "pass"}
    cur_pass = {cid for cid in common if cur[cid] == "pass"}
    report.newly_failing = sorted(base_pass - cur_pass)
    report.newly_passing = sorted(cur_pass - base_pass)

    if len(cur_pass) < len(base_pass) - tolerance:
        report.ok = False
        report.failures.append(
            f"regression vs baseline: {len(cur_pass)}/{len(common)} pass, baseline "
            f"{len(base_pass)}/{len(common)} (tolerance {tolerance})"
        )
    elif report.newly_failing:
        report.warnings.append(
            f"within tolerance, but newly failing: {', '.join(report.newly_failing)}"
        )

    unmeasured = [cid for cid in report.newly_failing if cur[cid] in ("empty_truncated", "error")]
    if unmeasured:
        report.warnings.append(
            "some newly failing cases were not measured (empty_truncated/error), check the "
            f"token budget or the server before blaming the model: {', '.join(unmeasured)}"
        )
    return report


def evaluate(
    result: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    tolerance: int = 0,
    min_pass_rate: float | None = None,
) -> GateReport:
    failures = check_thresholds(result, min_pass_rate)
    if baseline is not None:
        report = compare_to_baseline(result, baseline, tolerance)
    else:
        report = GateReport(ok=True, warnings=["no baseline given: absolute thresholds only"])
    report.failures = failures + report.failures
    report.ok = not report.failures
    return report
