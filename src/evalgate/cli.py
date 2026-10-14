"""Command line entry point.

  evalgate run GOLDEN.yaml [--out result.json] [--gate --baseline last.json]
  evalgate gate RESULT.json --baseline last.json      # offline: compare two saved results
  evalgate bench [...]                                 # lm-evaluation-harness wrapper

Exit codes: 0 ok / gate passed, 1 usage or config error, 2 gate FAILED (do not promote),
3 endpoint unusable (preflight failed or every case errored), 4 bench task failures.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx

from . import __version__
from .client import ChatClient
from .gate import GateReport, evaluate
from .runner import GoldenError, load_golden, preflight, run_golden

EXIT_OK, EXIT_USAGE, EXIT_GATE_FAILED, EXIT_ENDPOINT = 0, 1, 2, 3

STATUS_LABEL = {"pass": "PASS ", "fail": "FAIL ", "empty_truncated": "TRUNC", "error": "ERROR"}


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def print_result(result: dict[str, Any], out=None) -> None:
    out = out or sys.stdout
    ep, golden, s = result["endpoint"], result["golden"], result["summary"]
    print(f"evalgate: {golden['name']} ({golden['cases']} cases) on {ep['model']}", file=out)
    width = max(len(c["id"]) for c in result["cases"])
    for c in result["cases"]:
        crit = "*" if c.get("critical") else " "
        line = f"  {STATUS_LABEL.get(c['status'], c['status'])} {crit}{c['id'].ljust(width)}"
        if c["status"] != "pass" and c.get("detail"):
            line += f"  {c['detail']}"
        print(line.rstrip(), file=out)
        for w in c.get("warnings", []):
            print(f"         {' ' * width}  warning: {w}", file=out)
    reasons = " ".join(f"{k}={v}" for k, v in sorted(s["finish_reasons"].items()))
    print(f"score {s['passed']}/{s['total']} ({s['pass_rate']:.0%})   finish_reason: {reasons}"
          f"   (* = critical)", file=out)
    if s["status_counts"].get("empty_truncated"):
        print("note: TRUNC = finish_reason=length with empty content. The token budget ran out "
              "before the answer was emitted; raise max_tokens for those cases.", file=out)


def print_gate(report: GateReport, out=None) -> None:
    out = out or sys.stdout
    for f in report.failures:
        print(f"  gate failure: {f}", file=out)
    for w in report.warnings:
        print(f"  gate warning: {w}", file=out)
    if report.newly_passing:
        print(f"  newly passing: {', '.join(report.newly_passing)}", file=out)
    print("GATE: PASS" if report.ok else "GATE: FAIL - do not promote", file=out)


def _default_out(golden_name: str, model: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{golden_name}-{model}")
    return os.path.join("results", f"{slug}-{stamp}.json")


def cmd_run(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    try:
        golden = load_golden(args.golden)
    except (OSError, GoldenError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not args.model:
        print("error: --model (or EVALGATE_MODEL) is required", file=sys.stderr)
        return EXIT_USAGE
    if args.baseline and not args.gate:
        print("error: --baseline only makes sense with --gate", file=sys.stderr)
        return EXIT_USAGE

    baseline = None
    if args.gate and args.baseline:
        # Fail fast: a gate pointed at a missing baseline must not quietly pass.
        try:
            baseline = _load_json(args.baseline)
        except (OSError, ValueError) as exc:
            print(f"error: cannot read baseline {args.baseline}: {exc}", file=sys.stderr)
            return EXIT_USAGE

    api_key = os.environ.get(args.api_key_env) or None
    with ChatClient(args.base_url, args.model, api_key=api_key, timeout=args.timeout,
                    transport=transport) as client:
        if not args.no_preflight:
            problem = preflight(client)
            if problem:
                print(f"error: endpoint not usable, aborting before the suite: {problem}", file=sys.stderr)
                return EXIT_ENDPOINT
        result = run_golden(golden, client, concurrency=args.concurrency)

    out_path = args.out or _default_out(golden.name, args.model)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    report = evaluate(result, baseline, tolerance=args.tolerance, min_pass_rate=args.min_pass_rate) \
        if args.gate else None
    if report is not None:
        result["gate"] = report.as_dict()
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print_result(result)
    print(f"result written to {out_path}")
    if result["summary"]["status_counts"].get("error", 0) == result["summary"]["total"]:
        print("error: every case errored; this is an endpoint problem, not a verdict", file=sys.stderr)
        return EXIT_ENDPOINT
    if report is None:
        return EXIT_OK
    print_gate(report)
    return EXIT_OK if report.ok else EXIT_GATE_FAILED


def cmd_gate(args: argparse.Namespace) -> int:
    try:
        result = _load_json(args.result)
        baseline = _load_json(args.baseline) if args.baseline else None
        report = evaluate(result, baseline, tolerance=args.tolerance, min_pass_rate=args.min_pass_rate)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print_result(result)
    print_gate(report)
    return EXIT_OK if report.ok else EXIT_GATE_FAILED


def _add_gate_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--baseline", help="baseline result JSON to compare against")
    p.add_argument("--tolerance", type=int, default=0,
                   help="number of cases allowed to drop vs the baseline (default 0)")
    p.add_argument("--min-pass-rate", type=float, default=None,
                   help="override gate.min_pass_rate from the golden set (0-1)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evalgate",
        description="Run a golden-set eval against an OpenAI-compatible endpoint and gate "
                    "model promotion on the result.",
        epilog="exit codes: 0 ok, 1 usage error, 2 gate failed, 3 endpoint unusable",
    )
    ap.add_argument("--version", action="version", version=f"evalgate {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a golden set against an OpenAI-compatible endpoint")
    run.add_argument("golden", help="golden-set YAML file")
    run.add_argument("--base-url", default=os.environ.get("EVALGATE_BASE_URL", "http://localhost:8000/v1"),
                     help="default: $EVALGATE_BASE_URL or http://localhost:8000/v1")
    run.add_argument("--model", default=os.environ.get("EVALGATE_MODEL"), help="default: $EVALGATE_MODEL")
    run.add_argument("--api-key-env", default="EVALGATE_API_KEY",
                     help="NAME of the env var holding the API key (default EVALGATE_API_KEY). "
                          "Keys are never accepted on the command line.")
    run.add_argument("--out", help="where to write the result JSON (default results/<name>-<time>.json)")
    run.add_argument("--gate", action="store_true", help="exit 2 if thresholds fail or the baseline regresses")
    _add_gate_opts(run)
    run.add_argument("--concurrency", type=int, default=2,
                     help="parallel requests (default 2; keep it low against a production server)")
    run.add_argument("--timeout", type=float, default=600.0, help="per-request timeout, seconds")
    run.add_argument("--no-preflight", action="store_true", help="skip the endpoint health request")

    gate = sub.add_parser("gate", help="gate a saved result against a saved baseline (no server needed)")
    gate.add_argument("result", help="result JSON from `evalgate run`")
    _add_gate_opts(gate)

    sub.add_parser("bench", help="lm-evaluation-harness wrapper (see `evalgate bench --help`)",
                   add_help=False)
    return ap


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bench":
        from .bench import main as bench_main
        return bench_main(argv[1:])
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if args.concurrency < 1:
            print("error: --concurrency must be >= 1", file=sys.stderr)
            return EXIT_USAGE
        return cmd_run(args, transport)
    return cmd_gate(args)


if __name__ == "__main__":
    sys.exit(main())
