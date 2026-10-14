"""`evalgate bench`: a careful wrapper around EleutherAI lm-evaluation-harness.

This is a positioning tool ("where does this model sit on public benchmarks"), not the
regression gate. It builds one `lm_eval` command per task and runs them only if lm_eval
is installed; otherwise (or with --dry-run) it prints them.

Lessons encoded here, each learned the hard way against a reasoning model on vLLM:

* chat/reasoning models go through the chat-completions API, which needs
  --apply_chat_template, and a generous max_gen_toks (2048). Reasoning parsers buffer the
  visible answer, so a short budget returns empty content and scores as wrong.
* GSM8K's headline is exact_match,flexible-extract, not strict-match. strict-match wants
  the "#### 42" format and penalises a verbose but correct reasoning answer; commercial
  chat models are in the same boat, so the comparison stays fair.
* one lm_eval process per task: one failing task (e.g. gated GPQA without a Hugging Face
  token) must not throw away the tasks that completed.
* code tasks (HumanEval, MBPP) execute model-written code and need
  --confirm_run_unsafe_code. Run them somewhere you are comfortable executing that code.
* concurrency defaults to 2. Concurrency 4 on a long-context task (MMLU-Pro) exhausted the
  GPU KV cache and crashed a production vLLM server. Above 2 you get a warning.
* `full` samples 500 items per task (uncapped MMLU-Pro is ~12k questions, more than a day
  at low concurrency). --limit 0 means genuinely uncapped.
* head-to-head with commercial APIs uses the SAME harness and settings, opt-in per
  provider with --compare. Never compare against numbers from vendor model cards.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PANEL = ["ifeval", "gsm8k", "mmlu_pro", "gpqa_diamond_cot_zeroshot", "humaneval", "mbpp"]
QUICK = ["ifeval", "gsm8k", "mmlu_pro"]
SCOPE_LIMITS = {"quick": 100, "full": 500}
SAFE_CONCURRENCY = 2
COMPARE_CONCURRENCY = 4  # hosted APIs are not your GPU; their rate limits are theirs to manage
DEFAULT_MAX_GEN_TOKS = 2048
REQUEST_TIMEOUT_S = 900
CODE_TASK_PREFIXES = ("humaneval", "mbpp")
GATED_TASK_PREFIXES = ("gpqa",)

# Headline metric preference (first match wins). An entry containing a comma is an exact
# key, so gsm8k picks flexible-extract before strict-match. An entry without a comma is a
# metric stem matching e.g. "prompt_level_strict_acc,none".
METRIC_PREF = [
    "prompt_level_strict_acc",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match",
    "acc",
    "acc_norm",
    "pass_at_1",
    "pass@1",
]
_SKIP = ("stderr", "sample_len", "alias")

# provider -> (lm-eval model type, env var holding the key)
COMPARE_BACKENDS = {
    "openai": ("openai-chat-completions", "OPENAI_API_KEY"),
    "anthropic": ("anthropic-chat-completions", "ANTHROPIC_API_KEY"),
}


def pick_metric(metrics: dict[str, Any]) -> tuple[str, float] | None:
    def numeric(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    for pref in METRIC_PREF:
        if "," in pref:
            if numeric(metrics.get(pref)):
                return pref, float(metrics[pref])
            continue
        for k, v in metrics.items():
            if numeric(v) and (k == pref or k.startswith(pref + ",")) and "stderr" not in k:
                return k, float(v)
    for k, v in metrics.items():
        if numeric(v) and not any(s in k for s in _SKIP):
            return k, float(v)
    return None


@dataclass
class Target:
    label: str
    model_type: str
    model_args: str


def chat_completions_url(endpoint: str) -> str:
    """lm-eval's local-chat-completions wants the full .../chat/completions URL."""
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"


def local_target(endpoint: str, model: str, concurrency: int) -> Target:
    args = (
        f"base_url={chat_completions_url(endpoint)},model={model},num_concurrent={concurrency},"
        f"max_retries=3,tokenized_requests=False,timeout={REQUEST_TIMEOUT_S}"
    )
    return Target(label=model, model_type="local-chat-completions", model_args=args)


def compare_target(spec: str, env: dict[str, str]) -> tuple[Target | None, str | None]:
    """'openai:gpt-4o' -> Target. Returns (None, reason) if it cannot run."""
    provider, _, model = spec.partition(":")
    if provider not in COMPARE_BACKENDS or not model:
        return None, f"--compare {spec!r}: expected PROVIDER:MODEL with PROVIDER in {sorted(COMPARE_BACKENDS)}"
    model_type, key_var = COMPARE_BACKENDS[provider]
    if not env.get(key_var):
        return None, f"--compare {spec}: skipped, {key_var} is not set"
    args = f"model={model},num_concurrent={COMPARE_CONCURRENCY},max_retries=3"
    return Target(label=f"{provider}:{model}", model_type=model_type, model_args=args), None


def resolve_limit(scope: str, limit: int | None) -> int | None:
    if limit is not None:
        return None if limit == 0 else limit
    return SCOPE_LIMITS[scope]


def concurrency_warning(concurrency: int) -> str | None:
    if concurrency <= SAFE_CONCURRENCY:
        return None
    return (
        f"WARNING: concurrency {concurrency} > {SAFE_CONCURRENCY}. Long-context tasks (MMLU-Pro) "
        "at concurrency 4 exhausted the GPU KV cache and crashed a production vLLM server. "
        "Point heavy runs at a non-production replica, or keep concurrency at 1-2."
    )


def is_code_task(task: str) -> bool:
    return task.startswith(CODE_TASK_PREFIXES)


def is_gated_task(task: str) -> bool:
    return task.startswith(GATED_TASK_PREFIXES)


def hf_token_available(env: dict[str, str]) -> bool:
    if env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    hf_home = env.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.isfile(os.path.join(hf_home, "token"))


def task_env_overrides(task: str) -> dict[str, str]:
    extra = {"HF_HUB_ENABLE_HF_TRANSFER": "0"}  # avoid needing the optional hf_transfer package
    if is_code_task(task):
        extra["HF_ALLOW_CODE_EVAL"] = "1"  # required by the code_eval metric
    return extra


def build_command(
    lm_eval: str,
    target: Target,
    task: str,
    limit: int | None,
    task_dir: str,
    max_gen_toks: int = DEFAULT_MAX_GEN_TOKS,
) -> list[str]:
    cmd = [
        lm_eval,
        "--model", target.model_type,
        "--model_args", target.model_args,
        "--tasks", task,
        "--gen_kwargs", f"max_gen_toks={max_gen_toks}",
        "--batch_size", "1",
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        "--output_path", task_dir,
        "--log_samples",
    ]
    if is_code_task(task):
        cmd.append("--confirm_run_unsafe_code")
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def format_command(cmd: list[str], env_overrides: dict[str, str]) -> str:
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_overrides.items())
    return (prefix + " " if prefix else "") + shlex.join(cmd)


def read_headlines(task_dir: str) -> dict[str, tuple[str, float]]:
    files = sorted(
        glob.glob(os.path.join(task_dir, "**", "results_*.json"), recursive=True),
        key=os.path.getmtime,
    )
    if not files:
        return {}
    with open(files[-1]) as fh:
        results = json.load(fh).get("results", {})
    out: dict[str, tuple[str, float]] = {}
    for name, metrics in results.items():
        headline = pick_metric(metrics)
        if headline:
            out[name] = headline
    return out


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def run_task(cmd: list[str], env: dict[str, str], task_dir: str) -> str | None:
    """Run one lm_eval invocation; returns an error string or None. Output goes to a log."""
    os.makedirs(task_dir, exist_ok=True)
    log_path = os.path.join(task_dir, "lm_eval.log")
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        return None
    try:
        with open(log_path) as fh:
            tail = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        tail = []
    return f"rc={rc}: {tail[-1][:160] if tail else 'no output'} (log: {log_path})"


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="evalgate bench",
        description="Run (or print) lm-evaluation-harness commands with safe defaults for "
                    "reasoning models served behind an OpenAI-compatible API.",
    )
    ap.add_argument("--endpoint", default=os.environ.get("EVALGATE_BASE_URL", "http://localhost:8000/v1"),
                    help="OpenAI-compatible base URL (default: $EVALGATE_BASE_URL or http://localhost:8000/v1)")
    ap.add_argument("--model", default=os.environ.get("EVALGATE_MODEL"),
                    help="served model name (default: $EVALGATE_MODEL)")
    ap.add_argument("--scope", choices=sorted(SCOPE_LIMITS), default="quick",
                    help="quick = ifeval,gsm8k,mmlu_pro @100/task; full = whole panel @500/task")
    ap.add_argument("--tasks", help="comma-separated lm-eval task names (overrides --scope's task list)")
    ap.add_argument("--limit", type=int, default=None, help="items per task; 0 = uncapped")
    ap.add_argument("--concurrency", type=int, default=SAFE_CONCURRENCY,
                    help=f"concurrent requests to YOUR endpoint (default {SAFE_CONCURRENCY}; warns above)")
    ap.add_argument("--max-gen-toks", type=int, default=DEFAULT_MAX_GEN_TOKS)
    ap.add_argument("--compare", action="append", default=[], metavar="PROVIDER:MODEL",
                    help="also run the same tasks on a hosted model, e.g. openai:gpt-4o. "
                         "Keys come from OPENAI_API_KEY / ANTHROPIC_API_KEY. Prompts leave your machine.")
    ap.add_argument("--out", default=None, help="output directory (default ./bench-runs/<timestamp>)")
    ap.add_argument("--lm-eval", default=None, help="path to the lm_eval executable")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    if not args.model:
        print("error: --model (or EVALGATE_MODEL) is required", file=sys.stderr)
        return 1
    if args.concurrency < 1:
        print("error: --concurrency must be >= 1", file=sys.stderr)
        return 1

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else (
        QUICK if args.scope == "quick" else PANEL)
    limit = resolve_limit(args.scope, args.limit)

    warn = concurrency_warning(args.concurrency)
    if warn:
        print(warn, file=sys.stderr)
    if args.max_gen_toks < DEFAULT_MAX_GEN_TOKS:
        print(f"WARNING: max_gen_toks={args.max_gen_toks} < {DEFAULT_MAX_GEN_TOKS}. Reasoning models "
              "may return empty content at small budgets and score as wrong.", file=sys.stderr)
    gated = [t for t in tasks if is_gated_task(t)]
    if gated and not hf_token_available(env):
        print(f"WARNING: {', '.join(gated)} use a gated Hugging Face dataset. Accept the dataset "
              "licence on Hugging Face and set HF_TOKEN, or these tasks will fail (the others "
              "still run: one process per task).", file=sys.stderr)

    targets = [local_target(args.endpoint, args.model, args.concurrency)]
    for spec in args.compare:
        tgt, reason = compare_target(spec, env)
        if tgt:
            targets.append(tgt)
        else:
            print(reason, file=sys.stderr)

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = args.out or os.path.join("bench-runs", stamp)
    lm_eval = args.lm_eval or shutil.which("lm_eval")
    plan = []
    for tgt in targets:
        for task in tasks:
            task_dir = os.path.join(outdir, _slug(tgt.label), task)
            cmd = build_command(lm_eval or "lm_eval", tgt, task, limit, task_dir, args.max_gen_toks)
            plan.append((tgt, task, task_dir, cmd))

    print(f"tasks={','.join(tasks)} limit={limit or 'uncapped'} concurrency={args.concurrency} "
          f"targets={[t.label for t in targets]}")
    if args.dry_run or not lm_eval:
        if not lm_eval:
            print("lm_eval not found on PATH; printing commands only. "
                  "Install with: pip install 'llm-eval-gate[bench]'", file=sys.stderr)
        for _, task, _, cmd in plan:
            print(format_command(cmd, task_env_overrides(task)))
        return 0

    scores: dict[str, dict[str, Any]] = {t.label: {} for t in targets}
    errors: dict[str, list[str]] = {t.label: [] for t in targets}
    for tgt, task, task_dir, cmd in plan:
        print(f"=== {tgt.label} :: {task}", flush=True)
        task_env = {**env, **task_env_overrides(task)}
        err = run_task(cmd, task_env, task_dir)
        if err:
            errors[tgt.label].append(f"{task}: {err}")
            print(f"    FAILED {err}", flush=True)
            continue
        headlines = read_headlines(task_dir)
        if not headlines:
            errors[tgt.label].append(f"{task}: no results json")
            continue
        scores[tgt.label].update({k: list(v) for k, v in headlines.items()})
        head = headlines.get(task)
        print(f"    {task}: {head[0]} = {_pct(head[1])}" if head else f"    done ({len(headlines)} entries)")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    summary = {"tasks": tasks, "limit": limit, "scores": scores, "errors": errors}
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    labels = [t.label for t in targets]
    print("\n" + "task".ljust(28) + "".join(lbl[:22].ljust(24) for lbl in labels))
    for task in tasks:
        row = task.ljust(28)
        for lbl in labels:
            v = scores[lbl].get(task)
            row += _pct(v[1] if v else None).ljust(24)
        print(row)
    for lbl, errs in errors.items():
        for e in errs:
            print(f"! {lbl}: {e}")
    print(f"\nsummary: {os.path.join(outdir, 'summary.json')}")
    print("Small samples are noisy (about +/-5-10% at 100 items) and every model may have seen "
          "some test items in training. Differences on the same harness are the signal.")
    return 0 if not any(errors.values()) else 4


if __name__ == "__main__":
    sys.exit(main())
