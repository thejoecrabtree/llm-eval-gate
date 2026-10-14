import json
import os
import stat
import sys
import textwrap

import pytest

from evalgate import bench
from evalgate.cli import main as cli_main


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    for var in ("EVALGATE_BASE_URL", "EVALGATE_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))  # no cached token


# ------------------------------------------------------------- command construction

def test_command_encodes_chat_template_and_generation_budget(tmp_path):
    tgt = bench.local_target("http://localhost:8000/v1", "my-model", 2)
    cmd = bench.build_command("lm_eval", tgt, "gsm8k", 100, str(tmp_path / "gsm8k"))
    assert "--apply_chat_template" in cmd
    assert "--fewshot_as_multiturn" in cmd
    assert cmd[cmd.index("--gen_kwargs") + 1] == "max_gen_toks=2048"
    assert cmd[cmd.index("--tasks") + 1] == "gsm8k"  # one task per process
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    assert cmd[cmd.index("--limit") + 1] == "100"
    assert "--confirm_run_unsafe_code" not in cmd


def test_model_args_use_full_chat_url_and_concurrency():
    args = bench.local_target("http://host:8000/v1/", "m", 2).model_args
    assert "base_url=http://host:8000/v1/chat/completions," in args
    assert "num_concurrent=2" in args and "tokenized_requests=False" in args
    already = bench.local_target("http://host:8000/v1/chat/completions", "m", 1).model_args
    assert "chat/completions/chat" not in already


def test_code_tasks_need_unsafe_code_flag_and_env(tmp_path):
    tgt = bench.local_target("http://x/v1", "m", 1)
    for task in ("humaneval", "mbpp"):
        cmd = bench.build_command("lm_eval", tgt, task, None, str(tmp_path))
        assert "--confirm_run_unsafe_code" in cmd
        assert "--limit" not in cmd
        assert bench.task_env_overrides(task)["HF_ALLOW_CODE_EVAL"] == "1"
    assert "HF_ALLOW_CODE_EVAL" not in bench.task_env_overrides("ifeval")


@pytest.mark.parametrize("scope, limit, expected", [
    ("quick", None, 100), ("full", None, 500), ("full", 0, None), ("quick", 25, 25)])
def test_limits(scope, limit, expected):
    assert bench.resolve_limit(scope, limit) == expected


def test_concurrency_guard():
    assert bench.concurrency_warning(1) is None
    assert bench.concurrency_warning(2) is None
    warning = bench.concurrency_warning(4)
    assert warning and "KV cache" in warning


# ------------------------------------------------------------- headline metric

def test_gsm8k_headline_is_flexible_extract():
    metrics = {"alias": "gsm8k", "exact_match,strict-match": 0.41, "exact_match_stderr,strict-match": 0.02,
               "exact_match,flexible-extract": 0.93, "exact_match_stderr,flexible-extract": 0.01}
    assert bench.pick_metric(metrics) == ("exact_match,flexible-extract", 0.93)


def test_other_headlines():
    ifeval = {"prompt_level_strict_acc,none": 0.8, "prompt_level_strict_acc_stderr,none": 0.01,
              "inst_level_loose_acc,none": 0.9}
    assert bench.pick_metric(ifeval) == ("prompt_level_strict_acc,none", 0.8)
    mmlu_pro = {"exact_match,custom-extract": 0.66, "exact_match_stderr,custom-extract": 0.02}
    assert bench.pick_metric(mmlu_pro) == ("exact_match,custom-extract", 0.66)
    humaneval = {"pass@1,create_test": 0.85, "pass@1_stderr,create_test": 0.03}
    assert bench.pick_metric(humaneval) == ("pass@1,create_test", 0.85)
    assert bench.pick_metric({"alias": "x", "weird_stderr,none": 0.1}) is None


# ------------------------------------------------------------- comparison targets

def test_compare_target_needs_its_key():
    tgt, reason = bench.compare_target("openai:gpt-4o", {})
    assert tgt is None and "OPENAI_API_KEY" in reason
    tgt, reason = bench.compare_target("openai:gpt-4o", {"OPENAI_API_KEY": "test-placeholder"})
    assert reason is None and tgt.model_type == "openai-chat-completions"
    assert "test-placeholder" not in tgt.model_args  # keys travel via env, never argv
    tgt, reason = bench.compare_target("nope", {})
    assert tgt is None and "PROVIDER:MODEL" in reason


# ------------------------------------------------------------- main(): dry run / no lm_eval

def test_dry_run_prints_one_command_per_task(capsys, monkeypatch):
    monkeypatch.setattr(bench.shutil, "which", lambda name: None)
    code = bench.main(["--model", "m", "--scope", "full", "--concurrency", "4"])
    out, err = capsys.readouterr()
    assert code == 0
    lines = [ln for ln in out.splitlines() if "--tasks" in ln]
    assert len(lines) == len(bench.PANEL)
    assert all("--apply_chat_template" in ln and "max_gen_toks=2048" in ln for ln in lines)
    assert any(ln.startswith("HF_HUB_ENABLE_HF_TRANSFER=0 HF_ALLOW_CODE_EVAL=1 ") for ln in lines)
    assert "--limit 500" in lines[0]
    assert "lm_eval not found" in err
    assert "WARNING: concurrency 4" in err
    assert "gated Hugging Face dataset" in err  # GPQA, no token configured


def test_cli_dispatches_bench_subcommand(capsys):
    assert cli_main(["bench", "--model", "m", "--tasks", "ifeval", "--dry-run"]) == 0
    assert "--tasks ifeval" in capsys.readouterr().out


def test_model_required():
    assert bench.main(["--dry-run"]) == 1


# ------------------------------------------------------------- main(): real run with a fake lm_eval

FAKE_LM_EVAL_PY = textwrap.dedent("""\
    import json, os, sys
    a = sys.argv
    task = a[a.index("--tasks") + 1]
    out = a[a.index("--output_path") + 1]
    if task.startswith("gpqa"):
        print("DatasetNotFoundError: gated dataset, set HF_TOKEN", file=sys.stderr)
        sys.exit(1)
    metrics = {
        "gsm8k": {"exact_match,strict-match": 0.5, "exact_match,flexible-extract": 0.9},
        "ifeval": {"prompt_level_strict_acc,none": 0.8},
    }[task]
    d = os.path.join(out, "model")
    os.makedirs(d, exist_ok=True)
    json.dump({"results": {task: metrics}}, open(os.path.join(d, "results_2026.json"), "w"))
""")


def make_fake_lm_eval(tmp_path):
    """An executable stand-in for lm_eval: writes a results json, fails on GPQA."""
    script = tmp_path / "fake_lm_eval.py"
    script.write_text(FAKE_LM_EVAL_PY)
    wrapper = tmp_path / "lm_eval"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper


def test_per_task_isolation_keeps_completed_tasks(tmp_path, capsys):
    fake = make_fake_lm_eval(tmp_path)
    outdir = tmp_path / "run"
    code = bench.main(["--model", "m", "--tasks", "ifeval,gpqa_diamond_cot_zeroshot,gsm8k",
                       "--lm-eval", str(fake), "--out", str(outdir)])
    assert code == 4  # one task failed
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["scores"]["m"]["gsm8k"] == ["exact_match,flexible-extract", 0.9]
    assert summary["scores"]["m"]["ifeval"] == ["prompt_level_strict_acc,none", 0.8]
    assert "gpqa_diamond_cot_zeroshot" in summary["errors"]["m"][0]
    assert os.path.exists(outdir / "m" / "gpqa_diamond_cot_zeroshot" / "lm_eval.log")
    out = capsys.readouterr().out
    assert "90.0%" in out and "80.0%" in out
