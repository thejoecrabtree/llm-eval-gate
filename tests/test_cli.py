import json

import httpx
import pytest

from evalgate.cli import EXIT_ENDPOINT, EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE, main

from conftest import FakeServer, completion, good_model, overriding


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("EVALGATE_BASE_URL", "EVALGATE_MODEL", "EVALGATE_API_KEY", "MY_GATEWAY_KEY"):
        monkeypatch.delenv(var, raising=False)


def run_cli(args, responder):
    server = FakeServer(responder)
    code = main(args, transport=server.transport)
    return code, server


def degraded_model(body):
    # A candidate that breaks strict JSON (critical) and one maths case.
    return overriding(good_model, {
        "JSON object": completion('Here you go: {"name": "sample", "count": "4", "active": "true"}'),
        "bat and a ball": completion("10"),
    })(body)


def test_run_writes_result_and_exits_zero(tmp_path, example_golden_path, capsys):
    out = tmp_path / "r.json"
    code, server = run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(out)], good_model)
    assert code == EXIT_OK
    result = json.loads(out.read_text())
    assert result["summary"]["passed"] == 14
    assert result["endpoint"] == {"base_url": "http://localhost:8000/v1", "model": "m"}
    assert "score 14/14" in capsys.readouterr().out
    assert "Reply with OK." in server.bodies[0]["messages"][0]["content"]  # preflight first


def test_default_output_path_under_results(tmp_path, example_golden_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVALGATE_MODEL", "my/model")
    code, _ = run_cli(["run", str(example_golden_path)], good_model)
    assert code == EXIT_OK
    files = list((tmp_path / "results").glob("generic-golden-v1-my_model-*.json"))
    assert len(files) == 1


def test_gate_blocks_regression_against_baseline(tmp_path, example_golden_path, capsys):
    baseline = tmp_path / "last.json"
    assert run_cli(["run", str(example_golden_path), "--model", "prod", "--out", str(baseline)], good_model)[0] == 0
    cand = tmp_path / "cand.json"
    code, _ = run_cli(["run", str(example_golden_path), "--model", "candidate", "--out", str(cand),
                       "--gate", "--baseline", str(baseline)], degraded_model)
    out = capsys.readouterr().out
    assert code == EXIT_GATE_FAILED
    assert "GATE: FAIL - do not promote" in out
    assert "critical case 'json-strict'" in out
    saved = json.loads(cand.read_text())
    assert saved["gate"]["ok"] is False
    assert saved["gate"]["newly_failing"] == ["json-strict", "math-trap"]
    # the gate never rewrites the baseline
    assert json.loads(baseline.read_text())["endpoint"]["model"] == "prod"


def test_gate_passes_identical_candidate(tmp_path, example_golden_path, capsys):
    baseline = tmp_path / "last.json"
    run_cli(["run", str(example_golden_path), "--model", "prod", "--out", str(baseline)], good_model)
    code, _ = run_cli(["run", str(example_golden_path), "--model", "cand", "--out", str(tmp_path / "c.json"),
                       "--gate", "--baseline", str(baseline)], good_model)
    assert code == EXIT_OK
    assert "GATE: PASS" in capsys.readouterr().out


def test_gate_fails_when_measurement_is_truncated(tmp_path, example_golden_path, capsys):
    """A too-small budget on a critical safety probe blocks the gate, and says why."""
    buffered = completion("", "length")
    code, _ = run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(tmp_path / "r.json"),
                       "--gate"], overriding(good_model, {"kill a runaway": buffered}))
    out = capsys.readouterr().out
    assert code == EXIT_GATE_FAILED
    assert "TRUNC" in out and "raise max_tokens" in out


def test_missing_baseline_is_a_usage_error_before_any_request(tmp_path, example_golden_path):
    code, server = run_cli(["run", str(example_golden_path), "--model", "m", "--gate",
                            "--baseline", str(tmp_path / "nope.json")], good_model)
    assert code == EXIT_USAGE
    assert server.bodies == []


def test_baseline_without_gate_is_rejected(tmp_path, example_golden_path):
    code, _ = run_cli(["run", str(example_golden_path), "--model", "m", "--baseline", "x.json"], good_model)
    assert code == EXIT_USAGE


def test_model_is_required(example_golden_path):
    assert run_cli(["run", str(example_golden_path)], good_model)[0] == EXIT_USAGE


def test_preflight_failure_aborts_with_endpoint_code(tmp_path, example_golden_path):
    code, server = run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(tmp_path / "r.json")],
                           lambda body: httpx.Response(503, text="loading"))
    assert code == EXIT_ENDPOINT
    assert len(server.bodies) == 1  # only the preflight request


def test_every_case_erroring_is_an_endpoint_problem(tmp_path, example_golden_path):
    def responder(body):
        if "Reply with OK." in body["messages"][-1]["content"]:
            return completion("OK")
        return completion("", "abort")
    code, _ = run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(tmp_path / "r.json"),
                       "--gate"], responder)
    assert code == EXIT_ENDPOINT


def test_api_key_comes_from_env_var(tmp_path, example_golden_path, monkeypatch):
    monkeypatch.setenv("EVALGATE_API_KEY", "test-key-abc")
    _, server = run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(tmp_path / "r.json")],
                        good_model)
    assert all(h["authorization"] == "Bearer test-key-abc" for h in server.headers)

    monkeypatch.delenv("EVALGATE_API_KEY")
    monkeypatch.setenv("MY_GATEWAY_KEY", "test-key-xyz")
    _, server = run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(tmp_path / "r2.json"),
                         "--api-key-env", "MY_GATEWAY_KEY"], good_model)
    assert server.headers[0]["authorization"] == "Bearer test-key-xyz"
    # the key is never written into the result file
    assert "test-key" not in (tmp_path / "r2.json").read_text()


def test_offline_gate_command(tmp_path, example_golden_path, capsys):
    base, good, bad = tmp_path / "base.json", tmp_path / "good.json", tmp_path / "bad.json"
    run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(base)], good_model)
    run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(good)], good_model)
    run_cli(["run", str(example_golden_path), "--model", "m", "--out", str(bad)], degraded_model)
    assert main(["gate", str(good), "--baseline", str(base)]) == EXIT_OK
    assert main(["gate", str(bad), "--baseline", str(base)]) == EXIT_GATE_FAILED
    assert main(["gate", str(bad), "--baseline", str(base), "--tolerance", "5", "--min-pass-rate", "0.5"]) \
        == EXIT_GATE_FAILED  # json-strict is critical: tolerance does not excuse it
    assert main(["gate", str(tmp_path / "missing.json")]) == EXIT_USAGE
