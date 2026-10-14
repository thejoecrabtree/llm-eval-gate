import itertools
import textwrap

import httpx
import pytest

from evalgate.client import ChatClient
from evalgate.runner import GoldenError, load_golden, preflight, run_golden

from conftest import FakeServer, completion, good_model, overriding


def run_example(path, responder, concurrency=2):
    server = FakeServer(responder)
    golden = load_golden(path)
    with ChatClient("http://fake/v1", "test-model", transport=server.transport) as client:
        result = run_golden(golden, client, concurrency=concurrency)
    return result, server


def case(result, cid):
    return next(c for c in result["cases"] if c["id"] == cid)


def test_example_golden_set_loads(example_golden_path):
    g = load_golden(example_golden_path)
    assert 10 <= len(g.cases) <= 15
    assert g.gate["min_pass_rate"] == 0.9
    assert set(g.gate["critical"]) == {
        "json-strict", "tool-call", "reasoning-separation", "refuse-harmful", "no-over-refuse", "determinism"}


def test_good_model_passes_every_example_case(example_golden_path):
    result, server = run_example(example_golden_path, good_model)
    s = result["summary"]
    assert (s["passed"], s["total"]) == (14, 14), [c for c in result["cases"] if c["status"] != "pass"]
    assert s["status_counts"] == {"pass": 14}
    # 14 cases + 1 extra call for the determinism canary
    assert sum(s["finish_reasons"].values()) == 15
    assert s["finish_reasons"]["tool_calls"] == 1
    assert s["errors"] == 0
    assert result["golden"]["name"] == "generic-golden-v1"
    assert len(result["golden"]["sha256"]) == 64


def test_per_case_budgets_and_tools_reach_the_request(example_golden_path):
    _, server = run_example(example_golden_path, good_model)
    assert server.body_for("kill a runaway")["max_tokens"] == 800
    assert server.body_for("phishing email")["max_tokens"] == 512
    assert server.body_for("bat and a ball")["max_tokens"] == 1600  # golden-set default
    weather = server.body_for("weather")
    assert weather["tools"][0]["function"]["name"] == "get_weather"
    assert weather["tool_choice"] == "auto"
    assert all(b["temperature"] == 0.0 for b in server.bodies)


# --------------------------------------------- the empty-content / finish_reason=length case

def test_empty_content_with_length_is_empty_truncated_not_over_refusal(example_golden_path):
    """The 120B-class reasoning model bug: a buffered parser + small max_tokens returns
    finish_reason=length and EMPTY content. That must not be scored as a refusal/blank."""
    buffered = completion("", "length", reasoning="Okay, the user wants to stop a process. Let me")
    result, _ = run_example(example_golden_path, overriding(good_model, {"kill a runaway": buffered}))
    c = case(result, "no-over-refuse")
    assert c["status"] == "empty_truncated"
    assert c["passed"] is False
    assert c["checks"] == []  # checks are not run on a non-measurement
    assert "max_tokens=800" in c["detail"] and "not a refusal" in c["detail"]
    assert c["reasoning_chars"] > 0 and c["content_chars"] == 0
    assert result["summary"]["status_counts"]["empty_truncated"] == 1
    assert result["summary"]["finish_reasons"]["length"] == 1


def test_empty_truncation_on_must_refuse_is_not_counted_as_compliance(example_golden_path):
    buffered = completion("", "length")
    result, _ = run_example(example_golden_path, overriding(good_model, {"phishing email": buffered}))
    assert case(result, "refuse-harmful")["status"] == "empty_truncated"


def test_empty_content_with_stop_is_a_real_blank_answer(example_golden_path):
    result, _ = run_example(example_golden_path, overriding(good_model, {"kill a runaway": completion("", "stop")}))
    c = case(result, "no-over-refuse")
    assert c["status"] == "fail"
    assert "not a substantive answer" in c["detail"]


def test_truncated_but_non_empty_content_is_scored_with_a_warning(example_golden_path):
    result, _ = run_example(example_golden_path, overriding(good_model, {"bat and a ball": completion("5", "length")}))
    c = case(result, "math-trap")
    assert c["status"] == "pass"
    assert any("truncated" in w for w in c["warnings"])


@pytest.mark.parametrize("reason", ["abort", "error"])
def test_abort_and_error_finish_reasons_are_errors(example_golden_path, reason):
    result, _ = run_example(example_golden_path, overriding(good_model, {"3x + 7": completion("-12", reason)}))
    c = case(result, "algebra")
    assert c["status"] == "error" and reason in c["detail"]
    assert result["summary"]["errors"] == 1
    assert result["summary"]["finish_reasons"][reason] == 1


def test_http_failure_is_an_error_not_a_fail(example_golden_path):
    resp = httpx.Response(500, text="engine dead")
    result, _ = run_example(example_golden_path, overriding(good_model, {"tungsten": resp}))
    c = case(result, "knowledge-symbol")
    assert c["status"] == "error" and "HTTP 500" in c["detail"]
    assert result["summary"]["finish_reasons"]["transport_error"] == 1


def test_determinism_drift_fails_the_canary(example_golden_path):
    answers = itertools.cycle(["2, 3, 5, 7, 11, 13, 17, 19", "2,3,5,7,11,13,17,19"])
    drift = lambda body: completion(next(answers))  # noqa: E731
    result, _ = run_example(example_golden_path, overriding(good_model, {"first 8 prime": drift}), concurrency=1)
    c = case(result, "determinism")
    assert c["status"] == "fail" and "DRIFT" in c["detail"]


def test_reasoning_leak_fails_reasoning_separation(example_golden_path):
    leak = completion("<think>51 = 3 x 17</think> No")
    result, _ = run_example(example_golden_path, overriding(good_model, {"Is 51 prime": leak}))
    assert case(result, "reasoning-separation")["status"] == "fail"


# ------------------------------------------------------------- client details

def test_api_key_sent_as_bearer_only_when_given():
    server = FakeServer(lambda body: completion("ok"))
    with ChatClient("http://fake/v1", "m", api_key="test-key-123", transport=server.transport) as c:
        c.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    with ChatClient("http://fake/v1", "m", transport=server.transport) as c:
        c.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert server.headers[0]["authorization"] == "Bearer test-key-123"
    assert "authorization" not in server.headers[1]


def test_request_goes_to_chat_completions_under_base_url():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=completion("ok"))

    with ChatClient("http://fake:8000/v1/", "m", transport=httpx.MockTransport(handler)) as c:
        c.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert seen == ["http://fake:8000/v1/chat/completions"]


def test_preflight_tolerates_buffered_empty_reply_but_not_http_errors():
    ok = FakeServer(lambda body: completion("", "length"))
    with ChatClient("http://fake/v1", "m", transport=ok.transport) as c:
        assert preflight(c) is None
    down = FakeServer(lambda body: httpx.Response(503, text="loading model"))
    with ChatClient("http://fake/v1", "m", transport=down.transport) as c:
        assert "HTTP 503" in preflight(c)


# ------------------------------------------------------------- golden-set validation

def write(tmp_path, text):
    p = tmp_path / "g.yaml"
    p.write_text(textwrap.dedent(text))
    return p


@pytest.mark.parametrize("body, message", [
    ("cases: []", "non-empty 'cases'"),
    ("cases:\n  - prompt: hi\n    checks: [{contains: a}]", "needs an 'id'"),
    ("cases:\n  - id: a\n    checks: [{contains: a}]", "'prompt' or 'messages'"),
    ("cases:\n  - id: a\n    prompt: hi", "no checks"),
    ("cases:\n  - id: a\n    prompt: hi\n    checks: [{bogus: 1}]", "unknown check"),
    ("cases:\n  - {id: a, prompt: hi, checks: [{contains: a}]}\n  - {id: a, prompt: hi, checks: [{contains: a}]}",
     "duplicate"),
    ("gate: {critical: [zzz]}\ncases:\n  - {id: a, prompt: hi, checks: [{contains: a}]}", "unknown case ids"),
])
def test_golden_validation_errors(tmp_path, body, message):
    with pytest.raises(GoldenError, match=message):
        load_golden(write(tmp_path, body))


def test_system_prompt_and_critical_flag(tmp_path):
    g = load_golden(write(tmp_path, """
        gate: {critical: [b]}
        cases:
          - {id: a, prompt: hi, system: be brief, checks: [{contains: a}], critical: true}
          - {id: b, prompt: hi, checks: [{contains: b}]}
          - {id: c, prompt: hi, checks: [{contains: c}]}
    """))
    assert g.cases[0].messages[0] == {"role": "system", "content": "be brief"}
    assert [c.critical for c in g.cases] == [True, True, False]
    assert g.gate["critical"] == ["a", "b"]
