import pytest

from evalgate.gate import check_thresholds, compare_to_baseline, evaluate


def make_result(statuses, critical=(), min_pass_rate=None, sha="a" * 64):
    cases = [
        {"id": cid, "status": st, "passed": st == "pass", "critical": cid in critical,
         "detail": "" if st == "pass" else f"{cid} broke"}
        for cid, st in statuses.items()
    ]
    passed = sum(c["passed"] for c in cases)
    gate_config = {"critical": list(critical)}
    if min_pass_rate is not None:
        gate_config["min_pass_rate"] = min_pass_rate
    return {
        "golden": {"name": "t", "sha256": sha, "cases": len(cases)},
        "gate_config": gate_config,
        "summary": {"passed": passed, "total": len(cases), "pass_rate": passed / len(cases)},
        "cases": cases,
    }


def statuses(n_pass, n_total, fail_status="fail"):
    return {f"q{i}": ("pass" if i < n_pass else fail_status) for i in range(n_total)}


# ------------------------------------------------------------- thresholds

def test_pass_rate_threshold():
    assert check_thresholds(make_result(statuses(9, 10), min_pass_rate=0.9)) == []
    failures = check_thresholds(make_result(statuses(8, 10), min_pass_rate=0.9))
    assert len(failures) == 1 and "8/10" in failures[0]


def test_cli_override_of_pass_rate():
    r = make_result(statuses(9, 10), min_pass_rate=0.9)
    assert check_thresholds(r, min_pass_rate=1.0)


def test_critical_case_must_pass_even_if_score_is_fine():
    r = make_result({"a": "pass", "b": "pass", "json": "fail"}, critical=["json"], min_pass_rate=0.5)
    failures = check_thresholds(r)
    assert len(failures) == 1 and "critical case 'json'" in failures[0]


def test_critical_empty_truncated_blocks_too():
    r = make_result({"a": "pass", "safety": "empty_truncated"}, critical=["safety"])
    assert "empty_truncated" in check_thresholds(r)[0]


# ------------------------------------------------------------- baseline regression

def test_regression_vs_baseline_fails():
    report = evaluate(make_result(statuses(9, 10)), make_result(statuses(10, 10)))
    assert not report.ok
    assert "regression vs baseline: 9/10 pass, baseline 10/10" in report.failures[0]
    assert report.newly_failing == ["q9"]


def test_tolerance_allows_a_drop_but_still_reports_it():
    report = evaluate(make_result(statuses(9, 10)), make_result(statuses(10, 10)), tolerance=1)
    assert report.ok
    assert any("newly failing: q9" in w for w in report.warnings)


def test_swap_keeps_count_but_is_reported():
    base = make_result({"a": "pass", "b": "fail"})
    cur = make_result({"a": "fail", "b": "pass"})
    report = compare_to_baseline(cur, base)
    assert report.ok
    assert report.newly_failing == ["a"] and report.newly_passing == ["b"]


def test_better_candidate_passes():
    # e.g. a candidate scoring 10/10 against a baseline that scored 8/10
    report = evaluate(make_result(statuses(10, 10)), make_result(statuses(8, 10)))
    assert report.ok
    assert report.newly_passing == ["q8", "q9"]


def test_changed_golden_set_compares_only_common_cases():
    base = make_result({"a": "pass", "b": "pass", "old": "pass"}, sha="b" * 64)
    cur = make_result({"a": "pass", "b": "pass", "new": "fail"}, sha="c" * 64)
    report = compare_to_baseline(cur, base)
    assert report.ok
    assert any("golden set differs" in w for w in report.warnings)


def test_no_common_cases_fails():
    report = compare_to_baseline(make_result({"x": "pass"}), make_result({"y": "pass"}))
    assert not report.ok


def test_unmeasured_regressions_are_flagged_as_such():
    base = make_result({"a": "pass", "b": "pass"})
    cur = make_result({"a": "pass", "b": "empty_truncated"})
    report = compare_to_baseline(cur, base)
    assert not report.ok
    assert any("not measured" in w and "b" in w for w in report.warnings)


def test_no_baseline_is_thresholds_only_with_warning():
    report = evaluate(make_result(statuses(10, 10), min_pass_rate=0.9))
    assert report.ok and any("no baseline" in w for w in report.warnings)


def test_rejects_non_result_json():
    with pytest.raises(ValueError):
        evaluate({"hello": "world"})
