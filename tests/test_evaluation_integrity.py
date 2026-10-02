"""Offline checks for comparable, complete evaluation evidence (no model calls)."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_results as analysis
import ci_eval
import fit_routing
import judge_agreement as agreement


def row(tid=1, reward=1, **extra):
    return {"task_id": tid, "org_type": "b2b", "interactive": False,
            "task_type": "case_routing", "reward": reward, "gt_answer": ["ABC"],
            "traj": [{"role": "user", "content": "Which case?"}, {"role": "assistant", "content": "ABC"}], **extra}


def results(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "results_fixture_b2b.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    path.with_name("config_" + path.name).write_text(json.dumps({
        "model": "fixture", "agent_strategy": "remote", "org_type": "b2b", "interactive": False,
        "agent_eval_mode": "aided", "judge_model": "fixture-judge", "judge_provider": "fixture",
        "user_model": "fixture-user", "user_provider": "fixture", "generation": {}, "tasks_sha256": "fixture",
    }), encoding="utf-8")
    return path


def split(tmp_path, ids=(1, 2)):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"b2b": {"single_turn": list(ids), "multi_turn": []},
                                "b2c": {"single_turn": [], "multi_turn": []}}), encoding="utf-8")
    return path


def test_duplicate_result_identity_is_not_silently_overwritten(tmp_path):
    results(tmp_path, [row(), row(reward=0)])
    with pytest.raises(ValueError, match="duplicate task"):
        analysis.load_system(tmp_path)


@pytest.mark.parametrize("records", [[row()], [row(), row(2, error="HTTP 500")], [row(), row(2), row(3)]])
def test_complete_report_rejects_missing_errors_and_unexpected_tasks(tmp_path, records):
    wanted = analysis.expected_keys(split(tmp_path))
    results(tmp_path, records)
    with pytest.raises(ValueError, match="incomplete evaluation"):
        analysis.validate_complete({"agent": analysis.load_system(tmp_path)}, wanted)


def test_deterministic_system_failure_remains_a_completed_zero(tmp_path):
    results(tmp_path, [row(1, reward=0, failure_reason="request_too_large"), row(2)])
    analysis.validate_complete({"agent": analysis.load_system(tmp_path)}, analysis.expected_keys(split(tmp_path)))


def test_routing_rejects_equal_size_different_tasks():
    with pytest.raises(ValueError, match="matching nonempty"):
        fit_routing.paired_rates({("b2b", False, "1"): row()}, {("b2b", False, "2"): row(2)}, 0.5)


def test_routing_rejects_error_and_inconsistent_type():
    key = ("b2b", False, "1")
    for bad in (row(error="quota"), row(task_type="knowledge_qa")):
        with pytest.raises(ValueError):
            fit_routing.paired_rates({key: row()}, {key: bad}, 0.5)


def test_routing_pairs_by_identity_not_input_order():
    a, b = ("b2b", False, "1"), ("b2b", False, "2")
    big, small = fit_routing.paired_rates({a: row(), b: row(2, 0)}, {b: row(2, 1), a: row(1, 0)}, 0.5)
    assert big["case_routing"] == [1, 0]
    assert small["case_routing"] == [0, 1]


def test_unpriced_or_missing_cost_is_not_free():
    assert analysis.cost_of(row()) is None
    assert analysis.cost_of(row(agent_info={"total_cost": 0, "usage": {"cost_complete": False}})) is None
    assert analysis.cost_of(row(agent_info={"total_cost": 0, "usage": {"cost_complete": True}})) == 0


def test_ci_does_not_create_baseline_implicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_eval, "CASES", split(tmp_path))
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(ci_eval, "BASELINE", baseline)
    results(tmp_path, [row(), row(2)])
    assert ci_eval.compare(str(tmp_path), False, 0.12) == 1
    assert not baseline.exists()
    assert ci_eval.compare(str(tmp_path), True, 0.12) == 0
    assert ci_eval.compare(str(tmp_path), False, 0.12) == 0


def test_ci_rejects_partial_even_when_updating_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_eval, "CASES", split(tmp_path))
    monkeypatch.setattr(ci_eval, "BASELINE", tmp_path / "baseline.json")
    results(tmp_path, [row()])
    assert ci_eval.compare(str(tmp_path), True, 0.12) == 1
    assert not ci_eval.BASELINE.exists()


def test_ci_rejects_model_changes_but_allows_a_new_code_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(ci_eval, "CASES", split(tmp_path))
    monkeypatch.setattr(ci_eval, "BASELINE", tmp_path / "baseline.json")
    source = results(tmp_path, [row(), row(2)])
    assert ci_eval.compare(str(tmp_path), True, 0.12) == 0
    sidecar = source.with_name("config_" + source.name)
    config = json.loads(sidecar.read_text())
    config["generation"]["CRMARENA_RUN_FINGERPRINT"] = "new-PR-source"
    sidecar.write_text(json.dumps(config))
    assert ci_eval.compare(str(tmp_path), False, 0.12) == 0
    config["judge_model"] = "different-judge"
    sidecar.write_text(json.dumps(config))
    assert ci_eval.compare(str(tmp_path), False, 0.12) == 1


def labelled_sheet(tmp_path):
    source = results(tmp_path / "run", [row(1, 1), row(2, 0)])
    sheet = tmp_path / "labels.csv"
    agreement.make_sheet([f"agent={source.parent}"], 2, str(sheet), 40)
    with sheet.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["human_correct"] = "1" if r["key"].endswith("|1") else "0"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agreement.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return source, sheet


def test_label_sheet_is_preserved_and_judge_agreement_is_gated(tmp_path):
    source, sheet = labelled_sheet(tmp_path)
    assert agreement.kappa(str(sheet), min_labels=2) == 0
    assert agreement.kappa(str(sheet), min_labels=40) == 1
    before = sheet.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        agreement.make_sheet([f"agent={source.parent}"], 2, str(sheet), 40)
    assert sheet.read_bytes() == before


def test_labels_cannot_grade_changed_source_response_or_verdict(tmp_path):
    source, sheet = labelled_sheet(tmp_path)
    records = json.loads(source.read_text())
    records[0]["reward"] = 0
    source.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="grading changed"):
        agreement.kappa(str(sheet), min_labels=2)


def test_analysis_requires_explicit_partial_mode_and_marks_report(tmp_path, monkeypatch, capsys):
    results(tmp_path, [row()])
    monkeypatch.setattr(sys, "argv", ["analyze_results.py", "--system", f"agent={tmp_path}"])
    with pytest.raises(SystemExit) as exc:
        analysis.main()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, "argv", ["analyze_results.py", "--system", f"agent={tmp_path}", "--allow-partial", "--iters", "100"])
    analysis.main()
    assert "EXPLORATORY PARTIAL REPORT" in capsys.readouterr().out


def test_incomplete_cost_is_reported_as_a_lower_bound_not_a_total():
    complete = row(agent_info={"total_cost": 0.02, "usage": {"cost_complete": True}})
    partial = row(2, agent_info={"total_cost": 0.04, "usage": {"cost_complete": False}})
    rows_ = [complete, partial]
    text = analysis.cost_summary([analysis.cost_of(r) for r in rows_], rows_)
    assert text == ">= 0.0300 (1 of 2 tasks incomplete)"
    assert analysis.cost_summary([analysis.cost_of(complete)], [complete]) == "0.0200"
