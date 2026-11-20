import json
import subprocess
import sys

from arbor_ab import REPO_ROOT, report

RAW = REPO_ROOT / "evidence" / "2026-11-19_0432_port" / "raw" / "dsv41flash__c1_p90.json"
EPOCH = "dsv41flash__c1_p90__86699be610_55915fb0"


def make_run(tmp_path):
    """A run dir built from the real DSv4.1 C1 campaign's before/after files (evidence bundle)."""
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    run = tmp_path / "r1"
    for phase in ("before", "after"):
        d = run / "dsv41flash__c1_p90" / phase
        d.mkdir(parents=True)
        (d / "agg.json").write_text(json.dumps(raw[f"{phase}_agg"][0]["json"]), encoding="utf-8")
        (d / "gsm8k.log").write_text("\n".join(raw[f"{phase}_gsm8k"][0]["gsm8k_lines"]) + "\n", encoding="utf-8")
    (run / "dsv41flash__c1_p90" / "weights.json").write_text(json.dumps({"weights_source": "user_path", "path": "/w"}))
    (run / "run_config.json").write_text(json.dumps({
        "run_id": "r1", "commit": "f7abce978dc2d3f98f3efc6102cf4bb37983b537",
        "cells": [{"model": "dsv41flash", "target": "c1_p90", "epoch_id": EPOCH}]}), encoding="utf-8")
    return run


def test_generate_report_reproduces_known_result(tmp_path):
    run = make_run(tmp_path)
    p = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_report.py"), "--run-dir", str(run)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    c = json.loads((run / "dsv41flash__c1_p90" / "comparison.json").read_text(encoding="utf-8"))
    d = c["deltas"]["e2e_norm_intvty.p90"]
    assert (d["before"], d["after"], d["change_pct"]) == (13.81163, 124.05897, 798.221)
    assert c["ceiling"]["after_vs_ceiling_pct"] == 0.0            # this campaign's after IS the stored ceiling
    g = c["gsm8k"]
    assert g["scores"]["strict_match"]["drop"] == 0.0083 and g["status"] == "pass"   # 0.9522 -> 0.9439 at max_drop 0.01
    assert c["verdict"] is None and c["baseline_flags"]["numeric_gate"] == "pass"
    assert c["weights"] == {"weights_source": "user_path", "path": "/w"}


def test_gsm8k_skipped_when_both_phases_skipped():
    from arbor_ab import baseline
    r = baseline.load_latest("dsv41flash", "c1_p90")
    m = {"total_tput_tps": 1.0, "tput_per_gpu": 1.0, "intvty": {"p50": 1, "p90": 1, "p95": 1},
         "e2e_norm_intvty": {"p50": 1, "p90": 1, "p95": 1},
         "gsm8k": {"flexible_extract": None, "strict_match": None, "accuracy": None, "num_questions": None}}
    c = report.comparison("r", r, m, m, 0.01, "x", phases={"before": {"gsm8k_questions": 0}, "after": {"gsm8k_questions": 0}})
    assert c["gsm8k"]["status"] == "skipped"
    c = report.comparison("r", r, m, m, 0.01, "x", phases={"before": {"gsm8k_questions": 1319}, "after": {"gsm8k_questions": 0}})
    assert c["gsm8k"]["status"] == "missing"


def test_incomplete_cell_gets_no_comparison(tmp_path):
    run = make_run(tmp_path)
    (run / "dsv41flash__c1_p90" / "after" / "agg.json").unlink()
    p = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_report.py"), "--run-dir", str(run)],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "incomplete" in p.stdout
    assert not (run / "dsv41flash__c1_p90" / "comparison.json").exists()


def test_gsm8k_check_fail_and_missing():
    def m(flex, strict):
        x = {"gsm8k": {"flexible_extract": flex, "strict_match": strict, "accuracy": None, "num_questions": 1319}}
        return x
    assert report.gsm8k_check(m(0.96, 0.95), m(0.94, 0.95), 0.01)["status"] == "fail"
    assert report.gsm8k_check(m(0.96, 0.95), m(None, 0.95), 0.01)["status"] == "missing"
    assert report.gsm8k_check(m(None, None), m(None, None), 0.01)["status"] == "missing"
