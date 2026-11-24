"""make_baseline finalize on a tmp copy of baselines/: a gate pass stores the record (latest moves, the old epoch gets
superseded_by), a gate fail stores nothing. The draft comes from the offline CI fixtures (test_draft_offline); the
measured `before` is the CI's own AgentX record, so the numeric gate compares like with like.
Live counterpart: Crusoe 308511 (2026-11-24), gate -0.24 %, stored, then restored by hand.
"""
import argparse
import json
import shutil
import subprocess

import pytest

from arbor_ab import REPO_ROOT, baseline

import make_baseline
from test_draft_offline import RUN, draft, offline  # noqa: F401  (offline is a fixture)


def with_metrics(b):
    """The fixtures' bmk JSON is trimmed: give it the c1_p90 primary metric (e2e_norm_intvty p90, CI 44689368479)."""
    b["request_metrics"] = {"latency": {"e2e_norm_intvty": {"p50": 12.0, "p90": 13.73835}}}


@pytest.fixture
def tmp_baselines(monkeypatch, tmp_path):
    root = tmp_path / "baselines"
    shutil.copytree(REPO_ROOT / "baselines" / "dsv41flash" / "c1_p90", root / "dsv41flash" / "c1_p90")
    monkeypatch.setattr(baseline, "BASELINES_DIR", root)
    return root


def run_dir(tmp_path, relay, model="dsv41flash", target="c1_p90"):
    r = tmp_path / "runs" / "20261124-000000_test"
    before = r / f"{model}__{target}" / "before"
    before.mkdir(parents=True)
    (before / "agg.json").write_text(json.dumps(relay.bmk), encoding="utf-8")
    (before / "phase.json").write_text(json.dumps({"ended_at": "2026-11-24T08:00:00Z", "node": "n1", "job": "1",
                                                   "duration_s": 3600}), encoding="utf-8")
    (r / "run_config.json").write_text(json.dumps({"run_id": r.name, "backend": "slurm", "commit": "c" * 40}),
                                       encoding="utf-8")
    return r


def finalize(d, r):
    return make_baseline.cmd_finalize(argparse.Namespace(draft=str(d), run_dir=str(r), replace=False))


def git_status_baselines():
    return subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "baselines"],
                          capture_output=True, text=True).stdout


def test_gate_pass_stores_and_supersedes(offline, tmp_baselines, tmp_path, capsys):  # noqa: F811
    before_status = git_status_baselines()
    relay = offline("vllm_srtctl", bmk_edit=with_metrics)
    d, rec = draft(relay, "dsv41flash", "c1_p90")
    old = baseline.load(tmp_baselines / "dsv41flash" / "c1_p90" / "latest.json")["epoch_id"]
    assert finalize(d, run_dir(tmp_path, relay)) == 0
    out = capsys.readouterr().out
    assert "gate: structural pass, numeric pass" in out and "stored:" in out
    new = baseline.load(tmp_baselines / "dsv41flash" / "c1_p90" / "latest.json")["epoch_id"]
    assert new != old and (tmp_baselines / "dsv41flash" / "c1_p90" / f"{new}.json").exists()
    stored = baseline.load(tmp_baselines / "dsv41flash" / "c1_p90" / f"{new}.json")
    assert baseline.validation_errors(stored) == [] and stored["verification"]["ci_verified"] is True
    assert baseline.load(tmp_baselines / "dsv41flash" / "c1_p90" / f"{old}.json")["superseded_by"] == new
    assert (d / "finalized.json").exists()
    assert git_status_baselines() == before_status                     # the real baselines/ did not change


def test_gate_fail_stores_nothing(offline, tmp_baselines, tmp_path, capsys):  # noqa: F811
    relay = offline("vllm_srtctl", bmk_edit=with_metrics)
    d, rec = draft(relay, "dsv41flash", "c1_p90")
    files = sorted(p.name for p in (tmp_baselines / "dsv41flash" / "c1_p90").iterdir())
    r = run_dir(tmp_path, relay)
    agg = json.loads((r / "dsv41flash__c1_p90" / "before" / "agg.json").read_text(encoding="utf-8"))
    agg["tp"] = 8                                                     # the CI record says TP 8, the draft TP 4
    (d / "ci_bmk.json").write_text(json.dumps({"artifact": {"name": relay.name, "id": 1}, "json": agg}), encoding="utf-8")
    assert finalize(d, r) == 1
    out = capsys.readouterr().out
    assert "gate: structural fail" in out and "not stored" in out
    assert sorted(p.name for p in (tmp_baselines / "dsv41flash" / "c1_p90").iterdir()) == files
    assert (d / "finalized.json").exists()
