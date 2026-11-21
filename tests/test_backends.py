"""The execution-backend boundary (design.md 5.1): the orchestrator works through the contract alone."""
import json
import shutil
import subprocess
import sys

import pytest

from arbor_ab import REPO_ROOT, backends
from arbor_ab.backends import crusoe_slurm

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import poll_status  # noqa: E402
import submit_ab_test  # noqa: E402

LOCAL = {"backend": "crusoe_slurm", "ssh": {"login_host": "h", "github_relay_host": "r"},
         "remote": {"work_root": "/w", "infx_clone": "/x/InferenceX", "node_local_root": "/nl", "shared_hf_hub": "/hub"},
         "slurm": {"qos": "amd-agentx-qos", "account": "amd-agentx"}, "arbor": {"clone": "../Arbor", "gateway_env": None}}
CLUSTER = {"partition": "amd-spur-v2", "qos_max_wall_minutes": {"amd-burst-qos": 1440, "amd-agentx-qos": 2880}}


class FakeBackend(backends.Backend):
    """In memory: 'stages' by copying into a directory, every job 'succeeds' and leaves a result file."""
    name = "fake"

    def __init__(self, root):
        self.root, self.submitted = root, []

    def site(self):
        return {"node_local_root": "/nl", "shared_hf_hub": "-", "infx_clone": "/x", "gateway_env": None}

    def run_dir(self, run_id):
        return str(self.root / run_id)

    def job_script(self, cell_dir, job_name, minutes, exclude=(), options=None):
        return "job.sh", f"#!/bin/bash\nexec bash {cell_dir}/runner.sh {cell_dir}\n"

    def stage(self, run_id, local_run_dir, cells, extra_files):
        for c in cells:
            shutil.copytree(local_run_dir / c["cell"], c["remote_dir"])
        for f in extra_files:
            shutil.copy(f, self.run_dir(run_id))

    def submit(self, cells):
        self.submitted += [c["cell"] for c in cells]
        for c in cells:   # the 'job' runs at once and writes what a runner writes
            (self.root / c["remote_dir"]).mkdir(parents=True, exist_ok=True)
            open(f"{c['remote_dir']}/state", "w").write("done")
        return [f"j{i}" for i, _ in enumerate(cells)], None

    def status(self, cells):
        return [{"state": "SUCCEEDED", "raw": "ok", "node": "local", "elapsed": "0", "time_limit": "", "reason": "",
                 "runner_state": open(f"{c['remote_dir']}/state").read(), "phases_done": [], "log_tail": []} for c in cells]

    def fetch(self, cell, local_dir):
        shutil.copy(f"{cell['remote_dir']}/state", local_dir / "state")
        return 1

    def cancel(self, cell):
        return "cancelled"


def _git_repo(path):
    path.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(path), *args], check=True)
    (path / "run.sh").write_bytes(b"echo hi\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "x"], check=True)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def test_orchestrator_runs_on_any_backend(tmp_path):
    sha = _git_repo(tmp_path / "arbor")
    fake = FakeBackend(tmp_path / "cluster")
    (tmp_path / "cluster").mkdir()
    req = {"commit": sha, "mode": "smoke",
           "targets": [{"model": "dsv41flash", "target": "c1_p90", "weights_path": "/shared/weights/dsv41"}]}
    rc = submit_ab_test.submit(req, "fake-test.yaml", fake, tmp_path / "arbor")
    run_dir = REPO_ROOT / "runs" / rc["run_id"]
    try:
        assert rc["submitted"] and rc["backend"] == "fake" and fake.submitted == ["dsv41flash__c1_p90"]
        cell_dir = tmp_path / "cluster" / rc["run_id"] / "dsv41flash__c1_p90"
        assert (cell_dir / "job.sh").exists() and (cell_dir / "runner.sh").exists()
        env = (cell_dir / "cell.env").read_text()
        assert f"CELL_DIR={rc['cells'][0]['remote_dir']}" in env.replace("'", "") and "MODE=smoke" in env
        assert "WEIGHTS_PATH=/shared/weights/dsv41" in env and rc["cells"][0]["weights_path"] == "/shared/weights/dsv41"
        doc = poll_status.poll(run_dir, fake)
        assert doc["cells"][0]["state"] == "SUCCEEDED" and (run_dir / "dsv41flash__c1_p90" / "state").read_text() == "done"
        assert json.loads((run_dir / "status.json").read_text())["backend"] == "fake"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_relative_weights_path_refused(tmp_path):
    import pytest
    sha = _git_repo(tmp_path / "arbor")
    req = {"commit": sha, "mode": "smoke", "targets": [{"model": "dsv41flash", "target": "c1_p90", "weights_path": "w/x"}]}
    with pytest.raises(SystemExit, match="absolute path"):
        submit_ab_test.submit(req, "fake-test.yaml", FakeBackend(tmp_path / "cluster"), tmp_path / "arbor")


def test_get_selects_backend():
    b = backends.get(LOCAL, CLUSTER)
    assert b.name == "crusoe_slurm" and b.run_dir("r1") == "/w/runs/r1"


def test_crusoe_job_script():
    b = crusoe_slurm.CrusoeSlurm(LOCAL, CLUSTER)
    name, text = b.job_script("/w/runs/r1/c", "ab-c", 1890)
    assert name == "job.sbatch" and "--exclude" not in text
    assert "#SBATCH --exclude=n1,n2" in b.job_script("/w/c", "ab-c", 150, exclude=("n1", "n2"))[1]
    burst = b.job_script("/w/c", "ab-c", 150, options={"qos": "amd-burst-qos", "account": "amd-hyperloom"})[1]
    with pytest.raises(ValueError, match="allows 1440 min"):
        b.job_script("/w/c", "ab-c", 1441, options={"qos": "amd-burst-qos"})
    assert "#SBATCH --qos=amd-burst-qos" in burst and "#SBATCH --account=amd-hyperloom" in burst
    with pytest.raises(ValueError, match="unknown"):
        b.job_script("/w/c", "ab-c", 150, options={"nodes": 2})
    for line in ("#SBATCH --partition=amd-spur-v2", "#SBATCH --qos=amd-agentx-qos", "#SBATCH --account=amd-agentx",
                 "#SBATCH --exclusive", "#SBATCH --time=1-07:30:00", "#SBATCH --error=/w/runs/r1/c/logs/slurm-%j.err",
                 "exec bash /w/runs/r1/c/runner.sh /w/runs/r1/c 2>&1"):
        assert line in text


SITES = {"backend": "slurm", "arbor": {"clone": "../Arbor", "gateway_env": None}, "default_site": "crusoe",
         "sites": {"crusoe": {"ssh": {"login_host": "c", "github_relay_host": "r"}, "remote": {"work_root": "/c/w"},
                              "slurm": {"qos": "q1", "account": "a1"}, "arbor": {"gateway_env": "/c/w/gw.env"}},
                   "meta64": {"ssh": {"login_host": "m", "github_relay_host": "r"}, "remote": {"work_root": "/m/w"}}}}


def test_sites_and_a_cluster_without_qos():
    name, view = backends.site_local(SITES)
    assert name == "crusoe" and view["cluster"] == "crusoe" and view["arbor"] == {"clone": "../Arbor", "gateway_env": "/c/w/gw.env"}
    name, view = backends.site_local(SITES, "meta64")
    assert view["ssh"]["login_host"] == "m" and view["arbor"]["clone"] == "../Arbor" and "sites" not in view
    with pytest.raises(KeyError, match="not in local.yaml sites"):
        backends.site_local(SITES, "nowhere")
    assert backends.site_local(LOCAL) == ("crusoe", LOCAL)                       # the older flat local.yaml: one site
    meta = backends.for_site(SITES, "meta64")                                   # config/cluster/meta64.yaml
    assert meta.site_name == "meta64" and meta.gpu_type == "mi350x" and meta.run_dir("r1") == "/m/w/runs/r1"
    text = meta.job_script("/m/w/runs/r1/c", "ab-c", 600)[1]
    assert "#SBATCH --partition=meta64" in text and "--qos" not in text and "--account" not in text
    with pytest.raises(ValueError, match="partition meta64 allows 7200 min"):
        meta.job_script("/m/w/c", "ab-c", 7201)


def test_crusoe_time_conversions():
    assert crusoe_slurm.to_minutes("1-07:30:00") == 1890 and crusoe_slurm.to_minutes("02:30:00") == 150
    assert crusoe_slurm.to_minutes("45:00") == 45 and crusoe_slurm.slurm_time(150) == "0-02:30:00"


def test_crusoe_status_normalizes(monkeypatch):
    raw = [{"job": {"raw": "RUNNING", "elapsed": "1:15", "time_limit": "02:30:00", "node": "n1"}, "runner_state": "fetch",
            "phases_done": [], "log_tail": []},
           {"job": {"raw": "CANCELLED+", "elapsed": "0", "time_limit": "", "node": ""}, "runner_state": None,
            "phases_done": [], "log_tail": []},
           {"job": {}, "runner_state": None, "phases_done": [], "log_tail": []}]
    monkeypatch.setattr(crusoe_slurm.remote, "run_python", lambda host, prog, payload, timeout=600: raw)
    b = crusoe_slurm.CrusoeSlurm(LOCAL, CLUSTER)
    s = b.status([{"job_id": "1", "remote_dir": "/a"}, {"job_id": "2", "remote_dir": "/b"}, {"job_id": None, "remote_dir": "/c"}])
    assert [x["state"] for x in s] == ["RUNNING", "CANCELLED", "UNKNOWN"]
    assert b.extend({"job_id": "1", "job_minutes": 150}, s[0]) is None          # already has its wall time
