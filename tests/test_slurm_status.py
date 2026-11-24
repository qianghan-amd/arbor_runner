"""The slurm backend's remote programs (STATUS_PROGRAM, HOLD_SUBMIT_PROGRAM), run locally as remote.run_python would run
them on the login node, with squeue / sacct / scontrol answered by a fake subprocess.run (no ssh, no cluster).

Bugs these guard: a Spur job held after 5 requeues looked PENDING forever (Crusoe 308822, 2026-11-24); hold-job cells
have no Slurm job of their own and take their state from the ctrl file and the cell's markers.
"""
import contextlib
import getpass
import io
import json
import subprocess
import types

import pytest

from arbor_ab import remote
from arbor_ab.backends import slurm
from test_backends import CLUSTER, LOCAL


def fake_commands(monkeypatch, answers):
    """answers: {command name: stdout text}; any other command prints nothing."""
    def run(cmd, *a, **k):
        return types.SimpleNamespace(stdout=answers.get(cmd[0], ""), stderr="", returncode=0)
    monkeypatch.setattr(subprocess, "run", run)


def local_run_python(host, program, payload, timeout=600):
    ns = {"json": json, "PAYLOAD": json.loads(json.dumps(payload)), "__name__": "__main__"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(program, "<remote program>", "exec"), ns)
    return json.loads(buf.getvalue())


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(remote, "run_python", local_run_python)
    return slurm.Slurm(LOCAL, CLUSTER)


def cell(tmp_path, name, job_id, state=None, done=(), **extra):
    d = tmp_path / name
    (d / "phases").mkdir(parents=True)
    (d / "logs").mkdir()
    for p in done:
        (d / "phases" / f"{p}.done").touch()
    if state:
        (d / "state").write_text(state + "\n")
    (d / "logs" / "runner.log").write_text("[runner] line 1\n[runner] line 2\n")
    return {"cell": name, "remote_dir": d.as_posix(), "job_id": job_id, **extra}


# ---- status: Slurm jobs ----------------------------------------------------------------------------------------------

def test_status_running_preempted_and_held(tmp_path, monkeypatch, backend):
    fake_commands(monkeypatch, {
        "squeue": "11|RUNNING|5:00|3:30:00|n1|None\n13|PENDING|0:00|7:50:00||JobHoldMaxRequeue\n",
        "sacct": "11 RUNNING 00:05:00 03:30:00 n1 0:0\n12 PREEMPTED 00:31:00 03:30:00 n2 0:0\n",
    })
    cells = [cell(tmp_path, "a", "11", "campaign", ["fetch", "prepare"]), cell(tmp_path, "b", "12", "campaign"),
             cell(tmp_path, "c", "13")]
    a, b, c = backend.status(cells)
    assert (a["state"], a["node"], a["phases_done"], a["runner_state"]) == ("RUNNING", "n1", ["fetch", "prepare"], "campaign")
    assert a["log_tail"] == ["[runner] line 1", "[runner] line 2"]
    assert (b["state"], b["raw"]) == ("CANCELLED", "PREEMPTED")
    # Spur held it after 5 requeues: over, and marked PREEMPTED so --resubmit-preempted takes it
    assert c["state"] == "CANCELLED" and c["raw"].startswith("PREEMPTED (JobHoldMaxRequeue")


def test_status_unknown_job(tmp_path, monkeypatch, backend):
    fake_commands(monkeypatch, {})
    (s,) = backend.status([cell(tmp_path, "a", "99")])
    assert s["state"] == "UNKNOWN"


# ---- status: hold-job cells ------------------------------------------------------------------------------------------

def hold_cell(tmp_path, name, state=None, ctrl=None, start=False, rc=None):
    ctrl_dir = tmp_path / "ctrl" / "node1"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    c = cell(tmp_path, name, "hold:9:n1", state, hold_ctrl=(ctrl_dir / f"ab_{name}.sh").as_posix())
    d = tmp_path / name
    if ctrl:
        (ctrl_dir / f"ab_{name}{ctrl}").write_text("#!/bin/bash\n")
    if start:
        (d / "hold_start").write_text("2026-11-24T10:00:00Z\n")
        (d / "hold_node").write_text("crsuse2-m2m-v3-002\n")
    if rc is not None:
        (d / "hold_exit").write_text(f"{rc}\n")
    return c


@pytest.mark.parametrize("kw,hold_running,want", [
    ({"ctrl": ".sh"}, True, "PENDING"),                                    # the agent has not picked it up yet
    ({"ctrl": ".sh.cancelled"}, True, "CANCELLED"),
    ({"ctrl": ".sh.started", "start": True, "state": "smoke"}, True, "RUNNING"),
    ({"ctrl": ".sh.started", "start": True, "state": "smoke"}, False, "FAILED"),   # the hold job ended under it
    ({"ctrl": ".sh.started", "start": True, "rc": 0, "state": "done"}, True, "SUCCEEDED"),
    ({"ctrl": ".sh.started", "start": True, "rc": 0, "state": "smoke"}, True, "FAILED"),
    ({"ctrl": ".sh.started", "start": True, "rc": 1, "state": "failed:prepare"}, True, "FAILED"),
])
def test_status_hold_cells(tmp_path, monkeypatch, backend, kw, hold_running, want):
    fake_commands(monkeypatch, {"squeue": "9|RUNNING|1-00:00:00|2-00:00:00|n0,n1|None\n" if hold_running else ""})
    (s,) = backend.status([hold_cell(tmp_path, "a", **kw)])
    assert s["state"] == want, s
    assert "(hold 9)" in s["raw"]
    if kw.get("start"):
        assert s["node"] == "crsuse2-m2m-v3-002" and s["elapsed"]


# ---- hold submit -----------------------------------------------------------------------------------------------------

def hold_root(tmp_path):
    root = tmp_path / "hold4"
    (root / "ctrl" / "node0").mkdir(parents=True)
    (root / "ctrl" / "node1").mkdir(parents=True)
    (root / "NODES").write_text("n0\nn1\n")
    return root


def scontrol_text(root, state="RUNNING", user="tester(1000)", end="2099-01-01T00:00:00"):
    return (f"JobId=9 JobName=hold4-arbor\n   UserId={user} GroupId=g(1)\n   JobState={state} Reason=None\n"
            f"   StartTime=2026-11-22T21:58:50 EndTime={end}\n   WorkDir={root.as_posix()}\n")


def hold_cells(tmp_path, minutes=150):
    c = cell(tmp_path, "dsv41flash__c1_p90", None, backend_options={"hold_job": 9, "nodelist": "n1"})
    c["remote_dir"] = c["remote_dir"].replace(tmp_path.as_posix(), tmp_path.as_posix() + "/runs/20261124-000000_x_t")
    c["job_minutes"] = minutes
    return [c]


@pytest.mark.parametrize("scontrol,node,minutes,why", [
    ({"state": "PENDING"}, "n1", 150, "is PENDING"),
    ({"user": "someone(2000)"}, "n1", 150, "belongs to someone"),
    ({}, "n7", 150, "is not in hold job 9"),
    ({"end": "2000-01-01T00:00:00"}, "n1", 150, "ends in"),
])
def test_hold_submit_refuses(tmp_path, monkeypatch, backend, scontrol, node, minutes, why):
    root = hold_root(tmp_path)
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    fake_commands(monkeypatch, {"scontrol": scontrol_text(root, **scontrol)})
    cells = hold_cells(tmp_path, minutes)
    cells[0]["backend_options"]["nodelist"] = node
    ids, err = backend.submit(cells)
    assert ids == [None] and why in err
    assert not list((root / "ctrl" / "node1").iterdir())


def test_hold_submit_refuses_a_non_agent_job(tmp_path, monkeypatch, backend):
    root = tmp_path / "plain"
    root.mkdir()
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    fake_commands(monkeypatch, {"scontrol": scontrol_text(root)})
    ids, err = backend.submit(hold_cells(tmp_path))
    assert ids == [None] and "not a node-agent hold job" in err


def test_hold_submit_drops_the_cell_script(tmp_path, monkeypatch, backend):
    root = hold_root(tmp_path)
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    fake_commands(monkeypatch, {"scontrol": scontrol_text(root)})
    cells = hold_cells(tmp_path)
    ids, err = backend.submit(cells)
    assert err is None and ids == ["hold:9:n1"]
    files = sorted(p.name for p in (root / "ctrl" / "node1").iterdir())
    assert files == ["ab_20261124-000000_x_t_dsv41flash__c1_p90.sh"]            # renamed into place: no .tmp left
    text = (root / "ctrl" / "node1" / files[0]).read_text()
    assert text == slurm.hold_script(cells[0]["remote_dir"], 150)
    assert cells[0]["hold_ctrl"].endswith("/ctrl/node1/" + files[0])
