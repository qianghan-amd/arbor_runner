"""Status of a PR-test run, and its result files fetched home (design.md section 6: short ssh calls only).

Usage: python scripts/poll_status.py --run-dir runs/<run_id> [--no-fetch] [--raise-time]
For each cell: Slurm state (squeue, else sacct), the runner's own state file and finished phases, then (unless
--no-fetch) the small result files of finished phases copied into runs/<run_id>/<cell>/ (no model data, no traces).
--raise-time: a RUNNING job whose time limit is below what the cell needs gets `scontrol update TimeLimit` (the QOS
blocks a long PENDING job, so jobs are submitted short and raised once running: crusoe-arbor-run skill).
Writes runs/<run_id>/status.json. Stateless: run it as often as you like.
"""
import argparse
import datetime as dt
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

FETCH = ("state", "phases", "logs/runner.log", "campaign_check.json", "campaign_exit.json",
         "*/agg.json", "*/bench_output.json", "*/gsm8k.log", "*/phase.json", "*/launch.json", "*/cmdline.txt",
         "*/tree.txt", "*/gsm8k/results_*.json", "*/gsm8k/*/results_*.json")
REMOTE = r'''
import json, os, subprocess, sys
cells = PAYLOAD["cells"]
ids = [c["job_id"] for c in cells if c["job_id"]]
def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout
sq = {}
if ids:
    for line in run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%M|%l|%N|%R"]).splitlines():
        i, st, el, lim, node, reason = line.split("|", 5)
        sq[i] = {"state": st, "elapsed": el, "time_limit": lim, "node": node, "reason": reason}
    # Spur's sacct (config/cluster/crusoe.yaml scheduler: spur) has no -X/-P: fixed fields, whitespace-separated
    for line in run(["sacct", "-n", "-j", ",".join(ids), "-o", "JobID,State,Elapsed,Timelimit,NodeList,ExitCode"]).splitlines():
        f = line.split()
        if len(f) >= 2 and f[0] in ids:
            sq.setdefault(f[0], {"state": f[1], "elapsed": f[2] if len(f) > 2 else "", "time_limit": f[3] if len(f) > 3 else "",
                                 "node": f[4] if len(f) > 4 else "", "exit_code": f[5] if len(f) > 5 else ""})
out = []
for c in cells:
    d = c["remote_dir"]
    state = open(f"{d}/state").read().strip() if os.path.exists(f"{d}/state") else None
    done = sorted(f[:-5] for f in os.listdir(f"{d}/phases") if f.endswith(".done")) if os.path.isdir(f"{d}/phases") else []
    tail = []
    if os.path.exists(f"{d}/logs/runner.log"):
        tail = open(f"{d}/logs/runner.log", errors="replace").read().splitlines()[-5:]
    out.append({"cell": c["cell"], "job": sq.get(c["job_id"] or "", {}), "runner_state": state, "phases_done": done, "log_tail": tail})
print(json.dumps(out))
'''


def to_minutes(limit):
    """Slurm 'D-HH:MM:SS' / 'HH:MM:SS' / 'MM:SS' -> minutes."""
    days, _, rest = limit.partition("-") if "-" in limit else ("0", "", limit)
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days) * 1440 + parts[0] * 60 + parts[1]


def fetch_cell(host, remote_dir, local_dir):
    """Small result files of one cell, via one tar stream (missing patterns are skipped)."""
    pats = " ".join(FETCH)
    # the script goes on stdin: the login shell is tcsh, which mangles $(...) inside a quoted command line
    cmd = f"cd {remote_dir} 2>/dev/null || exit 0\nshopt -s nullglob\nf=$(ls -d {pats} 2>/dev/null)\n[ -n \"$f\" ] && tar -cf - $f\n"
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash", "-s"], input=cmd.encode(), capture_output=True, timeout=300)
    if not p.stdout:
        return 0
    n = 0
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:
        for m in t.getmembers():
            if m.isfile() and not m.name.startswith(("/", "..")) and ".." not in m.name.split("/"):
                dest = local_dir / m.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(t.extractfile(m).read())
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--raise-time", action="store_true")
    args = ap.parse_args()
    from arbor_ab import remote
    rc = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    host = rc["login_host"]
    if not rc.get("submitted"):
        sys.exit(f"{rc['run_id']} was not submitted (dry run or failed sbatch)")
    status = remote.run_python(host, REMOTE, {"cells": rc["cells"]})
    for s, c in zip(status, rc["cells"]):
        job = s["job"]
        if args.raise_time and job.get("state") == "RUNNING" and job.get("time_limit"):
            if to_minutes(job["time_limit"]) < c["job_minutes"]:
                d, m = divmod(c["job_minutes"], 1440)
                lim = f"{d}-{m // 60:02d}:{m % 60:02d}:00"
                p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "scontrol", "update", f"JobId={c['job_id']}",
                                    f"TimeLimit={lim}"], capture_output=True, text=True)
                s["time_raised_to"] = lim if p.returncode == 0 else f"failed: {p.stderr.strip()[-200:]}"
        if not args.no_fetch:
            s["files_fetched"] = fetch_cell(host, c["remote_dir"], args.run_dir / c["cell"])
        print(f"{s['cell']}: slurm {job.get('state', '?')} {job.get('elapsed', '')} on {job.get('node', '')} "
              f"| runner {s['runner_state']} | done {s['phases_done']}"
              + (f" | time raised to {s['time_raised_to']}" if "time_raised_to" in s else ""))
    doc = {"run_id": rc["run_id"], "polled_at": dt.datetime.now(dt.timezone.utc).isoformat(), "cells": status}
    (args.run_dir / "status.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
