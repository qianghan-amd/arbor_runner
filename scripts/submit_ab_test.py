"""Submit a PR test: request YAML -> run_id -> runs/<run_id>/ -> one sbatch per cell -> exit (design.md sections 4-6).

Usage:
  python scripts/submit_ab_test.py --request runs/requests/<name>.yaml [--dry-run]
Request:
  commit: <Arbor commit SHA>             # identified by commit, not PR number (design.md section 4)
  mode: smoke | full                     # smoke = fetch + prepare + 600 s AgentX + 200 GSM8K questions, no campaign
  campaign_minutes: 1440                 # full mode only (Arbor's time_budget_minutes)
  targets:
    - { model: dsv41flash, target: c1_p90 }
Stateless: everything it decides is written to runs/<run_id>/run_config.json; it never waits for results.
--dry-run renders every file locally (runs/<run_id>/) and touches no cluster. A cell that is not runnable
(render.check_runnable) refuses the whole submission, with the reasons.
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from arbor_ab import REPO_ROOT, baseline, config, render

RUNNER_FILES = ("runner.sh", "serve_cmd.py", "fetch.py")


SHELL_EXT = (".sh", ".bash", ".sbatch")


def is_shell(name, data):
    first = data.split(b"\n", 1)[0]
    return name.endswith(SHELL_EXT) or (first.startswith(b"#!") and first.rstrip(b"\r").endswith((b"sh", b"bash")))


def archive_crlf_files(tar_path):
    """(shell scripts with CRLF line ends, other text members with CRLF). Binary members are skipped.

    Upstream commits some files with CRLF (reference docs, a skill's .py, a package.json; verified 2026-11-19 against
    the blobs of f7abce97): they ship byte-exact as committed, and Python/JSON/YAML parsers accept CRLF. bash does not,
    so a shell script with CRLF refuses the submission.
    """
    import tarfile
    bad, other = [], []
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if m.isfile() and m.size < 5_000_000:
                b = t.extractfile(m).read()
                if b"\r\n" in b and b"\0" not in b:
                    (bad if is_shell(m.name, b) else other).append(m.name)
    return bad, other


def git(clone, *args):
    return subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    req = yaml.safe_load(args.request.read_text(encoding="utf-8"))
    mode = req.get("mode", "full")
    if mode not in render.MODES:
        sys.exit(f"mode must be one of {render.MODES}")
    minutes = int(req.get("campaign_minutes", 1440)) if mode == "full" else 0
    local = config.load_local()
    cluster = config.load_cluster()
    clone = (REPO_ROOT / local["arbor"]["clone"]).resolve()
    try:
        commit = git(clone, "rev-parse", "--verify", f"{req['commit']}^{{commit}}")
    except subprocess.CalledProcessError:
        sys.exit(f"commit {req['commit']} not found in {clone} (git fetch it first)")
    if mode == "full" and not local["arbor"].get("gateway_env"):
        sys.exit("full mode needs local.yaml arbor.gateway_env (the LLM gateway for Arbor's agents)")

    records, refusals = [], []
    for t in req["targets"]:
        r = baseline.load_latest(t["model"], t["target"])
        why = ["no baseline record"] if r is None else render.check_runnable(r)
        if why:
            refusals.append(f"{t['model']}/{t['target']}: " + "; ".join(why))
        records.append(r)
    if refusals:
        sys.exit("not submitted, cells not runnable:\n  " + "\n  ".join(refusals))

    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}_{commit[:8]}_{args.request.stem}"
    run_dir = REPO_ROOT / "runs" / run_id
    remote_run = f"{local['remote']['work_root']}/runs/{run_id}"
    site = {"node_local_root": local["remote"]["node_local_root"], "shared_hf_hub": local["remote"]["shared_hf_hub"],
            "infx_clone": local["remote"]["infx_clone"], "gateway_env": local["arbor"].get("gateway_env"),
            "partition": cluster["partition"], "qos": local["slurm"]["qos"], "account": local["slurm"].get("account")}

    tmp = Path(tempfile.mkdtemp(prefix="arbor-ab-"))
    tar = tmp / "arbor.tar"
    # LF bytes whatever this machine's git settings: with core.autocrlf=true, git archive on Windows writes CRLF
    # text files, and Arbor's shell scripts would break on the Linux node
    subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-C", str(clone), "archive", "--format=tar",
                    "-o", str(tar), commit], check=True)
    crlf, crlf_other = archive_crlf_files(tar)
    if crlf:
        sys.exit(f"arbor.tar has CRLF shell scripts, e.g. {crlf[:3]}: refusing to ship it")
    if crlf_other:
        print(f"note: {len(crlf_other)} non-script files are committed upstream with CRLF (shipped as committed), "
              f"e.g. {crlf_other[0]}")
    tar_sha = hashlib.sha256(tar.read_bytes()).hexdigest()

    cells = []
    for r in records:
        cell = render.cell_name(r["model"], r["target"]["key"])
        cdir, rcdir = run_dir / cell, f"{remote_run}/{cell}"
        cdir.mkdir(parents=True)
        container = render.container_name(run_id, r["model"], r["target"]["key"])
        env = render.cell_env(r, run_id, commit, mode, rcdir, container, minutes, site, f"{remote_run}/arbor.tar", tar_sha)
        (cdir / "cell.env").write_text(render.env_file(env), encoding="utf-8", newline="\n")
        acfg = render.arbor_config(r, run_id, commit, rcdir, container, minutes)
        (cdir / "arbor_config.json").write_text(json.dumps(acfg, indent=2) + "\n", encoding="utf-8", newline="\n")
        wall = render.job_minutes(mode, minutes, r)
        (cdir / "job.sbatch").write_text(render.sbatch(rcdir, f"ab-{cell}"[:40], wall, site), encoding="utf-8", newline="\n")
        for f in RUNNER_FILES:
            shutil.copyfile(REPO_ROOT / "scripts" / "runner" / f, cdir / f)
        cells.append({"model": r["model"], "target": r["target"]["key"], "epoch_id": r["epoch_id"], "cell": cell,
                      "remote_dir": rcdir, "container": container, "job_minutes": wall, "job_id": None})

    run_config = {"schema_version": 1, "run_id": run_id, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "request": req, "request_file": args.request.as_posix(), "commit": commit, "arbor_repo": "AMD-AGI/Arbor",
                  "arbor_tar_sha256": tar_sha, "mode": mode, "campaign_minutes": minutes, "remote_run_dir": remote_run,
                  "login_host": local["ssh"]["login_host"], "site": {k: v for k, v in site.items() if k != "gateway_env"},
                  "gateway_configured": bool(site["gateway_env"]),
                  "reference": req.get("reference"), "cells": cells, "submitted": False}
    rc_path = run_dir / "run_config.json"
    rc_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"run {run_id}: {len(cells)} cell(s), mode {mode}, Arbor {commit[:12]} (tar sha256 {tar_sha[:12]}), local {run_dir}")
    if args.dry_run:
        shutil.rmtree(tmp)
        print("dry run: nothing sent to the cluster")
        return 0

    host = local["ssh"]["login_host"]
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, "mkdir", "-p", remote_run], check=True)
    subprocess.run(["scp", "-q", "-r", str(tar), *[str(run_dir / c["cell"]) for c in cells], f"{host}:{remote_run}/"], check=True)
    shutil.rmtree(tmp)
    script = "set -e\n" + "".join(f"sbatch --parsable {c['remote_dir']}/job.sbatch\n" for c in cells)
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash", "-s"], input=script, capture_output=True, text=True)
    ids = [ln.split(";")[0] for ln in p.stdout.split()]
    for c, j in zip(cells, ids):
        c["job_id"] = j
    run_config["submitted"] = p.returncode == 0 and len(ids) == len(cells)
    run_config["submit_stderr"] = p.stderr[-2000:] or None
    rc_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"submitted: {[(c['cell'], c['job_id']) for c in cells]}" if run_config["submitted"] else f"sbatch failed: {p.stderr}")
    return 0 if run_config["submitted"] else 1


if __name__ == "__main__":
    sys.exit(main())
