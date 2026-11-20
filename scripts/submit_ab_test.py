"""Submit a PR test: request YAML -> run_id -> runs/<run_id>/ -> one job per cell -> exit (design.md sections 4-6).

Usage:
  python scripts/submit_ab_test.py --request runs/requests/<name>.yaml [--dry-run]
Request:
  commit: <Arbor commit SHA>             # identified by commit, not PR number (design.md section 4)
  mode: campaign                         # default. campaign = fetch + prepare + the Arbor campaign, no measurement;
                                         # full = before + campaign + after (3600 s AgentX each side; the A/B);
                                         # smoke = 600 s AgentX, no campaign; campaign_smoke = a short campaign
  gsm8k: true                            # optional, default false: GSM8K in the measured phases (else "skipped")
  campaign_minutes: 480                  # campaign modes (Arbor's time_budget_minutes; the runner stops Arbor then)
  exclude_nodes: [crsuse2-m2m-v3-177]    # optional: hosts not to use (e.g. left dirty by another user)
  backend_options: { qos: amd-burst-qos, account: amd-hyperloom }   # optional, backend-specific (crusoe_slurm)
  targets:
    - { model: dsv41flash, target: c1_p90 }
    - { model: glm52, target: c1_p90, weights_path: /shared_nfs/<...>/snapshots/<rev> }   # optional, see README
Where the jobs run is the execution backend's business (local.yaml `backend`, default crusoe_slurm; design.md 5.1).
Stateless: everything it decides is written to runs/<run_id>/run_config.json; it never waits for results.
--dry-run renders every file locally and touches no cluster. A cell that is not runnable (render.check_runnable)
refuses the whole submission, with the reasons.
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

from arbor_ab import REPO_ROOT, backends, baseline, config, render

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
    bad, other = [], []
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if m.isfile() and m.size < 5_000_000:
                b = t.extractfile(m).read()
                if b"\r\n" in b and b"\0" not in b:
                    (bad if is_shell(m.name, b) else other).append(m.name)
    return bad, other


def arbor_archive(clone, commit, dest):
    """LF bytes whatever this machine's git settings (core.autocrlf=true on Windows would write CRLF)."""
    subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-C", str(clone), "archive", "--format=tar",
                    "-o", str(dest), commit], check=True)
    bad, other = archive_crlf_files(dest)
    if bad:
        sys.exit(f"arbor.tar has CRLF shell scripts, e.g. {bad[:3]}: refusing to ship it")
    if other:
        print(f"note: {len(other)} non-script files are committed upstream with CRLF (shipped as committed), e.g. {other[0]}")
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def write(p, text):
    p.write_text(text, encoding="utf-8", newline="\n")


def submit(req, request_file, backend, clone, gateway_needed=True, dry_run=False, now=None, records=None):
    """Render runs/<run_id>/ and (unless dry_run) start one job per cell through `backend`. Returns run_config."""
    mode = req.get("mode", "campaign")   # default: campaign only (user decision 2026-11-20)
    # GSM8K is off unless the request asks for it (user decision 2026-11-20); `skip_gsm8k: true` (older requests) still wins
    skip_gsm8k = not req.get("gsm8k", False) or bool(req.get("skip_gsm8k"))
    if mode not in render.MODES:
        sys.exit(f"mode must be one of {render.MODES}")
    minutes = int(req.get("campaign_minutes", 1440)) if mode in render.CAMPAIGN_MODES else 0
    try:
        commit = subprocess.run(["git", "-C", str(clone), "rev-parse", "--verify", f"{req['commit']}^{{commit}}"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        sys.exit(f"commit {req['commit']} not found in {clone} (git fetch it first)")
    site = backend.site()
    if mode in render.CAMPAIGN_MODES and gateway_needed and not site.get("gateway_env"):
        sys.exit("full mode needs local.yaml arbor.gateway_env (the LLM gateway for Arbor's agents; setup_remote.py)")

    given, records, refusals = records, [], []   # records: measure these (e.g. a make_baseline.py draft), not the stored ones
    for i, t in enumerate(req["targets"]):
        r = given[i] if given else baseline.load_latest(t["model"], t["target"])
        why = ["no baseline record"] if r is None else render.check_runnable(r)
        if t.get("weights_path") and not str(t["weights_path"]).startswith("/"):
            why.append(f"weights_path {t['weights_path']!r} must be an absolute path on the node")
        if why:
            refusals.append(f"{t['model']}/{t['target']}: " + "; ".join(why))
        records.append(r)
    if refusals:
        sys.exit("not submitted, cells not runnable:\n  " + "\n  ".join(refusals))

    run_id = f"{(now or dt.datetime.now(dt.timezone.utc)):%Y%m%d-%H%M%S}_{commit[:8]}_{Path(request_file).stem}"
    run_dir, remote_run = REPO_ROOT / "runs" / run_id, backend.run_dir(run_id)
    tmp = Path(tempfile.mkdtemp(prefix="arbor-ab-"))
    tar = tmp / "arbor.tar"
    tar_sha = arbor_archive(clone, commit, tar)

    cells = []
    for t, r in zip(req["targets"], records):
        cell = render.cell_name(r["model"], r["target"]["key"])
        cdir, rcdir = run_dir / cell, f"{remote_run}/{cell}"
        cdir.mkdir(parents=True)
        container = render.container_name(run_id, r["model"], r["target"]["key"])
        env = render.cell_env(r, run_id, commit, mode, rcdir, container, minutes, site, f"{remote_run}/arbor.tar", tar_sha,
                              weights_path=t.get("weights_path"), skip_gsm8k=skip_gsm8k)
        write(cdir / "cell.env", render.env_file(env))
        write(cdir / "arbor_config.json", json.dumps(render.arbor_config(r, run_id, commit, rcdir, container, minutes), indent=2) + "\n")
        wall = render.job_minutes(mode, minutes, r)
        jname, jtext = backend.job_script(rcdir, f"ab-{cell}"[:40], wall, exclude=tuple(req.get("exclude_nodes") or ()),
                                          options=req.get("backend_options"))
        write(cdir / jname, jtext)
        for f in RUNNER_FILES:
            shutil.copyfile(REPO_ROOT / "scripts" / "runner" / f, cdir / f)
        cells.append({"model": r["model"], "target": r["target"]["key"], "epoch_id": r["epoch_id"], "cell": cell,
                      "remote_dir": rcdir, "container": container, "job_minutes": wall, "job_id": None,
                      "weights_path": t.get("weights_path")})

    run_config = {"schema_version": 1, "run_id": run_id, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "request": req, "request_file": Path(request_file).as_posix(), "commit": commit, "arbor_repo": "AMD-AGI/Arbor",
                  "arbor_tar_sha256": tar_sha, "mode": mode, "campaign_minutes": minutes, "backend": backend.name,
                  "remote_run_dir": remote_run, "site": {k: v for k, v in site.items() if k != "gateway_env"},
                  "gateway_configured": bool(site.get("gateway_env")), "reference": req.get("reference"),
                  "exclude_nodes": list(req.get("exclude_nodes") or []), "backend_options": req.get("backend_options"),
                  "skip_gsm8k": skip_gsm8k,
                  "cells": cells, "submitted": False}
    rc_path = run_dir / "run_config.json"
    write(rc_path, json.dumps(run_config, indent=2) + "\n")
    print(f"run {run_id}: {len(cells)} cell(s), mode {mode}, Arbor {commit[:12]} (tar sha256 {tar_sha[:12]}), "
          f"backend {backend.name}, local {run_dir}")
    if dry_run:
        shutil.rmtree(tmp)
        print("dry run: nothing sent")
        return run_config

    backend.stage(run_id, run_dir, cells, [tar])
    shutil.rmtree(tmp)
    ids, err = backend.submit(cells)
    for c, j in zip(cells, ids):
        c["job_id"] = j
    run_config["submitted"] = all(ids)
    run_config["submit_stderr"] = err
    write(rc_path, json.dumps(run_config, indent=2) + "\n")
    print(f"submitted: {[(c['cell'], c['job_id']) for c in cells]}" if run_config["submitted"] else f"submit failed: {err}")
    return run_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    local = config.load_local()
    backend = backends.get(local, config.load_cluster())
    req = yaml.safe_load(args.request.read_text(encoding="utf-8"))
    rc = submit(req, args.request, backend, (REPO_ROOT / local["arbor"]["clone"]).resolve(), dry_run=args.dry_run)
    return 0 if args.dry_run or rc["submitted"] else 1


if __name__ == "__main__":
    sys.exit(main())
