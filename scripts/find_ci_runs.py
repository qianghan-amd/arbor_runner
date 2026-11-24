"""Find recent InferenceX AgentX CI runs ("Run Sweep") that measured our models, and say which this runner can start.

Usage: python scripts/find_ci_runs.py [--pages 2] [--model glm52] [--site crusoe] [--dry-run] [--json out.json]
- Lists each successful Run Sweep run with AgentX results of a matrix model: date, run id, hardware, framework, our cells
  (model, conc), branch, image; and whether this site can run it (gfx arch, framework) or why not.
- --dry-run: for every runnable run, `run_from_ci.py --dry-run` on its first cell, so a CI format change shows up here
  before anyone launches (2026-11-23 scan found three: kv_offload_backend as an object, a glm5.2 cell serving
  GLM-5.3, the tokenspeed framework). Exit 1 when a dry run fails for a reason that is not a known refusal.
Reads GitHub through the relay (config ssh.github_relay_host): no token on this machine.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbor_ab import REPO_ROOT, backends, ci_baseline, config, github_relay as gh  # noqa: E402

REPO = "SemiAnalysisAI/InferenceX"
WORKFLOW = 214438218            # "Run Sweep" (the workflow every AgentX CI point comes from)
# refusals the tool gives on purpose: a run that ends in one of these is reported, not counted as a failure
KNOWN_REFUSALS = ("only ('vllm', 'sglang') are supported", "ran on", "another model under the same CI name",
                  "has no live AgentX artifacts", "is gone from the registry", "no live bmk_", "expired")


def prefixes(matrix):
    out = {}
    for model, spec in matrix["models"].items():
        p = spec.get("infmax_prefix", model)
        for x in (p if isinstance(p, list) else [p]):
            out[x] = model
    return out


def scan(relay, pages, matrix, only_model=None):
    pre = prefixes(matrix)
    concs = {m: {t["conc"] for t in s["targets"].values()} for m, s in matrix["models"].items()}
    runs = []
    for page in range(1, pages + 1):
        d = gh.call(relay, [gh.get(f"/repos/{REPO}/actions/workflows/{WORKFLOW}/runs?per_page=100&page={page}"
                                   "&status=success")])[0].get("data") or {}
        runs += d.get("workflow_runs") or []
        if len(d.get("workflow_runs") or []) < 100:
            break
    found = []
    for i in range(0, len(runs), 40):
        batch = runs[i:i + 40]
        res = gh.call(relay, [gh.artifacts(REPO, r["id"]) for r in batch], timeout=1800)
        for r, a in zip(batch, res):
            arts = a.get("data") or []
            pts = [p for p in ci_baseline.ci_points(arts) if p[0] in pre and p[2] in concs[pre[p[0]]]
                   and (only_model is None or pre[p[0]] == only_model)]
            if pts:
                bmk = next(x for x in arts if ci_baseline.BMK_NAME.match(x["name"])
                           and ci_baseline.BMK_NAME.match(x["name"]).group(1) in pre and not x.get("expired"))
                found.append({"id": r["id"], "date": r["created_at"][:10], "branch": r["head_branch"],
                              "cells": sorted({(pre[p[0]], p[2]) for p in pts}), "bmk_id": bmk["id"]})
    res = gh.call(relay, [gh.artifact_json(REPO, c["bmk_id"]) for c in found], timeout=1800) if found else []
    for c, r in zip(found, res):
        j = next(iter((r.get("data") or {}).values()), {})
        c.update(hw=j.get("hw"), framework=j.get("framework"), image=j.get("image"), served=j.get("model"))
    return sorted(found, key=lambda c: c["date"], reverse=True), len(runs)


def verdict(c, arch):
    if ci_baseline.ci_arch(c.get("hw")) != arch:
        return f"no: {c.get('hw')} is not {arch}"
    if c.get("framework") not in ci_baseline.FRAMEWORKS:
        return f"no: framework {c.get('framework')}"
    known = ci_baseline.known_model_ids(c["cells"][0][0])
    if known and c.get("served") not in known:
        return f"other model: serves {c.get('served')} (--allow-other-model)"
    return "yes"


def dry_run(c, site):
    model, conc = c["cells"][0]
    target = next(t for t, v in config.load_matrix()["models"][model]["targets"].items() if v["conc"] == conc)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_from_ci.py"), "--ci-run", str(c["id"]), "--model", model,
           "--target", target, "--arbor-commit", "HEAD", "--dry-run", "--name", f"scan-{c['id']}-{target}"]
    if site:
        cmd += ["--site", site]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    out = (p.stdout + p.stderr).strip().splitlines()
    # a check, not a launch: drop the request, record and local run dir the dry run wrote (the draft stays in runs/drafts)
    name = f"scan-{c['id']}-{target}"
    for f in [*(REPO_ROOT / "runs" / "requests").glob(f"{name}.*"), *(REPO_ROOT / "runs").glob(f"*_{name}")]:
        shutil.rmtree(f) if f.is_dir() else f.unlink()
    if p.returncode == 0 and any("dry run: nothing sent" in x for x in out):
        return "ok", ""
    last = next((x for x in reversed(out) if x.strip()), "")
    return ("refused" if any(k in last for k in KNOWN_REFUSALS) else "FAILED"), last[:240]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=2, help="pages of 100 successful runs, newest first (default 2)")
    ap.add_argument("--model", help="matrix model key: only this model")
    ap.add_argument("--site", help="local.yaml site whose GPU arch decides 'runnable' (default: default_site)")
    ap.add_argument("--dry-run", action="store_true", help="run_from_ci --dry-run on each runnable run")
    ap.add_argument("--json", type=Path, help="write the list here")
    args = ap.parse_args()
    local, matrix = config.load_local(), config.load_matrix()
    backend = backends.for_site(local, args.site)
    arch = config.load_cluster(config.local_view(args.site).get("cluster", backend.site_name)).get("gpu_arch", "gfx950")
    relay = config.local_view(args.site)["ssh"]["github_relay_host"]
    found, n = scan(relay, args.pages, matrix, args.model)
    print(f"{n} successful Run Sweep runs scanned; {len(found)} measured our models", flush=True)
    failed = 0
    for c in found:
        c["runnable"] = verdict(c, arch)
        line = (f"{c['date']} {c['id']} {(c.get('hw') or '?').split(':')[-1]:<16} {c.get('framework') or '?':<10} "
                f"{','.join(f'{m}/c{k}' for m, k in c['cells'])[:48]:<48} {c['runnable']}")
        if args.dry_run and c["runnable"] == "yes":
            c["dry_run"], c["dry_run_detail"] = dry_run(c, args.site)
            failed += c["dry_run"] == "FAILED"
            line += f" | dry run {c['dry_run']}" + (f": {c['dry_run_detail']}" if c["dry_run"] != "ok" else "")
        print(line, flush=True)
    if args.json:
        args.json.write_text(json.dumps(found, indent=1, default=list) + "\n", encoding="utf-8", newline="\n")
    if failed:
        print(f"{failed} runnable CI run(s) failed the dry run for an unknown reason: a CI format change? "
              "Read the detail above; add a parser fix and a fixture (tests/test_ci_formats.py).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
