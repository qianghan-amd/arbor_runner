"""Run an Arbor campaign from an AgentX CI run: the first use case, one command.

  python scripts/run_from_ci.py --ci-run <InferenceX Actions run id> --arbor-commit <sha> [options]

What it does, in order (each step prints what it decided):
  1. reads the CI run's AgentX points (bmk_agentic_* artifacts) and picks the cell: model, target (by concurrency), TP.
     A run with several matching points stops and lists them: choose with --target / --tp.
  2. builds the starting point from that CI run (make_baseline.py draft: image + digest, server flags, model and corpus
     revisions exact from the CI; server env copied from the same model's stored baseline, see review.md).
  3. weights: uses the pinned snapshot in the shared HF hub when it is there (checked against the pin on the node),
     else the job downloads it.
  4. fetches the Arbor commit into the local clone if it is missing.
  5. writes runs/requests/<name>.yaml (the run can be repeated from it) and submits one job (submit_ab_test.submit).
Default: mode campaign (no before/after measurement), 480 min, no GSM8K, the QOS in local.yaml. --follow waits and
reports (run_pr_test.py); --dry-run renders everything and submits nothing.
"""
import argparse
import subprocess
import sys

import yaml

from arbor_ab import REPO_ROOT, backends, ci_baseline, config, github_relay

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import make_baseline  # noqa: E402
import run_pr_test  # noqa: E402
import submit_ab_test  # noqa: E402


def write(p, text):
    p.write_text(text, encoding="utf-8", newline="\n")


def arbor_sha(clone, commit):
    """The full SHA of `commit` in the clone; `git fetch origin` once when it is missing."""
    def resolve():
        p = subprocess.run(["git", "-C", str(clone), "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
                           capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    sha = resolve()
    if not sha:
        print(f"Arbor {commit} not in {clone}: git fetch origin")
        subprocess.run(["git", "-C", str(clone), "fetch", "-q", "origin"], check=True)
        sha = resolve()
    if not sha:
        sys.exit(f"Arbor commit {commit} not found in {clone} even after git fetch")
    return sha


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ci-run", required=True, help="InferenceX GitHub Actions run id of the AgentX sweep")
    ap.add_argument("--arbor-commit", required=True)
    ap.add_argument("--model", help="matrix model key, when the CI run measured several models")
    ap.add_argument("--target", help="matrix target, e.g. c1_p90, when several match")
    ap.add_argument("--tp", type=int, help="TP of the CI point, when several match")
    ap.add_argument("--mode", default="campaign", choices=("campaign", "full", "campaign_smoke"))
    ap.add_argument("--campaign-minutes", type=int, default=480)
    ap.add_argument("--gsm8k", action="store_true", help="GSM8K in the measured phases (full mode)")
    ap.add_argument("--weights-path", default="auto", help="auto (shared hub when present), none (download), or a path")
    ap.add_argument("--exclude-nodes", nargs="*", default=[])
    ap.add_argument("--qos")
    ap.add_argument("--account")
    ap.add_argument("--name", help="request name (default ci<ID>-<model>-<target>-<sha8>)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--follow", action="store_true", help="wait for the job and print the report")
    args = ap.parse_args()

    local, matrix = config.load_local(), config.load_matrix()
    backend = backends.get(local, config.load_cluster())
    clone = (REPO_ROOT / local["arbor"]["clone"]).resolve()
    sha = arbor_sha(clone, args.arbor_commit)

    # 1. the cell, from the CI run's own points
    relay = local["ssh"]["github_relay_host"]
    run, arts = github_relay.call(relay, [github_relay.get(f"/repos/{make_baseline.INFX}/actions/runs/{args.ci_run}"),
                                          github_relay.artifacts(make_baseline.INFX, args.ci_run)])
    if run.get("http") != 200:
        sys.exit(f"CI run {args.ci_run}: http {run.get('http')}")
    run, arts = run["data"], arts.get("data") or []
    points = ci_baseline.ci_points(arts)
    if not points:
        sys.exit(f"CI run {args.ci_run} ({run['display_title'][:80]}) has no live AgentX artifacts (bmk_agentic_*): "
                 "the run reused older results, skipped the AgentX jobs, or its artifacts expired. Use the run that "
                 "produced the numbers (same branch, earlier run).")
    try:
        model, target, tp = ci_baseline.choose_cell(points, matrix, args.model, args.target, args.tp)
    except ValueError as e:
        sys.exit(f"CI run {args.ci_run}: {e}")
    print(f"1. CI run {args.ci_run} ({run['head_branch']}, {run['created_at']}): cell {model}/{target}, tp{tp}")

    # 2. the starting point
    d, rec = make_baseline.draft(model, target, args.ci_run, tp, run_info=run, arts=arts)
    print(f"2. start point {rec['epoch_id']}: image {rec['fingerprint']['image']}@{(rec['fingerprint']['image_digest'] or '')[:19]}, "
          f"model rev {rec['launch']['model_rev'][:10]}; where each field came from: "
          f"{d.relative_to(REPO_ROOT).as_posix()}/review.md")

    # 3. weights
    weights = None if args.weights_path == "none" else args.weights_path
    if weights == "auto":
        org_name = rec["launch"]["model_id"].replace("/", "--")
        snap = f"{backend.site()['shared_hf_hub']}/models--{org_name}/snapshots/{rec['launch']['model_rev']}"
        have = backend.files_exist([f"{snap}/config.json", f"{snap}/model.safetensors.index.json"])
        weights = snap if have and all(have.values()) else None
    print(f"3. weights: {weights or 'downloaded by the job at the pinned revision'}")
    bench = rec["measured"]["bench_client_commit"]
    got = backend.ensure_bench_commits([bench]).get(bench)
    if got is not None and not got["ok"]:
        sys.exit(f"InferenceX {bench} (the CI's bench client) or a submodule of it could not be fetched into the "
                 f"cluster's InferenceX clone: {got}")
    print(f"4. Arbor {sha}; InferenceX bench client {bench[:10]} {'present' if got else '(not checked by this backend)'}")

    # 5. request file, then submit
    name = args.name or f"ci{args.ci_run}-{model}-{target}-{sha[:8]}"
    req = {"commit": sha, "mode": args.mode, "campaign_minutes": args.campaign_minutes, "ci_run": str(args.ci_run),
           "targets": [{"model": model, "target": target, **({"weights_path": weights} if weights else {})}]}
    if args.gsm8k:
        req["gsm8k"] = True
    if args.exclude_nodes:
        req["exclude_nodes"] = args.exclude_nodes
    if args.qos:
        req["backend_options"] = {"qos": args.qos, **({"account": args.account} if args.account else {})}
    rf = REPO_ROOT / "runs" / "requests" / f"{name}.yaml"
    write(rf, f"# run_from_ci.py: InferenceX CI run {args.ci_run} ({run['display_title'][:100]})\n"
              f"# start point: {d.relative_to(REPO_ROOT).as_posix()}/draft.json (rebuild it with make_baseline.py draft "
              f"--model {model} --target {target} --ci-run {args.ci_run} --tp {tp})\n"
              + yaml.safe_dump(req, sort_keys=False))
    print(f"5. request {rf.relative_to(REPO_ROOT).as_posix()}")
    rc = submit_ab_test.submit(req, rf, backend, clone, dry_run=args.dry_run, records=[rec])
    if args.dry_run:
        return 0
    if not rc["submitted"]:
        return 1
    run_dir = REPO_ROOT / "runs" / rc["run_id"]
    print(f"follow: python scripts/run_pr_test.py --run-dir {run_dir.relative_to(REPO_ROOT).as_posix()}")
    if args.follow:
        doc = run_pr_test.follow(run_dir, backend, 10)
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_report.py"), "--run-dir", str(run_dir)])
        return 0 if all(c["state"] == "SUCCEEDED" for c in doc["cells"]) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
