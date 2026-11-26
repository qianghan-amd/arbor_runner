"""Run an Arbor campaign from an AgentX CI run: the first use case, one command.

  python scripts/run_from_ci.py --ci-run <InferenceX Actions run id or URL> --arbor-commit <sha> [options]

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
import json
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
    ap.add_argument("--ci-run", required=True, help="the InferenceX AgentX CI run: its Actions run id or URL (https://github.com/SemiAnalysisAI/InferenceX/actions/runs/<id>)")
    ap.add_argument("--arbor-commit", required=True)
    ap.add_argument("--model", help="matrix model key, when the CI run measured several models")
    ap.add_argument("--target", help="matrix target, e.g. c1_p90, when several match")
    ap.add_argument("--tp", type=int, help="TP of the CI point, when several match")
    ap.add_argument("--mode", default="campaign", choices=("campaign", "full", "campaign_smoke", "smoke", "bench"))
    ap.add_argument("--conc", type=int, nargs="+",
                    help="mode bench: the concurrencies to bench, one server for all (default: the CI point's)")
    ap.add_argument("--duration", type=int, help="mode bench: seconds of each AgentX run (default 600)")
    ap.add_argument("--campaign-minutes", type=int, default=480)
    ap.add_argument("--gsm8k", action="store_true", help="GSM8K in the measured phases (full mode)")
    ap.add_argument("--weights-path", default="auto", help="auto (shared hub when present), none (download), or a path")
    ap.add_argument("--exclude-nodes", nargs="*", default=[])
    ap.add_argument("--site", help="local.yaml site (default: local.yaml default_site)")
    ap.add_argument("--allow-other-hw", action="store_true", help="launch a CI run of another GPU architecture anyway")
    ap.add_argument("--allow-other-model", action="store_true",
                    help="launch a CI cell whose weights differ from the model's stored baselines anyway")
    ap.add_argument("--partition",help="Slurm partition for this run (e.g. meta64's interactive)")
    ap.add_argument("--nodelist", help="run on this node (Slurm --nodelist)")
    ap.add_argument("--hold-job", help="run on a free node of your own running hold job (its node agent; needs --nodelist)")
    ap.add_argument("--objective", help='what Arbor optimizes, as Arbor names it: "total throughput", "P50 interactivity", '
                                        '"P90 interactivity", "output throughput", or a bench key with a direction '
                                        '("tpot_p50:min"); default: the metric of the target')
    ap.add_argument("--goal", help="a sentence for Arbor's agents (config `goal`); default: one from the target")
    ap.add_argument("--arbor-standalone", action="store_true",
                    help="run Arbor in one process (ARBOR_STANDALONE=1) instead of under its launcher (the default)")
    ap.add_argument("--qos")
    ap.add_argument("--account")
    ap.add_argument("--name", help="request name (default ci<ID>-<model>-<target>-<sha8>)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--follow", action="store_true", help="wait for the job and print the report")
    args = ap.parse_args()
    if getattr(args, "ci_run", None):
        try:                               # a run id, or a run/job URL pasted from GitHub
            args.ci_run = ci_baseline.parse_ci_run(args.ci_run)
        except ValueError as e:
            sys.exit(str(e))

    local, matrix = config.load_local(), config.load_matrix()
    backend = backends.for_site(local, args.site)
    clone = (REPO_ROOT / local["arbor"]["clone"]).resolve()
    sha = arbor_sha(clone, args.arbor_commit)

    # 1. the cell, from the CI run's own points
    relay = config.local_view(args.site)["ssh"]["github_relay_host"]
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
    # the hardware first (one result JSON): an NVIDIA or gfx942 run is refused before any question about its cells
    arch = config.load_cluster(config.local_view(args.site).get("cluster", backend.site_name)).get("gpu_arch", "gfx950")
    first = next(a for a in arts if ci_baseline.BMK_NAME.match(a["name"]) and not a.get("expired"))
    hw = next(iter((github_relay.call(relay, [github_relay.artifact_json(make_baseline.INFX, first["id"])])[0].get("data")
                    or {}).values()), {}).get("hw")
    if ci_baseline.ci_arch(hw) != arch and not args.allow_other_hw:
        sys.exit(f"CI run {args.ci_run} ran on {hw} ({ci_baseline.ci_arch(hw)}); this site's GPUs are {arch}: its image "
                 "and recipe are not for them. Pick an MI355X/MI350X run (or --allow-other-hw)")
    try:
        model, target, tp = ci_baseline.choose_cell(points, matrix, args.model, args.target, args.tp)
    except ValueError as e:
        sys.exit(f"CI run {args.ci_run}: {e}")
    print(f"1. CI run {args.ci_run} ({run['head_branch']}, {run['created_at']}): cell {model}/{target}, tp{tp}")

    # 2. the starting point
    d, rec = make_baseline.draft(model, target, args.ci_run, tp, run_info=run, arts=arts, gpu_arch=arch,
                                 allow_other_hw=args.allow_other_hw, allow_other_model=args.allow_other_model)
    print(f"2. start point {rec['epoch_id']}: image {rec['fingerprint']['image']}@{(rec['fingerprint']['image_digest'] or '')[:19]}, "
          f"model rev {rec['launch']['model_rev'][:10]}; where each field came from: "
          f"{d.relative_to(REPO_ROOT).as_posix()}/review.md")

    # 3. weights
    weights = None if args.weights_path == "none" else args.weights_path
    if weights == "auto" and not backend.site().get("shared_hf_hub"):
        weights = None                     # this site has no shared hub: the job downloads at the pin
    if weights == "auto":
        org_name = rec["launch"]["model_id"].replace("/", "--")
        snap = f"{backend.site()['shared_hf_hub']}/models--{org_name}/snapshots/{rec['launch']['model_rev']}"
        complete, detail = backend.snapshot_complete(snap)
        weights = snap if complete else None
        if complete is False and "no snapshot" not in detail:
            print(f"   the shared hub's snapshot is incomplete ({detail}): the job downloads instead")
    print(f"3. weights: {weights or 'downloaded by the job at the pinned revision'}")
    # the image must still be pullable (by its digest, else its tag): a deleted nightly tag would fail on the node
    from arbor_ab import registry
    fp = rec["fingerprint"]
    avail, detail = registry.summarize(registry.check(relay, [(fp["image"], fp["image_digest"])])[0], fp["image_digest"])
    if avail is False:
        sys.exit(f"image {fp['image']}@{(fp['image_digest'] or '<no digest>')[:19]} cannot be pulled ({detail}): this CI "
                 "run's image is gone from the registry. Pick a newer run, or give a launch record with an available image")
    print(f"   image: {detail}")
    bench = rec["measured"]["bench_client_commit"]
    got = backend.ensure_bench_commits([bench]).get(bench)
    if got is not None and not got["ok"]:
        sys.exit(f"InferenceX {bench} (the CI's bench client) or a submodule of it could not be fetched into the "
                 f"cluster's InferenceX clone: {got}")
    if args.gsm8k and got is not None and got.get("benchmark_lib") is False:
        # found before any node time (meta64 153110, 2026-11-26: refused in prepare after a 6-min fetch)
        sys.exit(f"--gsm8k: InferenceX {bench[:10]} (this CI run's bench client) has the newer infx layout without "
                 "benchmarks/benchmark_lib.sh, and the runner's GSM8K needs it: run without --gsm8k")
    print(f"4. Arbor {sha}; InferenceX bench client {bench[:10]} {'present' if got else '(not checked by this backend)'}")

    # 5. request file, then submit
    name = args.name or f"ci{args.ci_run}-{model}-{target}-{sha[:8]}"
    # the launch record goes next to the request (both tracked), and the request names it: resubmitting the request
    # with submit_ab_test.py runs this launch again, not the stored baseline
    record = REPO_ROOT / "runs" / "requests" / f"{name}.record.json"
    write(record, json.dumps(rec, indent=2) + "\n")
    req = {"commit": sha, "mode": args.mode, "campaign_minutes": args.campaign_minutes, "ci_run": str(args.ci_run),
           "targets": [{"model": model, "target": target, "record": record.relative_to(REPO_ROOT).as_posix(),
                        **({"weights_path": weights} if weights else {})}]}
    if args.mode == "bench" and (args.conc or args.duration):
        req["bench"] = {**({"concurrencies": args.conc} if args.conc else {}),
                        **({"duration_s": args.duration} if args.duration else {})}
    elif args.conc or args.duration:
        sys.exit("--conc / --duration are for --mode bench")
    if args.gsm8k:
        req["gsm8k"] = True
    if args.site:
        req["site"] = args.site
    if args.exclude_nodes:
        req["exclude_nodes"] = args.exclude_nodes
    if args.qos:
        req["backend_options"] = {"qos": args.qos, **({"account": args.account} if args.account else {})}
    if args.partition:
        req.setdefault("backend_options", {})["partition"] = args.partition
    if args.nodelist:
        req.setdefault("backend_options", {})["nodelist"] = args.nodelist
    if args.objective:
        req["objective"] = args.objective
    if args.goal:
        req["goal"] = args.goal
    if args.arbor_standalone:
        req["arbor_standalone"] = True
    if args.hold_job:
        req.setdefault("backend_options", {})["hold_job"] = args.hold_job
    rf = REPO_ROOT / "runs" / "requests" / f"{name}.yaml"
    write(rf, f"# run_from_ci.py: InferenceX CI run {args.ci_run} ({run['display_title'][:100]})\n"
              f"# launch: {record.name} (from {d.relative_to(REPO_ROOT).as_posix()}/draft.json; review.md there says where "
              f"each field came from). Rerun: python scripts/submit_ab_test.py --request <this file>\n"
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
        sys.stdout.flush()   # our buffered lines first, then the report's (piped output was out of order, G3 2026-11-22)
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_report.py"), "--run-dir", str(run_dir)])
        return 0 if all(c["state"] == "SUCCEEDED" for c in doc["cells"]) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
