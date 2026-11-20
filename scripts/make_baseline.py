"""Build a new baseline from an InferenceX CI run (docs/make_baseline_design.md, simplified MVP, vLLM only).

  python scripts/make_baseline.py draft    --model M --target T --ci-run ID [--tp N]
      -> runs/drafts/<model>__<target>__ci<ID>/{draft.json, review.md, ci_bmk.json, ci_commands.txt}
         A person reads review.md and confirms (or edits draft.json for) every `copied`/`derived` row.
  python scripts/make_baseline.py measure  --draft DIR --arbor-commit SHA [--gsm8k] [--weights-path P] [--qos Q --account A]
      -> one `baseline` job (fetch, prepare, before) through the execution backend; prints the run dir
  python scripts/make_baseline.py finalize --draft DIR --run-dir runs/<run_id> [--replace]
      -> metrics from the run; gate vs CI (structural + numeric, matrix tolerance); stored in baselines/ only on `pass`
Each step only reads and writes files; stop between them at any time.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

from arbor_ab import REPO_ROOT, agentx, backends, baseline, ci_baseline, config, gate, github_relay, registry, remote, report

INFX = "SemiAnalysisAI/InferenceX"
HF_COMMITS = r'''
import urllib.request
c = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{PAYLOAD['name']}/commits/main", timeout=60))
print(json.dumps([{"id": x["id"], "date": x["date"]} for x in c]))
'''


def write(p, text):
    p.write_text(text, encoding="utf-8", newline="\n")


def cmd_draft(args):
    local, matrix = config.load_local(), config.load_matrix()
    relay = local["ssh"]["github_relay_host"]
    tgt = config.target_spec(matrix, args.model, args.target)
    tp = args.tp or (matrix["models"][args.model]["targets"][args.target].get("expected_host") or {}).get("tp")
    run, arts = github_relay.call(relay, [github_relay.get(f"/repos/{INFX}/actions/runs/{args.ci_run}"),
                                          github_relay.artifacts(INFX, args.ci_run)])
    if run.get("http") != 200 or not arts.get("data"):
        sys.exit(f"CI run {args.ci_run}: run http {run.get('http')}, {len(arts.get('data') or [])} artifacts (expired?)")
    run = run["data"]
    cands = [a for a in gate.bmk_candidates(arts["data"], tp, tgt["conc"]) if not a["expired"]]
    if not cands:
        sys.exit(f"no live bmk_* artifact for tp{tp} conc{tgt['conc']} in run {args.ci_run}")
    got = github_relay.call(relay, [github_relay.artifact_json(INFX, a["id"]) for a in cands])
    best = max(((a, next(iter((g.get("data") or {}).values()), {})) for a, g in zip(cands, got)),
               key=lambda x: x[1].get("num_requests_successful") or 0)
    bmk_art, bmk = best
    agentic = "agentic_" + bmk_art["name"][len("bmk_agentic_"):]
    agentic_art = next((a for a in arts["data"] if a["name"] == agentic and not a["expired"]), None)
    if not agentic_art:
        sys.exit(f"no live {agentic} artifact (needed for the server command)")
    texts = github_relay.call(relay, [github_relay.artifact_text(
        INFX, agentic_art["id"], ["vllm_command.txt", "benchmark_command.txt", "server.log"], grep={"server.log": "snapshots/"})])[0]
    files = texts.get("data") or {}
    if "vllm_command.txt" not in files:
        sys.exit(f"{agentic} has no vllm_command.txt (SGLang CI runs are not supported in this MVP)")
    cmd = ci_baseline.parse_vllm_command(files["vllm_command.txt"])
    model_rev = ci_baseline.model_rev_from_log(files.get("server.log", ""), cmd["model"])
    ds = bmk.get("dataset") or {}
    commits = remote.run_python(relay, HF_COMMITS, {"name": ds["hf_dataset_name"]})
    corpus_rev = ci_baseline.corpus_rev_at(commits, run["created_at"])
    image, digest = ci_baseline.split_image(bmk.get("image"))
    if not digest:   # the tag's digest now; review.md marks it `derived`
        reg = registry.check(relay, [(bmk["image"], None)])[0]
        digest = reg.get("tag_digest")
    tpl = ci_baseline.template_record(args.model, args.target)
    rec, rows = ci_baseline.build_draft(args.model, tgt, run, bmk_art["name"], bmk, cmd, model_rev, ds["hf_dataset_name"],
                                        corpus_rev, digest, tpl)
    errs = baseline.validation_errors(rec)
    if errs:
        sys.exit("draft is not schema-valid:\n  " + "\n  ".join(errs))
    notes = [f"bmk dataset entries: {ds.get('num_dataset_entries')} (the runner's offline corpus load must print the same "
             "number in prepare)", f"CI aiperf command: {files.get('benchmark_command.txt', '')[:300]}"]
    d = REPO_ROOT / "runs" / "drafts" / f"{args.model}__{args.target}__ci{args.ci_run}"
    d.mkdir(parents=True, exist_ok=True)
    write(d / "draft.json", json.dumps(rec, indent=2) + "\n")
    write(d / "ci_bmk.json", json.dumps({"artifact": bmk_art, "json": bmk}, indent=1) + "\n")
    write(d / "ci_commands.txt", files["vllm_command.txt"] + "\n\n" + files.get("benchmark_command.txt", "") + "\n")
    write(d / "review.md", ci_baseline.review_md(rec, rows, run, bmk_art["name"], notes))
    print(f"draft: {d.relative_to(REPO_ROOT).as_posix()} (epoch {rec['epoch_id']}); read review.md next")
    return 0


def cmd_measure(args):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import submit_ab_test
    d = Path(args.draft)
    rec = json.loads((d / "draft.json").read_text(encoding="utf-8"))
    local = config.load_local()
    req = {"commit": args.arbor_commit, "mode": "baseline", "gsm8k": args.gsm8k,
           "targets": [{"model": rec["model"], "target": rec["target"]["key"], "weights_path": args.weights_path}]}
    if args.qos:
        req["backend_options"] = {"qos": args.qos, **({"account": args.account} if args.account else {})}
    rc = submit_ab_test.submit(req, f"baseline-{d.name}", backends.get(local, config.load_cluster()),
                               (REPO_ROOT / local["arbor"]["clone"]).resolve(), records=[rec])
    if rc["submitted"]:
        print(f"then: python scripts/poll_status.py --run-dir runs/{rc['run_id']}; "
              f"python scripts/make_baseline.py finalize --draft {d.as_posix()} --run-dir runs/{rc['run_id']}")
    return 0 if rc["submitted"] else 1


def cmd_finalize(args):
    d, run_dir = Path(args.draft), Path(args.run_dir)
    rec = json.loads((d / "draft.json").read_text(encoding="utf-8"))
    bmk = json.loads((d / "ci_bmk.json").read_text(encoding="utf-8"))
    cell = run_dir / f"{rec['model']}__{rec['target']['key']}" / "before"
    if not (cell / "agg.json").exists():
        sys.exit(f"{cell}/agg.json not found: poll_status.py fetches it once the job's `before` phase is done")
    lines = (cell / "gsm8k.log").read_text(encoding="utf-8", errors="replace").splitlines() if (cell / "gsm8k.log").exists() else []
    rec["metrics"] = report.phase_metrics(json.loads((cell / "agg.json").read_text(encoding="utf-8")), lines)
    ph = json.loads((cell / "phase.json").read_text(encoding="utf-8")) if (cell / "phase.json").exists() else {}
    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    rec["measured"].update(date=ph.get("ended_at"), cluster=rc.get("backend"), node=ph.get("node"), job=ph.get("job"),
                           replays=1, duration_s=ph.get("duration_s"), arbor_commit=rc["commit"])
    rec["ceiling"] = {"metrics": rec["metrics"], "source": {"kind": "pristine_before", "ref": f"runs/{rc['run_id']}",
                                                            "date": ph.get("ended_at"), "cluster": rc.get("backend"),
                                                            "node": ph.get("node"), "job": ph.get("job")}}
    rec["artifacts"] = {"bench_output": f"runs/{rc['run_id']}/{cell.parent.name}/before/bench_output.json",
                        "agg_json": f"runs/{rc['run_id']}/{cell.parent.name}/before/agg.json",
                        "gsm8k_log": f"runs/{rc['run_id']}/{cell.parent.name}/before/gsm8k.log" if lines else None}
    t = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    diffs, skipped = gate.structural_diff(rec, bmk["json"])
    tol = config.tolerance_pct(config.load_matrix(), rec["model"], rec["target"]["key"])
    n_status, ci_v, ours, delta = gate.numeric(rec, bmk["json"], tol, rec["fingerprint"]["ci_commit"])
    v = rec["verification"]
    v["structural"] = {"status": "fail" if diffs else "pass", "checked_at": t,
                       "detail": "; ".join(diffs) or f"matches CI's bmk record ({bmk['artifact']['name']})"}
    v["numeric"] = {"status": n_status, "checked_at": t, "ci_artifact": f"{INFX}/actions/artifacts/{bmk['artifact']['id']}",
                    "ci_value": ci_v, "measured_value": ours, "delta_pct": round(delta, 3) if delta is not None else None,
                    "tolerance_pct": tol}
    v["ci_verified"], v["unverified_against_ci"] = gate.flags(v["structural"]["status"], n_status)
    rec["epoch_id"] = baseline.derive_epoch_id(rec)
    print(f"gate: structural {v['structural']['status']}, numeric {n_status} (ours {ours} vs CI {ci_v}, delta {v['numeric']['delta_pct']}%, tol {tol}%)")
    write(d / "finalized.json", json.dumps(rec, indent=2) + "\n")
    if not v["ci_verified"]:
        print(f"not stored (design.md 3.4: halt for human review); the full record is {d.as_posix()}/finalized.json")
        return 1
    existing = baseline.cell_dir(rec["model"], rec["target"]["key"]) / f"{rec['epoch_id']}.json"
    if existing.exists() and not args.replace:
        print(f"epoch {rec['epoch_id']} already stored ({existing}); rerun with --replace to overwrite it")
        return 1
    print(f"stored: {baseline.store(rec)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draft")
    p.add_argument("--model", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--ci-run", required=True)
    p.add_argument("--tp", type=int)
    p = sub.add_parser("measure")
    p.add_argument("--draft", required=True)
    p.add_argument("--arbor-commit", required=True, help="the Arbor whose launch code renders the server command")
    p.add_argument("--gsm8k", action="store_true", help="also run GSM8K (off by default)")
    p.add_argument("--weights-path")
    p.add_argument("--qos")
    p.add_argument("--account")
    p = sub.add_parser("finalize")
    p.add_argument("--draft", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--replace", action="store_true")
    args = ap.parse_args()
    return {"draft": cmd_draft, "measure": cmd_measure, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
