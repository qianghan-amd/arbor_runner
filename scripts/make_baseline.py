"""Build a new baseline from an InferenceX CI run (docs/make_baseline_design.md, simplified MVP; vLLM and SGLang).
Baselines are measured on a baseline site only (config/cluster/<name>.yaml `baselines: true`: Crusoe); `measure` refuses
any other site.

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
# server log lines read: the model/draft snapshot paths and the server's own argument dump (ci_baseline.check_against_server)
LOG_GREP = r"snapshots/|non-default args|server_args="
HF_COMMITS = r'''
import urllib.request
c = json.load(urllib.request.urlopen(f"https://huggingface.co/api/{PAYLOAD.get('kind', 'datasets')}/{PAYLOAD['name']}/commits/main",
                                     timeout=60))
print(json.dumps([{"id": x["id"], "date": x["date"]} for x in c]))
'''


def write(p, text):
    p.write_text(text, encoding="utf-8", newline="\n")


def cmd_draft(args):
    d, rec = draft(args.model, args.target, args.ci_run, args.tp)
    print(f"draft: {d.relative_to(REPO_ROOT).as_posix()} (epoch {rec['epoch_id']}); read review.md next")
    return 0


def draft(model, target, ci_run, tp=None, run_info=None, arts=None, gpu_arch="gfx950", allow_other_hw=False,
          allow_other_model=False):
    """Write runs/drafts/<model>__<target>__ci<ID>/ and return (dir, record). run_info/arts: already fetched."""
    local, matrix = config.load_local(), config.load_matrix()
    relay = config.local_view()["ssh"]["github_relay_host"]
    tgt = config.target_spec(matrix, model, target)
    tp = tp or (matrix["models"][model]["targets"][target].get("expected_host") or {}).get("tp")
    if run_info is None or arts is None:
        run, arts = github_relay.call(relay, [github_relay.get(f"/repos/{INFX}/actions/runs/{ci_run}"),
                                              github_relay.artifacts(INFX, ci_run)])
        if run.get("http") != 200 or not arts.get("data"):
            sys.exit(f"CI run {ci_run}: run http {run.get('http')}, {len(arts.get('data') or [])} artifacts (expired?)")
        run, arts = run["data"], arts["data"]
    else:
        run = run_info
    cands = [a for a in gate.bmk_candidates(arts, tp, tgt["conc"]) if not a["expired"]]
    if not cands:
        sys.exit(f"no live bmk_* artifact for tp{tp} conc{tgt['conc']} in run {ci_run}")
    got = github_relay.call(relay, [github_relay.artifact_json(INFX, a["id"]) for a in cands])
    best = max(((a, next(iter((g.get("data") or {}).values()), {})) for a, g in zip(cands, got)),
               key=lambda x: x[1].get("num_requests_successful") or 0)
    bmk_art, bmk = best
    if ci_baseline.ci_arch(bmk.get("hw")) != gpu_arch and not allow_other_hw:
        sys.exit(f"CI run {ci_run} ran on {bmk.get('hw')} ({ci_baseline.ci_arch(bmk.get('hw'))}); this site's GPUs are "
                 f"{gpu_arch}: its image and recipe are not for them. Pick an MI355X/MI350X run (or --allow-other-hw)")
    known = ci_baseline.known_model_ids(model)
    if not known:
        print(f"note: no stored baseline of {model} names its weights; the CI's {bmk.get('model')} is not checked")
    elif bmk.get("model") not in known:
        # CI 44550646425 (2026-11-19): a cell named glm5.2 served amd/GLM-5.3-Quark-MXFP4-AttnFP8 (user decision
        # 2026-11-23: refuse, with an override; a new model gets its own matrix entry)
        if not allow_other_model:
            sys.exit(f"CI run {ci_run}: cell {model}/{target} served {bmk.get('model')}, but {model}'s stored baselines "
                     f"use {sorted(known)}: another model under the same CI name. Add it to config/matrix.yaml as its "
                     "own model (or --allow-other-model)")
        print(f"note: --allow-other-model: {model}/{target} runs {bmk.get('model')} (stored baselines: {sorted(known)})")
    agentic ="agentic_" + bmk_art["name"][len("bmk_agentic_"):]
    agentic_art = next((a for a in arts if a["name"] == agentic and not a["expired"]), None)
    if not agentic_art:
        sys.exit(f"no live {agentic} artifact (needed for the server command)")
    texts = github_relay.call(relay, [github_relay.artifact_text(
        INFX, agentic_art["id"], ["vllm_command.txt", "benchmark_command.txt", "server.log"], grep={"server.log": LOG_GREP})])[0]
    files = texts.get("data") or {}
    server_env, source, draft_rev, draft_from_log = None, "agentic_*/vllm_command.txt + server.log", None, True
    if "vllm_command.txt" in files:   # direct-launch CI format
        cmd = ci_baseline.parse_vllm_command(files["vllm_command.txt"])
        log = files.get("server.log", "")
        launch_text = files["vllm_command.txt"]
    else:                             # srtctl CI format: the resolved recipe and the server log in server_logs_*
        srv_name = "server_logs_" + bmk_art["name"][len("bmk_agentic_"):]
        srv = next((a for a in arts if a["name"] == srv_name and not a["expired"]), None)
        if not srv:
            sys.exit(f"{agentic} has no vllm_command.txt and there is no live {srv_name} (srtctl recipe): "
                     "an unknown CI artifact format; add its parser and a fixture (tests/test_ci_formats.py)")
        tar = "srt-single-node-logs.tar.gz"
        rc, rl = github_relay.call(relay, [github_relay.artifact_tar(INFX, srv["id"], tar, pattern=r"^\./logs/config\.yaml$"),
                                           github_relay.artifact_tar(INFX, srv["id"], tar, pattern=r"_agg_w0\.out$",
                                                                     grep=LOG_GREP)])
        recipe = next(iter(((rc.get("data") or {}).get("text") or {}).values()), None)
        if not recipe:
            sys.exit(f"{srv_name}: no logs/config.yaml in {tar}")
        try:
            cmd, server_env, _ = ci_baseline.parse_srt_recipe(recipe)   # the image comes from the bmk JSON, as for direct runs
        except ValueError as e:                                         # e.g. engine atom, or a disaggregated recipe
            sys.exit(f"CI run {ci_run}: {e}")
        log = "\n".join(((rl.get("data") or {}).get("text") or {}).values())
        source, launch_text = f"{srv_name[:60]}.../{tar}: logs/config.yaml + server log", recipe

    def rev_of(repo):
        """(revision, from the log?): the snapshot path the CI's server loaded; else, when it loaded by repo id
        without one (Kimi K3 run 40037005947 downloaded main during the run), the repo's HF main at the CI run."""
        try:
            return ci_baseline.model_rev_from_log(log, repo), True
        except ValueError:
            try:
                commits = remote.run_python(relay, HF_COMMITS, {"name": repo, "kind": "models"})
                return ci_baseline.corpus_rev_at(commits, run["created_at"]), False
            except (ValueError, RuntimeError) as e:
                sys.exit(f"CI run {ci_run}: no revision of {repo}: not in the server log, and HF history: {e}")
    model_rev, model_from_log = rev_of(cmd["model"])
    if cmd.get("draft"):   # a speculative draft model (SGLang draft path, or vLLM speculative-config "model")
        draft_rev, draft_from_log = rev_of(cmd["draft"])
    # the launch we parsed must be what the CI's server says it ran (catches a CI format change that still parses)
    fw = cmd.get("framework") or bmk.get("framework")
    try:
        check_notes = []
        problems = ci_baseline.check_against_server(fw, cmd["flags"], log, notes=check_notes)
    except ValueError as e:
        sys.exit(f"CI run {ci_run}: {e}")
    if problems:
        sys.exit(f"CI run {ci_run}: the launch parsed from {source} does not match its server's own arguments "
                 "(a CI format change?); nothing written:\n  " + "\n  ".join(problems))
    n_args = len(ci_baseline.server_logged_args(log)[1])
    ds = bmk.get("dataset") or {}
    commits = remote.run_python(relay, HF_COMMITS, {"name": ds["hf_dataset_name"]})
    corpus_rev = ci_baseline.corpus_rev_at(commits, run["created_at"])
    image, digest = ci_baseline.split_image(bmk.get("image"))
    if not digest:   # the tag's digest now; review.md marks it `derived`
        reg = registry.check(relay, [(bmk["image"], None)])[0]
        digest = reg.get("tag_digest")
    if not digest:   # the tag is gone upstream (nightly tags are deleted: Kimi K3 run 40037005947, 2026-11-22):
        # a stored record that pinned the same tag earlier knows its digest, which Docker Hub may still serve
        same = [r for mo, t in config.cells(config.load_matrix()) for r in [baseline.load_latest(mo, t)]
                if r and r["fingerprint"].get("image") == image and r["fingerprint"].get("image_digest")]
        digest = same[0]["fingerprint"]["image_digest"] if same else None
        if digest:
            print(f"note: image tag {image} is gone upstream; using the digest {digest[:19]} that "
                  f"{same[0]['epoch_id']} pinned for it")
    try:
        tpl = ci_baseline.template_record(model, target, cmd.get("framework") or bmk.get("framework"))
    except ValueError as e:
        sys.exit(f"CI run {ci_run}: {e}")
    rec, rows = ci_baseline.build_draft(model, tgt, run, bmk_art["name"], bmk, cmd, model_rev, ds["hf_dataset_name"],
                                        corpus_rev, digest, tpl, server_env=server_env, source=source, draft_rev=draft_rev)
    derived_revs = [p for p, ok in (("launch.model", model_from_log), ("launch.draft", draft_from_log)) if not ok]
    if derived_revs:   # review.md: a revision not in the log is derived, like the corpus's
        rows = [(f, v, "HF commit history at the CI run (not in the server log)", "derived")
                if f.startswith(tuple(derived_revs)) else (f, v, s, k) for f, v, s, k in rows]
    rows += [("check note", n, "server log argument dump", "derived") for n in check_notes]
    rows.append(("check: server's own arguments", f"every parsed flag matches; {n_args} logged arguments read",
                 f"CI run {ci_run}: server log `{'non-default args' if fw == 'vllm' else 'server_args='}` dump", "exact"))
    errs = baseline.validation_errors(rec)
    if errs:
        sys.exit("draft is not schema-valid:\n  " + "\n  ".join(errs))
    notes = [f"bmk dataset entries: {ds.get('num_dataset_entries')} (the runner's offline corpus load must print the same "
             "number in prepare)", f"CI aiperf command: {files.get('benchmark_command.txt', '')[:300]}"]
    d = REPO_ROOT / "runs" / "drafts" / f"{model}__{target}__ci{ci_run}"
    d.mkdir(parents=True, exist_ok=True)
    write(d / "draft.json", json.dumps(rec, indent=2) + "\n")
    write(d / "ci_bmk.json", json.dumps({"artifact": bmk_art, "json": bmk}, indent=1) + "\n")
    write(d / "ci_commands.txt", launch_text + "\n\n" + files.get("benchmark_command.txt", "") + "\n")
    write(d / "review.md", ci_baseline.review_md(rec, rows, run, bmk_art["name"], notes))
    return d, rec


def cmd_measure(args):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import submit_ab_test
    d = Path(args.draft)
    rec = json.loads((d / "draft.json").read_text(encoding="utf-8"))
    local = config.load_local()
    backend = backends.for_site(local, args.site)
    if not config.load_cluster(config.local_view(args.site).get("cluster", backend.site_name)).get("baselines"):
        sys.exit(f"site {backend.site_name} is a runner only: baselines are measured on a baseline site "
                 "(config/cluster/<name>.yaml `baselines: true`; today crusoe)")
    req = {"commit": args.arbor_commit, "mode": "baseline", "gsm8k": args.gsm8k,
           "targets": [{"model": rec["model"], "target": rec["target"]["key"], "weights_path": args.weights_path}]}
    if args.qos:
        req["backend_options"] = {"qos": args.qos, **({"account": args.account} if args.account else {})}
    if args.exclude_nodes:   # a node the runner refused as not clean (Crusoe 308407, 2026-11-24: v3-207)
        req["exclude_nodes"] = args.exclude_nodes
    rc = submit_ab_test.submit(req, f"baseline-{d.name}", backend,
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
    p.add_argument("--ci-run", required=True, help="InferenceX Actions run id or URL")
    p.add_argument("--tp", type=int)
    p = sub.add_parser("measure")
    p.add_argument("--draft", required=True)
    p.add_argument("--arbor-commit", required=True, help="the Arbor whose launch code renders the server command")
    p.add_argument("--gsm8k", action="store_true", help="also run GSM8K (off by default)")
    p.add_argument("--weights-path")
    p.add_argument("--site", help="local.yaml site (default: local.yaml default_site)")
    p.add_argument("--qos")
    p.add_argument("--account")
    p.add_argument("--exclude-nodes", nargs="*", default=[], help="nodes not to use (e.g. one the runner refused)")
    p = sub.add_parser("finalize")
    p.add_argument("--draft", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--replace", action="store_true")
    args = ap.parse_args()
    if getattr(args, "ci_run", None):
        try:                               # a run id, or a run/job URL pasted from GitHub
            args.ci_run = ci_baseline.parse_ci_run(args.ci_run)
        except ValueError as e:
            sys.exit(str(e))
    return {"draft": cmd_draft, "measure": cmd_measure, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
