"""AgentX official results database: the AgentX results InferenceX ingested, on B300 and MI355X (agentx_db/README.md).

Usage:
  python scripts/agentx_db.py sync [--pages 3] [--hw b300-dsxe mi355x-amds] [--refresh]
  python scripts/agentx_db.py index                      # rebuild points.csv from runs/
  python scripts/agentx_db.py query [--model dsv41flash] [--hw mi355x-amds] [--conc 32] [--framework vllm]
                                    [--latest] [--csv]
  python scripts/agentx_db.py compare --model dsv41flash [--metric tput_per_gpu] [--latest]

Official = what InferenceX's "Merge Ingest" workflow (.github/workflows/merge-ingest.yml) sends to the AgentX
database: on each merge to main that changes perf-changelog.yaml it reuses the merged PR's "Run Sweep" run, named by an
authorized `/use <run_id>` PR comment, and dispatches `ingest-agentic-results` with that source run id. A PR's other
sweeps are not official. sync lists the successful Merge Ingest runs (--pages of 100, newest first), resolves each to
its PR and source run (the last `/use` comment; the bot's ingest report is kept), and stores that source run.

sync reads GitHub through the relay (config ssh.github_relay_host; no token here) and writes one file per source run,
agentx_db/official/runs/<run_id>.json.gz, then points.csv (one row per point). Existing runs are not fetched again
(--refresh does). It keeps, verbatim with a sha256 each, the run's small official artifacts: results_bmk
(agg_bmk.json, every point's metrics), eval_results_all, changelog-metadata, run-stats and each point's job_event
(recipe path, topology, image). The large per-point artifacts (agentic_*, server_logs_*, power_audit_*, eval_*) are
indexed only (name, id, size, expires_at): GitHub deletes them after 90 days, and run_from_ci.py can still read them
while they live. A point's launch stays reproducible from its recipe path at the run's InferenceX commit (head_sha).
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbor_ab import REPO_ROOT, agentx, config, github_relay as gh  # noqa: E402

REPO = "SemiAnalysisAI/InferenceX"
WORKFLOW = 214438218                       # "Run Sweep"
MERGE_WORKFLOW = 370551520                 # "Merge Ingest" (merge-ingest.yml)
USE_RE = re.compile(r"(?m)^\s*/use\s+(\d+)\b")
DB = REPO_ROOT / "agentx_db" / "official"
HW_DEFAULT = ("b300-dsxe", "mi355x-amds")  # run-stats names; GB300 (gb300-nv) is another machine
SCHEMA = "agentx-db/official-run/1"
RUN_ARTIFACTS = {"results_bmk": "results_bmk", "eval_results_all": "eval_results_all",
                 "changelog-metadata": "changelog", "run-stats": "run_stats"}
POINT_KINDS = ("bmk_agentic_", "agentic_", "server_logs_", "power_audit_", "eval_server_logs_", "eval_", "job_event_")
COLUMNS = ["merged_at", "pr", "run_id", "date", "branch", "hw", "model_prefix", "model", "framework", "precision",
           "spec_decoding", "tp", "ep", "pp", "dp_attention", "dcp_size", "num_gpus", "conc", "kv_offloading",
           "kv_offload_backend", "image", "recipe_fingerprint", "requests_ok", "requests_total",
           "tput_per_gpu", "output_tput_per_gpu", "total_tput_tps", "e2e_norm_intvty_p50", "e2e_norm_intvty_p90",
           "intvty_p50", "intvty_p90", "ttft_p50_ms", "tpot_p50_ms", "gsm8k_em_strict", "avg_power_w",
           "joules_per_output_token", "power_valid", "status"]
EVAL_COLUMNS = ["merged_at", "pr", "run_id", "hw", "model_prefix", "framework", "precision", "spec_decoding", "tp", "ep",
                "dp_attention", "conc", "task", "em_strict", "em_strict_se", "em_flexible", "n_eff", "has_bench_point"]


def hw_name(hw):
    """'cluster:mi355x-amds' / 'CLUSTER:MI355X-AMDS' -> 'mi355x-amds'."""
    return (hw or "").split(":")[-1].lower()


def digest(obj):
    """sha256 of the artifact's JSON as stored (canonical: sorted keys, no spaces); the relay returns parsed JSON."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def kind_of(name):
    return next((k.rstrip("_") for k in POINT_KINDS if name.startswith(k)), None)


def source_of(comments):
    """(source run id, ingest report) from a merged PR's comments: the last `/use <id>` (Merge Ingest reuses it only
    when authorized, and the merge run succeeded), and the bot's ingest report after it."""
    src, report = None, None
    for c in comments:
        ids = USE_RE.findall(c.get("body") or "")
        if ids:
            src = int(ids[-1])
        elif (c.get("user") or {}).get("type") == "Bot" and "ingest" in (c.get("body") or "").lower():
            report = {"at": c.get("created_at"), "by": (c.get("user") or {}).get("login"), "body": c["body"][:4000]}
    return src, report


def official_merges(relay, pages):
    """The successful Merge Ingest runs, each resolved to its merged PR and source Run Sweep run."""
    merges = []
    for page in range(1, pages + 1):
        d = call(relay, [gh.get(f"/repos/{REPO}/actions/workflows/{MERGE_WORKFLOW}/runs?per_page=100&page={page}"
                                "&status=success")])[0].get("data") or {}
        merges += d.get("workflow_runs") or []
        if len(d.get("workflow_runs") or []) < 100:
            break
    out = []
    for i in range(0, len(merges), 40):
        batch = merges[i:i + 40]
        pulls = call(relay, [gh.get(f"/repos/{REPO}/commits/{m['head_sha']}/pulls") for m in batch])
        prs = []
        for m, p in zip(batch, pulls):
            ps = p.get("data") or []
            pr = next((x for x in ps if x.get("merge_commit_sha") == m["head_sha"]), ps[0] if ps else None)
            prs.append(pr)
        comments = call(relay, [gh.get(f"/repos/{REPO}/issues/{pr['number']}/comments?per_page=100") for pr in prs if pr])
        it = iter(comments)
        for m, pr in zip(batch, prs):
            if not pr:
                continue
            src, report = source_of(next(it).get("data") or [])
            if src:
                out.append({"source_run_id": src, "merge_run_id": m["id"], "merge_commit": m["head_sha"],
                            "merged_at": pr.get("merged_at"), "pr": pr["number"], "pr_title": pr.get("title"),
                            "pr_url": pr.get("html_url"), "ingest_report": report})
    return out


def build_run(run, arts, fetched, hw_wanted, official=None):
    """The stored record of one CI run (None when it has no point on the wanted hardware)."""
    agg = fetched.get("results_bmk") or []
    points = [p for p in agg if hw_name(p.get("hw")) in hw_wanted]
    if not points:
        return None
    evals = [e for e in (fetched.get("eval_results_all") or []) if hw_name(e.get("hw")) in hw_wanted]
    stats = fetched.get("run_stats") or {}
    return {
        "schema": SCHEMA,
        "official": official,                               # the Merge Ingest that ingested this run (merge, PR)
        "run": {k: run.get(k) for k in ("id", "html_url", "created_at", "head_branch", "head_sha", "event",
                                        "display_title", "run_attempt", "conclusion")},
        "hardware": sorted({hw_name(p.get("hw")) for p in points}),
        "points": points,                                   # results_bmk/agg_bmk.json entries, verbatim
        "evals": evals,                                     # eval_results_all/agg_eval_all.json entries, verbatim
        "changelog": fetched.get("changelog"),
        "run_stats": {k: v for k, v in stats.items() if k in hw_wanted} if isinstance(stats, dict) else stats,
        "job_events": fetched.get("job_events") or [],
        "artifacts": [{"name": a["name"], "id": a["id"], "kind": kind_of(a["name"]) or "run",
                       "size": a["size_in_bytes"], "expires_at": a.get("expires_at"), "expired": a.get("expired")}
                      for a in arts],
        "sha256": {k: digest(fetched[k]) for k in ("results_bmk", "eval_results_all", "changelog", "run_stats")
                   if fetched.get(k) is not None},
        "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run_path(db, run_id):
    return db / "runs" / f"{run_id}.json.gz"


def write_run(path, rec):
    """gzip with mtime 0: a run's file is written once and its bytes do not depend on when it was written."""
    raw = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(gzip.compress(raw, mtime=0))


def read_run(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def first_file(resp):
    data = resp.get("data") or {}
    return next(iter(data.values()), None) if isinstance(data, dict) else None


def call(relay, reqs):
    """One relay call, tried twice: a dropped ssh connection must not end a long sync (2026-11-25, run 23 of the
    first backfill: 'Connection closed by ... port 22')."""
    try:
        return gh.call(relay, reqs, timeout=1800)
    except gh.RelayError as e:
        print(f"  relay call failed ({str(e)[-120:]}); trying once more", flush=True)
        return gh.call(relay, reqs, timeout=1800)


def sync(relay, pages, hw_wanted, refresh=False, db=DB):
    (db / "runs").mkdir(parents=True, exist_ok=True)
    merges = official_merges(relay, pages)
    by_src = {}
    for m in sorted(merges, key=lambda m: m["merged_at"] or ""):   # a run reused by two merges: keep the first
        by_src.setdefault(m["source_run_id"], m)
    # source runs checked before with no point on the wanted hardware (or nothing left to read): not fetched again
    checked_file = db / "checked.txt"
    checked = set(int(x) for x in checked_file.read_text(encoding="utf-8").split()) if checked_file.exists() else set()
    todo = [s for s in by_src if refresh or not (run_path(db, s).exists() or s in checked)]
    print(f"{len(merges)} Merge Ingest runs -> {len(by_src)} source Run Sweep runs; {len(todo)} to check "
          f"({len(checked)} checked before without B300/MI355X AgentX points)", flush=True)
    stored, skipped = 0, []
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        try:
            runs = [x.get("data") or {} for x in call(relay, [gh.get(f"/repos/{REPO}/actions/runs/{s}") for s in batch])]
            listed = call(relay, [gh.artifacts(REPO, s) for s in batch])
        except gh.RelayError:
            skipped += batch
            continue
        for s, r, a in zip(batch, runs, listed):
            try:
                n = store_run(relay, r, a, hw_wanted, db, by_src[s]) if r.get("id") else 0
            except gh.RelayError:
                skipped.append(s)
                continue
            stored += n
            if not n:
                checked.add(s)
    checked_file.write_text("".join(f"{s}\n" for s in sorted(checked)), encoding="utf-8", newline="\n")
    n = write_index(db)
    print(f"stored {stored} run(s); points.csv: {n} points"
          + (f"; {len(skipped)} run(s) skipped on relay errors (sync again to retry): {skipped[:10]}" if skipped else ""),
          flush=True)


def store_run(relay, r, a, hw_wanted, db, official=None):
    """Fetch and write one run; 1 if stored, 0 if it has no AgentX point on the wanted hardware (or its artifacts
    expired: GitHub keeps them 90 days)."""
    arts = [x for x in (a.get("data") or []) if not x.get("expired")]
    by_name = {x["name"]: x for x in arts}
    if "results_bmk" not in by_name:        # no AgentX result in this run (or it expired)
        return 0
    want = [(key, by_name[name]) for name, key in RUN_ARTIFACTS.items() if name in by_name]
    got = call(relay, [gh.artifact_json(REPO, x["id"]) for _, x in want])
    fetched = {key: first_file(g) for (key, _), g in zip(want, got)}
    if not any(hw_name(p.get("hw")) in hw_wanted for p in fetched.get("results_bmk") or []):
        return 0
    evs = [x for x in arts if x["name"].startswith("job_event_")]
    evg = call(relay, [gh.artifact_json(REPO, x["id"]) for x in evs]) if evs else []
    fetched["job_events"] = [first_file(g) for g in evg if first_file(g)]
    rec = build_run(r, [x for x in arts if kind_of(x["name"]) or x["name"] in RUN_ARTIFACTS], fetched, hw_wanted,
                    official)
    if not rec:
        return 0
    write_run(run_path(db, r["id"]), rec)
    print(f"  {r['created_at'][:10]} {r['id']} {','.join(rec['hardware'])}: {len(rec['points'])} points "
          f"({r['head_branch'][:40]})", flush=True)
    return 1


def eval_key(d):
    return (d.get("model_prefix") or d.get("infmax_model_prefix"), d.get("framework"), d.get("precision"),
            d.get("spec_decoding"), d.get("tp"), d.get("ep"), d.get("conc"), str(d.get("dp_attention")).lower())


def point_row(rec, p, evals):
    m, _ = agentx.metrics_from_agg(p)
    lat = ((p.get("request_metrics") or {}).get("latency") or {})
    per_gpu = ((p.get("request_metrics") or {}).get("throughput") or {}).get("per_gpu") or {}
    ev = evals.get(eval_key(p)) or {}
    kvb = p.get("kv_offload_backend")
    run, off = rec["run"], rec.get("official") or {}
    return {"merged_at": (off.get("merged_at") or "")[:16], "pr": off.get("pr"), "run_id": run["id"],
            "date": (run.get("created_at") or "")[:10], "branch": run.get("head_branch"), "hw": hw_name(p.get("hw")), "model_prefix": p.get("infmax_model_prefix"),
            "model": p.get("model"), "framework": p.get("framework"), "precision": p.get("precision"),
            "spec_decoding": p.get("spec_decoding"), "tp": p.get("tp"), "ep": p.get("ep"), "pp": p.get("pp"),
            "dp_attention": p.get("dp_attention"), "dcp_size": p.get("dcp_size"), "num_gpus": p.get("num_gpus"),
            "conc": p.get("conc"), "kv_offloading": p.get("kv_offloading"),
            "kv_offload_backend": kvb.get("name") if isinstance(kvb, dict) else kvb, "image": p.get("image"),
            "recipe_fingerprint": p.get("recipe_fingerprint"), "requests_ok": p.get("num_requests_successful"),
            "requests_total": p.get("num_requests_total"), "tput_per_gpu": m.get("tput_per_gpu"),
            "output_tput_per_gpu": per_gpu.get("output_tput_tps"), "total_tput_tps": m.get("total_tput_tps"),
            "e2e_norm_intvty_p50": m["e2e_norm_intvty"]["p50"], "e2e_norm_intvty_p90": m["e2e_norm_intvty"]["p90"],
            "intvty_p50": m["intvty"]["p50"], "intvty_p90": m["intvty"]["p90"],
            "ttft_p50_ms": (lat.get("ttft") or {}).get("p50"), "tpot_p50_ms": (lat.get("tpot") or {}).get("p50"),
            "gsm8k_em_strict": ev.get("em_strict"), "avg_power_w": p.get("avg_power_w"),
            "joules_per_output_token": p.get("joules_per_output_token"), "power_valid": p.get("power_valid"),
            # failed: the point ran but no request succeeded (no metrics), e.g. run 36567581989 minimaxm3 c48
            "status": "ok" if p.get("num_requests_successful") and m.get("tput_per_gpu") is not None else "failed"}


def load_runs(db=DB):
    return [read_run(f) for f in sorted((db / "runs").glob("*.json.gz"))]


def rows(db=DB):
    out = []
    for rec in load_runs(db):
        if not rec.get("official"):      # a sweep no Merge Ingest took (e.g. from an earlier, all-runs sync)
            continue
        evals = {eval_key(e): e for e in rec.get("evals") or []}
        out += [point_row(rec, p, evals) for p in rec["points"]]
    return sorted(out, key=lambda r: (r["merged_at"], r["run_id"]), reverse=True)


def eval_rows(db=DB):
    """Every official GSM8K result (eval_results_all), also those at a configuration with no bench point (InferenceX
    evaluates a few points; 30 of 139 had no matching bench point on 2026-11-25)."""
    out = []
    for rec in load_runs(db):
        if not rec.get("official"):
            continue
        keys = {eval_key(p) for p in rec["points"]}
        off = rec["official"]
        for e in rec.get("evals") or []:
            out.append({"merged_at": (off.get("merged_at") or "")[:16], "pr": off.get("pr"), "run_id": rec["run"]["id"],
                        "hw": hw_name(e.get("hw")), "model_prefix": e.get("model_prefix"),
                        **{k: e.get(k) for k in ("framework", "precision", "spec_decoding", "tp", "ep", "dp_attention",
                                                 "conc", "task", "em_strict", "em_strict_se", "em_flexible", "n_eff")},
                        "has_bench_point": eval_key(e) in keys})
    return sorted(out, key=lambda r: (r["merged_at"], r["run_id"]), reverse=True)


def write_csv(path, columns, rs):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rs)


def write_index(db=DB):
    rs = rows(db)
    write_csv(db / "points.csv", COLUMNS, rs)
    write_csv(db / "evals.csv", EVAL_COLUMNS, eval_rows(db))
    return len(rs)


CONFIG_KEY = ("hw", "model_prefix", "framework", "precision", "spec_decoding", "tp", "ep", "dp_attention",
              "kv_offloading", "conc")


def latest_only(rs):
    seen, out = set(), []
    for r in rs:                                      # rows are newest first
        k = tuple(r[c] for c in CONFIG_KEY)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def select(rs, model=None, hw=None, conc=None, framework=None, branch=None, latest=False):
    rs = [r for r in rs if (model is None or r["model_prefix"] == model) and (hw is None or r["hw"].startswith(hw))
          and (conc is None or r["conc"] == conc) and (framework is None or r["framework"] == framework)
          and (branch is None or r["branch"] == branch)]
    return latest_only(rs) if latest else rs


def fmt(v):
    return f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v))


def print_table(rs, cols):
    widths = [max(len(c), *(len(fmt(r[c])) for r in rs)) if rs else len(c) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rs:
        print("  ".join(fmt(r[c]).ljust(w) for c, w in zip(cols, widths)))


def compare(rs, metric):
    """Per (model, conc): the best point of each hardware on `metric` (higher is better), side by side."""
    best = {}
    for r in rs:
        if r[metric] is None:
            continue
        k = (r["model_prefix"], r["conc"], r["hw"])
        if k not in best or r[metric] > best[k][metric]:
            best[k] = r
    hws = sorted({k[2] for k in best})
    out = []
    for model, conc in sorted({(k[0], k[1]) for k in best}, key=lambda x: (x[0], x[1])):
        row = {"model_prefix": model, "conc": conc}
        for h in hws:
            b = best.get((model, conc, h))
            row[h] = b[metric] if b else None
            row[f"{h} config"] = (f"{b['framework']} tp{b['tp']} ep{b['ep']} kv:{b['kv_offloading']} run {b['run_id']}"
                                  if b else None)
        out.append(row)
    return out, hws


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--pages", type=int, default=3, help="pages of 100 completed runs, newest first")
    s.add_argument("--hw", nargs="+", default=list(HW_DEFAULT))
    s.add_argument("--refresh", action="store_true", help="fetch runs already in the database again")
    sub.add_parser("index")
    for name in ("query", "compare"):
        q = sub.add_parser(name)
        q.add_argument("--model")
        q.add_argument("--hw")
        q.add_argument("--conc", type=int)
        q.add_argument("--framework")
        q.add_argument("--branch")
        q.add_argument("--latest", action="store_true", help="newest point per configuration only")
        if name == "query":
            q.add_argument("--csv", action="store_true")
        else:
            q.add_argument("--metric", default="tput_per_gpu", choices=[c for c in COLUMNS if c.endswith(
                ("per_gpu", "tps", "_p50", "_p90", "em_strict"))])
    args = ap.parse_args()
    if args.cmd == "sync":
        sync(config.local_view()["ssh"]["github_relay_host"], args.pages, set(args.hw), args.refresh)
    elif args.cmd == "index":
        print(f"points.csv: {write_index()} points")
    else:
        rs = select(rows(), args.model, args.hw, args.conc, args.framework, args.branch, args.latest)
        if args.cmd == "query":
            if args.csv:
                w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
                w.writeheader()
                w.writerows(rs)
            else:
                print_table(rs, ["merged_at", "pr", "run_id", "hw", "model_prefix", "framework", "tp", "ep", "conc",
                                 "kv_offloading", "tput_per_gpu", "e2e_norm_intvty_p50", "e2e_norm_intvty_p90",
                                 "gsm8k_em_strict"])
        else:
            out, hws = compare(rs, args.metric)
            print(f"best {args.metric} per (model, conc) and hardware")
            print_table(out, ["model_prefix", "conc", *hws, *[f"{h} config" for h in hws]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
