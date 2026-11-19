"""Run the two-layer verification gate (docs/design.md 3.4) on stored baselines and record the result in each record.

Usage:
  python scripts/verify_baseline.py [--model M] [--target T] [--evidence-dir DIR] [--dry-run]
GitHub calls go through the relay host in local.yaml (ssh.github_relay_host). No GPU, no cluster jobs.
A `fail` is written into the record (verification.numeric.status) so nobody trusts it; it is not hidden.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from arbor_ab import baseline, config, gate, github_relay

REPO = "SemiAnalysisAI/InferenceX"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_records(model=None, target=None):
    m = config.load_matrix()
    out = []
    for mod, tgt in config.cells(m):
        if (model and mod != model) or (target and tgt != target):
            continue
        r = baseline.load_latest(mod, tgt)
        if r is not None:
            out.append(r)
    return m, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--target")
    ap.add_argument("--evidence-dir", type=Path, help="save each downloaded CI bmk JSON here")
    ap.add_argument("--dry-run", action="store_true", help="evaluate and print; do not write records")
    args = ap.parse_args()

    relay = config.load_local()["ssh"]["github_relay_host"]
    matrix, records = load_records(args.model, args.target)
    agentx_ci = [r for r in records if r["fingerprint"]["source_type"] == "agentx_ci"]
    runs = sorted({r["fingerprint"]["ci_run_id"] for r in agentx_ci})

    res = github_relay.call(relay, [q for run in runs for q in (github_relay.get(f"/repos/{REPO}/actions/runs/{run}"),
                                                                 github_relay.artifacts(REPO, run))])
    run_info = {run: (res[2 * i], res[2 * i + 1]) for i, run in enumerate(runs)}

    wanted = {}
    for r in agentx_ci:
        arts = run_info[r["fingerprint"]["ci_run_id"]][1].get("data") or []
        for a in gate.bmk_candidates(arts, r["host"]["tp"], r["target"]["conc"]):
            if not a["expired"]:
                wanted[a["id"]] = a
    ids = sorted(wanted)
    dl = github_relay.call(relay, [github_relay.artifact_json(REPO, i) for i in ids]) if ids else []
    ci_json = {}
    for i, d in zip(ids, dl):
        members = d.get("data") or {}
        if len(members) == 1:
            ci_json[i] = next(iter(members.values()))
            if args.evidence_dir:
                args.evidence_dir.mkdir(parents=True, exist_ok=True)
                (args.evidence_dir / f"{i}.json").write_text(json.dumps(
                    {"artifact": wanted[i], "member": next(iter(members)), "json": ci_json[i]}, indent=1) + "\n", encoding="utf-8", newline="\n")

    rows = []
    for r in records:
        cell = f"{r['model']}/{r['target']['key']}"
        v = r["verification"]
        t = now()
        fp = r["fingerprint"]
        if fp["source_type"] != "agentx_ci":
            rows.append((cell, "not_run", "not_run", "user_custom: layer 2 needs a fresh measurement (make_baseline.py)"))
            continue
        run, arts = run_info[fp["ci_run_id"]]
        head = (run.get("data") or {}).get("head_sha") or ""
        pin_ok = run.get("http") == 200 and head.startswith(fp["ci_commit"][:10])
        cands = gate.bmk_candidates(arts.get("data") or [], r["host"]["tp"], r["target"]["conc"])
        live = [(a, ci_json[a["id"]]) for a in cands if a["id"] in ci_json]
        chosen = gate.pick(r, live)
        notes = [f"run {fp['ci_run_id']} head_sha {head[:12] or '?'} {'matches' if pin_ok else 'DOES NOT match'} ci_commit"]
        notes.append(f"bmk candidates tp{r['host']['tp']} conc{r['target']['conc']}: {len(cands)} ({len(live)} live)")

        if chosen:
            diffs, skipped = gate.structural_diff(r, chosen[1])
            s_status = "pass" if pin_ok and not diffs else "fail"
            notes.append(f"chosen artifact {chosen[0]['id']} ({chosen[1].get('num_requests_successful')} requests)")
            if skipped:
                notes.append("not compared (null in record): " + ",".join(skipped))
            tol = config.tolerance_pct(matrix, r["model"], r["target"]["key"])
            n_status, ci_v, ours, delta = gate.numeric(r, chosen[1], tol, fp["ci_commit"])
            if n_status == "client_mismatch":
                notes.append(f"bench client {r['measured']['bench_client_commit'][:10]} != CI {fp['ci_commit'][:10]}")
            if r["measured"].get("bench_client_commit") is None:
                notes.append("bench client commit unknown")
            v["numeric"] = {"status": n_status, "checked_at": t, "ci_artifact": f"{REPO}/actions/artifacts/{chosen[0]['id']}",
                            "ci_value": ci_v, "measured_value": ours,
                            "delta_pct": round(delta, 3) if delta is not None else None, "tolerance_pct": tol}
        else:
            reason = "no artifacts left for this run (expired)" if not (arts.get("data") or []) else \
                "no live structurally matching bmk artifact with data"
            s_status = "pass" if pin_ok else "fail"
            notes.append(reason + "; structural check limited to the pin")
            v["numeric"] = {"status": "artifact_expired", "checked_at": t, "ci_artifact": None, "ci_value": None,
                            "measured_value": gate.primary_value(r["metrics"], r["target"]), "delta_pct": None,
                            "tolerance_pct": config.tolerance_pct(matrix, r["model"], r["target"]["key"])}
        v["structural"] = {"status": s_status, "checked_at": t, "detail": "; ".join(notes)}
        v["ci_verified"], v["unverified_against_ci"] = gate.flags(s_status, v["numeric"]["status"])
        rows.append((cell, s_status, v["numeric"]["status"],
                     f"delta {v['numeric']['delta_pct']}% (ours {v['numeric']['measured_value']} vs CI {v['numeric']['ci_value']}); "
                     + "; ".join(notes)))
        if not args.dry_run:
            baseline.store(r)

    print("| cell | structural | numeric | detail |\n|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
