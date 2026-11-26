"""Build comparison.json for every cell of a PR-test run that has both phases (docs/design.md section 8).
A campaign-only run (mode campaign / campaign_smoke) has no before/after: its row shows the campaign's exit instead.

Usage: python scripts/generate_report.py --run-dir runs/<run_id> [--print]
Reads runs/<run_id>/run_config.json (written by submit_ab_test.py) and each <model>__<target>/{before,after}/ dir.
Stateless and idempotent: re-running rewrites the same comparison.json (only `generated_at` changes).
A cell missing a phase is reported as incomplete and gets no comparison.json.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from arbor_ab import agentx, baseline, config, report


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_phase(d):
    if not (d / "agg.json").exists():
        return None, None
    log = d / "gsm8k.log"
    lines = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines() if log.exists() else []
    return report.phase_metrics(read_json(d / "agg.json"), lines), read_json(d / "phase.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--print", action="store_true", help="also print each comparison.json")
    args = ap.parse_args()
    rc = read_json(args.run_dir / "run_config.json")
    if rc is None:
        sys.exit(f"{args.run_dir}/run_config.json not found")
    matrix = config.load_matrix()
    rows = []
    for cell in rc["cells"]:
        model, target = cell["model"], cell["target"]
        d = args.run_dir / f"{model}__{target}"
        # the record the cell ran from (written by submit since 2026-11-21; a CI draft is not in baselines/)
        record = (read_json(d / "baseline_record.json")
                  or baseline.load(baseline.cell_dir(model, target) / f"{cell['epoch_id']}.json"))
        src = report.baseline_source(record)
        src_txt = f"from {src['source_type']} run {src['ci_run_id']} ({src['ci_commit_date']})"
        if rc.get("mode") == "bench":     # one row per concurrency (runner.sh bench_sweep: bench/c<N>/agg.json)
            ph = read_json(d / "bench" / "phase.json") or {}
            for c in ph.get("concurrencies") or (rc.get("bench") or {}).get("concurrencies") or []:
                m, _ = read_phase(d / "bench" / f"c{c}")
                if m is None:
                    rows.append((model, target, f"bench c{c}", "no result (see logs/runner.log)"))
                    continue
                rows.append((model, target, f"bench c{c}", f"tput/GPU {m['tput_per_gpu']}, e2e_norm P50 "
                             f"{m['e2e_norm_intvty']['p50']} P90 {m['e2e_norm_intvty']['p90']}, intvty P90 "
                             f"{m['intvty']['p90']}; launch {src_txt}, {ph.get('duration_s')} s each"))
            log = d / "bench" / "gsm8k.log"
            if log.exists():
                g = agentx.gsm8k_from_log(log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines())
                rows.append((model, target, "bench GSM8K", str(g)))
            continue
        if rc.get("mode") in ("campaign", "campaign_smoke"):   # no before/after by design: the campaign's own result
            ex = read_json(d / "campaign_exit.json") or {}
            rows.append((model, target, "campaign", f"baseline {src_txt}; Arbor exit {ex.get('reason')}, "
                                                    f"iterations {ex.get('iteration')}, kept {ex.get('keeps')}"))
            continue
        before, before_meta = read_phase(d / "before")
        after, after_meta = read_phase(d / "after")
        if before is None or after is None:
            rows.append((model, target, "incomplete", f"missing: {[p for p, x in (('before', before), ('after', after)) if x is None]}"))
            continue
        comp = report.comparison(rc["run_id"], record, before, after, config.gsm8k_max_drop(matrix, model, target),
                                 rc["commit"], reference=rc.get("reference"),
                                 phases={"before": before_meta, "after": after_meta}, weights=read_json(d / "weights.json"))
        site = rc.get("site_name") or "crusoe"
        if not config.load_cluster(config.local_view(site).get("cluster", site)).get("baselines"):
            comp["ceiling"]["after_vs_ceiling_pct"] = None
            comp["ceiling"]["note"] = (f"run on {site} ({rc.get('gpu_type')}), a runner-only site: the ceiling is a baseline "
                                       "site's number and is not compared; before vs after (same node) still holds")
        ex = read_json(d / "campaign_exit.json") or {}
        comp["campaign_exit"] = {k: ex.get(k) for k in ("reason", "iteration", "keeps")} if ex else None
        if ex and not ex.get("keeps"):
            # nothing kept: after ran the same launch as before (the workspace config Arbor seeds is the pristine one),
            # so the difference is run-to-run noise (an A/A reading; meta64 test C 2026-11-21: P90 +5.5 %, P50 +0.6 %)
            comp["note"] = "the campaign kept nothing: after is the same launch as before, so before vs after is noise"
        comp["generated_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (d / "comparison.json").write_text(json.dumps(comp, indent=2) + "\n", encoding="utf-8", newline="\n")
        p = comp["deltas"][record["target"]["primary_metric_path"].split(".", 1)[1]]
        rows.append((model, target, "ok" if not comp.get("note") else "ok, NOTHING KEPT (A/A)", f"baseline {src_txt}; primary {p['before']} -> {p['after']} ({p['change_pct']}%), "
                                          f"after vs ceiling {comp['ceiling']['after_vs_ceiling_pct']}%, GSM8K {comp['gsm8k']['status']}, "
                                          f"weights {comp['weights']['weights_source']}"))
        if args.print:
            print(json.dumps(comp, indent=2))
    if rc.get("objective") or rc.get("goal"):
        # the request asked Arbor for its own objective/goal: the rows above still compare the cell's primary metric;
        # the campaign's effective objective is <cell>/sessions/full/objective.json (arbor/objective.py)
        print(f"note: the request set objective={rc.get('objective')!r} goal={rc.get('goal')!r}; 'primary' above is "
              "the cell's metric, compare the asked one in comparison.json deltas")
    for r in rows:
        print(" | ".join(map(str, r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
