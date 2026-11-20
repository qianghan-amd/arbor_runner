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

from arbor_ab import baseline, config, report


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
        record = baseline.load(baseline.cell_dir(model, target) / f"{cell['epoch_id']}.json")
        src = report.baseline_source(record)
        src_txt = f"from {src['source_type']} run {src['ci_run_id']} ({src['ci_commit_date']})"
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
        comp["generated_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (d / "comparison.json").write_text(json.dumps(comp, indent=2) + "\n", encoding="utf-8", newline="\n")
        p = comp["deltas"][record["target"]["primary_metric_path"].split(".", 1)[1]]
        rows.append((model, target, "ok", f"baseline {src_txt}; primary {p['before']} -> {p['after']} ({p['change_pct']}%), "
                                          f"after vs ceiling {comp['ceiling']['after_vs_ceiling_pct']}%, GSM8K {comp['gsm8k']['status']}, "
                                          f"weights {comp['weights']['weights_source']}"))
        if args.print:
            print(json.dumps(comp, indent=2))
    for r in rows:
        print(" | ".join(map(str, r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
