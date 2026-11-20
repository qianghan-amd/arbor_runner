"""comparison.json for one (model, target) of one PR-test run (docs/design.md section 8). Pure functions.

Run-dir contract (written by the node-side runner, read here):
  runs/<run_id>/<model>__<target>/<phase>/   phase = before | after
      agg.json          AgentX agg JSON (source of record; schema metrics via agentx.metrics_from_agg)
      bench_output.json the pack's bench summary (kept for debugging; not used for numbers, see agentx.py traps)
      gsm8k.log         lm-eval table (vLLM) or `Accuracy:` line (SGLang)
      phase.json        {"started_at", "ended_at", "tree", "launch_from", "node", "job"} (optional)
  runs/<run_id>/<model>__<target>/weights.json   {"weights_source": "downloaded" | "user_path", ...} (fetch.py)
`verdict` stays null: a human reads the report (design.md section 8 / 11). Automating it later = filling that field.
"""
from . import agentx

GSM8K_KEYS = ("flexible_extract", "strict_match", "accuracy")


def phase_metrics(agg, gsm8k_lines, gsm8k_n=None):
    m, _ = agentx.metrics_from_agg(agg) if agg else (agentx.empty_metrics(), {})
    m["gsm8k"] = agentx.gsm8k_from_log(gsm8k_lines or [], gsm8k_n)
    return m


def _pct(a, b):
    return None if a is None or b in (None, 0) else round((a - b) / b * 100.0, 3)


def delta(before, after, path):
    b, a = agentx.get_path(before, path), agentx.get_path(after, path)
    return {"before": b, "after": a, "change_pct": _pct(a, b)}


def gsm8k_check(before, after, max_drop):
    """Per score: drop = before - after (absolute). status: pass | fail | missing."""
    out, statuses = {}, []
    for k in GSM8K_KEYS:
        b, a = before["gsm8k"][k], after["gsm8k"][k]
        if b is None and a is None:
            continue
        if b is None or a is None:
            out[k] = {"before": b, "after": a, "drop": None, "status": "missing"}
        else:
            drop = round(b - a, 4)
            out[k] = {"before": b, "after": a, "drop": drop, "status": "pass" if drop <= max_drop else "fail"}
        statuses.append(out[k]["status"])
    status = "missing" if not statuses or "missing" in statuses else ("fail" if "fail" in statuses else "pass")
    return {"max_drop": max_drop, "status": status, "scores": out}


def baseline_source(record):
    """Where the cell's starting point comes from: the baseline's AgentX CI run (or other source) and image."""
    f = record["fingerprint"]
    return {"epoch_id": record["epoch_id"], "source_type": f.get("source_type"), "ci_run_id": f.get("ci_run_id"),
            "ci_commit": f.get("ci_commit"), "ci_commit_date": f.get("ci_commit_date"), "image": f.get("image"),
            "image_digest": f.get("image_digest")}


def comparison(run_id, record, before, after, max_drop, arbor_commit, reference=None, phases=None, weights=None):
    """record: the baseline record of the cell (current epoch). before/after: schema metrics blocks."""
    tgt = record["target"]
    primary = tgt["primary_metric_path"].split(".", 1)[1]
    paths = [primary] + [p.split(".", 1)[1] for p in (tgt["primary_metric_per_gpu_path"], tgt["secondary_metric_path"]) if p]
    ceiling = record.get("ceiling")
    ceil_v = agentx.get_path(ceiling["metrics"], primary) if ceiling else None
    after_v = agentx.get_path(after, primary)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": record["model"],
        "target": tgt,
        "arbor_commit": arbor_commit,
        "epoch_id": record["epoch_id"],
        "baseline_source": baseline_source(record),
        "reference": reference or {"kind": "baseline", "id": record["epoch_id"],
                                   "note": "before = the baseline's pristine launch, re-measured fresh in this run"},
        "before": before,
        "after": after,
        "deltas": {p: delta(before, after, p) for p in paths},
        "ceiling": {"primary": ceil_v, "source": ceiling["source"] if ceiling else None,
                    "after_vs_ceiling_pct": _pct(after_v, ceil_v)},
        "gsm8k": ({"max_drop": max_drop, "status": "skipped", "scores": {}, "note": "request skip_gsm8k"}
                  if all((phases or {}).get(p) and (phases or {})[p].get("gsm8k_questions") == 0 for p in ("before", "after"))
                  else gsm8k_check(before, after, max_drop)),
        "baseline_flags": {"ci_verified": record["verification"]["ci_verified"],
                           "numeric_gate": record["verification"]["numeric"]["status"],
                           "data_quality_warning": [w["code"] for w in record["data_quality_warning"] or []]},
        "phases": phases or {},
        # where the served weights came from: downloaded at the pin, or a user path checked against the pin (fetch.py)
        "weights": weights or {"weights_source": "unknown", "note": "no weights.json in the run (made before 2026-11-19)"},
        "verdict": None,
    }
