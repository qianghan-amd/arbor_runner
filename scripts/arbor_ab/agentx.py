"""Field map: raw AgentX / lm-eval / SGLang outputs -> the baseline schema's `metrics` block.

Verified against real files on 2026-11-19 (experiments/arbor-ab-system/2026-11-19_0432_port/raw/):
- The AgentX agg JSON (`agg_<prefix>_<fw>_agentx_conc<N>.json`) is the source of record. The schema mirrors its
  nested `request_metrics.latency.{intvty,e2e_norm_intvty}` blocks.
- Trap: Arbor's `bench_output.json` field `total_throughput` is the PER-GPU value (it equals
  `request_metrics.throughput.per_gpu.total_tput_tps`), not the total. Do not map it to `total_tput_tps`.
- Trap: older packs (`agentx-dsv41-*` validator) report `p90_interactivity` = agg `intvty.p90`, while the C1 P90
  campaigns optimize `e2e_norm_intvty.p90`. They are different metrics.
- Trap: agg `dcp_size` can read 1 when the server ran `--decode-context-parallel-size 8` (kimik3-c1-p90-sa). Take
  DCP from the server flags.
- Older agg files (glm52-c16-tput, 2026-11-09) have no `e2e_norm_intvty` block and no `num_gpus`.
"""
import re

PCTL = ("p50", "p90", "p95")
AGG_PATHS = {
    "total_tput_tps": "request_metrics.throughput.total.tokens_per_second",
    "tput_per_gpu": "request_metrics.throughput.per_gpu.total_tput_tps",
    **{f"intvty.{p}": f"request_metrics.latency.intvty.{p}" for p in PCTL},
    **{f"e2e_norm_intvty.{p}": f"request_metrics.latency.e2e_norm_intvty.{p}" for p in PCTL},
}


def get_path(obj, dotted):
    for k in dotted.split("."):
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def set_path(obj, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        obj = obj[k]
    obj[keys[-1]] = value


def empty_gsm8k():
    return {"flexible_extract": None, "strict_match": None, "accuracy": None, "num_questions": None}


def empty_metrics():
    return {"total_tput_tps": None, "tput_per_gpu": None,
            "intvty": dict.fromkeys(PCTL), "e2e_norm_intvty": dict.fromkeys(PCTL), "gsm8k": empty_gsm8k()}


def metrics_from_agg(agg):
    """(metrics without GSM8K, {schema field: agg path}) for every field present in the agg JSON."""
    m, src = empty_metrics(), {}
    for field, path in AGG_PATHS.items():
        v = get_path(agg, path)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            set_path(m, field, float(v))
            src[field] = path
    return m, src


_LM_EVAL_ROW = re.compile(r"^\|[^|]*\|[^|]*\|\s*(flexible-extract|strict-match)\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9.]+)\s*\|")


def gsm8k_from_log(lines, num_questions=None):
    """lm-eval-harness summary table rows (vLLM packs) or SGLang few_shot_gsm8k `Accuracy:` line."""
    g = empty_gsm8k()
    for line in lines:
        m = _LM_EVAL_ROW.match(line)
        if m:
            g["flexible_extract" if m.group(1) == "flexible-extract" else "strict_match"] = float(m.group(2))
        m = re.match(r"^Accuracy:\s*([0-9.]+)", line)
        if m:
            g["accuracy"] = float(m.group(1))
    g["num_questions"] = num_questions
    return g


def gsm8k_from_lm_eval_results(results):
    """lm-eval `results_*.json` (task `gsm8k`)."""
    r = results["results"]["gsm8k"]
    g = empty_gsm8k()
    g["flexible_extract"] = r.get("exact_match,flexible-extract")
    g["strict_match"] = r.get("exact_match,strict-match")
    n = r.get("sample_len") or (results.get("n-samples", {}).get("gsm8k", {}) or {}).get("effective")
    g["num_questions"] = int(n) if n else None
    return g


def has_gsm8k(g):
    return any(g[k] is not None for k in ("flexible_extract", "strict_match", "accuracy"))


def dcp_from_flags(flags):
    m = re.search(r"(?:--decode-context-parallel-size|-dcp)[ =](\d+)", flags or "")
    return int(m.group(1)) if m else 1
