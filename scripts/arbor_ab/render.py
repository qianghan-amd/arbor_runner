"""Render one cell of a PR-test run from its baseline record (pure; no I/O). Used by submit_ab_test.py.

Everything a node needs comes from the record (design.md section 7: "all parameters come from the stored baseline
JSON") plus the run request (commit, mode, campaign minutes) and site settings (paths, QOS).
"""
import re
import shlex

PORT = 31400
SESSION_MOUNT = "/workspace/sessions"
INFX_IN_CONTAINER = "/workspace/InferenceX"
BENCH_ENV_REQUIRED = ("MODEL_PREFIX", "FRAMEWORK")          # packs/agentx/validator/bench_agentx.sh:76-77
P90_OBJECTIVE = "P90 interactivity"                          # arbor/objective.py:54,71 -> e2e_norm_intvty_p90
# campaign (the default, user decision 2026-11-20): fetch, prepare, the Arbor campaign; no before/after measurement.
# full: before, campaign, after (the A/B comparison). campaign_smoke: campaign with a short budget (same steps).
# smoke: one short measurement, no campaign. baseline: fetch, prepare, before (make_baseline.py).
# bench: the AgentX bench alone at the request's concurrencies, one server for all (request `bench:`).
MODES = ("campaign", "full", "smoke", "campaign_smoke", "baseline", "bench")
BENCH_DEFAULT_DURATION_S = 600
CAMPAIGN_MODES = ("campaign", "full", "campaign_smoke")


class NotRunnable(ValueError):
    pass


def cell_name(model, target):
    return f"{model}__{target}"


def container_name(run_id, model, target):
    stamp = run_id.split("_", 1)[0]                       # "20261119-075002": unique per submission
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", f"ab_{stamp}_{model}_{target}")[:63]


def check_runnable(record):
    """List of reasons this cell cannot run a PR test (empty = runnable)."""
    why = []
    lau = record.get("launch")
    if not lau:
        return ["no launch block (summary-only cell: re-baseline first)"]
    why += [f"launch.missing: {m}" for m in lau["missing"]]
    env = record["host"].get("server_env") or {}
    why += [f"host.server_env lacks {k} (the pack's bench needs it)" for k in BENCH_ENV_REQUIRED if k not in env]
    if not record["host"].get("model_path"):
        why.append("host.model_path missing")
    if not (record.get("measured") or {}).get("bench_client_commit"):
        # the InferenceX client the bench runs (runner.sh INFX_SHA): without it prepare stops after the fetch
        # (meta64 151758, 2026-11-24, stored dsv41flash c32_tput: "InferenceX  not in ...")
        why.append("measured.bench_client_commit missing (start from a CI run, or re-baseline)")
    gpus = [g for g in str(lau.get("gpu_ids") or "").split(",") if g.strip()]
    if len(gpus) != record["host"]["num_gpus"]:
        why.append(f"launch.gpu_ids {lau.get('gpu_ids')} != host.num_gpus {record['host']['num_gpus']}")
    if record["target"]["direction"] != "max":
        why.append("only max-direction targets are wired")
    return why


def check_objective(objective):
    """What is wrong with a request's `objective` (None = fine). Arbor's arbor/objective.py resolves it for real at
    `arbor check` (it refuses a name it does not know); this only catches the shape early."""
    if objective is None or (isinstance(objective, str) and objective.strip()):
        return None
    if isinstance(objective, dict):
        return None if objective.get("metric") else "objective: the structured form needs `metric`"
    return "objective: a metric name (text), or {metric, bar_pct, guards}"


def arbor_config(record, run_id, commit, cell_dir, container, minutes, objective=None, goal=None):
    """The Arbor campaign config (stock packs/agentx of the commit under test; keys per docs/reference.md).
    objective: the request's (text: passed as --opt-objective, see cell_env; structured: written as opt_objective,
    arbor/objective.py). goal: the request's sentence for Arbor's agents; else one from the target."""
    h, lau = record["host"], record["launch"]
    env = {k: str(v) for k, v in (h.get("server_env") or {}).items()}
    env["INFMAX_CONTAINER_WORKSPACE"] = INFX_IN_CONTAINER
    return {
        "_note": (f"arbor-ab-system PR test {run_id}: cell {record['model']}/{record['target']['key']}, baseline epoch "
                  f"{record['epoch_id']}, Arbor {commit}. Launch = the baseline's pristine launch; rendered by "
                  f"scripts/arbor_ab/render.py, not hand-written."),
        "pack": "agentx",
        "identity": f"arbor-ab:{record['model']}:{record['target']['key']}:{run_id}",
        "goal": goal or f"Optimize {record['target']['primary_metric_path']} at concurrency {record['target']['conc']}.",
        "framework": h["framework"],
        "model_path": h["model_path"],
        "model_name": lau["served_model_name"],
        "tp": h["tp"],
        "port": PORT,
        "gpu_ids": lau["gpu_ids"],
        "server_flags": h.get("server_flags") or "",
        "server_env": env,
        "docker_container": container,
        "workspace_root": lau["workspace_root"],
        "editable_packages": lau.get("editable_packages") or [],
        "concurrency": lau["concurrency"],
        "num_prompts": lau.get("num_prompts") or 320,
        "bench_timeout": lau.get("bench_timeout_s") or 7200,
        # Arbor's docs/reference.md: absolute paths to the pack's validator scripts, as the container sees them.
        # Without them Arbor's init looks for './validator' in the container and stops (P6 job 299696, 2026-11-20);
        # `arbor check` did not flag it. The Arbor tree is mounted at /workspace/Arbor (runner.sh).
        "bench_script": "/workspace/Arbor/packs/agentx/validator/bench_agentx.sh",
        "correctness_script": "/workspace/Arbor/packs/agentx/validator/correctness.sh",
        "session_root": f"{cell_dir}/sessions/full",
        "session_mount": SESSION_MOUNT,
        "time_budget_minutes": minutes,
        "review_gate": False,
        **({"opt_objective": objective} if isinstance(objective, dict) else {}),
    }


def opt_objective(record):
    """None keeps the pack's objective (total_throughput); P90 cells pass --opt-objective."""
    return P90_OBJECTIVE if record["target"]["primary_metric_path"] == "metrics.e2e_norm_intvty.p90" else None


def cell_env(record, run_id, commit, mode, cell_dir, container, minutes, site, arbor_tar, arbor_tar_sha256, weights_path=None,
             skip_gsm8k=False, arbor_launcher=True, objective=None, bench=None):
    """Shell variables for scripts/runner/runner.sh (written as cell.env). weights_path: optional pre-downloaded weights
    of the served model on the node (request `weights_path`); fetch.py checks them against the pin and links them."""
    h, lau, fp = record["host"], record["launch"], record["fingerprint"]
    return {
        "ARBOR_TAR": arbor_tar, "ARBOR_TAR_SHA256": arbor_tar_sha256,
        "RUN_ID": run_id, "CELL": cell_name(record["model"], record["target"]["key"]), "MODE": mode,
        "CELL_DIR": cell_dir, "NL_ROOT": site["node_local_root"], "SHARED_HUB": site.get("shared_hf_hub") or "",
        # the node-local disk limits (fetch.py make_room): free space kept, our cache's cap
        "NL_MIN_FREE_GB": site.get("node_local_min_free_gb") or "", "NL_MAX_CACHE_GB": site.get("node_local_max_cache_gb") or "",
        "IMAGE_TAG": fp["image"], "IMAGE_DIGEST": fp["image_digest"] or "",
        "CONTAINER": container, "PORT": str(PORT), "GPUS": lau["gpu_ids"], "FRAMEWORK": h["framework"],
        "MODEL_ID": lau["model_id"], "MODEL_REV": lau["model_rev"],
        "DRAFT_ID": lau.get("draft_id") or "", "DRAFT_REV": lau.get("draft_rev") or "",
        "CORPUS_ID": lau["corpus_id"], "CORPUS_REV": lau["corpus_rev"],
        "MODEL_PATH": h["model_path"], "MODEL_NAME": lau["served_model_name"],
        "WORKSPACE_ROOT": lau["workspace_root"],
        "INFX_CLONE": site["infx_clone"], "INFX_SHA": record["measured"]["bench_client_commit"] or "",
        "ARBOR_SHA": commit, "CONC": str(lau["concurrency"]), "NUM_PROMPTS": str(lau.get("num_prompts") or 320),
        "BENCH_TIMEOUT": str(lau.get("bench_timeout_s") or 7200),
        "GSM8K_TOOL": lau["gsm8k"]["tool"], "GSM8K_CONC": str(lau["gsm8k"].get("concurrency") or 16),
        # skip_gsm8k (request): 0 questions = no GSM8K in any phase; phase.json and the report say "skipped"
        "DUR_FULL": "3600", "NQ_FULL": "0" if skip_gsm8k else str(lau["gsm8k"].get("num_questions") or 1319),
        "DUR_SMOKE": "600", "NQ_SMOKE": "0" if skip_gsm8k else "200",
        "CAMPAIGN_MINUTES": str(minutes), # --opt-objective: the request's text objective; none when it is structured (in the config, which the flag
        # would override); else the cell's (P90 cells)
        "OPT_OBJECTIVE": objective if isinstance(objective, str) else "" if objective else (opt_objective(record) or ""),
        "GATEWAY_ENV": site.get("gateway_env") or "",
        "WEIGHTS_PATH": weights_path or "",
        # `arbor run` under Arbor's launcher (its default; #74 closed 2026-11-24, not reproduced); request
        # `arbor_standalone: true`: one process (ARBOR_STANDALONE=1, kept by Arbor for one release)
        "ARBOR_LAUNCHER": "1" if arbor_launcher else "0",
        # mode bench: the concurrencies (default: the cell's), the duration of each run, GSM8K once (0 = skipped)
        "BENCH_CONCS": " ".join(str(c) for c in bench_concurrencies(record, bench)),
        "BENCH_DUR": str((bench or {}).get("duration_s") or BENCH_DEFAULT_DURATION_S),
        "NQ_BENCH": "0" if skip_gsm8k else str(lau["gsm8k"].get("num_questions") or 1319),
        # bench.restart_each: a fresh server for each concurrency (GSM8K on the last one), else one shared server
        "BENCH_RESTART": "1" if (bench or {}).get("restart_each") else "0",
    }


def env_file(env):
    return "".join(f"{k}={shlex.quote(str(v))}\n" for k, v in env.items())


def bench_concurrencies(record, bench):
    """The request's bench concurrencies (positive ints, in its order, no repeats), else the cell's."""
    concs = (bench or {}).get("concurrencies") or [record["launch"]["concurrency"]]
    if not all(isinstance(c, int) and c > 0 for c in concs):
        raise ValueError(f"bench.concurrencies: positive integers, got {concs!r}")
    return list(dict.fromkeys(concs))


def job_minutes(mode, minutes, record, bench=None):
    """Wall clock the cell's job needs (the backend turns it into its own time limit). Fetch <= 60, prepare <= 30, one measurement <= 75 + GSM8K (vLLM C1: 70)."""
    meas = 75 + (75 if record["host"]["framework"] == "vllm" else 10)
    if mode == "smoke":
        return 60 + 30 + 30 + 30
    if mode == "bench":      # fetch, prepare, one server start (<= 40), each run + 10 min, GSM8K, margin
        dur = (bench or {}).get("duration_s") or BENCH_DEFAULT_DURATION_S
        concs = bench_concurrencies(record, bench)
        starts = len(concs) if (bench or {}).get("restart_each") else 1
        return 60 + 30 + 40 * starts + sum(dur // 60 + 10 for _ in concs) + 75 + 30
    if mode in ("campaign", "campaign_smoke"):
        return 60 + 30 + minutes + 30
    if mode == "baseline":
        return 60 + 30 + meas + 30
    return 60 + 30 + 2 * meas + minutes + 60
