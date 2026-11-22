"""A baseline draft from an InferenceX CI run (docs/make_baseline_design.md, simplified MVP). Pure functions.

Two CI artifact formats:
- direct launch, vLLM only (e.g. run 34657748450): agentic_*/vllm_command.txt (server flags) and agentic_*/server.log
  (model revision). The server env is not recorded: it is copied from the same model's stored baseline (review.md).
- srtctl launch, vLLM and SGLang (e.g. vLLM run 44689368479, SGLang run 38877885539):
  server_logs_*/srt-single-node-logs.tar.gz holds the resolved recipe logs/config.yaml (roles.agg.args = server
  flags, roles.agg.env = server env, model.container = image@digest, `engine` = the framework) and the server log
  logs/<host>_agg_w0.out (model and draft revisions). Flags and server env are both exact. One aggregated server only.
Always exact: commit/date (Actions API), topology + corpus name (bmk_* JSON), corpus revision (the dataset's last HF
commit before the run). Copied from the template (the same model's stored baseline): the bench-side env (AgentX
loader, workspace paths) and the node-side launch fields.
"""
import copy
import re
import shlex

import yaml

from . import agentx, baseline

ARBOR_ADDS = {"--host": 1, "--port": 1, "--tensor-parallel-size": 1}   # arbor/frameworks/vllm.py serve_cmd adds these
# arbor/frameworks/sglang.py serve_cmd adds --model-path, --served-model-name, --host, --port; TP stays in the flags
SGLANG_ARBOR_ADDS = {"--host", "--port", "--model-path"}
FRAMEWORKS = ("vllm", "sglang")
# server-side env keys; with an exact server env from the CI recipe, the template's own values of these are dropped
SERVER_ENV_PREFIXES = ("VLLM_", "AITER_", "SGLANG_", "HIP_", "NCCL_", "RCCL_", "TORCH_", "OMP_", "MIOPEN_")
KEEP_FROM_TEMPLATE = ("VLLM_ENGINE_READY_TIMEOUT_S",)   # a harness wait, not a tuning knob (srtctl sets it outside the recipe)


def parse_srt_recipe(text):
    """srtctl recipe (logs/config.yaml) -> ({framework, model, served_name, tp, draft, flags}, server_env, image_ref).
    Flags: roles.agg.args as `--key value` in recipe order (true -> `--key`; false/empty -> left out), minus what
    Arbor adds itself: vLLM: served name, TP, host, port; SGLang: served name, model path, host, port (TP stays).
    draft: the speculative draft model's repo id (SGLang `speculative-draft-model-path`), when it is one."""
    cfg = yaml.safe_load(text)
    roles = cfg.get("roles") or {}
    if set(roles) != {"agg"}:
        raise ValueError(f"srtctl recipe roles {sorted(roles)}: only one aggregated server (agg) is supported")
    engine = cfg.get("engine")
    fw = (engine.get("type") if isinstance(engine, dict) else engine) or (cfg.get("frontend") or {}).get("type")
    if fw not in FRAMEWORKS:
        raise ValueError(f"srtctl engine {fw!r}: only {FRAMEWORKS} are supported")
    args, toks = roles["agg"].get("args") or {}, []
    out = {"framework": fw, "tp": None, "served_name": None, "draft": None}
    for k, v in args.items():
        if k == "served-model-name":
            out["served_name"] = v[0] if isinstance(v, list) else v
            continue
        if k in ("tensor-parallel-size", "tp", "tp-size"):
            out["tp"] = int(v)
            if fw == "vllm":
                continue                       # vLLM: Arbor adds --tensor-parallel-size; SGLang: it stays a flag
        if k == "speculative-draft-model-path" and isinstance(v, str) and "/" in v and not v.startswith("/"):
            out["draft"] = v                   # a repo id: pinned and fetched like the model (launch.draft_id/rev)
        if (f"--{k}" in (ARBOR_ADDS if fw == "vllm" else SGLANG_ARBOR_ADDS)) or v is None or v is False or v == "":
            continue
        toks += [f"--{k}"] if v is True else [f"--{k}", v if isinstance(v, str) else str(v)]
    model = str((cfg.get("model") or {}).get("path", "")).removeprefix("hf:")
    env = {k: str(v) for k, v in (roles["agg"].get("env") or {}).items()}
    flags = shlex.join(toks)
    out.update(model=model, flags=flags, draft=out["draft"] or vllm_draft(flags))
    return out, env, str((cfg.get("model") or {}).get("container") or "").strip()


# ---- the launch we parsed, checked against what the CI's server says it ran -------------------------------------------
# Whatever the CI's artifact format (it has changed: direct launch -> srtctl, 2026-11), the server logs its own resolved
# arguments: vLLM `non-default args: {...}` (the arguments it was given), SGLang `server_args={...}` (all of them).
# A flag our parser dropped, renamed or mangled shows up here, so a format change cannot pass silently.
# vLLM: keys the harness, not the recipe, sets (Arbor's serve_cmd adds model/served name/host/port/TP; srtctl adds
# api_server_count and device_ids).
VLLM_HARNESS_KEYS = {"model_tag", "model", "served_model_name", "host", "port", "tensor_parallel_size", "api_server_count",
                     "device_ids"}
SGLANG_KEY_ALIASES = {"tensor_parallel_size": "tp_size", "expert_parallel_size": "ep_size", "pipeline_parallel_size": "pp_size",
                      "data_parallel_size": "dp_size",
                      # run 44231353783 (SGLang DSv4.1 MI355X, 2026-11-18): stored under another name
                      "fp8_gemm_backend": "fp8_gemm_runner_backend", "fp4_gemm_backend": "fp4_gemm_runner_backend",
                      "monokernel_backend": "enable_amd_monokernel"}
# keys whose value SGLang rewrites (rescaled, or a deprecated flag mapped to a switch): compared by key only
SGLANG_ADJUSTED = {"mem_fraction_static",       # 0.85 -> 0.7225 (run 38877885539)
                   "enable_amd_monokernel"}     # `--monokernel-backend amd` -> True (deprecated; run 44231353783)


def _top_level_items(text):
    """'{'a': 1, 'b': X(c=2), 'd': {...}}' -> [('a', '1'), ('b', 'X(c=2)'), ('d', '{...}')]: split at depth 1, quotes kept."""
    items, depth, quote, start = [], 0, None, 1
    for i, ch in enumerate(text):
        if quote:
            if ch == quote and text[i - 1] != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                items.append(text[start:i])
                break
        elif ch == "," and depth == 1:
            items.append(text[start:i])
            start = i + 1
    out = []
    for it in items:
        k, sep, v = it.strip().partition(":")
        if sep:
            out.append((k.strip().strip("'\""), v.strip()))
    return out


def server_logged_args(log_text):
    """('vllm'|'sglang', {key: value text}) from the server log's own argument dump; (None, {}) when it has none."""
    for kind, pat in (("vllm", r"non-default args: (\{.*\})\s*$"), ("sglang", r"server_args=(\{.*\})\s*$")):
        m = re.search(pat, log_text, re.M)
        if m:
            return kind, dict(_top_level_items(m.group(1)))
    return None, {}


def _flag_dict(flags):
    toks, out, i = shlex.split(flags), {}, 0
    while i < len(toks):
        name = toks[i].lstrip("-").replace("-", "_")
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out[name], i = toks[i + 1], i + 2
        else:
            out[name], i = True, i + 1
    return out


def _same(ours, logged_text):
    """Whether our flag value equals the logged one; None when the logged value is an object we cannot read."""
    import ast
    import json
    try:
        theirs = ast.literal_eval(logged_text)
    except (ValueError, SyntaxError):
        return None                                   # e.g. AttentionConfig(...): only the key is checked
    if ours is True:
        return theirs is True
    if isinstance(theirs, dict):
        try:
            mine = json.loads(ours)
        except (TypeError, ValueError):
            return False
        return all(theirs.get(k) == v for k, v in mine.items())
    if isinstance(theirs, list):
        theirs = theirs[0] if len(theirs) == 1 else theirs
    if isinstance(theirs, (int, float)) and not isinstance(theirs, bool):
        try:
            return float(ours) == float(theirs)
        except ValueError:
            return False
    return str(ours) == str(theirs)


def check_against_server(framework, flags, log_text, notes=None):
    """Problems (empty = the parsed flags are what the CI's server ran). A flag of ours that the server logged must
    have the same value (objects: by key only; `auto`: the server resolves it). vLLM logs only its non-default
    arguments: every one of them but the harness's must be among ours (a dropped flag is caught), and a flag of ours
    it does not list is a note (set to the default). SGLang logs every argument, some under another name (aliases,
    deprecated flags): one it does not list is a note (a name SGLang did not know would stop our own server at its
    start). Raises ValueError without a dump."""
    kind, logged = server_logged_args(log_text)
    if kind is None:
        raise ValueError("the server log has no argument dump (vLLM `non-default args`, SGLang `server_args=`): "
                         "cannot check the parsed launch")
    if kind != framework:
        return [f"the server log is {kind}'s, the launch is {framework}"]
    ours, problems = _flag_dict(flags), []
    notes = notes if notes is not None else []
    for k, v in ours.items():
        key = SGLANG_KEY_ALIASES.get(k, k) if framework == "sglang" else k
        if key not in logged:
            # vLLM lists only arguments that differ from its defaults: a recipe that sets one to the default value
            # (Kimi K3 run 40037005947: --moe-backend) is absent; its other direction below stays strict
            msg = f"--{k.replace('_', '-')} is in the parsed flags but not among the server's arguments" + (
                " (vLLM lists non-default arguments only: probably its default value)" if framework == "vllm" else "")
            notes.append(msg)
        elif str(v).lower() == "auto":
            notes.append(f"--{k.replace('_', '-')} auto: the server resolved it to {logged[key][:60]}")
        elif not (framework == "sglang" and key in SGLANG_ADJUSTED) and _same(v, logged[key]) is False:
            problems.append(f"--{k.replace('_', '-')}: parsed {v!r}, the server ran {logged[key][:80]}")
    if framework == "vllm":
        for key, val in logged.items():
            if key not in ours and key not in VLLM_HARNESS_KEYS:
                problems.append(f"the server ran {key}={val[:80]}, which the parsed flags lack")
    return problems


def vllm_draft(flags):
    """The draft model repo of a vLLM `--speculative-config {"model": "<org>/<name>", ...}` (Kimi K3's DSpark, run
    40037005947), pinned like the served model; None for a native draft (dspark/mtp without a model) or a local path."""
    import json
    t = shlex.split(flags)
    for i, tok in enumerate(t[:-1]):
        if tok == "--speculative-config":
            try:
                m = json.loads(t[i + 1]).get("model")
            except (ValueError, AttributeError):
                return None
            return m if isinstance(m, str) and "/" in m and not m.startswith("/") else None
    return None


def parse_vllm_command(text):
    """'vllm serve <model> --served-model-name N [path] --host H --port P --tensor-parallel-size T <flags>' ->
    {model, served_name, tp, flags}; flags = everything Arbor does not add itself, shell-quoted."""
    t = shlex.split(text.strip())
    if t[:2] != ["vllm", "serve"] or len(t) < 3:
        raise ValueError(f"not a 'vllm serve' command: {text[:80]!r}")
    model, rest, i, out = t[2], t[3:], 0, {"tp": None, "served_name": None}
    flags = []
    while i < len(rest):
        tok = rest[i]
        if tok == "--served-model-name":
            i += 1
            names = []
            while i < len(rest) and not rest[i].startswith("--"):
                names.append(rest[i])
                i += 1
            out["served_name"] = names[0] if names else None
            continue
        if tok in ARBOR_ADDS:
            if tok == "--tensor-parallel-size":
                out["tp"] = int(rest[i + 1])
            i += 2
            continue
        flags.append(tok)
        i += 1
    out.update(model=model, flags=shlex.join(flags), draft=vllm_draft(shlex.join(flags)))
    return out


def model_rev_from_log(log_text, model_repo):
    """The snapshot revision the CI server loaded: models--<org>--<name>/snapshots/<rev> in server.log."""
    pat = r"models--" + re.escape(model_repo.replace("/", "--")) + r"/snapshots/([0-9a-f]{40})"
    revs = sorted(set(re.findall(pat, log_text)))
    if len(revs) != 1:
        raise ValueError(f"server.log names {len(revs)} snapshot revisions of {model_repo}: {revs[:3]}")
    return revs[0]


def corpus_rev_at(commits, ci_created_at):
    """The dataset's main at the CI run: the latest commit dated at or before it. commits: [{id, date}] (HF API)."""
    before = [c for c in commits if c["date"] <= ci_created_at]
    if not before:
        raise ValueError("the dataset has no commit before the CI run")
    return max(before, key=lambda c: c["date"])["id"]


CI_REPO = "SemiAnalysisAI/InferenceX"
# GPU architecture of a CI run's hardware (bmk JSON `hw`, e.g. "cluster:mi355x-amds"); images and recipes are built and
# tuned per architecture, so a run of one is not launched on another (NVIDIA runs carry CUDA images)
HW_ARCH = {"mi355x": "gfx950", "mi350x": "gfx950", "mi325x": "gfx942", "mi300x": "gfx942"}


def ci_arch(hw):
    """'cluster:mi355x-amds' -> 'gfx950'; an NVIDIA or unknown name -> the name itself (never equal to a gfx arch)."""
    name = (hw or "").split(":")[-1].split("-")[0].lower()
    return HW_ARCH.get(name, name or "unknown")


def parse_ci_run(arg):
    """The Actions run id from what a user pastes: `44689368479`, or a run / job / attempt URL of InferenceX
    (https://github.com/SemiAnalysisAI/InferenceX/actions/runs/44689368479[/job/123][/attempts/2])."""
    arg = str(arg).strip()
    if arg.isdigit():
        return arg
    m = re.match(r"^https?://github\.com/([^/]+/[^/]+)/actions/runs/(\d+)(?:/.*)?$", arg)
    if not m:
        raise ValueError(f"--ci-run {arg!r}: give the run id or its URL (https://github.com/{CI_REPO}/actions/runs/<id>)")
    if m.group(1).lower() != CI_REPO.lower():
        raise ValueError(f"--ci-run {arg!r}: a run of {m.group(1)}, not of {CI_REPO}")
    return m.group(2)


BMK_NAME = re.compile(r"^bmk_agentic_([a-z0-9.]+)_tp(\d+)_conc(\d+)_")   # glm5.2 has a dot


def ci_points(artifacts):
    """The AgentX points a CI run measured: sorted unique (model_prefix, tp, conc) of its live bmk_agentic_* artifacts."""
    pts = set()
    for a in artifacts:
        m = BMK_NAME.match(a["name"])
        if m and not a.get("expired"):
            pts.add((m.group(1), int(m.group(2)), int(m.group(3))))
    return sorted(pts)


def choose_cell(points, matrix, model=None, target=None, tp=None):
    """(model, target, tp) for a CI run's points; raises ValueError listing the choices when it is not one.
    A model is the matrix key whose `infmax_prefix` (default: the key) is the CI's model prefix, or `model`; a target
    matches the point's concurrency.
    With several TPs at that concurrency, the matrix's expected_host.tp decides, else `tp` must."""
    models = matrix["models"]
    by_prefix = {spec.get("infmax_prefix", k): k for k, spec in models.items()}
    opts = []
    for prefix, p_tp, conc in points:
        key = model or by_prefix.get(prefix)
        if key not in models or (model and by_prefix.get(prefix) not in (None, model)):
            continue
        for tkey, t in models[key]["targets"].items():
            if t["conc"] == conc and (target in (None, tkey)) and (tp in (None, p_tp)):
                opts.append((key, tkey, p_tp))
    opts = sorted(set(opts))
    if len(opts) > 1:
        preferred = [o for o in opts if (models[o[0]]["targets"][o[1]].get("expected_host") or {}).get("tp") == o[2]]
        if len(preferred) == 1:
            return preferred[0]
    if len(opts) == 1:
        return opts[0]
    have = ", ".join(f"{p} tp{t} conc{c}" for p, t, c in points) or "none"
    raise ValueError(("no matrix cell matches" if not opts else "several cells match: "
                      + ", ".join(f"{m}/{t} tp{x}" for m, t, x in opts) + "; choose with --target/--tp")
                     + f" (CI points: {have})")


def split_image(ref):
    return (ref.split("@", 1) + [None])[:2] if ref and "@" in ref else (ref, None)


# Fixed by the runner's own container (scripts/runner/runner.sh docker run mounts): never copied, always set
CONTAINER_BENCH_ENV = {"HF_HUB_CACHE": "/hf-hub", "HF_HUB_OFFLINE": "1", "INFMAX_CONTAINER_WORKSPACE": "/workspace/InferenceX",
                       "AIPERF_DATASET_MMAP_CACHE_DIR": "/aiperf_mmap_cache"}
# Bench-side env keys that do not depend on the model (InferenceX's AgentX scripts read them); with a template of
# another model only these are kept, and the model-specific ones are set from the CI run (build_draft).
GENERIC_BENCH_KEYS = {"AIPERF_DATASET_MMAP_CACHE_DIR", "AIPERF_HTTP_TCP_USER_TIMEOUT", "AIPERF_REQUIRED_SERVER_METRIC_PREFIX",
                      "DP_ATTENTION", "DURATION", "EP_SIZE", "FRAMEWORK", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "IMAGE",
                      "INFMAX_CONTAINER_WORKSPACE", "KV_OFFLOADING", "MODEL_PREFIX", "PRECISION", "PYTHONUNBUFFERED",
                      "RUNNER_TYPE", "SPEC_DECODING", "TOTAL_CPU_DRAM_GB", "TP", "WEKA_LOADER_OVERRIDE", "AGENTX_TOKENIZER"}


def template_record(model, target, framework=None):
    """The stored baseline whose launch layout (framework install path, editable packages, GSM8K tool) a draft copies.
    In order: the same model (same target first) with the CI's framework; else another model with that framework
    (the layout belongs to the framework's images, not to the model: a model's first SGLang run needs this)."""
    from . import config
    m = config.load_matrix()
    def usable(r):
        return r and r.get("launch") and not r["launch"]["missing"] and framework in (None, r["host"]["framework"])
    for t in [target] + [t for t in m["models"][model]["targets"] if t != target]:
        r = baseline.load_latest(model, t)
        if usable(r):
            return r
    others = [r for r in (baseline.load_latest(mo, t) for mo, t in config.cells(m) if mo != model) if usable(r)]
    if others:   # the most complete layout: the same target first, then the most generic bench keys
        return max(others, key=lambda r: (r["target"]["key"] == target,
                                          len(GENERIC_BENCH_KEYS & set(r["host"].get("server_env") or {}))))
    raise ValueError(f"no stored baseline with a launch block for framework {framework!r}: nothing to copy the "
                     "container layout from")


def build_draft(model, tgt, run, bmk_name, bmk, cmd, model_rev, corpus_id, corpus_rev, image_digest, template,
                server_env=None, source="agentic_*/vllm_command.txt + server.log", draft_rev=None):
    """A schema-v2 record (metrics empty) and review rows [(field, value, source, kind)]; kind: exact|copied|derived.
    server_env: the CI's exact server env (srtctl recipe); None = the template's env (direct-launch CI runs).
    cmd["draft"] / draft_rev: the speculative draft model (SGLang), pinned like the model."""
    if bmk.get("framework") not in FRAMEWORKS:
        raise ValueError(f"framework {bmk.get('framework')!r}: only {FRAMEWORKS} CI runs are supported")
    if template["host"]["framework"] != bmk["framework"]:   # template_record(framework=...) prevents this
        raise ValueError(f"the stored baseline {template['epoch_id']} is {template['host']['framework']}, the CI run is "
                         f"{bmk['framework']}: its container layout (workspace_root, editable_packages) cannot be copied")
    tpl = template
    image, digest_in_ref = split_image(bmk.get("image"))
    digest = digest_in_ref or image_digest
    flags = cmd["flags"]
    env = dict(tpl["host"].get("server_env") or {})
    if server_env is not None:   # exact from the CI: the template keeps only its bench-side keys
        env = {k: v for k, v in env.items() if not k.startswith(SERVER_ENV_PREFIXES) or k in KEEP_FROM_TEMPLATE}
        env.update(server_env)
    env.update(MODEL_PREFIX=bmk.get("infmax_model_prefix") or env.get("MODEL_PREFIX"), FRAMEWORK=bmk["framework"])
    # bench-side keys that describe the CI point: from the bmk JSON, never the template's older point
    point = {"IMAGE": image, "TP": cmd["tp"] or bmk.get("tp"), "EP_SIZE": bmk.get("ep"), "PRECISION": bmk.get("precision"),
             "DP_ATTENTION": bmk.get("dp_attention"), "KV_OFFLOADING": bmk.get("kv_offloading"),
             "SPEC_DECODING": bmk.get("spec_decoding")}
    env.update({k: str(v) for k, v in point.items() if k in env and v is not None})
    if tpl["model"] != model:   # a template of another model: only its generic bench keys; the rest from the CI run
        env = {k: v for k, v in env.items() if k in GENERIC_BENCH_KEYS or (server_env or {}).get(k) == v}
        hw = (bmk.get("hw") or "").split(":")[-1].split("-")[0]
        derived = {"AGENTX_TOKENIZER": cmd["model"], "WEKA_LOADER_OVERRIDE": (bmk.get("dataset") or {}).get("loader"),
                   "RUNNER_TYPE": hw or None}
        env.update({k: v for k, v in derived.items() if v and (k in env or k != "AGENTX_TOKENIZER")})
    for k, v in CONTAINER_BENCH_ENV.items():   # the container's own paths, whatever the template had
        env.setdefault(k, v)
    tp = cmd["tp"] or bmk.get("tp")
    num_gpus = bmk.get("num_gpus") or tp
    # model path: the template's convention (snapshot path in the node hub, or the repo id)
    tpl_path = tpl["host"].get("model_path") or ""
    model_path = (f"/hf-hub/models--{cmd['model'].replace('/', '--')}/snapshots/{model_rev}" if tpl_path.startswith("/hf-hub/")
                  else cmd["model"])
    lau = copy.deepcopy(tpl["launch"])
    draft = cmd.get("draft")
    lau.update(draft_id=draft, draft_rev=draft_rev if draft else None)
    lau.update(model_id=cmd["model"], model_rev=model_rev, corpus_id=corpus_id, corpus_rev=corpus_rev,
               served_model_name=cmd["served_name"] or cmd["model"], gpu_ids=",".join(str(i) for i in range(num_gpus)),
               concurrency=tgt["conc"], missing=[],
               source=f"InferenceX CI run {run['id']} ({bmk_name}) + env/launch copied from {tpl['epoch_id']}")
    rec = {
        "schema_version": 2, "model": model, "target": tgt,
        "host": {"framework": bmk["framework"], "tp": tp, "ep": bmk.get("ep") or 1, "dcp": agentx.dcp_from_flags(flags),
                 "num_gpus": num_gpus, "recipe_sh": None, "recipe_point": None, "model_path": model_path,
                 "server_flags": flags, "server_env": env},
        "fingerprint": {"source_type": "agentx_ci", "image": image, "image_digest": digest, "local_mirror_path": None,
                        "ci_commit": run["head_sha"], "ci_run_id": str(run["id"]), "ci_commit_date": run["created_at"][:10],
                        "recipe_fingerprint": bmk.get("recipe_fingerprint") or None},
        "epoch_id": "",
        "health": {"ci_artifact_available": True, "image_available": None, "local_mirror_exists": False, "checked_at": None},
        "verification": {"ci_verified": False, "unverified_against_ci": False,
                         "structural": {"status": "not_run", "checked_at": None, "detail": None},
                         "numeric": {"status": "not_run", "checked_at": None, "ci_artifact": None, "ci_value": None,
                                     "measured_value": None, "delta_pct": None, "tolerance_pct": None}},
        "measured": {"date": None, "cluster": None, "node": None, "job": None, "replays": None, "duration_s": None,
                     "bench_client_commit": run["head_sha"], "arbor_commit": None},
        "metrics": agentx.empty_metrics(), "ceiling": None,
        "artifacts": {"bench_output": None, "agg_json": None, "gsm8k_log": None},
        "provenance": {"_source": f"make_baseline.py draft from InferenceX CI run {run['id']}"},
        "superseded_by": None, "data_quality_warning": None, "launch": lau,
    }
    rec["epoch_id"] = baseline.derive_epoch_id(rec)
    src = f"CI run {run['id']}"
    rows = [
        ("fingerprint.ci_commit", run["head_sha"], f"{src}: Actions API head_sha", "exact"),
        ("fingerprint.image", image, f"{src}: {bmk_name} image", "exact"),
        ("fingerprint.image_digest", digest, f"{src}: {bmk_name} image" if digest_in_ref else "registry: tag's digest now",
         "exact" if digest_in_ref else "derived"),
        ("host.framework/tp/ep/num_gpus", f"{bmk['framework']}/{tp}/{rec['host']['ep']}/{num_gpus}", f"{src}: {bmk_name}", "exact"),
        ("host.dcp", rec["host"]["dcp"], "server flags (--decode-context-parallel-size)", "exact"),
        ("host.server_flags", flags, f"{src}: {source}", "exact"),
        ("launch.model_id/model_rev", f"{cmd['model']} @ {model_rev}", f"{src}: {source} (server log snapshot path)", "exact"),
        *([("launch.draft_id/draft_rev", f"{draft} @ {draft_rev}", f"{src}: {source} (server log snapshot path)", "exact")]
          if draft else []),
        ("launch.corpus_id/corpus_rev", f"{corpus_id} @ {corpus_rev}", "bmk dataset name + HF commit history at the CI run",
         "derived"),
        ("host.server_env", "; ".join(f"{k}={v}" for k, v in sorted(env.items())),
         (f"server keys exact from {src} ({source}); bench keys copied from {tpl['epoch_id']}" if server_env is not None
          else f"copied from {tpl['epoch_id']} (MODEL_PREFIX, FRAMEWORK from the bmk JSON)"),
         "exact+copied" if server_env is not None else "copied"),
        ("launch.workspace_root/editable_packages/gsm8k", f"{lau['workspace_root']} / {lau.get('editable_packages')} / {lau['gsm8k']}",
         f"copied from {tpl['epoch_id']}" + (f" (another model, same framework {tpl['host']['framework']}: the image "
                                              "layout; the runner checks it on the node)" if tpl["model"] != model else ""),
         "copied"),
        ("host.model_path", model_path, f"the template's convention ({'node hub snapshot' if model_path.startswith('/') else 'repo id'})",
         "derived"),
    ]
    return rec, rows


def review_md(rec, rows, run, bmk_name, notes):
    lines = [f"# Baseline draft: {rec['model']}/{rec['target']['key']} from InferenceX CI run {run['id']}", "",
             f"epoch `{rec['epoch_id']}`, CI commit `{run['head_sha']}` ({run['created_at']}), artifact `{bmk_name}`.", "",
             "**Confirm every `copied` and `derived` row before `make_baseline.py measure`.** Edit `draft.json` to fix one.",
             "", "| field | value | source | kind |", "|---|---|---|---|"]
    lines += [f"| {f} | `{str(v)[:300]}` | {s} | **{k}** |" if k != "exact" else f"| {f} | `{str(v)[:300]}` | {s} | {k} |"
              for f, v, s, k in rows]
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"
