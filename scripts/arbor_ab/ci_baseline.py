"""A baseline draft from an InferenceX CI run (docs/make_baseline_design.md, simplified MVP). Pure functions.

Exact from the CI run: commit/date (Actions API), topology + image + model prefix + corpus name (bmk_* JSON), server
flags (agentic_*/vllm_command.txt), model revision (agentic_*/server.log snapshot path), corpus revision (the dataset's
last HF commit before the run, checked against the bmk entry count by the caller).
Copied, for a person to confirm (review.md): server env and the node-side launch fields, from the latest baseline of
the same model (the "template"). Recipe parsing is not done (user decision 2026-11-20).
vLLM only for now: the SGLang CI command file is not verified.
"""
import copy
import re
import shlex

from . import agentx, baseline

ARBOR_ADDS = {"--host": 1, "--port": 1, "--tensor-parallel-size": 1}   # arbor/frameworks/vllm.py serve_cmd adds these


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
    out.update(model=model, flags=shlex.join(flags))
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


def split_image(ref):
    return (ref.split("@", 1) + [None])[:2] if ref and "@" in ref else (ref, None)


def template_record(model, target):
    """The latest baseline of the same model with a usable launch block; the same target first."""
    from . import config
    m = config.load_matrix()
    order = [target] + [t for t in m["models"][model]["targets"] if t != target]
    for t in order:
        r = baseline.load_latest(model, t)
        if r and r.get("launch") and not r["launch"]["missing"]:
            return r
    raise ValueError(f"no baseline of {model} with a launch block to copy the env from")


def build_draft(model, tgt, run, bmk_name, bmk, cmd, model_rev, corpus_id, corpus_rev, image_digest, template):
    """A schema-v2 record (metrics empty) and review rows [(field, value, source, kind)]; kind: exact|copied|derived."""
    if bmk.get("framework") != "vllm":
        raise ValueError(f"framework {bmk.get('framework')!r}: only vLLM CI runs are supported in this MVP")
    tpl = template
    image, digest_in_ref = split_image(bmk.get("image"))
    digest = digest_in_ref or image_digest
    flags = cmd["flags"]
    env = dict(tpl["host"].get("server_env") or {})
    env.update(MODEL_PREFIX=bmk.get("infmax_model_prefix") or env.get("MODEL_PREFIX"), FRAMEWORK=bmk["framework"])
    tp = cmd["tp"] or bmk.get("tp")
    num_gpus = bmk.get("num_gpus") or tp
    # model path: the template's convention (snapshot path in the node hub, or the repo id)
    tpl_path = tpl["host"].get("model_path") or ""
    model_path = (f"/hf-hub/models--{cmd['model'].replace('/', '--')}/snapshots/{model_rev}" if tpl_path.startswith("/hf-hub/")
                  else cmd["model"])
    lau = copy.deepcopy(tpl["launch"])
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
        ("host.server_flags", flags, f"{src}: agentic_*/vllm_command.txt", "exact"),
        ("launch.model_id/model_rev", f"{cmd['model']} @ {model_rev}", f"{src}: vllm_command.txt + server.log snapshot path", "exact"),
        ("launch.corpus_id/corpus_rev", f"{corpus_id} @ {corpus_rev}", "bmk dataset name + HF commit history at the CI run",
         "derived"),
        ("host.server_env", "; ".join(f"{k}={v}" for k, v in sorted(env.items())), f"copied from {tpl['epoch_id']} "
         "(MODEL_PREFIX, FRAMEWORK from the bmk JSON)", "copied"),
        ("launch.workspace_root/editable_packages/gsm8k", f"{lau['workspace_root']} / {lau.get('editable_packages')} / {lau['gsm8k']}",
         f"copied from {tpl['epoch_id']}", "copied"),
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
