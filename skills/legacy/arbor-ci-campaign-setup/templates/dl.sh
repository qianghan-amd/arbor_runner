#!/bin/bash
# <<CODENAME>>: pinned <<model>><<, draft model>> and the trace corpus into one HF cache on the node's NVMe
# (refs/main written = the pins, for offline loads). HF token on the node host only, never in a container.
#
# Adapted from experiments/musespark-c1-p90-sa/dl.sh. Skip this whole file if following the dsv41f-c32-tput shared-
# cache style instead (point env.sh's MODEL_C at the shared /shared_nfs/huggingface/hub snapshot path).
set -uo pipefail
NL=<<node-local dir, must match env.sh's HUB_NL's parent>>; HUB=$NL/hf-hub; mkdir -p "$HUB"
PYDL=<<path to a shared download venv, or create one below if none exists>>
[ -x "$PYDL" ] || { python3 -m venv $NL/dlvenv && $NL/dlvenv/bin/pip -q install huggingface_hub && PYDL=$NL/dlvenv/bin/python; }
export CORPUS_REV=<<corpus revision — InferenceX pins no revision for this one, so pick the shared hub's current main>>
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
echo "[dl] $(date -u +%T) start; hub $HUB; huggingface_hub $($PYDL -c 'import huggingface_hub as h; print(h.__version__)')"
$PYDL - "$HUB" <<'PY'
import json, os, sys, time
from huggingface_hub import snapshot_download
hub = sys.argv[1]
items = [
    ("<<MODEL_ID>>", "<<MODEL_REV>>", "model"),
    # ("<<DRAFT_ID>>", "<<DRAFT_REV>>", "model"),   # uncomment if speculative decode
    ("<<CORPUS_REPO>>", os.environ.get("CORPUS_REV", ""), "dataset"),
]
for repo, rev, kind in items:
    t = time.time()
    p = snapshot_download(repo, revision=rev or None, repo_type=kind, cache_dir=hub, max_workers=16)
    rev = rev or os.path.basename(p)
    files = os.listdir(p)
    idx = [f for f in files if f.endswith(".safetensors.index.json")]
    want = set(json.load(open(os.path.join(p, idx[0])))["weight_map"].values()) if idx else set()
    missing = [f for f in want if not os.path.isfile(os.path.join(p, f))]
    assert not missing, f"{repo}: missing {missing[:5]}"
    d = os.path.join(hub, ("datasets--" if kind == "dataset" else "models--") + repo.replace("/", "--"), "refs")
    os.makedirs(d, exist_ok=True); open(os.path.join(d, "main"), "w").write(rev)
    n = sum(1 for f in files if f.endswith(".safetensors"))
    print(f"[dl] ok {repo}@{rev[:10]} ({kind}): {n} shards, {len(files)} entries, {time.time()-t:.0f} s -> refs/main pinned", flush=True)
PY
rc=$?
du -shL $HUB/models--* $HUB/datasets--* 2>/dev/null
echo "[dl] $(date -u +%T) done rc=$rc"
