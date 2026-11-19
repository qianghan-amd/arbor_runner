#!/bin/bash
# Provision the node for <<CODENAME>>. Idempotent; run on the GPU node through ctrl/. Steps:
#   1. node checks: docker, GPUs idle, other users' containers listed (never touched)
#   2. the CI image by digest (anonymous pull if the shared Docker Hub token has expired)
#   3. Arbor at the pinned commit (archive + a local git commit so `has_changes` works), our pack and config
#   4. the driver's venv (pyyaml, claude-agent-sdk)
#   5. the HF cache on NVMe: model(+draft), corpus at their pins (dl.sh), every shard resolving
#   6. the container: the NVMe hub at /hf-hub, InferenceX, Arbor, sessions
#   7. stack versions; the framework tree (workspace_root) clean at its HEAD; editable packages into the config
#   8. the aiperf environment for the bench (InferenceX's install_agentic_deps)
#   9. ruppell for the profiler skill (not fatal)
#
# Adapted from experiments/musespark-c1-p90-sa/prepare.sh (SGLang). Where this differs for vLLM, the alternate
# command is given inline as a comment — pick the one matching env.sh's framework.
set -uo pipefail
. "$(dirname "$0")/env.sh"
log "prepare on $(hostname -s) job=${SLURM_JOB_ID:-none}"

# ---- 1. node -----------------------------------------------------------------------------------------
docker info >/dev/null 2>&1 || die "no working docker daemon"
docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | grep -v "^$CONTAINER " | sed 's/^/[prepare] other container (not ours, not touched): /'
vram=$(rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {print $NF}' | sort -n | tail -1)
log "highest GPU VRAM% before start: ${vram:-?}"
[ "${vram:-0}" -le 1 ] || die "a GPU is in use before start"

# ---- 2. image ----------------------------------------------------------------------------------------
if ! docker image inspect "${IMAGE_TAG%%:*}@$IMAGE_DIGEST" >/dev/null 2>&1; then
    mkdir -p "$EXP/.docker_anon"
    DOCKER_CONFIG=$EXP/.docker_anon docker pull -q "${IMAGE_TAG%%:*}@$IMAGE_DIGEST" || die "image pull"
fi
docker tag "${IMAGE_TAG%%:*}@$IMAGE_DIGEST" "$IMAGE_TAG" || die "image tag"
case "$(docker image inspect --format '{{join .RepoDigests " "}}' "$IMAGE_TAG")" in
    *"$IMAGE_DIGEST"*) log "image ok: $IMAGE_TAG = $IMAGE_DIGEST";; *) die "image digest mismatch";; esac

# ---- 3. Arbor ----------------------------------------------------------------------------------------
# <<EXP/stage/arbor_*.tgz must exist: an archive of the pinned Arbor commit/branch, staged before this script runs.>>
if [ ! -f "$ARBOR/SOURCE_COMMIT" ]; then
    mkdir -p "$ARBOR" && tar -xzf "$EXP/stage/<<arbor archive filename>>" -C "$ARBOR" || die "Arbor archive"
    echo "$ARBOR_SHA" > "$ARBOR/SOURCE_COMMIT"
    git -C "$ARBOR" init -q && git -C "$ARBOR" -c user.email=qh@local -c user.name=qh add -A \
        && git -C "$ARBOR" -c user.email=qh@local -c user.name=qh commit -q -m "Arbor <<BRANCH>> $ARBOR_SHA (archive)" || die "Arbor git"
    log "Arbor $ARBOR_SHA at $ARBOR"
fi
[ "$(cat "$ARBOR/SOURCE_COMMIT")" = "$ARBOR_SHA" ] || die "$ARBOR is not Arbor $ARBOR_SHA"
mkdir -p "$ARBOR/packs/$PACK" && cp -a "$ARBOR/packs/agentx/validator" "$ARBOR/packs/agentx/tools" "$ARBOR/packs/$PACK/" \
    && cp "$EXP/stage/pack.yaml" "$ARBOR/packs/$PACK/pack.yaml" \
    && cp "$EXP/stage/bench_agentx.sh" "$ARBOR/packs/$PACK/validator/bench_agentx.sh" \
    && mkdir -p "$(dirname "$CONFIG")" && cp "$EXP/stage/config.json" "$CONFIG" || die "pack/config"
[ -z "$(grep -rlI $'\r' "$ARBOR/packs/$PACK" "$CONFIG")" ] || die "CRLF in pack/config"
log "pack $PACK and config $CONFIG_REL installed"

# ---- 4. venv -----------------------------------------------------------------------------------------
if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV" && "$PY" -m pip install -q --upgrade pip && "$PY" -m pip install -q pyyaml claude-agent-sdk || die "venv"
fi
"$PY" - <<'PY' || die "venv"
import importlib.metadata as m
print(f"[prepare] venv: pyyaml {m.version('pyyaml')}, claude-agent-sdk {m.version('claude-agent-sdk')}")
PY

# ---- 5. HF cache -------------------------------------------------------------------------------------
# Skip this whole step if following the dsv41f-c32-tput shared-cache style (no node-local dl.sh).
for r in "models--${MODEL_ID//\//--} $MODEL_REV"; do   # add "models--${DRAFT_ID//\//--} $DRAFT_REV" if speculative decode
    set -- $r
    [ "$(cat "$HUB_NL/$1/refs/main" 2>/dev/null)" = "$2" ] || die "$1: refs/main is not $2 (run dl.sh)"
    n=$(find -L "$HUB_NL/$1/snapshots/$2" -maxdepth 1 -name '*.safetensors' -type f | wc -l)
    k=$(find "$HUB_NL/$1/snapshots/$2" -maxdepth 1 -name '*.safetensors' | wc -l)
    [ "$n" -gt 0 ] && [ "$n" -eq "$k" ] || die "$1: $n of $k shards resolve"
    log "$1 @ ${2:0:10}: $n shards resolve"
done
CORPUS_DIR=datasets--<<corpus repo, slashes to double-dash>>
[ -s "$HUB_NL/$CORPUS_DIR/refs/main" ] && [ -n "$(find -L "$HUB_NL/$CORPUS_DIR/snapshots" -type f | head -1)" ] || die "corpus missing (run dl.sh)"
log "corpus $CORPUS_DIR @ $(cat "$HUB_NL/$CORPUS_DIR/refs/main" | cut -c1-10)"

# ---- 6. container ------------------------------------------------------------------------------------
mkdir -p "$SESSIONS" "$MMAP_CACHE" "$AGENT_HOME" "$EXP/runs" && chmod 777 "$SESSIONS" "$MMAP_CACHE" "$EXP/runs"
if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    render=$(getent group render | cut -d: -f3); video=$(getent group video | cut -d: -f3)
    docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
        --network host --ipc host --device /dev/kfd --device /dev/dri \
        --group-add "${render:-109}" --group-add "${video:-44}" \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
        --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$HUB_NL":/hf-hub \
        -v "$INFX":/workspace/InferenceX \
        -v "$ARBOR":/workspace/Arbor \
        -v "$SESSIONS":/workspace/sessions \
        -v "$MMAP_CACHE":/aiperf_mmap_cache \
        -v "$EXP":/workspace/exp \
        -v "$ARBOR/.claude/skills/profiler/ruppell":/ruppell:ro \
        -e HF_HUB_CACHE=/hf-hub -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -e RUPPELL_BUILD=/root/ruppell-work/build \
        "$IMAGE_TAG" -c "sleep infinity" >/dev/null || die "docker run failed"
    log "container $CONTAINER started"
else
    log "container $CONTAINER already running"
fi

# ---- 7. versions, workspace, editable packages -------------------------------------------------------
docker exec "$CONTAINER" bash -c '
  echo "# stack versions, $(date -Is)"
  echo "python3: $(command -v python3) $(python3 -V 2>&1)"
  cat /opt/rocm/.info/version 2>/dev/null | sed "s/^/rocm: /"
  python3 -c "import sglang, torch; print(\"sglang:\", sglang.__version__, sglang.__file__); print(\"torch:\", torch.__version__, torch.version.hip)"
  # vLLM equivalent: python3 -c "import vllm, torch; print(\"vllm:\", vllm.__version__, vllm.__file__); print(\"torch:\", torch.__version__, torch.version.hip)"
  pip list 2>/dev/null | grep -iE "^(sglang|vllm|torch|aiter|amd-aiter|flydsl|triton|transformers|lm.eval) "
  git -C <<WORKSPACE_ROOT>> log -1 --format="framework tree: %H %ci" 2>/dev/null
' > "$LOGS/stack_versions.txt" 2>&1
log "stack: $(grep -E '^(python3|rocm|sglang|vllm|framework tree):' "$LOGS/stack_versions.txt" | tr '\n' ' ' | cut -c1-260)"
docker exec "$CONTAINER" bash -c "git -C $WORKSPACE_ROOT config user.email >/dev/null || { git -C $WORKSPACE_ROOT config user.email arbor@local; git -C $WORKSPACE_ROOT config user.name Arbor; }"
# The image may ship its framework tree with modified tracked files (the image's own state). Commit them as a baseline,
# so Arbor's diffs start from the code as shipped; the list and the diff go to the logs.
dirty=$(docker exec "$CONTAINER" bash -c "git -C $WORKSPACE_ROOT status --porcelain --untracked-files=no")
if [ -n "$dirty" ]; then
    docker exec "$CONTAINER" bash -c "git -C $WORKSPACE_ROOT diff" > "$LOGS/image_framework_tree.diff"
    log "image's framework tree as shipped has $(echo "$dirty" | wc -l) modified tracked files: $(echo "$dirty" | awk '{print $2}' | tr '\n' ' ') (diff: logs/image_framework_tree.diff)"
    docker exec "$CONTAINER" bash -c "cd $WORKSPACE_ROOT && git add -u && git commit -q -m 'baseline: working-tree changes as shipped in $IMAGE_TAG@$IMAGE_DIGEST'" || die "baseline commit"
fi
log "workspace $WORKSPACE_ROOT at $(docker exec "$CONTAINER" git -C $WORKSPACE_ROOT log -1 --format='%h %s' | cut -c1-90)"
"$PY" - "$CONFIG" "$CONTAINER" <<'PY' || die "editable packages"
import json, subprocess, sys
p, c = sys.argv[1], sys.argv[2]
cfg = json.load(open(p))
pk = [e for e in cfg.get("editable_packages", []) if e.get("name") not in ("aiter", "flydsl")]
for name in ("aiter", "flydsl"):   # drop entries that don't apply to this framework/image
    r = subprocess.run(["docker", "exec", c, "python3", "-c", f"import os, {name}; print(os.path.dirname({name}.__file__))"],
                       capture_output=True, text=True)
    path = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
    if path:
        pk.append({"name": name, "path": path, "rebuild": None})
    print(f"[prepare] editable {name}: {path or 'not importable'}")
cfg["editable_packages"] = pk
json.dump(cfg, open(p, "w"), indent=2)
PY

# ---- 8. aiperf (bench) ---------------------------------------------------------------------------------
docker exec "$CONTAINER" bash -c '
  export INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX AIPERF_RUNTIME_DIR=/root/inferencex-agentic
  source /workspace/InferenceX/benchmarks/benchmark_lib.sh
  if [ -x "${AIPERF_CLI:-}" ] && [ -f "$AIPERF_RUNTIME_DIR/.ready" ]; then echo "aiperf present"; exit 0; fi
  install_agentic_deps && touch "$AIPERF_RUNTIME_DIR/.ready" && "$AIPERF_CLI" --version 2>&1 | tail -1
' 2>&1 | tail -2 | sed 's/^/[prepare] aiperf: /'
docker exec "$CONTAINER" bash -c 'test -f /root/inferencex-agentic/.ready' || die "aiperf environment not ready"
[ -z "$(git -C "$INFX" status --porcelain)" ] && log "InferenceX still clean at $(git -C "$INFX" rev-parse --short HEAD)" || log "WARNING InferenceX tree changed: $(git -C "$INFX" status --porcelain | head -3 | tr '\n' ' ')"
docker exec "$CONTAINER" python3 -c "import sglang.test.few_shot_gsm8k" 2>/dev/null && log "GSM8K tool present (sglang.test.few_shot_gsm8k)" || log "WARNING GSM8K tool not importable (check the vLLM-equivalent eval entry point instead if this is a vLLM campaign)"

# ---- 9. ruppell, not fatal -----------------------------------------------------------------------------
docker exec "$CONTAINER" bash -c '
  [ -f /ruppell/runtime/build.sh ] || { echo "no vendored ruppell at /ruppell"; exit 0; }
  mkdir -p /root/ruppell-work && bash /ruppell/runtime/build.sh > /root/ruppell-work/build.log 2>&1 \
    && bash /ruppell/runtime/build_cpu_ops.sh >> /root/ruppell-work/build.log 2>&1; echo "rc=$? (RUPPELL_BUILD=$RUPPELL_BUILD)"
' 2>&1 | tail -1 | sed 's/^/[prepare] ruppell build: /'

touch "$EXP/PREPARED"
log "prepare done"
