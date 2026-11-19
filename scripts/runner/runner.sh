#!/bin/bash
# One PR-test cell, end to end, on the Slurm node that runs it (docs/design.md sections 5-7). Deterministic: every
# parameter comes from <cell_dir>/cell.env and <cell_dir>/arbor_config.json, both rendered from the baseline record by
# scripts/submit_ab_test.py. Nothing runs on the login node; nothing here asks a person anything.
#
# usage (from the sbatch script): bash runner.sh <cell_dir>
# modes:  smoke = fetch, prepare, smoke (600 s AgentX + 200 GSM8K questions)
#         full  = fetch, prepare, before, campaign, after (3600 s AgentX + full GSM8K each side)
# state:  <cell_dir>/state holds the phase running now, `done`, or `failed:<phase>`; <cell_dir>/phases/<p>.{start,done}
#         mark each phase, so a re-submitted job resumes after the last finished phase.
# output: <cell_dir>/<phase>/{agg.json, bench_output.json, gsm8k.log, phase.json, launch.json, cmdline.txt, ...}
#
# Sources (each step's proven original): skills/arbor-ci-campaign-setup/templates/{dl,prepare,ab,campaign}.sh and their
# filled versions for dsv41f-c1-p90-sa2 / musespark-c16-ab (evidence/2026-11-19_0830_launch). Fixes over those: GSM8K
# with no result is fatal; the after-launch is Arbor's own live launch (serve_cmd.py), not a hand patch; no HF token in
# any container; corpus offline-load is a hard gate; arbor check runs first with the same flags as arbor run.
#
# Known limit (MVP): the container is removed when the job exits, so a job that dies between `campaign` and `after`
# loses the kept tree inside it; re-running `after` then needs the session restored (`python3 -m arbor restore`).
set -uo pipefail
CELL_DIR=${1:?usage: runner.sh <cell_dir>}
. "$CELL_DIR/cell.env"
LOGS=$CELL_DIR/logs; mkdir -p "$LOGS" "$CELL_DIR/phases"
# every error message goes where the job's output goes: a scheduler may send stderr elsewhere (Spur wrote it to
# $HOME/spur-<job>.out, so smoke job 299452's real error was not in the run's logs)
exec 2>&1
NL=$NL_ROOT; HUB=$NL/hf-hub
ARBOR=$CELL_DIR/arbor; INFX=$CELL_DIR/infx; VENV=$CELL_DIR/.venv; PY=$VENV/bin/python
PHASE=start; SERVER_UP=0

log() { echo "[runner] $(date -u +%FT%TZ) $*" | tee -a "$LOGS/runner.log"; }
die() { log "FATAL ($PHASE): $*"; echo "failed:$PHASE" > "$CELL_DIR/state"; exit 1; }
in_c() { docker exec "$CONTAINER" bash -c "$1"; }
# retry <tries> <command...>: for network steps (image pull, downloads, pip). A transient registry/CDN error must not
# end a multi-hour job (smoke job 299452: Docker Hub CDN "504 Gateway Time-out" on the first try). Backoff 30 s, 60 s, ...
retry() {
    local max=$1 n=1; shift
    until "$@"; do
        [ "$n" -ge "$max" ] && { log "gave up after $n tries: $*"; return 1; }
        log "try $n/$max failed, retrying in $((30 * n)) s: $*"; sleep $((30 * n)); n=$((n + 1))
    done
}

cleanup() {
    [ "$SERVER_UP" = 1 ] && stop_server
    # our container only (by its unique name); other users' containers are never touched
    docker rm -f "$CONTAINER" >/dev/null 2>&1 && log "container $CONTAINER removed"
}
trap cleanup EXIT

run_phase() {   # run_phase <name> <function> [args...]: skipped when already done
    PHASE=$1; shift
    if [ -f "$CELL_DIR/phases/$PHASE.done" ]; then log "$PHASE: done earlier, skipped"; return 0; fi
    echo "$PHASE" > "$CELL_DIR/state"; date -u +%FT%TZ > "$CELL_DIR/phases/$PHASE.start"
    log "$PHASE: start on $(hostname -s) job=${SLURM_JOB_ID:-none}"
    "$@" || die "phase returned non-zero"
    date -u +%FT%TZ > "$CELL_DIR/phases/$PHASE.done"; log "$PHASE: done"
}

# ---- fetch: pinned model / draft / corpus into node-local NVMe ----------------------------------------------------------
fetch_once() {   # one attempt; fetch.py skips what is already pinned and complete, so a retry resumes
    local items=("$MODEL_ID:$MODEL_REV:model" "$CORPUS_ID:$CORPUS_REV:dataset")
    [ -n "$DRAFT_ID" ] && items+=("$DRAFT_ID:$DRAFT_REV:model")
    # the token, if this user has one, stays in this one process's environment on the host
    local tok=; [ -f ~/.cache/huggingface/token ] && tok=$(cat ~/.cache/huggingface/token)
    if [ -n "$tok" ]; then   # an env assignment, not an argument: it never shows in `ps`
        HF_TOKEN="$tok" "$NL/dlvenv/bin/python" "$CELL_DIR/fetch.py" "$HUB" "${SHARED_HUB:--}" "${items[@]}" 2>&1 | tee -a "$LOGS/fetch.log"
    else
        "$NL/dlvenv/bin/python" "$CELL_DIR/fetch.py" "$HUB" "${SHARED_HUB:--}" "${items[@]}" 2>&1 | tee -a "$LOGS/fetch.log"
    fi
    return "${PIPESTATUS[0]}"
}

fetch() {
    mkdir -p "$NL" || return 1
    if [ ! -x "$NL/dlvenv/bin/python" ]; then
        python3 -m venv "$NL/dlvenv" && retry 3 "$NL/dlvenv/bin/pip" -q install 'huggingface_hub==2.4.0' || return 1
    fi
    retry 3 fetch_once
}

# archive_tree <repo> <commit> <dest>: the commit's files, plus every submodule at the commit its gitlink records,
# recursively. `git archive` alone leaves submodules empty: InferenceX's utils/aiperf is one, and aiperf's install then
# fails ("does not appear to be a Python project"; smoke job 299333, 2026-11-19).
archive_tree() {
    local repo=$1 sha=$2 dest=$3 mode type gsha path
    mkdir -p "$dest" && git -C "$repo" archive "$sha" | tar -x -C "$dest" || return 1
    while read -r mode type gsha path; do
        [ "$type" = commit ] || continue
        git -C "$repo/$path" cat-file -e "$gsha^{commit}" 2>/dev/null \
            || { log "submodule $path @ $gsha is not available in $repo/$path (git submodule update there)"; return 1; }
        archive_tree "$repo/$path" "$gsha" "$dest/$path" || return 1
        log "submodule $path @ ${gsha:0:10} extracted"
    done < <(git -C "$repo" ls-tree -r "$sha")
}

# ---- prepare: image, Arbor tree, InferenceX client, venv, container, framework tree, aiperf, corpus gate ----------------
prepare() {
    docker info >/dev/null 2>&1 || { log "no working docker daemon"; return 1; }
    docker ps --format '{{.Names}} {{.Image}}' | grep -v "^$CONTAINER " | sed 's/^/[runner] other container (not ours, not touched): /'
    local vram; vram=$(rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {print $NF}' | sort -n | tail -1)
    log "highest GPU VRAM% before start: ${vram:-?}"; [ "${vram:-0}" -le 1 ] || { log "a GPU is in use"; return 1; }

    local repo=${IMAGE_TAG%:*}
    mkdir -p "$CELL_DIR/.docker_anon"   # anonymous pulls: no registry credential of anyone's is used
    if [ -n "$IMAGE_DIGEST" ]; then
        docker image inspect "$repo@$IMAGE_DIGEST" >/dev/null 2>&1 || \
            retry 3 env DOCKER_CONFIG="$CELL_DIR/.docker_anon" docker pull -q "$repo@$IMAGE_DIGEST" || { log "image pull"; return 1; }
        docker tag "$repo@$IMAGE_DIGEST" "$IMAGE_TAG" || return 1
        docker image inspect --format '{{join .RepoDigests " "}}' "$IMAGE_TAG" | grep -q "$IMAGE_DIGEST" \
            || { log "image digest mismatch"; return 1; }
    else
        retry 3 env DOCKER_CONFIG="$CELL_DIR/.docker_anon" docker pull -q "$IMAGE_TAG" || return 1
    fi
    log "image $IMAGE_TAG @ ${IMAGE_DIGEST:-<tag only>}"

    if [ ! -f "$ARBOR/SOURCE_COMMIT" ]; then
        [ "$(sha256sum "$ARBOR_TAR" | cut -d' ' -f1)" = "$ARBOR_TAR_SHA256" ] || { log "$ARBOR_TAR checksum mismatch"; return 1; }
        mkdir -p "$ARBOR" && tar -xf "$ARBOR_TAR" -C "$ARBOR" || return 1
        echo "$ARBOR_SHA" > "$ARBOR/SOURCE_COMMIT"
        git -C "$ARBOR" init -q && git -C "$ARBOR" -c user.email=ab@local -c user.name=arbor-ab add -A \
            && git -C "$ARBOR" -c user.email=ab@local -c user.name=arbor-ab commit -q -m "AMD-AGI/Arbor $ARBOR_SHA (git archive)" || return 1
    fi
    [ "$(cat "$ARBOR/SOURCE_COMMIT")" = "$ARBOR_SHA" ] || { log "$ARBOR is not Arbor $ARBOR_SHA"; return 1; }
    log "Arbor $ARBOR_SHA at $ARBOR"

    if [ ! -f "$INFX/.source_commit" ]; then   # the marker is written last, so a half-built tree is rebuilt on resume
        git -C "$INFX_CLONE" cat-file -e "$INFX_SHA^{commit}" || { log "InferenceX $INFX_SHA not in $INFX_CLONE"; return 1; }
        rm -rf "$INFX" && archive_tree "$INFX_CLONE" "$INFX_SHA" "$INFX" || return 1
        echo "$INFX_SHA" > "$INFX/.source_commit"
    fi
    [ -f "$INFX/benchmarks/benchmark_lib.sh" ] || [ -d "$INFX/infx" ] || { log "InferenceX client has neither benchmark_lib.sh nor infx"; return 1; }
    log "InferenceX client $INFX_SHA at $INFX"

    if [ ! -x "$PY" ]; then
        python3 -m venv "$VENV" && retry 3 "$PY" -m pip install -q --upgrade pip \
            && retry 3 "$PY" -m pip install -q pyyaml claude-agent-sdk || return 1
    fi
    "$PY" -c 'import importlib.metadata as m; print("[runner] venv: pyyaml", m.version("pyyaml"), "claude-agent-sdk", m.version("claude-agent-sdk"))'

    # only the directories root in the container writes to are opened up (NFS root squash)
    mkdir -p "$CELL_DIR/sessions" "$CELL_DIR/mmap" "$CELL_DIR/agent_home" && chmod 777 "$CELL_DIR/sessions" "$CELL_DIR/mmap"
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]; then
        docker rm -f "$CONTAINER" >/dev/null 2>&1
        local render video; render=$(getent group render | cut -d: -f3); video=$(getent group video | cut -d: -f3)
        docker run -d --name "$CONTAINER" --entrypoint /bin/bash \
            --network host --ipc host --device /dev/kfd --device /dev/dri \
            --group-add "${render:-109}" --group-add "${video:-44}" \
            --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
            --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
            -v "$HUB":/hf-hub -v "$INFX":/workspace/InferenceX -v "$ARBOR":/workspace/Arbor \
            -v "$CELL_DIR/sessions":/workspace/sessions -v "$CELL_DIR/mmap":/aiperf_mmap_cache -v "$CELL_DIR":/workspace/exp \
            -e HF_HUB_CACHE=/hf-hub -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
            "$IMAGE_TAG" -c "sleep infinity" >/dev/null || { log "docker run"; return 1; }
    fi
    case "$MODEL_PATH" in
        /*) in_c "find -L '$MODEL_PATH' -maxdepth 1 -name '*.safetensors' -type f | head -1 | grep -q ." \
                || { log "no shards at $MODEL_PATH in the container"; return 1; } ;;
    esac

    local pkg=vllm; [ "$FRAMEWORK" = sglang ] && pkg=sglang
    # __path__, not __file__: SGLang can be a namespace package (None __file__); cd /tmp so the cwd cannot shadow it
    local where; where=$(in_c "cd /tmp && python3 -c 'import $pkg; print(list($pkg.__path__)[0])'" 2>/dev/null | tail -1)
    case "$WORKSPACE_ROOT" in "$where"|"${where%/python/$pkg}"|"${where%/$pkg}") ;; *)
        log "workspace_root $WORKSPACE_ROOT does not hold the installed $pkg ($where)"; return 1;; esac
    in_c "cd '$WORKSPACE_ROOT' && { git rev-parse --git-dir >/dev/null 2>&1 || { git init -q . && git config user.email arbor@local \
          && git config user.name Arbor && git config core.fileMode false && printf '__pycache__/\n*.pyc\n' > .git/info/exclude \
          && git add -A 2>/dev/null && git commit -q -m 'baseline: framework as shipped in $IMAGE_TAG'; }; } \
          && echo \"framework tree $(basename "$WORKSPACE_ROOT") at \$(git rev-parse --short HEAD), dirty: \$(git status --porcelain --untracked-files=no | wc -l)\"" \
        | sed 's/^/[runner] /' || return 1

    retry 3 in_c 'export INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX AIPERF_RUNTIME_DIR=/root/inferencex-agentic
          [ -f "$AIPERF_RUNTIME_DIR/.ready" ] && exit 0
          source /workspace/InferenceX/benchmarks/benchmark_lib.sh && install_agentic_deps && touch "$AIPERF_RUNTIME_DIR/.ready"' \
        >> "$LOGS/aiperf_install.log" 2>&1 || { log "aiperf install (see logs/aiperf_install.log)"; return 1; }

    # The corpus as the bench loads it: aiperf's weka loader calls datasets.load_dataset(<corpus>, split="train") with
    # the launch's HF_HUB_OFFLINE=1 (recorded server_env; Arbor gives the bench the server's env, claude_loop.py
    # _bench_env). Offline, `datasets` needs its own cache of the dataset (smoke job 299461 failed exactly there; the
    # baseline campaigns' patched bench had unset HF_HUB_OFFLINE instead, which would also let the server resolve
    # repo ids online). So: warm the container's datasets cache ONLINE at the pinned full revision, then prove the
    # OFFLINE load the bench will do. The bench's python is InferenceX's own $AIPERF_PYTHON (benchmark_lib.sh; its path
    # differs between InferenceX versions: smoke job 299409).
    local crev; crev=$(cat "$HUB/datasets--${CORPUS_ID//\//--}/refs/main" 2>/dev/null)
    [ -n "$crev" ] || { log "no pinned corpus revision in $HUB"; return 1; }
    local aienv="cd /tmp && export INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX AIPERF_RUNTIME_DIR=/root/inferencex-agentic \
          && source /workspace/InferenceX/benchmarks/benchmark_lib.sh >/dev/null && export HF_HUB_CACHE=/hf-hub"
    retry 3 in_c "$aienv && echo \"aiperf python: \$AIPERF_PYTHON\" && HF_HUB_OFFLINE=0 \"\$AIPERF_PYTHON\" -c \
          \"from datasets import load_dataset; d = load_dataset('$CORPUS_ID', revision='$crev', split='train'); print('corpus cache warmed:', len(d))\"" \
        >> "$LOGS/corpus_check.log" 2>&1 || { log "corpus cache warm-up failed (see logs/corpus_check.log)"; return 1; }
    in_c "$aienv && HF_HUB_OFFLINE=1 \"\$AIPERF_PYTHON\" -c \
          \"from datasets import load_dataset; d = load_dataset('$CORPUS_ID', split='train'); print('corpus offline load ok:', len(d))\"" \
        >> "$LOGS/corpus_check.log" 2>&1
    tail -1 "$LOGS/corpus_check.log" | tee -a "$LOGS/runner.log" | grep -q 'offline load ok' \
        || { log "corpus does not load offline after warm-up (see logs/corpus_check.log)"; return 1; }
    return 0
}

# ---- one measurement: fresh server (Arbor's own launch), AgentX replay, GSM8K ---------------------------------------
stop_server() {
    local pidf=/workspace/exp/$TAG/server.pid
    in_c "pid=\$(cat $pidf 2>/dev/null); [ -n \"\$pid\" ] && kill -TERM -- -\$pid 2>/dev/null; for i in \$(seq 1 24); do kill -0 \$pid 2>/dev/null || exit 0; sleep 5; done; exit 1" \
        || log "$TAG: server group still alive after 120 s"
    if [ "$FRAMEWORK" = vllm ]; then   # vLLM engine workers can outlive the API server; ours carry our GPU mask
        in_c "for p in \$(pgrep -f VLLM::); do grep -qa 'ROCR_VISIBLE_DEVICES=$GPUS' /proc/\$p/environ 2>/dev/null && kill -9 \$p; done; true"
    fi   # SGLang is never SIGKILLed: it holds KFD memory until a clean exit (arbor/frameworks/sglang.py)
    in_c "pkill -9 -f '[a]iperf' 2>/dev/null; true"
    SERVER_UP=0; log "$TAG: server stopped"
}

measure() {   # measure <tag> <duration_s> <gsm8k_questions>
    TAG=$1; local dur=$2 nq=$3 out=$CELL_DIR/$1 cin=/workspace/exp/$1
    mkdir -p "$out" && chmod 777 "$out"
    [ -f "$out/agg.json" ] && { log "$TAG: $out already has a result"; return 1; }
    in_c "curl -sf -m 3 localhost:$PORT/health >/dev/null" && { log "$TAG: a server already answers on $PORT"; return 1; }
    in_c "git -C '$WORKSPACE_ROOT' log -1 --format='%H %s'; git -C '$WORKSPACE_ROOT' status --porcelain --untracked-files=no | head -20" > "$out/tree.txt"
    (cd "$ARBOR" && PYTHONPATH=$ARBOR "$PY" "$CELL_DIR/serve_cmd.py" "$CELL_DIR/arbor_config.json" "$out" "$cin" --set "DURATION=$dur") \
        | tee -a "$LOGS/runner.log" || return 1
    local t0; t0=$(date -u +%FT%TZ)
    docker exec -d "$CONTAINER" bash "$cin/serve.sh" || return 1
    SERVER_UP=1; local T0; T0=$(date +%s)
    for i in $(seq 1 160); do
        in_c "curl -sf -m 5 localhost:$PORT/health >/dev/null" && break
        in_c "kill -0 \$(cat $cin/server.pid 2>/dev/null) 2>/dev/null" || { log "$TAG: server died"; tail -30 "$out/server.log"; return 1; }
        sleep 15
    done
    in_c "curl -sf -m 5 localhost:$PORT/health >/dev/null" || { log "$TAG: not healthy after 40 min"; return 1; }
    log "$TAG: healthy after $(( $(date +%s) - T0 )) s"
    in_c "tr '\0' ' ' < /proc/\$(cat $cin/server.pid)/cmdline" > "$out/cmdline.txt"

    # the bench gets the static config's server_env (Arbor: claude_loop.py _bench_env), DURATION set per phase
    "$PY" - "$CELL_DIR/arbor_config.json" "$dur" > "$out/bench.env" <<'PY' || return 1
import json, shlex, sys
c = json.load(open(sys.argv[1])); env = {k: str(v) for k, v in c["server_env"].items() if not k.startswith("_")}
env["DURATION"] = sys.argv[2]
print("".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items()), end="")
PY
    local T1; T1=$(date +%s)
    timeout "$BENCH_TIMEOUT" docker exec "$CONTAINER" bash -c ". $cin/bench.env; export PORT=$PORT MODEL_PATH='$MODEL_PATH' \
        MODEL_NAME='$MODEL_NAME' CONCURRENCY=$CONC NUM_PROMPTS=$NUM_PROMPTS BENCH_OUTPUT=$cin/bench_output.json; \
        cd /tmp && bash /workspace/Arbor/packs/agentx/validator/bench_agentx.sh" > "$out/bench.log" 2>&1
    log "$TAG: AgentX rc=$? after $(( $(date +%s) - T1 )) s"
    "$PY" - "$out" "$cin" <<'PY' || { log "$TAG: no AgentX aggregate"; return 1; }
import json, shutil, sys
out, cin = sys.argv[1:3]
b = json.load(open(f"{out}/bench_output.json"))
agg = b.get("agg_json_path") or ""
host = agg.replace(cin, out, 1) if agg.startswith(cin) else None
assert host, f"agg_json_path {agg!r} is not under {cin}"
shutil.copyfile(host, f"{out}/agg.json")
print(f"[runner] AgentX status={b.get('status')} total_throughput_per_gpu={b.get('total_throughput_per_gpu')} "
      f"e2e_norm_intvty_p90={b.get('e2e_norm_intvty_p90')} requests={b.get('num_requests_successful')}/{b.get('num_requests_total')}")
PY

    local T2; T2=$(date +%s)
    if [ "$GSM8K_TOOL" = infx_lm_eval ]; then   # public dataset fetched by lm-eval itself; no token in the container
        docker exec -e PORT="$PORT" -e MODEL="$MODEL_NAME" -e MODEL_NAME="$MODEL_NAME" -e EVAL_MAX_MODEL_LEN=4096 \
            -e EVAL_CONCURRENT_REQUESTS="$GSM8K_CONC" -e EVAL_RESULT_DIR="$cin/gsm8k" -e EVAL_LIMIT="$nq" -e HF_HUB_OFFLINE=0 \
            "$CONTAINER" bash -c "cd /workspace/InferenceX && timeout 5400 bash -c 'source benchmarks/benchmark_lib.sh && \
            run_lm_eval --port $PORT --task infx/evals/gsm8k.yaml --results-dir $cin/gsm8k'" > "$out/gsm8k.log" 2>&1
        ls "$out"/gsm8k/results_*.json "$out"/gsm8k/*/results_*.json >/dev/null 2>&1 || { log "$TAG: GSM8K produced no results"; return 1; }
    else
        in_c "cd /tmp && timeout 5400 python3 -m sglang.test.few_shot_gsm8k --num-questions $nq --port $PORT --parallel $GSM8K_CONC" \
            > "$out/gsm8k.log" 2>&1
        grep -qa '^Accuracy:' "$out/gsm8k.log" || { log "$TAG: GSM8K printed no accuracy"; return 1; }
    fi
    log "$TAG: GSM8K ($nq) after $(( $(date +%s) - T2 )) s: $(grep -aE '^Accuracy:|flexible-extract|strict-match' "$out/gsm8k.log" | tr -s ' ' | tr '\n' ' ' | cut -c1-200)"
    stop_server

    "$PY" - "$out/phase.json" "$TAG" "$t0" "$(date -u +%FT%TZ)" "$(head -1 "$out/tree.txt")" "$out/launch.json" \
        "$(hostname -s)" "${SLURM_JOB_ID:-}" "$dur" "$nq" <<'PY'
import json, sys
p, tag, t0, t1, tree, launch, node, job, dur, nq = sys.argv[1:11]
kept = json.load(open(launch))["live_workspace_config"]
json.dump({"phase": tag, "started_at": t0, "ended_at": t1, "tree": tree, "launch_from": "kept" if kept else "pristine",
           "node": node, "job": job, "duration_s": int(dur), "gsm8k_questions": int(nq)}, open(p, "w"), indent=2)
PY
}

# ---- campaign: Arbor under test, unattended (design.md section 7) ---------------------------------------------------
campaign() {
    [ -n "$GATEWAY_ENV" ] && [ -f "$GATEWAY_ENV" ] || { log "no LLM gateway env file configured (local.yaml arbor.gateway_env)"; return 1; }
    export HOME=$CELL_DIR/agent_home              # pins ~/.arbor/settings.env and the agents' ~/.claude
    mkdir -p "$HOME/.claude" && printf '{"env": {"BASH_DEFAULT_TIMEOUT_MS": "7200000", "BASH_MAX_TIMEOUT_MS": "7200000"}}\n' > "$HOME/.claude/settings.json"
    export BASH_DEFAULT_TIMEOUT_MS=7200000 BASH_MAX_TIMEOUT_MS=7200000
    set -a; . "$GATEWAY_ENV"; set +a             # variables only; never echoed, never written into a container
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    local flags=(--config "$CELL_DIR/arbor_config.json" --no-review-gate)   # unattended by design (design.md section 7)
    [ -n "$OPT_OBJECTIVE" ] && flags+=(--opt-objective "$OPT_OBJECTIVE")
    # Arbor's AGENTS.md: check with the same flags first; exit 2 refuses; nothing may wait; `lacks` must be empty
    (cd "$ARBOR" && "$PY" -m arbor check "${flags[@]}") > "$CELL_DIR/campaign_check.json" 2> "$LOGS/campaign_check.err"
    local rc=$?
    "$PY" - "$CELL_DIR/campaign_check.json" "$rc" "${ALLOW_LACKS:-0}" <<'PY' | tee -a "$LOGS/runner.log" || return 1
import json, sys
doc = json.load(open(sys.argv[1])); rc = int(sys.argv[2]); allow_lacks = sys.argv[3] == "1"
lacks = (doc.get("makeup") or {}).get("lacks") or []
print(f"[runner] arbor check rc={rc} ok={doc.get('ok')} refused={doc.get('refused')} waits={doc.get('waits')} "
      f"lacks={lacks} warnings={len(doc.get('warnings') or [])}")
for w in doc.get("warnings") or []:
    print(f"[runner] arbor check warning: {w}")
sys.exit(0 if rc == 0 and doc.get("ok") and not doc.get("waits") and (allow_lacks or not lacks) else 1)
PY
    (cd "$ARBOR" && "$PY" -m arbor run "${flags[@]}" --cooldown 30) > "$LOGS/campaign.log" 2>&1
    log "campaign: arbor run rc=$?"
    local ex=$CELL_DIR/sessions/full/exit.json
    [ -f "$ex" ] && cp "$ex" "$CELL_DIR/campaign_exit.json" && log "campaign exit: $(tr -d '\n' < "$ex" | cut -c1-300)"
    [ -f "$ex" ] || { log "campaign wrote no exit.json"; return 1; }
}

case "$MODE" in
    smoke) run_phase fetch fetch; run_phase prepare prepare; run_phase smoke measure smoke "$DUR_SMOKE" "$NQ_SMOKE" ;;
    full)  run_phase fetch fetch; run_phase prepare prepare; run_phase before measure before "$DUR_FULL" "$NQ_FULL"
           run_phase campaign campaign; run_phase after measure after "$DUR_FULL" "$NQ_FULL" ;;
    *) die "unknown MODE '$MODE'" ;;
esac
echo done > "$CELL_DIR/state"; PHASE=end; log "cell $CELL: all phases done"
