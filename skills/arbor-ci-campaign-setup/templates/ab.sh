#!/bin/bash
# Smoke and before/after measurements for <<CODENAME>> (full AgentX at <<CONC>> and GSM8K, before and after the
# campaign). Serves the tree in the container exactly as Arbor builds the launch (arbor/frameworks/{sglang,vllm}.py:
# the server command, with the config's server_env and ROCR_VISIBLE_DEVICES=<gpus>, no HIP_VISIBLE_DEVICES:
# arbor/frameworks/base.py), on a fresh server; runs the pack's bench_agentx.sh (the AgentX replay, DURATION s) and
# then the GSM8K eval; stops that server (its own process group only).
#
# Adapted from experiments/musespark-c1-p90-sa/ab.sh (SGLang). The vLLM-equivalent serve command and GSM8K eval entry
# point are marked inline — swap them in if env.sh's framework is vLLM.
#
# usage: bash ab.sh <tag> <duration_s> <gsm8k_questions>      e.g.  before 3600 1319 | after 3600 1319 | smoke 600 200
set -uo pipefail
. "$(dirname "$0")/env.sh"
TAG=${1:?tag}; DUR=${2:?duration}; NQ=${3:?gsm8k questions}; GPUS=<<GPU ids reserved for ab.sh, e.g. 0,1,2,3 — leave the rest free for Arbor's own probes/profiling>>
OUT_H=$EXP/runs/$TAG; OUT_C=/workspace/exp/runs/$TAG
[ -e "$OUT_H/bench_output.json" ] && die "$OUT_H already has a result"
mkdir -p "$OUT_H" && chmod 777 "$OUT_H"
docker exec "$CONTAINER" bash -c "curl -sf -m 3 localhost:$PORT/health >/dev/null" && die "a server already answers on port $PORT"
docker exec "$CONTAINER" bash -c "git -C $WORKSPACE_ROOT log -1 --format='tree %H %s'; git -C $WORKSPACE_ROOT status --porcelain --untracked-files=no | head -20" > "$OUT_H/tree.txt"
# THE LAUNCH TO SERVE: once a campaign has started, Arbor keeps the live launch in <workspace_root>/arbor/config.json (seeded
# at init, edited by every KEEP that changes a flag or env, committed with the tree). arbor/execute/bench.py: its
# server_flags win; its server_env is merged over $CONFIG's. $CONFIG alone is only the campaign's INPUT: an "after" run built
# from it silently drops every config-level KEEP (kimik3-c1-p90-sa, 2026-11-16: 3 kept flag/env changes missing). So: use
# the workspace config when it exists (AB_CONFIG overrides), and record which one was used.
LAUNCH=${AB_CONFIG:-}
if [ -z "$LAUNCH" ] && docker exec "$CONTAINER" test -f "$WORKSPACE_ROOT/arbor/config.json"; then
    docker exec "$CONTAINER" cat "$WORKSPACE_ROOT/arbor/config.json" > "$OUT_H/workspace_config.json" && LAUNCH=$OUT_H/workspace_config.json
fi
python3 - "$CONFIG" "$OUT_H/server.env" "$OUT_H/serve.sh" "$GPUS" "$PORT" "$DUR" "$OUT_C" "$LAUNCH" <<'PY' || die "config"
import json, shlex, sys
cfg, envf, serve, gpus, port, dur, outc, launch = sys.argv[1:9]
c = json.load(open(cfg))
if launch:   # Arbor's rule (arbor/execute/bench.py _live_flags/_live_env): workspace server_flags win unless the key is
    w = json.load(open(launch))      # absent/None ("" is a deliberate clear); server_env is merged, workspace values win
    if w.get("server_flags") is not None:
        c["server_flags"] = str(w["server_flags"])
    c["server_env"] = {**{k: str(v) for k, v in c["server_env"].items()}, **{k: str(v) for k, v in (w.get("server_env") or {}).items()}}
print(f"[ab] launch from {launch or cfg + ' (no workspace arbor/config.json yet: pre-campaign)'}")
env = {k: v for k, v in c["server_env"].items() if not k.startswith("_")}; env["DURATION"] = dur; env["ROCR_VISIBLE_DEVICES"] = gpus
with open(envf, "w") as f:
    for k, v in env.items(): f.write(f"export {k}={shlex.quote(str(v))}\n")
    f.write("unset HIP_VISIBLE_DEVICES\n")
# SGLang:
cmd = (f"python3 -m sglang.launch_server --model-path {c['model_path']} --served-model-name {c['model_name']} "
       f"--host 0.0.0.0 --port {port} {c['server_flags']}")
# vLLM equivalent (swap in if env.sh's framework is vLLM; arbor/frameworks/vllm.py adds --tensor-parallel-size itself, so
# server_flags must not repeat it):
# cmd = (f"vllm serve {c['model_path']} --served-model-name {c['model_name']} "
#        f"--host 0.0.0.0 --port {port} --tensor-parallel-size {c['tp']} {c['server_flags']}")
open(serve, "w").write(f"#!/bin/bash\n. {outc}/server.env\ncd /tmp\nexec setsid {cmd} > {outc}/server.log 2>&1\n")
print(f"[ab] server: {cmd}")
PY
log "$TAG: tree $(head -1 "$OUT_H/tree.txt" | cut -c1-80); server on GPUs $GPUS port $PORT; AgentX ${DUR}s; GSM8K $NQ"
docker exec -d "$CONTAINER" bash "$OUT_C/serve.sh" || die "docker exec"
T0=$(date +%s)
for i in $(seq 1 160); do
    docker exec "$CONTAINER" bash -c "curl -sf -m 5 localhost:$PORT/health >/dev/null" && break
    # SGLang process-alive check; vLLM equivalent: pgrep -f "vllm serve .*--port $PORT "
    docker exec "$CONTAINER" pgrep -f "sglang.launch_server .*--port $PORT " >/dev/null || { log "$TAG: FATAL server died"; tail -30 "$OUT_H/server.log"; exit 1; }
    sleep 15
done
docker exec "$CONTAINER" bash -c "curl -sf -m 5 localhost:$PORT/health >/dev/null" || { log "$TAG: FATAL not healthy after 40 min"; tail -20 "$OUT_H/server.log"; exit 1; }
log "$TAG: healthy after $(( $(date +%s) - T0 )) s"

stop_server() {
    # SGLang process-group stop + stragglers-by-GPU-mask sweep; vLLM equivalent: match "vllm serve .*--port $PORT " and
    # whatever the vLLM worker process name is instead of "sglang".
    docker exec "$CONTAINER" bash -c "pid=\$(pgrep -f 'sglang.launch_server .*--port $PORT ' | head -1); [ -n \"\$pid\" ] && kill -TERM -- -\$(ps -o pgid= -p \$pid | tr -d ' ') 2>/dev/null; sleep 30; \
      for p in \$(pgrep -f 'sglang'); do grep -qa 'ROCR_VISIBLE_DEVICES=$GPUS' /proc/\$p/environ 2>/dev/null && kill -9 \$p; done; true"
    # aiperf (the bench client bench_agentx.sh launches) can leak past this: the container is started with
    # --entrypoint /bin/bash -c "sleep infinity", so PID 1 never reaps anything, and an aiperf process whose parent
    # already exited is reparented to PID 1 and sits as a zombie (STAT Z) instead of disappearing. A zombie still
    # matches `pgrep -f aiperf`, so any later "is a bench still running" check that uses plain pgrep (ours or a
    # restarted/fixed-up copy of this script) can misread "nothing left but zombies" as "still running" and refuse
    # to proceed. `kill` is a no-op on a zombie (there is no process left to signal), so we don't try to kill one —
    # we only wait out and confirm the non-zombie aiperf stragglers are gone, filtering STAT so zombies don't count.
    docker exec "$CONTAINER" bash -c "pkill -9 -f '[a]iperf' 2>/dev/null; \
      for i in \$(seq 1 20); do ps -eo pid=,stat=,cmd= | awk '\$2 !~ /^Z/ && \$0 ~ /[a]iperf/' | grep -q . || exit 0; sleep 3; done; \
      echo '[ab] WARNING: non-zombie aiperf stragglers still present after 60s' >&2"
    log "$TAG: server stopped"
}
trap 'stop_server' EXIT

T1=$(date +%s)
docker exec "$CONTAINER" bash -c ". $OUT_C/server.env; export PORT=$PORT MODEL_PATH=$MODEL_ID MODEL_NAME=$MODEL_ID CONCURRENCY=<<CONC>> NUM_PROMPTS=<<prompt count, e.g. 320>> \
  BENCH_OUTPUT=$OUT_C/bench_output.json; cd /tmp && bash /workspace/Arbor/packs/$PACK/validator/bench_agentx.sh" > "$OUT_H/bench.log" 2>&1
log "$TAG: AgentX rc=$? after $(( $(date +%s) - T1 )) s: $(grep -a '^\[bench_agentx\] p90_intvty' "$OUT_H/bench.log" | tail -1)"
log "$TAG: corpus revision $(cat "$HUB_NL/$CORPUS_DIR/refs/main" | cut -c1-10)"   # CORPUS_DIR as defined in prepare.sh step 5; skip if shared-cache style
python3 - "$OUT_H/bench_output.json" <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: sys.exit(f"[ab] no bench_output.json: {e}")
keys = ("e2e_norm_intvty_p90", "e2e_norm_intvty_p50", "total_throughput", "output_throughput", "num_requests_successful",
        "num_requests_total", "num_requests_measured", "kv_cache_hit_rate")
print("[ab] " + ", ".join(f"{k}={d.get(k)}" for k in keys))
PY

T2=$(date +%s)
# SGLang's GSM8K tool; vLLM equivalent: use InferenceX's own GSM8K eval entry point (lm-eval-harness based — see
# inferencex-e2e/perf-changelog.yaml / utils/evals/gsm8k.yaml for the exact invocation Arbor's own pack uses).
docker exec "$CONTAINER" bash -c "cd /tmp && timeout 5400 python3 -m sglang.test.few_shot_gsm8k --num-questions $NQ --port $PORT --parallel 64" > "$OUT_H/gsm8k.log" 2>&1
log "$TAG: GSM8K ($NQ) rc=$? after $(( $(date +%s) - T2 )) s: $(grep -aiE 'accuracy|invalid|latency' "$OUT_H/gsm8k.log" | tail -3 | tr '\n' ' ')"
