#!/bin/bash
# The 24 h Arbor standalone campaign of <<CODENAME>>, on a provisioned node (prepare.sh ran; the "before" measurement
# exists). Adapted from experiments/musespark-c1-p90-sa/campaign.sh.
#   1. the budget: what is left of the campaign's 24 h, and of this job less a buffer (the "after" AgentX + GSM8K)
#   2. the agents' Bash tool may wait 2 h (a long replay with its setup outlasts the CLI's 30 min default)
#   3. the driver: python -m arbor run, unattended (no review gate). touch $SESSION/STOP ends it after the iteration.
set -uo pipefail
. "$(dirname "$0")/env.sh"
CAMPAIGN_MIN=${CAMPAIGN_MIN:-1440}
[ -s "$EXP/runs/before/bench_output.json" ] || die "no before measurement"
[ -f "$SESSION/STOP" ] && die "operator STOP present at $SESSION/STOP"

# ---- 1. budget ---------------------------------------------------------------------------------------------
[ -s "$EXP/CAMPAIGN_T0" ] || date +%s > "$EXP/CAMPAIGN_T0"
used=$(( ($(date +%s) - $(cat "$EXP/CAMPAIGN_T0")) / 60 ))
left_campaign=$(( CAMPAIGN_MIN - used ))
left_job=$( { squeue -j "${SLURM_JOB_ID:-0}" -h -o %L 2>/dev/null || true; } | awk -F'[-:]' '
    NF==4 {print $1*1440+$2*60+$3; exit} NF==3 {print $1*60+$2; exit} NF==2 {print $1; exit}')
left_job=${left_job:-$left_campaign}
BUFFER_MIN=<<minutes reserved for the after-measurement, e.g. 180>>
budget=$(( left_campaign < left_job - BUFFER_MIN ? left_campaign : left_job - BUFFER_MIN ))
[ "$budget" -gt 60 ] || die "budget $budget min: campaign left $left_campaign, job left $left_job"
"$PY" - "$CONFIG" "$budget" <<'PYEOF'
import json, sys
p, b = sys.argv[1], int(sys.argv[2])
c = json.load(open(p)); c["time_budget_minutes"] = b; json.dump(c, open(p, "w"), indent=2)
PYEOF
log "budget $budget min (campaign used $used of $CAMPAIGN_MIN, job left $left_job)"

# ---- 2. agents' settings -----------------------------------------------------------------------------------
mkdir -p "$AGENT_HOME/.claude"
"$PY" - "$AGENT_HOME/.claude/settings.json" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault("env", {}).update({"BASH_DEFAULT_TIMEOUT_MS": "7200000", "BASH_MAX_TIMEOUT_MS": "7200000"})
json.dump(d, open(p, "w"), indent=2)
print(f"[campaign] {p}: env {d['env']}")
PYEOF

# ---- 3. driver ---------------------------------------------------------------------------------------------
cleanup() {
    log "cleanup: stopping the server and the bench in $CONTAINER"
    [ -n "${DRIVER:-}" ] && kill -TERM "$DRIVER" 2>/dev/null
    # [x]yz patterns: a plain `pkill -f aiperf` matches this bash -c line itself and kills it first.
    # SGLang process names shown; vLLM equivalent: swap "sglang.launch_server"/"sglang::" for vLLM's process names.
    docker exec "$CONTAINER" bash -c 'pkill -f [a]iperf; pkill -f "[s]glang.launch_server"; sleep 10; pkill -9 -f "[s]glang::"; true' 2>/dev/null
}
trap 'cleanup; exit 0' TERM INT
cd "$ARBOR" || exit 1
set -a; . "$GATEWAY_ENV"; set +a
export HOME=$AGENT_HOME
export TMPDIR=/tmp/qiangh-arbor-<<short codename abbrev>> && mkdir -p "$TMPDIR" && chmod 700 "$TMPDIR"
export CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1 BASH_DEFAULT_TIMEOUT_MS=7200000 BASH_MAX_TIMEOUT_MS=7200000
export PYTHONPATH="$ARBOR" PATH="$VENV/bin:$PATH" ARBOR_REVIEW_GATE=0
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
log "arbor driver starting: Arbor $(cat "$ARBOR/SOURCE_COMMIT" | cut -c1-10), config $CONFIG_REL"
"$PY" -m arbor run --config "$CONFIG_REL" --cooldown 30 --no-review-gate &
DRIVER=$!
wait "$DRIVER"
rc=$?
log "driver exited rc=$rc"
cleanup
