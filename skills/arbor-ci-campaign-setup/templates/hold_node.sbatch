#!/bin/bash
#SBATCH --job-name=<<CODENAME>>
#SBATCH --partition=amd-spur-v2
#SBATCH --qos=amd-hyperloom-qos
#SBATCH --account=amd-hyperloom
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:mi355x:8
#SBATCH --time=1-00:00:00
#SBATCH --output=/shared_nfs/qiangh/mywork/<<WORKSPACE NAME>>/experiments/<<CODENAME>>/logs/hold_%j.out
#
# Hold one MI355X node (exclusive) for <<CODENAME>>: <<one-line description>>. Submitted at 1 day (the QOS max wall
# blocks a pending 3-day job); raise with `scontrol update JobId=<id> TimeLimit=3-00:00:00` once RUNNING.
#
# Node agent: every 20 s this job looks in ctrl/ for *.sh files. Each one runs once, in the background, as a child of
# this batch step; it is renamed to *.sh.started and its output goes to logs/<name>.<jobid>.log. Touch ctrl/RELEASE to
# end the job early. See the `crusoe-arbor-run` skill for the full mechanics (atomic ctrl/ drop, stopping cleanly).
#
# Never `bash -l`: a login shell sources the profile off /home, which has wedged on this cluster.
set -u
ROOT=/shared_nfs/qiangh/mywork/<<WORKSPACE NAME>>/experiments/<<CODENAME>>
CTRL=$ROOT/ctrl
LOGS=$ROOT/logs
echo "[hold] node=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is)"
rocm-smi --showproductname 2>/dev/null | grep -E "GPU\[" | head -8
echo "$(hostname) ${SLURM_JOB_ID:-none} $(date -Is)" > "$ROOT/NODE"
while [ ! -e "$CTRL/RELEASE" ]; do
    for f in "$CTRL"/*.sh; do
        [ -e "$f" ] || continue
        name=$(basename "$f" .sh)
        mv "$f" "$f.started"
        echo "[hold] $(date -Is) start $name"
        ( cd "$ROOT" && exec bash "$f.started" ) > "$LOGS/$name.${SLURM_JOB_ID:-none}.log" 2>&1 < /dev/null &
        echo "$!" > "$CTRL/$name.pid"
    done
    sleep 20
done
echo "[hold] RELEASE seen at $(date -Is)"
