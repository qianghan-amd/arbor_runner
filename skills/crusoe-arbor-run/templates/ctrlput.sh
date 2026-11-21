#!/bin/bash
# Drop a script into an experiment's ctrl/ atomically, so the hold job's node agent never runs a half-written file.
# usage: LOGIN=<user>@<Crusoe login host> EXP_ROOT=<experiments dir> bash ctrlput.sh <local script> <name> <code name>
#   e.g. LOGIN=you@crsuse2-slog-006.crusoe.amd.com EXP_ROOT=/home/you/arbor_dsv41flash/experiments
#   -> $LOGIN:$EXP_ROOT/<code name>/ctrl/<name>.sh ; output: logs/<name>.<jobid>.log
set -eu
SRC=${1:?local script}; NAME=${2:?name}; CODE=${3:?experiment code name}
LOGIN=${LOGIN:?set LOGIN=<user>@<Crusoe login host>}; EXP_ROOT=${EXP_ROOT:?set EXP_ROOT=<experiments dir on the cluster>}
C=$EXP_ROOT/$CODE/ctrl
bash -n "$SRC"
scp -q "$SRC" "$LOGIN:$C/.$NAME.tmp"
ssh "$LOGIN" "mv $C/.$NAME.tmp $C/$NAME.sh"
echo "queued $NAME in $CODE/ctrl"
