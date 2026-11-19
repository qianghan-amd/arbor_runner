#!/bin/bash
# Drop a script into an experiment's ctrl/ atomically, so the hold job's node agent never runs a half-written file.
# usage: bash ctrlput.sh <local script> <name> <experiment code name>
#   -> crs5:/home/qiangh/arbor_dsv41flash/experiments/<code name>/ctrl/<name>.sh ; output: logs/<name>.<jobid>.log
set -eu
SRC=${1:?local script}; NAME=${2:?name}; CODE=${3:?experiment code name}
C=/home/qiangh/arbor_dsv41flash/experiments/$CODE/ctrl
bash -n "$SRC"
scp -q "$SRC" "crs5:$C/.$NAME.tmp"
ssh crs5 "mv $C/.$NAME.tmp $C/$NAME.sh"
echo "queued $NAME in $CODE/ctrl"
