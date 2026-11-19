#!/bin/bash
# Workstation setup, step 1 of README "Setup": a repo-local venv with the pinned deps, local.yaml from the example,
# and the test suite. Idempotent. Linux/macOS (python3) and Windows Git Bash (python) both work.
# usage: bash scripts/setup_dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# run each candidate: on Windows `python3` can be the Microsoft Store alias, which is on PATH but is not Python
PYBIN=
for c in python3 python py; do
    if "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then PYBIN=$c; break; fi
done
[ -n "$PYBIN" ] || { echo "need python 3.10+ (python3, python or py)"; exit 1; }
[ -d .venv ] || "$PYBIN" -m venv .venv
VPY=.venv/bin/python; [ -x "$VPY" ] || VPY=.venv/Scripts/python
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q -r requirements.txt
if [ ! -f local.yaml ]; then
    cp local.yaml.example local.yaml
    echo "created local.yaml from local.yaml.example: fill in the <user> paths (and arbor.gateway_env after setup_remote.py)"
fi
"$VPY" -m pytest -q tests
git config core.hooksPath .githooks   # pre-commit: tests must pass
echo "dev setup ok: $VPY; next: edit local.yaml, then $VPY scripts/setup_remote.py --key-file <file>, then $VPY scripts/check_setup.py"
