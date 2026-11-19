"""One-time setup on the execution side, step 3 of README "Setup". Idempotent.

Usage: python scripts/setup_remote.py [--key-file <file>] [--test-gateway]
What it does is the backend's (local.yaml `backend`; crusoe_slurm: work dirs 755/700 on /shared_nfs, the LLM gateway
env file (mode 600) built from the key in <file>, the InferenceX clone's bench-client commits, one gateway call).
The key is read on this workstation and handed to the backend only: never printed, never an argument, never in the repo.
"""
import argparse
import sys
from pathlib import Path

from arbor_ab import backends, baseline, config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key-file", type=Path)
    ap.add_argument("--test-gateway", action="store_true")
    args = ap.parse_args()
    key = None
    if args.key_file:
        key = args.key_file.read_text(encoding="utf-8").strip()
        if not key or any(c.isspace() or not c.isprintable() for c in key):
            sys.exit(f"{args.key_file}: expected one key with no whitespace")
    shas = sorted({r["measured"]["bench_client_commit"] for m, t in config.cells(config.load_matrix())
                   for r in [baseline.load_latest(m, t)] if r and r.get("launch") and r["measured"]["bench_client_commit"]})
    backend = backends.get(config.load_local(), config.load_cluster())
    results = backend.setup(key=key, test_gateway=args.test_gateway, infx_shas=shas)
    for name, ok, msg in results:
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {msg}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
