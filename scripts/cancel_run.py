"""Cancel the jobs of a PR-test run (its own jobs only, by the ids in run_config.json).

Usage: python scripts/cancel_run.py --run-dir runs/<run_id> [--cell <model>__<target>]
The runner removes its own container when its job ends. Writes nothing but the printed result.
"""
import argparse
import json
import sys
from pathlib import Path

from arbor_ab import backends, config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--cell")
    args = ap.parse_args()
    rc = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    backend = backends.get(config.load_local(), config.load_cluster())
    if rc.get("backend", "crusoe_slurm") != backend.name:
        sys.exit(f"{rc['run_id']} ran on backend {rc.get('backend')}, local.yaml selects {backend.name}")
    for c in rc["cells"]:
        if not args.cell or c["cell"] == args.cell:
            print(f"{c['cell']} job {c['job_id']}: {backend.cancel(c)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
