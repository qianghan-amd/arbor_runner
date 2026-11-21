"""Status of a PR-test run, and its result files fetched home (design.md section 6: short calls only).

Usage: python scripts/poll_status.py --run-dir runs/<run_id> [--no-fetch] [--extend]
Per cell: the job's state (normalized: PENDING RUNNING SUCCEEDED FAILED CANCELLED UNKNOWN, plus the scheduler's own
word), the runner's state file and finished phases, then (unless --no-fetch) the small result files
(backends.RESULT_FILES) copied into runs/<run_id>/<cell>/. --extend: a running job below the wall time its cell
needs gets more, where the backend supports it. Writes runs/<run_id>/status.json. Stateless: run it as often as you like.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from arbor_ab import backends, config


def poll(run_dir, backend=None, fetch=True, extend=False):
    """backend: None = the site the run was submitted to (run_config.json `site_name`)."""
    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    backend = backend or backends.for_run(config.load_local(), rc)
    if not rc.get("submitted"):
        sys.exit(f"{rc['run_id']} was not submitted (dry run or failed submit)")
    status = backend.status(rc["cells"])
    for s, c in zip(status, rc["cells"]):
        s["cell"] = c["cell"]
        if extend:
            s["extended_to"] = backend.extend(c, s)
        if fetch:
            s["files_fetched"] = backend.fetch(c, run_dir / c["cell"])
        print(f"{s['cell']}: {s['state']} ({s['raw']}) {s['elapsed']} on {s['node']} | runner {s['runner_state']} "
              f"| done {s['phases_done']}" + (f" | extended to {s['extended_to']}" if s.get("extended_to") else ""))
    doc = {"run_id": rc["run_id"], "backend": backend.name, "polled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "cells": status}
    (run_dir / "status.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--extend", action="store_true")
    args = ap.parse_args()
    poll(args.run_dir, None, not args.no_fetch, args.extend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
