"""The main path in one command: submit a PR test, wait for it, report it.

Usage:
  python scripts/run_pr_test.py --request runs/requests/<name>.yaml        # submit, then follow
  python scripts/run_pr_test.py --run-dir runs/<run_id>                    # follow (and report) a run already submitted
Options: --every MINUTES (poll interval, default 10), --no-wait (submit and return, like submit_ab_test.py).
A convenience over the stateless tools (submit_ab_test.submit, poll_status.poll, generate_report.py; design.md 6): it
runs on a workstation, keeps nothing but runs/<run_id>/ on disk, and can be stopped and started again at any time.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from arbor_ab import REPO_ROOT, backends, config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import poll_status  # noqa: E402
import submit_ab_test  # noqa: E402

TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED")


def follow(run_dir, backend, every_min):
    """Poll until every cell is terminal; returns the last status document."""
    while True:
        doc = poll_status.poll(run_dir, backend, fetch=True, extend=False, resubmit=True)
        if all(c["state"] in TERMINAL for c in doc["cells"]):
            return doc
        time.sleep(every_min * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--request", type=Path)
    src.add_argument("--run-dir", type=Path)
    ap.add_argument("--every", type=float, default=10.0, help="minutes between polls")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--site", help="local.yaml site for --request (default: the request's `site`, else default_site)")
    args = ap.parse_args()
    local = config.load_local()
    if args.request:
        req = yaml.safe_load(args.request.read_text(encoding="utf-8"))
        backend = backends.for_site(local, args.site or req.get("site"))
        rc = submit_ab_test.submit(req, args.request, backend, (REPO_ROOT / local["arbor"]["clone"]).resolve())
        if not rc["submitted"]:
            return 1
        run_dir = REPO_ROOT / "runs" / rc["run_id"]
        if args.no_wait:
            print(f"follow later: python scripts/run_pr_test.py --run-dir {run_dir.relative_to(REPO_ROOT).as_posix()}")
            return 0
    else:
        run_dir = args.run_dir
        backend = backends.for_run(local, json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")))
    doc = follow(run_dir, backend, args.every)
    sys.stdout.flush()   # our buffered lines first, then the report's
    rep = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "generate_report.py"), "--run-dir", str(run_dir)])
    failed = [c["cell"] for c in doc["cells"] if c["state"] != "SUCCEEDED"]
    print(f"run {run_dir.name}: " + ("all cells succeeded" if not failed else f"not succeeded: {failed} (see their logs/runner.log)"))
    return 1 if failed or rep.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
