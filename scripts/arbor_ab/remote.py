"""Run a short stdlib-only python3 program on a cluster host over one ssh call (design.md section 6: short-lived calls only).

The program gets PAYLOAD (JSON) and must print one JSON document. Nothing persists on the host.
"""
import json
import subprocess


def run_bash(host, script, timeout=600):
    """A bash script on the host's stdin (the login shell is tcsh). Sent as bytes: text-mode stdin on Windows turns
    `\\n` into `\\r\\n`, and bash then reads `set -e\\r` and `job.sbatch\\r` (happened 2026-11-19)."""
    if "\r" in script:
        raise ValueError("script contains CR")
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash", "-s"], input=script.encode("utf-8"),
                          capture_output=True, timeout=timeout)


def run_python(host, program, payload, timeout=600):
    script = f"import json\nPAYLOAD = json.loads({json.dumps(json.dumps(payload))})\n" + program
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"], input=script.encode(), capture_output=True,
                       timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"ssh {host} python3 rc={p.returncode}: {p.stderr.decode(errors='replace')[-1500:]}")
    return json.loads(p.stdout)
