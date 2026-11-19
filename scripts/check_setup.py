"""Day-1 smoke test (docs/design.md section 10): connectivity only. Not a permission or quota check.

Usage: python scripts/check_setup.py
Checks: local.yaml, Python deps, ssh to the login host, GitHub API through the relay host (token present there, never
printed), vendored skills, stored baselines valid. Exit code 1 if any check fails.
"""
import subprocess
import sys

from arbor_ab import REPO_ROOT

SKILLS = ("arbor-ci-campaign-setup", "crusoe-arbor-run")
GITHUB_REPOS = ("SemiAnalysisAI/InferenceX", "AMD-AGI/Arbor")


def check_local():
    from arbor_ab import config
    local = config.load_local()
    return local, f"ssh.login_host={local['ssh']['login_host']}, ssh.github_relay_host={local['ssh']['github_relay_host']}"


def check_ssh(host):
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "hostname"],
                       capture_output=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace").strip()[-300:])
    return f"{host} -> {p.stdout.decode().strip()}"


def check_github(relay):
    from arbor_ab import github_relay
    res = github_relay.call(relay, [github_relay.get(f"/repos/{r}") for r in GITHUB_REPOS], timeout=120)
    codes = {r: x.get("http") for r, x in zip(GITHUB_REPOS, res)}
    if any(c != 200 for c in codes.values()):
        raise RuntimeError(f"http codes {codes}")
    return f"token on {relay} works (not shown); " + ", ".join(f"{r} 200" for r in codes)


REMOTE_LAYOUT = r'''
import os, shutil, stat
def mode(p):
    return oct(stat.S_IMODE(os.stat(p).st_mode)) if os.path.exists(p) else None
wr, gw = PAYLOAD["work_root"], PAYLOAD["gateway_env"]
print(json.dumps({"work_root": mode(wr), "runs": mode(wr + "/runs"), "secrets": mode(wr + "/secrets"),
                  "gateway_env": mode(gw) if gw else "not configured",
                  "commands": {c: bool(shutil.which(c)) for c in ("sbatch", "squeue", "sacct", "scontrol")}}))
'''


def check_remote_layout(local):
    """What scripts/setup_remote.py sets up: dir modes, the gateway env file's mode (its contents are never read)."""
    from arbor_ab import remote
    r = remote.run_python(local["ssh"]["login_host"], REMOTE_LAYOUT,
                          {"work_root": local["remote"]["work_root"], "gateway_env": local["arbor"].get("gateway_env")})
    want = {"work_root": "0o755", "runs": "0o755", "secrets": "0o700"}
    bad = [f"{k} is {r[k]} (want {v})" for k, v in want.items() if r[k] != v]
    if r["gateway_env"] not in ("0o600", "not configured"):
        bad.append(f"gateway_env is {r['gateway_env']} (want 0o600; run setup_remote.py --key-file)")
    bad += [f"{c} not on PATH" for c, ok in r["commands"].items() if not ok]
    if bad:
        raise RuntimeError("; ".join(bad) + " -> run scripts/setup_remote.py")
    return f"work_root 755, runs 755, secrets 700, gateway_env {r['gateway_env']}, sbatch/squeue/sacct/scontrol present"


def check_skills():
    missing = [s for s in SKILLS if not (REPO_ROOT / "skills" / s / "SKILL.md").exists()]
    if missing:
        raise RuntimeError(f"missing under {REPO_ROOT / 'skills'}: {missing}")
    return ", ".join(SKILLS)


def check_baselines():
    from arbor_ab import baseline, config
    m = config.load_matrix()
    cells = config.cells(m)
    recs = {c: baseline.load_latest(*c) for c in cells}
    absent = [f"{a}/{b}" for (a, b), r in recs.items() if r is None]
    invalid = [f"{a}/{b}" for (a, b), r in recs.items() if r is not None and baseline.validation_errors(r)]
    if invalid:
        raise RuntimeError(f"schema-invalid: {invalid}")
    return f"{len(cells) - len(absent)}/{len(cells)} active cells have a valid record" + (f"; none for {absent}" if absent else "")


def main():
    results, local = [], None
    steps = [("local.yaml", None), ("python deps", None), ("ssh login host", None), ("remote layout", None),
             ("github via relay", None), ("vendored skills", check_skills), ("baselines", check_baselines)]
    for name, fn in steps:
        try:
            if name == "local.yaml":
                local, msg = check_local()
            elif name == "python deps":
                import jsonschema, yaml  # noqa: F401
                msg = "yaml, jsonschema importable"
            elif name == "ssh login host":
                msg = check_ssh(local["ssh"]["login_host"])
            elif name == "remote layout":
                msg = check_remote_layout(local)
            elif name == "github via relay":
                msg = check_github(local["ssh"]["github_relay_host"])
            else:
                msg = fn()
            results.append(("PASS", name, msg))
        except Exception as e:
            results.append(("FAIL", name, f"{type(e).__name__}: {e}"))
    for status, name, msg in results:
        print(f"[{status}] {name}: {msg}")
    return 1 if any(s == "FAIL" for s, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
