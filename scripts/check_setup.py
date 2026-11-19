"""Day-1 smoke test (docs/design.md section 10): connectivity and setup only. Not a permission or quota check.

Usage: python scripts/check_setup.py
Checks: local.yaml, Python deps, the execution backend's own checks (crusoe_slurm: ssh to the login node, work-dir
modes, gateway env mode, scheduler commands), GitHub API through the relay host (token present there, never printed),
vendored skills, stored baselines valid. Exit code 1 if any check fails.
"""
import sys

from arbor_ab import REPO_ROOT

SKILLS = ("arbor-ci-campaign-setup", "crusoe-arbor-run")
GITHUB_REPOS = ("SemiAnalysisAI/InferenceX", "AMD-AGI/Arbor")


def check_github(relay):
    from arbor_ab import github_relay
    res = github_relay.call(relay, [github_relay.get(f"/repos/{r}") for r in GITHUB_REPOS], timeout=120)
    codes = {r: x.get("http") for r, x in zip(GITHUB_REPOS, res)}
    if any(c != 200 for c in codes.values()):
        raise RuntimeError(f"http codes {codes}")
    return f"token on {relay} works (not shown); " + ", ".join(f"{r} 200" for r in codes)


def check_skills():
    missing = [s for s in SKILLS if not (REPO_ROOT / "skills" / s / "SKILL.md").exists()]
    if missing:
        raise RuntimeError(f"missing under {REPO_ROOT / 'skills'}: {missing}")
    return ", ".join(SKILLS)


def check_baselines():
    from arbor_ab import baseline, config
    cells = config.cells(config.load_matrix())
    recs = {c: baseline.load_latest(*c) for c in cells}
    absent = [f"{a}/{b}" for (a, b), r in recs.items() if r is None]
    invalid = [f"{a}/{b}" for (a, b), r in recs.items() if r is not None and baseline.validation_errors(r)]
    if invalid:
        raise RuntimeError(f"schema-invalid: {invalid}")
    return f"{len(cells) - len(absent)}/{len(cells)} active cells have a valid record" + (f"; none for {absent}" if absent else "")


def main():
    results = []

    def step(name, fn):
        try:
            results.append(("PASS", name, fn()))
            return True
        except Exception as e:
            results.append(("FAIL", name, f"{type(e).__name__}: {e}"))
            return False

    from arbor_ab import backends, config
    local = {}

    def load_local():
        local.update(config.load_local())
        return (f"backend={local.get('backend', 'crusoe_slurm')}, "
                f"github_relay_host={local['ssh']['github_relay_host']}")

    def deps():
        import jsonschema, yaml  # noqa: F401
        return "yaml, jsonschema importable"

    if step("local.yaml", load_local):
        step("python deps", deps)
        try:
            backend = backends.get(local, config.load_cluster())
            for name, ok, msg in backend.check():
                results.append(("PASS" if ok else "FAIL", f"{backend.name}: {name}", msg))
        except Exception as e:
            results.append(("FAIL", "backend", f"{type(e).__name__}: {e}"))
        step("github via relay", lambda: check_github(local["ssh"]["github_relay_host"]))
    step("vendored skills", check_skills)
    step("baselines", check_baselines)
    for status, name, msg in results:
        print(f"[{status}] {name}: {msg}")
    return 1 if any(s == "FAIL" for s, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
