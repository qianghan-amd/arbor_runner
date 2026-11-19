# Arbor A/B test system

A regression benchmark for **Arbor itself**: given an Arbor commit, run it on fixed `(model, target)` cells on the
Crusoe MI355X cluster, re-measure before and after (AgentX replay + GSM8K), and compare with the stored baseline and
its ceiling. Design: [`docs/design.md`](docs/design.md). Crusoe only.

## Setup (a fresh clone; nothing else is needed)

Prerequisites: Python 3.10+, git, ssh access to the Crusoe login node (an alias in `~/.ssh/config`, e.g. `crs5`), a
node/QOS allocation, gh credentials on a cluster host (for the GitHub API relay), and an LLM gateway key for Arbor's
agents (only for full runs with a campaign).

1. **Workstation:** `bash scripts/setup_dev.sh`. It makes `.venv` with `requirements.txt`, copies
   `local.yaml.example` to `local.yaml` (gitignored) and runs the tests. Fill in the `<user>` paths in `local.yaml`.
2. **Arbor clone:** `git clone https://github.com/AMD-AGI/Arbor ../Arbor` (or set `arbor.clone` in `local.yaml`).
3. **Cluster:** `.venv/bin/python scripts/setup_remote.py --key-file <file with the gateway key> --test-gateway`.
   It creates the remote work dirs with the right modes and writes the gateway env file (mode 600). The key never
   enters the repo, a log or a process argument list. Set `arbor.gateway_env` in `local.yaml` to the path it prints.
4. **Check:** `.venv/bin/python scripts/check_setup.py`. Every line must be PASS.

(On Windows Git Bash the interpreter is `.venv/Scripts/python`.)

## Use

```bash
# 1. write a request (the commit is the PR's head commit)
cat runs/requests/smoke-dsv41-c1.yaml
# 2. render it locally and look at runs/<run_id>/ first
.venv/bin/python scripts/submit_ab_test.py --request runs/requests/smoke-dsv41-c1.yaml --dry-run
# 3. submit (one job per cell through the execution backend, design.md 5.1; the script exits at once)
.venv/bin/python scripts/submit_ab_test.py --request runs/requests/smoke-dsv41-c1.yaml
# 4. poll as often as you like (fetches the small result files)
.venv/bin/python scripts/poll_status.py --run-dir runs/<run_id>      # --extend: more wall time if needed
# (stop a run: .venv/bin/python scripts/cancel_run.py --run-dir runs/<run_id>)
# 5. report: runs/<run_id>/<cell>/comparison.json (verdict stays null: a person reads it)
.venv/bin/python scripts/generate_report.py --run-dir runs/<run_id>
```

Modes: `smoke` (fetch, prepare, 600 s AgentX + 200 GSM8K questions; no campaign) and `full` (before, an Arbor campaign
of `campaign_minutes`, after). Which cells can run: `render.check_runnable` (see `scripts/check_setup.py` output).

Baselines: `scripts/verify_baseline.py` (two-layer gate against the InferenceX CI artifacts) and
`scripts/check_baseline_health.py` (CI artifact, image digest/tag, mirror). Evidence behind every number:
[`evidence/README.md`](evidence/README.md).

## Layout

| Path | What |
|---|---|
| `config/` | matrix (cells, tolerances), per-model facts, cluster facts (QOS, scheduler, gateway format) |
| `schemas/baseline.schema.json` | the baseline record (v2) |
| `baselines/<model>/<target>/` | one record per epoch + `latest.json` |
| `evidence/` | raw ground truth the baselines were built from |
| `scripts/` | CLIs; `scripts/arbor_ab/` the shared library; `scripts/arbor_ab/backends/` where jobs run (design.md 5.1); `scripts/runner/` what runs on the node |
| `runs/requests/` | request files (tracked); `runs/<run_id>/` local run dirs (ignored; commit results with `git add -f`) |
| `skills/` | the two campaign skills the runner was built from |

## Contributing

`CLAUDE.md` holds the rules for agents and people. Run `python -m pytest -q tests` before you commit. Push to
`origin main` only.
