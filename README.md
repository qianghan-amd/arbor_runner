# Arbor A/B test system

A regression benchmark for **Arbor itself**: given an Arbor commit, run it on fixed `(model, target)` cells on the
Crusoe MI355X cluster, re-measure before and after (AgentX replay + GSM8K), and compare with the stored baseline and
its ceiling. Design: [`docs/design.md`](docs/design.md). Crusoe only.

## Setup (a fresh clone; nothing else is needed)

Prerequisites: Python 3.10+, git, ssh access (key login) to each cluster's login node, by its hostname (`local.yaml.example`), a
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

One command for the whole path (submit, wait, report):

```bash
.venv/bin/python scripts/run_pr_test.py --request runs/requests/<name>.yaml   # or --run-dir runs/<run_id> to follow one
```

The same steps one at a time:

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

Modes: `campaign` (the default: fetch, prepare, an Arbor campaign of `campaign_minutes`; no measurement of our own),
`full` (before, campaign, after: the A/B measurement) and `smoke` (one 600 s AgentX measurement, no campaign). GSM8K
runs only with `gsm8k: true` in the request. Step by step, in easy English: [`docs/how_to_run_a_campaign.md`](docs/how_to_run_a_campaign.md). Which cells can run: `render.check_runnable` (see `scripts/check_setup.py` output).

### A node left dirty by someone else (optional `exclude_nodes`)

The runner refuses a node where a GPU is already in use (a foreign server would disturb the measurement); its log
names the node and the other containers (never touched). Resubmit with that node excluded:

```yaml
exclude_nodes: [crsuse2-m2m-v3-177]
```

### Choosing the QOS per run (optional `backend_options`)

Backend-specific settings for one run, recorded in `run_config.json`. For `crusoe_slurm`, `qos` and `account`
override `local.yaml`:

```yaml
backend_options: { qos: amd-burst-qos, account: amd-hyperloom }
```

`amd-burst-qos` has a 24 h wall and preempt mode "cancel": fine for smoke runs and short campaigns.

### Pre-downloaded weights (optional)

When the served model's weights already sit on the node or on shared storage, skip the download per target:

```yaml
targets:
  - { model: dsv41flash, target: c1_p90, weights_path: /shared_nfs/<you>/models/DeepSeek-V4.1-Flash/<rev> }
```

- `weights_path` is an absolute path, as the node sees it, to a directory holding the model files
  (`config.json`, `model.safetensors.index.json`, the shards). An HF hub snapshot works directly, e.g. the shared hub's
  `/shared_nfs/huggingface/hub/models--<org>--<name>/snapshots/<pinned rev>` (its shards are links into `../../blobs`;
  the runner mounts the whole `models--<org>--<name>` entry). On Crusoe the shared hub holds the pinned weights of
  DSv4.1 Flash, Kimi K3, MiniMax M3.1 Flash and GLM-5.2 (not Muse Spark; checked 2026-11-19). Without it, nothing changes: the weights are downloaded
  at the baseline's pinned revision.
- The weights must be the **pinned** ones: the runner downloads only `config.json` and `model.safetensors.index.json`
  at the pinned revision and requires both to be byte-identical to yours, and every indexed shard to exist; otherwise
  the run fails. The launch does not change (the pinned hub entry is linked to your path; it is mounted read-only).
- Every run records where the weights came from: `runs/<run_id>/<cell>/weights.json` and the `weights` block of
  `comparison.json` (`weights_source: downloaded | user_path`, the path, the pinned revision, the two files' sha256).
  If the pinned weights are already on the node, they are used and recorded as `downloaded`, with a note.

New baseline from an InferenceX CI run (vLLM or SGLang; measured on the baseline site, Crusoe; docs/make_baseline_design.md):

```bash
.venv/bin/python scripts/make_baseline.py draft --model dsv41flash --target c1_p90 --ci-run 34657748450
#   read runs/drafts/<...>/review.md: confirm each `copied` (env, launch) and `derived` (corpus rev, digest) row
.venv/bin/python scripts/make_baseline.py measure --draft runs/drafts/<...> --arbor-commit <sha> [--gsm8k]
.venv/bin/python scripts/make_baseline.py finalize --draft runs/drafts/<...> --run-dir runs/<run_id>   # stores only on gate pass
```

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

`CLAUDE.md` holds the rules for agents and people. Run `python -m pytest -q tests` before you commit. The team repo is
`AMD-BRAIN-Internal/arbor-ci` (branch `main`); never force-push.
