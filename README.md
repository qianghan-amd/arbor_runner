# Arbor CI runner

Run an Arbor campaign on a GPU cluster, with one command.

You give two things:

1. an **AgentX CI run** (from InferenceX, Actions, "Run Sweep"): it tells us the model, the docker image, the server
   flags and the benchmark;
2. an **Arbor commit**: the Arbor version to test.

The tool reads the CI run, builds the exact same server launch, finds or downloads the weights, and starts the job on
the cluster. Everything it decided is saved, so you can check it and run it again.

Step-by-step guide: [`docs/how_to_run_a_campaign.md`](docs/how_to_run_a_campaign.md).

## Set up (once)

You need: Python 3.10+, git, ssh key login to the cluster login node, and the LLM gateway key (Arbor's agents use it).

```bash
bash scripts/setup_dev.sh                       # makes .venv, copies local.yaml.example to local.yaml, runs the tests
# edit local.yaml: your user name and paths for each cluster (site)
git clone https://github.com/AMD-AGI/Arbor ../Arbor
.venv/bin/python scripts/setup_remote.py --site crusoe --key-file <file with the gateway key> --test-gateway
.venv/bin/python scripts/check_setup.py --site crusoe   # every line must say PASS
```

On Windows Git Bash, use `.venv/Scripts/python` and put `MSYS_NO_PATHCONV=1` before the command.

## Examples

### 1. Run a campaign from a CI run (the main use)

```bash
.venv/bin/python scripts/run_from_ci.py \
    --ci-run https://github.com/SemiAnalysisAI/InferenceX/actions/runs/44689368479 \
    --arbor-commit 020767ac02a7bfad05a37f8ad7fb09419e529584
```

This starts an 8-hour Arbor campaign. The CI run can be a URL or just the number. If the CI run measured more than one
of our cells, the tool lists them; then add `--target c1_p90` (and `--tp 4` if needed).

### 2. Look before you launch

```bash
.venv/bin/python scripts/run_from_ci.py --ci-run 44689368479 --arbor-commit <sha> --dry-run
```

Nothing is sent. It prints the cell, the image, the weights and the request file it wrote. Read
`runs/drafts/<cell>__ci<id>/review.md` to see where each value came from.

### 3. Pick the cluster

```bash
--site crusoe     # MI355X, burst QOS (the default site)
--site meta64     # MI350X
```

Sites are in `local.yaml`. Other options: `--qos`, `--partition`, `--nodelist`, `--exclude-nodes`.

Have your own "hold" job with a node agent (it runs `ctrl/node<i>/*.sh` on node i)? Use one of its free nodes:

```bash
--hold-job 304634 --nodelist crsuse2-m2m-v3-002
```

The tool checks that the hold job is yours, holds that node, and has enough time left. The runner still refuses a
node whose GPUs are in use.

### 4. Pick what to run

```bash
--mode campaign                                    # default: Arbor campaign only
--mode full                                        # measure before, campaign, measure after: the A/B result
--mode smoke                                       # one short measurement, no campaign: "does it start?"
--mode campaign_smoke --campaign-minutes 15        # a short campaign: "does Arbor start?"
--campaign-minutes 480                             # campaign length (default 480)
--objective "total throughput"                     # what Arbor optimizes (default: the target's metric)
--goal "Raise total throughput at conc 16"         # a sentence for Arbor's agents
--gsm8k                                            # add the GSM8K accuracy check (off by default)
```

### 5. Watch, stop, and get the report

```bash
.venv/bin/python scripts/poll_status.py --run-dir runs/<run_id>       # state, finished phases, fetches results
.venv/bin/python scripts/cancel_run.py --run-dir runs/<run_id>        # stop it (your own job only)
.venv/bin/python scripts/generate_report.py --run-dir runs/<run_id>   # the report
```

Or add `--follow` to `run_from_ci.py`: it waits for the job and prints the report.

### 6. Run from a launch file instead of a CI run

`run_from_ci.py` saves the full launch as `runs/requests/<name>.record.json` and a request that points to it. You can
edit the record (for example one server flag) and run the request again:

```yaml
# runs/requests/my-test.yaml
commit: 020767ac02a7bfad05a37f8ad7fb09419e529584
mode: smoke
site: meta64
targets:
  - { model: dsv41flash, target: c1_p90, record: runs/requests/my-test.record.json }
```

```bash
.venv/bin/python scripts/submit_ab_test.py --request runs/requests/my-test.yaml
```

### 7. Weights

```bash
--weights-path auto     # default: the shared hub if it has every file, else the job downloads them
--weights-path /path    # your own copy (it must match the pinned revision)
--weights-path none     # always download
```

The job keeps a cache on the node's local disk and removes your oldest models when space is low.

## What the tool refuses, and why

It stops early, with a clear message, instead of running a bad test:

| Case | Why |
|---|---|
| The CI ran on NVIDIA or MI300/MI325 | its image and flags are not for our GPUs (`--allow-other-hw` to force) |
| The CI cell serves other weights than our baselines (e.g. a `glm5.2` cell that serves GLM-5.3) | it is another model (`--allow-other-model` to force) |
| The CI used `atom`, `tokenspeed` or a multi-node setup | only single-node vLLM and SGLang are supported |
| Our launch does not match the args the CI server printed | the CI format may have changed |
| The CI's docker image is gone from the registry | we cannot get the same image (pick a newer CI run) |
| The node disk has no room, or a GPU on the node is busy | resubmit; for a busy node add `--exclude-nodes <node>` |

During a run, the job stops the campaign at its time budget, removes its own containers at the end (also after a
kill), and frees the GPUs.

## Clusters

| Site | GPUs | Notes |
|---|---|---|
| `crusoe` | MI355X | default; burst QOS (24 h, can be preempted); baselines are measured only here |
| `meta64` | MI350X | runner only (no baselines); partitions `meta64` (5 days) and `interactive` (1 day) |

Tested setups and the bugs they found: [`docs/runner_test_matrix.md`](docs/runner_test_matrix.md).

## For maintainers

- New baseline from a CI run (Crusoe only): `scripts/make_baseline.py draft | measure | finalize`
  ([`docs/make_baseline_design.md`](docs/make_baseline_design.md)).
- Design: [`docs/design.md`](docs/design.md). Every number in `baselines/` traces to a file in `evidence/`.

| Path | What |
|---|---|
| `config/` | models and targets (`matrix.yaml`), cluster facts |
| `baselines/` | stored baselines (one record per model, target and epoch) |
| `scripts/` | the commands; `scripts/arbor_ab/` the library; `scripts/runner/` what runs on the node |
| `runs/requests/` | request files; `runs/<run_id>/` local run folders (not in git) |
| `skills/arbor-ci-run/` | the agent skill: an agent (Claude Code, Codex, Cursor) follows it to set up and run |

## Contributing

Rules for people and agents: [`CLAUDE.md`](CLAUDE.md). Run `python -m pytest -q tests` before each commit. Team repo:
`AMD-BRAIN-Internal/arbor-ci`, branch `main`. Never force-push.
