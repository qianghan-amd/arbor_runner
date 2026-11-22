# How to run a campaign

This page tells you how to test one Arbor commit. You do not need to read the design first.

## What a run does

You give the system an Arbor commit. For each model and target you pick (a "cell"), it starts one cluster job. The job
does these steps on one node:

1. **fetch**: get the model weights and the test data (or use weights that are already on shared storage).
2. **prepare**: pull the Docker image, start a container, and check that the node is free.
3. **campaign**: run Arbor for `campaign_minutes`. Arbor tries to make the server faster.

That is the default (`mode: campaign`). The campaign starts from the cell's baseline: the server settings of one
AgentX CI run. The report names that CI run.

If you also want our own A/B measurement, use `mode: full`. It adds two steps around the campaign:

- **before**: start the server with the baseline settings and measure it (one AgentX replay of 3600 s).
- **after**: start the server with the settings Arbor kept and measure it again, the same way.

Then you get `comparison.json`: before, after, and the best known result (the "ceiling") for that cell. Before and
after always run on the same node, in the same job. That makes the comparison fair. GSM8K (the accuracy check) is off
unless you set `gsm8k: true`.

## The inputs of a campaign

**These inputs decide what a campaign tests. Get them right first.** A campaign has three main inputs:

| # | Input | What it is | Where it comes from |
|---|---|---|---|
| 1 | **The starting point** (the "baseline") | The server launch Arbor starts from: framework (vLLM or SGLang), TP/EP, server flags, server env, concurrency. | One **AgentX CI run** of InferenceX. `scripts/make_baseline.py` turns a CI run into a baseline record in `baselines/<model>/<target>/`. |
| 2 | **The Docker image** | The serving image (vLLM or SGLang build), pinned by its digest. | The same AgentX CI run: the image that run used. |
| 3 | **The Arbor commit** | The Arbor code under test. | You. `commit:` in the request. |

You pick 1 and 2 together, by picking a cell (`model` + `target`). Each cell has a stored baseline, and the baseline
holds the image. To start from a different CI run or a different image, first make a new baseline from that CI run
(`make_baseline.py draft`, `measure`, `finalize`; vLLM or SGLang; measured on Crusoe only). After that, every request for that cell uses the new baseline.

The baseline also pins these inputs. They come from the same CI run, and you do not set them in a request:

- **Model weights**: the model and its exact Hugging Face revision.
- **Test data** (the "corpus"): the AgentX dataset and its revision.
- **Benchmark client**: the InferenceX commit that replays the AgentX traffic.
- **The goal**: the target says what Arbor must improve. `c1_p90` means P90 interactivity (`e2e_norm_intvty.p90`) at concurrency 1;
  `c32_tput` means throughput at concurrency 32.

You set these in the request or in your one-time setup:

| Input | Where | Note |
|---|---|---|
| Campaign length | `campaign_minutes` in the request | The runner stops Arbor at this time. 8 h is a good start (see "How long it takes"). |
| Weights folder | `weights_path` in the request (optional) | Saves the download. The files must match the pinned revision. |
| Cluster and QOS | `local.yaml` (or `backend_options` in the request) | One node per cell. |
| LLM gateway key | one-time setup, step 5 | Arbor's agents call the LLM through it. A campaign cannot start without it. |

What is not possible yet: starting from a hand-written full config, without a CI run. The closest way today is to run
`make_baseline.py draft` on a CI run and edit the `draft.json` it writes before `measure`.

## One-time setup

Do this once per person.

1. Get these first:
   - ssh access to the Crusoe login node, with an alias in `~/.ssh/config`.
   - A QOS you can submit jobs to.
   - `gh` logged in on one cluster host (used to read GitHub CI results).
   - An LLM gateway key (Arbor's agents need it during the campaign).
2. On your workstation, in this repo:

   ```bash
   bash scripts/setup_dev.sh
   ```

   This makes `.venv`, copies `local.yaml.example` to `local.yaml`, and runs the tests.
3. Edit `local.yaml`. Replace every `<user>` with your user name. Set your ssh alias in `ssh.login_host`.
4. Clone Arbor next to this repo:

   ```bash
   git clone https://github.com/AMD-AGI/Arbor ../Arbor
   ```

5. Set up your cluster folders and the gateway key:

   ```bash
   .venv/bin/python scripts/setup_remote.py --key-file <file with the gateway key> --test-gateway
   ```

   Copy the path it prints into `arbor.gateway_env` in `local.yaml`. The key itself never goes into the repo.
6. Check everything:

   ```bash
   .venv/bin/python scripts/check_setup.py
   ```

   Every line must say PASS. It also lists the cells you can run.

On Windows (Git Bash), use `.venv/Scripts/python` instead of `.venv/bin/python`.

## Run a campaign from an AgentX CI run (the usual way)

You give two things: the AgentX CI run and the Arbor commit. Everything else is read from the CI run.

```bash
python scripts/run_from_ci.py --ci-run <run id or URL> --arbor-commit <Arbor SHA> [--site crusoe|meta64]
```

**What to give as `--ci-run`:** an InferenceX GitHub Actions run that ran the AgentX benchmark. Either form works:

- the run's URL, as you copy it from the browser:
  `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/44689368479` (a `.../job/<n>` URL of that run also works);
- or just the number: `44689368479`.

**Where to find one:** in the InferenceX repository, Actions tab, workflow "Run Sweep". Or open the pull request
that changed the recipe and click its "Run Sweep" check. Pick a run that:

- **finished and ran the benchmark.** Its Artifacts list has `bmk_agentic_<model>_tp<N>_conc<C>_...` files. A run
  whose jobs were skipped because it reused older results has no such files: the tool refuses it and says so. Use the
  earlier run on the same branch that produced the numbers.
- **is less than 90 days old.** GitHub deletes artifacts after that.
- **measured the model you want**, at the concurrency of your target. The tool reads the model and concurrency from the
  artifact names. If the run measured several of our cells (for example C1 and C16, or TP4 and TP8), it lists them and
  stops. Then add `--target c1_p90` and, if needed, `--tp 4`.

Supported: vLLM and SGLang runs, in both artifact formats seen so far (the older `vllm_command.txt` and the srtctl
recipe). The model must already have a stored baseline in this repo, for its container layout.

**What the tool prints**, step by step: the cell it chose, the image and model revision from the CI run, where the
weights come from, the request file it wrote (`runs/requests/ci<id>-...yaml`, to repeat the run exactly), and the job.
`runs/drafts/<model>__<target>__ci<id>/review.md` shows where each setting came from.

Options: `--mode campaign` (default) | `full` | `smoke` | `campaign_smoke`, `--campaign-minutes 480`, `--gsm8k`,
`--exclude-nodes ...`, `--dry-run` (write everything, submit nothing), `--follow` (wait and print the report).

### The launch JSON: saved, and yours to give

`run_from_ci.py` saves what it read from the CI run as one JSON, the **launch record**, next to its request file:
`runs/requests/<name>.record.json`. The request names it (`record:` in the target), so
`python scripts/submit_ab_test.py --request runs/requests/<name>.yaml` repeats exactly that launch.

You can also give a launch record yourself, without a CI run:

1. Copy one: a `runs/requests/*.record.json`, a `runs/drafts/*/draft.json`, or a stored baseline
   (`baselines/<model>/<target>/<epoch>.json`).
2. Edit what you want to change. The fields you usually touch:
   - `host.server_flags`, `host.server_env`: the server's flags and env;
   - `fingerprint.image`, `fingerprint.image_digest`: the container image;
   - `launch.model_id`, `launch.model_rev`: the weights (and `launch.draft_id`, `launch.draft_rev`);
   - `launch.corpus_id`, `launch.corpus_rev`: the AgentX traces.
3. Name it in a request target:

   ```yaml
   commit: <Arbor SHA>
   campaign_minutes: 480
   targets:
     - { model: dsv41flash, target: c1_p90, record: runs/requests/my-launch.record.json }
   ```

4. Submit: `python scripts/submit_ab_test.py --request runs/requests/my-launch.yaml --dry-run`, then without
   `--dry-run`.

Submit refuses a record that is not valid (`schemas/baseline.schema.json`) or names another cell, and says why. It
recomputes `epoch_id` if your edit changed the fields it comes from. A hand-written launch has no CI check behind it
(the server-argument check runs only on CI runs): you are responsible for what it says.

## Run a campaign from a request file

Use this to run a stored baseline, several cells at once, or a request you wrote by hand.

### 1. Pick the commit

Use the full SHA of the Arbor commit you want to test. Make sure your Arbor clone has it:

```bash
git -C ../Arbor fetch origin
git -C ../Arbor rev-parse --verify <sha>
```

### 2. Write a request file

Make a new file in `runs/requests/`, for example `runs/requests/my-test.yaml`:

```yaml
# What: test Arbor <sha> on DSv4.1 Flash C1 P90, 8 h campaign
commit: <full Arbor SHA>
campaign_minutes: 480
targets:
  - { model: dsv41flash, target: c1_p90,
      weights_path: /shared_nfs/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 }
```

The name of the file becomes part of the run name. Write a short comment at the top that says why you run it.

| Field | Need it? | What it means |
|---|---|---|
| `commit` | yes | The Arbor commit to test. |
| `mode` | no | `campaign` (default): only the campaign. `full`: before, campaign, after (the A/B measurement). `smoke`: a short measurement, no campaign (checks the setup). `campaign_smoke`: a short campaign (checks that Arbor starts). |
| `campaign_minutes` | yes | How long Arbor runs. The runner stops Arbor at this time, even in the middle of a step. Always set it: the default is 1440 (24 h), and that job is too long for burst. |
| `gsm8k` | no | `true` adds the GSM8K accuracy check to the measured phases. Default: off, so correctness is not checked. |
| `targets` | yes | The cells to run. One job per cell. |
| `weights_path` | no | A folder that already holds the pinned model weights. The job then does not download them. It must be an absolute path that the node can see. |
| `exclude_nodes` | no | Nodes the job must not use, for example `[crsuse2-m2m-v3-177]`. |
| `backend_options` | no | Use a different QOS for this run, for example `{ qos: amd-agentx-qos, account: amd-agentx }`. |

Cells: see `config/matrix.yaml` for the names, and `scripts/check_setup.py` for the cells that can run now.

### 3. Try it without the cluster

```bash
.venv/bin/python scripts/submit_ab_test.py --request runs/requests/my-test.yaml --dry-run
```

This writes all files into `runs/<run_id>/` and sends nothing. If a cell cannot run, it tells you why.

### 4. Submit and wait

```bash
.venv/bin/python scripts/run_pr_test.py --request runs/requests/my-test.yaml
```

This submits the jobs, checks them every 10 minutes, and writes the report at the end. You can stop it with Ctrl-C at
any time. The jobs keep running on the cluster. To follow the run again later:

```bash
.venv/bin/python scripts/run_pr_test.py --run-dir runs/<run_id>
```

If you only want to submit and come back later, add `--no-wait`.

### 5. Check on a run

```bash
.venv/bin/python scripts/poll_status.py --run-dir runs/<run_id>
```

It prints the state of each cell and the phase it is in, and copies the small result files to `runs/<run_id>/`. The
full logs stay on the cluster in `<work_root>/runs/<run_id>/<cell>/logs/`. Start with `runner.log`; the Arbor log is
`campaign.log`.

To stop a run:

```bash
.venv/bin/python scripts/cancel_run.py --run-dir runs/<run_id>
```

### 6. Read the result

```bash
.venv/bin/python scripts/generate_report.py --run-dir runs/<run_id>
```

For a `campaign` run, the report shows the baseline's CI run and how the campaign ended (`reason`, number of
iterations, number of kept changes). Arbor's own results are in `<work_root>/runs/<run_id>/<cell>/sessions/full/`.

For a `full` run, open `runs/<run_id>/<cell>/comparison.json`:

- `baseline_source`: the AgentX CI run (id, commit, date, image) the baseline comes from.
- `before`, `after`: the measured numbers. `change_pct` is after compared with before.
- `ceiling`: the best known result for this cell.
- `gsm8k`: `pass`, `fail`, or `skipped`. If it says `skipped`, correctness was not checked in this run.
- `weights`: where the model weights came from.
- `verdict` is always empty. A person reads the numbers and decides.

## How long it takes

| Step | Time |
|---|---|
| fetch + prepare | 5 to 60 min (fast when the node already has the files) |
| campaign | `campaign_minutes` |
| before, after (`full` only) | about 75 min each |

A `campaign` job asks the scheduler for `120 + campaign_minutes` minutes. A `full` job on a vLLM cell asks for `450 +
campaign_minutes` (SGLang: `320 + campaign_minutes`). On burst a job can be 24 h at most. The submit step refuses a
job that is too long for its QOS.

Choose a campaign that is long enough. Arbor's first round studies the model (the "map" phase) before it changes
anything. For DSv4.1 Flash, that phase alone took more than 3 hours. A 3-hour campaign can end with no change at all.
We use 8 hours.

## Choose the cluster (site)

Two clusters work today. Each is a "site" in your `local.yaml` (see `local.yaml.example`):

| Site | Cluster | GPUs | Login |
|---|---|---|---|
| `crusoe` | Crusoe (Spur scheduler) | 8 x MI355X | `crsuse2-slog-006.crusoe.amd.com` |
| `meta64` | meta64 (Slurm 25) | 8 x MI350X | `ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com` |

Pick one in any of these ways. The first one found wins:

1. On the command line: `--site meta64` (on `run_from_ci.py`, `submit_ab_test.py`, `run_pr_test.py`,
   `make_baseline.py measure`, `setup_remote.py`, `check_setup.py`).
2. In the request file: `site: meta64`.
3. `default_site` in `local.yaml` (now `crusoe`).

You do not choose it again later. The run records its site, and `poll_status.py`, `cancel_run.py` and
`run_pr_test.py --run-dir` go to that cluster by themselves.

Before the first run on a site, do setup steps 5 and 6 for it: `setup_remote.py --site <site> ...` and
`check_setup.py --site <site>`.

Crusoe is the baseline site: every stored baseline and ceiling is measured there, and only there
(`make_baseline.py measure` refuses another site). meta64 is a runner: run campaigns and experiments on it, but its
numbers are MI350X numbers. A `full` run on meta64 still compares before with after (same node); its report does not
compare with the Crusoe ceiling.

## QOS

This project uses `amd-burst-qos` (account `amd-hyperloom`) by default on Crusoe. meta64 needs no QOS: its partition
`meta64` allows 5 days, and `interactive` allows 1 day (`backend_options: { partition: interactive }`). A burst job can be cancelled at any time when
other work needs the node ("preempted"). If that happens, run the whole request again. Do not resume it: a new job can
land on another node, and then before and after would not be on the same node.

## When something goes wrong

| What you see | What to do |
|---|---|
| `node ... is not clean (a GPU is in use)` in `runner.log` | Another user left something on that node. Add the node to `exclude_nodes` and submit again. We never touch other users' containers. |
| The state is `CANCELLED (PREEMPTED)` | Burst gave the node to other work. Submit the same request again. |
| `the job needs N min but <qos> allows M min` | Make `campaign_minutes` smaller, or use another QOS with `backend_options`. |
| `commit ... not found` | Run `git -C ../Arbor fetch origin` and try again. |
| `not submitted, cells not runnable` | That cell does not have enough baseline data yet. Pick another cell. |
| `full mode needs local.yaml arbor.gateway_env` | Do setup step 5. |
| A path in the request turns into `C:/.../Git/...` | You are on Windows Git Bash. Write the path in the YAML file with an editor, or put `MSYS_NO_PATHCONV=1` in front of the command. |

## Good habits

- Run a `smoke` request first when you change the setup or use a new cell.
- Compare only runs with the same cell, node type, and settings. Change one thing at a time.
- C1 P90 numbers are noisy. On three nodes the same setup gave e2e_norm P90 from 13.83 to 14.72 (6 % apart). A small
  change in that number from one run is not proof of anything.
- Keep the request files in git, so others can see what was run and why.
