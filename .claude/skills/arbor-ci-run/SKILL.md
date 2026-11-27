---
name: arbor-ci-run
description: Run an Arbor campaign on a GPU cluster with this repo's runner, from an AgentX CI run (or a launch record JSON) and an Arbor commit. Use when you clone arbor-ci to set it up, launch a campaign, smoke or A/B (full) run, pick a cluster, site or hold-job node, watch, stop or report a run, or find why a run was refused or failed.
---

# Run an Arbor campaign with arbor-ci

The scripts do the work. You collect the inputs, run a dry run, read what it found, submit, and watch.
Short guide for people: `README.md`. Every detail: `docs/how_to_run_a_campaign.md`. Rules: `CLAUDE.md`.
Python is `.venv/bin/python` (Windows Git Bash: `.venv/Scripts/python`, and put `MSYS_NO_PATHCONV=1` first).

## 1. Set up once

1. `bash scripts/setup_dev.sh`: makes `.venv`, copies `local.yaml.example` to `local.yaml`, runs the tests.
2. Fill in `local.yaml`: one entry per cluster under `sites:` (user, login host as `user@hostname`, paths),
   and `default_site`. Never use names from a personal ssh config.
3. `git clone https://github.com/AMD-AGI/Arbor ../Arbor` (or set `arbor.clone`).
4. `python scripts/setup_remote.py --site <site> --key-file <file with the gateway key> --test-gateway`.
   The user gives the key file. Never print, log or commit the key.
5. `python scripts/check_setup.py --site <site>`: every line must say PASS. Stop and report any FAIL.

## 2. Get the inputs from the user

- An **AgentX CI run**: a URL `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/<id>` or the id, and
- an **Arbor commit** (a full or short sha that exists in the Arbor clone).

A good CI run: workflow "Run Sweep" in InferenceX; ran on MI355X or MI350X (gfx950); has `bmk_agentic_*` artifacts
(a run that reused older results has none); less than 90 days old (artifacts expire).
Instead of a CI run, the user can give a **launch record JSON**: a request with `record:` (see "The launch JSON:
saved, and yours to give" in `docs/how_to_run_a_campaign.md`). Ask the user if the model or target is not clear; do not guess.

## 3. Dry run first

```bash
python scripts/run_from_ci.py --ci-run <url or id> --arbor-commit <sha> --dry-run [--target c1_p90] [--tp 4]
```

Nothing is sent. Read `runs/drafts/<model>__<target>__ci<id>/review.md`: it says where each value came from
(`exact`, `copied`, `derived`). The tool also writes `runs/requests/<name>.yaml` and `<name>.record.json`.
If the CI run measured several cells, it lists them: add `--target` (and `--tp`).

## 4. Choose where and how

| Choice | Flags |
|---|---|
| Cluster | `--site crusoe` (MI355X, default, burst QOS) or `--site meta64` (MI350X, runner only) |
| Scheduler | `--qos`, `--account`, `--partition` (meta64: `meta64` or `interactive`), `--nodelist`, `--exclude-nodes` |
| Own hold job | `--hold-job <job id> --nodelist <node>` (a job of yours whose node agent runs `ctrl/node<i>/*.sh`) |
| Arbor in one process | `--arbor-standalone` (ARBOR_STANDALONE=1); the default runs Arbor under its launcher |
| AgentX bench alone | `--mode bench --conc 1 4 16 --duration 600`: one server, a run per concurrency (`--bench-restart`: a fresh server each), no campaign, no LLM key; `generate_report.py` prints a row per concurrency |
| What Arbor optimizes | `--objective "total throughput"` (or `"P50 interactivity"`, ...; request `objective:` may be `{metric, bar_pct, guards}`), `--goal "<sentence for the agents>"`; concurrency = the target (`c16_tput`) |
| Mode | `--mode campaign` (default), `full` (before, campaign, after), `smoke`, `campaign_smoke` |
| Length | `--campaign-minutes` (default 480); `--gsm8k` adds the accuracy check |
| Weights | `--weights-path auto` (default), `none`, or an absolute path at the pinned revision |

Before you use a hold-job node, check that its GPUs hold no memory and that it runs no container. Never touch a
container, job or `ctrl/` file of another user or session. Ask the user before runs that cost many node-hours.

## 5. Submit, watch, stop, report

```bash
python scripts/run_from_ci.py ...same flags, without --dry-run...   # add --follow to wait and print the report
python scripts/poll_status.py --run-dir runs/<run_id>               # state, finished phases; fetches results
python scripts/cancel_run.py --run-dir runs/<run_id> [--cell <cell>] # our job only
python scripts/generate_report.py --run-dir runs/<run_id>
python scripts/submit_ab_test.py --request runs/requests/<name>.yaml # rerun a request (or an edited record)
```

Logs are in the remote run dir `<work_root>/runs/<run_id>/<cell>/`: `logs/runner.log` (each phase),
`logs/campaign.log` (Arbor), `logs/fetch.log`, `logs/slurm-<job>.out`, `logs/hold.out` (hold-job mode). The node keeps its own copy of `runner.log`, `campaign.log` and `arbor_rc` in `<node_local_root>/runlogs/<run>_<cell>/`.
After a restart (meta64 requeues a preempted job) the earlier attempt is in `attempts/<n>/`.

## 6. When the tool refuses or a run fails

| You see | Cause | Do this |
|---|---|---|
| "ran on ... this site's GPUs are gfx950" | NVIDIA or MI300/MI325 CI run | Pick a gfx950 run (`--allow-other-hw` only if the user asks) |
| "served ..., but ...'s stored baselines use ..." | Another model under the same CI name | Ask the user: new model in `config/matrix.yaml`, or `--allow-other-model` |
| "srtctl engine '...': only ('vllm', 'sglang') are supported" | atom, tokenspeed, or multi-node | Not supported; pick another run |
| "does not match its server's own arguments" | CI format changed | Do not edit around it; report it with the CI run id |
| "image ... cannot be pulled" | Image deleted upstream | Pick a newer CI run, or a record with an image that exists |
| "KV_OFFLOAD_BACKEND must be empty ..." | Record mixes KV settings | Rebuild the record with `run_from_ci.py` |
| `failed:fetch`, "not enough room" (exit 3) | Node disk full | Use `--weights-path`, or another node or site |
| "node ... is not clean (a GPU is in use)" | Another job's server on the node | Resubmit with `--exclude-nodes <node>` |
| "hint: you have N job(s) in QOS" | QOS submit limit reached | Wait, or choose another QOS or site |
| `failed:after` or another phase | Read `logs/runner.log` from the end | Compare with `docs/runner_test_matrix.md` (bugs found and fixed) |

## 7. Rules

- Never print, log or commit a key or token.
- Never touch other users' or other sessions' jobs, containers or files.
- Ask before runs that cost many node-hours.
- What you decided is recorded in the request file and `review.md`; tell the user which request you ran.
