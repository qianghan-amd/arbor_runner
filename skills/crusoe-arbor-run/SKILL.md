---
name: crusoe-arbor-run
description: Crusoe cluster only. How to run and watch an Arbor campaign (standalone or under Hyperloom) on the Crusoe MI355X Slurm cluster from this workstation — the hold job with its node agent, the ctrl/ drop, where the Arbor driver and its Claude agents run, how they reach the GPUs, how to stop, and how to read progress. Use when you start, restart, stop or monitor a campaign on Crusoe, or when someone asks how the run is wired (ssh? tmux? where do the agents run?).
---

# Running Arbor on the Crusoe cluster (Crusoe only)

Scope: the Crusoe MI355X Slurm cluster (login node crs5, partition amd-spur-v2, QOS amd-hyperloom-qos, `/shared_nfs`, node NVMe `/mnt/m2m_nobackup`). Other clusters (e.g. Ruby) differ in login node, shells, storage and docker policy; do not apply this as is.

No ssh to the compute node, no tmux. Every process on the node is a child of one Slurm hold job.

## 1. Launch chain

1. From the workstation, `ssh crs5` (the login node; `crusoe2_006` hangs). Its login shell is **tcsh**: put any multi-step command in a bash script and run it with `bash`.
2. A **hold job** (`templates/hold_node.sbatch`) holds one MI355X node, exclusive, on `amd-spur-v2` / QOS `amd-hyperloom-qos`. Its only work is a **node agent**: every 20 s it looks in `<experiment>/ctrl/` for `*.sh`, renames each to `*.sh.started`, and runs it once in the background on the node; output goes to `<experiment>/logs/<name>.<jobid>.log`.
3. To run anything on the node, write a bash script, put it in `ctrl/` **atomically** (`templates/ctrlput.sh`: scp to a hidden tmp name, then `mv`), and read its log. `/shared_nfs/...` (= `/home/qiangh/arbor_dsv41flash` on crs5) is the same directory on crs5 and on the node.
4. The campaign script (e.g. `campaign.sh`) starts the **Arbor driver** on the node: `python -m arbor run --config <config> --no-review-gate`, from the experiment's own venv (`<experiment>/.venv`: pyyaml, claude-agent-sdk), with `cwd` = the Arbor tree.

## 2. Where the agents run

- The driver starts each agent session through **claude-agent-sdk's bundled Claude Code CLI** (2.1.x), as a child process of the driver, **on the node itself, not in a container**.
- `HOME=<experiment>/agent_home`; the LLM gateway variables come from `gateway.env` (`set -a; . gateway.env; set +a`). Only variables pass; no token is ever written into a container.
- Set the agents' Bash timeout to 2 h from the start (`BASH_DEFAULT_TIMEOUT_MS` / `BASH_MAX_TIMEOUT_MS = 7200000`, as env and in `$HOME/.claude/settings.json`): a 3600 s AgentX replay with its setup outlasts the CLI's 30 min default.
- Unset `ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES` before the driver; Arbor sets the GPU mask per server (`gpu_ids`).

## 3. How the agents reach GPUs and code

- One long-lived container of the CI image (started once by `prepare.sh`, `--entrypoint /bin/bash ... sleep infinity`, `--device /dev/kfd /dev/dri`, `--network host --ipc host`). Arbor never creates it.
- Agents and the harness run `docker exec <container> ...` to edit the framework source (SGLang / vLLM, AITER, FlyDSL — the config's `workspace_root` and `editable_packages`), start and stop the server, profile, and run the bench.
- Mounts: the HF cache (model, draft, corpus, `refs/main` = the pins) at `/hf-hub`, InferenceX at `/workspace/InferenceX`, the Arbor tree at `/workspace/Arbor`, sessions at `/workspace/sessions`, the experiment at `/workspace/exp`.

## 4. Lifecycle

- Disconnecting the workstation changes nothing: the processes belong to the Slurm job.
- **Stop a campaign cleanly:** `touch <session>/STOP` — Arbor ends after the current iteration.
- **Stop a stray script of ours:** a ctrl script that kills only our own processes (match our script paths) and removes only our container.
- **End the node:** `touch <experiment>/ctrl/RELEASE` (the agent loop exits, the job ends). Ask the user first: a released node may not come back (QOS group limit: 5 nodes).
- The job's end kills everything; budget the campaign to end before it.

## 5. Reading progress

- `ssh crs5` and read the shared logs and session files (`logs/<name>.<jobid>.log`, `<session>/...`).
- Node-local paths (`/mnt/m2m_nobackup/...`) and `ps`/`docker`/`rocm-smi` exist only on the node: query them through a ctrl script.
- Files owned by root or a sandbox user: read them through a short-lived own container (`docker run --rm --name qh_<x>_rd -v <dir>:<dir>:ro <image> bash -c '...'`). `--pid=host` is denied on this cluster.

## Pitfalls (each cost time before)

- **The Claude CLI ends a Monitor that prints nothing for 10 min** (CLI 2.1.323, measured 2026-11-13: silent watch ended at 608 s; a 30-s heartbeat ran 939 s). Arbor v0.9.2-bt's bench prompt suggested a silent `until [ -s bench_output.json ]; do sleep 30; done` watch, so every bench > 10 min was cut and the run aborted after 3 NOT MEASURED. Check the Arbor version's bench prompt before a long-bench campaign; campaign 5's copy carries a local fix (heartbeat line).

- **Keep the Arbor tree, the agents' HOME and the session OUTSIDE this repo** (e.g. `/shared_nfs/qiangh/arbor_runs/<code name>/`). Claude Code loads every `CLAUDE.md` above the agent's cwd: with the tree under the repo, campaign 5's agents loaded the repo's CLAUDE.md, read `MEMORY.md` and an earlier campaign's results (an answer leak), and wrote into our worklog and MEMORY.md (2026-11-13). Also add a pack rule that forbids reading or writing the operator's repo.

- **Non-atomic drop:** scp straight into `ctrl/` lets the agent run a half-written script; it exits silently with an empty log. Always use `ctrlput.sh`.
- **Never pipe a long run through `head`:** it closes the pipe and kills the run mid-way (left a stale `.git/index.lock`).
- **Never overwrite a script while it runs:** bash reads scripts lazily. Run a frozen copy per run (`cp x.sh logs/x.$TAG.sh; exec bash logs/x.$TAG.sh`).
- **tcsh on crs5:** `2>/dev/null`, `$(...)` with quotes, and `'...'` inside `"..."` break; wrap in `bash -c '...'` or a script.
- **`/tmp` on crs5 is not writable** for scp; use the experiment dir.
- **Hold jobs:** submit at `--time=1-00:00:00` (a pending 3-day job is blocked by the QOS max wall); once RUNNING, `scontrol update JobId=<id> TimeLimit=3-00:00:00`.
- **Other users' containers on the node** (e.g. an idle leftover): never touch them; list them in `prepare.sh` and report.
- **No `curl` / `gh` on the workstation:** read GitHub through `git` (partial clone; a CI commit may live only on `refs/pull/*`).

## Templates

- `templates/hold_node.sbatch` — the hold job with its node agent (`sed "s/CODE_NAME/<code name>/g"`, then `sbatch` from the experiment dir).
- `templates/ctrlput.sh` — atomic drop of a script into `ctrl/`.

Worked examples: `experiments/dsv41f-c32-tput/` (campaign 1, standalone), `experiments/musespark-c1-p90-sa/` (campaign 5, standalone: `env.sh`, `prepare.sh`, `ab.sh`, `campaign.sh`, `GUIDE.md`).
