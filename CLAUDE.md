# Rules for agents (and people) working in this repo

Setup and use: `README.md`. Design: `docs/design.md`. Each rule below exists because its problem happened once; fix a
new class of problem in the repo (script, config, check, doc), never as a one-off local patch.

## Repo

- Remote: `origin` = `qianghan-amd/arbor_runner`, branch `main`. Never force-push, never add a remote, never `git init`
  (denied in `.claude/settings.json`).
- `.claude/settings.json` (checked in) allows the routine commands (git status/diff/log/add/commit/pull --rebase/push
  origin main, pytest, the repo venv). Add a rule there, with a reason in the commit, instead of re-approving the same
  prompt; machine-only choices go in the untracked `.claude/settings.local.json`.
- Tests: `python -m pytest -q tests` before every commit.
- Commit locally as you go; push at checkpoints only (end of a phase, or a stop-and-report point), not after every
  commit (user decision 2026-11-19).
- Bytes are LF everywhere (`.gitattributes`). Provenance sha256 values in `baselines/` are over LF files. Python on
  Windows: pass `newline="\n"` to `write_text`/`open`. `git archive`: `-c core.autocrlf=false -c core.eol=lf`.
- Edit code that contains backslash escapes with an editor tool, not a shell heredoc or `sed`: the shell layer turned
  `b"\r\n"` into raw bytes once. After any scripted edit: `python -m py_compile` and `grep -c $'\r'`.

## Data

- Every number in `baselines/` traces to `evidence/` (raw files with sha256) or to a cited source line; nothing is
  guessed. A value not found stays null and is listed (`launch.missing`, `data_quality_warning`).
- Field names mirror AgentX's raw nested structure (`intvty`, `e2e_norm_intvty`); see `scripts/arbor_ab/agentx.py`
  for the traps (bench_output `total_throughput` is per GPU; old validators' `p90_interactivity` = `intvty.p90`).
- Never soften a finding: a failed gate, a client mismatch, a moved tag stay in the record as measured.

## Cluster (Crusoe)

- Scheduler/cluster specifics live only in `scripts/arbor_ab/backends/<name>.py` (design.md 5.1). The orchestrator,
  baselines, gate and report call the `Backend` contract; `scripts/runner/runner.sh` stays backend-neutral.

- The scheduler is **Spur** (Slurm-compatible, `spur 0.15.0`): check a flag with `--help` before use; `sacct` has no
  `-X`/`-P`; stderr is not merged into `--output` (it goes to `$HOME/spur-<job>.out` unless `--error` is set).
- Network steps on the node (image pull, downloads, pip) fail transiently (Docker Hub CDN 504 seen): wrap them in the
  runner's `retry`.
- Login shells are tcsh: send multi-step commands as a script on stdin (`ssh host bash -s`, `ssh host python3 -`).
- Nothing long-lived on the login node (design.md section 6); node work runs only inside the cell's Slurm job.
- Remote work root (`local.yaml remote.work_root`) stays outside any git repo: Arbor's agents load every `CLAUDE.md`
  above their working directory.
- Never touch other users' jobs, containers or files. Ask before a run that costs significant node-hours.

## Secrets

- Never print, log, commit or put on a command line a token or key. GitHub: through the relay
  (`scripts/arbor_ab/github_relay.py`: token read on the relay host, given to curl on stdin). LLM gateway: the file
  `scripts/setup_remote.py` writes (mode 600); jobs source it, containers never see it.
