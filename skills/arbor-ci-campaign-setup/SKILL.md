---
name: arbor-ci-campaign-setup
description: Set up and launch a new standalone Arbor campaign for a model, matching a pinned AgentX CI point's env/config/flags exactly. Use when the user gives an Arbor version/commit, a docker image (SGLang or vLLM), a pinned AgentX CI reference (commit and/or run ID), a model, and an optimization target (e.g. conc32+total_throughput, conc1+p90_interactivity) and wants a new 24h campaign set up and started. Covers both halves: resolving the CI point down to exact flags ("CI archaeology"), and scaffolding + launching the experiment. For cluster mechanics of an already-running campaign (hold job, ctrl/, stop/restart, reading progress), use `crusoe-arbor-run` instead — this skill hands off to it at Step 4.
---

# Setting up a new Arbor campaign against a pinned AgentX CI point

Scope: Crusoe MI355X Slurm cluster (same scope as `crusoe-arbor-run`). This skill covers everything *before and including* the first `sbatch`, i.e. producing a ready-to-run experiment directory and launching its 24h campaign. Cluster-side lifecycle after that (stop, restart, reading progress) is `crusoe-arbor-run`'s job — don't duplicate it here, cross-reference it.

Two worked examples back every template in `templates/`, cited throughout:
- **`musespark-c1-p90-sa`** (Muse Spark Open, SGLang + DFlash2 spec decode, conc1 / P90 interactivity) — the fuller, more recent pattern: node-local NVMe download via its own `dl.sh`, hand-rolled `ab.sh` for before/after, `GUIDE.md` written first. **Default template source.**
- **`dsv41f-c32-tput`** (DeepSeek-V4.1-Flash, vLLM, conc32 / total_throughput) — an earlier, leaner pattern: shared `/shared_nfs/huggingface/hub` cache (no per-campaign download), a `stage1.sh` pipeline (`prepare → smoke → ci_ref → bar → calibrate → campaign`) driving a Python `tools/measure.py` instead of hand-rolled bash for before/after, and a statistically-calibrated `GSM8K_MIN` floor. Cite this when the user wants the leaner/older style, or when GPUs are shared rather than NVMe-local.

Default to the `musespark-c1-p90-sa` style (`dl.sh` + `ab.sh`) unless the user asks for the `stage1.sh`/`tools/measure.py` style.

## Required inputs

Ask for whichever of these the user hasn't already given:

1. **Arbor version** — branch/tag and its resolved commit (e.g. `v0.9.2-baseline-test` = `2c2fd3df`). **Never hardcode or silently infer this** — not from this skill's own examples, not from another campaign's choice, not even from a prior stalled attempt on the *same* model that already picked one. A different campaign can need a different Arbor version for real correctness reasons (e.g. a specific version bug with synthetic-acceptance launches, or a prefix-cache fix only present after a certain commit) that are easy to get wrong by analogy. If the user hasn't stated an exact version/commit for *this* campaign, ask — do not proceed on an inherited or assumed default, even a previously-proven one.
2. **Docker image family** — SGLang or vLLM (determines which `ab.sh`/`campaign.sh` serve-command variant and container-mount pattern to use). **Same rule as Arbor version: never hardcode or silently reuse an exact image tag+digest from a prior attempt (even a stalled one on the same model) without confirming it with the user first.** The CI's own pinned image is usually the right answer and is worth citing as independently-confirmed ground truth (e.g. from the CI run itself, cross-checked across multiple sources) — but say so explicitly and let the user object, rather than silently carrying a cached value forward as if it were already agreed.
3. **Pinned AgentX CI reference** — a commit and/or an InferenceX Actions run ID. One can resolve the other (see Step 1).
4. **Model** (+ draft model, if speculative decode).
5. **Optimization target** — concurrency + metric, e.g. `conc32 + total_throughput` or `conc1 + p90_interactivity` (+ whether a secondary metric must be held as a non-regression guard, e.g. "P50 within 5%").

## Step 0 — codename and scope

Pick a short unique kebab-case codename: `<model>-c<conc>-<target>` (e.g. `musespark-c1-p90-sa`, `dsv41f-c32-tput`). Confirm the target host is Crusoe — if not, stop and say this skill and `crusoe-arbor-run` don't apply as-is.

## Step 1 — CI archaeology (do this before touching any node)

Goal: turn "model X, CI run/commit Y, concurrency Z" into an exact config key, recipe file, docker digest, and model/draft revisions — the same process used to build `master/README.md`'s Project Scope table and `master/reports/2026-11-08_model_summary.md`.

1. **Resolve the config key** for the pinned CI run (the string like `musespark-fp4-mi355x-sglang-agentic-dflash` or `kimik3.1-fp4-mi355x-vllm-agentic-dspark`) — usually findable in the session's own prior `meta.json` `_note` field if one exists, or from the CI run's job name/matrix entry on GitHub Actions.

2. **Two recipe eras — check which applies before reading further:**
   - **Newer models (static YAML):** `inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/<model>/<framework>/<board>/agentic.yaml`, with named `override_*:` blocks each setting `CONC: '<value>'` or `CONC: [...]` directly. Read the block matching the requested concurrency — the exact `server_flags`/`server_env` are right there.
   - **Older models (parameterized `.sh`):** `benchmarks/single_node/agentic/<model>.sh`, where `CONC` is an environment variable the CI workflow dispatch sets externally — **the static file has no sweep list in it.** The actual historical concurrency values are reconstructed from `inferencex-e2e/perf-changelog.yaml`'s prose `description:` field for that model's config-key entry (cite the `pr-link:` there) — e.g. MiniMax's entry says "Search space: TP4 at conc 1-64"; GLM-5.2's recurring entries mention a standing "TP8/EP1 arm at conc [1, 2, 4]". Don't claim a static value exists when it's workflow-dispatch-driven; cite the changelog prose instead.

3. **Confirm the docker image by digest, not tag.** Tags get deleted (confirmed case: Kimi K3's tag was later deleted; only the digest still pulls). `docker image inspect --format '{{join .RepoDigests " "}}' <tag>` after pulling, compare against the recorded digest.

4. **Confirm the exact HF model revision** (+ draft model revision, if speculative decode) from the recipe/config `_note`, not assumed from a README summary — READMEs can lag.

5. **Pitfall — `git ls-remote` only lists ref *tips*.** A commit buried mid-history in a PR branch will falsely look unreachable if you only check `refs/heads/*`, `refs/tags/*`, `refs/pull/*/head` tips. Before concluding a commit "isn't reachable," fetch the PR's full branch (`git fetch origin pull/<N>/head:refs/heads/<name>`) and `git log --oneline <branch> | grep <short-sha>` — the commit is very likely an ancestor, just not a tip. (Root-caused this exact mistake for MiniMax M3.1 Flash and GLM-5.2 commits this way; don't re-make it.)

6. **Pitfall — the bench client's InferenceX commit may need to be *older* than the CI pin.** If the Arbor version's bench script depends on `benchmarks/benchmark_lib.sh` and a later InferenceX restructure removed or moved it, pin the bench client to the last commit that still has it, not the CI's own commit. Confirmed case: Arbor `v0.9.2-baseline-test` + InferenceX `8a76a78e41`-and-later no longer ship `benchmark_lib.sh`; `musespark-c1-p90-sa` pins its bench client to `86699be610` instead, and notes in its `GUIDE.md` that absolute numbers won't match CI's own `infx` client as a result (before/after/every-iteration all use the *same* client, so the comparison stays internally valid even though it isn't numerically identical to the CI dashboard).

## Step 2 — write `GUIDE.md` first

Before any node action — matches the project's standing "write intent before launch" convention. Use `templates/GUIDE.md` as the skeleton: goal, pins table (Arbor, CI run+commit, recipe+point, image+digest, model+draft+revisions, bench client+corpus), decisions made with the user, known weaknesses/caveats discovered in Step 1, layout, constraints (never touch other users' containers/processes, don't loosen gates/tolerances/CI flags without the user, log every action to `reports/<date>_worklog.md`).

## Step 3 — `env.sh`

Single source of truth every other script sources. Fill in `templates/env.sh`'s placeholders: `TOP`/`EXP` paths, Arbor sha/branch, InferenceX bench-client sha, config/pack names, image tag+digest, container name, port, HF hub paths + model/draft/corpus id+revision, `WORKSPACE_ROOT` (the framework's editable source tree inside the container — SGLang or vLLM's install path).

## Step 4 — hold job

This is cluster mechanics — **use `crusoe-arbor-run` for the hold-job/node-agent/`ctrl/`-drop mechanics** (how the node agent polls `ctrl/`, atomic script drop via `ctrlput.sh`, how to stop/release). This skill only covers filling in `templates/hold_node.sbatch`'s codename/time/QOS/output-path fields and submitting it (`sbatch`, at `--time=1-00:00:00` since the QOS max-wall blocks a pending 3-day job; raise with `scontrol update JobId=<id> TimeLimit=3-00:00:00` once RUNNING).

## Step 5 — `prepare.sh`

Template the provisioning sequence from `templates/prepare.sh` (9 steps, reuse verbatim where model-agnostic, edit only the marked substitution points):
1. node checks (docker daemon up, GPUs idle, list — never touch — other users' containers)
2. pull the CI image by digest (anonymous pull if the shared Docker Hub token has expired), tag it, verify the digest
3. install Arbor at the pinned commit (archive + local git commit for `has_changes` to work), plus the pack + config
4. the driver's venv (`pyyaml`, `claude-agent-sdk`)
5. verify the HF cache (model/draft/corpus) resolves at the pinned revisions — fail loudly if `dl.sh` hasn't been run
6. start the long-lived container with the right mounts (`/hf-hub`, InferenceX, Arbor, sessions, mmap cache, the experiment dir, the profiler skill read-only)
7. capture stack versions, commit the framework tree's as-shipped working-tree state as a baseline (so Arbor's diffs start clean), resolve `editable_packages` paths (notably `aiter`/`flydsl`, whose install path isn't static)
8. verify the aiperf bench environment is ready (InferenceX's `install_agentic_deps`)
9. build `ruppell` for the profiler skill (not fatal if it fails)

## Step 6 — `dl.sh`

Model + draft + corpus download to node-local NVMe at exact pinned revisions (`templates/dl.sh`) — writes `refs/main` = the pin for each, for offline loads; verifies every safetensors shard the index expects actually resolves. Skip this step (and prepare.sh's step 5 check) if following the `dsv41f-c32-tput`-style shared-cache pattern instead.

## Step 7 — `ab.sh` (before)

Smoke test first (short duration, relaxed gates), then a full AgentX replay + GSM8K on a fresh server — **before the campaign starts.** `templates/ab.sh` serves the tree exactly as Arbor's own launcher would (`arbor/frameworks/{sglang,vllm}.py`'s command, server_env, GPU mask), waits for health, runs the pack's `bench_agentx.sh`, then `sglang.test.few_shot_gsm8k` (or vLLM's equivalent), and records everything under `runs/<tag>/`.

**This is the "number of record" — not the same thing as Arbor's own internal rolling `baseline` field** used for the campaign's own KEEP/REVERT decisions. **This repeatedly diverges in practice — always report both, flag if they disagree:**
- Muse Spark Campaign5: +27.0% iterative vs. +24.4% full before/after re-measure.
- Kimi K3.1 Pro C1 P90: +7.9% iterative vs. essentially flat **−0.2%** full before/after re-measure (the landed KEEPs didn't hold up under independent re-measurement — this was reported as an open, unresolved discrepancy, not silently reconciled).

## Step 8 — Arbor pack config

`configs/<board>/<codename>.json`. Map the user's plain-language optimization target to Arbor's `metric_name`/`metric_direction`/`secondary_name` (+ gate thresholds for a non-regression guard metric, e.g. `secondary_threshold_rel`/`secondary_noise_band_pct`). Set `time_budget_minutes: 1440` for a 24h campaign.

**Pitfall:** never put a raw number under `target.<metric_name>` unless it is genuinely meant as an early-exit threshold — Arbor's `_resolve_target` reads *any* numeric value under the metric's own key as a literal early-exit bar, not as a documentation reference. This cost a real campaign exit after only ~2h once (Kimi K3.1 Pro C1 P90, root-caused and fixed by keeping only prose + a `baseline_*`-prefixed reference block, no raw number under `target.<metric>`).

## Step 9 — `campaign.sh`

`templates/campaign.sh`'s budget computation (campaign-minutes-used-so-far vs. job-walltime-left minus a buffer for the after-measurement) and cleanup/trap pattern are already generic — reuse near-verbatim, just update the container/process-kill match pattern for the framework (SGLang's `sglang.launch_server`/`sglang::` vs. vLLM's `vllm serve`/equivalent). Launches `python -m arbor run --config <config> --cooldown 30 --no-review-gate`, sets the 2h Bash timeout for the driver's agents (`BASH_DEFAULT_TIMEOUT_MS`/`BASH_MAX_TIMEOUT_MS = 7200000` — a 3600s AgentX replay with setup outlasts the CLI's 30 min default), unsets `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` before the driver (Arbor sets the GPU mask per server).

## Step 10 — after the campaign

Run `ab.sh` again (tag `after`) for the after-measurement. Report the iterative best *and* the full before/after re-measure side by side, explicitly flagging if they disagree (see Step 7's pitfall) — don't just report the iterative number as "the" result.

**The after-run must serve the KEPT launch, not the campaign's input config.** Arbor keeps the live launch in `<workspace_root>/arbor/config.json`: it seeds that file at init, every KEEP that changes a flag or env edits it, and it is committed with the tree. `arbor/execute/bench.py` (`_live_flags`/`_live_env`) serves its `server_flags` (unless the key is absent/None) and merges its `server_env` over the static config's. `configs/<board>/<codename>.json` (`$CONFIG`) is only the campaign's *input*. An after-run built from `$CONFIG` alone silently drops every config-level KEEP. `templates/ab.sh` now reads the workspace config when it exists (override: `AB_CONFIG=<file>`) and prints `[ab] launch from …`. **Verify before trusting the after number:** the after-run's `serve.sh`/`server.env` (or the live `/proc/<pid>/cmdline`) carries every flag/env change in the KEEP list. Confirmed case: kimik3-c1-p90-sa (2026-11-16), the first after-run lacked all 3 kept config changes (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, capture size 8192, `--prefix-match-unit 256`) and was stopped and redone. Suspected same cause, unverified: Kimi K3.1 Pro C1's +7.9 % iterative vs −0.2 % after (its only real KEEP was the config change `num_speculative_tokens` 7→5).

## Pitfalls appendix

Cluster-general pitfalls (silent-Monitor 10-min timeout, `CLAUDE.md`/`AGENTS.md` leak if the Arbor tree sits inside the operator's own repo, non-atomic `ctrl/` drop, tcsh quoting on `crs5`, `/tmp` not writable on `crs5`, hold-job wall-time submission limits) are **`crusoe-arbor-run`'s** — don't duplicate, cross-reference. This skill's own additions, specific to setting up a CI-matched campaign, are the ones called out inline above in Steps 1, 7, 8, and 10:
- `ls-remote` tips-only blind spot when resolving a pinned commit (Step 1)
- bench-client InferenceX commit may need to predate the CI pin (Step 1)
- iterative-baseline vs. full-before/after divergence — always report both (Step 7)
- raw numeric `target.<metric>` triggers an unintended early-exit threshold (Step 8)
- after-run built from `$CONFIG` instead of the kept `<workspace_root>/arbor/config.json` drops every config-level KEEP (Step 10)
