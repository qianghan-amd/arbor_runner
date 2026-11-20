# Arbor A/B Test System — Design Plan (for review)

**Date:** 2026-11-18
**Status:** draft, pre-implementation — nothing below is built yet
**Scope:** MVP, extensible toward CI/CD; Crusoe cluster only for now

## 1. Purpose

This is not a model-tuning tool. It is a **continuous regression/validation benchmark for Arbor itself**: given an Arbor code change (a PR touching `gate.py` / search policy / measure module / skills, or a routine weekly check), does Arbor still find real improvements on a fixed set of `(model, target)` combinations — and does it match or beat the best result ever found for that combination?

Every result is judged the same way we already validated all 12 tracked baselines this session: a real before/after re-measure plus a GSM8K correctness check, never just a live/iterative number.

## 2. Campaign matrix

- **`model` and `target` are selectable parameters, not hardcoded.** A `target` is a `(concurrency, optimization metric)` pair, e.g. "C16 + total_throughput" or "C1 + interactivity_P90".
- **Host configuration (framework, TP/EP/DCP, GPU count) is inherited from whatever that model+target's baseline used** — the system does not invent its own packing/placement scheme per model.
- Currently known matrix (**corrected 2026-11-19** — a cell-by-cell audit of `model_summary.md` against this table found it had dropped Kimi K3, a fully complete model, in favor of Kimi K3.1 Pro, which has real data-quality problems in both its cells — see §2.1):

| Model | Throughput target | P90 target |
|---|---|---|
| Muse Spark | C16 (SGLang, TP4+EP4) | C1 (SGLang, TP4+EP4) |
| Kimi K3 | C32 (vLLM, TP8+DCP8) | C1 (vLLM, TP8+DCP8) |
| Kimi K3.1 Pro | C32 (vLLM, TP8+DCP8) | C1 (vLLM, TP8+DCP8) |
| DSv4.1 Flash | C32 (vLLM, TP4) | C1 (vLLM, TP4) |
| GLM-5.2 | C16 (SGLang, TP4+EP4) | C1 (SGLang, TP4+EP4) |
| MiniMax-M3.1-Flash | C32 (vLLM, TP4) | C1 (vLLM, TP4) |

### 2.1 Baseline data-quality audit (do this before porting — do not treat all cells as equally trustworthy)

Verified cell-by-cell against `master/reports/2026-11-08_model_summary.md` on 2026-11-19. Of the (now 7-model) matrix:

| Target | Status |
|---|---|
| Muse Spark tput | ✅ complete (real before/after + GSM8K before/after, both by us) |
| Muse Spark P90 | ✅ complete |
| **Kimi K3 tput** | ✅ complete — this is the cell that was missing from the matrix entirely |
| **Kimi K3 P90** | ✅ complete |
| Kimi K3.1 Pro tput | ❌ **incomplete** — no real before/after re-measure, no GSM8K at all |
| Kimi K3.1 Pro P90 | ⚠️ **unresolved** — has a real before/after P90 measurement, but it *contradicts* the iterative tracking (+7.9% iterative vs. −0.2%/essentially flat on re-measure) and was never reconciled. GSM8K is after-only (no before), so correctness drift can't be checked either. Do not port this as a trustworthy baseline without re-running it. |
| DSv4.1 Flash tput | ⚠️ **partial** — real before/after throughput exists, but GSM8K is before-only (0.624) and never re-measured after. That 0.624 is also suspiciously low vs. the same model's own C1 column (~0.965) — worth independently re-checking before trusting, not just porting as-is. |
| DSv4.1 Flash P90 | ✅ complete |
| GLM-5.2 tput | ✅ complete |
| GLM-5.2 P90 | ✅ complete |
| MiniMax tput | ❌ **incomplete** — the only number here is sourced from another team (Chris Cai, Agent Board `ArborDevelopment#37`), not independently measured by us, and has no GSM8K at all |
| MiniMax P90 | ✅ complete (this one *is* our own independent measurement) |
| Qwen3.8 (both targets) | ❌ **no data in this project** — only exists as a differently-sourced table on the Confluence wiki, not in `model_summary.md` |

**Net: of 14 cells across 7 models, 8 are solid (both throughput models' worth), 4 have a real, specific problem (not just "missing," each with its own distinct issue), 2 have zero data.** `make_baseline.py` should port the 8 solid cells as `ci_verified` baselines normally; the 4 problem cells should still be recorded (so the gap is visible in the system, not silently dropped) but flagged with an explicit `data_quality_warning` field rather than treated as equal to the solid ones — porting bad numbers as if they were clean baselines would make the A/B system's own reference points untrustworthy from day one.
| Qwen3.8 | C32 (TBD) | C1 (TBD) |

## 3. Baseline & epoch system

### 3.1 Lifecycle
Baselines are **run once and stored, not re-run on every PR test.** They are refreshed on a human-paced (weekly/biweekly) cadence, or on demand when the pinned image/recipe changes.

### 3.2 Pristine config source (two first-class paths)
- `source_type: agentx_ci` — resolved via CI archaeology (commit/run/recipe), using the existing `arbor-ci-campaign-setup` skill.
- `source_type: user_custom` — the user supplies their own launch config directly (server flags, image, model path); no CI archaeology needed.

### 3.3 Epoch model
- **`epoch_id` is derived from a fingerprint**: `(image, ci_commit/ci_run)` for the CI path, or a content hash of the custom config for the custom path. Not manually assigned.
- **Same fingerprint on a baseline refresh → same epoch continues** (ceiling keeps accumulating; a large unexplained drift between the old and new measurement is flagged, not silently absorbed).
- **Different fingerprint → new epoch.** The old epoch is **archived, never deleted**, tagged `superseded_by: <new_epoch_id>`. Comparisons default to **within the current epoch only** — a new image's results are never silently compared against an old image's ceiling.

### 3.4 Verification gate (mandatory before a baseline is trusted)
1. **Structural check (cheap, no GPU time):** diff the derived server flags/image/model path against the recipe source file itself. Catches transcription errors before spending compute.
2. **Numeric check (one real measurement):** run the derived config once, fetch the CI's own historical benchmark artifact for that exact commit/run (GitHub Actions artifact API — proven this session for DSv4.1 Flash, matched within 0.02%), compare within a tolerance (proposed ±3%). Within tolerance → baseline marked `ci_verified: true` and stored. Outside tolerance → **halt, do not auto-store, flag for human review.**
3. If the CI artifact has expired (not always available — artifacts have a retention window) → baseline is marked `unverified_against_ci: true` (usable, lower confidence), requiring a human look at the structural check instead.

### 3.5 Known failure mode: the pinned image itself can disappear
Confirmed real case this session (Kimi K3's image tag was deleted). Three independent health flags per baseline, checked on the same refresh cadence:
- `ci_artifact_available` — can we still fetch CI's historical number for comparison
- `image_available` — can the pinned image still be pulled from upstream
- `local_mirror_exists` — do we have our own copy

**Mitigation:** at baseline-creation time, mirror the image to local/shared storage (`local_mirror_path` recorded alongside the upstream reference) so the system is not permanently dependent on the upstream registry.

### 3.6 Schema (JSON, versioned)
Field names mirror AgentX's own raw metric structure directly — do not invent flattened names. This is a direct lesson from this session (the `intvty` vs `e2e_norm_intvty` mix-up cost real time twice).

```json
{
  "schema_version": 2,
  "model": "dsv41flash",
  "target": {
    "conc": 32,
    "primary_metric_path": "metrics.total_tput_tps",
    "primary_metric_per_gpu_path": "metrics.tput_per_gpu",
    "secondary_metric_path": "metrics.e2e_norm_intvty.p50"
  },
  "host": { "framework": "vllm", "tp": 4, "ep": 1, "num_gpus": 4, "recipe_sh": "..." },
  "fingerprint": {
    "source_type": "agentx_ci",
    "image": "vllm/vllm-openai-rocm:deepseekv41-flash-0909",
    "local_mirror_path": null,
    "ci_commit": "86699be610...", "ci_run_id": "34657748450",
    "ci_commit_date": "2026-09-11", "recipe_fingerprint": "239ed287f0..."
  },
  "epoch_id": "dsv41flash__c32_tput__86699be610",
  "health": { "ci_artifact_available": true, "image_available": true, "local_mirror_exists": false },
  "measured": { "date": "2026-11-07T10:40:22Z", "node": "crsuse2-m2m-v3-244", "job": "272282", "replays": 1, "duration_s": 3600 },
  "metrics": {
    "total_tput_tps": 68051.03812, "tput_per_gpu": 17012.75953,
    "intvty":          { "p50": 21.37104, "p90": 13.3463, "p95": null },
    "e2e_norm_intvty": { "p50": null, "p90": null, "p95": null },
    "gsm8k": { "flexible_extract": null, "strict_match": null }
  },
  "artifacts": { "bench_output": "...", "agg_json": "...", "gsm8k_log": "..." },
  "superseded_by": null,
  "data_quality_warning": null
}
```
Stored at `baselines/<model>/<target>/<epoch_id>.json`, with a `latest` pointer per `(model, target)`.

## 4. Triggering a PR test

- **Identified by commit SHA/URL, not PR number** — a PR mutates (new pushes); a commit does not.
- **Tests the branch head directly**, no simulated merge-into-main (simplicity over completeness; the trade-off is accepted).
- **MVP submission is a manually-edited request file**, not a web form:

```yaml
# runs/requests/my-test.yaml
commit: "2803f36ac4e09aedaf5d18b84d436c9365acb4c3"
targets:
  - { model: muse_spark, target: c16_tput }
  - { model: dsv41flash, target: c1_p90 }
```
```bash
python scripts/submit_ab_test.py --request runs/requests/my-test.yaml
```
A web UI is an explicit future layer on top of the same entry point, not designed yet. Permissions/login are deferred (human SSO now; will need a separate machine-auth/service-account path for CI/CD later — noted so the eventual design doesn't have to be redone for that).

## 5. Scheduling

**Reuses the cluster's scheduler directly — no custom scheduler.** Every selected `(model, target)` cell becomes its own job; the scheduler's own QOS/queueing handles contention. Node count is an open parameter, decided later. On Crusoe the scheduler is Spur, a Slurm-compatible implementation (verified 2026-11-19); QOS `amd-burst-qos`, account `amd-hyperloom` (user decision 2026-11-20; it was `amd-agentx-qos` until then): 24 h wall, preempt mode cancel, so a cancelled A/B run is rerun whole, never resumed on another node.

### 5.1 Execution backend boundary (extensibility point, added 2026-11-19)

How and where a cell's job runs is isolated behind one small contract, so another cluster's scheduler, or a "run on this host, no scheduler" mode for local/dev use, can be added without touching the orchestrator, the baselines, the gate or the report.

- **The payload is backend-neutral.** `scripts/runner/runner.sh` runs every phase of one cell (fetch, prepare, before, campaign, after) on one node with bash, docker and GPUs. It reads only its cell directory (`cell.env`, `arbor_config.json`), so it runs the same under any backend.
- **The contract** (`scripts/arbor_ab/backends/__init__.py`, class `Backend`): `site()` (paths the payload needs on the node), `run_dir(run_id)`, `job_script(cell_dir, name, minutes)`, `stage(...)` (make the files visible to the job), `submit(cells)`, `status(cells)` (normalized states `PENDING RUNNING SUCCEEDED FAILED CANCELLED UNKNOWN` + the scheduler's own word), `fetch(cell, dir)` (the small result files), `cancel(cell)`, optional `extend(cell, status)`, and `check()` / `setup()` for `check_setup.py` / `setup_remote.py`.
- **Backends:** `crusoe_slurm` (short ssh calls to the login node, Spur commands, `/shared_nfs` work root) is the only one. A new backend is one file, `backends/<name>.py` defining `BACKEND`, selected by `backend: <name>` in `local.yaml`. Each run records its backend in `run_config.json`; `poll_status.py` / `cancel_run.py` refuse a run made with another backend.
- **What stays outside:** the GitHub relay and the registry checks (baseline verification, §3.4–3.5) are not job execution; they use their own relay host.

## 6. Orchestrator placement (and why)

**Never runs on the Crusoe login node.** Crusoe's login node periodically kills long-running tmux/herdr-style processes (CPU-policing policy, observed repeatedly this session). The orchestrator runs anywhere else — a developer's machine today, a lightweight always-on service or CI runner later — and only talks to Crusoe via short-lived SSH calls (submit, `squeue`, fetch a result file). No long-lived foreground/background process ever sits on the login node. (These ssh calls are the `crusoe_slurm` backend's, §5.1; the orchestrator itself is cluster-agnostic.)

**Extensibility requirement:** `submit_ab_test.py`, `poll_status.py`, and `generate_report.py` are each **stateless, idempotent CLI tools** with no hidden state beyond what's in `runs/` on disk. `submit_ab_test.py` writes `run_config.json` and exits immediately — it does not block waiting for results. This means *who* calls these scripts (a human, cron, a future CI pipeline step) is fully pluggable without changing the scripts themselves.

## 7. Execution: skill (judgment) vs. script (deterministic)

| | Baseline creation | PR test execution |
|---|---|---|
| Frequency | Low (weekly/biweekly, or on image change) | High (every selected PR) |
| Driven by | `arbor-ci-campaign-setup` skill (+ `crusoe-arbor-run` for cluster mechanics) — CI archaeology, agent judgment | A deterministic, parameterized script derived from `templates/ab.sh` (already proven this session for every before/after measurement) |
| Needs an LLM agent at run time? | Yes | No — all parameters come from the stored baseline JSON |

**Explicit boundary (write this down, do not blur it):**
> Baseline management is human-paced; PR testing is fully automatable.
> Creating/refreshing a baseline requires agent-assisted CI archaeology and cannot run fully unattended.
> Testing a PR against an already-established baseline is fully deterministic and scriptable — this is the path intended to eventually run unattended in CI/CD.

**Guardrail — every PR test re-runs its own fresh before+after, never reuses the baseline's stored numbers as "before":**
- `before` = the pristine config (matching the baseline's recorded host/image/recipe), measured fresh on this test run
- `after` = the result of running the PR's Arbor code through its campaign and taking the final kept tree
- Both measured under matching conditions (same cluster/image/config; same physical node is a natural side effect of one hold-job covering both phases, not an independent requirement)
- Guardrail: full AgentX before+after **and** GSM8K before+after, every time

**Comparing against an arbitrary prior run (not just the registered baseline):** supported via a `--reference <run_id>` option on the comparison step — it reads that prior run's **config** (not its old stored numbers) and re-measures it fresh as the new "before," preserving the same-session rigor principle. This does not get promoted into the formal `baselines/` registry (which stays reserved for the official pristine reference per epoch).

## 8. Output

`comparison.json` per `(model, target)` per run: before → after → current-epoch ceiling, GSM8K pass/fail, plus a `verdict` field.

**`verdict` is manual for MVP** (`null` / `"manual_review_pending"` — a human reads the report). This is a deliberate placeholder: automating it later (pass/fail thresholds, gating a PR merge) only requires `generate_report.py` to populate this one field — no schema or downstream change needed.

## 9. Repo layout

```
arbor-ab/
├── README.md                      # clone -> configure -> run; Crusoe-only noted
├── config/
│   ├── matrix.yaml                # which (model, target) combos exist
│   ├── models/*.yaml              # per-model static facts
│   └── cluster/crusoe.yaml        # QOS names, node naming, ssh alias (isolated for future clusters)
├── local.yaml.example             # template for the user's own ssh/github-token config (gitignored when filled in)
├── skills/
│   ├── arbor-ci-campaign-setup/   # vendored, not a global-skill dependency
│   └── crusoe-arbor-run/          # vendored
├── scripts/                       # (as built, 2026-11-19)
│   ├── runner/                    # the backend-neutral job payload: runner.sh (all phases of one cell),
│   │                              #   serve_cmd.py (launch rendered by the Arbor under test), fetch.py
│   ├── arbor_ab/                  # shared library: baseline, gate, agentx field map, render, report, ...
│   │   └── backends/              # execution-backend boundary (§5.1): __init__.py (contract), crusoe_slurm.py
│   ├── make_baseline.py           # (P7, not built yet) CI archaeology or user-custom intake -> verify -> store + mirror
│   ├── verify_baseline.py         # two-layer gate (§3.4) against the CI's bmk artifacts
│   ├── check_baseline_health.py   # ci_artifact_available / image_available / local_mirror_exists
│   ├── setup_dev.sh, setup_remote.py, check_setup.py   # day-1 setup and its check (§10)
│   ├── submit_ab_test.py          # reads a request file -> one job per cell through the backend -> exits
│   ├── poll_status.py             # job + runner status, fetches result files; callable repeatedly
│   ├── cancel_run.py              # stops a run's own jobs
│   └── generate_report.py         # builds comparison.json (incl. verdict placeholder)
├── baselines/<model>/<target>/<epoch_id>.json (+ latest pointer)   # ships pre-populated with the 12 known baselines
├── runs/
│   ├── requests/<name>.yaml       # manually-edited submission files
│   └── <run_id>/<model>__<target>/{before,after}/{bench_output.json, gsm8k_result.json}, comparison.json
└── docs/design.md                 # this document, kept current as the design evolves
```

No web UI in the MVP. No raw trajectory/transcript/hook-trace data is stored in `baselines/` or `runs/` — only AgentX-leaderboard-equivalent summary metrics (a few KB per record regardless of how many runs accumulate). Full raw artifacts may exist transiently on the compute node / `shared_nfs` during a run for manual debugging, but are not curated long-term by this system.

## 10. User onboarding (day 1)

1. Prerequisites: own Crusoe SSH access + node/QOS quota, a GitHub token with InferenceX read access, Python 3.x.
2. `git clone` + `pip install -r requirements.txt`.
3. Copy `local.yaml.example` → `local.yaml`, fill in ssh host/user/github token (gitignored).
4. `python scripts/check_setup.py` — connectivity only (SSH reachable, GitHub token valid, skills present). Not a permissions/quota check.
5. `baselines/` ships pre-populated with the 12 already-validated baselines — no setup needed to start testing PRs against those models.
6. Edit a `runs/requests/*.yaml`, run `submit_ab_test.py --request ...`.
7. `poll_status.py` / re-read `runs/<run_id>/.../comparison.json` for results.
8. Only if testing a model/target not yet covered: `make_baseline.py --model ... --target ... [--ci-commit ... | --custom-config ...]`.

## 11. Known gaps / explicitly deferred (not solved in this MVP)

- **No automated pass/fail verdict** — `verdict` field exists but is always manual for now.
- **Full-matrix-per-PR does not scale to "test every push."** 24h+ per campaign makes a full 6×2 matrix infeasible at CI push frequency. A smaller per-PR "smoke" subset vs. a periodic full-matrix run is a real distinction this design does not yet make operational — flagged for the next design pass before wiring up real CI triggers.
- **No machine-to-machine auth.** "Add login later" (§4) should end up supporting a service-account/API-token path for CI, not only human SSO — noted so it isn't designed narrowly and then redone.
- **No commit de-duplication.** Submitting the same commit twice just creates two `run_id`s; no "already tested, reuse result" check.
- **No image-mirror storage retention policy.** Mirrored images accumulate per epoch with no cleanup defined yet.
- **Baseline creation is inherently human-paced** (agent-assisted CI archaeology) — this is a boundary, not a bug (see §7); only the PR-test path is meant to become fully unattended.
- **One execution backend.** Only `crusoe_slurm` exists (§5.1). The natural second one is a no-scheduler `local` backend (run `runner.sh` directly on a GPU host, for dev use); another cluster gets its own file. Neither is built.
- **A job that dies between `campaign` and `after` loses the kept tree** (the runner removes its container at exit); re-running `after` then needs the Arbor session restored (`python3 -m arbor restore`). The runner resumes cleanly at every other phase boundary.

## 12. Open parameters (deliberately not decided yet)

- Node count / pool size for the Slurm QOS this system uses.
- Numeric tolerance for the baseline verification gate (proposed ±3%, not finalized).
- Qwen3.8's host configuration (framework/TP/EP) — model entry is a placeholder.
