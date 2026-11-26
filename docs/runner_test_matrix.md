# Runner test matrix

Setups the runner has been run on end to end, and what each run found. A smoke run goes through fetch, prepare,
`arbor check` and a campaign that starts and is stopped by the runner at its budget (`campaign_smoke`, 15 min), or one
measurement (`smoke`). Request files: `runs/requests/sm-*.yaml` (and their `*.record.json`). Arbor `020767ac` unless
noted. Dates are 2026-11.

## Smoke matrix (2026-11-22)

| Id | Site | Model / target | Framework | AgentX CI run (date) | Path tested | Job | Result |
|---|---|---|---|---|---|---|---|
| C1 | crusoe | DSv4.1 Flash C1 | vLLM | 34657748450 (09-11) | oldest CI format (`vllm_command.txt`), InferenceX lib layout | 303721 | pass |
| C2 | crusoe | DSv4.1 Flash C1 | SGLang | 44231353783 (11-18) | first SGLang run of a model: layout of another model, checked on the image | 303722 | pass |
| C3 | crusoe | GLM-5.2 C1 TP8 | SGLang | 43487923974 (11-13) | | 303723 | pass |
| C4 | crusoe | MiniMax M3.1 Flash C1 | vLLM | 39929767608 (10-23) | | 303724 | pass |
| K1c | crusoe | Kimi K3 C1 TP8 | vLLM | 40037005947 (10-23) | 1.56 TB weights from the shared hub; image tag deleted upstream, pulled by digest; vLLM draft model pinned | 303803 | pass |
| K2b | crusoe | Kimi K3.1 Pro C1 TP8 | vLLM | 41954410041 (11-05) | 1.56 TB download (shared hub incomplete); layout corrected (vLLM in /opt/runtime-venv) | 303804 | pass |
| K3 | crusoe | Muse Spark C1 | SGLang | 38877885539 (10-16) | SGLang draft model | 303761 | pass |
| K4 | crusoe | DSv4.1 C1 + MiniMax C1 | vLLM | stored baselines | two cells in one request, `run_pr_test.py` | 303762/3 | pass |
| M1 | meta64 | DSv4.1 Flash C32 | vLLM | 44699914609 (11-20) | throughput target | 147550 | pass |
| M2 | meta64 | DSv4.1 Flash C1 | SGLang | 43344632854 (11-13) | | 147551 | pass |
| M3 | meta64 | GLM-5.2 C16 TP4 | SGLang | 43487923974 (11-13) | | 147552 | pass |
| M4 | meta64 | MiniMax M3.1 Flash C1 | vLLM | 43171131215 (11-12) | | 147554 | pass |
| M5 | meta64 | Kimi K3 C1 | vLLM | 40037005947 (10-23) | 1.57 TB > the node's cache cap: refused by the disk guard before any download (intended) | 147564 | pass |
| M6 | meta64 | DSv4.1 Flash C1 | vLLM | 34657748450 (09-11) | oldest CI format on meta64 (uv Python 3.12) | 147555 | pass |
| M7 | meta64 | Muse Spark C16 | SGLang | 38877885539 (10-16) | | 147557 | pass |
| G1 | meta64 | DSv4.1 Flash C1 | vLLM | stored baseline | `smoke` with `gsm8k: true` (GSM8K 200 Q: 0.965 / 0.940) | 147578 | pass |
| G2b | meta64 | MiniMax M3.1 Flash C1 | vLLM | 43171131215 (11-12) | `--partition interactive` | 147618 | pass |
| G3 | meta64 | DSv4.1 Flash C1 | vLLM | 44689368479 (11-20) | default `campaign` mode, `--follow` (report printed) | 147576 | pass |

Refused up front, as intended (`run_from_ci --dry-run`): NVIDIA runs (GB200 44906527874), gfx942 runs (MI325X
44880730346), the `atom` framework (43387638995), a run that reused older results (44864673715, no artifacts).

## Earlier end-to-end runs

| What | Site | Job | Result |
|---|---|---|---|
| P6: before / campaign / after, DSv4.1 C1, Arbor f7abce97 | crusoe | 302311 | pass: e2e_norm P90 13.875 -> 52.437 |
| First use case, CI + Arbor commit, 8 h campaign (one process) | crusoe | 302501 | pass: Arbor done, 5 iterations, 1 KEEP |
| `full` via run_from_ci, 30-min campaign | meta64 | 146989 | pass (nothing kept: an A/A reading) |
| `make_baseline measure` + `finalize` | crusoe | 303240 | pass (gate refused +5.9 %, as designed) |
| SIGKILL / cancel while the server starts: container gone, GPUs free | both | 303340, 147007, 147008 | pass |

## Restart, preemption, launcher, full mode (2026-11-23/24)

| What | Site | Job | Result |
|---|---|---|---|
| Restart after a requeue (requeued by hand in smoke) | meta64 | 149586 | pass: attempt 1 archived, started over on another node |
| Attempt limit (3 requeues) | meta64 | 149688 | pass: "attempt 4 of at most 3 ... giving up", FAILED, no endless requeue |
| SIGKILL on Crusoe: which cleanup ran | crusoe | 308348 + probe 308361 | pass: the dead-man switch (no trap line in the node-local log) |
| `poll_status --extend` | both | 147758, 308285 | Spur: owner may raise the limit; meta64 Slurm: refused (operator only) |
| `full` via run_from_ci, SGLang GLM-5.2 C16 | meta64 | 148019 | pass: A/A +0.2 % |
| `full` via run_from_ci, MiniMax C1 | crusoe | 309597 | pass after 2 automatic resubmits (JobHoldMaxRequeue): A/A -0.55 % |
| #74: Arbor 7c211696 under its launcher, DSv4.1 C1 | meta64, crusoe | 149931, 308827 | pass: root-baseline bench passed, exit deadline rc 0 (3 Crusoe tries before: burst preemption) |
| Baseline store path (`make_baseline measure` + `finalize`) | crusoe | 308511 | pass: gate -0.24 %, stored, then baselines/ restored |
| duckdb/pyarrow before the campaign | meta64 | 150380 | pass: duckdb 2.0.1, pyarrow 23.0.1 |

## User objective, start methods, newest CI (2026-11-24/25, Arbor 7c211696)

| Test | Site | Job | Result |
|---|---|---|---|
| `--objective "P50 interactivity"` (DSv4.1 C1, a P90 cell) | meta64 | 151742 | pass: objective.json e2e_norm_intvty_p50, asked "P50 interactivity" |
| `--objective "total throughput" --goal ...` at conc 16 (GLM-5.2 c16_tput) | meta64 | 151743 | pass: total_throughput; the goal in session.json |
| structured `objective:` P50 +2 % with two guards (MiniMax C1) | meta64 | 151749 | pass: threshold 1.02, guards total_token_throughput 5 %, request_error_rate 10 % |
| `--objective "banana speed"` | meta64 | 151747 | refused at `arbor check` with Arbor's message (intended) |
| stored baseline GLM-5.2 c1 (SGLang) | meta64 | 151756 | pass: AgentX ok 36/47 |
| stored baseline Muse Spark c1 (SGLang, draft) | meta64 | 151757 | pass: AgentX ok 80/91 |
| stored baseline DSv4.1 c32 | meta64 | 151758 | failed: no bench_client_commit -> now refused before submit |
| stored baseline Kimi K3 c1 (1.56 TB, shared hub) | crusoe | 311115 | pass after a preemption resubmit: AgentX ok 59/71 |
| hand-edited SGLang launch record (GLM-5.2 C16, env added) | meta64 | 151751 | pass: the env in launch.json and serve.sh; AgentX ok |
| newest CI 45233004275 (MiniMax 11-23) | meta64 | 151754 | pass: arbor run rc 0 |

Stored baselines usable as a start (`check_runnable`): 5 of 12 (dsv41flash c1, glm52 c1, kimik3 c1, minimax c1,
muse c1). The others lack a launch block, the corpus pins, bench env or the bench client commit: start them from a
CI run.

## Startup methods and mode bench (2026-11-25/26, Arbor 7c211696)

| Test | How it was started | Site | Job | Result |
|---|---|---|---|---|
| B1 | `run_from_ci --mode bench --conc 1 2 4 --duration 300`, a job URL | crusoe | 312862 | pass: one server, bench/c1 c2 c4, summary.json, a report row each |
| B2 | `run_pr_test.py --request` (submit, wait, report), stored DSv4.1 c1, smoke | crusoe | 312869 | pass; the report said "incomplete" -> a smoke row (fixed) |
| B3 | `run_from_ci` with no `--site` (default crusoe) and `--follow` | crusoe | 312871 | pass |
| B4 | one request, two cells: an edited record + a stored baseline (GLM-5.2) | crusoe | 312916, 312917 | pass: the edit in launch.json |
| B5 | `cancel_run` on a running smoke | crusoe | 312964, 313020, 313303 | Spur's scancel is a SIGKILL (no trap) -> TERM to the runner: 313303 stopped its server and removed its container |
| T1 | `run_from_ci --mode bench`, no `--conc` (the CI's 32), `--follow` | crusoe | 313154 | pass |
| T2 | request `record:` (launch JSON), mode bench, `submit_ab_test.py` | crusoe | 313396 | pass (after a preemption and a dirty node, v3-172) |
| T3 | stored baseline, mode bench, `run_pr_test.py --request` | crusoe | 313506 | pass |
| T4 | meta64, mode bench, `--gsm8k` | meta64 | 153110 | refused in prepare (infx layout has no benchmark_lib.sh) -> now refused before submit |
| - | dry run; `--conc` without mode bench; `--conc 0` | - | - | pass; both refused with a clear message |

Mode bench caveat: on one server the second concurrency read much lower than the first (B1 c2 5041 < c1 11136
tok/s/GPU; T3 c2 1990 < c1 5779) at 300 s each: likely state carried between runs and runs too short. Use 1800 s or
more; a restart per concurrency is the next change.

## Runner bugs these runs found (all fixed)

| Found by | Bug | Fix |
|---|---|---|
| campaign smoke 300088 | Arbor's time budget does not bound a running iteration | runner stops Arbor at the budget (`timeout`) |
| P6 302311 | after found Arbor's last server still up | container restarted after the campaign |
| 300404 (preempted) | a KILL skips the cleanup trap | container dead-man switch |
| 302466 | Arbor's launcher mode killed the job | `ARBOR_STANDALONE=1` (Arbor issue #74 on the board) |
| meta64 146743 | /scratch not writable; job /tmp private | node cache in /var/tmp |
| 302455 | InferenceX infx layout | layout detection, infx client venv |
| dry runs | CI hardware / framework not checked | gfx950 guard, clean refusals |
| C2 dry run | no SGLang template for DSv4.1 | layout from another model, checked on the image |
| M5 | model loaded by repo id; vLLM drafts unpinned | revision from HF history; vLLM draft pinned |
| K1 | image tag deleted upstream | digest from a stored record; registry check before submit |
| K2 | CI prefix differs from the record's | several prefixes per model |
| G2 | disk eviction freed nothing (shared `<hub>/blobs`) | blob-level eviction, orphan cleanup |
| K2 | shared hub snapshot without shards | every shard checked before use |
| G3 | piped output out of order | flush before the report |
| meta64 147754 | orphan scan saw only the sharded blob store's prefix dirs | recursive walk, sidecars removed |
| meta64 147755 | CI drafts kept the template's KV backend (none + hicache) | KV + PP/DCP/PCP from the CI point; refused at submit |
| meta64 147783 | a requeued job resumed on another node without its container | start over (attempts/<n>), max 3 |
| meta64 149589 | SGLang (hicache) held 2 % VRAM > 120 s after stop | 300 s for SGLang; holders logged |
| CI 44550646425 | a glm5.2 cell served GLM-5.3 weights | refused unless --allow-other-model; KV backend as an object |
| Crusoe 308407 | make_baseline measure could not skip a refused node | --exclude-nodes |
| Crusoe burst | every run > 20 min preempted | --requeue + poll_status --resubmit-preempted |
| Crusoe 308822 | Spur held a requeued job forever (JobHoldMaxRequeue) | reported as over; resubmit cancels it first |
| meta64 151758 | stored baseline without bench_client_commit passed the check | check_runnable requires it |
| B2 312869 | a smoke run reported "incomplete" | a smoke row in generate_report |
| B5 312964, 313020 | Spur's scancel kills with SIGKILL: no cleanup trap | cancel TERMs the runner (srun --overlap pkill), scancel after 90 s; state "cancelled" |
| Crusoe v3-179 | python3 without ensurepip; a half-built venv reused; no docker | uv fallback, .ok venv markers, docker checked first, bad_nodes |
| Crusoe v3-172 | another user's container holds the GPUs | bad_nodes |
| meta64 153110 | --gsm8k with the infx layout failed after the fetch | refused before submit |
