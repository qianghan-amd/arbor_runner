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
