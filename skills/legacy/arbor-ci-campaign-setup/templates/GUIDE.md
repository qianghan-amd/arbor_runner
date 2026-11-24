# Campaign: <<CODENAME>> — intent and constraints

Written <<DATE>> before launch (CLAUDE.md, autonomous mode). Source of truth: <<where the goal/pins came from — a README item, a user conversation, etc.>>

<!-- Adapted from experiments/musespark-c1-p90-sa/GUIDE.md -->

## Goal

Run a 24 h **Arbor standalone** campaign on <<MODEL>> with <<FRAMEWORK: SGLang or vLLM>>, MI355X, AgentX trace replay at **concurrency <<CONC>>**, with **<<METRIC, e.g. total_throughput_per_gpu / p90 interactivity>>** as the objective<<, without losing <<SECONDARY METRIC>> beyond <<TOLERANCE>>, if a secondary guard applies>>. Before and after the campaign, run the full AgentX benchmark at <<CONC>> and GSM8K.

## Pins

| What | Pin |
|---|---|
| Arbor | <<repo>> branch/tag `<<BRANCH>>` = `<<ARBOR_SHA>>` (<<date>>), stock code + our pack `<<PACK>>` |
| CI run | InferenceX run <<RUN_ID>> (<<date>>), commit `<<CI_COMMIT>>` (<<how it's reachable, e.g. "on refs/pull/NNNN" if not a tip — see SKILL.md Step 1 pitfall>>), key `<<CONFIG_KEY>>` |
| Recipe | `<<path to agentic.yaml or the parameterized .sh>>`<<, point \`<<override_NAME>>\` (<<TP/EP/CONC summary>>) — OR — CONC set externally by CI workflow dispatch; see perf-changelog.yaml entry for this config key>> |
| Image | `<<IMAGE_TAG>>` @ `<<IMAGE_DIGEST>>` |
| Model | `<<MODEL_ID>>` @ `<<MODEL_REV>>` |
| Draft | `<<DRAFT_ID>>` @ `<<DRAFT_REV>>` (<<spec-decode scheme, draft-token count>>) — omit this row if no speculative decode |
| Bench client | InferenceX `<<BENCH_CLIENT_SHA>>` (`benchmarks/benchmark_lib.sh`, aiperf) — <<note if this must differ from the CI commit per SKILL.md Step 1's compatibility pitfall>>, corpus `<<CORPUS_REPO>>` @ `<<CORPUS_REV>>`, DURATION <<N>> s |

## Decisions (user, <<date>>)

- **Node:** <<hold job id, node name, submitted wall-time, any raise>>.
- **CI point:** `<<override_NAME or equivalent>>` (<<TP/EP>>); GPUs <<which>> free for Arbor's probes and profiling.
- **Bench client:** <<which InferenceX commit and why, if it differs from the CI pin>>.
- **Before/after:** one full AgentX (<<CONC>>, <<DURATION>> s) and one GSM8K (<<N>> questions) each, on a fresh server.
- **Objective:** `<<metric_name>>`, KEEP at +<<X>>%<<, secondary \`<<secondary_name>>\` held within <<Y>>%>>.

## Known weaknesses of this Arbor (red flags reported to the user)

<<e.g. no cold prefix cache before a bench; the campaign's rolling internal baseline can diverge from the full before/after re-measure — see SKILL.md Step 7; anything else surfaced during Step 1 CI archaeology>>

## Layout

- `experiments/<<CODENAME>>/`: this guide, `env.sh`, `dl.sh`, `prepare.sh`, `ab.sh` (smoke/before/after), `campaign.sh`, `stage/` (Arbor archive, pack, bench, config), `runs/<tag>/`, `sessions/full/`, `logs/`, `ctrl/` (node agent).
- Node NVMe `<<HUB_NL path>>`: model, draft, corpus (refs/main = pins), mounted at `/hf-hub` in container `<<CONTAINER>>`.

## Constraints

Never touch other users' containers or processes (list them, never kill). Smoke test first. Do not loosen gates, tolerances or CI flags without the user; record any CI deviation. No tokens (GitHub, board, HF) inside the container. Log every action in `reports/<date>_worklog.md`.
