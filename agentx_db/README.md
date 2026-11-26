# AgentX results database (B300 and MI355X)

Two parts:

1. **`official/`**: AgentX's own results, from the InferenceX "Run Sweep" CI runs (this part).
2. `ours/` (next step): our baselines and our A/B experiments.

## official/

Source: the AgentX results InferenceX itself ingested. On each merge to `main` that changes `perf-changelog.yaml`,
its "Merge Ingest" workflow (`.github/workflows/merge-ingest.yml`) reuses the merged PR's "Run Sweep" run (named by an
authorized `/use <run_id>` comment on the PR) and sends it to the AgentX database (`ingest-agentic-results`). That
source run is what we store, with the merge (merge run id, PR number and title, merge time, the bot's ingest report)
in its `official` block. A PR's other sweeps are not official and are not stored. All models; hardware B300
(`b300-dsxe`) and MI355X (`mi355x-amds`). GB300 (`gb300-nv`) is another machine and is not included.

| File | What it holds |
|---|---|
| `runs/<run_id>.json.gz` | One CI run. The small official artifacts, verbatim: `points` (`results_bmk/agg_bmk.json`: every point's metrics, image with digest, recipe fingerprint, topology, KV offload, power), `evals` (`eval_results_all`: GSM8K), `changelog` (`changelog-metadata`: why the run was made, base and head commits), `run_stats`, `job_events` (per point: the recipe path in InferenceX, topology, image). `sha256` of each (canonical JSON). `artifacts`: the index of every artifact of the run (name, id, size, expiry). `run`: id, URL, date, branch, head commit, event. |
| `points.csv` | One row per point, newest first: the columns below. Rebuilt from `runs/` by `sync` and `index`. `status` is `failed` when no request succeeded. |
| `evals.csv` | Every official GSM8K result (InferenceX evaluates a few points), with `has_bench_point` (false: a configuration with no bench point in that run). |
| `checked.txt` | Source runs already checked that have no B300/MI355X AgentX point; `sync` does not fetch them again. |

Not copied: the large per-point artifacts (`agentic_*` with the server command and aiperf output, `server_logs_*`,
`power_audit_*`, `eval_*`; 5-100 MB each). They are indexed (id, expiry); GitHub deletes them after 90 days. A point's
launch stays reproducible: its recipe path (`job_events`) at the run's InferenceX commit (`run.head_sha`). While the
artifacts live, `scripts/run_from_ci.py` turns a run into a launch record.

`points.csv` columns (newest merge first): merged_at, pr, run_id, date (of the sweep), branch, hw, model_prefix, model, framework, precision, spec_decoding, tp,
ep, pp, dp_attention, dcp_size, num_gpus, conc, kv_offloading, kv_offload_backend, image, recipe_fingerprint,
requests_ok, requests_total, tput_per_gpu (total tokens/s per GPU), output_tput_per_gpu, total_tput_tps,
e2e_norm_intvty_p50/p90 (the AgentX interactivity axis), intvty_p50/p90, ttft_p50_ms, tpot_p50_ms, gsm8k_em_strict,
avg_power_w, joules_per_output_token, power_valid. Metric paths: `scripts/arbor_ab/agentx.py`.

## Use

```bash
python scripts/agentx_db.py sync --pages 1          # add new merges (newest 100 Merge Ingest runs); --refresh to fetch again
python scripts/agentx_db.py query --model dsv41flash --hw mi355x --latest
python scripts/agentx_db.py query --model glm5.3 --conc 16 --csv > glm53_c16.csv
python scripts/agentx_db.py compare --model dsv41flash --metric tput_per_gpu --latest   # B300 vs MI355X per conc
```

`model` is the CI's model prefix (`dsv41flash`, `glm5.2`, `musespark`, `kimik3.1`, ...). `--latest` keeps the newest
point of each configuration (hardware, model, framework, precision, spec decoding, TP, EP, DP attention, KV, conc).
