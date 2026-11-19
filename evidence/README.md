# Evidence

Ground truth behind `baselines/`. Every baseline record's `provenance` points into this directory, so each number can be
checked from the repo alone.

| Run | What | Produced by |
|---|---|---|
| `2026-11-19_0432_port/` | `raw/<model>__<target>.json`: read-only collection of each campaign's files on Crusoe (`/shared_nfs`), with path, sha256, size and mtime of every source file. `cells.yaml` / `summary_only_cells.yaml`: the port manifests (source paths, CI pins, data-quality notes with citations). `port_report.md`: the port's output table. | `collect.py` + `port.py` (harness `experiments/arbor-ab-system/2026-11-19_0432_port/`) |
| `2026-11-19_0640_verify/` | `ci_artifacts/<artifact_id>.json`: the InferenceX CI `bmk_*` artifacts the gate compared against (GitHub artifacts expire after 90 days; these copies stay). `gate_result.md`, `health_result.md`: gate and health tables at that time. | `scripts/verify_baseline.py --evidence-dir`, `scripts/check_baseline_health.py` |

## Check a number

1. In the record, read `provenance._raw_bundle` (path + sha256) and the field's provenance entry (source file + key).
2. `sha256sum evidence/2026-11-19_0432_port/raw/<cell>.json` must equal the recorded sha256.
3. In that raw file, the source file entry carries its own sha256 on the cluster; the value sits under `json` at the cited key.

Cells ported from cited text (`summary_only_cells.yaml`: Kimi K3 C32, Kimi K3.1 Pro C32, MiniMax C32) cite their source
lines (campaign reports in a local backup of conductor_355_2, Agent Board `boards/ArborDevelopment#37` comments) instead
of a raw bundle; their records carry `data_quality_warning` entries that say so.

No credentials are stored here: the collector drops credential-like keys, and the bundle was scanned for token patterns.
