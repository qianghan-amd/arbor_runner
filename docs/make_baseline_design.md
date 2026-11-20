# MVP design: build a new baseline from an AgentX CI run (`make_baseline.py`)

**Status:** draft for review, 2026-11-19. Nothing below is built. **Scope:** P7 of `experiments/arbor-ab-system/INTENT.md`;
design.md section 3.2 (`source_type: agentx_ci`) and section 3.4 (two-layer gate).

## 1. Goal

A user gives a `(model, target)` cell and an InferenceX CI run ID. The tool turns that CI run into a baseline record: the
same schema, the same pins and the same `launch` block as the 12 ported baselines. It then measures the pristine launch
once on Crusoe and stores the record only if the measurement agrees with CI's own number.

Today this needs a person and the `arbor-ci-campaign-setup` skill (about half a day of archaeology per cell). The MVP
automates the archaeology that the CI's own artifacts answer exactly. It asks a person only where they do not.

## 2. What the CI run already tells us (verified 2026-11-19, InferenceX run 34657748450, DSv4.1 Flash)

| Need | Source in the CI run | How exact |
|---|---|---|
| CI commit, date | Actions API `runs/{id}`: `head_sha`, `created_at` | exact |
| Image (+ digest) | `bmk_*` artifact JSON: `image` (some runs add `@sha256:…`); otherwise registry lookup by tag at that date | exact / near |
| Framework, TP, EP, DCP, GPUs, conc, precision, spec decoding, KV offload | `bmk_*` JSON: `framework, tp, ep, dcp_size, num_gpus, conc, precision, spec_decoding, kv_offloading` | exact (DCP: cross-check the flags) |
| **Server flags** | `agentic_*` artifact: **`vllm_command.txt`**, the exact command CI ran | exact |
| Model ID + **revision** | `vllm_command.txt` (repo ID) + `server.log`: `models--org--name/snapshots/<rev>` | exact |
| Bench settings | `agentic_*`: `benchmark_command.txt` (aiperf scenario, concurrency, duration, seed) | exact |
| `MODEL_PREFIX` (bench env) | `bmk_*` JSON: `infmax_model_prefix` | exact |
| Corpus name + size | `bmk_*` JSON: `dataset.hf_dataset_name`, `num_dataset_entries` (393) | exact |
| **Corpus revision** | not in the artifacts. CI loads HF `main`, so take the latest dataset commit before `created_at` (HF API); check that the entry count matches | derived, checked |
| CI's own metrics (layer-2 gate) | `bmk_*` JSON `request_metrics.*` (same structure as our agg) | exact |
| **Server env vars** | **not recorded** (`server.log` shows a few only) → the recipe at `head_sha`: `.sh` (env exports) or `agentic.yaml` (`override_*` block) | parsed; **needs review** |
| `workspace_root`, `editable_packages` | not in CI; depends on the image | found by the runner's `prepare` (it already checks the install path), with defaults per framework |
| Draft model (spec decode) | `vllm_command.txt` / `sglang` command (`--speculative-*`) + `server.log` snapshot path | exact when present |

**Not yet checked:** which command file an SGLang CI run writes (vLLM writes `vllm_command.txt`). Artifacts expire
after 90 days; an expired run cannot be used (the tool says so and stops).

## 3. Flow

```
make_baseline.py draft  --model M --target T --ci-run ID
   1. GitHub relay: run info, artifacts; pick the bmk_* artifact for (tp, conc) as verify_baseline.py does
   2. download bmk_*, agentic_* (command files + server.log only), the recipe file at head_sha
   3. build the record: host, fingerprint, launch, target (from config/matrix.yaml), metrics empty
   4. write runs/baselines/<model>__<target>__<run>/draft.json + review.md (each field: value, source, exact or derived)
   -> STOP for a person: read review.md; fix or confirm the env block and any field marked "derived"/"ambiguous"

make_baseline.py measure --draft <dir>          (after review)
   5. submit a `baseline` mode job (new runner mode = fetch, prepare, before only; 3600 s AgentX + full GSM8K)
      through the same backend; poll as usual

make_baseline.py finalize --draft <dir>         (after the job)
   6. metrics <- the job's before/agg.json + gsm8k; layer 1 = structural vs the bmk JSON (as now);
      layer 2 = numeric vs CI's own number, tolerance from matrix.yaml (+/-3 %)
   7. pass -> store (epoch from the fingerprint; supersedes the old epoch of that cell, never deletes)
      fail / client mismatch / expired -> do not store as trusted; write the result into review.md for a person
```

`draft`, `measure` and `finalize` are stateless steps over files in one directory, like the rest of the system. A run
can stop between them.

## 4. Decisions in this design (for review)

1. **Flags come from `vllm_command.txt`, not from re-rendering the recipe.** It is what CI actually ran. Recipe parsing
   is used only for the env, and as an advisory cross-check of the flags (the "recipe flag diff" listed as a P2 gap).
2. **Env needs a person, once per baseline.** CI does not record it, and the two recipe formats are fragile to parse.
   The tool pre-fills it from the recipe and marks it "review". This keeps design.md section 7's line: baseline
   creation is human-paced; PR testing is not.
3. **Corpus revision = latest dataset commit before the CI run**, accepted only if the entry count matches CI's.
4. **Bench client = the CI commit.** If that InferenceX commit cannot run the Arbor pack's bench (no
   `benchmark_lib.sh` / `infx`; the skill's Step 1 pitfall), the tool says so and asks for a client commit; the gate
   then reports `client_mismatch` (not `pass`), as it does today for four cells.
5. **No new schema.** The output is a normal v2 record with `launch`. `provenance` gets one entry per field from §2.
6. **One new runner mode, `baseline`** (fetch, prepare, before). Everything else is reused: backend, runner, gate,
   health check.

## 5. Out of scope for the MVP

- The `user_custom` path (manual config without CI). It reuses steps 5–7 with a hand-written draft. It is cheap to add
  next, but it is not in this MVP.
- Image mirroring (design.md 3.5), multi-node recipes, SGLang command-file discovery beyond one verified example,
  automatic env extraction without review, refreshing an existing epoch.

## 6. Test plan

- Offline (no GPU): `draft` against the CI run behind each already-ported, still-live baseline (DSv4.1 C1/C32,
  Kimi K3 C1, Muse Spark C1, MiniMax C1). The generated record must equal the stored one field by field (pins, flags,
  model/corpus revision, topology); every difference is either a bug or a finding.
- On the cluster: one `measure` + `finalize` on DSv4.1 C1 (the cell P6 already reproduced: +2.8 % P90, −0.04 %
  throughput), about 2.5 node-hours.

## 7. Effort and order

About 1.5 days of work: draft ~0.75 day (most of it in the recipe env parser and review.md), runner `baseline` mode
~0.25 day, finalize ~0.25 day, tests ~0.25 day. The order is draft, then its offline test against the stored baselines,
then measure/finalize.

## 8. Open questions for the reviewer

1. Is "env reviewed by a person, flags taken from CI's command file" the right split, or should the MVP try harder to
   avoid the review step?
2. Tolerance for the layer-2 gate: keep ±3 % on the primary metric (C1 P90 noise was about 3 % in P6's before run)?
3. When the gate fails, should the tool keep the draft as an `unverified` record (usable, marked), or keep nothing?
4. Should `user_custom` come right after this MVP, or before it?
