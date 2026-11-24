# Shared by every script of <<CODENAME>> (<<one-line description: model, framework, Arbor line, conc, target>>).
# Sourced, not run. All paths are under /shared_nfs or the node's NVMe (never /home on the node — its NFS mount has
# wedged on this cluster before).
#
# Adapted from experiments/musespark-c1-p90-sa/env.sh (SGLang + spec decode, node-local NVMe download) — see
# experiments/dsv41f-c32-tput/env.sh for the leaner vLLM + shared-/shared_nfs/huggingface/hub-cache variant if that
# style applies instead (no dl.sh/HUB_NL section needed there; set MODEL_C to the shared cache snapshot path instead).

TOP=/shared_nfs/qiangh/mywork/<<WORKSPACE NAME, e.g. arbor_dsv41flash>>
EXP=$TOP/experiments/<<CODENAME>>
LOGS=$EXP/logs
ARBOR=$EXP/src/arbor                      # Arbor at the pinned commit (archive) + pack; mounted at /workspace/Arbor
ARBOR_SHA=<<full Arbor commit sha>>
INFX=$TOP/codebases/InferenceX            # bench-client commit: see SKILL.md Step 1's compatibility pitfall
INFX_SHA=<<full InferenceX bench-client commit sha>>
VENV=$EXP/.venv                           # the driver's python
PY=$VENV/bin/python
CONFIG_REL=configs/<<board, e.g. mi355x>>/<<CODENAME>>.json
CONFIG=$ARBOR/$CONFIG_REL
PACK=<<pack name, e.g. agentx-musespark-c1sa>>
SESSIONS=$EXP/sessions                    # mounted at /workspace/sessions
SESSION=$SESSIONS/full
MMAP_CACHE=$EXP/aiperf_mmap_cache
AGENT_HOME=$EXP/agent_home                # HOME for the driver and its agents
GATEWAY_ENV=<<path to an existing campaign's agent_home/.arbor/gateway.env to reuse the same LLM gateway creds>>

# The CI image (InferenceX run <<RUN_ID>>), pulled by digest.
IMAGE_TAG=<<e.g. lmsysorg/sglang-rocm:... or vllm/vllm-openai-rocm:...>>
IMAGE_DIGEST=sha256:<<digest, confirmed per SKILL.md Step 1 — not just the tag>>
CONTAINER=<<short unique container name, e.g. qh_<codename-abbrev>>>
PORT=<<a free port not used by another campaign on the same node>>

# HF cache on the node's NVMe, mounted at /hf-hub: the model (and draft, if speculative decode) and the trace corpus
# at their pins (dl.sh), each with refs/main = its pin. The launch names both by repo id, as CI does, and runs offline.
# Omit this whole block and set MODEL_C directly if following the dsv41f-c32-tput shared-cache style instead.
HUB_NL=/mnt/m2m_nobackup/<<node-local dir name, unique per campaign>>/hf-hub
MODEL_ID=<<HF repo id>>
MODEL_REV=<<full commit sha, from the recipe/config, not a README summary>>
DRAFT_ID=<<HF repo id>>                   # omit if no speculative decode
DRAFT_REV=<<full commit sha>>             # omit if no speculative decode
WORKSPACE_ROOT=<<the framework's editable source tree inside the container, e.g. /sgl-workspace/sglang or vLLM's install path — confirmed by prepare.sh step 7, not assumed>>

mkdir -p "$LOGS"
log() { echo "[$(basename "$0" .sh)] $(date -u +%H:%M:%S) $*"; }
die() { log "FATAL: $*"; exit 1; }
