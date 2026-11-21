import pytest

from arbor_ab import ci_baseline

# Shape of run 34657748450's agentic_*/vllm_command.txt (shell-escaped JSON, trimmed)
CMD = ('vllm serve deepseek-ai/DeepSeek-V4.1-Flash --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --host 0.0.0.0 '
       '--port 8891 --tensor-parallel-size 4 --language-model-only --moe-backend aiter_triton_mxfp4_bf16 '
       '--speculative-config \\{\\"method\\":\\"dspark\\",\\"num_speculative_tokens\\":5\\} --max-num-seqs 2')


def test_parse_vllm_command_drops_what_arbor_adds():
    c = ci_baseline.parse_vllm_command(CMD)
    assert c["model"] == c["served_name"] == "deepseek-ai/DeepSeek-V4.1-Flash" and c["tp"] == 4
    assert c["flags"] == ("--language-model-only --moe-backend aiter_triton_mxfp4_bf16 "
                          "--speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":5}' --max-num-seqs 2")
    assert "--host" not in c["flags"] and "--port" not in c["flags"] and "--tensor-parallel-size" not in c["flags"]
    with pytest.raises(ValueError):
        ci_baseline.parse_vllm_command("python3 -m sglang.launch_server --model-path x")


# Shape of run 44689368479's server_logs_*/srt-single-node-logs.tar.gz logs/config.yaml (srtctl, trimmed)
SRT = """schema: 2
model:
  path: hf:deepseek-ai/DeepSeek-V4.1-Flash
  container:
    vllm/vllm-openai-rocm:nightly-9c1d45c@sha256:694108707f33
engine:
  type: vllm
roles:
  agg:
    args:
      served-model-name: deepseek-ai/DeepSeek-V4.1-Flash
      language-model-only: true
      moe-backend: aiter
      gpu-memory-utilization: 0.9
      attention-config: '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
      speculative-config: '{"method": "dspark", "num_speculative_tokens": 5}'
      tensor-parallel-size: 4
      max-num-seqs: 1
      enforce-eager: false
    env:
      VLLM_ROCM_USE_AITER: '1'
      OMP_NUM_THREADS: '1'
"""


def test_parse_srt_recipe():
    c, env, image = ci_baseline.parse_srt_recipe(SRT)
    assert c["model"] == c["served_name"] == "deepseek-ai/DeepSeek-V4.1-Flash" and c["tp"] == 4
    assert c["flags"] == ("--language-model-only --moe-backend aiter --gpu-memory-utilization 0.9 "
                          "--attention-config '{\"indexer_kv_dtype\":\"mxfp4\",\"indexer_sparse_logits\":true}' "
                          "--speculative-config '{\"method\": \"dspark\", \"num_speculative_tokens\": 5}' --max-num-seqs 1")
    assert env == {"VLLM_ROCM_USE_AITER": "1", "OMP_NUM_THREADS": "1"}
    assert image == "vllm/vllm-openai-rocm:nightly-9c1d45c@sha256:694108707f33"
    with pytest.raises(ValueError, match="only one aggregated server"):
        ci_baseline.parse_srt_recipe(SRT.replace("  agg:", "  prefill:"))


def test_model_rev_from_log():
    rev = "dba1be0a40aa45a94ad051997016db3960a90277"
    log = f"loading /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/{rev}/config.json\n" * 3
    assert ci_baseline.model_rev_from_log(log, "deepseek-ai/DeepSeek-V4.1-Flash") == rev
    with pytest.raises(ValueError):
        ci_baseline.model_rev_from_log("no snapshot here", "deepseek-ai/DeepSeek-V4.1-Flash")


MATRIX = {"models": {"dsv41flash": {"targets": {
    "c1_p90": {"kind": "p90", "conc": 1, "expected_host": {"tp": 4}},
    "c32_tput": {"kind": "tput", "conc": 32, "expected_host": {"tp": 4}}}}}}


def test_ci_points_and_choose_cell():
    arts = [{"name": "bmk_agentic_dsv41flash_tp2_conc1_kvnone_spec-mtp_fp4_vllm_x", "expired": False},
            {"name": "bmk_agentic_dsv41flash_tp4_conc1_kvnone_spec-mtp_fp4_vllm_x", "expired": False},
            {"name": "bmk_agentic_dsv41flash_tp4_conc256_kvdram_x", "expired": False},
            {"name": "bmk_agentic_dsv41flash_tp8_conc32_x", "expired": True},
            {"name": "agentic_dsv41flash_tp4_conc1_x", "expired": False}]
    pts = ci_baseline.ci_points(arts)
    assert pts == [("dsv41flash", 2, 1), ("dsv41flash", 4, 1), ("dsv41flash", 4, 256)]
    # tp2 and tp4 both measured conc 1: the matrix's expected tp (4) decides
    assert ci_baseline.choose_cell(pts, MATRIX) == ("dsv41flash", "c1_p90", 4)
    assert ci_baseline.choose_cell(pts, MATRIX, tp=2) == ("dsv41flash", "c1_p90", 2)
    with pytest.raises(ValueError, match="no matrix cell"):          # conc 32 only in an expired artifact
        ci_baseline.choose_cell(pts, MATRIX, target="c32_tput")
    with pytest.raises(ValueError, match="no matrix cell"):
        ci_baseline.choose_cell([("glm99", 8, 1)], MATRIX)


def test_corpus_rev_at_ci_time():
    commits = [{"id": "new", "date": "2026-10-01T00:00:00.000Z"}, {"id": "23f152f6", "date": "2026-06-21T10:00:00.000Z"},
               {"id": "first", "date": "2026-06-21T09:00:00.000Z"}]
    assert ci_baseline.corpus_rev_at(commits, "2026-09-11T23:21:29Z") == "23f152f6"   # the run predates "new"
    assert ci_baseline.corpus_rev_at(commits, "2026-11-01T00:00:00Z") == "new"
    with pytest.raises(ValueError):
        ci_baseline.corpus_rev_at(commits, "2026-01-01T00:00:00Z")
