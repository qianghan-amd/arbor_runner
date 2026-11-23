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


# Shape of SGLang run 38877885539's recipe (Muse Spark, trimmed): `engine` is a plain string, TP stays a flag, a draft
SRT_SGLANG = """model:
  path: hf:amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4
engine: sglang
roles:
  agg:
    args:
      served-model-name: amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4
      attention-backend: aiter
      speculative-draft-model-path: RadixArk/Muse-Spark-Open-DFlash2
      enable-metrics: true
      tensor-parallel-size: 4
      expert-parallel-size: 4
    env:
      SGLANG_SIMULATE_ACC_LEN: '4.05'
"""


def test_parse_srt_recipe_sglang():
    c, env, _ = ci_baseline.parse_srt_recipe(SRT_SGLANG)
    assert c["framework"] == "sglang" and c["tp"] == 4 and c["draft"] == "RadixArk/Muse-Spark-Open-DFlash2"
    assert c["flags"] == ("--attention-backend aiter --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 "
                          "--enable-metrics --tensor-parallel-size 4 --expert-parallel-size 4")   # TP kept for SGLang
    assert env == {"SGLANG_SIMULATE_ACC_LEN": "4.05"}
    with pytest.raises(ValueError, match="only"):
        ci_baseline.parse_srt_recipe(SRT_SGLANG.replace("engine: sglang", "engine: trtllm"))


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


def test_choose_cell_by_infmax_prefix():
    m = {"models": {"muse_spark": {"infmax_prefix": "musespark", "targets": {"c1_p90": {"conc": 1, "expected_host": {"tp": 4}}}},
                    "glm52": {"infmax_prefix": "glm5.2", "targets": {"c1_p90": {"conc": 1, "expected_host": {"tp": 4}}}}}}
    pts = ci_baseline.ci_points([{"name": "bmk_agentic_musespark_tp4_conc1_x"}, {"name": "bmk_agentic_musespark_tp8_conc1_x"}])
    assert ci_baseline.choose_cell(pts, m) == ("muse_spark", "c1_p90", 4)
    assert ci_baseline.ci_points([{"name": "bmk_agentic_glm5.2_tp4_conc1_x"}]) == [("glm5.2", 4, 1)]
    assert ci_baseline.choose_cell([("glm5.2", 4, 1)], m) == ("glm52", "c1_p90", 4)


def test_corpus_rev_at_ci_time():
    commits = [{"id": "new", "date": "2026-10-01T00:00:00.000Z"}, {"id": "23f152f6", "date": "2026-06-21T10:00:00.000Z"},
               {"id": "first", "date": "2026-06-21T09:00:00.000Z"}]
    assert ci_baseline.corpus_rev_at(commits, "2026-09-11T23:21:29Z") == "23f152f6"   # the run predates "new"
    assert ci_baseline.corpus_rev_at(commits, "2026-11-01T00:00:00Z") == "new"
    with pytest.raises(ValueError):
        ci_baseline.corpus_rev_at(commits, "2026-01-01T00:00:00Z")


def test_parse_ci_run_accepts_id_and_urls():
    for arg in ("44689368479", " 44689368479 ", "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/44689368479",
                "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/44689368479/job/987",
                "https://github.com/semianalysisai/inferencex/actions/runs/44689368479/attempts/2"):
        assert ci_baseline.parse_ci_run(arg) == "44689368479"
    with pytest.raises(ValueError, match="not of SemiAnalysisAI/InferenceX"):
        ci_baseline.parse_ci_run("https://github.com/AMD-AGI/Arbor/actions/runs/1")
    with pytest.raises(ValueError, match="run id or its URL"):
        ci_baseline.parse_ci_run("https://github.com/SemiAnalysisAI/InferenceX/pull/5151")


def test_ci_arch():
    assert ci_baseline.ci_arch("cluster:mi355x-amds") == ci_baseline.ci_arch("mi350x") == "gfx950"
    assert ci_baseline.ci_arch("cluster:mi325x-amd") == "gfx942"
    assert ci_baseline.ci_arch("cluster:gb200-nv") == "gb200" and ci_baseline.ci_arch(None) == "unknown"


def test_vllm_draft_from_speculative_config():
    assert ci_baseline.vllm_draft("--x 1 --speculative-config '{\"method\": \"dspark\", \"model\": \"Inferact/Kimi-K3-DSpark\"}'") \
        == "Inferact/Kimi-K3-DSpark"
    assert ci_baseline.vllm_draft("--speculative-config '{\"method\": \"dspark\", \"num_speculative_tokens\": 5}'") is None
    assert ci_baseline.vllm_draft("--speculative-config '{\"model\": \"/local/draft\"}'") is None
    assert ci_baseline.vllm_draft("--max-num-seqs 2") is None


def test_point_kv_and_parallel_from_ci_not_template():
    # meta64 147755: CI point kv none, template (stored GLM-5.2 baseline) dram + hicache -> AgentX aggregation refused
    tpl = {"KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "hicache", "KV_OFFLOAD_BACKEND_METADATA": '{"name":"hicache"}',
           "TOTAL_CPU_DRAM_GB": "1199", "TP": "4"}
    env = dict(tpl, KV_OFFLOADING="none")
    ci_baseline.point_kv_and_parallel(env, {"kv_offloading": "none", "kv_offload_backend": None, "pp": 1, "dcp_size": 4,
                                            "pcp_size": 1})
    assert env == {"KV_OFFLOADING": "none", "TP": "4", "PP_SIZE": "1", "DCP_SIZE": "4", "PCP_SIZE": "1"}
    env = {"KV_OFFLOADING": "dram"}
    ci_baseline.point_kv_and_parallel(env, {"kv_offloading": "dram", "kv_offload_backend": "lmcache",
                                            "allocated_cpu_dram_gb": 800})
    assert env["KV_OFFLOAD_BACKEND"] == "lmcache" and env["TOTAL_CPU_DRAM_GB"] == "800"
    assert ci_baseline.kv_env_problem(env) is None
    env = {"KV_OFFLOADING": "dram"}   # newer bmk JSON (CI 44550646425): the backend as its metadata object
    ci_baseline.point_kv_and_parallel(env, {"kv_offloading": "dram", "kv_offload_backend": {"name": "hicache"},
                                            "allocated_cpu_dram_gb": 1274})
    assert (env["KV_OFFLOAD_BACKEND"], env["KV_OFFLOAD_BACKEND_METADATA"], env["TOTAL_CPU_DRAM_GB"]) == \
        ("hicache", '{"name":"hicache"}', "1274")
    env = {"KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "hicache"}   # the template had no metadata (lib layout)
    ci_baseline.point_kv_and_parallel(env, {"kv_offloading": "dram", "kv_offload_backend": "hicache"})
    assert env["KV_OFFLOAD_BACKEND_METADATA"] == '{"name":"hicache"}'
    env = dict(tpl)   # same backend as the template: its metadata kept
    ci_baseline.point_kv_and_parallel(env, {"kv_offloading": "dram", "kv_offload_backend": "hicache"})
    assert env["KV_OFFLOAD_BACKEND_METADATA"] == '{"name":"hicache"}'
    with pytest.raises(ValueError, match="required"):   # dram in the CI, but no backend named anywhere
        ci_baseline.point_kv_and_parallel({"KV_OFFLOADING": "dram"}, {"kv_offloading": "dram"})


def test_kv_env_problem():
    assert ci_baseline.kv_env_problem({}) is None
    assert ci_baseline.kv_env_problem({"KV_OFFLOADING": "none"}) is None
    assert "must be empty" in ci_baseline.kv_env_problem({"KV_OFFLOADING": "none", "KV_OFFLOAD_BACKEND": "hicache"})
    assert ci_baseline.kv_env_problem({"KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "hicache"}, strict=False) is None
    assert "must match" in ci_baseline.kv_env_problem({"KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "hicache",
                                                       "KV_OFFLOAD_BACKEND_METADATA": '{"name":"lmcache"}'})


def test_known_model_ids_from_stored_baselines():
    # CI 44550646425: a glm5.2 cell served amd/GLM-5.3-Quark-MXFP4-AttnFP8; make_baseline.draft refuses it by this set
    assert ci_baseline.known_model_ids("glm52") == {"amd/GLM-5.2-MXFP4"}
    assert "amd/GLM-5.3-Quark-MXFP4-AttnFP8" not in ci_baseline.known_model_ids("glm52")
