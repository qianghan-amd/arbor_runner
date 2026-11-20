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


def test_model_rev_from_log():
    rev = "dba1be0a40aa45a94ad051997016db3960a90277"
    log = f"loading /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/{rev}/config.json\n" * 3
    assert ci_baseline.model_rev_from_log(log, "deepseek-ai/DeepSeek-V4.1-Flash") == rev
    with pytest.raises(ValueError):
        ci_baseline.model_rev_from_log("no snapshot here", "deepseek-ai/DeepSeek-V4.1-Flash")


def test_corpus_rev_at_ci_time():
    commits = [{"id": "new", "date": "2026-10-01T00:00:00.000Z"}, {"id": "23f152f6", "date": "2026-06-21T10:00:00.000Z"},
               {"id": "first", "date": "2026-06-21T09:00:00.000Z"}]
    assert ci_baseline.corpus_rev_at(commits, "2026-09-11T23:21:29Z") == "23f152f6"   # the run predates "new"
    assert ci_baseline.corpus_rev_at(commits, "2026-11-01T00:00:00Z") == "new"
    with pytest.raises(ValueError):
        ci_baseline.corpus_rev_at(commits, "2026-01-01T00:00:00Z")
