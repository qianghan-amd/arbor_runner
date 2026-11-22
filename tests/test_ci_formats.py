"""Every InferenceX CI artifact format seen so far, from real (trimmed) artifacts in tests/fixtures/ci/<format>/.

A new CI format gets a new fixture directory here: the parser must read it, and the launch it parses must match the
argument dump of the CI's own server (ci_baseline.check_against_server), or the format change is caught.
Fixtures: vllm_direct = run 34657748450 (DSv4.1 Flash, 2026-09); vllm_srtctl = run 44689368479 (DSv4.1 Flash nightly,
2026-11); sglang_srtctl = run 38877885539 (Muse Spark, 2026-10). Secrets scrubbed (env names ending TOKEN/KEY/...).
"""
import json
from pathlib import Path

import pytest

from arbor_ab import ci_baseline, config

F = Path(__file__).parent / "fixtures" / "ci"
FORMATS = {   # format: (model key, target, model id, model rev, framework)
    "vllm_direct": ("dsv41flash", "c1_p90", "deepseek-ai/DeepSeek-V4.1-Flash", "dba1be0a40aa45a94ad051997016db3960a90277", "vllm"),
    "vllm_srtctl": ("dsv41flash", "c1_p90", "deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6", "vllm"),
    "sglang_srtctl": ("muse_spark", "c1_p90", "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4",
                      "c7bc74fd64371315a091bc15f9723882eead0dc1", "sglang"),
}


def parsed(fmt):
    d = F / fmt
    if (d / "vllm_command.txt").exists():
        cmd = ci_baseline.parse_vllm_command((d / "vllm_command.txt").read_text(encoding="utf-8"))
        cmd.setdefault("framework", "vllm")
        return cmd, None
    cmd, env, _ = ci_baseline.parse_srt_recipe((d / "config.yaml").read_text(encoding="utf-8"))
    return cmd, env


def log(fmt):
    return (F / fmt / "server_log_lines.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_each_real_format_parses_and_matches_its_server(fmt):
    model, target, model_id, rev, fw = FORMATS[fmt]
    cmd, env = parsed(fmt)
    assert cmd["framework"] == fw and cmd["model"] == model_id
    assert ci_baseline.check_against_server(fw, cmd["flags"], log(fmt)) == []
    assert ci_baseline.model_rev_from_log(log(fmt), model_id) == rev
    bmk = json.loads((F / fmt / "bmk.json").read_text(encoding="utf-8"))
    assert bmk["json"]["framework"] == fw
    pts = ci_baseline.ci_points([{"name": bmk["artifact"], "expired": False}])
    assert ci_baseline.choose_cell(pts, config.load_matrix()) == (model, target, bmk["json"]["tp"])


def test_the_check_catches_a_dropped_a_changed_and_an_unknown_flag():
    cmd, _ = parsed("vllm_srtctl")
    dropped = cmd["flags"].replace("--max-num-seqs 1", "")
    assert any("max_num_seqs" in p for p in ci_baseline.check_against_server("vllm", dropped, log("vllm_srtctl")))
    changed = cmd["flags"].replace("--max-num-seqs 1", "--max-num-seqs 4")
    assert any("--max-num-seqs: parsed '4'" in p for p in ci_baseline.check_against_server("vllm", changed, log("vllm_srtctl")))
    sg, _ = parsed("sglang_srtctl")
    notes = []
    probs = ci_baseline.check_against_server("sglang", sg["flags"] + " --renamed-flag 3", log("sglang_srtctl"), notes)
    assert probs == [] and notes == ["--renamed-flag is in the parsed flags but not among the server's arguments"]
    # SGLang: a changed value still stops it; `auto` is the server's to resolve
    assert ci_baseline.check_against_server("sglang", sg["flags"].replace("--page-size 16", "--page-size 32"), log("sglang_srtctl"))
    notes = []
    assert ci_baseline.check_against_server("sglang", sg["flags"] + " --attention-backend auto", log("sglang_srtctl"), notes) == []
    # SGLang rescales mem_fraction_static itself: not a mismatch; its TP flag is checked under its own name, tp_size
    assert "--mem-fraction-static 0.85" in sg["flags"] and "--tensor-parallel-size 4" in sg["flags"]
    with pytest.raises(ValueError, match="no argument dump"):
        ci_baseline.check_against_server("vllm", cmd["flags"], "no dump here")


def test_synthetic_acceptance_is_in_every_format():
    """The CI's spec decode is synthetic in each format (memory ci-synthetic-acceptance): vLLM in its flags, SGLang in env."""
    for fmt in ("vllm_direct", "vllm_srtctl"):
        assert '"rejection_sample_method": "synthetic"' in parsed(fmt)[0]["flags"].replace('":"', '": "')
    assert parsed("sglang_srtctl")[1]["SGLANG_SIMULATE_ACC_LEN"] == "4.05"
