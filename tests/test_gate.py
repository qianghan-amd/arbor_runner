import copy

from arbor_ab import gate
from test_baseline import full_record

# Shape of a real CI bmk artifact (InferenceX run 34657748450, artifact 10288756888), values trimmed.
CI = {"image": "vllm/vllm-openai-rocm:deepseekv41-flash-0909", "framework": "vllm", "tp": 4, "ep": 1, "dcp_size": 1,
      "num_gpus": 4, "conc": 32, "num_requests_successful": 2012,
      "request_metrics": {"throughput": {"total": {"tokens_per_second": 68067.85913},
                                         "per_gpu": {"total_tput_tps": 17016.96478}}}}


def record():
    r = full_record()
    r["host"]["framework"] = "vllm"
    r["fingerprint"]["image"] = "vllm/vllm-openai-rocm:deepseekv41-flash-0909"
    r["metrics"]["total_tput_tps"] = 68662.9774
    r["measured"]["bench_client_commit"] = "86699be610d5e6916eb6eb5b0c4a74676e8fb53d"
    return r


def test_bmk_candidates_exact_conc():
    arts = [{"name": n} for n in ("bmk_agentic_x_tp4_conc1_kvnone", "bmk_agentic_x_tp4_conc16_kvnone",
                                  "agentic_x_tp4_conc1_kvnone", "bmk_agentic_x_tp8_conc1_kvnone")]
    assert [a["name"] for a in gate.bmk_candidates(arts, 4, 1)] == ["bmk_agentic_x_tp4_conc1_kvnone"]


def test_structural_match_and_digest_in_ci_image():
    r = record()
    r["fingerprint"]["image_digest"] = "sha256:" + "a" * 64
    ci = dict(CI, image=CI["image"] + "@sha256:" + "a" * 64)
    assert gate.structural_diff(r, ci) == ([], [])
    ci["image"] = CI["image"] + "@sha256:" + "b" * 64
    assert any("digest" in d for d in gate.structural_diff(r, ci)[0])


def test_structural_mismatch_tp():
    diffs, _ = gate.structural_diff(record(), dict(CI, tp=8))
    assert diffs == ["tp 4 != CI 8"]


def test_pick_skips_empty_and_prefers_most_requests():
    r = record()
    empty = ({"id": 1}, dict(CI, num_requests_successful=0))
    short = ({"id": 2}, dict(CI, num_requests_successful=24))
    full = ({"id": 3}, dict(CI))
    wrong = ({"id": 4}, dict(CI, tp=8, num_requests_successful=9999))
    assert gate.pick(r, [empty, short, full, wrong])[0]["id"] == 3
    assert gate.pick(r, [empty]) is None


def test_numeric_pass_fail_client_mismatch():
    r = record()
    status, ci_v, ours, delta = gate.numeric(r, CI, 3.0, "86699be610d5e6916eb6eb5b0c4a74676e8fb53d")
    assert status == "pass" and round(delta, 3) == 0.874
    assert gate.numeric(r, CI, 0.5, "86699be610")[0] == "fail"
    assert gate.numeric(r, CI, 3.0, "8a76a78e41")[0] == "client_mismatch"
    unknown = copy.deepcopy(r)
    unknown["measured"]["bench_client_commit"] = None          # unknown client is judged on the number alone
    assert gate.numeric(unknown, CI, 3.0, "8a76a78e41")[0] == "pass"
    none = copy.deepcopy(r)
    none["metrics"]["total_tput_tps"] = None
    assert gate.numeric(none, CI, 3.0, "86699be610")[0] == "not_run"


def test_flags():
    assert gate.flags("pass", "pass") == (True, False)
    assert gate.flags("pass", "fail") == (False, False)
    assert gate.flags("pass", "artifact_expired") == (False, True)
    assert gate.flags("fail", "pass") == (False, False)
