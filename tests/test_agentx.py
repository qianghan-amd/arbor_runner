from arbor_ab import agentx

# Lines copied from real logs (dsv41f-c1-p90-sa2/runs/before/gsm8k.log, glm52-c1-p90-sa/runs/before/gsm8k.log).
LM_EVAL = ["|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9651|±  |0.0051|",
           "|     |       |strict-match    |     5|exact_match|↑  |0.9522|±  |0.0059|"]
SGLANG = ["Accuracy: 0.926", "Invalid: 0.000"]

AGG = {"request_metrics": {
    "throughput": {"total": {"tokens_per_second": 9882.60258}, "per_gpu": {"total_tput_tps": 2470.65065}},
    "latency": {"intvty": {"mean": 123.9, "p50": 128.45215, "p75": 112.7, "p90": 109.69379, "p95": 108.53033},
                "e2e_norm_intvty": {"p50": 105.2778, "p90": 29.23717, "p95": 13.49004}}}}


def test_metrics_from_agg():
    m, src = agentx.metrics_from_agg(AGG)
    assert m["total_tput_tps"] == 9882.60258 and m["tput_per_gpu"] == 2470.65065
    assert m["intvty"] == {"p50": 128.45215, "p90": 109.69379, "p95": 108.53033}
    assert m["e2e_norm_intvty"]["p90"] == 29.23717
    assert src["e2e_norm_intvty.p90"] == "request_metrics.latency.e2e_norm_intvty.p90"
    assert set(m["intvty"]) == {"p50", "p90", "p95"}            # p75/mean not carried: schema has p50/p90/p95 only


def test_old_agg_without_e2e_norm():
    old = {"request_metrics": {"latency": {"intvty": {"p90": 12.8}}, "throughput": {"per_gpu": {"total_tput_tps": 5959.5}}}}
    m, src = agentx.metrics_from_agg(old)
    assert m["e2e_norm_intvty"]["p90"] is None and m["total_tput_tps"] is None and "e2e_norm_intvty.p90" not in src


def test_gsm8k_lm_eval_and_sglang():
    g = agentx.gsm8k_from_log(LM_EVAL, 1319)
    assert (g["flexible_extract"], g["strict_match"], g["accuracy"], g["num_questions"]) == (0.9651, 0.9522, None, 1319)
    g = agentx.gsm8k_from_log(SGLANG, 1319)
    assert (g["flexible_extract"], g["accuracy"]) == (None, 0.926)
    assert not agentx.has_gsm8k(agentx.gsm8k_from_log([]))


def test_gsm8k_results_json():
    res = {"results": {"gsm8k": {"sample_len": 1319, "exact_match,strict-match": 0.4306, "exact_match,flexible-extract": 0.6240}}}
    g = agentx.gsm8k_from_lm_eval_results(res)
    assert (g["flexible_extract"], g["strict_match"], g["num_questions"]) == (0.6240, 0.4306, 1319)


def test_dcp_from_flags():
    assert agentx.dcp_from_flags("--stream-interval 10 --decode-context-parallel-size 8 --dcp-comm-backend a2a") == 8
    assert agentx.dcp_from_flags("--tensor-parallel-size 4") == 1
    assert agentx.dcp_from_flags(None) == 1
