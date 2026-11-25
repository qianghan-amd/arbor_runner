"""scripts/agentx_db.py offline: record building, the points index, latest/compare, deterministic gzip files."""
import sys

from arbor_ab import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import agentx_db  # noqa: E402


def point(hw, prefix="dsv41flash", conc=1, tput=100.0, p90=10.0, fw="vllm", tp=2, kv="none"):
    # the shape of results_bmk/agg_bmk.json entries (InferenceX run 44689368479, 2026-11-20)
    return {"hw": f"cluster:{hw}", "infmax_model_prefix": prefix, "model": "x/y", "framework": fw, "precision": "fp4",
            "spec_decoding": "mtp", "tp": tp, "ep": 1, "pp": 1, "dp_attention": "false", "dcp_size": 1,
            "num_gpus": tp, "conc": conc, "kv_offloading": kv, "kv_offload_backend": {"name": "hicache"} if kv == "dram" else None,
            "image": "img@sha256:ab", "recipe_fingerprint": "f", "num_requests_successful": 9, "num_requests_total": 10,
            "avg_power_w": 700.0, "joules_per_output_token": 5.0, "power_valid": 1,
            "request_metrics": {"throughput": {"total": {"tokens_per_second": tput * tp},
                                               "per_gpu": {"total_tput_tps": tput, "output_tput_tps": tput / 100}},
                                "latency": {"e2e_norm_intvty": {"p50": p90 * 2, "p90": p90, "p95": p90 / 2},
                                            "intvty": {"p50": 1.0, "p90": 2.0, "p95": 3.0},
                                            "ttft": {"p50": 900.0}, "tpot": {"p50": 12.0}}}}


def run(rid, date, branch="main"):
    return {"id": rid, "html_url": f"u/{rid}", "created_at": f"{date}T00:00:00Z", "head_branch": branch,
            "head_sha": "abc", "event": "workflow_dispatch", "display_title": "t", "run_attempt": 1, "conclusion": "success"}


def make_db(tmp_path):
    db = tmp_path / "official"
    (db / "runs").mkdir(parents=True)
    hw = set(agentx_db.HW_DEFAULT)
    evals = [{"hw": "CLUSTER:MI355X-AMDS", "model_prefix": "dsv41flash", "framework": "vllm", "precision": "fp4",
              "spec_decoding": "mtp", "tp": 2, "ep": 1, "conc": 1, "dp_attention": "false", "em_strict": 0.97}]
    r1 = agentx_db.build_run(run(1, "2026-11-20"), [{"name": "bmk_agentic_x", "id": 5, "size_in_bytes": 2,
                                                       "expires_at": "2027-02-18"}],
                             {"results_bmk": [point("mi355x-amds", tput=100.0), point("gb300-nv", tput=999.0)],
                              "eval_results_all": evals, "changelog": {"base_ref": "origin/main"},
                              "run_stats": {"mi355x-amds": {"n_success": 1}, "gb300-nv": {"n_success": 1}}}, hw,
                             {"pr": 11, "merged_at": "2026-11-21T10:00:00Z", "merge_run_id": 91})
    r2 = agentx_db.build_run(run(2, "2026-11-22", "dev/x"), [],
                             {"results_bmk": [point("mi355x-amds", tput=120.0), point("b300-dsxe", tput=150.0)]}, hw,
                             {"pr": 12, "merged_at": "2026-11-23T10:00:00Z", "merge_run_id": 92})
    unofficial = agentx_db.build_run(run(4, "2026-11-24"), [], {"results_bmk": [point("mi355x-amds", tput=1.0)]}, hw)
    assert agentx_db.build_run(run(3, "2026-11-23"), [], {"results_bmk": [point("h100-dsxe")]}, hw) is None
    for rec in (r1, r2, unofficial):
        agentx_db.write_run(agentx_db.run_path(db, rec["run"]["id"]), rec)
    return db, r1


def test_build_run_keeps_only_the_wanted_hardware_and_hashes(tmp_path):
    db, r1 = make_db(tmp_path)
    assert r1["hardware"] == ["mi355x-amds"] and len(r1["points"]) == 1 and r1["run_stats"] == {"mi355x-amds": {"n_success": 1}}
    assert set(r1["sha256"]) == {"results_bmk", "eval_results_all", "changelog", "run_stats"}
    assert r1["artifacts"] == [{"name": "bmk_agentic_x", "id": 5, "kind": "bmk_agentic", "size": 2,
                                "expires_at": "2027-02-18", "expired": None}]
    # a run file is written once: the same record gives the same bytes
    p = agentx_db.run_path(db, 1)
    first = p.read_bytes()
    agentx_db.write_run(p, agentx_db.read_run(p))
    assert p.read_bytes() == first


def test_index_rows_latest_and_compare(tmp_path):
    db, _ = make_db(tmp_path)
    assert agentx_db.write_index(db) == 3
    lines = (db / "points.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == agentx_db.COLUMNS and "\r" not in "".join(lines)
    rs = agentx_db.rows(db)
    assert [r["run_id"] for r in rs] == [2, 2, 1]                                  # newest first
    old = next(r for r in rs if r["run_id"] == 1)
    assert old["tput_per_gpu"] == 100.0 and old["e2e_norm_intvty_p90"] == 10.0 and old["gsm8k_em_strict"] == 0.97
    assert old["ttft_p50_ms"] == 900.0 and old["output_tput_per_gpu"] == 1.0
    latest = agentx_db.select(rs, model="dsv41flash", latest=True)
    assert sorted((r["hw"], r["run_id"]) for r in latest) == [("b300-dsxe", 2), ("mi355x-amds", 2)]
    assert [r["run_id"] for r in agentx_db.select(rs, branch="main")] == [1]
    assert {r["pr"] for r in rs} == {11, 12} and rs[0]["merged_at"] == "2026-11-23T10:00"   # run 4: not official
    out, hws = agentx_db.compare(rs, "tput_per_gpu")
    assert hws == ["b300-dsxe", "mi355x-amds"]
    assert out[0]["b300-dsxe"] == 150.0 and out[0]["mi355x-amds"] == 120.0 and "run 2" in out[0]["mi355x-amds config"]


def test_hw_name():
    assert agentx_db.hw_name("cluster:mi355x-amds") == "mi355x-amds" == agentx_db.hw_name("CLUSTER:MI355X-AMDS")


def test_source_of_pr_comments():
    # PR 5201 (Merge Ingest 45520339963, 2026-11-25): "/use 45314602375", then the bot's ingest report
    comments = [{"body": "/use 45200000000", "user": {"login": "a", "type": "User"}},
                {"body": "looks good", "user": {"login": "b", "type": "User"}},
                {"body": "/use 45314602375\n", "user": {"login": "Oseltamivir", "type": "User"}},
                {"body": "Ingest report: 7 points ingested", "user": {"login": "github-actions[bot]", "type": "Bot"},
                 "created_at": "2026-11-25T00:47:00Z"}]
    src, report = agentx_db.source_of(comments)
    assert src == 45314602375 and report["by"] == "github-actions[bot]" and "ingested" in report["body"]
    assert agentx_db.source_of([{"body": "no reuse here", "user": {"type": "User"}}]) == (None, None)
