import io
import sys
import tarfile

from arbor_ab import REPO_ROOT, baseline, render

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import submit_ab_test  # noqa: E402

SITE = {"node_local_root": "/mnt/nl/u/arbor_ab", "shared_hf_hub": "/shared_nfs/huggingface/hub", "infx_clone": "/x/InferenceX",
        "gateway_env": None}


def test_check_runnable_on_real_records():
    assert render.check_runnable(baseline.load_latest("dsv41flash", "c1_p90")) == []
    assert any("MODEL_PREFIX" in w for w in render.check_runnable(baseline.load_latest("muse_spark", "c16_tput")))
    assert render.check_runnable(baseline.load_latest("kimik3", "c32_tput"))[0].startswith("no launch block")
    assert any("corpus_id" in w for w in render.check_runnable(baseline.load_latest("glm52", "c16_tput")))
    # corpus pins recovered from evidence (evidence/2026-11-19_0830_launch/corpus_pins.yaml)
    assert render.check_runnable(baseline.load_latest("glm52", "c1_p90")) == []
    assert render.check_runnable(baseline.load_latest("dsv41flash", "c32_tput")) == []


def test_arbor_config_dsv41_c1():
    r = baseline.load_latest("dsv41flash", "c1_p90")
    c = render.arbor_config(r, "20261119-000000_f7abce97_t", "f7abce97", "/w/runs/x/dsv41flash__c1_p90", "ab_c", 1440)
    assert c["pack"] == "agentx" and c["review_gate"] is False and c["time_budget_minutes"] == 1440
    assert c["concurrency"] == 1 and c["gpu_ids"] == "0,1,2,3" and c["tp"] == 4 and c["framework"] == "vllm"
    assert c["session_root"] == "/w/runs/x/dsv41flash__c1_p90/sessions/full" and c["session_mount"] == "/workspace/sessions"
    assert c["server_env"]["INFMAX_CONTAINER_WORKSPACE"] == "/workspace/InferenceX"
    assert c["server_flags"] == r["host"]["server_flags"]                      # the pristine launch, unchanged
    assert c["bench_script"] == "/workspace/Arbor/packs/agentx/validator/bench_agentx.sh"   # P6 job 299696 lacked it
    assert c["correctness_script"] == "/workspace/Arbor/packs/agentx/validator/correctness.sh"
    assert render.opt_objective(r) == "P90 interactivity"
    assert render.opt_objective(baseline.load_latest("kimik3", "c1_p90")) == "P90 interactivity"


def test_cell_env_and_sbatch():
    r = baseline.load_latest("dsv41flash", "c1_p90")
    e = render.cell_env(r, "20261119-000000_f7abce97_t", "f7abce97", "smoke", "/w/c", "ab_c", 0, SITE, "/w/arbor.tar", "ab" * 32)
    assert e["MODEL_REV"] == "dba1be0a40aa45a94ad051997016db3960a90277" and e["CORPUS_REV"] == "23f152f6"
    assert e["INFX_SHA"].startswith("86699be610") and e["GSM8K_TOOL"] == "infx_lm_eval" and e["GSM8K_CONC"] == "16"
    assert render.env_file({"A": "x y", "B": "it's"}) == "A='x y'\nB='it'\"'\"'s'\n"   # shlex-quoted for bash
    assert render.container_name("20261119-075610_f7abce97_smoke", "dsv41flash", "c1_p90") == "ab_20261119-075610_dsv41flash_c1_p90"
    assert render.job_minutes("smoke", 0, r) == 150
    assert render.job_minutes("campaign_smoke", 20, r) == 60 + 30 + 20 + 30
    skip = render.cell_env(r, "20261119-000000_f7abce97_t", "f7abce97", "full", "/w/c", "ab_c", 180, SITE, "/w/a.tar", "ab" * 32,
                           skip_gsm8k=True)
    assert skip["NQ_FULL"] == "0" and skip["NQ_SMOKE"] == "0"
    assert render.job_minutes("full", 1440, r) == 60 + 30 + 2 * 150 + 1440 + 60


def _tar(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def test_archive_crlf_refuses_shell_only(tmp_path):
    p = tmp_path / "a.tar"
    p.write_bytes(_tar({"run.sh": b"echo a\r\n", "tool": b"#!/bin/bash\r\necho b\r\n", "doc.md": b"x\r\n",
                        "x.py": b"print(1)\r\n", "ok.sh": b"echo ok\n", "bin.dat": b"\0\r\n"}).read())
    bad, other = submit_ab_test.archive_crlf_files(p)
    assert sorted(bad) == ["run.sh", "tool"] and sorted(other) == ["doc.md", "x.py"]


def test_arbor_launcher_is_default():
    # #74 closed 2026-11-24 (not reproduced): Arbor runs under its launcher unless the request asks arbor_standalone
    r = baseline.load_latest("dsv41flash", "c1_p90")
    args = (r, "20261124-000000_7c211696_t", "7c211696", "campaign", "/w/c", "ab_c", 90, SITE, "/w/a.tar", "ab" * 32)
    assert render.cell_env(*args)["ARBOR_LAUNCHER"] == "1"
    assert render.cell_env(*args, arbor_launcher=False)["ARBOR_LAUNCHER"] == "0"
