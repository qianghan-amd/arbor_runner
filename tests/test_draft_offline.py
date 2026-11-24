"""make_baseline.draft end to end, offline: a fake GitHub relay serves the recorded CI artifacts in tests/fixtures/ci/,
HF history and the registry are faked, and the draft is written under a tmp dir (the real runs/drafts is untouched).

This is the core of run_from_ci (step 2, "the starting point"); run_from_ci itself also needs the machine's local.yaml
and a `git archive` of the Arbor clone, so it is not run here. Bugs these guard: KV offload keys taken from the
template instead of the CI point (meta64 147755), kv_offload_backend given as an object (CI 44550646425), a glm5.2
cell that served GLM-5.3 (CI 44550646425), non-gfx950 and non-vLLM/SGLang CI runs.
"""
import copy
import json
from pathlib import Path

import pytest

from arbor_ab import ci_baseline, config, github_relay, registry, remote

import make_baseline

F = Path(__file__).parent / "fixtures" / "ci"
RUN = {"id": 44689368479, "created_at": "2026-11-20T06:39:42Z", "head_sha": "8f15ac7136382ee56734348269fd8980abc17528",
       "head_branch": "test-branch", "display_title": "test run", "html_url": "https://example/run"}
HF = [{"id": "a" * 40, "date": "2026-11-01T00:00:00Z"}, {"id": "b" * 40, "date": "2026-12-01T00:00:00Z"}]


class FakeRelay:
    def __init__(self, fmt, bmk_edit=None, recipe_edit=None):
        d = F / fmt
        b = json.loads((d / "bmk.json").read_text(encoding="utf-8"))
        self.name, self.bmk = b["artifact"], b["json"]
        if bmk_edit:
            bmk_edit(self.bmk)
        self.log = (d / "server_log_lines.txt").read_text(encoding="utf-8")
        self.cmd = (d / "vllm_command.txt").read_text(encoding="utf-8") if (d / "vllm_command.txt").exists() else None
        self.recipe = (d / "config.yaml").read_text(encoding="utf-8") if (d / "config.yaml").exists() else None
        if recipe_edit and self.recipe:
            self.recipe = recipe_edit(self.recipe)
        suffix = self.name[len("bmk_agentic_"):]
        self.arts = [{"id": 1, "name": self.name, "expired": False},
                     {"id": 2, "name": "agentic_" + suffix, "expired": False},
                     {"id": 3, "name": "server_logs_" + suffix, "expired": False}]

    def call(self, host, requests, timeout=900):
        out = []
        for r in requests:
            if r["op"] == "artifact_json":
                out.append({"http": 200, "data": {"result.json": self.bmk}})
            elif r["op"] == "artifact_text":
                out.append({"http": 200, "data": {"vllm_command.txt": self.cmd, "server.log": self.log} if self.cmd else {}})
            elif r["op"] == "artifact_tar" and "config" in r["pattern"]:
                out.append({"http": 200, "data": {"text": {"./logs/config.yaml": self.recipe}}})
            elif r["op"] == "artifact_tar":
                out.append({"http": 200, "data": {"text": {"./logs/x_agg_w0.out": self.log}}})
            else:
                raise AssertionError(f"unexpected relay request {r}")
        return out


@pytest.fixture
def offline(monkeypatch, tmp_path):
    def setup(fmt, **kw):
        relay = FakeRelay(fmt, **kw)
        monkeypatch.setattr(github_relay, "call", relay.call)
        monkeypatch.setattr(remote, "run_python", lambda host, prog, payload, timeout=600: HF)
        monkeypatch.setattr(registry, "check", lambda host, refs: [{"tag_digest": "sha256:" + "0" * 64}])
        monkeypatch.setattr(config, "load_local", lambda: {"ssh": {"github_relay_host": "relay"}})
        monkeypatch.setattr(config, "local_view", lambda site=None: {"ssh": {"github_relay_host": "relay"}})
        monkeypatch.setattr(make_baseline, "REPO_ROOT", tmp_path)
        return relay
    return setup


def draft(relay, model, target, **kw):
    return make_baseline.draft(model, target, RUN["id"], relay.bmk["tp"], run_info=RUN, arts=relay.arts, **kw)


@pytest.mark.parametrize("fmt,model,target", [("vllm_direct", "dsv41flash", "c1_p90"),
                                              ("vllm_srtctl", "dsv41flash", "c1_p90"),
                                              ("sglang_srtctl", "muse_spark", "c1_p90")])
def test_draft_from_each_ci_format(offline, tmp_path, fmt, model, target):
    relay = offline(fmt)
    d, rec = draft(relay, model, target)
    assert d == tmp_path / "runs" / "drafts" / f"{model}__{target}__ci{RUN['id']}"
    for f in ("draft.json", "review.md", "ci_bmk.json", "ci_commands.txt"):
        assert (d / f).exists(), f
    assert rec["launch"]["model_id"] == relay.bmk["model"]
    assert rec["host"]["framework"] == relay.bmk["framework"]
    assert rec["launch"]["corpus_rev"] == "a" * 40                       # HF main at the CI run's date
    assert rec["fingerprint"]["image_digest"]                            # from the image ref, or the registry
    env = rec["host"]["server_env"]
    assert env["KV_OFFLOADING"] == "none" and "KV_OFFLOAD_BACKEND" not in env
    assert ci_baseline.kv_env_problem(env) is None


def test_refuses_other_gpu_arch(offline):
    relay = offline("vllm_srtctl", bmk_edit=lambda b: b.update(hw="cluster:mi325x-amd"))
    with pytest.raises(SystemExit, match="ran on cluster:mi325x-amd"):
        draft(relay, "dsv41flash", "c1_p90")
    d, _ = draft(relay, "dsv41flash", "c1_p90", allow_other_hw=True)
    assert d.exists()


def test_refuses_other_model_unless_allowed(offline):
    relay = offline("vllm_srtctl", bmk_edit=lambda b: b.update(model="deepseek-ai/DeepSeek-V5-Flash"))
    with pytest.raises(SystemExit, match="another model under the same CI name"):
        draft(relay, "dsv41flash", "c1_p90")


@pytest.mark.parametrize("engine", ["atom", "tokenspeed"])
def test_refuses_unsupported_framework(offline, engine):
    relay = offline("sglang_srtctl", recipe_edit=lambda t: t.replace("engine: sglang", f"engine: {engine}"))
    with pytest.raises(SystemExit, match=f"srtctl engine '{engine}': only"):
        draft(relay, "muse_spark", "c1_p90")


def test_kv_none_point_drops_the_templates_backend(offline, monkeypatch):
    # meta64 147755: CI point kv none, template dram + hicache -> AgentX aggregation refused after the benchmark
    real = ci_baseline.template_record

    def template_with_hicache(model, target, framework=None):
        t = copy.deepcopy(real(model, target, framework))
        t["host"]["server_env"].update(KV_OFFLOADING="dram", KV_OFFLOAD_BACKEND="hicache",
                                       KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}', TOTAL_CPU_DRAM_GB="1199")
        return t
    monkeypatch.setattr(ci_baseline, "template_record", template_with_hicache)
    relay = offline("vllm_srtctl")
    _, rec = draft(relay, "dsv41flash", "c1_p90")
    env = rec["host"]["server_env"]
    assert env["KV_OFFLOADING"] == "none"
    assert not {"KV_OFFLOAD_BACKEND", "KV_OFFLOAD_BACKEND_METADATA", "TOTAL_CPU_DRAM_GB"} & set(env)


def test_kv_dram_point_takes_the_backend_object(offline):
    # CI 44550646425: kv_offload_backend is the metadata object {"name": "hicache"}
    def dram(b):
        b.update(kv_offloading="dram", kv_offload_backend={"name": "hicache"}, allocated_cpu_dram_gb=1274, dcp_size=1)
    relay = offline("sglang_srtctl", bmk_edit=dram)
    _, rec = draft(relay, "muse_spark", "c1_p90")
    env = rec["host"]["server_env"]
    assert (env["KV_OFFLOADING"], env["KV_OFFLOAD_BACKEND"], env["KV_OFFLOAD_BACKEND_METADATA"], env["TOTAL_CPU_DRAM_GB"]) \
        == ("dram", "hicache", '{"name":"hicache"}', "1274")
    assert env["DCP_SIZE"] == "1"
