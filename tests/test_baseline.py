import copy
import json
from pathlib import Path

import pytest

from arbor_ab import baseline

FIX = Path(__file__).parent / "fixtures"
ADDED_REQUIRED = {  # fields this repo adds to the design.md 3.6 example (INTENT.md F4/F6/Q4, design 3.4)
    "<root>": {"verification", "ceiling", "provenance"},
    "target": {"key", "direction"},
    "host": {"dcp", "recipe_point", "model_path", "server_flags", "server_env"},
    "fingerprint": {"image_digest"},
    "health": {"checked_at"},
    "measured": {"cluster", "bench_client_commit", "arbor_commit"},
    "metrics/gsm8k": {"accuracy", "num_questions"},
}


def design_example():
    return json.loads((FIX / "design_3_6_example.json").read_text(encoding="utf-8"))


def full_record():
    """The design example plus the fields this repo adds; epoch_id derived from the fingerprint."""
    r = design_example()
    r["target"].update(key="c32_tput", direction="max")
    r["host"].update(dcp=1, recipe_point=None, model_path=None, server_flags=None, server_env=None)
    r["fingerprint"]["image_digest"] = None
    r["health"]["checked_at"] = None
    r["measured"].update(cluster="crusoe", bench_client_commit=None, arbor_commit=None)
    r["metrics"]["gsm8k"].update(accuracy=None, num_questions=None)
    r["verification"] = {"ci_verified": False, "unverified_against_ci": False,
                         "structural": {"status": "not_run", "checked_at": None, "detail": None},
                         "numeric": {"status": "not_run", "checked_at": None, "ci_artifact": None, "ci_value": None,
                                     "measured_value": None, "delta_pct": None, "tolerance_pct": None}}
    r["ceiling"] = None
    r["provenance"] = {}
    r["epoch_id"] = baseline.derive_epoch_id(r)
    return r


def test_design_example_only_misses_added_fields():
    errs = baseline.validation_errors(design_example())
    assert errs, "the raw example lacks the added fields, so it must not validate"
    for e in errs:
        path, msg = e.split(": ", 1)
        assert "is a required property" in msg, e   # nothing the example already has is rejected
        field = msg.split("'")[1]
        assert field in ADDED_REQUIRED.get(path, set()), e


def test_full_record_valid():
    assert baseline.validation_errors(full_record()) == []


@pytest.mark.parametrize("mutate", [
    lambda r: r["metrics"].update(p90_intvty=13.3),                     # flattened AgentX name
    lambda r: r["metrics"].update(e2e_norm_intvty_p90=13.3),
    lambda r: r["metrics"]["intvty"].update(p99=1.0),
    lambda r: r["metrics"].pop("e2e_norm_intvty"),
    lambda r: r["target"].update(primary_metric_path="total_tput_tps"),  # must be a path into metrics
    lambda r: r.update(data_quality_warning="bad"),                      # must be a list of {code, detail}
    lambda r: r.update(data_quality_warning=[{"code": "made_up", "detail": "x"}]),
    lambda r: r["fingerprint"].update(ci_run_id=None),                   # agentx_ci needs the CI run
    lambda r: r["verification"]["numeric"].update(status="ok"),
])
def test_rejects(mutate):
    r = full_record()
    mutate(r)
    assert baseline.validation_errors(r)


def test_data_quality_warning_list_ok():
    r = full_record()
    r["data_quality_warning"] = [{"code": "gsm8k_anomalous", "detail": "0.624 vs ~0.965 at C1"}]
    assert baseline.validation_errors(r) == []


def test_epoch_id_stable_and_sensitive():
    r = full_record()
    eid = r["epoch_id"]
    assert eid.startswith("dsv41flash__c32_tput__86699be610_")
    full = copy.deepcopy(r)
    full["fingerprint"]["ci_commit"] = "86699be610" + "0" * 30          # full SHA == short SHA
    assert baseline.derive_epoch_id(full) == eid
    other = copy.deepcopy(r)
    other["metrics"]["total_tput_tps"] = 1.0                           # measurements are not part of the fingerprint
    assert baseline.derive_epoch_id(other) == eid
    for field, value in [("image", "vllm/other:tag"), ("ci_run_id", "1"), ("ci_commit", "aaaaaaaaaa")]:
        x = copy.deepcopy(r)
        x["fingerprint"][field] = value
        assert baseline.derive_epoch_id(x) != eid, field


def test_custom_epoch_follows_config():
    r = full_record()
    r["fingerprint"].update(source_type="user_custom", ci_commit=None, ci_run_id=None)
    r["host"]["server_flags"] = "--max-num-seqs 2"
    a = baseline.derive_epoch_id(r)
    assert "__custom_" in a
    r["host"]["server_flags"] = "--max-num-seqs 4"
    assert baseline.derive_epoch_id(r) != a


def test_store_latest_idempotent_and_supersede(tmp_path):
    r = full_record()
    p = baseline.store(r, root=tmp_path)
    first = p.read_bytes()
    assert baseline.store(r, root=tmp_path) == p and p.read_bytes() == first
    assert baseline.load_latest("dsv41flash", "c32_tput", root=tmp_path)["epoch_id"] == r["epoch_id"]

    new = copy.deepcopy(r)
    new["fingerprint"]["image"] = "vllm/vllm-openai-rocm:newer"
    new["epoch_id"] = baseline.derive_epoch_id(new)
    baseline.store(new, root=tmp_path)
    assert baseline.load_latest("dsv41flash", "c32_tput", root=tmp_path)["epoch_id"] == new["epoch_id"]
    old = baseline.load(p)
    assert old["superseded_by"] == new["epoch_id"]                     # archived, not deleted
    assert len(list((tmp_path / "dsv41flash" / "c32_tput").glob("*__*.json"))) == 2


def test_store_rejects_wrong_epoch_id(tmp_path):
    r = full_record()
    r["epoch_id"] = "dsv41flash__c32_tput__86699be610_deadbeef"
    with pytest.raises(ValueError, match="does not match fingerprint"):
        baseline.store(r, root=tmp_path)
