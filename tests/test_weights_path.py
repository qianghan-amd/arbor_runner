"""Request `weights_path`: pre-downloaded weights are checked against the pin, linked (launch unchanged), and recorded."""
import json
import os
import sys
import types

import pytest

from arbor_ab import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts" / "runner"))
import fetch  # noqa: E402

REPO, REV = "org/model", "a" * 40
INDEX = json.dumps({"weight_map": {"w1": "model-1.safetensors", "w2": "model-2.safetensors"}})
CONFIG = '{"model_type": "x"}'


def symlinks_ok(tmp_path):
    try:
        os.symlink(tmp_path, tmp_path / "probe")
        return True
    except (OSError, NotImplementedError):
        return False


@pytest.fixture
def pinned_files(tmp_path, monkeypatch):
    """A fake huggingface_hub whose pinned index/config are INDEX/CONFIG."""
    src = tmp_path / "pinned"
    src.mkdir()
    (src / "model.safetensors.index.json").write_text(INDEX)
    (src / "config.json").write_text(CONFIG)
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda repo, name, revision, cache_dir: str(src / name)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return src


def user_dir(tmp_path, index=INDEX, config=CONFIG, shards=("model-1.safetensors", "model-2.safetensors")):
    d = tmp_path / "weights"
    d.mkdir()
    (d / "model.safetensors.index.json").write_text(index)
    (d / "config.json").write_text(config)
    for s in shards:
        (d / s).write_bytes(b"x")
    return d


def test_user_weights_matching_pin_are_linked(tmp_path, pinned_files):
    if not symlinks_ok(tmp_path):
        pytest.skip("this OS cannot create symlinks here")
    hub, w = tmp_path / "hub", user_dir(tmp_path)
    prov = fetch.link_user_model(str(hub), REPO, REV, str(w))
    snap = hub / "models--org--model" / "snapshots" / REV
    assert snap.is_symlink() and os.path.realpath(snap) == os.path.realpath(w)
    assert (hub / "models--org--model" / "refs" / "main").read_text() == REV
    assert prov["weights_source"] == "user_path" and prov["indexed_shards"] == 2
    assert all(c["matches_pinned"] for c in prov["files"].values())


def test_user_weights_not_matching_pin_refuse(tmp_path, pinned_files):
    w = user_dir(tmp_path, config='{"model_type": "other"}')
    with pytest.raises(SystemExit, match="not the pinned weights"):
        fetch.link_user_model(str(tmp_path / "hub"), REPO, REV, str(w))


def test_user_weights_missing_shard_refuse(tmp_path, pinned_files):
    w = user_dir(tmp_path, shards=("model-1.safetensors",))
    with pytest.raises(SystemExit, match="shards missing"):
        fetch.link_user_model(str(tmp_path / "hub"), REPO, REV, str(w))


def test_existing_download_wins_and_says_so(tmp_path, pinned_files):
    hub, w = tmp_path / "hub", user_dir(tmp_path)
    (hub / "models--org--model" / "snapshots" / REV).mkdir(parents=True)       # a real download already on the node
    prov = fetch.link_user_model(str(hub), REPO, REV, str(w))
    assert prov["weights_source"] == "downloaded" and "not needed" in prov["note"]


def test_short_revision_refused(tmp_path, pinned_files):
    with pytest.raises(SystemExit, match="full pinned revision"):
        fetch.link_user_model(str(tmp_path / "hub"), REPO, "abc123", str(user_dir(tmp_path)))
