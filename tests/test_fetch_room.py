"""fetch.py's room check (runner): a node's disk may be its root disk, so the cache evicts before it downloads."""
import importlib.util
import os
import time
from collections import namedtuple

import pytest

from arbor_ab import REPO_ROOT

spec = importlib.util.spec_from_file_location("fetch", REPO_ROOT / "scripts" / "runner" / "fetch.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
Usage = namedtuple("Usage", "total used free")


def entry(hub, name, nbytes, used_ago_s):
    d = hub / name / "blobs"
    d.mkdir(parents=True)
    (d / "x").write_bytes(b"\0" * nbytes)
    stamp = hub / name / fetch.LAST_USED
    stamp.write_text("t")
    t = time.time() - used_ago_s
    os.utime(stamp, (t, t))
    return str(hub / name)


def test_make_room_evicts_lru_not_needed_then_stops(tmp_path, monkeypatch):
    fetch.GB = 1000                                    # sizes in KB-scale "GB" for the test
    hub = tmp_path / "hub"
    old = entry(hub, "models--a--old", 3000, 300)
    mid = entry(hub, "models--b--mid", 3000, 200)
    keep = entry(hub, "models--c--needed", 3000, 900)  # oldest, but this run needs it
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda p: Usage(0, 0, 10 ** 12))
    fetch.make_room(str(hub), need=2000, keep={keep}, min_free_gb=None, max_cache_gb=8.1)   # 9003 + 2000 > 8100
    assert not os.path.exists(old) and os.path.exists(mid) and os.path.exists(keep)      # one LRU eviction was enough


def test_make_room_fails_before_download_when_nothing_left(tmp_path, monkeypatch):
    fetch.GB = 1000
    hub = tmp_path / "hub"
    keep = entry(hub, "models--c--needed", 3000, 10)
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda p: Usage(0, 0, 5000))       # 5 "GB" free
    with pytest.raises(SystemExit) as e:
        fetch.make_room(str(hub), need=4000, keep={keep}, min_free_gb=2, max_cache_gb=None)
    assert e.value.code == fetch.NO_ROOM == 3 and os.path.exists(keep)


def test_make_room_noop_when_it_fits(tmp_path, monkeypatch):
    fetch.GB = 1000
    hub = tmp_path / "hub"
    a = entry(hub, "models--a--x", 1000, 100)
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda p: Usage(0, 0, 10 ** 9))
    fetch.make_room(str(hub), need=1000, keep=set(), min_free_gb=1, max_cache_gb=10)
    assert os.path.exists(a)


def shared_entry(hub, name, blobs, used_ago_s):
    """An entry as huggingface_hub 2.4 writes it with a cache_dir: snapshot links into the SHARED <hub>/blobs."""
    snap = hub / name / "snapshots" / "rev"
    snap.mkdir(parents=True)
    for b, nbytes in blobs.items():
        blob = hub / "blobs" / b
        blob.parent.mkdir(exist_ok=True)
        if not blob.exists():
            blob.write_bytes(b"\0" * nbytes)
        try:
            os.symlink(blob, snap / b)
        except OSError:
            pytest.skip("no symlink permission on this workstation")
    stamp = hub / name / fetch.LAST_USED
    stamp.write_text("t")
    t = time.time() - used_ago_s
    os.utime(stamp, (t, t))
    return str(hub / name)


def test_shared_blobs_evicted_through_links_and_orphans_first(tmp_path, monkeypatch):
    fetch.GB = 1000
    hub = tmp_path / "hub"
    old = shared_entry(hub, "models--a--old", {"x1": 3000, "shared": 1000}, 300)
    keep = shared_entry(hub, "models--c--needed", {"k1": 2000, "shared": 1000}, 900)
    (hub / "blobs" / "orphan").write_bytes(b"\0" * 500)                              # an earlier eviction's leftover
    user = tmp_path / "user_weights"
    user.mkdir()
    (user / "w").write_bytes(b"\0" * 9000)
    os.symlink(user, hub / "models--c--needed" / "snapshots" / "user")              # a user-model link: never deleted
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda p: Usage(0, 0, 10 ** 12))
    fetch.make_room(str(hub), need=1000, keep={keep}, min_free_gb=None, max_cache_gb=4.2)   # 6500 + 1000 > 4200
    assert not (hub / "blobs" / "orphan").exists() and not (hub / "blobs" / "x1").exists()
    assert (hub / "blobs" / "shared").exists() and (hub / "blobs" / "k1").exists()  # still used by the kept entry
    assert not os.path.exists(old) and (user / "w").exists()
