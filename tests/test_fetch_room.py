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
    with pytest.raises(SystemExit, match="not enough room"):
        fetch.make_room(str(hub), need=4000, keep={keep}, min_free_gb=2, max_cache_gb=None)
    assert os.path.exists(keep)


def test_make_room_noop_when_it_fits(tmp_path, monkeypatch):
    fetch.GB = 1000
    hub = tmp_path / "hub"
    a = entry(hub, "models--a--x", 1000, 100)
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda p: Usage(0, 0, 10 ** 9))
    fetch.make_room(str(hub), need=1000, keep=set(), min_free_gb=1, max_cache_gb=10)
    assert os.path.exists(a)
