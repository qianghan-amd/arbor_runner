"""Fetch the pinned model, draft and corpus into a node-local HF cache, offline-loadable (refs/main = the pin).

Runs on the node host in a venv with huggingface_hub (the version every earlier campaign's dl.sh used: 2.4.0).
Usage: python fetch.py <hub_dir> <shared_hub_or_-> repo:rev:kind [repo:rev:kind ...]     kind = model | dataset
- A dataset whose revision matches the shared hub's copy is rsynced from there (classic layout). A corpus written by
  huggingface_hub 2.4 could not be read offline by aiperf's `datasets` (musespark-c1-p90-sa/fixcorpus.sh).
- Every *.safetensors file the index names must resolve; refs/main is written so offline loads find the pin.
- HF_TOKEN, if the runner set it, lives in this process's environment on the host only; it is never printed.
Exit 1 on any failure. Idempotent: a pinned, complete entry is left as it is.
"""
import json
import os
import subprocess
import sys
import time


def entry_dir(hub, repo, kind):
    return os.path.join(hub, ("datasets--" if kind == "dataset" else "models--") + repo.replace("/", "--"))


def pinned(hub, repo, rev, kind):
    ref = os.path.join(entry_dir(hub, repo, kind), "refs", "main")
    return os.path.isfile(ref) and open(ref).read().strip().startswith(rev)


def check_shards(path, repo):
    idx = [f for f in os.listdir(path) if f.endswith(".safetensors.index.json")]
    want = set(json.load(open(os.path.join(path, idx[0])))["weight_map"].values()) if idx else set()
    missing = [f for f in want if not os.path.isfile(os.path.join(path, f))]
    if missing:
        sys.exit(f"[fetch] {repo}: {len(missing)} shards missing, e.g. {missing[:3]}")
    return len(want)


def copy_shared(hub, shared, repo, rev, kind):
    src = entry_dir(shared, repo, kind)
    ref = os.path.join(src, "refs", "main")
    if shared == "-" or not os.path.isfile(ref) or not open(ref).read().strip().startswith(rev):
        return False
    dst = entry_dir(hub, repo, kind)
    r = subprocess.run(["rsync", "-a", "--exclude", "trees/", src + "/", dst + "/"])
    if r.returncode != 0:
        sys.exit(f"[fetch] rsync {src} failed rc={r.returncode}")
    return True


def main(hub, shared, items):
    from huggingface_hub import __version__ as hv, snapshot_download
    os.makedirs(hub, exist_ok=True)
    print(f"[fetch] hub {hub}; huggingface_hub {hv}", flush=True)
    for item in items:
        repo, rev, kind = item.rsplit(":", 2)
        t = time.time()
        if pinned(hub, repo, rev, kind):
            how = "already pinned"
        elif kind == "dataset" and copy_shared(hub, shared, repo, rev, kind):
            how = "copied from the shared hub"
        else:
            p = snapshot_download(repo, revision=rev, repo_type=kind, cache_dir=hub, max_workers=16)
            d = os.path.join(entry_dir(hub, repo, kind), "refs")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "main"), "w").write(os.path.basename(p))
            how = "downloaded"
        snaps = os.path.join(entry_dir(hub, repo, kind), "snapshots")
        full = open(os.path.join(entry_dir(hub, repo, kind), "refs", "main")).read().strip()
        path = os.path.join(snaps, full)
        if not os.path.isdir(path):
            sys.exit(f"[fetch] {repo}: snapshot {full} not on disk")
        n = check_shards(path, repo) if kind == "model" else 0
        print(f"[fetch] ok {repo}@{full[:10]} ({kind}, {how}): {n} indexed shards, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
