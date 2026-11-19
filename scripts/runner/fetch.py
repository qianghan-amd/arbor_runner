"""Fetch the pinned model, draft and corpus into a node-local HF cache, offline-loadable (refs/main = the pin).

Runs on the node host in a venv with huggingface_hub (the version every earlier campaign's dl.sh used: 2.4.0).
Usage: python fetch.py <hub_dir> <shared_hub_or_-> [--user-model PATH] [--provenance FILE] repo:rev:kind [...]
       kind = model | dataset; the first `model` item is the served model.
- A dataset whose revision matches the shared hub's copy is rsynced from there (classic layout). A corpus written by
  huggingface_hub 2.4 could not be read offline by aiperf's `datasets` (musespark-c1-p90-sa/fixcorpus.sh).
- --user-model PATH: the served model's weights are already at PATH (pre-downloaded; node or shared storage). Instead of
  downloading them, the pinned hub entry (snapshots/<rev>) becomes a symlink to PATH, so the launch (model_path) stays
  exactly the baseline's. PATH must hold the pinned weights: its model.safetensors.index.json and config.json must be
  byte-identical to the pinned revision's (those two small files are downloaded to compare), and every indexed shard
  must exist. Otherwise the fetch fails. A pinned snapshot already complete on the node is used as it is (the user path
  is then not needed, and the provenance says so).
- --provenance FILE: writes {weights_source: downloaded | user_path, ...} for the served model, so every output says
  where the weights came from.
- Every *.safetensors file the index names must resolve; refs/main is written so offline loads find the pin.
- HF_TOKEN, if the runner set it, lives in this process's environment on the host only; it is never printed.
Exit 1 on any failure. Idempotent: a pinned, complete entry is left as it is.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

COMPARE = ("model.safetensors.index.json", "config.json")


def entry_dir(hub, repo, kind):
    return os.path.join(hub, ("datasets--" if kind == "dataset" else "models--") + repo.replace("/", "--"))


def pinned(hub, repo, rev, kind):
    ref = os.path.join(entry_dir(hub, repo, kind), "refs", "main")
    if not (os.path.isfile(ref) and open(ref).read().strip().startswith(rev)):
        return False
    return os.path.isdir(os.path.join(entry_dir(hub, repo, kind), "snapshots", open(ref).read().strip()))


def write_ref(hub, repo, kind, full_rev):
    d = os.path.join(entry_dir(hub, repo, kind), "refs")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "main"), "w").write(full_rev)


def check_shards(path, repo):
    idx = [f for f in os.listdir(path) if f.endswith(".safetensors.index.json")]
    want = set(json.load(open(os.path.join(path, idx[0])))["weight_map"].values()) if idx else set()
    missing = [f for f in want if not os.path.isfile(os.path.join(path, f))]
    if missing:
        sys.exit(f"[fetch] {repo}: {len(missing)} shards missing, e.g. {missing[:3]}")
    return len(want)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def link_user_model(hub, repo, rev, user_path):
    """Check user_path against the pinned revision, then make snapshots/<rev> a symlink to it. Returns provenance."""
    from huggingface_hub import hf_hub_download
    user_path = os.path.realpath(user_path)
    if not os.path.isdir(user_path):
        sys.exit(f"[fetch] --user-model {user_path}: not a directory on this node")
    if len(rev) != 40:
        sys.exit(f"[fetch] --user-model needs the full pinned revision of {repo}, got {rev!r}")
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in COMPARE:
            mine = os.path.join(user_path, name)
            if not os.path.isfile(mine):
                sys.exit(f"[fetch] --user-model {user_path}: has no {name}")
            ref = hf_hub_download(repo, name, revision=rev, cache_dir=tmp)
            checks[name] = {"sha256": sha256(mine), "matches_pinned": sha256(mine) == sha256(ref)}
    bad = [n for n, c in checks.items() if not c["matches_pinned"]]
    if bad:
        sys.exit(f"[fetch] --user-model {user_path}: {bad} differ from {repo}@{rev[:10]}: not the pinned weights")
    n = check_shards(user_path, repo)
    snap = os.path.join(entry_dir(hub, repo, "model"), "snapshots", rev)
    os.makedirs(os.path.dirname(snap), exist_ok=True)
    if os.path.islink(snap) or not os.path.exists(snap):
        if os.path.islink(snap):
            os.unlink(snap)
        os.symlink(user_path, snap)
        write_ref(hub, repo, "model", rev)
        return {"weights_source": "user_path", "path": user_path, "indexed_shards": n, "files": checks}
    return {"weights_source": "downloaded", "path": snap, "files": checks,
            "note": f"the pinned snapshot was already complete on this node; --user-model {user_path} was not needed"}


def main(argv):
    hub, shared, *rest = argv
    user_model = provenance = None
    while rest and rest[0].startswith("--"):
        flag, val, *rest = rest
        if flag == "--user-model":
            user_model = val or None
        elif flag == "--provenance":
            provenance = val
        else:
            sys.exit(f"unknown option {flag}")
    from huggingface_hub import __version__ as hv, snapshot_download
    os.makedirs(hub, exist_ok=True)
    print(f"[fetch] hub {hub}; huggingface_hub {hv}" + (f"; user model {user_model}" if user_model else ""), flush=True)
    served, prov = None, None
    for item in rest:
        repo, rev, kind = item.rsplit(":", 2)
        t = time.time()
        is_served = kind == "model" and served is None
        served = served or (repo if kind == "model" else None)
        if pinned(hub, repo, rev, kind) and not (is_served and user_model and
                                                  os.path.islink(os.path.join(entry_dir(hub, repo, kind), "snapshots", rev))):
            how = "already pinned"
            if is_served:
                snap = os.path.join(entry_dir(hub, repo, kind), "snapshots", open(os.path.join(entry_dir(hub, repo, kind), "refs", "main")).read().strip())
                prov = ({"weights_source": "user_path", "path": os.path.realpath(snap)} if os.path.islink(snap)
                        else {"weights_source": "downloaded", "path": snap})
                if user_model and not os.path.islink(snap):
                    prov["note"] = f"the pinned snapshot was already complete on this node; --user-model {user_model} was not needed"
        elif is_served and user_model:
            prov = link_user_model(hub, repo, rev, user_model)
            how = f"weights from {prov['path']} ({prov['weights_source']})"
        elif kind == "dataset" and copy_shared(hub, shared, repo, rev, kind):
            how = "copied from the shared hub"
        else:
            p = snapshot_download(repo, revision=rev, repo_type=kind, cache_dir=hub, max_workers=16)
            write_ref(hub, repo, kind, os.path.basename(p))
            how = "downloaded"
            if is_served:
                prov = {"weights_source": "downloaded", "path": p}
        full = open(os.path.join(entry_dir(hub, repo, kind), "refs", "main")).read().strip()
        path = os.path.join(entry_dir(hub, repo, kind), "snapshots", full)
        if not os.path.isdir(path):
            sys.exit(f"[fetch] {repo}: snapshot {full} not on disk")
        n = check_shards(path, repo) if kind == "model" else 0
        print(f"[fetch] ok {repo}@{full[:10]} ({kind}, {how}): {n} indexed shards, {time.time() - t:.0f} s", flush=True)
        if is_served and prov is not None:
            prov.update({"model_id": repo, "model_rev": full})
    if provenance and prov is not None:
        with open(provenance, "w") as f:
            json.dump(prov, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1:])
