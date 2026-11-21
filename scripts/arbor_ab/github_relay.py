"""GitHub API through a relay host that holds gh credentials (INTENT.md Q2).

The token never leaves the relay host and is never printed:
- it is read there from ~/.config/gh/hosts.yml with awk into a shell variable,
- handed to python3 through the environment of that one process,
- and given to curl on stdin (`-H @-`), so it is not in any process's argv (visible in `ps` on a shared node).
curl -L does not forward the Authorization header when a redirect changes host (artifact downloads go to blob storage).
All requests run in one ssh call; the result is one JSON document on stdout.
"""
import json
import subprocess

REMOTE = r'''
set -u
T=$(awk '/github.com:/{f=1} f && /oauth_token:/{print $2; exit}' ~/.config/gh/hosts.yml 2>/dev/null)
if [ -z "$T" ]; then echo '{"error": "no gh oauth_token on relay host"}'; exit 0; fi
GH_RELAY_TOKEN="$T" python3 - <<'PY'
import io, json, os, subprocess, tempfile, zipfile
TOKEN = os.environ.pop("GH_RELAY_TOKEN")
API = "https://api.github.com"
REQS = json.loads(r"""__REQUESTS__""")

def curl(url, out=None):
    args = ["curl", "-sS", "-L", "--max-time", "300", "-H", "@-", "-H", "Accept: application/vnd.github+json",
            "-w", "%{http_code}", "-o", out or "-", url]
    p = subprocess.run(args, input=("Authorization: token " + TOKEN + "\n").encode(), capture_output=True)
    if out:
        return int(p.stdout.decode() or 0), None
    body = p.stdout[:-3]
    return int(p.stdout[-3:].decode() or 0), body

def get_json(path):
    code, body = curl(API + path)
    return code, (json.loads(body) if code == 200 else None)

res = []
for r in REQS:
    try:
        if r["op"] == "get":
            code, data = get_json(r["path"])
            res.append({"http": code, "data": data})
        elif r["op"] == "artifacts":
            arts, page, code = [], 1, 0
            while True:
                code, data = get_json(f"/repos/{r['repo']}/actions/runs/{r['run_id']}/artifacts?per_page=100&page={page}")
                if code != 200:
                    break
                arts += [{k: a[k] for k in ("id", "name", "size_in_bytes", "expired", "created_at", "expires_at")}
                         for a in data["artifacts"]]
                if len(arts) >= data["total_count"] or not data["artifacts"]:
                    break
                page += 1
            res.append({"http": code, "data": arts})
        elif r["op"] == "artifact_json":
            with tempfile.TemporaryDirectory() as d:
                zp = os.path.join(d, "a.zip")
                code, _ = curl(f"{API}/repos/{r['repo']}/actions/artifacts/{r['artifact_id']}/zip", out=zp)
                members = {}
                if code == 200:
                    z = zipfile.ZipFile(zp)
                    for n in z.namelist():
                        if n.endswith(".json") and z.getinfo(n).file_size < 4_000_000:
                            members[n] = json.loads(z.read(n))
            res.append({"http": code, "data": members})
        elif r["op"] == "artifact_text":
            # only the named members (matched by base name); a big one returns only its lines matching r["grep"][name]
            import re
            out = {}
            with tempfile.TemporaryDirectory() as d:
                zp = os.path.join(d, "a.zip")
                code, _ = curl(f"{API}/repos/{r['repo']}/actions/artifacts/{r['artifact_id']}/zip", out=zp)
                if code == 200:
                    z = zipfile.ZipFile(zp)
                    for n in z.namelist():
                        base = os.path.basename(n)
                        if base not in r["names"]:
                            continue
                        text = z.read(n).decode("utf-8", "replace")
                        pat = (r.get("grep") or {}).get(base)
                        out[base] = "\n".join(l for l in text.splitlines() if re.search(pat, l))[:200000] if pat else text[:200000]
            res.append({"http": code, "data": out})
        elif r["op"] == "artifact_tar":
            # a .tar.gz member of the zip (srtctl CI runs: srt-single-node-logs.tar.gz): its member list, and the text
            # of members whose path matches r["pattern"] (only lines matching r["grep"] when given)
            import re, tarfile
            out = {"members": [], "text": {}}
            with tempfile.TemporaryDirectory() as d:
                zp = os.path.join(d, "a.zip")
                code, _ = curl(f"{API}/repos/{r['repo']}/actions/artifacts/{r['artifact_id']}/zip", out=zp)
                if code == 200:
                    z = zipfile.ZipFile(zp)
                    n = next((x for x in z.namelist() if os.path.basename(x) == r["member"]), None)
                    if n:
                        t = tarfile.open(fileobj=io.BytesIO(z.read(n)), mode="r:*")
                        for m in t.getmembers():
                            out["members"].append([m.name, m.size])
                            if m.isfile() and r.get("pattern") and re.search(r["pattern"], m.name):
                                text = t.extractfile(m).read().decode("utf-8", "replace")
                                if r.get("grep"):
                                    text = "\n".join(l for l in text.splitlines() if re.search(r["grep"], l))
                                out["text"][m.name] = text[:200000]
            res.append({"http": code, "data": out})
        elif r["op"] == "artifact_names":
            with tempfile.TemporaryDirectory() as d:
                zp = os.path.join(d, "a.zip")
                code, _ = curl(f"{API}/repos/{r['repo']}/actions/artifacts/{r['artifact_id']}/zip", out=zp)
                names = [[i.filename, i.file_size] for i in zipfile.ZipFile(zp).infolist()] if code == 200 else []
            res.append({"http": code, "data": names})
        else:
            res.append({"http": 0, "error": "unknown op"})
    except Exception as e:  # report, never raise: one bad request must not hide the others
        res.append({"http": 0, "error": f"{type(e).__name__}: {e}"})
print(json.dumps({"results": res}))
PY
'''


class RelayError(RuntimeError):
    pass


def call(host, requests, timeout=900):
    """Run GitHub API requests on the relay host. Returns one result dict per request ({http, data} or {error})."""
    script = REMOTE.replace("__REQUESTS__", json.dumps(requests))
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash", "-s"], input=script.encode(), capture_output=True,
                       timeout=timeout)
    if p.returncode != 0:
        raise RelayError(f"ssh {host} rc={p.returncode}: {p.stderr.decode(errors='replace')[-1500:]}")
    out = json.loads(p.stdout)
    if "error" in out:
        raise RelayError(out["error"])
    return out["results"]


def get(path):
    return {"op": "get", "path": path}


def artifacts(repo, run_id):
    return {"op": "artifacts", "repo": repo, "run_id": str(run_id)}


def artifact_json(repo, artifact_id):
    return {"op": "artifact_json", "repo": repo, "artifact_id": int(artifact_id)}


def artifact_tar(repo, artifact_id, member, pattern=None, grep=None):
    """Member list of a .tar.gz inside an artifact zip, plus the text of its members matching `pattern`."""
    return {"op": "artifact_tar", "repo": repo, "artifact_id": int(artifact_id), "member": member, "pattern": pattern,
            "grep": grep}


def artifact_names(repo, artifact_id):
    """[[member name, size]] of an artifact zip."""
    return {"op": "artifact_names", "repo": repo, "artifact_id": int(artifact_id)}


def artifact_text(repo, artifact_id, names, grep=None):
    """Text of the named members of an artifact zip; for a member in `grep`, only its lines matching that regex."""
    return {"op": "artifact_text", "repo": repo, "artifact_id": int(artifact_id), "names": list(names), "grep": grep or {}}
