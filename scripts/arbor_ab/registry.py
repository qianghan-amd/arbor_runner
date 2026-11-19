"""Image availability without pulling: anonymous Docker Hub manifest HEAD by digest and by tag (design.md section 3.5).

Known case: Kimi K3's tag was deleted upstream while its digest still resolves (verified 2026-11-19: tag 404, digest 200).
"""
from . import remote

PROGRAM = r'''
import urllib.error, urllib.request
ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json", "application/vnd.oci.image.manifest.v1+json"])
tokens, out = {}, []
def head(repo, ref):
    if repo not in tokens:
        u = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        tokens[repo] = json.load(urllib.request.urlopen(u, timeout=30))["token"]
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{ref}", method="HEAD",
                                 headers={"Authorization": "Bearer " + tokens[repo], "Accept": ACCEPT})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.headers.get("Docker-Content-Digest")
    except urllib.error.HTTPError as e:
        return e.code, None
for repo, tag, digest in PAYLOAD:
    try:
        t = head(repo, tag) if tag else (None, None)
        d = head(repo, digest) if digest else (None, None)
        out.append({"tag_http": t[0], "tag_digest": t[1], "digest_http": d[0]})
    except Exception as e:
        out.append({"error": f"{type(e).__name__}: {e}"})
print(json.dumps(out))
'''


def parse_ref(image):
    """'repo:tag[@sha256:x]' -> (repo, tag, digest). Docker Hub only (all current images are on Docker Hub)."""
    digest = None
    if "@" in image:
        image, digest = image.split("@", 1)
    repo, _, tag = image.rpartition(":") if ":" in image.split("/")[-1] else (image, "", "latest")
    return repo, tag, digest


def check(host, images):
    """images: [(image_ref, recorded_digest or None)] -> one result dict each."""
    payload = []
    for ref, digest in images:
        repo, tag, d = parse_ref(ref)
        payload.append([repo, tag, digest or d])
    return remote.run_python(host, PROGRAM, payload)


def summarize(result, recorded_digest):
    """(image_available, detail). Available = the pinned digest resolves (or, with no digest pinned, the tag)."""
    if "error" in result:
        return None, "registry check error: " + result["error"]
    tag_ok, dig_ok = result["tag_http"] == 200, result["digest_http"] == 200
    parts = [f"tag {'resolves' if tag_ok else 'MISSING (http %s)' % result['tag_http']}"]
    if recorded_digest:
        parts.append(f"digest {'resolves' if dig_ok else 'MISSING (http %s)' % result['digest_http']}")
        if tag_ok and result["tag_digest"] and result["tag_digest"] != recorded_digest:
            parts.append(f"TAG MOVED: now {result['tag_digest'][:19]} != pinned {recorded_digest[:19]}")
        return dig_ok, "; ".join(parts)
    parts.append("no digest pinned")
    return tag_ok, "; ".join(parts)
