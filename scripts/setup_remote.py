"""Cluster-side setup, step 2 of README "Setup". Idempotent; short ssh calls only.

Usage: python scripts/setup_remote.py [--key-file <file>] [--test-gateway]
  - creates <work_root> and <work_root>/runs (755: root in the containers must traverse them, NFS root-squash) and
    <work_root>/secrets (700)
  - --key-file: writes <work_root>/secrets/gateway.env (600) for Arbor's agents, in the format of config/cluster
    llm_gateway, with the subscription key read from <file> on this workstation. The key is streamed over ssh stdin:
    never printed, never in a process argument list, never in the repo. The upload is verified by sha256.
  - checks the InferenceX clone holds every runnable baseline's bench-client commit
  - --test-gateway: one minimal request through the gateway from the login host (prints HTTP status and model only)
Then set local.yaml arbor.gateway_env to the printed path and run scripts/check_setup.py.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from arbor_ab import baseline, config, remote

VERIFY = r'''
import hashlib, os, stat
p = PAYLOAD["path"]; out = {"exists": os.path.exists(p)}
if out["exists"]:
    out["mode"] = oct(stat.S_IMODE(os.stat(p).st_mode))
    val = None
    for line in open(p):
        if line.startswith("ANTHROPIC_CUSTOM_HEADERS="):
            val = line.split(": ", 1)[1].rstrip().rstrip('"')
    out["key_sha256_12"] = hashlib.sha256(val.encode()).hexdigest()[:12] if val else None
print(json.dumps(out))
'''
CHECK_INFX = r'''
import subprocess
print(json.dumps({s: subprocess.run(["git", "-C", PAYLOAD["clone"], "cat-file", "-e", s + "^{commit}"]).returncode == 0
                  for s in PAYLOAD["shas"]}))
'''
TEST_GATEWAY = r'''
import urllib.request, urllib.error
env = {}
for line in open(PAYLOAD["path"]):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); env[k] = v.strip('"')
name, val = env["ANTHROPIC_CUSTOM_HEADERS"].split(": ", 1)
req = urllib.request.Request(env["ANTHROPIC_BASE_URL"].rstrip("/") + "/v1/messages", method="POST",
    data=json.dumps({"model": env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "max_tokens": 5,
                     "messages": [{"role": "user", "content": "Say ok."}]}).encode(),
    headers={"content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": env["ANTHROPIC_API_KEY"], name: val})
try:
    r = urllib.request.urlopen(req, timeout=60); body = json.load(r)
    print(json.dumps({"http": r.status, "model": body.get("model")}))
except urllib.error.HTTPError as e:
    print(json.dumps({"http": e.code, "model": None}))
'''


def ssh(host, cmd, data=None):
    """One short ssh call. `cmd` must be valid for the login shell (tcsh): no $(...), no bash-only syntax."""
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], input=data, capture_output=True, timeout=120)


def gateway_env_text(gw, key):
    m = gw["models"]
    return ("# arbor-ab-system: LLM gateway for Arbor's agents (written by scripts/setup_remote.py). Never print or copy.\n"
            f"ANTHROPIC_API_KEY={gw['api_key_placeholder']}\n"
            f"ANTHROPIC_BASE_URL={gw['base_url']}\n"
            f"ANTHROPIC_CUSTOM_HEADERS=\"{gw['key_header']}: {key}\"\n"
            f"ANTHROPIC_DEFAULT_HAIKU_MODEL={m['haiku']}\nANTHROPIC_DEFAULT_OPUS_MODEL={m['opus']}\n"
            f"ANTHROPIC_DEFAULT_SONNET_MODEL={m['sonnet']}\nANTHROPIC_MODEL={m['default']}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key-file", type=Path)
    ap.add_argument("--test-gateway", action="store_true")
    args = ap.parse_args()
    local, cluster = config.load_local(), config.load_cluster()
    host, wr = local["ssh"]["login_host"], local["remote"]["work_root"]
    gw_path = f"{wr}/secrets/gateway.env"

    p = ssh(host, f"mkdir -p {wr}/runs {wr}/secrets && chmod 755 {wr} {wr}/runs && chmod 700 {wr}/secrets")
    print(f"[setup] {wr}: {'ok (755, runs 755, secrets 700)' if p.returncode == 0 else 'FAILED ' + p.stderr.decode()[-300:]}")
    if p.returncode:
        return 1

    if args.key_file:
        key = args.key_file.read_text(encoding="utf-8").strip()
        if not key or any(c.isspace() or not c.isprintable() for c in key):
            sys.exit(f"[setup] {args.key_file}: expected one key with no whitespace")
        data = gateway_env_text(cluster["llm_gateway"], key).encode()
        p = ssh(host, f"umask 077 && cat > {gw_path} && chmod 600 {gw_path}", data=data)
        if p.returncode:
            sys.exit(f"[setup] writing {gw_path} failed: {p.stderr.decode()[-300:]}")
        v = remote.run_python(host, VERIFY, {"path": gw_path})
        ok = v.get("key_sha256_12") == hashlib.sha256(key.encode()).hexdigest()[:12] and v.get("mode") == "0o600"
        print(f"[setup] {gw_path}: written, mode {v.get('mode')}, key verified by sha256: {'yes' if ok else 'NO'}")
        if not ok:
            return 1

    shas = sorted({r["measured"]["bench_client_commit"] for m, t in config.cells(config.load_matrix())
                   for r in [baseline.load_latest(m, t)] if r and r.get("launch") and r["measured"]["bench_client_commit"]})
    found = remote.run_python(host, CHECK_INFX, {"clone": local["remote"]["infx_clone"], "shas": shas})
    print(f"[setup] InferenceX clone {local['remote']['infx_clone']}: "
          + ", ".join(f"{s[:10]} {'ok' if ok else 'MISSING'}" for s, ok in found.items()))

    if args.test_gateway:
        r = remote.run_python(host, TEST_GATEWAY, {"path": gw_path})
        print(f"[setup] gateway test from {host}: HTTP {r['http']}, model {r['model']}")
    print(f"[setup] set local.yaml arbor.gateway_env: {gw_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
