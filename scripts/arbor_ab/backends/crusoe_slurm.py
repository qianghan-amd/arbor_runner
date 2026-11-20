"""Backend `crusoe_slurm`: Crusoe MI355X through its Slurm-compatible scheduler (Spur), over short ssh calls only.

Nothing long-lived runs on the login node (design.md section 6): each call is one ssh session that returns.
Config: local.yaml `ssh.login_host`, `remote.*`, `slurm.qos/account`, `arbor.gateway_env`; config/cluster/crusoe.yaml.
Facts this module relies on (verified 2026-11-19): the scheduler is Spur 0.15.0, whose sacct has no -X/-P; login shells
are tcsh, so every multi-step command goes as a script on stdin (remote.run_bash / run_python); /shared_nfs is the same
path on the login node and the compute nodes; root in the containers needs the work dirs world-traversable (755).
"""
import hashlib
import io
import subprocess
import tarfile

from .. import remote
from . import RESULT_FILES, Backend

STATE_MAP = {"PENDING": "PENDING", "CONFIGURING": "PENDING", "REQUEUED": "PENDING", "RUNNING": "RUNNING",
             "COMPLETING": "RUNNING", "COMPLETED": "SUCCEEDED", "FAILED": "FAILED", "NODE_FAIL": "FAILED",
             "TIMEOUT": "FAILED", "OUT_OF_MEMORY": "FAILED", "BOOT_FAIL": "FAILED", "DEADLINE": "FAILED",
             "CANCELLED": "CANCELLED", "PREEMPTED": "CANCELLED"}

STATUS_PROGRAM = r'''
import os, subprocess
cells = PAYLOAD["cells"]
ids = [c["job_id"] for c in cells if c["job_id"]]
def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout
sq = {}
if ids:
    for line in run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%M|%l|%N|%R"]).splitlines():
        i, st, el, lim, node, reason = line.split("|", 5)
        sq[i] = {"raw": st, "elapsed": el, "time_limit": lim, "node": node, "reason": reason}
    # Spur's sacct has no -X/-P: fixed fields, whitespace-separated
    for line in run(["sacct", "-n", "-j", ",".join(ids), "-o", "JobID,State,Elapsed,Timelimit,NodeList,ExitCode"]).splitlines():
        f = line.split()
        if len(f) >= 2 and f[0] in ids:
            sq.setdefault(f[0], {"raw": f[1], "elapsed": f[2] if len(f) > 2 else "", "time_limit": f[3] if len(f) > 3 else "",
                                 "node": f[4] if len(f) > 4 else "", "exit_code": f[5] if len(f) > 5 else ""})
out = []
for c in cells:
    d = c["remote_dir"]
    state = open(f"{d}/state").read().strip() if os.path.exists(f"{d}/state") else None
    done = sorted(f[:-5] for f in os.listdir(f"{d}/phases") if f.endswith(".done")) if os.path.isdir(f"{d}/phases") else []
    tail = open(f"{d}/logs/runner.log", errors="replace").read().splitlines()[-5:] if os.path.exists(f"{d}/logs/runner.log") else []
    out.append({"job": sq.get(c["job_id"] or "", {}), "runner_state": state, "phases_done": done, "log_tail": tail})
print(json.dumps(out))
'''

LAYOUT_PROGRAM = r'''
import os, shutil, stat
def mode(p):
    return oct(stat.S_IMODE(os.stat(p).st_mode)) if os.path.exists(p) else None
wr, gw = PAYLOAD["work_root"], PAYLOAD["gateway_env"]
print(json.dumps({"work_root": mode(wr), "runs": mode(wr + "/runs"), "secrets": mode(wr + "/secrets"),
                  "gateway_env": mode(gw) if gw else "not configured",
                  "commands": {c: bool(shutil.which(c)) for c in ("sbatch", "squeue", "sacct", "scontrol", "scancel")}}))
'''

KEY_PROGRAM = r'''
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

GATEWAY_PROGRAM = r'''
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

INFX_PROGRAM = r'''
import subprocess
print(json.dumps({s: subprocess.run(["git", "-C", PAYLOAD["clone"], "cat-file", "-e", s + "^{commit}"]).returncode == 0
                  for s in PAYLOAD["shas"]}))
'''


def to_minutes(limit):
    """Slurm 'D-HH:MM:SS' / 'HH:MM:SS' / 'MM:SS' -> minutes."""
    days, _, rest = limit.partition("-") if "-" in limit else ("0", "", limit)
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days) * 1440 + parts[0] * 60 + parts[1]


def slurm_time(minutes):
    d, m = divmod(minutes, 1440)
    return f"{d}-{m // 60:02d}:{m % 60:02d}:00"


def gateway_env_text(gw, key):
    m = gw["models"]
    return ("# arbor-ab-system: LLM gateway for Arbor's agents (written by scripts/setup_remote.py). Never print or copy.\n"
            f"ANTHROPIC_API_KEY={gw['api_key_placeholder']}\n"
            f"ANTHROPIC_BASE_URL={gw['base_url']}\n"
            f"ANTHROPIC_CUSTOM_HEADERS=\"{gw['key_header']}: {key}\"\n"
            f"ANTHROPIC_DEFAULT_HAIKU_MODEL={m['haiku']}\nANTHROPIC_DEFAULT_OPUS_MODEL={m['opus']}\n"
            f"ANTHROPIC_DEFAULT_SONNET_MODEL={m['sonnet']}\nANTHROPIC_MODEL={m['default']}\n")


class CrusoeSlurm(Backend):
    name = "crusoe_slurm"

    def __init__(self, local, cluster):
        self.local, self.cluster = local, cluster
        self.host = local["ssh"]["login_host"]
        self.work_root = local["remote"]["work_root"]
        slurm = local.get("slurm") or {}
        self.partition = cluster["partition"]
        self.qos = slurm.get("qos") or cluster.get("qos_default")
        self.account = slurm.get("account") or cluster.get("account_default")

    def _ssh(self, *argv, data=None, timeout=120):
        """One short ssh call with plain arguments (no shell syntax: the login shell is tcsh)."""
        return subprocess.run(["ssh", "-o", "BatchMode=yes", self.host, *argv], input=data, capture_output=True,
                              timeout=timeout)

    def site(self):
        r = self.local["remote"]
        return {"node_local_root": r["node_local_root"], "shared_hf_hub": r["shared_hf_hub"],
                "infx_clone": r["infx_clone"], "gateway_env": (self.local.get("arbor") or {}).get("gateway_env")}

    def run_dir(self, run_id):
        return f"{self.work_root}/runs/{run_id}"

    def job_script(self, cell_dir, job_name, minutes, exclude=(), options=None):
        """options (request backend_options): {qos, account} override local.yaml slurm.qos/account for this run."""
        opts = options or {}
        unknown = set(opts) - {"qos", "account"}
        if unknown:
            raise ValueError(f"crusoe_slurm backend_options: unknown {sorted(unknown)} (known: qos, account)")
        qos, account = opts.get("qos", self.qos), opts.get("account", self.account)
        cap = (self.cluster.get("qos_max_wall_minutes") or {}).get(qos)
        if cap and minutes > cap:   # Spur keeps such a job pending forever (QOSMaxWallDurationPerJobLimit)
            raise ValueError(f"the job needs {minutes} min but {qos} allows {cap} min: shorten campaign_minutes or use "
                             "another QOS (backend_options)")
        lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}", f"#SBATCH --partition={self.partition}",
                 f"#SBATCH --qos={qos}"]
        if account:
            lines.append(f"#SBATCH --account={account}")
        if exclude:
            lines.append(f"#SBATCH --exclude={','.join(exclude)}")
        lines += ["#SBATCH --nodes=1", "#SBATCH --exclusive", f"#SBATCH --time={slurm_time(minutes)}",
                  f"#SBATCH --output={cell_dir}/logs/slurm-%j.out",
                  # Spur does not merge stderr into --output: without --error it goes to $HOME/spur-<job>.out
                  f"#SBATCH --error={cell_dir}/logs/slurm-%j.err",
                  "# One PR-test cell: the frozen runner does every phase; nothing else runs on the node.",
                  f"exec bash {cell_dir}/runner.sh {cell_dir} 2>&1", ""]
        return "job.sbatch", "\n".join(lines)

    def stage(self, run_id, local_run_dir, cells, extra_files):
        rd = self.run_dir(run_id)
        self._ssh("mkdir", "-p", rd).check_returncode()
        subprocess.run(["scp", "-q", "-r", *map(str, extra_files), *[str(local_run_dir / c["cell"]) for c in cells],
                        f"{self.host}:{rd}/"], check=True)
        # what landed must be LF before anything is queued (Windows tooling added CR once)
        check = "".join(f"if grep -l $'\\r' {c['remote_dir']}/*.sh {c['remote_dir']}/*.env {c['remote_dir']}/job.sbatch; "
                        f"then echo 'CRLF in uploaded files' >&2; exit 3; fi\n" for c in cells)
        p = remote.run_bash(self.host, check)
        if p.returncode:
            raise RuntimeError(p.stderr.decode(errors="replace"))

    def submit(self, cells):
        script = "".join(f"sbatch --parsable {c['remote_dir']}/job.sbatch || echo FAILED\n" for c in cells)
        p = remote.run_bash(self.host, script)
        out = p.stdout.decode(errors="replace").split()
        ids = [None if (t == "FAILED") else t.split(";")[0] for t in out][:len(cells)]
        ids += [None] * (len(cells) - len(ids))
        return ids, p.stderr.decode(errors="replace")[-2000:] or None

    def status(self, cells):
        raw = remote.run_python(self.host, STATUS_PROGRAM, {"cells": cells})
        out = []
        for r in raw:
            job = r["job"]
            word = (job.get("raw") or "").split("+")[0].upper()
            state = "UNKNOWN" if not job else STATE_MAP.get(word.split()[0] if word else "", "UNKNOWN")
            out.append({"state": state, "raw": job.get("raw"), "node": job.get("node", ""), "elapsed": job.get("elapsed", ""),
                        "time_limit": job.get("time_limit", ""), "reason": job.get("reason", ""),
                        "runner_state": r["runner_state"], "phases_done": r["phases_done"], "log_tail": r["log_tail"]})
        return out

    def fetch(self, cell, local_dir):
        pats = " ".join(RESULT_FILES)
        script = (f"cd {cell['remote_dir']} 2>/dev/null || exit 0\nshopt -s nullglob\n"
                  f"f=$(ls -d {pats} 2>/dev/null)\n[ -n \"$f\" ] && tar -cf - $f\n")
        p = remote.run_bash(self.host, script, timeout=300)
        if not p.stdout:
            return 0
        n = 0
        with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:
            for m in t.getmembers():
                if m.isfile() and not m.name.startswith(("/", "..")) and ".." not in m.name.split("/"):
                    dest = local_dir / m.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(t.extractfile(m).read())
                    n += 1
        return n

    def cancel(self, cell):
        if not cell.get("job_id"):
            return "no job id"
        p = self._ssh("scancel", str(cell["job_id"]))
        return "cancelled" if p.returncode == 0 else f"scancel failed: {p.stderr.decode(errors='replace').strip()[-200:]}"

    def extend(self, cell, status):
        """A RUNNING job below the wall time its cell needs gets `scontrol update TimeLimit`."""
        if status["state"] != "RUNNING" or not status.get("time_limit"):
            return None
        if to_minutes(status["time_limit"]) >= cell["job_minutes"]:
            return None
        lim = slurm_time(cell["job_minutes"])
        p = self._ssh("scontrol", "update", f"JobId={cell['job_id']}", f"TimeLimit={lim}")
        return lim if p.returncode == 0 else f"failed: {p.stderr.decode(errors='replace').strip()[-200:]}"

    def check(self):
        res = []
        p = self._ssh("hostname", timeout=60)
        res.append(("ssh login host", p.returncode == 0,
                    f"{self.host} -> {p.stdout.decode().strip()}" if p.returncode == 0 else p.stderr.decode(errors="replace")[-300:]))
        if p.returncode:
            return res
        r = remote.run_python(self.host, LAYOUT_PROGRAM, {"work_root": self.work_root, "gateway_env": self.site()["gateway_env"]})
        want = {"work_root": "0o755", "runs": "0o755", "secrets": "0o700"}
        bad = [f"{k} is {r[k]} (want {v})" for k, v in want.items() if r[k] != v]
        if r["gateway_env"] not in ("0o600", "not configured"):
            bad.append(f"gateway_env is {r['gateway_env']} (want 0o600)")
        bad += [f"{c} not on PATH" for c, ok in r["commands"].items() if not ok]
        res.append(("remote layout", not bad, "; ".join(bad) + " -> run scripts/setup_remote.py" if bad else
                    f"work_root 755, runs 755, secrets 700, gateway_env {r['gateway_env']}, scheduler commands present; "
                    f"qos {self.qos}, account {self.account}"))
        return res

    def setup(self, key=None, test_gateway=False, infx_shas=()):
        res, wr = [], self.work_root
        gw_path = f"{wr}/secrets/gateway.env"
        p = remote.run_bash(self.host, f"mkdir -p {wr}/runs {wr}/secrets && chmod 755 {wr} {wr}/runs && chmod 700 {wr}/secrets\n")
        res.append(("work dirs", p.returncode == 0, f"{wr} 755, runs 755, secrets 700" if p.returncode == 0
                    else p.stderr.decode(errors="replace")[-300:]))
        if key:
            # the key goes on ssh stdin into a 600 file: never printed, never an argument, never in the repo
            data = gateway_env_text(self.cluster["llm_gateway"], key).encode()
            p = self._ssh("umask 077 && cat > " + gw_path + " && chmod 600 " + gw_path, data=data)
            v = remote.run_python(self.host, KEY_PROGRAM, {"path": gw_path}) if p.returncode == 0 else {}
            ok = v.get("key_sha256_12") == hashlib.sha256(key.encode()).hexdigest()[:12] and v.get("mode") == "0o600"
            res.append(("gateway env", ok, f"{gw_path}: mode {v.get('mode')}, key verified by sha256: {'yes' if ok else 'NO'}"))
        if infx_shas:
            found = remote.run_python(self.host, INFX_PROGRAM, {"clone": self.local["remote"]["infx_clone"], "shas": list(infx_shas)})
            res.append(("InferenceX commits", all(found.values()),
                        ", ".join(f"{s[:10]} {'ok' if ok else 'MISSING'}" for s, ok in found.items())))
        if test_gateway:
            r = remote.run_python(self.host, GATEWAY_PROGRAM, {"path": gw_path})
            res.append(("gateway call", r["http"] == 200, f"from {self.host}: HTTP {r['http']}, model {r['model']}"))
        res.append(("next", True, f"local.yaml arbor.gateway_env: {gw_path}"))
        return res


BACKEND = CrusoeSlurm
