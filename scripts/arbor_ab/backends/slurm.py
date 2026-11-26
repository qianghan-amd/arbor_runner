"""Backend `slurm`: any Slurm(-compatible) cluster, over short ssh calls to its login node only.

Nothing long-lived runs on the login node (design.md section 6): each call is one ssh session that returns.
Config: the site's local.yaml keys `ssh.login_host`, `remote.*`, `slurm.{partition,qos,account}`, `arbor.gateway_env`
(backends.for_site), and config/cluster/<cluster>.yaml (partition, QOS defaults and wall limits, GPU type).
Written to the common subset of Slurm 25 and Crusoe's Spur 0.15 (verified 2026-11-19/21): sacct without -X/-P (Spur
has neither), stderr sent with --error (Spur does not merge it into --output); every multi-step command goes as a
script on stdin (remote.run_bash / run_python), since a login shell may be tcsh (Crusoe); the work root is the same path
on the login node and the compute nodes; root in the containers needs the work dirs world-traversable (755).
Clusters: crusoe (Spur, MI355X), meta64 (Slurm 25.05, MI350X).
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
import datetime as dt, os, subprocess
cells = PAYLOAD["cells"]
ids = [c["job_id"] for c in cells if c["job_id"] and not c["job_id"].startswith("hold:")]
ids += sorted({c["job_id"].split(":")[1] for c in cells if (c["job_id"] or "").startswith("hold:")})
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
def read(p):
    return open(p).read().strip() if os.path.exists(p) else None
def hold_job(c, d, state):
    """A hold-mode cell's job record, from its ctrl file and its own markers (it has no Slurm job)."""
    _, j, node = c["job_id"].split(":", 2)
    ctrl, start, rc = c.get("hold_ctrl") or "", read(f"{d}/hold_start"), read(f"{d}/hold_exit")
    if os.path.exists(ctrl):
        raw = "PENDING"                                     # the agent has not picked it up yet
    elif os.path.exists(ctrl + ".cancelled"):
        raw = "CANCELLED"
    elif rc is not None:
        raw = "COMPLETED" if rc == "0" and state == "done" else "FAILED"
    elif start and sq.get(j, {}).get("raw", "").startswith("RUNNING"):
        raw = "RUNNING"
    else:                                                   # the hold job ended under the cell (or never started it)
        raw = "FAILED" if start else "PENDING"
    el = ""
    if start:
        s = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        end = dt.datetime.fromtimestamp(os.path.getmtime(f"{d}/hold_exit"), dt.timezone.utc) if rc is not None else dt.datetime.now(dt.timezone.utc)
        el = str(end - s).split(".")[0]
    return {"raw": f"{raw} (hold {j})", "elapsed": el, "time_limit": "", "node": read(f"{d}/hold_node") or node,
            "reason": "", "exit_code": rc or ""}
out = []
for c in cells:
    d = c["remote_dir"]
    state = open(f"{d}/state").read().strip() if os.path.exists(f"{d}/state") else None
    done = sorted(f[:-5] for f in os.listdir(f"{d}/phases") if f.endswith(".done")) if os.path.isdir(f"{d}/phases") else []
    tail = open(f"{d}/logs/runner.log", errors="replace").read().splitlines()[-5:] if os.path.exists(f"{d}/logs/runner.log") else []
    job = hold_job(c, d, state) if (c["job_id"] or "").startswith("hold:") else sq.get(c["job_id"] or "", {})
    out.append({"job": job, "runner_state": state, "phases_done": done, "log_tail": tail})
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

# Fetch missing InferenceX commits (and the commits of their submodules, e.g. utils/aiperf: the runner extracts each
# gitlink at its commit) into the clone. `git fetch origin <sha>` only adds objects: no checkout, no branch moves.
INFX_FETCH_PROGRAM = r'''
import subprocess
def has(repo, s):
    return subprocess.run(["git", "-C", repo, "cat-file", "-e", s + "^{commit}"], capture_output=True).returncode == 0
import os, re
def fetch(repo, s):
    if not has(repo, s):
        subprocess.run(["git", "-C", repo, "fetch", "-q", "origin", s], capture_output=True, timeout=900)
    return has(repo, s)
def submodule_url(clone, s, path):
    text = subprocess.run(["git", "-C", clone, "show", f"{s}:.gitmodules"], capture_output=True, text=True).stdout
    for block in re.split(r"\n(?=\[submodule )", text):
        if re.search(rf"^\s*path\s*=\s*{re.escape(path)}\s*$", block, re.M):
            m = re.search(r"^\s*url\s*=\s*(\S+)", block, re.M)
            return m.group(1) if m else None
out = {}
for s in PAYLOAD["shas"]:
    ok, subs = fetch(PAYLOAD["clone"], s), {}
    if ok:
        tree = subprocess.run(["git", "-C", PAYLOAD["clone"], "ls-tree", "-r", s], capture_output=True, text=True).stdout
        for line in tree.splitlines():
            mode, kind, gsha, path = line.split(None, 3)
            if kind == "commit":
                sub = PAYLOAD["clone"] + "/" + path
                # a required submodule the clone never had (a fresh clone of a newer InferenceX, e.g. on meta64):
                # clone it from the URL .gitmodules gives at that commit, without a checkout
                url = submodule_url(PAYLOAD["clone"], s, path) if path in PAYLOAD["required"] else None
                if url and not os.path.isdir(sub + "/.git") and not os.path.isfile(sub + "/.git"):
                    os.makedirs(os.path.dirname(sub), exist_ok=True)
                    if os.path.isdir(sub) and not os.listdir(sub):
                        os.rmdir(sub)
                    subprocess.run(["git", "clone", "-q", "--no-checkout", url, sub], capture_output=True, timeout=900)
                subs[path] = fetch(sub, gsha)
    # only the submodules the runner requires decide (runner.sh REQUIRED_SUBMODULES); others are reported
    out[s] = {"ok": ok and all(subs.get(p, False) for p in PAYLOAD["required"] if p in subs), "submodules": subs,
              # the runner's GSM8K needs benchmarks/benchmark_lib.sh (the older `lib` layout); the infx layout lacks it
              "benchmark_lib": ok and subprocess.run(["git", "-C", PAYLOAD["clone"], "cat-file", "-e",
                                                      f"{s}:benchmarks/benchmark_lib.sh"], capture_output=True).returncode == 0}
print(json.dumps(out))
'''


SNAPSHOT_PROGRAM = r'''
import os
p = PAYLOAD["path"]
idx = [f for f in os.listdir(p) if f.endswith(".safetensors.index.json")] if os.path.isdir(p) else []
if not idx:
    print(json.dumps({"complete": False, "detail": "no snapshot or no *.safetensors.index.json"}))
else:
    want = sorted(set(json.load(open(os.path.join(p, idx[0])))["weight_map"].values()))
    missing = [f for f in want if not os.path.isfile(os.path.join(p, f))]
    print(json.dumps({"complete": not missing, "detail": f"{len(want) - len(missing)}/{len(want)} shards present"}))
'''


# Hold-job mode (backend_options {hold_job, nodelist}): a node of a running "hold" job of one's own, whose node agent
# runs each ctrl/node<i>/*.sh once on node i (skills/crusoe-arbor-run; Crusoe hold4-arbor 304634, 2026-11-23: the
# user's hyperloom nodes, shared with other sessions). The cell gets no Slurm job: its job.sbatch runs under the hold
# job, with the cell's wall time as a hard `timeout`; status comes from the ctrl file and the cell's own markers.
HOLD_SUBMIT_PROGRAM = r'''
import datetime as dt, getpass, os, re, subprocess
j, node = str(PAYLOAD["hold_job"]), PAYLOAD["node"]
info = subprocess.run(["scontrol", "show", "job", j], capture_output=True, text=True).stdout
def field(k):
    m = re.search(rf"(?<![A-Za-z]){k}=(\S+)", info)
    return m.group(1) if m else None
root, state, user, end = field("WorkDir"), field("JobState"), field("UserId") or "", field("EndTime")
problem = None
if state != "RUNNING":
    problem = f"hold job {j} is {state or 'not found'}"
elif not user.startswith(getpass.getuser()):
    problem = f"hold job {j} belongs to {user}, not {getpass.getuser()}"
elif not (root and os.path.isfile(f"{root}/NODES") and os.path.isdir(f"{root}/ctrl")):
    problem = f"hold job {j}: no NODES file and ctrl/ in its WorkDir {root}: not a node-agent hold job"
nodes = open(f"{root}/NODES").read().split() if not problem else []
if not problem and node not in nodes:
    problem = f"node {node} is not in hold job {j} ({' '.join(nodes)})"
left = None
if not problem and end:
    try:
        left = int((dt.datetime.fromisoformat(end) - dt.datetime.now()).total_seconds() // 60)
    except ValueError:
        left = None
need = max(c["minutes"] for c in PAYLOAD["cells"])
if not problem and left is not None and left < need:
    problem = f"hold job {j} ends in {left} min; the cell needs {need} min"
out = {"problem": problem, "root": root, "left": left, "ids": []}
if not problem:
    i = nodes.index(node)
    for c in PAYLOAD["cells"]:
        d = f"{root}/ctrl/node{i}"
        with open(f"{d}/{c['name']}.tmp", "w", newline="\n") as f:
            f.write(c["script"])
        os.replace(f"{d}/{c['name']}.tmp", f"{d}/{c['name']}.sh")   # the agent picks up *.sh only: complete or nothing
        out["ids"].append({"job_id": f"hold:{j}:{node}", "ctrl": f"{d}/{c['name']}.sh"})
print(json.dumps(out))
'''


def hold_script(cell_dir, minutes):
    """What the hold job's node agent runs for one cell: the cell's job.sbatch, bounded by its wall time."""
    return ("#!/bin/bash\n"
            f"# arbor-ci cell {cell_dir} (scripts/arbor_ab/backends/slurm.py hold mode)\n"
            f"d={cell_dir}\n"
            "mkdir -p $d/logs; date -u +%FT%TZ > $d/hold_start; hostname -s > $d/hold_node\n"
            f"timeout --signal=TERM --kill-after=15m {minutes}m bash $d/job.sbatch > $d/logs/hold.out 2>&1\n"
            "echo $? > $d/hold_exit\n")


def cancel_script(job_id, cell_dir):
    """A running job: SIGTERM to our runner on its node (a step in the job: srun --overlap; pkill on the runner's
    unique cell path), so its EXIT trap stops the server and removes the container; wait up to 90 s for the job to
    leave the queue; else (or a pending job) scancel. Spur's scancel kills with SIGKILL whatever -s says (Crusoe
    312964, 313020 with -s TERM -b, 2026-11-25: ExitCode 0:9), which skips the trap."""
    return (f"J={job_id}; [ -z \"$(squeue -h -j $J -o %T 2>/dev/null)\" ] && {{ echo already-ended; exit 0; }}\n"
            f"N=$(squeue -h -j $J -o %N 2>/dev/null)\n"
            f"if [ -n \"$N\" ] && [ \"$(squeue -h -j $J -o %T 2>/dev/null)\" = RUNNING ] && timeout 90 srun --jobid=$J "
            f"--overlap -w \"$N\" -N1 -n1 pkill -TERM -f 'runner[.]sh {cell_dir}$' >/dev/null 2>&1; then\n"
            f"  for i in $(seq 1 18); do [ -z \"$(squeue -h -j $J -o %T 2>/dev/null)\" ] && {{ echo ended-on-term; exit 0; }}; sleep 5; done\n"
            f"fi\n"
            f"scancel $J && echo killed\n")


def hold_options(opts):
    """(hold job id, node) from backend_options, or None."""
    if not (opts or {}).get("hold_job"):
        return None
    if not opts.get("nodelist") or "," in opts["nodelist"] or "[" in opts["nodelist"]:
        raise ValueError("backend_options hold_job needs nodelist: the one node of the hold job to run on")
    return str(opts["hold_job"]), opts["nodelist"]


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


class Slurm(Backend):
    name = "slurm"

    def __init__(self, local, cluster):
        self.local, self.cluster = local, cluster
        self.host = local["ssh"]["login_host"]
        self.work_root = local["remote"]["work_root"]
        slurm = local.get("slurm") or {}
        self.partition = slurm.get("partition") or cluster["partition"]
        self.qos = slurm.get("qos") or cluster.get("qos_default")          # None: the partition's default QOS
        self.account = slurm.get("account") or cluster.get("account_default")

    def _ssh(self, *argv, data=None, timeout=120):
        """One short ssh call with plain arguments (no shell syntax: the login shell is tcsh)."""
        return subprocess.run(["ssh", "-o", "BatchMode=yes", self.host, *argv], input=data, capture_output=True,
                              timeout=timeout)

    def site(self):
        r = self.local["remote"]
        return {"node_local_root": r["node_local_root"], "shared_hf_hub": r.get("shared_hf_hub"),
                "node_local_min_free_gb": self.cluster.get("node_local_min_free_gb"),
                "node_local_max_cache_gb": self.cluster.get("node_local_max_cache_gb"),
                "infx_clone": r["infx_clone"], "gateway_env": (self.local.get("arbor") or {}).get("gateway_env")}

    def run_dir(self, run_id):
        return f"{self.work_root}/runs/{run_id}"

    def ensure_bench_commits(self, shas):
        return remote.run_python(self.host, INFX_FETCH_PROGRAM, {"clone": self.local["remote"]["infx_clone"],
                                                                 "shas": list(shas), "required": ["utils/aiperf"]},
                                 timeout=1800)

    def snapshot_complete(self, path):
        """A shared-hub snapshot can hold config.json and the index but no shards (Kimi K3.1 Pro on Crusoe, 2026-11-22:
        96 of 96 missing): every shard the index names must exist."""
        r = remote.run_python(self.host, SNAPSHOT_PROGRAM, {"path": path}, timeout=300)
        return r["complete"], r["detail"]

    def files_exist(self, paths):
        """Shared storage looks the same from the login node and the compute nodes (/shared_nfs)."""
        return remote.run_python(self.host, "import os\nprint(json.dumps({p: os.path.exists(p) for p in PAYLOAD}))",
                                 list(paths), timeout=120)

    def job_script(self, cell_dir, job_name, minutes, exclude=(), options=None):
        """options (request backend_options): {partition, qos, account} override the site's for this run."""
        opts = options or {}
        unknown = set(opts) - {"partition", "qos", "account", "nodelist", "hold_job"}
        if unknown:
            raise ValueError(f"slurm backend_options: unknown {sorted(unknown)} (known: partition, qos, account, nodelist, "
                             "hold_job)")
        hold_options(opts)   # hold_job needs its one node
        part = opts.get("partition", self.partition)
        qos, account = opts.get("qos", self.qos), opts.get("account", self.account)
        # a longer --time than the QOS / partition allows pends forever (QOSMaxWallDurationPerJobLimit, PartitionTimeLimit)
        for what, cap in ((f"QOS {qos}", (self.cluster.get("qos_max_wall_minutes") or {}).get(qos)),
                          (f"partition {part}", (self.cluster.get("partition_max_wall_minutes") or {}).get(part))):
            if cap and minutes > cap:
                raise ValueError(f"the job needs {minutes} min but {what} allows {cap} min: shorten campaign_minutes "
                                 "or choose another one (backend_options)")
        lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}", f"#SBATCH --partition={part}"]
        if qos:
            lines.append(f"#SBATCH --qos={qos}")
        if account:
            lines.append(f"#SBATCH --account={account}")
        # the cluster's known-bad nodes (config bad_nodes) too, unless the request pins one on purpose
        bad = [n for n in (self.cluster.get("bad_nodes") or {}) if n not in (opts.get("nodelist") or "").split(",")]
        exclude = list(dict.fromkeys([*exclude, *bad]))
        if exclude:
            lines.append(f"#SBATCH --exclude={','.join(exclude)}")
        if opts.get("nodelist"):   # a node of one's choice (a test on a given node's cache, or the node of an earlier run)
            lines.append(f"#SBATCH --nodelist={opts['nodelist']}")
        # --requeue: a preempted job goes back to the queue (where the QOS allows it) instead of ending; the runner
        # then starts the cell over (runner.sh start_over). Crusoe burst preempted every run > 20 min on 2026-11-24.
        lines += ["#SBATCH --nodes=1", "#SBATCH --exclusive", "#SBATCH --requeue", f"#SBATCH --time={slurm_time(minutes)}",
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
        holds = [c for c in cells if hold_options(c.get("backend_options"))]
        if holds:
            if len(holds) != len(cells) or len({hold_options(c["backend_options"]) for c in cells}) != 1:
                return [None] * len(cells), "hold_job: every cell of a request goes to the same hold job and node"
            j, node = hold_options(cells[0]["backend_options"])
            r = remote.run_python(self.host, HOLD_SUBMIT_PROGRAM, {"hold_job": j, "node": node, "cells": [
                {"name": f"ab_{c['remote_dir'].rstrip('/').split('/')[-2]}_{c['cell']}", "minutes": c["job_minutes"],
                 "script": hold_script(c["remote_dir"], c["job_minutes"])} for c in cells]})
            if r["problem"]:
                return [None] * len(cells), f"hold_job: {r['problem']}"
            for c, got in zip(cells, r["ids"]):
                c["hold_ctrl"] = got["ctrl"]
            print(f"hold job {j}: {len(cells)} cell(s) dropped for {node} ({r['left']} min left); its agent starts "
                  "them within 20 s", flush=True)
            return [got["job_id"] for got in r["ids"]], None
        script = "".join(f"sbatch --parsable {c['remote_dir']}/job.sbatch || echo FAILED\n" for c in cells)
        p = remote.run_bash(self.host, script)
        out = p.stdout.decode(errors="replace").split()
        ids = [None if (t == "FAILED") else t.split(";")[0] for t in out][:len(cells)]
        ids += [None] * (len(cells) - len(ids))
        err = p.stderr.decode(errors="replace")[-2000:] or None
        if not all(ids) and self.qos:
            # Spur says only "Client specified an invalid argument" when the user's QOS submit limit is reached (burst: 4;
            # 2026-11-22, the user's other jobs held all 4): say how many jobs this user has in the QOS
            q = remote.run_bash(self.host, f"squeue -u $(whoami) -h -q {self.qos} | wc -l").stdout.decode().strip()
            cap = (self.cluster.get("qos_max_submit_per_user") or {}).get(self.qos)
            err = (err or "") + (f"\nhint: you have {q} job(s) in QOS {self.qos}" + (f" (its submit limit is {cap})" if cap else "")
                                 + ": at the limit, wait for one to end, or choose another QOS (backend_options)")
        return ids, err

    def status(self, cells):
        raw = remote.run_python(self.host, STATUS_PROGRAM, {"cells": cells})
        out = []
        for r in raw:
            job = r["job"]
            word = (job.get("raw") or "").split("+")[0].upper()
            state = "UNKNOWN" if not job else STATE_MAP.get(word.split()[0] if word else "", "UNKNOWN")
            raw_word = job.get("raw")
            if r["runner_state"] == "cancelled":   # cancel_run TERMed the runner: the scheduler says FAILED (rc 143)
                state = "CANCELLED"
            if state == "PENDING" and "JobHoldMaxRequeue" in (job.get("reason") or ""):
                # Spur requeues a --requeue job on preemption or failure up to 5 times, then holds it: it would show
                # PENDING forever (Crusoe 308822, 2026-11-24: Restarts=5, never reached its batch script). It is over.
                state, raw_word = "CANCELLED", "PREEMPTED (JobHoldMaxRequeue: requeued 5 times, then held)"
            out.append({"state": state, "raw": raw_word, "node": job.get("node", ""), "elapsed": job.get("elapsed", ""),
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
        if cell["job_id"].startswith("hold:"):
            # not picked up yet: take the ctrl file back; running: stop our runner only (its unique cell path), whose
            # EXIT trap removes our container. The hold job and other sessions' work are not touched.
            _, j, node = cell["job_id"].split(":", 2)
            ctrl = cell.get("hold_ctrl") or ""
            p = remote.run_bash(self.host, f"if [ -e {ctrl} ]; then mv {ctrl} {ctrl}.cancelled && echo taken-back; else "
                                f"srun --jobid={j} --overlap -w {node} -N1 -n1 pkill -TERM -f 'runner[.]sh {cell['remote_dir']}$' "
                                f"&& echo stopped; fi", timeout=180)
            out = p.stdout.decode(errors="replace")
            return ("cancelled (not started)" if "taken-back" in out else "cancelled (runner stopped)" if "stopped" in out
                    else f"cancel failed: {(out + p.stderr.decode(errors='replace')).strip()[-200:]}")
        # SIGTERM to our runner first, so its EXIT trap stops the server and removes the container (cancel_script);
        # a plain scancel on Spur is a SIGKILL that leaves the cleanup to the dead-man switch.
        j = str(int(cell["job_id"]))
        p = remote.run_bash(self.host, cancel_script(j, cell["remote_dir"]), timeout=300)
        out = p.stdout.decode(errors="replace")
        if "already-ended" in out:
            return "not cancelled: the job had already ended"
        if "ended-on-term" in out:
            return "cancelled (runner stopped on SIGTERM)"
        if "killed" in out:
            return "cancelled (SIGKILL after 90 s: the dead-man switch removes the container)"
        return f"scancel failed: {(out + p.stderr.decode(errors='replace')).strip()[-200:]}"

    def extend(self, cell, status):
        """A RUNNING job below the wall time its cell needs gets `scontrol update TimeLimit`."""
        if (cell.get("job_id") or "").startswith("hold:") or status["state"] != "RUNNING" or not status.get("time_limit"):
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


BACKEND = Slurm
