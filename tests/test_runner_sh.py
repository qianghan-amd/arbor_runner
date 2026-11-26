"""Unit tests of scripts/runner/runner.sh functions, run in real bash with stubbed commands (no docker, no GPU).

Each test takes the function bodies from runner.sh as they are (no copy), defines stubs as bash functions (docker,
rocm-smi, sleep, hostname, in_c, the Arbor CLI), and checks files, return codes and log lines. Bugs these guard:
resume after a requeue (meta64 147783), SGLang slow to free GPU memory (meta64 149589), the launcher guard (#74),
the analysis extras install (#74 c10021).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from arbor_ab import REPO_ROOT

RUNNER = (REPO_ROOT / "scripts" / "runner" / "runner.sh").read_text(encoding="utf-8")


def find_bash():
    if os.name == "nt":   # not WSL's System32\bash.exe: Git Bash
        for c in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
            if c and "system32" not in c.lower() and Path(c).exists():
                return c
        return None
    return shutil.which("bash")


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash")


def func(name):
    """The text of runner.sh's `name() {` ... first line that is exactly `}`."""
    m = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}$", RUNNER, re.M | re.S)
    assert m, f"{name} not found in runner.sh"
    return m.group(0)


def attempt_paths():
    return re.search(r"^ATTEMPT_PATHS=.*$", RUNNER, re.M).group(0)


STUBS = r'''
log() { echo "[runner] $*" >> "$LOGS/runner.log"; echo "[runner] $*"; }
die() { log "FATAL ($PHASE): $*"; echo "failed:$PHASE" > "$CELL_DIR/state"; exit 1; }
sleep() { :; }
hostname() { echo testnode; }
'''


def run(tmp_path, body, env=None):
    cell = tmp_path / "cell"
    (cell / "logs").mkdir(parents=True, exist_ok=True)
    script = ("set -uo pipefail\n" + f"CELL_DIR={cell.as_posix()}\nLOGS=$CELL_DIR/logs\n" + STUBS + body + "\n")
    f = tmp_path / "t.sh"
    f.write_bytes(script.encode("utf-8"))
    # 900 s: a bash start can take minutes on a loaded Windows box (2026-11-26: two of these at 300 s timed out in a
    # 37-min suite run); the scripts themselves do no waiting (sleep is stubbed)
    p = subprocess.run([BASH, f.as_posix()], capture_output=True, text=True, timeout=900,
                       env={**os.environ, "MSYS_NO_PATHCONV": "1", **(env or {})})
    log = (cell / "logs" / "runner.log").read_text(encoding="utf-8") if (cell / "logs" / "runner.log").exists() else ""
    return p, cell, log


# ---- start_over -------------------------------------------------------------------------------------------------------

def start_over_body(setup):
    return attempt_paths() + "\n" + func("start_over") + "\n" + setup + "\nstart_over\necho rc=$?\n"


def test_start_over_first_attempt_moves_nothing(tmp_path):
    p, cell, _ = run(tmp_path, start_over_body("mkdir -p $CELL_DIR/phases"))
    assert "rc=0" in p.stdout and not (cell / "attempts").exists()


def test_start_over_archives_and_starts_fresh(tmp_path):
    setup = ("mkdir -p $CELL_DIR/phases $CELL_DIR/before $CELL_DIR/sessions/full $CELL_DIR/arbor $CELL_DIR/infx\n"
             "for x in fetch prepare before campaign; do touch $CELL_DIR/phases/$x.start $CELL_DIR/phases/$x.done; done\n"
             "touch $CELL_DIR/phases/after.start; echo after > $CELL_DIR/state; echo c > $LOGS/campaign.log\n"
             "log 'after: start on nodeA job=1'")
    p, cell, log = run(tmp_path, start_over_body(setup))
    assert "rc=0" in p.stdout, p.stderr
    a = cell / "attempts" / "1"
    assert (a / "phases" / "before.done").exists() and (a / "before").is_dir() and (a / "sessions" / "full").is_dir()
    assert (a / "logs" / "campaign.log").exists() and (a / "arbor").is_dir()
    assert (cell / "phases").is_dir() and not list((cell / "phases").iterdir())
    assert not (cell / "before").exists() and (cell / "infx").is_dir()          # shared trees stay
    assert "attempt 1 (on nodeA, stopped in after) moved to attempts/1/; starting over on testnode" in log


def test_start_over_leaves_a_done_cell(tmp_path):
    setup = "mkdir -p $CELL_DIR/phases; touch $CELL_DIR/phases/fetch.start; echo done > $CELL_DIR/state"
    p, cell, _ = run(tmp_path, start_over_body(setup))
    assert "rc=0" in p.stdout and not (cell / "attempts").exists()


def test_start_over_gives_up_after_max_attempts(tmp_path):
    setup = ("mkdir -p $CELL_DIR/phases $CELL_DIR/attempts/1 $CELL_DIR/attempts/2; touch $CELL_DIR/phases/fetch.start;"
             " echo fetch > $CELL_DIR/state")
    p, cell, log = run(tmp_path, start_over_body(setup))
    assert p.returncode == 1 and "rc=" not in p.stdout                          # die exits
    assert (cell / "state").read_text().strip() == "failed:restart"
    assert "attempt 4 of at most 3" in log and not (cell / "attempts" / "3").exists()


# ---- stop_server ------------------------------------------------------------------------------------------------------

def stop_body(framework, vram_seq):
    """rocm-smi --showmemuse prints the next value of vram_seq (the last repeats); calls are counted."""
    seq = " ".join(str(v) for v in vram_seq)
    return f'''
FRAMEWORK={framework}; TAG=smoke; GPUS=0,1; CONTAINER=c; SERVER_UP=1
VR=({seq}); echo 0 > $CELL_DIR/calls
in_c() {{ return 0; }}
docker() {{ echo "PID ELAPSED COMMAND"; echo "123 01:00 python3 -m sglang.launch_server"; }}
rocm-smi() {{
  if [ "$1" = --showpids ]; then echo "4567 python3 1 2147483648"; return; fi
  local n; read -r n < $CELL_DIR/calls; echo $((n + 1)) > $CELL_DIR/calls   # builtins: forks are slow on Windows
  local i=$n; [ $i -ge ${{#VR[@]}} ] && i=$((${{#VR[@]}} - 1))
  echo "GPU[0]		: GPU Memory Allocated (VRAM%): ${{VR[$i]}}"
}}
{func("stop_server")}
stop_server; echo rc=$?
read -r calls < $CELL_DIR/calls; echo calls=$calls
'''


def test_stop_server_gpus_free(tmp_path):
    p, _, log = run(tmp_path, stop_body("vllm", [5, 3, 0]))
    assert "rc=0" in p.stdout and "calls=3" in p.stdout, p.stderr
    assert "smoke: server stopped, GPUs free" in log


@pytest.mark.skipif(os.name == "nt", reason="24-60 polls of forking pipes take minutes under Git Bash; Linux runs it")
@pytest.mark.parametrize("framework,tries", [("vllm", 24), ("sglang", 60)])
def test_stop_server_memory_held(tmp_path, framework, tries):
    # SGLang gets 300 s (60 polls), vLLM 120 s (24): meta64 149589 (hicache) held 2 % past 120 s
    p, _, log = run(tmp_path, stop_body(framework, [2]))
    assert "rc=1" in p.stdout and f"calls={tries}" in p.stdout, p.stderr
    assert "still holds memory (VRAM 2%)" in log
    assert "smoke: GPU processes: 4567 python3" in log and "smoke: left in the container: 123 01:00 python3" in log


# ---- arbor_check -----------------------------------------------------------------------------------------------------

def check_body(doc, launcher=None):
    py = Path(sys.executable).as_posix()
    lines = [f"CHECK_DOC='{json.dumps(doc)}'", f'REALPY="{py}"', "ARBOR=$CELL_DIR; OPT_OBJECTIVE=''",
             'pystub() { if [ "$1" = -m ]; then echo "$CHECK_DOC"; return 0; fi; "$REALPY" "$@"; }', "PY=pystub"]
    if launcher is not None:
        lines.append(f"ARBOR_LAUNCHER={launcher}")
    return "\n".join(lines) + "\n" + func("arbor_flags") + "\n" + func("arbor_check") + "\narbor_check; echo rc=$?\n"


OK = {"ok": True, "refused": [], "waits": [], "makeup": {"lacks": []}, "warnings": []}


@pytest.mark.parametrize("doc,launcher,rc", [
    ({**OK, "runs_under": "process"}, None, 0),
    ({**OK, "runs_under": "launcher"}, None, 0),       # the launcher is the default since #74 closed (1eef607)
    ({**OK, "runs_under": "launcher"}, "1", 0),
    ({**OK, "runs_under": "launcher"}, "0", 1),        # request arbor_standalone: one process only
    ({**OK, "runs_under": "process", "waits": ["gpu"]}, None, 1),
    ({**OK, "runs_under": "process", "makeup": {"lacks": ["x"]}}, None, 1),
    ({**OK, "ok": False}, None, 1),
])
def test_arbor_check(tmp_path, doc, launcher, rc):
    p, _, log = run(tmp_path, check_body(doc, launcher))
    assert f"rc={rc}" in p.stdout, (p.stdout, p.stderr)
    assert "[runner] arbor check rc=0" in log
    if launcher == "0":
        assert "not in one process: refused" in log


# ---- analysis_extras -------------------------------------------------------------------------------------------------

def test_analysis_extras_logs_versions(tmp_path):
    body = ('in_c() { echo "duckdb 2.0.1 pyarrow 23.0.1"; }\n' + func("analysis_extras") + "\nanalysis_extras; echo rc=$?\n")
    p, _, log = run(tmp_path, body)
    assert "rc=0" in p.stdout
    assert "analysis extras installed in the container (--no-deps): duckdb 2.0.1 pyarrow 23.0.1" in log


def test_analysis_extras_failure_is_not_fatal(tmp_path):
    body = ('in_c() { case "$1" in *"pip install"*) return 1;; *) echo "ERROR: no network";; esac; }\n'
            + func("analysis_extras") + "\nanalysis_extras; echo rc=$?\n")
    p, _, log = run(tmp_path, body)
    assert "rc=0" in p.stdout
    assert "analysis extras NOT installed" in log and "ERROR: no network" in log


# ---- bench_sweep (mode bench) -----------------------------------------------------------------------------------------

def test_bench_sweep_one_server_all_concurrencies_and_a_failure(tmp_path):
    # the real bench_sweep with its steps stubbed: one serve_up/stop_server, an agentx_bench per concurrency, the
    # failed one (4) does not stop the others, the phase fails, summary.json lists every concurrency
    py = Path(sys.executable).as_posix()
    stubs = (f'PY="{py}"\nBENCH_CONCS="1 4 16"; BENCH_DUR=300; NQ_BENCH=0\n'
             'serve_up() { OUT=$CELL_DIR/$1; CIN=/workspace/exp/$1; mkdir -p "$OUT"; echo serve_up >> $CELL_DIR/calls; }\n'
             'stop_server() { echo stop >> $CELL_DIR/calls; }\n'
             'gsm8k_eval() { echo "gsm8k $1" >> $CELL_DIR/calls; }\n'
             'phase_record() { echo "phase $*" >> $CELL_DIR/calls; }\n'
             'agentx_bench() { echo "bench $1 $3" >> $CELL_DIR/calls; mkdir -p "$2"; [ "$1" = 4 ] && return 1; '
             'echo "{\\"status\\": \\"ok\\", \\"total_throughput_per_gpu\\": $1000, \\"e2e_norm_intvty_p90\\": 9.5, '
             '\\"num_requests_successful\\": 9, \\"num_requests_total\\": 10}" > "$2/bench_output.json"; echo "{}" > "$2/agg.json"; }\n')
    p, cell, log = run(tmp_path, stubs + func("bench_sweep") + "\nbench_sweep\necho rc=$?\n")
    assert "rc=1" in p.stdout, p.stdout + p.stderr
    calls = (cell / "calls").read_text().splitlines()
    assert calls == ["serve_up", "bench 1 /workspace/exp/bench/c1", "bench 4 /workspace/exp/bench/c4",
                     "bench 16 /workspace/exp/bench/c16", "gsm8k 0", "stop", "phase 300 0 1 4 16"]
    s = json.loads((cell / "bench" / "summary.json").read_text())
    assert [(r["concurrency"], r["ok"]) for r in s["concurrencies"]] == [(1, True), (4, False), (16, True)]
    assert s["concurrencies"][2]["total_throughput_per_gpu"] == 16000 and "bench c4: ok=False" in log
