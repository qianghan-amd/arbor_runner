"""Render the server launch exactly as the Arbor under test would (arbor/policy/claude_loop.py `_start_server`).

Runs on the node host with PYTHONPATH=<arbor tree> (the commit under test) and docker access.
Usage: python3 serve_cmd.py <arbor_config.json> <phase_dir_on_host> <phase_dir_in_container> [--set K=V ...]
Writes <phase_dir>/serve.sh (run inside the container) and <phase_dir>/launch.json (flags, env, command, and whether
the workspace arbor/config.json was found, i.e. whether this is the pristine or the kept launch).

Composition, in Arbor's order: flags = _live_flags(config); env = {**_live_env(config), **--set}; env =
device_env(env, gpu_ids); env.update(extra_env(flags)); command = launch_cmd(...). Before a campaign the workspace
has no arbor/config.json, so live == static (the pristine launch); after it, the workspace file holds the kept launch.
The pid_file form of launch_cmd starts the server in its own process group, so the runner stops that group only.
"""
import json
import sys
from pathlib import Path

from arbor import component, frameworks
from arbor.execute.bench import _live_env, _live_flags
from arbor.execute.workspace import read_workspace_config


def main(argv):
    cfg_path, host_dir, cont_dir, *rest = argv
    overrides = {}
    while rest:
        flag, kv, *rest = rest
        assert flag == "--set", flag
        k, v = kv.split("=", 1)
        overrides[k] = v
    config = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    ws = read_workspace_config(config["docker_container"], config["workspace_root"]) or {}
    flags = _live_flags(config)
    env = {**_live_env(config), **overrides}
    fw = frameworks.get(config)
    env = fw.device_env(env, ",".join(map(str, component.gpu_list(config.get("gpu_ids")) or [])))
    env.update(fw.extra_env(flags))
    cmd = fw.launch_cmd(config, env, flags, log_path=f"{cont_dir}/server.log", pid_file=f"{cont_dir}/server.pid")
    host = Path(host_dir)
    host.mkdir(parents=True, exist_ok=True)
    (host / "serve.sh").write_text(f"#!/bin/bash\ncd /tmp\n{cmd}\n", encoding="utf-8", newline="\n")
    (host / "launch.json").write_text(json.dumps({
        "framework": frameworks.name_of(config), "live_workspace_config": bool(ws), "workspace_config": ws or None,
        "server_flags": flags, "server_env": env, "command": cmd}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[serve_cmd] {'kept (workspace arbor/config.json)' if ws else 'pristine (no workspace config)'} launch: "
          f"{frameworks.name_of(config)}, {len(flags.split())} flag words, {len(env)} env vars")


if __name__ == "__main__":
    main(sys.argv[1:])
