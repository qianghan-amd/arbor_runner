"""Execution backends: WHERE and HOW one PR-test cell's job runs (docs/design.md section 5.1).

The orchestrator (submit_ab_test.py, poll_status.py, cancel_run.py, check_setup.py, setup_remote.py) talks only to
the `Backend` contract below; baselines, the gate and the report never see a backend. The job payload,
scripts/runner/runner.sh, is backend-neutral: it needs bash, docker and the GPUs of one node, and the run's files
at `run_dir(run_id)`. A backend decides how those files get there and how the payload is started and watched.

Backends: `slurm` (any Slurm(-compatible) cluster over short ssh calls; Crusoe's Spur and meta64's Slurm 25), with
`crusoe_slurm` as its older name. A cluster is config/cluster/<name>.yaml; a site (local.yaml `sites:`) is one
cluster plus its own paths and credentials (`for_site`). To add a scheduler of another kind (or "run on this host, no
scheduler" for dev use): write backends/<name>.py with a `Backend` subclass named `BACKEND`, and name it as a site's
`backend`. Nothing else changes.
"""
import importlib

# Normalized job states; each backend maps its scheduler's own words onto these (and keeps the raw word too).
STATES = ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "UNKNOWN")

# What the runner writes that is small enough to bring home (relative to a cell dir). No model data, no traces.
RESULT_FILES = ("state", "phases", "logs/runner.log", "weights.json", "campaign_check.json", "campaign_exit.json",
                "*/agg.json", "*/bench_output.json", "*/gsm8k.log", "*/phase.json", "*/launch.json", "*/cmdline.txt",
                "*/tree.txt", "*/gsm8k/results_*.json", "*/gsm8k/*/results_*.json")


class Backend:
    """The contract. A `cell` is one entry of run_config.json["cells"]: cell, remote_dir, job_id, job_minutes, ..."""
    name = "abstract"

    def site(self):
        """Paths the runner needs on the node: node_local_root, shared_hf_hub, infx_clone, gateway_env."""
        raise NotImplementedError

    def run_dir(self, run_id):
        """Where the run's files live, as the job sees them."""
        raise NotImplementedError

    def job_script(self, cell_dir, job_name, minutes, exclude=(), options=None):
        """(file name, text) of the wrapper that starts `bash <cell_dir>/runner.sh <cell_dir>`. exclude: hosts the job
        must not land on (request `exclude_nodes`, e.g. a node another user left dirty); ignore if not applicable.
        options: the request's `backend_options` (backend-specific, e.g. crusoe_slurm {qos, account})."""
        raise NotImplementedError

    def stage(self, run_id, local_run_dir, cells, extra_files):
        """Make the rendered cell dirs (and extra_files, e.g. arbor.tar) visible at run_dir(run_id)."""
        raise NotImplementedError

    def submit(self, cells):
        """Start each cell's job; return the job ids in order (None where a start failed) and an error text."""
        raise NotImplementedError

    def status(self, cells):
        """Per cell: {state (one of STATES), raw, node, elapsed, time_limit, runner_state, phases_done, log_tail}."""
        raise NotImplementedError

    def fetch(self, cell, local_dir):
        """Copy the cell's RESULT_FILES into local_dir; return how many files."""
        raise NotImplementedError

    def cancel(self, cell):
        """Stop the cell's job (its own job only); return a short result text."""
        raise NotImplementedError

    def extend(self, cell, status):
        """Optional: give a running job the wall time it needs. None = nothing done / not supported."""
        return None

    def check(self):
        """Setup checks for check_setup.py: [(name, ok, message)]."""
        return []

    def ensure_bench_commits(self, shas):
        """Optional: make the InferenceX clone the jobs read hold these commits (and their submodules' commits):
        {sha: {"ok": bool, "submodules": {path: bool}}}; {} = not supported (the job then checks and fails)."""
        return {}

    def snapshot_complete(self, path):
        """Optional: (complete?, detail) for a model snapshot dir: its index's every shard exists. (None, why) = unknown."""
        return None, "not supported by this backend"

    def files_exist(self, paths):
        """Optional: which of these paths exist as the job sees them ({path: bool}); {} = cannot tell."""
        return {}

    def setup(self, key=None, test_gateway=False, infx_shas=()):
        """Idempotent one-time setup (dirs, the LLM gateway env from `key`, the InferenceX clone holds `infx_shas`,
        optionally one gateway call): [(name, ok, message)]."""
        return []


def get(local, cluster):
    """The backend local.yaml names (default crusoe_slurm)."""
    name = local.get("backend", "crusoe_slurm")
    mod = importlib.import_module(f"{__name__}.{name}")
    return mod.BACKEND(local, cluster)


def site_local(local, site=None):
    """(site name, the local.yaml view for that site). local.yaml may hold `sites: {<name>: {...}}` and
    `default_site`: a site's keys override the shared ones (one level deep for dicts, e.g. its own `arbor.gateway_env`
    beside the shared `arbor.clone`); `cluster` names config/cluster/<cluster>.yaml (default: the site's name).
    Without `sites`, the file is one site, named by its `cluster` key (default crusoe)."""
    sites = local.get("sites")
    if not sites:
        name = local.get("cluster", "crusoe")
        if site not in (None, name):
            raise KeyError(f"site {site!r}: local.yaml has no `sites:` (only {name})")
        return name, local
    name = site or local.get("default_site") or next(iter(sites))
    if name not in sites:
        raise KeyError(f"site {name!r} is not in local.yaml sites ({', '.join(sites)})")
    merged = {k: v for k, v in local.items() if k not in ("sites", "default_site")}
    for k, v in sites[name].items():
        merged[k] = {**merged[k], **v} if isinstance(v, dict) and isinstance(merged.get(k), dict) else v
    merged.setdefault("cluster", name)
    return name, merged


def for_run(local, run_config):
    """The backend of the site a run was submitted to (run_config.json `site_name`; runs before 2026-11-21: crusoe).
    (run_config.json `site` is the node paths the run was rendered with.)"""
    return for_site(local, run_config.get("site_name") or "crusoe")


def for_site(local, site=None):
    """The backend of a site (local.yaml `sites`, see site_local); its `site_name` and `gpu_type` are set."""
    from .. import config
    name, view = site_local(local, site)
    cluster = config.load_cluster(view.get("cluster", name))
    b = get(view, cluster)
    b.site_name, b.gpu_type = name, cluster.get("gpu_type")
    return b
