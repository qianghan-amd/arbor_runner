"""Load config/matrix.yaml, config/cluster/<name>.yaml and local.yaml, and answer matrix lookups."""
from pathlib import Path

import yaml

from . import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "config"
LOCAL_PATH = REPO_ROOT / "local.yaml"
TARGET_KEYS = ("direction", "primary_metric_path", "primary_metric_per_gpu_path", "secondary_metric_path")


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_matrix(path=None):
    m = _load_yaml(path or CONFIG_DIR / "matrix.yaml")
    for model, spec in m["models"].items():
        if spec["status"] not in ("active", "placeholder"):
            raise ValueError(f"matrix: {model}: unknown status {spec['status']!r}")
        for tkey, t in spec["targets"].items():
            if t["kind"] not in m["target_kinds"]:
                raise ValueError(f"matrix: {model}/{tkey}: unknown kind {t['kind']!r}")
            if not tkey.startswith(f"c{t['conc']}_"):
                raise ValueError(f"matrix: {model}/{tkey}: key does not match conc {t['conc']}")
    return m


def cells(matrix, include_placeholder=False):
    """[(model, target_key)] in matrix order."""
    return [(model, tkey) for model, spec in matrix["models"].items()
            if include_placeholder or spec["status"] == "active" for tkey in spec["targets"]]


def target_spec(matrix, model, target):
    """The `target` block of a baseline record for this cell (docs/design.md section 3.6)."""
    try:
        t = matrix["models"][model]["targets"][target]
    except KeyError:
        raise KeyError(f"{model}/{target} is not in config/matrix.yaml") from None
    kind = matrix["target_kinds"][t["kind"]]
    return {"key": target, "conc": t["conc"], **{k: kind[k] for k in TARGET_KEYS}}


def tolerance_pct(matrix, model, target):
    t = matrix["models"][model]["targets"][target]
    return float(t.get("tolerance_pct", matrix["default_tolerance_pct"]))


def gsm8k_max_drop(matrix, model, target):
    t = matrix["models"][model]["targets"][target]
    return float(t.get("gsm8k_max_drop", matrix["default_gsm8k_max_drop"]))


def load_cluster(name="crusoe"):
    return _load_yaml(CONFIG_DIR / "cluster" / f"{name}.yaml")


def load_model(key):
    return _load_yaml(CONFIG_DIR / "models" / f"{key}.yaml")


def load_local(path=None):
    p = Path(path or LOCAL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: copy local.yaml.example to local.yaml and fill it in")
    return _load_yaml(p)


def local_view(site=None, path=None):
    """local.yaml as one site sees it (its `ssh`, `remote`, ... over the shared keys); see backends.site_local."""
    from .backends import site_local
    return site_local(load_local(path), site)[1]
