"""Baseline epoch records: schema validation, fingerprint -> epoch_id, storage with a per-cell latest pointer.

docs/design.md section 3.3: epoch_id is derived from a fingerprint, never assigned by hand.
  agentx_ci   -> (image, ci_commit, ci_run_id)
  user_custom -> content hash of the launch config (image + host)
Same fingerprint on a refresh -> same epoch; different fingerprint -> new epoch, old one gets superseded_by.
"""
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from . import REPO_ROOT

SCHEMA_PATH = REPO_ROOT / "schemas" / "baseline.schema.json"
BASELINES_DIR = REPO_ROOT / "baselines"
COMMIT_PREFIX = 10   # summaries record 10-char SHAs; normalize so short and full SHAs give the same fingerprint
CUSTOM_HOST_KEYS = ("framework", "tp", "ep", "dcp", "num_gpus", "model_path", "server_flags", "server_env")


def _validator():
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")), format_checker=FormatChecker())


def validation_errors(record):
    """List of 'path: message' strings; empty means valid."""
    errs = sorted(_validator().iter_errors(record), key=lambda e: list(e.absolute_path))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errs]


def validate(record):
    errs = validation_errors(record)
    if errs:
        raise ValueError("baseline record invalid:\n  " + "\n  ".join(errs))


def fingerprint_material(record):
    fp = record["fingerprint"]
    if fp["source_type"] == "agentx_ci":
        return {"source_type": "agentx_ci", "image": fp["image"],
                "ci_commit": fp["ci_commit"][:COMMIT_PREFIX], "ci_run_id": str(fp["ci_run_id"])}
    host = record["host"]
    return {"source_type": "user_custom", "image": fp["image"], "host": {k: host.get(k) for k in CUSTOM_HOST_KEYS}}


def fingerprint_hash(record):
    canon = json.dumps(fingerprint_material(record), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def derive_epoch_id(record):
    h = fingerprint_hash(record)[:8]
    fp = record["fingerprint"]
    tag = fp["ci_commit"][:COMMIT_PREFIX] if fp["source_type"] == "agentx_ci" else "custom"
    return f"{record['model']}__{record['target']['key']}__{tag}_{h}"


def cell_dir(model, target, root=None):
    return Path(root or BASELINES_DIR) / model / target


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_latest(model, target, root=None):
    """The current-epoch record for a cell, or None if the cell has no baseline."""
    d = cell_dir(model, target, root)
    ptr = d / "latest.json"
    if not ptr.exists():
        return None
    return load(d / f"{load(ptr)['epoch_id']}.json")


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def store(record, root=None):
    """Validate and write a record as the cell's current epoch. Idempotent for an unchanged record.

    A different fingerprint for the same cell archives the previous epoch (superseded_by set, file kept, never deleted).
    """
    validate(record)
    if record["epoch_id"] != derive_epoch_id(record):
        raise ValueError(f"epoch_id {record['epoch_id']!r} does not match fingerprint ({derive_epoch_id(record)!r})")
    model, target = record["model"], record["target"]["key"]
    d = cell_dir(model, target, root)
    prev = load_latest(model, target, root)
    if prev is not None and prev["epoch_id"] != record["epoch_id"]:
        prev["superseded_by"] = record["epoch_id"]
        _write_json(d / f"{prev['epoch_id']}.json", prev)
    path = d / f"{record['epoch_id']}.json"
    _write_json(path, record)
    _write_json(d / "latest.json", {"epoch_id": record["epoch_id"]})
    return path
