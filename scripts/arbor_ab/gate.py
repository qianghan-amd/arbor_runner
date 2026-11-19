"""Baseline verification gate, docs/design.md section 3.4 (pure functions; I/O lives in scripts/verify_baseline.py).

Layer 1 (structural, no GPU): the record's launch identity vs the CI run's own record of what it ran:
  - (ci_commit, ci_run_id) pin: the run's head_sha must start with the recorded commit;
  - image tag (+ digest when both sides have one), framework, tp, ep, dcp, num_gpus, conc vs the CI bmk artifact.
  Recipe-file flag diff is not done here (see INTENT.md P2 note).
Layer 2 (numeric): the record's measured primary metric vs the CI bmk artifact's value, within tolerance_pct.
  Outcomes: pass | fail (halt: not trusted) | client_mismatch (bench client != CI commit: delta is reported but is not
  apples-to-apples) | artifact_expired (no live artifact: unverified_against_ci) | not_run (nothing comparable).
"""
import re

from . import agentx

STRUCT_FIELDS = (("framework", "framework"), ("tp", "tp"), ("ep", "ep"), ("dcp", "dcp_size"), ("num_gpus", "num_gpus"))


def bmk_candidates(artifacts, tp, conc):
    pat = re.compile(rf"^bmk_.*_tp{tp}_conc{conc}_")
    return [a for a in artifacts if pat.search(a["name"])]


def split_image(ref):
    """'repo:tag@sha256:x' -> ('repo:tag', 'sha256:x'); digest may be None."""
    if ref and "@" in ref:
        tag, digest = ref.split("@", 1)
        return tag, digest
    return ref, None


def structural_diff(record, ci):
    """List of mismatch strings (empty = match) and list of fields that could not be compared."""
    diffs, skipped = [], []
    tag, digest = split_image(record["fingerprint"]["image"])
    digest = record["fingerprint"].get("image_digest") or digest
    ci_tag, ci_digest = split_image(ci.get("image"))
    if tag != ci_tag:
        diffs.append(f"image tag {tag} != CI {ci_tag}")
    if digest and ci_digest and digest != ci_digest:
        diffs.append(f"image digest {digest[:19]} != CI {ci_digest[:19]}")
    for ours, theirs in STRUCT_FIELDS:
        a, b = record["host"].get(ours), ci.get(theirs)
        if a is None or b is None:
            skipped.append(ours)
        elif a != b:
            diffs.append(f"{ours} {a} != CI {b}")
    if record["target"]["conc"] != ci.get("conc"):
        diffs.append(f"conc {record['target']['conc']} != CI {ci.get('conc')}")
    return diffs, skipped


def pick(record, downloaded):
    """downloaded: [(artifact, ci_json)]. Keep structurally matching runs with data; prefer the most requests."""
    usable = [(a, j) for a, j in downloaded if (j.get("num_requests_successful") or 0) > 0 and not structural_diff(record, j)[0]]
    usable.sort(key=lambda x: x[1].get("num_requests_successful") or 0, reverse=True)
    return usable[0] if usable else None


def primary_value(metrics, target):
    return agentx.get_path(metrics, target["primary_metric_path"].split(".", 1)[1])


def numeric(record, ci_json, tolerance_pct, ci_commit):
    """(status, ci_value, measured_value, delta_pct)."""
    measured = primary_value(record["metrics"], record["target"])
    ci_metrics, _ = agentx.metrics_from_agg(ci_json)
    ci_value = primary_value(ci_metrics, record["target"])
    if measured is None or ci_value is None:
        return "not_run", ci_value, measured, None
    delta = (measured - ci_value) / ci_value * 100.0
    client = record["measured"].get("bench_client_commit")
    if client and not client.startswith(ci_commit[:10]) and not ci_commit.startswith(client[:10]):
        return "client_mismatch", ci_value, measured, delta
    return ("pass" if abs(delta) <= tolerance_pct else "fail"), ci_value, measured, delta


def flags(structural_status, numeric_status):
    """(ci_verified, unverified_against_ci) per design.md 3.4."""
    ci_verified = structural_status == "pass" and numeric_status == "pass"
    unverified = numeric_status in ("artifact_expired", "client_mismatch", "not_run")
    return ci_verified, unverified
