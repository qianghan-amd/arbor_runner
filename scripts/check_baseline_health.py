"""Refresh the three health flags of stored baselines (docs/design.md section 3.5) and write them into each record.

Usage:
  python scripts/check_baseline_health.py [--model M] [--target T] [--dry-run]
  ci_artifact_available  a live CI bmk artifact with data exists for (tp, conc) in the pinned CI run (GitHub relay)
  image_available        the pinned image digest still resolves on Docker Hub (manifest HEAD, no pull); tag checked too
  local_mirror_exists    fingerprint.local_mirror_path exists on the login host (false while no mirror is recorded)
Stateless and idempotent apart from `checked_at`. Short ssh calls only; no GPU, no Slurm.
"""
import argparse
import datetime as dt
import sys

from arbor_ab import baseline, config, gate, github_relay, registry, remote

REPO = "SemiAnalysisAI/InferenceX"
MIRROR_PROGRAM = "import os\nprint(json.dumps([os.path.exists(p) for p in PAYLOAD]))\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--target")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    local = config.load_local()
    relay, login = local["ssh"]["github_relay_host"], local["ssh"]["login_host"]

    matrix = config.load_matrix()
    records = [r for m, t in config.cells(matrix) if (not args.model or m == args.model) and (not args.target or t == args.target)
               for r in [baseline.load_latest(m, t)] if r is not None]

    ci_recs = [r for r in records if r["fingerprint"]["source_type"] == "agentx_ci"]
    runs = sorted({r["fingerprint"]["ci_run_id"] for r in ci_recs})
    arts = dict(zip(runs, github_relay.call(relay, [github_relay.artifacts(REPO, run) for run in runs]))) if runs else {}

    images = sorted({(r["fingerprint"]["image"], r["fingerprint"]["image_digest"]) for r in records})
    img_res = dict(zip(images, registry.check(relay, images)))

    mirrors = sorted({r["fingerprint"]["local_mirror_path"] for r in records if r["fingerprint"]["local_mirror_path"]})
    mirror_ok = dict(zip(mirrors, remote.run_python(login, MIRROR_PROGRAM, mirrors))) if mirrors else {}

    t = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("| cell | ci_artifact_available | image_available | local_mirror_exists | detail |\n|---|---|---|---|---|")
    for r in records:
        fp, notes = r["fingerprint"], []
        ci_ok = None
        if fp["source_type"] == "agentx_ci":
            a = arts[fp["ci_run_id"]]
            if a.get("http") == 200:
                live = [x for x in gate.bmk_candidates(a["data"], r["host"]["tp"], r["target"]["conc"]) if not x["expired"]]
                ci_ok = bool(live)
                notes.append(f"bmk tp{r['host']['tp']} conc{r['target']['conc']}: {len(live)} live"
                             + (f", earliest expiry {min(x['expires_at'] for x in live)[:10]}" if live else
                                (" (run has no artifacts left)" if not a["data"] else "")))
            else:
                notes.append(f"artifact list http {a.get('http')}")
        img_ok, img_note = registry.summarize(img_res[(fp["image"], fp["image_digest"])], fp["image_digest"])
        notes.append(img_note)
        mp = fp["local_mirror_path"]
        mir_ok = bool(mirror_ok.get(mp)) if mp else False
        notes.append(f"mirror {mp}" if mp else "no mirror recorded")
        r["health"] = {"ci_artifact_available": ci_ok, "image_available": img_ok, "local_mirror_exists": mir_ok,
                       "checked_at": t, "detail": "; ".join(notes)}
        print(f"| {r['model']}/{r['target']['key']} | {ci_ok} | {img_ok} | {mir_ok} | {'; '.join(notes)} |")
        if not args.dry_run:
            baseline.store(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
