"""
One-time move of the production history from local files to the S3 registry (P2 phase D).

    python -m pipeline.migrate_registry --dry-run     # what would happen
    python -m pipeline.migrate_registry               # publish every run in the history, replay the history on S3

Reads models/<cat>/production_history.jsonl (oldest first), publishes each run it names (+ gold snapshot) and writes
the same promotions to the S3 pointer in the same order, keeping run_id, reason, promoted_at and metrics; S3 gives
them versions 1..n and `migrated_at`. The last entry must be the local production.json, so the S3 pointer ends where
local production is. Refuses when the S3 pointer already exists (nothing is overwritten; run it once).
The local files are left as they are: switching the engine and the pipeline is MODEL_REGISTRY=s3.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from pipeline.config import Paths, load_config
from pipeline.gate import run_path
from pipeline.pointer import LocalPointer, S3Pointer, manifest_sha256, registry_for
from pipeline.storage import load_s3_env


def migrate(paths, registry, dry_run=False, log=print) -> list[dict]:
    local = LocalPointer(paths)
    hist, cur = local.history(), local.read()
    if not hist or cur is None:
        raise RuntimeError(f"no local production history in {paths.models}")
    if hist[-1]["run_id"] != cur["run_id"]:
        raise RuntimeError(f"history ends with {hist[-1]['run_id']} but production.json is {cur['run_id']}: "
                           "fix the local files first")
    s3 = S3Pointer(registry)
    if s3.read() is not None:
        raise RuntimeError(f"s3://{registry.cfg.bucket}/{s3.key} already exists: the registry is already in use")
    missing = sorted({h["run_id"] for h in hist if not run_path(paths, h).exists()})
    if missing:
        raise FileNotFoundError(f"runs named in the history are not on this disk: {missing}")

    runs = list(dict.fromkeys(h["run_id"] for h in hist))          # distinct, first-promotion order
    log(f"{len(hist)} promotions, {len(runs)} runs: {', '.join(runs)} -> s3://{registry.cfg.bucket}/{registry.models}")
    if dry_run:
        for h in hist:
            log(f"  would write: {h['run_id']} ({h.get('reason')}, promoted {h.get('promoted_at')})")
        return []
    registry.ensure_bucket()
    manifests = {}
    for rid in runs:
        manifests[rid] = registry.publish_run(run_path(paths, next(h for h in hist if h["run_id"] == rid)),
                                              paths.gold, log)
    now = datetime.now(timezone.utc).isoformat()
    written = []
    for h in hist:
        m = manifests[h["run_id"]]
        entry = {k: v for k, v in h.items() if k != "version"} | {
            "path": f"runs/{h['run_id']}", "manifest_sha256": manifest_sha256(m), "gold": m["gold"], "migrated_at": now}
        written.append(s3.write(entry))
        log(f"  v{written[-1]['version']}: {h['run_id']} ({h.get('reason')})")
    log(f"S3 production = {written[-1]['run_id']} v{written[-1]['version']} (local production.json unchanged)")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. category=Video_Games")
    a = ap.parse_args(argv)
    load_s3_env()
    paths = Paths(load_config(None, a.set))
    migrate(paths, registry_for(paths), a.dry_run)


if __name__ == "__main__":
    main()
