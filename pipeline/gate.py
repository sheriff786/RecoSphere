"""
Eval gate + promotion + rollback.

Gate (candidate run vs current production), on the CANDIDATE's test window:
  1. sanity            all metrics finite, candidate recall > 0
  2. beats baseline    ranker beats recent popularity on NDCG@10 and Recall@100 (same window)
  3. not older data    candidate's test window starts at/after production's (else production has
                       already trained on the candidate's test window = leakage, and we'd be
                       replacing a newer model with an older one)
  4. no regression     per metric: paired bootstrap over test users of (candidate - production)
                       -> relative change + CI. FAIL if confidently worse by more than min_effect
                       (CI upper bound < -min_effect) OR the point change < -max_point_drop.
                       (Not a non-inferiority test: with sparse NDCG@10 the CI is too wide for even
                       an identical model to "prove" it is not worse, so every retrain would fail.)
                       Metrics: ranker NDCG@10, ranker Recall@100, candidate recall (retrieval can
                       not silently get worse behind a ranker that hides it).
The production models are RE-EVALUATED on the candidate's data, never compared via old numbers
(an older model's numbers come from an older window and are not comparable).

What the bootstrap does and does NOT cover: it measures test-set sampling noise ("would this
artifact still look this way on other users?"). It does not measure training randomness (seed
variance) - two seeds of the same config ARE different artifacts and the gate judges the artifact.
Less seed variance has to come from more stable training, not from a looser gate.

Promotion = move the production pointer (the serving API follows it) and log it in the history.
Rollback = point back to the previous entry. Model files are never copied or deleted, so every past
model stays available. Where the pointer lives (local files or S3, P2) is pipeline/pointer.py; in S3 mode
a run must be published (pipeline/storage.py) before it can be promoted.
"""
import json
import math

import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from pipeline.pointer import manifest_sha256, pointer_store, registry_for
from pipeline.steps import evaluate_models


def run_path(paths, entry) -> Path:
    """Pointer paths are stored RELATIVE to models/<cat>/ (e.g. runs/2026...), so the same pointer works
    on Windows, in WSL and inside Docker where models/ is mounted somewhere else."""
    p = Path(entry["path"])
    return p if p.is_absolute() else paths.models / p


def load_per_user(path) -> dict:
    z = np.load(path)
    return {k.replace("|", "/"): z[k] for k in z.files}


def paired_bootstrap(cand: dict, prod: dict, metrics, n_boot=2000, confidence=0.95, seed=0) -> dict:
    """Resample USERS (same resample for both models -> paired). Returns, per metric, the relative change
    mean(cand)/mean(prod) - 1 and its percentile CI. Pairing removes the user-to-user variance that both
    models share, so the CI is much tighter than comparing two independent means."""
    rng = np.random.default_rng(seed)
    n = len(cand["uid"])
    sums = {m: [] for m in metrics}
    for s in range(0, n_boot, 100):                            # chunks: 100 x n indices at a time
        idx = rng.integers(0, n, size=(min(100, n_boot - s), n))
        for m in metrics:
            sums[m].append((cand[m][idx].mean(1), prod[m][idx].mean(1)))
    out, a = {}, (1 - confidence) / 2
    for m in metrics:
        c_b = np.concatenate([x for x, _ in sums[m]]); p_b = np.concatenate([y for _, y in sums[m]])
        rel = c_b / np.maximum(p_b, 1e-12) - 1
        c_mean, p_mean = float(cand[m].mean()), float(prod[m].mean())
        out[m] = {"candidate": c_mean, "production": p_mean, "rel_change": c_mean / max(p_mean, 1e-12) - 1,
                  "ci_low": float(np.quantile(rel, a)), "ci_high": float(np.quantile(rel, 1 - a))}
    return out


def current_production(paths):
    return pointer_store(paths).read()


def production_history(paths) -> list[dict]:
    return pointer_store(paths).history()


def resolve_run(paths, entry) -> Path:
    """The run's folder on this machine. In S3 mode a run trained elsewhere is fetched (verified) into the cache
    MODEL_CACHE_DIR (default .local/registry-cache)."""
    p = run_path(paths, entry)
    if p.exists() or pointer_store(paths).kind == "local":
        return p
    import os
    model_dir, _ = registry_for(paths).fetch_run(entry["run_id"], Path(os.environ.get("MODEL_CACHE_DIR",
                                                                                      ".local/registry-cache")))
    return model_dir


def gate(cfg, paths, rd, log=print) -> dict:
    g = cfg["gate"]
    cand = json.loads((rd.run_dir / "metrics.json").read_text())
    checks = []

    finite = all(math.isfinite(v) for v in cand.values() if isinstance(v, (int, float)))
    checks.append({"check": "sanity: metrics finite and candidate recall > 0",
                   "passed": finite and cand["candidates/recall_union"] > 0})

    if g.get("must_beat_baseline", True):
        for m in ("all/ndcg@10", "all/recall@100"):
            c_v, b_v = cand[f"ranker/{m}"], cand[f"recent_pop/{m}"]
            checks.append({"check": f"beats recent popularity on {m}", "candidate": c_v, "baseline": b_v,
                           "passed": c_v > b_v})

    prod = current_production(paths)
    prod_metrics, newer = None, True
    if prod is None:
        log("gate: no production model yet -> only sanity + baseline checks")
    elif prod["run_id"] == rd.run_dir.name:
        log("gate: this run IS production -> nothing to compare")
    else:
        prod_dir = resolve_run(paths, prod)
        # Leakage guard: production's final model was trained on everything before ITS test start.
        # If that is later than the candidate's test start, production has SEEN the candidate's test
        # window -> its score would be inflated, and the candidate is trained on older data anyway.
        pm = prod_dir / "data" / "manifest.json"
        prod_t_test = json.loads(pm.read_text())["t_test"] if pm.exists() else None
        newer = prod_t_test is None or rd.t_test >= prod_t_test
        checks.append({"check": "candidate data is not older than production's (no test-window leakage)",
                       "passed": newer})
    if prod is not None and prod["run_id"] != rd.run_dir.name and newer:
        cand_pu_path = rd.run_dir / "per_user_eval.npz"
        if not cand_pu_path.exists():                          # run evaluated before Phase 7d
            log("gate: candidate has no per-user eval -> evaluating it again ...")
            evaluate_models(rd.run_dir, rd, cfg["serving"]["new_item_slots"], log, per_user_path=cand_pu_path)
        cache = rd.run_dir / f"production_eval_{prod['run_id']}.json"
        prod_pu_path = rd.run_dir / f"production_per_user_{prod['run_id']}.npz"
        if cache.exists() and prod_pu_path.exists():
            prod_metrics = json.loads(cache.read_text())
        else:
            log(f"gate: re-evaluating production ({prod['run_id']}) on this run's test window ...")
            prod_metrics = evaluate_models(prod_dir, rd, cfg["serving"]["new_item_slots"], log, per_user_path=prod_pu_path)
            cache.write_text(json.dumps(prod_metrics, indent=2))
        cand_pu, prod_pu = load_per_user(cand_pu_path), load_per_user(prod_pu_path)
        assert np.array_equal(cand_pu["uid"], prod_pu["uid"]), "candidate and production evaluated on different users"
        for m, r in paired_bootstrap(cand_pu, prod_pu, g["metrics"], g["bootstrap"], g["confidence"], g["seed"]).items():
            sure_worse = r["ci_high"] < -g["min_effect"]
            big_drop = r["rel_change"] < -g["max_point_drop"]
            why = "confidently worse" if sure_worse else "point drop too big" if big_drop else ""
            checks.append({"check": f"no regression on {m} (CI<-{g['min_effect']:.0%} or drop>{g['max_point_drop']:.0%} blocks)"
                                    + (f" -> {why}" if why else ""),
                           "candidate": r["candidate"], "production": r["production"], "rel_change": r["rel_change"],
                           "ci_low": r["ci_low"], "ci_high": r["ci_high"], "passed": not (sure_worse or big_drop)})

    passed = all(c["passed"] for c in checks)
    report = {"run_id": rd.run_dir.name, "production_run_id": prod["run_id"] if prod else None,
              "passed": passed, "checks": checks, "decided_at": datetime.now(timezone.utc).isoformat()}
    (rd.run_dir / "gate_report.json").write_text(json.dumps(report, indent=2))
    for c in checks:
        extra = "".join(f" {k}={c[k]:.4f}" for k in ("candidate", "production", "baseline") if k in c)
        if "rel_change" in c:
            extra += f" change={c['rel_change']:+.1%} CI=[{c['ci_low']:+.1%}, {c['ci_high']:+.1%}]"
        log(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}{extra}")
    log(f"gate: {'PASSED' if passed else 'FAILED'}")
    return report


def promote(paths, run_dir: Path, reason: str, log=print) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    try:
        rel = run_dir.resolve().relative_to(paths.models.resolve()).as_posix()
    except ValueError:
        rel = str(run_dir.resolve())
    entry = {"run_id": run_dir.name, "path": rel, "reason": reason,
             "promoted_at": datetime.now(timezone.utc).isoformat(),
             "metrics": {k: metrics.get(k) for k in ("ranker/all/ndcg@10", "ranker/all/recall@100")}}
    store = pointer_store(paths)
    if store.kind == "s3":                                    # engines elsewhere can only load a published run
        m = store.reg.run_manifest(run_dir.name)
        if m is None:
            raise RuntimeError(f"run {run_dir.name} is not in the registry: "
                               f"python -m pipeline.storage publish-run {run_dir.name}")
        entry |= {"path": f"runs/{run_dir.name}", "manifest_sha256": manifest_sha256(m), "gold": m["gold"]}
    entry = store.write(entry)
    log(f"promoted {run_dir.name} -> production v{entry['version']} ({reason}, {store.kind})")
    return entry


def rollback(paths, log=print) -> dict:
    store = pointer_store(paths)
    hist = store.history()
    if not hist:
        raise RuntimeError("no production history")
    cur = store.read()
    previous = [h for h in hist if cur is None or h["run_id"] != cur["run_id"]]
    if not previous:
        raise RuntimeError("nothing to roll back to")
    target = previous[-1]
    return promote(paths, run_path(paths, target), f"rollback from {cur['run_id'] if cur else None}", log)
