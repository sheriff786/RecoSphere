"""
RecoSphere training pipeline (Phase 7a–7c).

    python -m pipeline.run                                  # full run: prepare -> ... -> gate -> promote
    python -m pipeline.run --set two_tower.epochs=2         # override any config value
    python -m pipeline.run --run-id 20261111-120000 --steps train_ranker,evaluate,gate,promote
    python -m pipeline.run --register-existing              # wrap notebook 03-06 models as the first production run
    python -m pipeline.run --run-id <id> --steps promote --force   # promote even if the gate failed
    python -m pipeline.run --rollback                       # production -> previous model

Exit codes: 0 ok, 2 gate failed (not promoted) — so a scheduler (Airflow, cron) can react.
"""
import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone

import yaml

from pipeline.config import Paths, load_config
from pipeline.data import load_run_data, prepare
from pipeline.gate import current_production, gate, promote, rollback
from pipeline.steps import build_index, evaluate_step, train_ranker, train_retrieval
from pipeline.tracking import Tracker

STEPS = ["prepare", "train_retrieval", "build_index", "train_ranker", "evaluate", "gate", "promote"]
SERVING_FILES = ["two_tower.pt", "two_tower_trainonly.pt", "item_emb.npy", "faiss.index", "faiss_meta.json",
                 "ranker.txt", "ranker_meta.json"]


def setup_log(run_dir=None):
    logger = logging.getLogger("pipeline")
    logger.handlers.clear(); logger.setLevel(logging.INFO); logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger.info


def register_existing(cfg, paths, log):
    """Notebook-built models (models/<cat>/*.pt, ranker.txt ...) become run 'nb-<date>' and production."""
    missing = [f for f in SERVING_FILES if f != "two_tower_trainonly.pt" and not (paths.models / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing in {paths.models}: {missing}")
    run_id = "nb-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id; run_dir.mkdir(parents=True)
    log = setup_log(run_dir)
    for f in SERVING_FILES:
        if (paths.models / f).exists():
            shutil.copy2(paths.models / f, run_dir / f)
    cfg["split"]["mode"] = "gold"
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    tr = Tracker(cfg, paths, run_id, log)
    tr.params(cfg); tr.tags({"source": "notebooks"})
    tr.tags({"data_hash": prepare(cfg, paths, run_dir, log)["data_hash"]})
    rd = load_run_data(run_dir, paths.gold)
    tr.metrics(evaluate_step(cfg, rd, log))
    for a in ("metrics.json", "serving_policy.json"):
        tr.artifact(run_dir / a)
    promote(paths, run_dir, "registered notebook models", log)
    tr.tags({"promoted": "true"}); tr.end()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. two_tower.epochs=2")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--force", action="store_true", help="promote even if the gate failed")
    ap.add_argument("--register-existing", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    paths = Paths(cfg)
    if a.rollback:
        rollback(paths, setup_log()); return
    if a.register_existing:
        register_existing(cfg, paths, setup_log()); return

    run_id = a.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = run_dir / "config.yaml"
    if saved.exists() and a.config is None:
        policy = {k: cfg[k] for k in ("gate", "mlflow")}       # deployment policy = TODAY's config
        cfg = load_config(saved, a.set)                       # continuing a run: keep ITS training config
        cfg.update(policy)
    saved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log = setup_log(run_dir)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"unknown steps {bad}; valid: {STEPS}")
    log(f"=== run {run_id} | steps: {', '.join(steps)} ===")

    tr = Tracker(cfg, paths, run_id, log)
    tr.params(cfg)
    rd, exit_code = None, 0
    try:
        for step in steps:
            log(f"--- {step}")
            if step == "prepare":
                tr.tags({"data_hash": prepare(cfg, paths, run_dir, log)["data_hash"], "split_mode": cfg["split"]["mode"]})
                rd = None
                continue
            rd = rd or load_run_data(run_dir, paths.gold)
            if step == "train_retrieval":
                tr.metrics(train_retrieval(cfg, rd, log))
            elif step == "build_index":
                build_index(cfg, rd, log)
                tr.tags({"emb_version": json.loads((run_dir / "faiss_meta.json").read_text())["emb_version"]})
            elif step == "train_ranker":
                tr.metrics(train_ranker(cfg, rd, log))
            elif step == "evaluate":
                tr.metrics(evaluate_step(cfg, rd, log))
                tr.artifact(run_dir / "metrics.json"); tr.artifact(run_dir / "serving_policy.json")
            elif step == "gate":
                rep = gate(cfg, paths, rd, log)
                tr.tags({"gate_passed": rep["passed"], "compared_to": rep["production_run_id"]})
                tr.artifact(run_dir / "gate_report.json")
            elif step == "promote":
                rep_path = run_dir / "gate_report.json"
                rep = json.loads(rep_path.read_text()) if rep_path.exists() else None
                cur = current_production(paths)
                stale = rep is not None and rep["production_run_id"] != (cur["run_id"] if cur else None) \
                    and not (cur and cur["run_id"] == run_id)
                if stale:
                    log(f"gate report is stale (made against {rep['production_run_id']}, production is now "
                        f"{cur['run_id'] if cur else None}) -> re-run the gate step")
                passed = bool(rep and rep["passed"] and not stale)
                if cur and cur["run_id"] == run_id:
                    log(f"{run_id} is already production")
                elif passed or a.force:
                    promote(paths, run_dir, "gate passed" if passed else "forced", log)
                    tr.tags({"promoted": "true"})   # "ever promoted"; production.json is the live truth
                else:
                    log(f"NOT promoted (gate failed or not run). Production stays: {cur['run_id'] if cur else None}")
                    exit_code = 2
        tr.end()
    except Exception:
        tr.end("FAILED")
        raise
    log(f"=== done: {run_dir}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
