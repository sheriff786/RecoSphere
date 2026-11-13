"""
A/B experiments in the serving API (Phase 8).

Experiment config (YAML, path = EXPERIMENT_CONFIG, default experiments/active.yaml; missing file = no experiment):

    name: ranker-seed2-vs-prod
    salt: ranker-seed2-vs-prod-v1      # NEW salt for every experiment -> assignments are independent
    enabled: true
    arms:
      - {name: control,   model: production,                traffic: 0.5}
      - {name: treatment, model: runs/20261112-045228,      traffic: 0.5}
      # optional per-arm serving overrides:  new_item_slots: 0

Assignment: bucket = sha256(salt + ":" + user_id) mod 10,000 -> arm by cumulative traffic.
  - deterministic: same user, same arm, on every worker and after every restart (no state to store)
  - salted: a user's arm in this experiment says nothing about their arm in the last one
  - by USER, not by request: a user must see one consistent experience, and the analysis unit = user

The arm named "control" is always "whatever production.json points to" (follows hot-swaps). Other arms
are pinned at experiment start: a run folder, or `production` = a pinned copy of today's production with
serving overrides (a POLICY test, e.g. new_item_slots: 0). Do not promote models during an experiment:
the arms would change under you. /health warns if production moved since the experiment started. If a treatment model fails to load, the experiment is DISABLED (all traffic ->
control) and /health says why: an experiment must never take the site down.
"""
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import yaml

log = logging.getLogger("recosphere")
BUCKETS = 10_000


def bucket(salt: str, user_id: str) -> int:
    h = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return int(h[:15], 16) % BUCKETS


def validate(exp: dict) -> dict:
    assert exp.get("name") and exp.get("salt"), "experiment needs name and salt"
    arms = exp.get("arms") or []
    assert len(arms) >= 2, "need at least 2 arms"
    names = [a["name"] for a in arms]
    assert len(set(names)) == len(names), "arm names must be unique"
    ctl = [a for a in arms if a["name"] == "control"]
    assert len(ctl) == 1 and ctl[0]["model"] == "production", "need one arm named 'control' with model: production"
    total = sum(float(a["traffic"]) for a in arms)
    assert abs(total - 1.0) < 1e-9, f"traffic must sum to 1 (got {total})"
    cum, edges = 0, []
    for a in arms:
        cum += round(float(a["traffic"]) * BUCKETS)
        edges.append(cum)
    edges[-1] = BUCKETS
    return {**exp, "edges": edges}


class Experiment:
    def __init__(self, cfg: dict):
        self.cfg = validate(cfg)
        self.name, self.salt = cfg["name"], cfg["salt"]
        self.arms = cfg["arms"]

    def assign(self, user_id: str) -> dict:
        b = bucket(self.salt, user_id)
        for arm, edge in zip(self.arms, self.cfg["edges"]):
            if b < edge:
                return arm
        return self.arms[-1]


class ExperimentManager:
    """Loads the treatment-arm Recommenders next to production; re-reads the config when it changes."""

    def __init__(self, serve_cfg, recommender_cls, store):
        self.serve_cfg, self.cls, self.store = serve_cfg, recommender_cls, store
        self.path = Path(os.environ.get("EXPERIMENT_CONFIG", "experiments/active.yaml"))
        self.exp, self.recs, self.error, self._mtime, self.prod_at_start = None, {}, None, None, None
        self._lock = threading.Lock()
        self.refresh(force=True)

    def _stamp(self):
        return self.path.stat().st_mtime if self.path.exists() else None

    def refresh(self, force: bool = False) -> dict:
        stamp = self._stamp()
        if not force and stamp == self._mtime:
            return {"status": "unchanged"}
        with self._lock:
            self._mtime = stamp
            if stamp is None:
                self.exp, self.recs, self.error = None, {}, None
                return {"status": "no experiment"}
            try:
                raw = yaml.safe_load(self.path.read_text())
                if not raw or not raw.get("enabled", True):
                    self.exp, self.recs, self.error = None, {}, None
                    return {"status": "disabled"}
                exp, recs = Experiment(raw), {}
                self.prod_at_start = self.serve_cfg.models
                for arm in exp.arms:
                    if arm["name"] == "control":
                        continue
                    mdir = self.prod_at_start if arm["model"] == "production" else Path(arm["model"])
                    mdir = mdir if mdir.is_absolute() else self.serve_cfg.models_root / mdir
                    cfg = replace(self.serve_cfg, new_item_slots=int(arm.get("new_item_slots", self.serve_cfg.new_item_slots)))
                    rec = self.cls(cfg, store=self.store, model_dir=mdir)
                    rec.smoke_test()
                    recs[arm["name"]] = rec
                self.exp, self.recs, self.error = exp, recs, None
                log.info("experiment '%s' active: %s", exp.name,
                         ", ".join(f"{a['name']}={a['model']}@{a['traffic']:.0%}" for a in exp.arms))
                return {"status": "active", "experiment": exp.name}
            except Exception as e:                               # bad config / model -> no experiment
                self.exp, self.recs = None, {}
                self.error = f"{type(e).__name__}: {e}"
                log.exception("experiment config rejected -> all traffic on control")
                return {"status": "rejected", "error": self.error}

    def route(self, user_id: str, control, force_arm: str | None = None):
        """-> (recommender, experiment name | None, arm name, forced?)"""
        exp = self.exp
        if exp is None:
            return control, None, "control", False
        if force_arm is not None:                                # QA: look at an arm on purpose (excluded from analysis)
            arm = next((a for a in exp.arms if a["name"] == force_arm), None)
            if arm is None:
                raise KeyError(force_arm)
            forced = True
        else:
            arm, forced = exp.assign(user_id), False
        rec = control if arm["name"] == "control" else self.recs[arm["name"]]
        return rec, exp.name, arm["name"], forced

    def info(self) -> dict:
        e = self.exp
        warn = None
        if e is not None and self.serve_cfg.models != self.prod_at_start:
            warn = (f"production changed during the experiment ({self.prod_at_start.name} -> "
                    f"{self.serve_cfg.models.name}): control arm is now a different model")
        return {"experiment": None if e is None else {"name": e.name, "salt": e.salt,
                                                      "arms": [{k: a.get(k) for k in ("name", "model", "traffic", "new_item_slots")}
                                                               for a in e.arms]},
                "experiment_error": self.error, "experiment_warning": warn}


class ExposureLogger:
    """One JSON line per served list: the 'who saw what' half of the A/B data. The client sends clicks with
    the same request_id (the other half). One file per worker process -> no locking across processes."""

    def __init__(self, directory: str | None):
        self.f = None
        if directory:
            d = Path(directory); d.mkdir(parents=True, exist_ok=True)
            self.f = open(d / f"exposures-{os.getpid()}.jsonl", "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    @staticmethod
    def new_request_id() -> str:
        return uuid.uuid4().hex

    def log(self, **rec):
        if self.f is None:
            return
        line = json.dumps({"ts": round(time.time(), 3), **rec}, separators=(",", ":"))
        with self._lock:
            self.f.write(line + "\n")

    def close(self):
        if self.f:
            self.f.close()
