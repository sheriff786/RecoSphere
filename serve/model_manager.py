"""ModelManager: owns the live Recommender and hot-swaps it (Phase 7d). The flow is described in serve/app.py."""
import logging
import threading
import time

from serve import metrics as M
from serve.config import ServeConfig
from serve.experiments import ExperimentManager
from serve.recommender import Recommender

log = logging.getLogger("recosphere")


class ModelManager:
    """Owns the live Recommender. `current` is swapped with a single assignment (atomic in CPython)."""

    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self.current = Recommender(cfg)
        M.set_model(None, None, self.current.model_run, self.current.emb_version)
        self.experiments = ExperimentManager(cfg, Recommender, self.current.store)
        self._lock = threading.Lock()                 # one reload at a time per worker
        self._stop = threading.Event()
        self.reloads, self.last_error, self.last_check = 0, None, None
        self.failed_target = None                     # don't retry a broken model every tick
        self._thread = None
        if cfg.watch_seconds > 0:
            self._thread = threading.Thread(target=self._watch, name="model-watch", daemon=True)
            self._thread.start()

    def wanted_dir(self):
        return self.cfg.models                        # re-reads production.json

    def reload(self, reason: str, force: bool = False) -> dict:
        if not self._lock.acquire(blocking=False):
            return {"status": "busy", "detail": "a reload is already running"}
        try:
            target = self.wanted_dir()
            if not force and target == self.current.model_dir:
                return {"status": "unchanged", "model_run": self.current.model_run}
            if not force and target == self.failed_target:
                return {"status": "skipped", "detail": f"{target.name} failed before; use force=true to retry",
                        "model_run": self.current.model_run}
            old = self.current
            log.info("model reload (%s): %s -> %s", reason, old.model_dir.name, target.name)
            t0 = time.perf_counter()
            try:
                new = Recommender(self.cfg, store=old.store)          # old keeps serving meanwhile
                new.smoke_test()
            except Exception as e:                                    # keep serving the old model
                self.last_error = f"{target.name}: {type(e).__name__}: {e}"
                self.failed_target = target
                log.exception("model reload FAILED, still serving %s", old.model_run)
                M.RELOADS.labels("failed").inc()
                return {"status": "failed", "error": self.last_error, "model_run": old.model_run}
            self.current = new                                        # <- the swap
            self.reloads += 1
            M.RELOADS.labels("swapped").inc()
            M.set_model(old.model_run, old.emb_version, new.model_run, new.emb_version)
            self.last_error, self.failed_target = None, None
            secs = time.perf_counter() - t0
            log.info("model swapped to %s in %.1fs", new.model_run, secs)
            return {"status": "swapped", "from": old.model_run, "to": new.model_run, "seconds": round(secs, 1)}
        finally:
            self._lock.release()

    def _watch(self):
        while not self._stop.wait(self.cfg.watch_seconds):
            try:
                self.last_check = time.time()
                want = self.wanted_dir()
                if want == self.current.model_dir and self.failed_target is not None:
                    self.failed_target, self.last_error = None, None   # pointer rolled back: issue resolved
                elif want not in (self.current.model_dir, self.failed_target):
                    self.reload("production.json changed")
                self.experiments.refresh()                                # experiment config edited?
            except Exception:                                         # a bad pointer must not kill the thread
                log.exception("model watch error")

    def stop(self):
        self._stop.set()

    def info(self) -> dict:
        return {"reloads": self.reloads, "last_reload_error": self.last_error, "watch_seconds": self.cfg.watch_seconds,
                **self.experiments.info()}
