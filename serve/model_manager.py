"""ModelManager: owns the live Recommender and hot-swaps it (Phase 7d). The flow is described in serve/app.py.

Where the model comes from is serve/model_store.py (P2 phase C): local disk, or the S3 registry, where a new
production version is fetched + verified in this background thread while the old model keeps serving."""
import logging
import threading
import time

from serve import metrics as M
from serve.config import ServeConfig
from serve.experiments import ExperimentManager
from serve.model_store import ModelUnavailable, model_source
from serve.recommender import Recommender

log = logging.getLogger("recosphere")


class ModelManager:
    """Owns the live Recommender. `current` is swapped with a single assignment (atomic in CPython)."""

    def __init__(self, cfg: ServeConfig, source=None):
        self.cfg = cfg
        self.source = source or model_source(cfg)
        target = self.source.wanted()                 # S3: no pointer / broken run -> fail at startup, loudly
        self.current = Recommender(cfg, **self.source.materialize(target))
        self.current_key, self.current_version = target.key, target.version
        M.set_model(None, None, self.current.model_run, self.current.emb_version)
        self.experiments = ExperimentManager(cfg, Recommender, self.current.store)
        self._lock = threading.Lock()                 # one reload at a time per worker
        self._stop = threading.Event()
        self.reloads, self.last_error, self.last_check = 0, None, None
        self.failed_target = None                     # key of a broken model: don't retry it every tick
        self._thread = None
        if cfg.watch_seconds > 0:
            self._thread = threading.Thread(target=self._watch, name="model-watch", daemon=True)
            self._thread.start()

    def reload(self, reason: str, force: bool = False) -> dict:
        if not self._lock.acquire(blocking=False):
            return {"status": "busy", "detail": "a reload is already running"}
        try:
            target = self.source.wanted()
            name = target.run_id or (target.model_dir.name if target.model_dir else target.key)
            if not force and target.key == self.current_key:
                self._same_as_current(target)
                return {"status": "unchanged", "model_run": self.current.model_run}
            if not force and target.key == self.failed_target:
                return {"status": "skipped", "detail": f"{name} failed before; use force=true to retry",
                        "model_run": self.current.model_run}
            old = self.current
            log.info("model reload (%s): %s -> %s", reason, old.model_run, name)
            t0 = time.perf_counter()
            try:
                new = Recommender(self.cfg, store=old.store, **self.source.materialize(target))  # old keeps serving
                new.smoke_test()
            except Exception as e:                                    # keep serving the old model
                self.last_error = f"{name}: {type(e).__name__}: {e}"
                self.failed_target = target.key
                if isinstance(e, ModelUnavailable):
                    log.error("model reload REFUSED (%s), still serving %s", e, old.model_run)
                else:
                    log.exception("model reload FAILED, still serving %s", old.model_run)
                M.RELOADS.labels("failed").inc()
                return {"status": "failed", "error": self.last_error, "model_run": old.model_run}
            self.current = new                                        # <- the swap
            self.current_key, self.current_version = target.key, target.version
            self.reloads += 1
            M.RELOADS.labels("swapped").inc()
            M.set_model(old.model_run, old.emb_version, new.model_run, new.emb_version)
            self.last_error, self.failed_target = None, None
            secs = time.perf_counter() - t0
            log.info("model swapped to %s in %.1fs", new.model_run, secs)
            return {"status": "swapped", "from": old.model_run, "to": new.model_run, "seconds": round(secs, 1)}
        finally:
            self._lock.release()

    def tick(self):
        """One poll (the watcher thread calls this every MODEL_WATCH_SECONDS; tests call it directly)."""
        self.last_check = time.time()
        want = self.source.wanted()
        if want.key == self.current_key:
            self._same_as_current(want)
        elif want.key != self.failed_target:
            self.reload("production pointer changed")
        self.experiments.refresh()                                    # experiment config edited?

    def _same_as_current(self, target):
        """The pointer names the model already served (e.g. a rollback away from a refused run): no reload, but the
        version moved on (v2 -> v4) and the refused run's error is no longer a problem."""
        self.current_version = target.version
        if self.failed_target is not None:
            self.failed_target, self.last_error = None, None

    def _watch(self):
        while not self._stop.wait(self.cfg.watch_seconds):
            try:
                self.tick()
            except Exception:                                         # a bad pointer must not kill the thread
                log.exception("model watch error")

    def stop(self):
        self._stop.set()

    def info(self) -> dict:
        return {"reloads": self.reloads, "last_reload_error": self.last_error, "watch_seconds": self.cfg.watch_seconds,
                "model_source": self.source.kind, "production_version": self.current_version,
                **self.experiments.info()}
