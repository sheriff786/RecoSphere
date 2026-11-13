"""Phase 7d: ModelManager hot-swap logic, with a fake Recommender (no models needed)."""
import json
from dataclasses import replace
from pathlib import Path

import serve.app as A
from serve.config import ServeConfig


class FakeRec:
    def __init__(self, cfg, store=None):
        self.model_dir = cfg.models
        if (self.model_dir / "BROKEN").exists():
            raise RuntimeError("cannot load")
        self.model_run, self.store, self.emb_version = self.model_dir.name, store or object(), "e"

    def smoke_test(self):
        pass


def setup(tmp_path, monkeypatch):
    root = tmp_path / "models" / "Video_Games"
    for r in ("a", "b", "bad"):
        (root / "runs" / r).mkdir(parents=True)
    (root / "runs" / "bad" / "BROKEN").touch()
    monkeypatch.setenv("RANKFORGE_REPO", str(tmp_path)); monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.setenv("EXPERIMENT_CONFIG", str(tmp_path / "none.yaml"))
    monkeypatch.setattr(A, "Recommender", FakeRec)
    point = lambda r: (root / "production.json").write_text(json.dumps({"run_id": r, "path": f"runs/{r}"}))
    point("a")
    cfg = replace(ServeConfig.from_env(), watch_seconds=0)
    return A.ModelManager(cfg), point


def test_swap_unchanged_and_shared_store(tmp_path, monkeypatch):
    mgr, point = setup(tmp_path, monkeypatch)
    store = mgr.current.store
    assert mgr.reload("t")["status"] == "unchanged"
    point("b")
    r = mgr.reload("t")
    assert r["status"] == "swapped" and mgr.current.model_run == "b"
    assert mgr.current.store is store                                  # live user state is kept


def test_broken_model_keeps_old_and_is_not_retried(tmp_path, monkeypatch):
    mgr, point = setup(tmp_path, monkeypatch)
    point("bad")
    r = mgr.reload("t")
    assert r["status"] == "failed" and mgr.current.model_run == "a" and "bad" in mgr.last_error
    assert mgr.reload("t")["status"] == "skipped"                     # no retry loop
    assert mgr.reload("t", force=True)["status"] == "failed"         # explicit retry still possible
    point("b")
    assert mgr.reload("t")["status"] == "swapped" and mgr.last_error is None
