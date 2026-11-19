"""P2 phase C: the engine loads production from the S3 registry (serve/model_store.py + ModelManager), on moto.
FakeRec reads the fetched files, so the tests see exactly what the engine would load."""
import json
from dataclasses import replace

import pytest

import pipeline.gate as G
import pipeline.pointer as P
import serve.model_manager as A
from pipeline.storage import GOLD_FILES, RUN_FILES, Registry, S3Config
from serve.config import ServeConfig
from serve.model_store import ModelUnavailable, S3Source
from tests.test_pointer import FakePaths


class FakeRec:
    """Recommender's constructor; 'loads' ranker.txt (its text = the run id) and checks the gold folder."""
    def __init__(self, cfg, store=None, model_dir=None, gold_dir=None, run_id=None):
        text = (model_dir / "ranker.txt").read_text()
        if text.startswith("BROKEN"):
            raise RuntimeError("cannot load")
        assert (gold_dir / "user_map.parquet").exists()
        self.model_dir, self.gold_dir, self.model_run = model_dir, gold_dir, run_id
        self.loaded = text
        self.store, self.emb_version = store or object(), "e"

    def smoke_test(self):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        reg = Registry(S3Config(None, "test", "test", "recosphere"), "Video_Games")
        reg.ensure_bucket()
        paths = FakePaths(tmp_path / "trainer")
        paths.gold.mkdir(parents=True)
        for n in GOLD_FILES:
            (paths.gold / n).write_bytes(n.encode())
        monkeypatch.setenv("MODEL_REGISTRY", "s3")
        monkeypatch.setenv("EXPERIMENT_CONFIG", str(tmp_path / "none.yaml"))
        monkeypatch.setattr(P, "registry_for", lambda _p: reg)
        monkeypatch.setattr(A, "Recommender", FakeRec)

        def publish(run_id, ranker=None, promote=True):
            d = paths.runs / run_id
            d.mkdir(parents=True, exist_ok=True)
            for n in RUN_FILES:
                (d / n).write_text(ranker or run_id if n == "ranker.txt" else n)
            reg.publish_run(d, paths.gold, log=lambda *_: None)
            if promote:
                G.promote(paths, d, "t", log=lambda *_: None)

        def manager():
            cfg = replace(ServeConfig.from_env(), watch_seconds=0)
            return A.ModelManager(cfg, source=S3Source(cfg, registry=reg, cache_dir=tmp_path / "engine-cache"))

        yield reg, paths, publish, manager


def test_engine_starts_from_the_s3_pointer(env):
    reg, _, publish, manager = env
    publish("a")
    mgr = manager()
    assert mgr.current.model_run == "a" and mgr.current.loaded == "a"
    assert mgr.current_version == 1 and mgr.info()["model_source"] == "s3"
    assert "engine-cache" in str(mgr.current.model_dir) and mgr.current.gold_dir.parent.name == "gold"


def test_promotion_is_picked_up_by_the_watcher(env):
    _, paths, publish, manager = env
    publish("a")
    mgr = manager()
    mgr.tick()
    assert mgr.reloads == 0                                       # unchanged pointer: nothing fetched
    publish("b")
    mgr.tick()
    assert mgr.current.model_run == "b" and mgr.current.loaded == "b" and mgr.current_version == 2
    G.rollback(paths, log=lambda *_: None)
    mgr.tick()
    assert mgr.current.model_run == "a" and mgr.current_version == 3


def test_corrupt_run_is_refused_old_model_keeps_serving_and_is_not_retried(env):
    reg, paths, publish, manager = env
    publish("a")
    mgr = manager()
    publish("c", promote=False)
    reg.s3.put_object(Bucket="recosphere", Key=f"{reg.run_prefix('c')}/item_emb.npy", Body=b"bit rot")
    G.promote(paths, paths.runs / "c", "t", log=lambda *_: None)
    mgr.tick()
    assert mgr.current.model_run == "a" and "item_emb.npy" in mgr.last_error
    calls = []
    real = reg.fetch_run
    reg.fetch_run = lambda *a, **k: (calls.append(a), real(*a, **k))[1]
    mgr.tick()
    assert calls == [], "a refused run is not downloaded again every tick"
    G.rollback(paths, log=lambda *_: None)                       # pointer back to a (v3): problem resolved
    assert mgr.reload("admin rollback")["status"] == "unchanged"  # what /admin/rollback does right after
    assert mgr.failed_target is None and mgr.last_error is None and mgr.current.model_run == "a"
    assert mgr.current_version == 3, "the served version follows the pointer even without a reload"


def test_unloadable_model_keeps_old(env):
    _, _, publish, manager = env
    publish("a")
    mgr = manager()
    publish("broken", ranker="BROKEN")                            # verifies fine, fails to load
    mgr.tick()
    assert mgr.current.model_run == "a" and "cannot load" in mgr.last_error


def test_manifest_must_be_the_promoted_one(env):
    reg, _, publish, manager = env
    publish("a")
    mgr = manager()
    publish("b")
    m = reg.run_manifest("b"); m["created_at"] = "rewritten"
    reg.put_json(f"{reg.run_prefix('b')}/MANIFEST.json", m)        # someone replaced the manifest after promotion
    mgr.tick()
    assert mgr.current.model_run == "a" and "not the one that was promoted" in mgr.last_error


def test_no_pointer_fails_at_startup(env):
    *_, manager = env
    with pytest.raises(ModelUnavailable, match="no production pointer"):
        manager()


def test_admin_paths_follow_the_engines_namespace(monkeypatch):
    """A demo engine (MODEL_NAMESPACE) must roll back ITS pointer, never the real one; local mode ignores it."""
    from serve.model_store import pipeline_paths
    cfg = ServeConfig.from_env()
    monkeypatch.setenv("MODEL_NAMESPACE", "demo/x")
    monkeypatch.setenv("MODEL_REGISTRY", "s3")
    assert pipeline_paths(cfg).namespace == "demo/x"
    monkeypatch.setenv("MODEL_REGISTRY", "local")
    assert pipeline_paths(cfg).namespace is None
