"""P2 phase B: the production pointer on local disk and on S3 (pipeline/pointer.py, pipeline/gate.py).
The same promote/rollback contract runs on both stores; S3-only behaviour (compare-and-swap, crash between the two
writes, unpublished runs) is tested on moto, and the race once more on the real server when S3_ENDPOINT_URL is set."""
import json
import os
import uuid
from pathlib import Path

import pytest

import pipeline.gate as G
import pipeline.pointer as P
from pipeline.storage import RUN_FILES, GOLD_FILES, Registry, S3Config


class FakePaths:
    def __init__(self, root: Path, namespace=None):
        self.category, self.namespace = "Video_Games", namespace
        self.models = root / "models" / "Video_Games"
        self.runs = self.models / "runs"
        self.gold = root / "gold"
        self.production_pointer = self.models / "production.json"
        self.production_history = self.models / "production_history.jsonl"


def make_run(paths, run_id):
    d = paths.runs / run_id
    (d / "data").mkdir(parents=True, exist_ok=True)
    for n in RUN_FILES:
        (d / n).write_bytes(f"{run_id}/{n}".encode())
    (d / "metrics.json").write_text(json.dumps({"ranker/all/ndcg@10": 0.02, "ranker/all/recall@100": 0.13}))
    (d / "data" / "manifest.json").write_text(json.dumps({"t_test": 100}))
    return d


def make_gold(paths):
    paths.gold.mkdir(parents=True, exist_ok=True)
    for n in GOLD_FILES:
        (paths.gold / n).write_bytes(n.encode())


@pytest.fixture
def moto_registry():
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        reg = Registry(S3Config(None, "test", "test", "recosphere"), "Video_Games")
        reg.ensure_bucket()
        yield reg


@pytest.fixture(params=["local", "s3"])
def store_paths(request, tmp_path, monkeypatch):
    """(paths, registry or None) with MODEL_REGISTRY set; in s3 mode the registry is moto."""
    paths = FakePaths(tmp_path)
    make_gold(paths)
    monkeypatch.setenv("MODEL_REGISTRY", request.param)
    if request.param == "local":
        yield paths, None
        return
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        reg = Registry(S3Config(None, "test", "test", "recosphere"), "Video_Games")
        reg.ensure_bucket()
        monkeypatch.setattr(P, "registry_for", lambda _p: reg)
        yield paths, reg


def promote(paths, reg, run_id, reason="x"):
    d = make_run(paths, run_id)
    if reg is not None:
        reg.publish_run(d, paths.gold, log=lambda *_: None)
    return G.promote(paths, d, reason, log=lambda *_: None)


def test_same_promote_rollback_contract_on_both_stores(store_paths):
    paths, reg = store_paths
    assert G.current_production(paths) is None and G.production_history(paths) == []
    assert promote(paths, reg, "a")["version"] == 1
    assert promote(paths, reg, "b")["version"] == 2
    assert G.current_production(paths)["run_id"] == "b"
    back = G.rollback(paths, log=lambda *_: None)
    assert back["run_id"] == "a" and back["version"] == 3 and back["reason"] == "rollback from b"
    assert [h["run_id"] for h in G.production_history(paths)] == ["a", "b", "a"]
    assert G.rollback(paths, log=lambda *_: None)["run_id"] == "b"           # rollback of a rollback = toggle
    if reg is not None:
        cur = G.current_production(paths)
        assert cur["gold"] and cur["manifest_sha256"] == P.manifest_sha256(reg.run_manifest("b"))


def test_s3_refuses_an_unpublished_run(tmp_path, monkeypatch, moto_registry):
    paths = FakePaths(tmp_path)
    monkeypatch.setenv("MODEL_REGISTRY", "s3")
    monkeypatch.setattr(P, "registry_for", lambda _p: moto_registry)
    with pytest.raises(RuntimeError, match="publish-run a"):
        G.promote(paths, make_run(paths, "a"), "x", log=lambda *_: None)
    assert G.current_production(paths) is None


def test_s3_concurrent_promotion_loses_instead_of_overwriting(tmp_path, monkeypatch, moto_registry):
    """Two promoters read the same pointer; the second write must fail (412), not silently win."""
    paths = FakePaths(tmp_path); make_gold(paths)
    monkeypatch.setenv("MODEL_REGISTRY", "s3")
    monkeypatch.setattr(P, "registry_for", lambda _p: moto_registry)
    promote(paths, moto_registry, "a")
    store = P.S3Pointer(moto_registry)
    stale = store._read_with_etag()                     # promoter 2 reads ...
    promote(paths, moto_registry, "b")                  # ... promoter 1 wins meanwhile
    monkeypatch.setattr(store, "_read_with_etag", lambda: stale)
    with pytest.raises(P.ConcurrentPromotion):
        store.write({"run_id": "c", "path": "runs/c"})
    assert G.current_production(paths)["run_id"] == "b" and G.current_production(paths)["version"] == 2


def test_s3_first_promotion_race_has_one_winner(moto_registry):
    a, b = P.S3Pointer(moto_registry), P.S3Pointer(moto_registry)
    monkeypatch_read = lambda: (None, None)             # both saw "no pointer yet"
    a._read_with_etag = b._read_with_etag = monkeypatch_read
    a.write({"run_id": "a"})
    with pytest.raises(P.ConcurrentPromotion):
        b.write({"run_id": "b"})


def test_s3_crash_between_pointer_and_history_loses_nothing(tmp_path, monkeypatch, moto_registry):
    paths = FakePaths(tmp_path); make_gold(paths)
    monkeypatch.setenv("MODEL_REGISTRY", "s3")
    monkeypatch.setattr(P, "registry_for", lambda _p: moto_registry)
    promote(paths, moto_registry, "a"); promote(paths, moto_registry, "b")
    moto_registry.s3.delete_object(Bucket="recosphere", Key="models/Video_Games/history/00000002.json")
    assert [h["run_id"] for h in G.production_history(paths)] == ["a", "b"]
    assert G.rollback(paths, log=lambda *_: None)["run_id"] == "a"


def test_s3_resolve_run_fetches_a_run_trained_elsewhere(tmp_path, monkeypatch, moto_registry):
    trainer = FakePaths(tmp_path / "trainer"); make_gold(trainer)
    monkeypatch.setenv("MODEL_REGISTRY", "s3")
    monkeypatch.setattr(P, "registry_for", lambda _p: moto_registry)
    monkeypatch.setattr(G, "registry_for", lambda _p: moto_registry)
    promote(trainer, moto_registry, "a")
    other = FakePaths(tmp_path / "other")                # a machine without the run folder
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))
    d = G.resolve_run(other, G.current_production(other))
    assert (d / "ranker.txt").read_bytes() == b"a/ranker.txt"
    assert json.loads((d / "data" / "manifest.json").read_text())["t_test"] == 100   # leakage guard input


def test_model_registry_setting(tmp_path, monkeypatch):
    paths = FakePaths(tmp_path)
    monkeypatch.delenv("MODEL_REGISTRY", raising=False)
    assert P.pointer_store(paths).kind == "local"                              # default: local, opt-in to s3
    monkeypatch.setenv("MODEL_REGISTRY", "S3x")
    with pytest.raises(ValueError):
        P.pointer_store(paths)
    monkeypatch.setenv("MODEL_REGISTRY", "s3")
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    with pytest.raises(RuntimeError, match="S3_ENDPOINT_URL"):
        P.pointer_store(paths)


@pytest.mark.skipif(not os.environ.get("S3_ENDPOINT_URL"), reason="needs a real S3 server (S3_ENDPOINT_URL)")
def test_real_server_compare_and_swap():
    """The race on the real server (SeaweedFS honours If-Match / If-None-Match). Throwaway namespace."""
    reg = Registry(S3Config.from_env(), "Video_Games", namespace=f"test/{uuid.uuid4().hex[:8]}")
    reg.ensure_bucket()
    try:
        a, b = P.S3Pointer(reg), P.S3Pointer(reg)
        assert a.write({"run_id": "r1"})["version"] == 1
        stale = b._read_with_etag()
        assert a.write({"run_id": "r2"})["version"] == 2
        b._read_with_etag = lambda: stale
        with pytest.raises(P.ConcurrentPromotion):
            b.write({"run_id": "r3"})
        assert [h["run_id"] for h in a.history()] == ["r1", "r2"]
    finally:
        for o in reg.s3.list_objects_v2(Bucket=reg.cfg.bucket, Prefix=reg.models).get("Contents", []):
            reg.s3.delete_object(Bucket=reg.cfg.bucket, Key=o["Key"])
