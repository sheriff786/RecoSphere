"""P2 phase A: the S3 model registry (pipeline/storage.py).

Unit tests run on moto (an in-process S3, no server). The last test runs the same round trip against a REAL
S3 server (SeaweedFS locally, the CI container) when S3_ENDPOINT_URL is set, else it is skipped."""
import json
import os
import uuid
from pathlib import Path

import pytest

from pipeline import storage
from pipeline.storage import MANIFEST, IntegrityError, Registry, S3Config


def _files(d: Path, names, salt=""):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(f"{n}{salt}".encode() * 1000)


@pytest.fixture
def tree(tmp_path):
    run = tmp_path / "runs" / "r1"
    _files(run, storage.RUN_FILES + ["metrics.json", "two_tower_trainonly.pt"])
    gold = tmp_path / "gold"
    _files(gold, storage.GOLD_FILES + ["item_text_emb_minilm.npy"])
    return run, gold, tmp_path / "cache"


@pytest.fixture
def reg():
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        cfg = S3Config(endpoint_url=None, access_key="test", secret_key="test", bucket="recosphere")
        r = Registry(cfg, "Video_Games")
        r.ensure_bucket()
        yield r


def test_round_trip_is_byte_identical(reg, tree):
    run, gold, cache = tree
    m = reg.publish_run(run, gold, log=lambda *_: None)
    assert set(m["files"]) == set(storage.RUN_FILES) | {"metrics.json"}       # not the 45 MB train-only copy
    model_dir, gold_dir = reg.fetch_run("r1", cache)
    for n in m["files"]:
        assert (model_dir / n).read_bytes() == (run / n).read_bytes()
    for n in storage.GOLD_FILES + ["item_text_emb_minilm.npy"]:
        assert (gold_dir / n).read_bytes() == (gold / n).read_bytes()
    assert gold_dir.name == m["gold"] and reg.list_runs() == ["r1"]


def test_gold_hash_depends_on_content_only(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _files(a, storage.GOLD_FILES); _files(b, storage.GOLD_FILES)
    h = lambda d: storage.gold_hash(storage._describe(d, storage.gold_files(d)))
    assert h(a) == h(b)
    (b / "user_map.parquet").write_bytes(b"changed")
    assert h(a) != h(b)


def test_corrupt_object_is_refused_and_nothing_is_cached(reg, tree):
    run, gold, cache = tree
    reg.publish_run(run, gold, log=lambda *_: None)
    reg.s3.put_object(Bucket="recosphere", Key=f"{reg.run_prefix('r1')}/ranker.txt", Body=b"tampered")
    with pytest.raises(IntegrityError, match="ranker.txt"):
        reg.fetch_run("r1", cache)
    assert not (cache / "runs" / "r1").exists()
    assert not [p for p in (cache / "runs").iterdir()], "no temp folder left behind"


def test_unfinished_upload_is_not_fetched(reg, tree):
    run, gold, cache = tree
    reg.publish_run(run, gold, log=lambda *_: None)
    reg.s3.delete_object(Bucket="recosphere", Key=f"{reg.run_prefix('r1')}/{MANIFEST}")   # = crashed before the end
    with pytest.raises(IntegrityError, match="no manifest"):
        reg.fetch_run("r1", cache)


def test_runs_are_immutable(reg, tree):
    run, gold, _ = tree
    first = reg.publish_run(run, gold, log=lambda *_: None)
    assert reg.publish_run(run, gold, log=lambda *_: None) == first             # same bytes: no-op
    (run / "ranker.txt").write_bytes(b"retrained")
    with pytest.raises(IntegrityError, match="immutable"):
        reg.publish_run(run, gold, log=lambda *_: None)


def test_cached_copy_is_reused_but_a_tampered_cache_is_replaced(reg, tree):
    run, gold, cache = tree
    reg.publish_run(run, gold, log=lambda *_: None)
    model_dir, _ = reg.fetch_run("r1", cache)
    calls = []
    real = reg.s3.download_file
    reg.s3.download_file = lambda *a, **k: (calls.append(a[1]), real(*a, **k))
    reg.fetch_run("r1", cache)
    assert calls == [], "a verified cache is not downloaded again"
    m = json.loads((model_dir / MANIFEST).read_text()); m["files"]["ranker.txt"]["sha256"] = "0" * 64
    (model_dir / MANIFEST).write_text(json.dumps(m))                            # cache no longer matches S3
    reg.fetch_run("r1", cache)
    assert any(k.endswith("ranker.txt") for k in calls)


def test_missing_serving_file_is_an_error(reg, tree):
    run, gold, _ = tree
    (run / "faiss.index").unlink()
    with pytest.raises(FileNotFoundError, match="faiss.index"):
        reg.publish_run(run, gold, log=lambda *_: None)


def test_namespace_isolates_runs_and_gold():
    cfg = S3Config(endpoint_url=None, access_key="k", secret_key="s", bucket="b")
    r = Registry(cfg, "Video_Games", namespace="replay/q1", s3=object())
    assert r.run_prefix("x") == "models/Video_Games/replay/q1/runs/x"
    assert r.gold_prefix("h") == "data/Video_Games/replay/q1/gold/h"


def test_from_env_is_none_without_endpoint(monkeypatch):
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    assert S3Config.from_env() is None


@pytest.mark.skipif(not os.environ.get("S3_ENDPOINT_URL"), reason="needs a real S3 server (S3_ENDPOINT_URL)")
def test_real_server_round_trip(tree):
    """Same contract on the real server (SeaweedFS): multipart upload of a >8 MB file, verify, corrupt -> refused.
    Uses a throwaway namespace so it never touches real runs."""
    run, gold, cache = tree
    (run / "two_tower.pt").write_bytes(os.urandom(12 * 1024 * 1024))          # > boto3's 8 MB multipart threshold
    reg = Registry(S3Config.from_env(), "Video_Games", namespace=f"test/{uuid.uuid4().hex[:8]}")
    reg.ensure_bucket()
    try:
        reg.publish_run(run, gold, log=lambda *_: None)
        model_dir, _ = reg.fetch_run("r1", cache)
        assert (model_dir / "two_tower.pt").read_bytes() == (run / "two_tower.pt").read_bytes()
        reg.s3.put_object(Bucket=reg.cfg.bucket, Key=f"{reg.run_prefix('r1')}/faiss.index", Body=b"x")
        with pytest.raises(IntegrityError):
            reg.fetch_run("r1", tmp := cache / "second")
        assert not (tmp / "runs" / "r1").exists()
    finally:
        for prefix in (reg.models, reg.gold):                               # both live under the test namespace
            for o in reg.s3.list_objects_v2(Bucket=reg.cfg.bucket, Prefix=prefix).get("Contents", []):
                reg.s3.delete_object(Bucket=reg.cfg.bucket, Key=o["Key"])


def test_load_s3_env_reads_only_s3_lines_and_never_overrides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=postgresql://secret\nS3_BUCKET=from-file\nS3_ACCESS_KEY_ID='k1'\n")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.setenv("S3_BUCKET", "from-env")
    storage.load_s3_env(env)
    assert "DATABASE_URL" not in os.environ                                    # other secrets are not read
    assert os.environ["S3_BUCKET"] == "from-env" and os.environ["S3_ACCESS_KEY_ID"] == "k1"
