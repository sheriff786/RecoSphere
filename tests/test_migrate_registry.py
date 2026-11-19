"""P2 phase D: moving the local production history to S3, and pipeline.run publishing before it promotes (moto)."""
import pytest

import pipeline.gate as G
import pipeline.pointer as P
import pipeline.run as R
from pipeline.migrate_registry import migrate
from pipeline.pointer import LocalPointer, S3Pointer
from tests.test_pointer import FakePaths, make_gold, make_run, moto_registry  # noqa: F401 (fixture)


@pytest.fixture
def local_history(tmp_path, monkeypatch, moto_registry):
    """Two local promotions (like the real models/Video_Games): nb -> a."""
    paths = FakePaths(tmp_path); make_gold(paths)
    monkeypatch.setenv("MODEL_REGISTRY", "local")
    G.promote(paths, make_run(paths, "nb"), "registered notebook models", log=lambda *_: None)
    G.promote(paths, make_run(paths, "a"), "gate passed", log=lambda *_: None)
    return paths, moto_registry


def test_migration_replays_the_history_in_order(local_history):
    paths, reg = local_history
    local = LocalPointer(paths).history()
    written = migrate(paths, reg, log=lambda *_: None)
    s3 = S3Pointer(reg)
    assert [(h["version"], h["run_id"], h["reason"]) for h in s3.history()] == \
        [(1, "nb", "registered notebook models"), (2, "a", "gate passed")]
    assert [h["promoted_at"] for h in s3.history()] == [h["promoted_at"] for h in local]   # dates kept
    assert s3.read()["run_id"] == LocalPointer(paths).read()["run_id"] == "a"
    assert all(h["manifest_sha256"] and h["gold"] and h["migrated_at"] for h in written)
    assert sorted(reg.list_runs()) == ["a", "nb"]
    assert LocalPointer(paths).history() == local                                         # local files untouched


def test_migration_runs_once(local_history):
    paths, reg = local_history
    migrate(paths, reg, log=lambda *_: None)
    with pytest.raises(RuntimeError, match="already exists"):
        migrate(paths, reg, log=lambda *_: None)
    assert S3Pointer(reg).read()["version"] == 2


def test_dry_run_writes_nothing(local_history):
    paths, reg = local_history
    migrate(paths, reg, dry_run=True, log=lambda *_: None)
    assert S3Pointer(reg).read() is None and reg.list_runs() == []


def test_history_must_end_at_local_production(local_history):
    paths, reg = local_history
    paths.production_pointer.write_text(paths.production_pointer.read_text().replace('"a"', '"nb"'))
    with pytest.raises(RuntimeError, match="history ends with a"):
        migrate(paths, reg, log=lambda *_: None)


def test_pipeline_publishes_before_promoting_in_s3_mode(tmp_path, monkeypatch, moto_registry):
    paths = FakePaths(tmp_path); make_gold(paths)
    monkeypatch.setenv("MODEL_REGISTRY", "s3")
    monkeypatch.setattr(P, "registry_for", lambda _p: moto_registry)
    monkeypatch.setattr(R, "registry_for", lambda _p: moto_registry)
    d = make_run(paths, "a")
    R.publish_for_promotion(paths, d, log=lambda *_: None)
    G.promote(paths, d, "gate passed", log=lambda *_: None)                # would raise if not published
    assert moto_registry.run_manifest("a") is not None and G.current_production(paths)["run_id"] == "a"
    assert not paths.production_pointer.exists()                          # S3 mode never writes the local pointer


def test_pipeline_publishes_nothing_in_local_mode(tmp_path, monkeypatch, moto_registry):
    paths = FakePaths(tmp_path); make_gold(paths)
    monkeypatch.setenv("MODEL_REGISTRY", "local")
    R.publish_for_promotion(paths, make_run(paths, "a"), log=lambda *_: None)
    assert moto_registry.list_runs() == []
