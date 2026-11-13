"""Phase 7 unit tests: config overrides, gate logic, promotion pointer, rollback. No training needed."""
import json
from pathlib import Path

import numpy as np
import pytest

import pipeline.gate as G
from pipeline.config import load_config


class FakePaths:
    def __init__(self, root: Path):
        self.models = root / "models" / "Video_Games"
        self.runs = self.models / "runs"
        self.production_pointer = self.models / "production.json"
        self.production_history = self.models / "production_history.jsonl"


class FakeRD:
    def __init__(self, run_dir, t_test):
        self.run_dir, self.t_test = run_dir, t_test


GATE_METRICS = ["ranker/all/ndcg@10", "ranker/all/recall@100", "candidates/recall_union"]


def make_run(paths, run_id, t_test, ndcg, recall, pop=0.05, n=4000, seed=0):
    """A fake run whose per-user metrics have the given means (Bernoulli-ish, like real NDCG/recall)."""
    d = paths.runs / run_id
    (d / "data").mkdir(parents=True)
    (d / "data" / "manifest.json").write_text(json.dumps({"t_test": t_test}))
    rng = np.random.default_rng(seed)
    base = rng.random(n)                                       # shared "user difficulty" -> paired data
    pu = {"uid": np.arange(n),
          "ranker/all/ndcg@10": (base < ndcg).astype(np.float32),
          "ranker/all/recall@100": (base < recall).astype(np.float32),
          "candidates/recall_union": (base < min(1, recall * 1.2)).astype(np.float32)}
    np.savez(d / "per_user_eval.npz", **{k.replace("/", "|"): v for k, v in pu.items()})
    m = {k: float(v.mean()) for k, v in pu.items() if k != "uid"}
    m.update({"recent_pop/all/ndcg@10": pop, "recent_pop/all/recall@100": pop})
    (d / "metrics.json").write_text(json.dumps(m))
    return d, m


@pytest.fixture
def env(tmp_path, monkeypatch):
    paths = FakePaths(tmp_path)
    cfg = load_config(None, ["gate.bootstrap=500"])
    # "re-evaluating production on the candidate window" = return that run's own metrics file
    def fake_eval(d, rd, k, log, per_user_path=None):
        if per_user_path is not None:
            per_user_path.write_bytes((Path(d) / "per_user_eval.npz").read_bytes())
        return json.loads((Path(d) / "metrics.json").read_text())
    monkeypatch.setattr(G, "evaluate_models", fake_eval)
    return cfg, paths


def test_overrides_are_typed():
    cfg = load_config(None, ["two_tower.epochs=7", "gate.max_point_drop=0.03", "split.mode=cutoff", "mlflow.enabled=false"])
    assert cfg["two_tower"]["epochs"] == 7 and cfg["gate"]["max_point_drop"] == 0.03
    assert cfg["split"]["mode"] == "cutoff" and cfg["mlflow"]["enabled"] is False


def test_first_run_needs_only_sanity_and_baseline(env):
    cfg, paths = env
    d, _ = make_run(paths, "r1", 100, 0.20, 0.80)
    assert G.gate(cfg, paths, FakeRD(d, 100), log=lambda *_: None)["passed"]


def test_must_beat_popularity(env):
    cfg, paths = env
    d, _ = make_run(paths, "r1", 100, 0.01, 0.02, pop=0.05)
    assert not G.gate(cfg, paths, FakeRD(d, 100), log=lambda *_: None)["passed"]


def test_regression_rules(env):
    cfg, paths = env
    p, _ = make_run(paths, "prod", 100, 0.200, 0.800)
    G.promote(paths, p, "init", log=lambda *_: None)
    same, _ = make_run(paths, "same", 100, 0.200, 0.800)        # identical -> CI around 0 -> pass
    bad, _ = make_run(paths, "bad", 100, 0.150, 0.800)          # ndcg -25% -> fail
    better, _ = make_run(paths, "better", 100, 0.230, 0.820)
    q = lambda *_: None
    assert G.gate(cfg, paths, FakeRD(same, 100), log=q)["passed"]
    assert G.gate(cfg, paths, FakeRD(better, 100), log=q)["passed"]
    rep = G.gate(cfg, paths, FakeRD(bad, 100), log=q)
    assert not rep["passed"]
    ndcg = next(c for c in rep["checks"] if "ndcg" in c["check"] and "regression" in c["check"])
    assert ndcg["ci_high"] < -0.2 and "confidently worse" in ndcg["check"]


def test_tiny_but_significant_drop_passes_big_noisy_drop_fails(env):
    cfg, paths = env
    q = lambda *_: None
    # many users: -0.6% is statistically real but below min_effect (1%) -> pass
    p, _ = make_run(paths, "prod", 100, 0.500, 0.800, n=20000)
    G.promote(paths, p, "init", log=q)
    c, _ = make_run(paths, "tiny", 100, 0.497, 0.800, n=20000)
    assert G.gate(cfg, paths, FakeRD(c, 100), log=q)["passed"]
    # few users: -8% point drop with a wide CI -> blocked by the max_point_drop safety net
    p2, _ = make_run(paths, "prod2", 100, 0.100, 0.800, n=600)
    G.promote(paths, p2, "init", log=q)
    c2, _ = make_run(paths, "noisy", 100, 0.092, 0.800, n=600)
    rep = G.gate(cfg, paths, FakeRD(c2, 100), log=q)
    ndcg = next(x for x in rep["checks"] if "ndcg" in x["check"] and "regression" in x["check"])
    assert not ndcg["passed"] and "point drop" in ndcg["check"]


def test_bootstrap_is_paired():
    rng = np.random.default_rng(1)
    base = rng.random(5000)
    a = {"uid": np.arange(5000), "m": (base < 0.30).astype(np.float32)}
    b = {"uid": np.arange(5000), "m": (base < 0.31).astype(np.float32)}      # b = a + a few extra hits
    r = G.paired_bootstrap(b, a, ["m"], n_boot=500)["m"]
    assert 0 < r["ci_low"] < r["rel_change"] < r["ci_high"]                 # paired -> CI excludes 0


def test_older_data_than_production_is_blocked(env):
    cfg, paths = env
    p, _ = make_run(paths, "prod", 200, 0.10, 0.50)
    G.promote(paths, p, "init", log=lambda *_: None)
    old, _ = make_run(paths, "old", 100, 0.90, 0.99)            # better numbers, but older window
    rep = G.gate(cfg, paths, FakeRD(old, 100), log=lambda *_: None)
    assert not rep["passed"]
    assert any("older" in c["check"] and not c["passed"] for c in rep["checks"])


def test_pointer_is_relative_and_rollback_works(env):
    _, paths = env
    a, _ = make_run(paths, "a", 100, 0.2, 0.8)
    b, _ = make_run(paths, "b", 100, 0.2, 0.8)
    G.promote(paths, a, "x", log=lambda *_: None)
    G.promote(paths, b, "y", log=lambda *_: None)
    cur = G.current_production(paths)
    assert cur["run_id"] == "b" and cur["path"] == "runs/b"     # portable across Windows / Docker
    assert G.run_path(paths, cur) == paths.models / "runs" / "b"
    G.rollback(paths, log=lambda *_: None)
    assert G.current_production(paths)["run_id"] == "a"
    assert not paths.production_pointer.with_suffix(".tmp").exists()


def test_serving_follows_pointer(env, monkeypatch):
    _, paths = env
    from serve.config import ServeConfig
    monkeypatch.setenv("RANKFORGE_REPO", str(paths.models.parent.parent))
    monkeypatch.delenv("MODEL_DIR", raising=False)
    cfg = ServeConfig.from_env()
    assert cfg.models == paths.models                           # no pointer -> flat notebook folder
    a, _ = make_run(paths, "a", 100, 0.2, 0.8)
    G.promote(paths, a, "x", log=lambda *_: None)
    assert cfg.models == paths.models / "runs" / "a"
