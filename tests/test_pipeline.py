"""Correctness tests — real data se pehle ye pass hone chahiye.  Run: pytest -q"""
import numpy as np
import polars as pl
import scipy.sparse as sp

from baselines.models import EASE, ItemKNN, Popularity
from data.prepare import build_gold, dedup, k_core
from data.split import temporal_split
from eval.metrics import _ndcg, build_matrix, evaluate, topk


def test_kcore_is_fixed_point():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"user_id": rng.integers(0, 300, 3000).astype(str),
                       "item_id": rng.integers(0, 200, 3000).astype(str),
                       "ts": np.arange(3000)})
    out = k_core(dedup(df), 5)
    assert out.group_by("user_id").len()["len"].min() >= 5
    assert out.group_by("item_id").len()["len"].min() >= 5


def test_kcore_logs_each_materialized_iteration():
    df = pl.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
        "item_id": ["a", "b", "a", "b", "a", "c"],
        "ts": [1, 2, 3, 4, 5, 6],
    })
    log = []
    out = k_core(df, 2, iteration_log=log)
    assert log[-1]["interactions"] == out.height
    assert all(row["users"] >= 0 and row["items"] >= 0 for row in log)
    assert out.group_by("user_id").len()["len"].min() >= 2
    assert out.group_by("item_id").len()["len"].min() >= 2


def test_dedup_keeps_first():
    df = pl.DataFrame({"user_id": ["a", "a"], "item_id": ["x", "x"], "ts": [20, 10]})
    assert dedup(df)["ts"].to_list() == [10]


def test_build_gold_writes_dense_mappings_and_conserves_rows(tmp_path):
    rows = [
        (f"u{u}", f"i{i}", float((u + i) % 5 + 1), u * 4 + i + 1, (u + i) % 2 == 0)
        for u in range(8)
        for i in range(4)
    ]
    rows.append(("u0", "i0", 1.0, 100, False))
    reviews = pl.DataFrame(
        rows,
        schema=["user_id", "item_id", "rating", "ts", "verified"],
        orient="row",
    )

    result = build_gold(reviews, tmp_path, k=2)

    assert result["feedback_decision"]["all_review_rows"] == 33
    assert result["feedback_decision"]["verified_true_rows"] == 16
    assert result["prepare"]["dedup"]["interactions"] == 32
    assert result["prepare"]["k_comparison"]["k"] == 10
    assert result["split"]["row_conservation"] is True
    assert result["id_mapping"]["uid_dense"] and result["id_mapping"]["iid_dense"]
    parts = [pl.read_parquet(tmp_path / f"{name}.parquet") for name in ("train", "val", "test")]
    assert sum(part.height for part in parts) == 32
    assert all(part.columns == ["uid", "iid", "ts", "rating"] for part in parts)
    assert pl.read_parquet(tmp_path / "user_map.parquet").height == 8
    assert pl.read_parquet(tmp_path / "item_map.parquet").height == 4
    assert (tmp_path / "prepare_stats.json").is_file()


def test_split_is_temporal():
    df = pl.DataFrame({"uid": np.zeros(100, int), "iid": np.arange(100), "ts": np.arange(100)})
    tr, va, te, _, _ = temporal_split(df)
    assert tr["ts"].max() < va["ts"].min() and va["ts"].max() < te["ts"].min()


def test_split_rejects_invalid_fractions():
    df = pl.DataFrame({"uid": [0], "iid": [0], "ts": [1]})
    try:
        temporal_split(df, val_frac=0.6, test_frac=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid fractions should be rejected")


def test_topk_and_ndcg():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert topk(s, 2).tolist() == [[1, 3]]
    assert np.isclose(_ndcg(np.array([1.0, 0.0]), 1, 2), 1.0)
    assert np.isclose(_ndcg(np.array([0.0, 1.0]), 1, 2), 1 / np.log2(3))


def test_evaluate_perfect_oracle_and_masking():
    # user 0 history: item 0; target: item 1. Oracle item 0 ko top par rakhta hai -> mask hona chahiye
    hist = pl.DataFrame({"uid": [0], "iid": [0]})
    ev = pl.DataFrame({"uid": [0, 0], "iid": [1, 0]})  # item 0 re-interaction -> target nahi
    H = build_matrix(hist, 1, 4)
    oracle = lambda u: np.array([[10.0, 5.0, 0.0, 0.0]])
    r = evaluate(oracle, H, ev, np.array([0]), ks=(1,))
    assert r["all/recall@1"] == 1.0 and r["all/ndcg@1"] == 1.0
    assert r["n_users_cold"] == 1  # item 1 train mein nahi tha -> cold slice


def test_ease_matches_formula():
    X = sp.random(50, 15, density=0.3, random_state=1, format="csr")
    X.data[:] = 1
    m = EASE(lam=10, max_items=100).fit(X)
    G = X.T.toarray() @ X.toarray() + 10 * np.eye(15)
    P = np.linalg.inv(G)
    B = np.eye(15) - P / np.diag(P)
    assert np.allclose(m.B, B, atol=1e-4)
    assert np.allclose(np.diag(m.B), 0)


def test_models_score_shapes():
    X = sp.random(30, 12, density=0.3, random_state=2, format="csr"); X.data[:] = 1
    for m in (Popularity().fit(X), ItemKNN(k=5).fit(X), EASE(lam=5).fit(X)):
        assert m.score(np.array([0, 3])).shape == (2, 12)
