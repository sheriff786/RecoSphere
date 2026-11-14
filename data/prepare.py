"""Prepare Silver interactions into model-ready Gold train/validation/test data."""
import argparse
import json
from pathlib import Path

import polars as pl

from data.split import describe, temporal_split

INTERACTION_COLUMNS = ["user_id", "item_id", "ts", "rating", "verified"]


def stats(df: pl.DataFrame, label: str) -> dict:
    n, u, i = df.height, df["user_id"].n_unique(), df["item_id"].n_unique()
    density = n / (u * i) if u and i else 0.0
    print(f"[{label:>10}] interactions={n:,}  users={u:,}  items={i:,}  density={density:.2e}")
    return {"step": label, "interactions": n, "users": u, "items": i, "density": density}


def dedup(df: pl.DataFrame) -> pl.DataFrame:
    """Keep the earliest review for each user/item pair."""
    return df.sort("ts", maintain_order=True).unique(
        subset=["user_id", "item_id"], keep="first", maintain_order=True
    )


def k_core(
    df: pl.DataFrame,
    k: int,
    max_iter: int = 1000,
    iteration_log: list[dict] | None = None,
) -> pl.DataFrame:
    """Filter users and items to a fixed point, recording each materialized pass."""
    if k < 1:
        raise ValueError("k must be at least 1")

    for it in range(max_iter):
        before = df.height
        good_users = df.group_by("user_id").len().filter(pl.col("len") >= k).select("user_id")
        df = df.join(good_users, on="user_id", how="semi")
        good_items = df.group_by("item_id").len().filter(pl.col("len") >= k).select("item_id")
        df = df.join(good_items, on="item_id", how="semi")
        row = {
            "iteration": it + 1,
            "interactions": df.height,
            "users": df["user_id"].n_unique(),
            "items": df["item_id"].n_unique(),
        }
        print(
            f"[k={k} iteration {it + 1}] interactions={row['interactions']:,}  "
            f"users={row['users']:,}  items={row['items']:,}"
        )
        if iteration_log is not None:
            iteration_log.append(row)
        if df.height == before:
            return df
    raise RuntimeError(f"{k}-core did not converge in {max_iter} iterations")


def map_ids(df: pl.DataFrame):
    user_map = (
        df.select("user_id").unique().sort("user_id").with_row_index("uid")
        .with_columns(pl.col("uid").cast(pl.Int32))
    )
    item_map = (
        df.select("item_id").unique().sort("item_id").with_row_index("iid")
        .with_columns(pl.col("iid").cast(pl.Int32))
    )
    out = (
        df.join(user_map, on="user_id").join(item_map, on="item_id")
        .select("uid", "iid", "ts", "rating")
        .sort("ts")
    )
    return out, user_map, item_map


def prepare(
    df: pl.DataFrame,
    k: int,
    iteration_log: list[dict] | None = None,
    deduped: pl.DataFrame | None = None,
):
    log = [stats(df, "raw")]
    df = dedup(df) if deduped is None else deduped
    log.append(stats(df, "dedup"))
    df = k_core(df, k, iteration_log=iteration_log)
    log.append(stats(df, f"{k}-core"))
    inter, user_map, item_map = map_ids(df)
    return inter, user_map, item_map, pl.DataFrame(log)


def _read_silver(silver_root: Path, category: str) -> pl.DataFrame:
    source = silver_root / "interactions" / f"category={category}"
    files = sorted(source.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Silver interaction Parquet files found under {source}")
    return pl.read_parquet(files, columns=INTERACTION_COLUMNS)


def _date(ts: int) -> str:
    return pl.from_epoch(pl.Series([ts]), time_unit="s").dt.strftime("%Y-%m-%d").item()


def _assert_dense_map(mapping: pl.DataFrame, id_col: str) -> None:
    values = mapping[id_col]
    if (
        mapping.height == 0
        or values.min() != 0
        or values.max() != mapping.height - 1
        or values.n_unique() != mapping.height
    ):
        raise AssertionError(f"{id_col} mapping is not dense from 0 to N-1")


def build_gold(
    reviews: pl.DataFrame,
    out: Path,
    k: int = 5,
    feedback: str = "all",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict:
    """Build Gold data from Silver-shaped reviews and return its measured statistics."""
    if feedback not in {"all", "verified"}:
        raise ValueError("feedback must be 'all' or 'verified'")
    missing = set(INTERACTION_COLUMNS) - set(reviews.columns)
    if missing:
        raise ValueError(f"Silver interactions missing columns: {sorted(missing)}")

    reviews = reviews.select(INTERACTION_COLUMNS)
    n_all = reviews.height
    n_verified = reviews.filter(pl.col("verified").fill_null(False)).height
    print(f"implicit-feedback rows: all={n_all:,}, verified=true={n_verified:,}")
    if feedback == "verified":
        reviews = reviews.filter(pl.col("verified").fill_null(False))
    reviews = reviews.select("user_id", "item_id", "ts", "rating")

    deduped = dedup(reviews)
    kcore_iterations: list[dict] = []
    inter, user_map, item_map, prep_log = prepare(
        reviews, k, kcore_iterations, deduped=deduped
    )
    core_stats = prep_log.row(2, named=True)
    if inter.height != core_stats["interactions"]:
        raise AssertionError("ID mapping changed the number of k-core interactions")
    if user_map.height != core_stats["users"] or item_map.height != core_stats["items"]:
        raise AssertionError("ID maps do not cover every user and item in the k-core")
    compare_k = 10 if k != 10 else 5
    compare_iterations: list[dict] = []
    comparison = k_core(deduped, compare_k, iteration_log=compare_iterations)
    comparison_stats = {
        "k": compare_k,
        "interactions": comparison.height,
        "users": comparison["user_id"].n_unique(),
        "items": comparison["item_id"].n_unique(),
        "iterations": compare_iterations,
    }

    if (
        inter.group_by("uid").len()["len"].min() < k
        or inter.group_by("iid").len()["len"].min() < k
    ):
        raise AssertionError(f"k-core output violates the minimum degree k={k}")

    # This project uses Polars eager DataFrames, so every semi-join pass is materialized;
    # unlike a lazy Spark plan, iterations do not retain the preceding join lineage.
    # Mapping is deliberately built on the full k-core so validation/test-only items
    # still have ids for cold-item analysis.
    _assert_dense_map(user_map, "uid")
    _assert_dense_map(item_map, "iid")
    train, val, test, t_val, t_test = temporal_split(inter, val_frac, test_frac)
    if train.height + val.height + test.height != inter.height:
        raise AssertionError("Temporal split lost or duplicated k-core interactions")
    if not train.height or not val.height or not test.height:
        raise AssertionError("Temporal split produced an empty partition")
    if not (train["ts"].max() < val["ts"].min() and val["ts"].max() < test["ts"].min()):
        raise AssertionError("Temporal split timestamps overlap or are not strictly ordered")

    train_users, train_items = train["uid"].unique(), train["iid"].unique()
    val_stats = describe("val", val, train_users, train_items)
    test_stats = describe("test", test, train_users, train_items)
    train_val = pl.concat([train, val])
    test_final_fit_stats = describe(
        "test_train_plus_val_known",
        test,
        train_val["uid"].unique(),
        train_val["iid"].unique(),
    )
    split_stats = [
        {
            "split": "train",
            "interactions": train.height,
            "users": train["uid"].n_unique(),
            "warm_users_pct": 1.0,
            "cold_item_interactions_pct": 0.0,
        },
        val_stats,
        test_stats,
    ]

    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.select("uid", "iid", "ts", "rating").write_parquet(out / f"{name}.parquet")
    user_map.write_parquet(out / "user_map.parquet")
    item_map.write_parquet(out / "item_map.parquet")

    result = {
        "feedback_decision": {
            "selected": feedback,
            "all_review_rows": n_all,
            "verified_true_rows": n_verified,
            "rows_excluded_if_verified_only": n_all - n_verified,
            "rating_filter": None,
            "rationale": (
                "Every review is treated as an implicit interaction, including 1-star reviews; "
                "verified purchase is a quality attribute, not a preference label."
                if feedback == "all"
                else "Only verified purchases are retained; ratings are not used as labels or filters."
            ),
        },
        "prepare": {
            "raw": prep_log.row(0, named=True),
            "dedup": prep_log.row(1, named=True),
            "k_core": core_stats,
            "k_core_iterations": kcore_iterations,
            "k_comparison": comparison_stats,
            "full_k_core_before_temporal_split": True,
            "dedup_rule": "earliest ts per (user_id, item_id)",
        },
        "split": {
            "fractions": {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac},
            "method": "global temporal timestamp quantiles; Polars exact quantile, lower interpolation",
            "cutoffs": {
                "val_start_ts_inclusive": t_val,
                "test_start_ts_inclusive": t_test,
                "train_end_exclusive": _date(t_val),
                "val_end_exclusive": _date(t_test),
            },
            "stats": split_stats,
            "test_known_after_train_plus_val": test_final_fit_stats,
            "row_conservation": train.height + val.height + test.height == inter.height,
        },
        "id_mapping": {
            "mapped_full_k_core_before_split": True,
            "users": user_map.height,
            "items": item_map.height,
            "uid_dense": True,
            "iid_dense": True,
        },
        "official_5_core_benchmark": {
            "available": False,
            "reason": "No downloaded official benchmark files found; see data/raw/<category>/benchmark.",
        },
    }
    (out / "prepare_stats.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"cutoffs: train < {_date(t_val)} <= val < {_date(t_test)} <= test")
    for row in split_stats:
        print(
            f"{row['split']}: interactions={row['interactions']:,} "
            f"users={row['users']:,} warm_users={row['warm_users_pct']:.1%} "
            f"cold_item_interactions={row['cold_item_interactions_pct']:.1%}"
        )
    print(f"saved Gold data -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--feedback", choices=("all", "verified"), default="all")
    ap.add_argument("--silver-root", default="data/silver")
    ap.add_argument("--raw-root", help="Optional legacy input: data/raw/<category>/interactions.parquet")
    ap.add_argument("--out-root", default="data/gold")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    a = ap.parse_args()

    if a.raw_root:
        reviews = pl.read_parquet(
            Path(a.raw_root) / a.category / "interactions.parquet",
            columns=INTERACTION_COLUMNS,
        )
    else:
        reviews = _read_silver(Path(a.silver_root), a.category)
    build_gold(
        reviews,
        Path(a.out_root) / a.category,
        k=a.k,
        feedback=a.feedback,
        val_frac=a.val_frac,
        test_frac=a.test_frac,
    )


if __name__ == "__main__":
    main()
