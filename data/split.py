"""
Phase 0, step 3: GLOBAL temporal split.

Kyun global (sab users ke liye ek hi cutoff) aur leave-last-out nahi?
  Leave-last-out mein user A ka test item 2019 ka ho sakta hai, jabki user B ka train item
  2022 ka. Model "future" dekh ke "past" predict karta hai -> leakage, numbers inflated.
  Global cutoff = production jaisa: kal tak ka data dekho, kal ke baad predict karo.

Cutoffs fixed days ki jagah QUANTILE se: Amazon data time mein bahut skewed hai, "last 30 days"
mein kabhi bahut kam interactions hote hain. Default: last 10% = test, uske pehle 10% = val.
Exact cutoff dates print hoti hain -> REPORT.md mein likho.

Usage:
    python -m data.split --category Video_Games
Output (data/processed/<category>/): train / val / test .parquet + split_stats.csv
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def temporal_split(df: pl.DataFrame, val_frac=0.1, test_frac=0.1):
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be positive and sum to less than 1")
    if df.is_empty():
        raise ValueError("Cannot split an empty interaction table")
    t_test = df["ts"].quantile(1 - test_frac, interpolation="lower")
    t_val = df["ts"].quantile(1 - test_frac - val_frac, interpolation="lower")
    train = df.filter(pl.col("ts") < t_val)
    val = df.filter((pl.col("ts") >= t_val) & (pl.col("ts") < t_test))
    test = df.filter(pl.col("ts") >= t_test)
    return train, val, test, int(t_val), int(t_test)


def describe(name, part, known_users, known_items):
    """Eval part ka breakdown: kitne users/items train mein pehle se hain."""
    users = part["uid"].unique()
    warm_users = users.is_in(known_users.implode()).sum()
    cold_item_rows = (~part["iid"].is_in(known_items.implode())).sum()
    return {
        "split": name,
        "interactions": part.height,
        "users": users.len(),
        "warm_users_pct": warm_users / max(users.len(), 1),
        "cold_item_interactions_pct": cold_item_rows / max(part.height, 1),
    }


def fmt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Video_Games")
    ap.add_argument("--root", default="data/processed")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    a = ap.parse_args()

    d = Path(a.root) / a.category
    df = pl.read_parquet(d / "interactions.parquet")
    train, val, test, t_val, t_test = temporal_split(df, a.val_frac, a.test_frac)
    print(f"train < {fmt(t_val)} <= val < {fmt(t_test)} <= test")

    ku, ki = train["uid"].unique(), train["iid"].unique()
    rows = [{"split": "train", "interactions": train.height, "users": ku.len(),
             "warm_users_pct": 1.0, "cold_item_interactions_pct": 0.0},
            describe("val", val, ku, ki)]
    # test ke liye "known" = train + val (final eval mein model train+val par refit hota hai)
    tv = pl.concat([train, val])
    rows.append(describe("test", test, tv["uid"].unique(), tv["iid"].unique()))
    s = pl.DataFrame(rows)
    print(s)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.write_parquet(d / f"{name}.parquet")
    s.write_csv(d / "split_stats.csv")


if __name__ == "__main__":
    main()
