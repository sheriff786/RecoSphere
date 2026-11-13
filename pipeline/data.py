"""
Step 1 (prepare): each run gets its OWN copy of the train/val/test windows, so a run is reproducible
and a later run (or the gate) can re-evaluate any model on exactly the same data.

  mode gold    -> notebook 02's split as-is (pipeline numbers == notebook numbers)
  mode cutoff  -> "today" = cutoff:  train < cutoff - val_days <= val < cutoff <= test < cutoff + test_days
                  (time-travel replay, Phase 7e)

Catalog-level inputs are read from data/gold/<cat>/ and NOT copied (they do not change per run):
  user_map, item_map, item_features.parquet, item_text_emb_<encoder>.npy   (notebooks 02 + 03)
"""
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

REQUIRED_GOLD = ["train.parquet", "val.parquet", "test.parquet", "user_map.parquet", "item_map.parquet",
                 "item_features.parquet"]


def _utc(ts):
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def prepare(cfg, paths, run_dir: Path, log=print) -> dict:
    missing = [f for f in REQUIRED_GOLD if not (paths.gold / f).exists()]
    enc = cfg["two_tower"]["text_encoder"]
    if not (paths.gold / f"item_text_emb_{enc}.npy").exists():
        missing.append(f"item_text_emb_{enc}.npy")
    if missing:
        raise FileNotFoundError(f"missing in {paths.gold}: {missing} -> run notebooks 02 and 03 (Tasks 1-2) first")

    g = {s: pl.read_parquet(paths.gold / f"{s}.parquet") for s in ("train", "val", "test")}
    sp = cfg["split"]
    if sp["mode"] == "gold":
        tr, va, te = g["train"], g["val"], g["test"]
    elif sp["mode"] == "cutoff":
        allx = pl.concat([g["train"], g["val"], g["test"]])
        cut = int(datetime.fromisoformat(str(sp["cutoff"])).replace(tzinfo=timezone.utc).timestamp())
        t_val, t_end = cut - sp["val_days"] * 86400, cut + sp["test_days"] * 86400
        tr = allx.filter(pl.col("ts") < t_val)
        va = allx.filter((pl.col("ts") >= t_val) & (pl.col("ts") < cut))
        te = allx.filter((pl.col("ts") >= cut) & (pl.col("ts") < t_end))
    else:
        raise ValueError(sp["mode"])
    assert tr.height and va.height and te.height, "empty split — check cutoff / window sizes"
    assert tr["ts"].max() < va["ts"].min() and va["ts"].max() < te["ts"].min(), "split is not time-ordered"

    d = run_dir / "data"; d.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("val", va), ("test", te)):
        part.write_parquet(d / f"{name}.parquet")
    h = hashlib.sha1()
    for part in (tr, va, te):
        h.update(np.ascontiguousarray(part.select("uid", "iid", "ts").to_numpy()).tobytes())
    man = {"mode": sp["mode"], "cutoff": None if sp.get("cutoff") is None else str(sp["cutoff"]), "rows": {"train": tr.height, "val": va.height, "test": te.height},
           "t_val": int(va["ts"].min()), "t_test": int(te["ts"].min()), "t_end": int(te["ts"].max()),
           "val_start_utc": _utc(va["ts"].min()), "test_start_utc": _utc(te["ts"].min()),
           "test_end_utc": _utc(te["ts"].max()), "data_hash": h.hexdigest()[:12]}
    (d / "manifest.json").write_text(json.dumps(man, indent=2))
    log(f"prepare: train={tr.height:,} val={va.height:,} test={te.height:,} | test {man['test_start_utc']} -> "
        f"{man['test_end_utc']} | data {man['data_hash']}")
    return man


@dataclass
class RunData:
    run_dir: Path
    gold: Path
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    train_val: pl.DataFrame
    n_users: int
    n_items: int
    item_feat: pl.DataFrame
    t_val: int
    t_test: int
    manifest: dict

    def text_emb(self, encoder: str) -> np.ndarray:
        return np.load(self.gold / f"item_text_emb_{encoder}.npy")


def load_run_data(run_dir: Path, gold: Path) -> RunData:
    d = run_dir / "data"
    tr, va, te = (pl.read_parquet(d / f"{s}.parquet") for s in ("train", "val", "test"))
    man = json.loads((d / "manifest.json").read_text())
    item_feat = pl.read_parquet(gold / "item_features.parquet").sort("iid")
    return RunData(run_dir=run_dir, gold=gold, train=tr, val=va, test=te, train_val=pl.concat([tr, va]),
                   n_users=pl.read_parquet(gold / "user_map.parquet").height, n_items=item_feat.height,
                   item_feat=item_feat, t_val=man["t_val"], t_test=man["t_test"], manifest=man)
