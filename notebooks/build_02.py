"""Builds 02_prepare_and_split.ipynb (notebook ko code se generate karna = git-diffable source).

Usage (repo root se):
    .reco/Scripts/python.exe notebooks/build_02.py
    .reco/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 notebooks/02_prepare_and_split.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# RankForge · 02 — Prepare & Split (Silver → Gold)

Silver "saaf data" hai, lekin model ke kaam ka nahi. Analogy: Silver ek saaf kitchen hai; yahan ingredients naap-tol ke
teen dibbon mein baantne hain — **train** (aaj tak), **val** (kal), **test** (parson). Model sirf pehla dibba dekhega.

```
Silver interactions
   → implicit-feedback decision → dedup (user, item) → iterative k-core → int id mapping
   → global temporal split (train | val | test)
   → Gold: train / val / test parquet + maps + prepare_stats.json
   → run_baselines.py --root data/gold
```

**Acceptance criteria** (sab neeche `assert` se check hote hain):
1. k-core ke baad har user aur har item ke ≥ k interactions
2. `max(train.ts) < min(val.ts)` aur `max(val.ts) < min(test.ts)`
3. train + val + test rows = k-core rows
4. Har uid/iid map mein hai aur 0..N-1 range mein hai
5. Official 5-core benchmark se counts compare
6. Independent Polars implementation (`data/prepare.py`) se cross-check
""")

md("## Step 0 — Environment + config")

code(r"""
import json, os, shutil, sys, time
from pathlib import Path

if os.name == "nt":   # Windows: Java + winutils (Spark ko local disk par likhne ke liye)
    os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-21.0.13.9-hotspot")
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    for var in ("JAVA_HOME", "HADOOP_HOME"):
        os.environ["PATH"] = os.environ[var] + r"\bin;" + os.environ["PATH"]
    missing = [f for f in ("winutils.exe", "hadoop.dll") if not (Path(os.environ["HADOOP_HOME"]) / "bin" / f).exists()]
    assert not missing, f"HADOOP_HOME\\bin mein missing: {missing}"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ---------------- CONFIG ----------------
CATEGORY    = os.environ.get("RANKFORGE_CATEGORY", "Video_Games")
DATA_ROOT   = Path(os.environ.get("RANKFORGE_DATA", "../data")).resolve()
REPO_ROOT   = DATA_ROOT.parent
DRIVER_MEM  = os.environ.get("RANKFORGE_DRIVER_MEM", "8g")
K           = int(os.environ.get("RANKFORGE_K", 5))
K_COMPARE   = 10 if K != 10 else 5           # decision table ke liye doosra k
FEEDBACK    = "all"                          # "all" | "verified" — Task 1 ka decision
VAL_FRAC, TEST_FRAC = 0.1, 0.1
QUANTILE_REL_ERR = 1e-4                      # approxQuantile -> cutoff APPROXIMATE hai (REPORT mein likho)

SILVER_INTER = DATA_ROOT / "silver" / "interactions" / f"category={CATEGORY}"
GOLD         = DATA_ROOT / "gold" / CATEGORY
BENCH        = DATA_ROOT / "benchmark" / CATEGORY
STATS = {"category": CATEGORY, "k": K, "engine": "pyspark"}    # -> prepare_stats.json
print("silver:", SILVER_INTER, "\ngold:  ", GOLD)
""")

code(r"""
from pyspark.sql import SparkSession, Window, functions as F

spark = (SparkSession.builder
         .master("local[*]")
         .appName(f"rankforge-prepare-{CATEGORY}")
         .config("spark.driver.memory", DRIVER_MEM)
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.shuffle.partitions", "64")
         .config("spark.sql.parquet.compression.codec", "zstd")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.version
""")

code(r"""
def shape(df) -> dict:
    # interactions / users / items — exact counts (ek hi scan mein)
    r = df.agg(F.count("*").alias("interactions"),
               F.countDistinct("user_id").alias("users"),
               F.countDistinct("item_id").alias("items")).first().asDict()
    r["density"] = r["interactions"] / (r["users"] * r["items"]) if r["users"] and r["items"] else 0.0
    return r

def show(label, s):
    print(f"[{label:>14}] interactions={s['interactions']:>10,}  users={s['users']:>9,}  items={s['items']:>7,}")
""")

md(r"""
## Task 1 — Load + implicit feedback decision

Sirf `user_id, item_id, ts, rating, verified` chahiye (`ts` Silver mein already **seconds** hai).

**Rating se filter NAHI karte.** 1-star review bhi batata hai ki user ne item khareeda/use kiya — ye *implicit feedback* hai
("user ne is item ke saath interact kiya"), preference label nahi. Sawal sirf ye hai: saare reviews positive, ya sirf `verified = true`?
Dono ka count nikaal ke decide karte hain (k-core ke baad ka asar Task 3 mein dikhega).
""")

code(r"""
silver = (spark.read.parquet(str(SILVER_INTER))
          .select("user_id", "item_id", "ts", "rating", "verified"))

is_verified = F.coalesce(F.col("verified"), F.lit(False))
fb = silver.agg(F.count("*").alias("all_rows"),
                F.sum(is_verified.cast("int")).alias("verified_rows")).first().asDict()
fb["verified_share"] = fb["verified_rows"] / fb["all_rows"]
print(f"all reviews: {fb['all_rows']:,}   verified=true: {fb['verified_rows']:,}  ({fb['verified_share']:.1%})")

interactions = silver if FEEDBACK == "all" else silver.filter(is_verified)
raw_stats = shape(interactions); show("raw", raw_stats)
""")

md(r"""
## Task 2 — (user, item) dedup

Same user ne same `item_id` (parent_asin) ko multiple baar review kiya → **sabse pehla (min ts)** rakho.
`min_by(rating, ts)` = us earliest review ki rating (rating sirf metadata hai, label nahi).

`localCheckpoint()` = result materialize + lineage cut. Iske baad k-core loop isi materialized table se shuru hoga.
""")

code(r"""
def dedup(df):
    return (df.groupBy("user_id", "item_id")
              .agg(F.min("ts").alias("ts"), F.min_by("rating", "ts").alias("rating")))

deduped = dedup(interactions).localCheckpoint()
dedup_stats = shape(deduped); show("dedup", dedup_stats)
print(f"removed repeat (user, item) rows: {raw_stats['interactions'] - dedup_stats['interactions']:,}")
""")

md(r"""
## Task 3 — Iterative k-core

Loop: users jinke ≥ k interactions → rakho; phir items jinke ≥ k → rakho. Users hatane se kuch items ka count k se neeche
gir jaata hai (aur ulta), isliye **fixed point** tak dohraate hain — jab row count badalna band ho jaaye.

**⚠️ Lineage explosion:** Spark lazy hai. Bina checkpoint ke iteration *n* ka plan = pichli saari *n−1* iterations ke joins +
aggregations. 6-7 iterations ke baad plan itna bada ho jaata hai ki optimizer hi minutes leta hai (ya driver OOM).
**Fix:** har iteration ke end mein `df = df.localCheckpoint()` → data materialize, plan truncate. Har iteration ka plan ab constant size hai.

`left_semi` join = "sirf filter karo, right side ka koi column mat lao" — filter-join ke liye sabse sasta.
""")

code(r"""
def k_core(df, k: int, max_iter: int = 50, verbose: bool = True):
    df = df.localCheckpoint()
    n_prev = df.count()
    log = []
    for it in range(1, max_iter + 1):
        t0 = time.time()
        good_users = df.groupBy("user_id").count().filter(F.col("count") >= k).select("user_id")
        df = df.join(good_users, "user_id", "left_semi")
        good_items = df.groupBy("item_id").count().filter(F.col("count") >= k).select("item_id")
        df = df.join(good_items, "item_id", "left_semi")
        df = df.localCheckpoint()                          # <- lineage yahin kat-ti hai
        s = shape(df)
        row = {"iteration": it, "interactions": s["interactions"], "users": s["users"],
               "items": s["items"], "seconds": round(time.time() - t0, 1)}
        log.append(row)
        if verbose:
            print(f"[k={k} it {it:>2}] interactions={row['interactions']:>9,}  users={row['users']:>8,}  "
                  f"items={row['items']:>7,}  ({row['seconds']}s)")
        if row["interactions"] == n_prev:
            return df, log
        n_prev = row["interactions"]
    raise RuntimeError(f"{k}-core did not converge in {max_iter} iterations")

core, kcore_log = k_core(deduped, K)
core_stats = shape(core); show(f"{K}-core", core_stats)
""")

code(r"""
# Acceptance 1: har user aur har item ke >= K interactions
min_user_deg = core.groupBy("user_id").count().agg(F.min("count")).first()[0]
min_item_deg = core.groupBy("item_id").count().agg(F.min("count")).first()[0]
assert min_user_deg >= K and min_item_deg >= K, (min_user_deg, min_item_deg)
print(f"OK  min user degree={min_user_deg}, min item degree={min_item_deg} (k={K})")
""")

md(r"""
### Decision table inputs: k = 5 vs k = 10, aur all vs verified-only

Same k-core function, alag inputs. Ye numbers REPORT ke "Decisions" section mein jaate hain.
""")

code(r"""
cmp_core, cmp_log = k_core(deduped, K_COMPARE, verbose=False)
cmp_stats = shape(cmp_core); show(f"{K_COMPARE}-core", cmp_stats)

verified_core, _ = k_core(dedup(silver.filter(is_verified)), K, verbose=False)
verified_stats = shape(verified_core); show(f"verified {K}-core", verified_stats)

import pandas as pd
decision = pd.DataFrame([
    {"variant": f"all reviews, k={K} (chosen)", **core_stats},
    {"variant": f"all reviews, k={K_COMPARE}", **cmp_stats},
    {"variant": f"verified only, k={K}", **verified_stats},
]).set_index("variant")
decision["users_kept_%_of_dedup"] = (100 * decision["users"] / dedup_stats["users"]).round(2)
decision["items_kept_%_of_dedup"] = (100 * decision["items"] / dedup_stats["items"]).round(2)
decision
""")

md(r"""
## Task 4 — Global temporal split

Sab users ke liye **ek hi cutoff** (production jaisa: kal tak ka data dekho, kal ke baad predict karo).
Last 10% interactions (time se) = test, uske pehle 10% = val, baaki train.

`approxQuantile(..., relativeError=1e-4)` → cutoff **approximate** hai (rank error ≤ 0.01% ≈ 80 rows yahan); returned value
hamesha data ki ek real `ts` hoti hai.

**⚠️ Ties:** same `ts` wali rows cutoff ke dono taraf nahi batni chahiye. Rule: `train: ts < t_val`, `val: t_val ≤ ts < t_test`,
`test: ts ≥ t_test` — ek row exactly ek bucket mein.
""")

code(r"""
q_val, q_test = 1 - VAL_FRAC - TEST_FRAC, 1 - TEST_FRAC
t_val, t_test = (int(x) for x in core.approxQuantile("ts", [q_val, q_test], QUANTILE_REL_ERR))

def split(df):
    return (df.filter(F.col("ts") < t_val),
            df.filter((F.col("ts") >= t_val) & (F.col("ts") < t_test)),
            df.filter(F.col("ts") >= t_test))

dates = spark.range(1).select(F.from_unixtime(F.lit(t_val)).alias("val_start"),
                              F.from_unixtime(F.lit(t_test)).alias("test_start")).first()
ties = core.agg(F.sum((F.col("ts") == t_val).cast("int")).alias("rows_at_t_val"),
                F.sum((F.col("ts") == t_test).cast("int")).alias("rows_at_t_test")).first().asDict()
print(f"train < {dates['val_start']} UTC <= val < {dates['test_start']} UTC <= test")
print("rows sharing the exact cutoff timestamp:", ties)
""")

md(r"""
## Task 5 — Integer id mapping

Matrices aur embeddings ko `0..N-1` dense integers chahiye. Mapping **poore k-core** par banti hai, sirf train par nahi —
test mein naye (cold) items ko bhi id chahiye, warna cold-item analysis ho hi nahi sakta.

`row_number().over(Window.orderBy(...))` bina `partitionBy` ke saari distinct ids **ek partition** mein le aata hai
(Spark warning bhi deta hai). L1 par ~95k users / ~26k items — chhota, chal jaata hai. **L3 ke liye:**
`df.select(col).distinct().rdd.zipWithIndex()` — distributed, no single-partition bottleneck.
Sort by string id → mapping deterministic hai (har run par same uid).
""")

code(r"""
def dense_map(df, key: str, idx: str):
    w = Window.orderBy(key)
    return (df.select(key).distinct()
              .withColumn(idx, (F.row_number().over(w) - 1).cast("int"))
              .select(idx, key))

user_map = dense_map(core, "user_id", "uid").localCheckpoint()
item_map = dense_map(core, "item_id", "iid").localCheckpoint()

inter = (core.join(user_map, "user_id").join(item_map, "item_id")
             .select("uid", "iid", "ts", F.col("rating").cast("float").alias("rating"))
             .localCheckpoint())
train, val, test = split(inter)
n_inter, n_train, n_val, n_test = inter.count(), train.count(), val.count(), test.count()
n_users, n_items = user_map.count(), item_map.count()
print(f"users={n_users:,} items={n_items:,} | train={n_train:,} val={n_val:,} test={n_test:,}")
print(f"actual fractions: train={n_train/n_inter:.4f} val={n_val/n_inter:.4f} test={n_test/n_inter:.4f}")
""")

code(r"""
# Acceptance 2: strictly ordered in time
b = {name: part.agg(F.min("ts").alias("min"), F.max("ts").alias("max")).first()
     for name, part in [("train", train), ("val", val), ("test", test)]}
assert b["train"]["max"] < b["val"]["min"] and b["val"]["max"] < b["test"]["min"], b
# Acceptance 3: na koi row khoyi, na duplicate hui (mapping join + split dono)
assert n_inter == core_stats["interactions"], "id mapping join changed row count"
assert n_train + n_val + n_test == core_stats["interactions"], "split lost/duplicated rows"
assert min(n_train, n_val, n_test) > 0
# Acceptance 4: dense 0..N-1 maps, aur har uid/iid map ke andar
for m, idx, n, s in [(user_map, "uid", n_users, "users"), (item_map, "iid", n_items, "items")]:
    r = m.agg(F.min(idx).alias("lo"), F.max(idx).alias("hi"), F.countDistinct(idx).alias("nd")).first()
    assert (r["lo"], r["hi"], r["nd"]) == (0, n - 1, n) and n == core_stats[s], (idx, r, n)
r = inter.agg(F.min("uid"), F.max("uid"), F.min("iid"), F.max("iid"),
              F.sum((F.col("uid").isNull() | F.col("iid").isNull()).cast("int"))).first()
assert r[0] == 0 and r[1] == n_users - 1 and r[2] == 0 and r[3] == n_items - 1 and r[4] == 0, r
print("OK  time order, row conservation, dense maps")
""")

md(r"""
## Task 6 — Eval slices ki stats

- **warm users %** = eval users jo train (history) mein pehle se hain — sirf yahi users ID-based models evaluate kar sakte hain
- **cold-item interactions %** = eval interactions jinka item history mein kabhi nahi aaya — ID-only model (Popularity, ItemKNN, EASE) inhe **kabhi** retrieve nahi kar sakta

Test ke liye do views: history = train (val jaisa), aur history = train+val (final eval mein model train+val par refit hota hai — `run_baselines.py` yahi karta hai).
""")

code(r"""
def eval_slice(name, part, history):
    n = part.count()
    users = part.select("uid").distinct()
    n_u = users.count()
    warm = users.join(history.select("uid").distinct(), "uid", "left_semi").count()
    cold = part.join(history.select("iid").distinct(), "iid", "left_anti").count()
    return {"split": name, "interactions": n, "users": n_u,
            "warm_users_pct": warm / max(n_u, 1), "cold_item_interactions_pct": cold / max(n, 1)}

train_val = train.unionByName(val)
split_stats = [
    {"split": "train", "interactions": n_train, "users": train.select("uid").distinct().count(),
     "warm_users_pct": 1.0, "cold_item_interactions_pct": 0.0},
    eval_slice("val (hist=train)", val, train),
    eval_slice("test (hist=train)", test, train),
    eval_slice("test (hist=train+val)", test, train_val),
]
pd.DataFrame(split_stats).set_index("split").style.format(
    {"warm_users_pct": "{:.1%}", "cold_item_interactions_pct": "{:.1%}", "interactions": "{:,}", "users": "{:,}"})
""")

md(r"""
**Iska matlab:** test interactions ka bada hissa aise items par hai jo history mein kabhi nahi aaye. Koi bhi ID-only model
un par recall = 0 dega — ye ID models ki **hard ceiling** hai, aur yahi gap Phase 2 ka text-aware item tower bharega.
""")

md(r"""
## Task 7 — Gold write

Columns `uid, iid, ts, rating` — taaki `run_baselines.py` aur `eval/metrics.py` bina badle chalein.

Spark normally ek **directory** likhta hai (`train.parquet/part-*.parquet` + `_SUCCESS` + `.crc`). Polars us directory ko
seedha nahi padh sakta (mixed extensions error). Gold L1 par chhota hai (~1M rows), isliye: Spark `repartition(1)` se likhe,
phir single part-file ko `train.parquet` naam se move karo. L3 par directories hi rakhenge aur readers glob (`**/*.parquet`) use karenge.
""")

code(r"""
def write_single_parquet(df, target: Path, sort_cols=()):
    tmp = target.with_name(f"_tmp_{target.name}")
    out = df.repartition(1)
    if sort_cols:
        out = out.sortWithinPartitions(*sort_cols)
    out.write.mode("overwrite").parquet(str(tmp))
    parts = list(tmp.glob("part-*.parquet"))
    assert len(parts) == 1, parts
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    os.replace(parts[0], target)
    shutil.rmtree(tmp)

GOLD.mkdir(parents=True, exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    write_single_parquet(part.select("uid", "iid", "ts", "rating"), GOLD / f"{name}.parquet", ("ts", "uid", "iid"))
write_single_parquet(user_map, GOLD / "user_map.parquet", ("uid",))
write_single_parquet(item_map, GOLD / "item_map.parquet", ("iid",))
sorted((p.name, f"{p.stat().st_size / 1e6:.2f} MB") for p in GOLD.glob("*.parquet"))
""")

md(r"""
## Acceptance 5 — Official 5-core benchmark se compare

McAuley Lab ne Amazon Reviews 2023 ke saath official 5-core files di hain:
- `benchmark/5core/rating_only/<cat>.csv` — poora 5-core (user, item, rating, timestamp)
- `benchmark/5core/timestamp/<cat>.{train,valid,test}.csv` — unka absolute-timestamp split

Sirf sanity check ke liye — hum unka split use nahi karte.
""")

code(r"""
from huggingface_hub import hf_hub_download
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
BENCH_FILES = {"rating_only.csv": f"benchmark/5core/rating_only/{CATEGORY}.csv",
               **{f"timestamp_{s}.csv": f"benchmark/5core/timestamp/{CATEGORY}.{s}.csv" for s in ("train", "valid", "test")}}

def fetch(repo_path: str, target: Path):
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / "_dl"
    os.replace(hf_hub_download(HF_REPO, repo_path, repo_type="dataset", local_dir=tmp), target)
    shutil.rmtree(tmp, ignore_errors=True)

for name, repo_path in BENCH_FILES.items():
    fetch(repo_path, BENCH / name)

read_bench = lambda name: (spark.read.option("header", True).csv(str(BENCH / name))
                           .select("user_id", F.col("parent_asin").alias("item_id"),
                                   (F.col("timestamp").cast("long") / 1000).cast("long").alias("ts")))
bench = read_bench("rating_only.csv").localCheckpoint()
bench_stats = shape(bench)

ours_pairs, bench_pairs = core.select("user_id", "item_id"), bench.select("user_id", "item_id").distinct()
overlap = {"both": ours_pairs.join(bench_pairs, ["user_id", "item_id"], "left_semi").count(),
           "ours_only": ours_pairs.join(bench_pairs, ["user_id", "item_id"], "left_anti").count(),
           "bench_only": bench_pairs.join(ours_pairs, ["user_id", "item_id"], "left_anti").count()}

official_split = {}
for s in ("train", "valid", "test"):
    r = read_bench(f"timestamp_{s}.csv").agg(F.count("*").alias("rows"),
                                             F.from_unixtime(F.min("ts")).alias("first"),
                                             F.from_unixtime(F.max("ts")).alias("last")).first().asDict()
    official_split[s] = r

bench_cmp = pd.DataFrame([{"source": "ours", **core_stats}, {"source": "official 5-core", **bench_stats}]).set_index("source")
display(bench_cmp)
print("(user, item) pair overlap:", overlap)
display(pd.DataFrame(official_split).T)
""")

md(r"""
## Acceptance 6 — Independent cross-check: Polars implementation

`data/prepare.py` same pipeline ko Polars (eager, single machine) mein implement karta hai. Do independent implementations ka
same answer dena = strong evidence ki logic sahi hai. k-core aur id maps **exactly** match hone chahiye; split counts mein
tiny farak ho sakta hai kyunki Polars exact quantile leta hai aur Spark approximate.
""")

code(r"""
sys.path.insert(0, str(REPO_ROOT))
import polars as pl
from data.prepare import _read_silver, dedup as pl_dedup, k_core as pl_k_core, map_ids
from data.split import temporal_split

pl_reviews = _read_silver(DATA_ROOT / "silver", CATEGORY).select("user_id", "item_id", "ts", "rating")
pl_core = pl_k_core(pl_dedup(pl_reviews), K)
pl_inter, pl_umap, pl_imap = map_ids(pl_core)
pl_tr, pl_va, pl_te, pl_tv, pl_tt = temporal_split(pl_inter, VAL_FRAC, TEST_FRAC)

g_umap = pl.read_parquet(GOLD / "user_map.parquet")
g_imap = pl.read_parquet(GOLD / "item_map.parquet")
xcheck = {
    "k_core_rows_equal": pl_core.height == core_stats["interactions"],
    "user_map_identical": g_umap.equals(pl_umap.select("uid", "user_id")),
    "item_map_identical": g_imap.equals(pl_imap.select("iid", "item_id")),
    "cutoffs_spark": [t_val, t_test], "cutoffs_polars_exact": [pl_tv, pl_tt],
    "split_rows_spark": [n_train, n_val, n_test], "split_rows_polars": [pl_tr.height, pl_va.height, pl_te.height],
}
assert xcheck["k_core_rows_equal"] and xcheck["user_map_identical"] and xcheck["item_map_identical"], xcheck
del pl_reviews
xcheck
""")

md("## prepare_stats.json — machine-readable record (REPORT.md ke numbers yahin se)")

code(r"""
STATS.update({
    "feedback_decision": {"selected": FEEDBACK, **fb, "rating_filter": None,
                          "verified_only_k_core": verified_stats},
    "prepare": {"raw": raw_stats, "dedup": dedup_stats, "k_core": core_stats,
                "k_core_iterations": kcore_log,
                "k_comparison": {"k": K_COMPARE, **cmp_stats, "iterations": cmp_log},
                "dedup_rule": "earliest ts per (user_id, item_id)",
                "full_k_core_before_temporal_split": True},
    "split": {"fractions": {"train": 1 - VAL_FRAC - TEST_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
              "method": f"global temporal; Spark approxQuantile relativeError={QUANTILE_REL_ERR} (approximate)",
              "cutoffs": {"val_start_ts_inclusive": t_val, "test_start_ts_inclusive": t_test,
                          "val_start_utc": dates["val_start"], "test_start_utc": dates["test_start"],
                          **ties},
              "stats": split_stats, "row_conservation": True},
    "id_mapping": {"mapped_full_k_core_before_split": True, "users": n_users, "items": n_items,
                   "method": "row_number over Window.orderBy(id) (single partition; zipWithIndex for L3)"},
    "official_5_core_benchmark": {"available": True, **bench_stats, "pair_overlap": overlap,
                                  "official_timestamp_split": official_split},
    "polars_cross_check": xcheck,
})
(GOLD / "prepare_stats.json").write_text(json.dumps(STATS, indent=2, default=str) + "\n", encoding="utf-8")
print("saved ->", GOLD / "prepare_stats.json")
""")

md(r"""
## Next

```bash
python run_baselines.py --category Video_Games --root data/gold
```
→ `results/Video_Games/baselines.md` — yahi Phase 1 ka real result hai. Numbers + decisions REPORT.md mein.
""")

code("spark.stop()")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("02_prepare_and_split.ipynb")
nbf.write(nb, out)
print("wrote", out)
