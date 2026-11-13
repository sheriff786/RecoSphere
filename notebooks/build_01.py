"""Builds 01_data_ingestion_research.ipynb (notebook ko code se generate karna = git-diffable source)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# RankForge · 01 — Data Ingestion Research

**Goal:** Amazon Reviews 2023 ki ek category ko samajhna aur *evidence ke saath* decide karna ki kya ingest karna hai.
Is notebook ke end par do cheezein hongi:

1. **Bronze** — original `.jsonl` files, bilkul untouched (immutable)
2. **Silver** — validated, typed, year-partitioned Parquet (zstd): `interactions` + `products`

**Sawal jinke jawab is notebook ko dene hain** (har ek ka answer neeche Findings mein likhna):

| # | Question | Kyun matter karta hai |
|---|---|---|
| Q1 | Raw records ka exact structure kya hai? Nested fields kaun se hain? | Schema decide karega |
| Q2 | Kitne rows corrupt / null / invalid hain? | Data quality baseline |
| Q3 | Timestamp kis unit mein hai, range kya hai, saal-wise volume kitna hai? | Temporal split + partitioning |
| Q4 | Duplicates kitne hain? `asin` vs `parent_asin` mein kitna farak hai? | Item id choice |
| Q5 | Users/items kitne sparse hain? 1-interaction users kitne %? | k-core decision |
| Q6 | Metadata kitne items cover karta hai? Text fields kitne bhare hain? | Cold-start / text tower feasibility |
""")

md(r"""
## Concepts pehle (3 decisions jo code se pehle samajhne zaroori hain)

**1. Bronze/Silver kyun?** Analogy: Bronze = original receipt ki photocopy, kabhi edit nahi hoti. Silver = us receipt se banayi saaf Excel sheet. Kal agar cleaning logic mein bug nikla, to Bronze se dobara Silver bana sakte ho — source dobara download nahi karna padega.

**2. `.jsonl` kyun, `.jsonl.gz` kyun nahi?** Gzip *splittable nahi* hota. Spark 22 GB ki `.gz` file ko sirf **ek task** mein padhega, baaki cores khaali baithenge. Plain `.jsonl` ko Spark chunks (~128 MB) mein tod ke parallel padhta hai. Hugging Face par files plain `.jsonl` hain — isliye wahi use kar rahe hain.

**3. Explicit schema kyun, `inferSchema` kyun nahi?** Inference ke liye Spark ko poora data ek extra baar scan karna padta hai (22 GB = minutes), aur agar kisi row mein `price` string ho aur kisi mein number, to type har run mein alag aa sakta hai. Explicit schema = fast + deterministic. Jo field type se match na kare, wo `_corrupt_record` mein jaata hai — jise hum count karenge (Q2).
""")

code(r"""
import json, os, shutil
from pathlib import Path

# ---------------- CONFIG ----------------
CATEGORY   = os.environ.get("RANKFORGE_CATEGORY", "Video_Games")   # L1. Baad mein: Movies_and_TV, Electronics
DATA_ROOT  = Path(os.environ.get("RANKFORGE_DATA", "../data")).resolve()
HF_REPO    = "McAuley-Lab/Amazon-Reviews-2023"
DRIVER_MEM = os.environ.get("RANKFORGE_DRIVER_MEM", "8g")          # 16 GB laptop par 8g theek hai

BRONZE_REVIEWS = DATA_ROOT / "bronze" / "reviews"  / f"category={CATEGORY}" / f"{CATEGORY}.jsonl"
BRONZE_META    = DATA_ROOT / "bronze" / "metadata" / f"category={CATEGORY}" / f"meta_{CATEGORY}.jsonl"
SILVER         = DATA_ROOT / "silver"
SILVER_INTER   = SILVER / "interactions" / f"category={CATEGORY}"
SILVER_PROD    = SILVER / "products"     / f"category={CATEGORY}"
print("data root:", DATA_ROOT)
""")

md(r"""
## Step 1 — Raw files → Bronze

File pehle se hai to skip. Hugging Face se seedha download (official README ka `load_dataset(..., trust_remote_code=True)` naye `datasets` versions mein fail ho sakta hai — isliye `hf_hub_download`).
""")

code(r"""
def to_bronze(repo_path: str, target: Path):
    if target.exists():
        print(f"exists, skip: {target.name}  ({target.stat().st_size / 1e9:.2f} GB)")
        return
    from huggingface_hub import hf_hub_download
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / "_dl"
    local = hf_hub_download(HF_REPO, repo_path, repo_type="dataset", local_dir=tmp)
    os.replace(local, target)          # move, copy nahi -> disk double nahi hoti
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"downloaded: {target.name}  ({target.stat().st_size / 1e9:.2f} GB)")

to_bronze(f"raw/review_categories/{CATEGORY}.jsonl", BRONZE_REVIEWS)
to_bronze(f"raw/meta_categories/meta_{CATEGORY}.jsonl", BRONZE_META)
""")

md("## Step 2 — Q1: Raw record ko aankh se dekho (Spark se pehle)\n\nSirf pehli 2 lines, plain Python. 20 GB file ko pandas mein load karne ki galti mat karna.")

code(r"""
def peek(path: Path, n=2, max_str=120):
    with open(path) as f:
        for _, line in zip(range(n), f):
            rec = json.loads(line)
            short = {k: (v[:max_str] + "…" if isinstance(v, str) and len(v) > max_str else v) for k, v in rec.items()}
            print(json.dumps(short, indent=2, default=str)[:2500]); print("-" * 60)
            print({k: type(v).__name__ for k, v in rec.items()}); print("=" * 60)

peek(BRONZE_REVIEWS)
""")

code("peek(BRONZE_META, n=1)")

md(r"""
**Dekhne layak cheezein:** `timestamp` mein kitne digits hain (13 = milliseconds)? `images` list of objects hai. Metadata mein `details` ek dict hai jiski keys har item mein alag hain, aur `price` kabhi number, kabhi `"None"` string hota hai. Ye sab schema decisions ko drive karta hai.
""")

md("## Step 3 — Spark session (local mode)\n\n`local[*]` = laptop ke saare cores. Same code baad mein EMR par chalega; sirf master aur paths (`s3a://...`) badlenge.")

code(r"""
from pyspark.sql import SparkSession, functions as F, types as T

spark = (SparkSession.builder
         .master("local[*]")
         .appName(f"rankforge-ingest-{CATEGORY}")
         .config("spark.driver.memory", DRIVER_MEM)
         .config("spark.sql.session.timeZone", "UTC")          # timestamps ka timezone fix
         .config("spark.sql.shuffle.partitions", "64")
         .config("spark.sql.parquet.compression.codec", "zstd")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
spark.version
""")

md(r"""
## Step 4 — Reviews: explicit schema ke saath read

Nested `images` ko `StringType` diya hai — Spark object ko raw JSON text ki tarah rakh deta hai. Silver mein images chahiye nahi, sirf "image tha ya nahi" count karna hai.
""")

code(r"""
REVIEW_SCHEMA = T.StructType([
    T.StructField("rating",            T.DoubleType()),
    T.StructField("title",             T.StringType()),
    T.StructField("text",              T.StringType()),
    T.StructField("images",            T.StringType()),
    T.StructField("asin",              T.StringType()),
    T.StructField("parent_asin",       T.StringType()),
    T.StructField("user_id",           T.StringType()),
    T.StructField("timestamp",         T.LongType()),
    T.StructField("helpful_vote",      T.LongType()),
    T.StructField("verified_purchase", T.BooleanType()),
    T.StructField("_corrupt_record",   T.StringType()),
])

reviews_raw = (spark.read
               .schema(REVIEW_SCHEMA)
               .option("mode", "PERMISSIVE")
               .option("columnNameOfCorruptRecord", "_corrupt_record")
               .json(str(BRONZE_REVIEWS))
               .cache())          # profiling mein baar-baar padhenge
n_raw = reviews_raw.count()
print(f"rows: {n_raw:,}   input partitions: {reviews_raw.rdd.getNumPartitions()}")
""")

md("## Step 5 — Q2: Data quality — corrupt rows, nulls, invalid values")

code(r"""
n_corrupt = reviews_raw.filter(F.col("_corrupt_record").isNotNull()).count()

valid_raw = reviews_raw.filter(F.col("_corrupt_record").isNull())   # corrupt rows alag gine gaye hain
null_counts = valid_raw.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in ["user_id", "parent_asin", "asin", "rating", "timestamp", "verified_purchase"]
]).first().asDict()

invalid = valid_raw.select(
    F.sum((~F.col("rating").between(1, 5)).cast("int")).alias("rating_out_of_range"),
    F.sum((F.col("timestamp") <= 0).cast("int")).alias("non_positive_ts"),
    F.sum((F.trim(F.coalesce("text", F.lit(""))) == "").cast("int")).alias("empty_text"),
).first().asDict()

print(f"corrupt rows: {n_corrupt:,} ({n_corrupt / n_raw:.4%})")
print("nulls:", null_counts)
print("invalid:", invalid)
""")

md("## Step 6 — Q3: Timestamp unit, range, saal-wise volume")

code(r"""
ts_stats = reviews_raw.select(
    F.min("timestamp").alias("min"), F.max("timestamp").alias("max"),
    F.expr("percentile_approx(length(cast(timestamp as string)), 0.5)").alias("median_digits"),
).first()
print(ts_stats)
TS_IS_MS = ts_stats["median_digits"] >= 13
print("timestamp unit:", "milliseconds" if TS_IS_MS else "seconds")

ts_sec = (F.col("timestamp") / 1000).cast("long") if TS_IS_MS else F.col("timestamp")
reviews = (reviews_raw
           .filter(F.col("_corrupt_record").isNull())
           .withColumn("ts", ts_sec)
           .withColumn("event_ts", F.col("ts").cast("timestamp"))
           .withColumn("event_year", F.year("event_ts")))

by_year = reviews.groupBy("event_year").count().orderBy("event_year").toPandas()
by_year["share"] = (by_year["count"] / by_year["count"].sum()).round(4)
by_year
""")

code(r"""
import matplotlib.pyplot as plt
ax = by_year.plot.bar(x="event_year", y="count", legend=False, figsize=(10, 3), color="#4C78A8")
ax.set_title(f"{CATEGORY}: interactions per year"); ax.set_ylabel("interactions"); ax.set_xlabel("")
plt.tight_layout(); plt.show()
""")

md(r"""
**Iska matlab:** agar shuru ke saalon mein bahut kam rows hain, to `event_year` partitioning se *small files problem* aata hai (bahut saari chhoti files = Spark ke liye slow). Isliye silver write mein per-partition files control karenge. Saath hi, temporal split ke cutoffs isi distribution se aate hain — "last 30 days" agar 0.5% data hai to test set bahut chhota hoga.
""")

md("## Step 7 — Q4: Duplicates aur `asin` vs `parent_asin`")

code(r"""
dup = reviews.select(
    F.count("*").alias("rows"),
    F.countDistinct("user_id", "parent_asin").alias("distinct_user_parent"),
    F.countDistinct("user_id", "asin", "timestamp").alias("distinct_exact"),
    F.countDistinct("asin").alias("asins"),
    F.countDistinct("parent_asin").alias("parent_asins"),
    F.countDistinct("user_id").alias("users"),
).first().asDict()

dup["exact_dup_rows"]      = dup["rows"] - dup["distinct_exact"]
dup["repeat_user_item_%"]  = round(100 * (1 - dup["distinct_user_parent"] / dup["rows"]), 3)
dup["asin_per_parent"]     = round(dup["asins"] / dup["parent_asins"], 3)
dup
""")

md(r"""
**Do alag cheezein, do alag decisions:**
- *Exact duplicates* (same user, asin, timestamp) = ingestion artefact → **Silver mein hatao**.
- *Repeat (user, parent_asin)* = user ne same product family ko dobara review kiya → ye **modeling decision** hai (implicit feedback mein ek signal), isliye Silver mein rakho, `prepare` step mein dedup karo. Silver ko modeling assumptions se saaf rakhna hai.
""")

md("## Step 8 — Q5: Sparsity — users aur items kitne thin hain")

code(r"""
def activity(col):
    per = reviews.groupBy(col).count()
    buckets = (per.withColumn("bucket", F.when(F.col("count") == 1, "1")
                                        .when(F.col("count") < 5, "2-4")
                                        .when(F.col("count") < 20, "5-19")
                                        .otherwise("20+"))
                  .groupBy("bucket").agg(F.count("*").alias(f"n_{col}"), F.sum("count").alias("interactions"))
                  .toPandas().set_index("bucket").loc[["1", "2-4", "5-19", "20+"]])
    buckets[f"%_{col}"] = (100 * buckets[f"n_{col}"] / buckets[f"n_{col}"].sum()).round(2)
    buckets["%_interactions"] = (100 * buckets["interactions"] / buckets["interactions"].sum()).round(2)
    return buckets

print("USERS"); display(activity("user_id"))
print("ITEMS (parent_asin)"); display(activity("parent_asin"))
""")

md(r"""
**k-core ka intuition:** `5-19` aur `20+` buckets dikhate hain ki 5-core ke baad roughly kitna data bachega (exact number iterative filter ke baad aayega, kyunki users hatane se items bhi girte hain). Agar 1-interaction users ka share bahut high hai, to ye data "personalization" ke liye kuch nahi deta — unke liye popularity hi best hai.
""")

md(r"""
## Step 9 — Metadata: Q6 coverage aur text quality

`price` ko string padh rahe hain (kyunki raw mein `"None"` aata hai) aur baad mein safe cast. `details` ka dict har item mein alag hai → raw JSON string rakho, zaroorat padne par `from_json`.
""")

code(r"""
META_SCHEMA = T.StructType([
    T.StructField("main_category",   T.StringType()),
    T.StructField("title",           T.StringType()),
    T.StructField("average_rating",  T.DoubleType()),
    T.StructField("rating_number",   T.LongType()),
    T.StructField("features",        T.ArrayType(T.StringType())),
    T.StructField("description",     T.ArrayType(T.StringType())),
    T.StructField("price",           T.StringType()),
    T.StructField("images",          T.StringType()),
    T.StructField("videos",          T.StringType()),
    T.StructField("store",           T.StringType()),
    T.StructField("categories",      T.ArrayType(T.StringType())),
    T.StructField("details",         T.StringType()),
    T.StructField("parent_asin",     T.StringType()),
    T.StructField("bought_together", T.StringType()),
    T.StructField("_corrupt_record", T.StringType()),
])
meta_raw = (spark.read.schema(META_SCHEMA).option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .json(str(BRONZE_META)).cache())

empty_arr = lambda c: (F.coalesce(F.size(c), F.lit(0)) <= 0)
blank     = lambda c: (F.trim(F.coalesce(F.col(c), F.lit(""))) == "")
meta_q = meta_raw.select(
    F.count("*").alias("rows"),
    F.countDistinct("parent_asin").alias("distinct_items"),
    F.sum(F.col("_corrupt_record").isNotNull().cast("int")).alias("corrupt"),
    F.avg(blank("title").cast("int")).alias("title_blank_%"),
    F.avg(empty_arr("description").cast("int")).alias("description_empty_%"),
    F.avg(empty_arr("features").cast("int")).alias("features_empty_%"),
    F.avg(empty_arr("categories").cast("int")).alias("categories_empty_%"),
    F.avg(blank("store").cast("int")).alias("store_blank_%"),
    F.avg(F.expr("try_cast(price as double)").isNotNull().cast("int")).alias("price_parsable_%"),
).first().asDict()
{k: (round(100 * v, 2) if k.endswith("%") else v) for k, v in meta_q.items()}
""")

code(r"""
# Coverage: kitne interacted items ke paas metadata hai, aur wo kitne % interactions cover karte hain
meta_items = meta_raw.select("parent_asin").distinct().withColumn("has_meta", F.lit(1))
cov = (reviews.groupBy("parent_asin").count()
       .join(meta_items, "parent_asin", "left")
       .select(F.avg(F.col("has_meta").isNotNull().cast("int")).alias("items_with_meta_%"),
               (F.sum(F.when(F.col("has_meta").isNotNull(), F.col("count"))) / F.sum("count")).alias("interactions_covered_%"))
       .first().asDict())
{k: round(100 * v, 2) for k, v in cov.items()}
""")

code(r"""
# details ki sabse common keys -> kaun se structured attributes features ban sakte hain
details_keys = (meta_raw
    .select(F.explode(F.map_keys(F.from_json("details", T.MapType(T.StringType(), T.StringType())))).alias("key"))
    .groupBy("key").count().orderBy(F.desc("count")).limit(15).toPandas())
details_keys["%_items"] = (100 * details_keys["count"] / meta_q["rows"]).round(1)
details_keys
""")

md(r"""
## Step 10 — Ingestion decision table

Upar ke numbers ke basis par. **Proposal** pre-filled hai — apne real numbers dekh ke "Why" column update karna.

**Silver `interactions`**

| Column | Keep? | Why |
|---|---|---|
| `user_id`, `parent_asin` → `item_id` | ✅ | Core interaction key; metadata join `parent_asin` par |
| `asin` | ✅ | Variant-level analysis ke liye sasta hai; modeling `item_id` par |
| `rating` | ✅ | Label nahi (implicit feedback), lekin feature/filter ke liye |
| `ts` (seconds), `event_ts`, `event_year` | ✅ | Split + partitioning |
| `verified_purchase`, `helpful_vote` | ✅ | Signal-quality features |
| `has_image` | ✅ (derived) | Sasta engagement proxy |
| `text`, `title` (review) | ❌ | Sabse heavy columns; abhi kisi model ko nahi chahiye. Bronze mein safe hain |
| `images` (raw) | ❌ | URLs, modeling value nahi |

**Silver `products`**

| Column | Keep? | Why |
|---|---|---|
| `parent_asin` → `item_id`, `title`, `features`, `description`, `categories` | ✅ | Item tower text + cold start |
| `store`, `main_category`, `price` (double), `average_rating`, `rating_number` | ✅ | Ranker features |
| `details` (raw JSON) | ✅ | Gold mein selected keys nikaalenge |
| `images`, `videos` | ❌ | Abhi multimodal scope mein nahi |
| `bought_together` | check | Agar mostly null nikla to drop |
""")

md("## Step 11 — Silver write")

code(r"""
silver_inter = (reviews
    .dropDuplicates(["user_id", "asin", "timestamp"])            # sirf EXACT duplicates
    .filter(F.col("user_id").isNotNull() & F.col("parent_asin").isNotNull() & F.col("ts").isNotNull())
    .select(
        "user_id",
        F.col("parent_asin").alias("item_id"),
        "asin",
        F.col("rating").cast("float"),
        "ts", "event_ts", "event_year",
        F.col("verified_purchase").alias("verified"),
        F.col("helpful_vote").cast("int"),
        (F.coalesce(F.col("images"), F.lit("[]")) != "[]").alias("has_image"),
    ))

(silver_inter
    .repartition("event_year")                                   # ek saal = ek task -> small files kam
    .write.mode("overwrite")
    .option("maxRecordsPerFile", 5_000_000)                       # bade saal ko split karo
    .partitionBy("event_year")
    .parquet(str(SILVER_INTER)))

silver_prod = (meta_raw
    .filter(F.col("_corrupt_record").isNull() & F.col("parent_asin").isNotNull())
    .dropDuplicates(["parent_asin"])
    .select(
        F.col("parent_asin").alias("item_id"),
        "title", "main_category", "store",
        F.expr("try_cast(price as double)").alias("price"),
        "average_rating", F.col("rating_number").cast("int"),
        "features", "description", "categories", "details",
    ))
silver_prod.coalesce(8).write.mode("overwrite").parquet(str(SILVER_PROD))
print("silver written")
""")

md("## Step 12 — Validation: read back, counts match, files sane")

code(r"""
si = spark.read.parquet(str(SILVER_INTER))
sp_ = spark.read.parquet(str(SILVER_PROD))
n_si, n_sp = si.count(), sp_.count()
n_valid = reviews.filter(F.col("user_id").isNotNull() & F.col("parent_asin").isNotNull()).count()
exact_dups = dup["exact_dup_rows"]

print(f"bronze rows {n_raw:,} -> corrupt {n_corrupt:,} -> exact dups {exact_dups:,} -> silver {n_si:,}")
assert n_si <= n_valid, "silver mein bronze se zyada rows? join/explode bug"
assert si.filter(F.col("ts").isNull()).count() == 0

def dir_stats(p: Path):
    files = list(p.rglob("*.parquet"))
    return len(files), sum(f.stat().st_size for f in files) / 1e9

for name, p in [("bronze reviews", BRONZE_REVIEWS), ("bronze meta", BRONZE_META)]:
    print(f"{name:<16} {p.stat().st_size / 1e9:7.3f} GB")
for name, p in [("silver inter", SILVER_INTER), ("silver products", SILVER_PROD)]:
    n, gb = dir_stats(p); print(f"{name:<16} {gb:7.3f} GB in {n} files")
si.printSchema()
""")

code(r"""
# Ingestion report -> REPORT.md ke liye machine-readable record
report = {
    "category": CATEGORY, "bronze_rows": n_raw, "corrupt_rows": n_corrupt,
    "exact_duplicate_rows": exact_dups, "silver_interactions": n_si, "silver_products": n_sp,
    "users": dup["users"], "items_parent_asin": dup["parent_asins"], "asins": dup["asins"],
    "timestamp_unit": "ms" if TS_IS_MS else "s",
    "first_ts": str(si.agg(F.min("event_ts")).first()[0]), "last_ts": str(si.agg(F.max("event_ts")).first()[0]),
    "repeat_user_item_pct": dup["repeat_user_item_%"],
    "meta_quality": {k: (round(100 * v, 2) if k.endswith("%") else v) for k, v in meta_q.items()},
    "meta_coverage": {k: round(100 * v, 2) for k, v in cov.items()},
}
out = SILVER / f"_ingest_report_{CATEGORY}.json"
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
""")

md(r"""
## Findings (REPORT.md mein copy karo — apne real numbers ke saath)

- **Q1 Schema:** …
- **Q2 Quality:** corrupt = …, nulls = …, empty review text = …
- **Q3 Time:** unit = …, range = … → …; last 2 saal = …% of data → split cutoffs ka implication: …
- **Q4 Duplicates:** exact = …, repeat (user, item) = …%, asin/parent = … → item id = `parent_asin` kyunki …
- **Q5 Sparsity:** 1-interaction users = …% (… % of interactions) → k-core choice: …
- **Q6 Metadata:** coverage = …% items / …% interactions; description empty = …% → text tower ke liye input = title + features + …
- **Storage:** bronze … GB → silver … GB (… x compression), kyunki text drop + Parquet/zstd

**Next notebook:** `02_prepare_and_split` — Silver se k-core + temporal split, PySpark mein.
""")

code("spark.stop()")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "01_data_ingestion_research.ipynb")
print("ok", len(C), "cells")
