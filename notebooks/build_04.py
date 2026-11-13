"""Builds 04_ann_faiss.ipynb  (edit this file, then: python build_04.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# RankForge · 04 — ANN Retrieval (FAISS)

**Sawal:** user ka vector aa gaya. Ab saare items mein se sabse kareeb 200 kaise nikaalein — **tez**, **kam memory** mein,
aur **bina zyada accuracy khoye**?

**Exact search** = user vector ko har item se multiply karo (N dot products). 25.6k items par ye milliseconds hai,
lekin 1.6M items × hazaron requests/second par mehenga ho jaata hai. **ANN (Approximate Nearest Neighbour)** thodi
accuracy chhod ke search ko bahut tez/sasta banata hai.

| Index | Analogy | Knob |
|---|---|---|
| **Flat** (exact) | Library ki har kitaab ek-ek karke dekho | — |
| **HNSW** | Doston ke doston se poochte-poochte sahi insaan tak pahuncho (graph) | `efSearch`: kitne raaste explore karo |
| **IVF** | Library ke sections — sirf sabse relevant shelves dekho | `nprobe`: kitni shelves |
| **IVF-PQ** | IVF + har kitaab ka chhota summary rakho (compression) | `nprobe`; memory bahut kam |

**Kya measure karte hain:**
1. **Index recall@K** — ANN ke top-K mein se kitne exact top-K mein bhi hain (sirf search ki quality)
2. **End-to-end Recall@100 / NDCG@10** — kya ANN se asli metric gira? (wahi `evaluate()` jo notebook 03 mein)
3. **Latency p50 / p95** — ek user, ek query, ek CPU thread (serving jaisa)
4. **Memory** aur **build time**

> **Honest note:** L1 par sirf ~25.6k items hain — yahan exact search bhi tez hai, isliye ANN ka faayda chhota dikhega.
> Asli decision L2 (747.8k items) par hoga. Section 5 mein ek **synthetic** scale test hai jo trade-off ka *shape* dikhata hai —
> uske numbers resume ke liye nahi hain.
""")

md("## Step 0 — Config\n\n`faiss-cpu` chahiye. Windows par torch aur faiss dono OpenMP library laate hain — `KMP_DUPLICATE_LIB_OK` us crash ko rokta hai.")

code(r"""
import os
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import hashlib, json, sys, time
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch

CATEGORY  = os.environ.get("RANKFORGE_CATEGORY", "Video_Games")
DATA_ROOT = Path(os.environ.get("RANKFORGE_DATA", "../data")).resolve()
REPO_ROOT = Path(os.environ.get("RANKFORGE_REPO", "..")).resolve()
GOLD    = DATA_ROOT / "gold" / CATEGORY
MODELS  = REPO_ROOT / "models" / CATEGORY
RESULTS = REPO_ROOT / "results" / CATEGORY
sys.path.insert(0, str(REPO_ROOT))
from eval.metrics import build_matrix, evaluate
from retrieval.two_tower import embed_items, embed_users, history_at_cutoff, load_two_tower

K_EVAL      = 100                                          # metric cutoff (notebook 03 jaisa)
K_SEARCH    = int(os.environ.get("K_SEARCH", 200))         # ANN se kitne candidates (seen items mask hone ke baad bhi ~100 bachein)
N_LAT       = int(os.environ.get("N_LAT", 1000))           # latency ke liye kitni single queries
SCALE_N     = int(os.environ.get("SCALE_N", 500_000))      # synthetic scale test (0 = skip)
SEED        = 42
N_THREADS   = os.cpu_count() or 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
faiss.omp_set_num_threads(N_THREADS)
print(f"faiss {faiss.__version__} | threads {N_THREADS} | device {DEVICE}")
""")

md(r"""
## Step 1 — Model load + embedding consistency check

Model `retrieval/two_tower.py` se load hota hai (`load_state_dict(strict=True)` — architecture notebook 03 se alag hua to yahin fail).
Phir item vectors **dobara** compute karke notebook 03 ke saved `item_emb.npy` se compare karte hain.
Ye production ka **"embedding version mismatch"** bug pakadne wala check hai: index ek model ke vectors se bana ho aur
user vectors doosre model se aayein, to results chupchaap kachra ho jaate hain.
""")

code(r"""
train = pl.read_parquet(GOLD / "train.parquet"); val = pl.read_parquet(GOLD / "val.parquet")
test  = pl.read_parquet(GOLD / "test.parquet")
N_USERS = pl.read_parquet(GOLD / "user_map.parquet").height
N_ITEMS = pl.read_parquet(GOLD / "item_map.parquet").height
train_val = pl.concat([train, val])

model, ck, dfa = load_two_tower(MODELS / "two_tower.pt", GOLD, DEVICE)
print("variant:", ck["variant"], "| flags:", ck["flags"])

V = embed_items(model, dfa, ck["t_ref"])
V_saved = np.load(MODELS / "item_emb.npy")
max_diff = float(np.abs(V - V_saved).max())
assert V.shape == V_saved.shape and max_diff < 1e-4, f"embedding version mismatch: max diff {max_diff}"
EMB_VERSION = hashlib.sha1(V_saved.tobytes()).hexdigest()[:12]
V = V_saved                                                # index hamesha saved artifact se banta hai
print(f"item vectors {V.shape} | recompute max diff {max_diff:.2e} | version {EMB_VERSION}")

H = history_at_cutoff(train_val, N_USERS, ck["cfg"]["hist_len"])
U = embed_users(model, H)
X_hist = build_matrix(train_val, N_USERS, N_ITEMS)
known_items = train_val["iid"].unique().to_numpy()

has_hist = np.diff(X_hist.indptr) > 0
EVAL_USERS = np.array(sorted(set(test["uid"].unique().to_list()) & set(np.flatnonzero(has_hist).tolist())))
Q = U[EVAL_USERS]
seen = np.diff(X_hist.indptr)[EVAL_USERS]
print(f"eval users: {len(EVAL_USERS):,} | users with > {K_SEARCH - K_EVAL} seen items "
      f"(mask ke baad < {K_EVAL} candidates bachenge): {(seen > K_SEARCH - K_EVAL).mean():.2%}")
""")

md(r"""
## Step 2 — Exact reference (ground truth)

Teen level ke numbers:
1. **Full brute force** — saare items score, phir top-100 (yahi notebook 03 ka test number hona chahiye)
2. **Flat @ K_SEARCH** — exact search, lekin sirf top-200 candidates → ANN se apples-to-apples comparison
3. ANN configs (Step 3) — inko #2 se compare karte hain
""")

code(r"""
SHOW = ["all/recall@100", "all/ndcg@10", "warm/recall@100", "cold/recall@100"]

full = evaluate(lambda ub: U[ub] @ V.T, X_hist, test, known_items, ks=(10, 50, 100))
print("full brute force:", {k: round(full[k], 4) for k in SHOW})
nb03 = RESULTS / "two_tower.csv"
if nb03.exists():
    ref = pl.read_csv(nb03).filter(pl.col("model") == f"two_tower_{ck['variant']}")
    if ref.height:
        print("notebook 03     :", {k: round(ref[k][0], 4) for k in SHOW})

row_of = np.full(N_USERS, -1); row_of[EVAL_USERS] = np.arange(len(EVAL_USERS))

def eval_from_ids(I, D):
    # sirf retrieved candidates ko score milta hai; baaki -inf (ANN ne unhe dekha hi nahi)
    def score_fn(ub):
        r = row_of[ub]
        S = np.full((len(ub), N_ITEMS), -np.inf, dtype=np.float32)
        ok = (r >= 0)[:, None] & (I[np.clip(r, 0, None)] >= 0)
        rows = np.broadcast_to(np.arange(len(ub))[:, None], ok.shape)
        S[rows[ok], I[np.clip(r, 0, None)][ok]] = D[np.clip(r, 0, None)][ok]
        return S
    return evaluate(score_fn, X_hist, test, known_items, ks=(10, 50, 100))

flat_ref = faiss.IndexFlatIP(V.shape[1]); flat_ref.add(V)
GT_D, GT_I = flat_ref.search(Q, K_SEARCH)
exact_k = eval_from_ids(GT_I, GT_D)
print(f"flat @ {K_SEARCH}      :", {k: round(exact_k[k], 4) for k in SHOW})
""")

md(r"""
## Step 3 — Index benchmark

| Index | Params | Note |
|---|---|---|
| Flat | — | exact |
| HNSW | M=32, efConstruction=200, efSearch ∈ {K, 2K, 4K} | FAISS andar se `max(efSearch, k)` use karta hai, isliye K se kam efSearch ka koi matlab nahi |
| IVF-Flat | nlist ≈ 2√N, nprobe ∈ {1, 4, 16, 64} | |
| IVF-PQ | nlist ≈ 2√N, 16 sub-vectors × 8 bit, nprobe ∈ {1, 4, 16, 64} | har vector 64 floats (256 B) → 16 B |

Vectors L2-normalized hain → inner product = cosine similarity.

Chhote data par FAISS `please provide at least N training points` warning de sakta hai (IVF/PQ ko clusters seekhne ke liye
kaafi vectors chahiye). L1 ke ~25.6k items par ye nahi aana chahiye; aaye to nlist chhota karo.
""")

code(r"""
def build_index(kind, X, nlist=None):
    d = X.shape[1]
    t0 = time.perf_counter()
    if kind == "flat":
        idx = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 200
    elif kind == "ivf":
        idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(X)
    elif kind == "ivfpq":
        idx = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 16, 8, faiss.METRIC_INNER_PRODUCT)
        idx.train(X)
    else:
        raise ValueError(kind)
    idx.add(X)
    return idx, time.perf_counter() - t0, faiss.serialize_index(idx).nbytes / 1e6

def set_knob(idx, kind, val):
    if kind == "hnsw":
        idx.hnsw.efSearch = val
    elif kind in ("ivf", "ivfpq"):
        idx.nprobe = val

def latency_ms(idx, queries, k):
    # serving jaisa: ek query, ek thread
    faiss.omp_set_num_threads(1)
    times = []
    for q in queries:
        t0 = time.perf_counter(); idx.search(q[None, :], k); times.append((time.perf_counter() - t0) * 1e3)
    faiss.omp_set_num_threads(N_THREADS)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))

def index_recall(I, GT):
    return float(np.mean([len(np.intersect1d(a[a >= 0], b)) / len(b) for a, b in zip(I, GT)]))

N = len(V)
NLIST = max(16, int(2 * np.sqrt(N)))
CONFIGS = [("flat", None)] + [("hnsw", e) for e in (K_SEARCH, 2 * K_SEARCH, 4 * K_SEARCH)] \
        + [("ivf", p) for p in (1, 4, 16, 64)] + [("ivfpq", p) for p in (1, 4, 16, 64)]
lat_q = Q[rng.choice(len(Q), size=min(N_LAT, len(Q)), replace=False)]

rows, built = [], {}
for kind, knob in CONFIGS:
    if kind not in built:
        built[kind] = build_index(kind, V, NLIST)
    idx, build_s, mem_mb = built[kind]
    set_knob(idx, kind, knob)
    t0 = time.perf_counter(); D, I = idx.search(Q, K_SEARCH); batch_s = time.perf_counter() - t0
    p50, p95 = latency_ms(idx, lat_q, K_SEARCH)
    r = eval_from_ids(I, D)
    rows.append({"index": kind, "knob": knob, "index_recall@K": index_recall(I, GT_I),
                 **{k: r[k] for k in SHOW}, "p50_ms": p50, "p95_ms": p95,
                 "batch_qps": len(Q) / batch_s, "memory_mb": mem_mb, "build_s": build_s})
    print(f"{kind:>6} knob={str(knob):>4}  idx_recall={rows[-1]['index_recall@K']:.4f}  "
          f"recall@100={r['all/recall@100']:.4f}  p95={p95:.3f}ms  mem={mem_mb:.1f}MB")

bench = pl.DataFrame(rows)
with pl.Config(tbl_rows=30, float_precision=4, tbl_width_chars=160):
    print(bench)
""")

code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for kind in ["flat", "hnsw", "ivf", "ivfpq"]:
    b = bench.filter(pl.col("index") == kind)
    ax[0].plot(b["p95_ms"], b["index_recall@K"], marker="o", label=kind)
    for kn, x, y in zip(b["knob"].to_list(), b["p95_ms"].to_list(), b["index_recall@K"].to_list()):
        if kn is not None:
            ax[0].annotate(str(kn), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[0].set_xlabel("p95 latency per query (ms, 1 thread)"); ax[0].set_ylabel(f"index recall@{K_SEARCH}")
ax[0].set_title("Recall vs latency (upar-baayen = best)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
mem = bench.group_by("index", maintain_order=True).agg(pl.col("memory_mb").first())
ax[1].bar(mem["index"], mem["memory_mb"], color="#4C78A8"); ax[1].set_ylabel("MB"); ax[1].set_title("Index memory")
plt.tight_layout(); plt.show()
""")

md(r"""
## Step 4 — Index choose karo aur save karo (serving ke liye)

**Rule:** retrieval step ka **latency budget** (default p95 ≤ 5 ms). Budget ke andar jo config sabse zyada index recall de, wahi;
tie ho to kam latency. Matlab: jab tak exact search budget mein hai, approximate lene ka koi faayda nahi.
L1 par isliye **Flat** aana expected hai — "is scale par ANN ki zaroorat nahi" ek valid, honest finding hai.
L2/L3 par jab Flat budget se bahar jaayega, tab ANN apne aap chuna jaayega.

Index ke saath `EMB_VERSION` save hota hai. Serving (Phase 6) check karega ki user tower aur index same version ke hain.
""")

code(r"""
LATENCY_BUDGET_MS = float(os.environ.get("LATENCY_BUDGET_MS", 5.0))
ok = bench.filter(pl.col("p95_ms") <= LATENCY_BUDGET_MS).sort(["index_recall@K", "p95_ms"], descending=[True, False])
if ok.height == 0:
    print(f"WARNING: koi config {LATENCY_BUDGET_MS} ms budget mein nahi -> sabse tez config le rahe hain")
chosen = (ok if ok.height else bench.sort("p95_ms")).row(0, named=True)
idx, _, _ = built[chosen["index"]]
set_knob(idx, chosen["index"], chosen["knob"])
name = chosen["index"] + (f"_{chosen['knob']}" if chosen["knob"] is not None else "")
MODELS.mkdir(parents=True, exist_ok=True)
faiss.write_index(idx, str(MODELS / "faiss.index"))
meta = {"index": chosen["index"], "knob": chosen["knob"], "k_search": K_SEARCH, "emb_version": EMB_VERSION,
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "n_items": N_ITEMS, "dim": int(V.shape[1]), "nlist": NLIST, "model_variant": ck["variant"],
        "metrics": {k: chosen[k] for k in ["index_recall@K", "all/recall@100", "p50_ms", "p95_ms", "memory_mb"]}}
(MODELS / "faiss_meta.json").write_text(json.dumps(meta, indent=2))

RESULTS.mkdir(parents=True, exist_ok=True)
bench.write_csv(RESULTS / "ann_benchmark.csv")
cols = ["index", "knob", "index_recall@K", "all/recall@100", "all/ndcg@10", "p50_ms", "p95_ms", "memory_mb", "build_s"]
lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
for r in bench.select(cols).iter_rows(named=True):
    lines.append("| " + " | ".join(str(r[c]) if c in ("index", "knob") else f"{r[c]:.4f}" for c in cols) + " |")
(RESULTS / "ann_benchmark.md").write_text(
    f"K_SEARCH={K_SEARCH}, items={N_ITEMS:,}, eval users={len(EVAL_USERS):,}, threads(latency)=1\n\n" + "\n".join(lines) + "\n",
    encoding="utf-8")
print(f"chosen: {name} -> {MODELS / 'faiss.index'}")
print(json.dumps(meta, indent=2))
""")

md(r"""
## Step 5 — Synthetic scale stress test (trade-off ka shape)

⚠️ **Synthetic data.** Real item vectors ko random sample karke unme thoda noise daal ke `SCALE_N` vectors banate hain,
queries asli user vectors. Ye sirf dikhata hai ki items badhne par Flat ki latency kaise badhti hai aur IVF-PQ memory kaise bachata hai.
**Ye numbers resume/REPORT ke "result" nahi hain** — asli numbers L2 (Movies_and_TV) par aayenge. `SCALE_N=0` = skip.

**Dekhne layak:** IVF-PQ mein agar `nprobe` badhane se recall nahi badhta, to bottleneck shelves nahi, **compression** hai
(PQ ne vectors ko itna chhota kiya ki fine detail kho gayi). Iska standard fix: PQ se zyada candidates lo aur unhe exact vectors se
re-score karo (`IndexRefineFlat`). Ye L2 par try karne layak experiment hai.
""")

code(r"""
if SCALE_N > 0:
    d = V.shape[1]
    Xs = V[rng.integers(0, N, SCALE_N)] + rng.normal(0, 0.08, (SCALE_N, d)).astype(np.float32)
    Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    qs = Q[rng.choice(len(Q), size=min(300, len(Q)), replace=False)]
    gt = faiss.IndexFlatIP(d); gt.add(Xs); _, GTs = gt.search(qs, K_EVAL)
    nl = max(16, int(2 * np.sqrt(SCALE_N)))
    srows = []
    for kind, knobs in [("flat", [None]), ("hnsw", [K_EVAL, 2 * K_EVAL, 4 * K_EVAL]), ("ivf", [8, 32]), ("ivfpq", [8, 32])]:
        idx_s, bs, mb = build_index(kind, Xs, nl)
        for kn in knobs:
            set_knob(idx_s, kind, kn)
            _, Is = idx_s.search(qs, K_EVAL)
            p50, p95 = latency_ms(idx_s, qs, K_EVAL)
            srows.append({"index": kind, "knob": kn, "index_recall@100": index_recall(Is, GTs),
                          "p50_ms": p50, "p95_ms": p95, "memory_mb": mb, "build_s": bs})
            print(srows[-1])
        del idx_s
    scale = pl.DataFrame(srows)
    scale.write_csv(RESULTS / f"ann_scale_synthetic_{SCALE_N}.csv")
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=140):
        print(f"SYNTHETIC — {SCALE_N:,} vectors"); print(scale)
""")

md(r"""
## Findings (REPORT.md — apne real numbers ke saath)

- **Consistency:** recompute vs saved item_emb max diff = … (version `…`); full brute force = notebook 03 ka number? haan/nahi
- **L1 result:** Flat p95 = … ms; best ANN = … (index recall …, p95 … ms) → L1 par ANN ki zaroorat hai/nahi
- **End-to-end:** ANN se Recall@100 kitna gira (Flat@K vs ANN): …
- **Memory:** Flat … MB vs IVF-PQ … MB (… x kam)
- **Synthetic scale (method demo, not a result):** items badhne par Flat latency … ; HNSW/IVF-PQ …
- **Decision:** chosen index = … ; L2 par dobara benchmark karna hai

**Next:** Phase 4 — candidates (two-tower via FAISS + recent popularity) → LightGBM ranker.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "04_ann_faiss.ipynb")
print("ok", len(C), "cells")
