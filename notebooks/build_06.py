"""Builds 06_beyond_accuracy.ipynb  (edit this file, then: python build_06.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# RankForge · 06 — Beyond Accuracy

**Analogy:** ek restaurant jo har customer ko sirf apni bestseller dish suggest karta hai. Short term mein orders theek aate hain,
lekin naye dishes kabhi try nahi hote, menu marta hai, aur har customer ko same cheez milti hai. Accuracy (Recall/NDCG) ye nahi pakadti.

Is notebook ke sawal:

| Sawal | Metric |
|---|---|
| Catalog ka kitna hissa kabhi dikhta bhi hai? | **Coverage@10** = top-10 lists mein alag games / kul games |
| Kya exposure kuch hit games par simat gaya? | **Gini@10** (0 = sabko barabar exposure, 1 = sab ek game ko) + Lorenz curve |
| Kya hum "obvious" cheezein hi dikha rahe? | **Novelty** = recommended games kitne kam-popular hain (−log₂ popularity) |
| Naye games ko jagah milti hai? | **New-item share@10** = top-10 mein kitne games history mein kabhi nahi aaye |
| List mein variety hai? | **Distinct publishers@10**, **top-publisher share** |
| Kaunse users ke liye system fail hota hai? | History length segments (1-2, 3-5, 6-10, 11-20, 21+ items) |
| Bina history wale users (test ke ~13.5%)? | **Cold-user fallback** — val par chuno, test par report |
| Kya simple rules variety badha sakte hain? | **Re-rank**: publisher cap, new-item slots → accuracy ka kitna nuksaan |

Notebook 05 ne dikhaya tha ki **ranker cold items par two-tower se haara** (0.0378 → 0.0282). Re-rank wala section isi ko test karta hai.
""")

md("## Step 0 — Config")

code(r"""
import os
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys, time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch

CATEGORY  = os.environ.get("RANKFORGE_CATEGORY", "Video_Games")
DATA_ROOT = Path(os.environ.get("RANKFORGE_DATA", "../data")).resolve()
REPO_ROOT = Path(os.environ.get("RANKFORGE_REPO", "..")).resolve()
GOLD, MODELS, RESULTS = DATA_ROOT / "gold" / CATEGORY, REPO_ROOT / "models" / CATEGORY, REPO_ROOT / "results" / CATEGORY
sys.path.insert(0, str(REPO_ROOT))
from eval.metrics import build_matrix, evaluate
from ranking.pipeline import FEATS, features, gen_candidates, make_context, recent_popularity
from retrieval.two_tower import load_two_tower

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_SHOW = 10                      # user ko jo dikhta hai
print("device:", DEVICE)
""")

md(r"""
## Step 1 — Notebook 05 ka pipeline dobara chalao (test window)

Same module (`ranking/pipeline.py`), same saved models. **Consistency check:** ranker ka Recall@100 notebook 05 ke `ranker.csv` se match hona chahiye.
""")

code(r"""
train = pl.read_parquet(GOLD / "train.parquet"); val = pl.read_parquet(GOLD / "val.parquet")
test  = pl.read_parquet(GOLD / "test.parquet"); train_val = pl.concat([train, val])
N_USERS = pl.read_parquet(GOLD / "user_map.parquet").height
item_feat = pl.read_parquet(GOLD / "item_features.parquet").sort("iid")
N_ITEMS = item_feat.height
T_VAL, T_TEST = int(val["ts"].min()), int(test["ts"].min())

model_tv, ck, dfa_t = load_two_tower(MODELS / "two_tower.pt", GOLD, DEVICE)
TEXT_EMB = np.load(GOLD / f"item_text_emb_{ck['cfg']['text_encoder']}.npy")
rmeta = json.loads((MODELS / "ranker_meta.json").read_text())
assert rmeta["features"] == FEATS, "ranker features != pipeline FEATS -> notebook 05 dobara chalao"
booster = lgb.Booster(model_file=str(MODELS / "ranker.txt"))
CTX = make_context(item_feat, TEXT_EMB, dfa_t, N_USERS, ck["cfg"]["hist_len"], rmeta["n_tt"], rmeta["n_pop"], rmeta["recent_frac"])
STORE = CTX.store

t0 = time.time()
c = gen_candidates(CTX, model_tv, train_val, test, T_TEST)
F = features(CTX, c, train_val)
print(f"candidates {len(c['u']):,} rows for {len(c['users']):,} users ({time.time() - t0:.0f}s)")

SCORES = {
    "recent_popularity": np.where(np.isnan(c["pop_rank"]), -1e9, -np.nan_to_num(c["pop_rank"])).astype(np.float32),
    "two_tower":         c["tt_score"],
    "ranker":            booster.predict(F).astype(np.float32),
}
known_tv = train_val["iid"].unique().to_numpy()
starts = np.searchsorted(c["u"], c["users"]); ends = np.append(starts[1:], len(c["u"]))

def score_fn_for(scores):
    where = dict(zip(c["users"].tolist(), zip(starts.tolist(), ends.tolist())))
    def fn(ub):
        S = np.full((len(ub), N_ITEMS), -np.inf, dtype=np.float32)
        for r, u in enumerate(ub):
            if u in where:
                s0, s1 = where[u]; S[r, c["i"][s0:s1]] = scores[s0:s1]
        return S
    return fn

r_rank = evaluate(score_fn_for(SCORES["ranker"]), c["X"], test, known_tv, ks=(10, 100))
ref = pl.read_csv(RESULTS / "ranker.csv").filter(pl.col("model") == "lightgbm ranker")["all/recall@100"][0]
print(f"ranker Recall@100 now {r_rank['all/recall@100']:.4f} vs notebook 05 {ref:.4f}")
assert abs(r_rank["all/recall@100"] - ref) < 1e-3, "pipeline output changed since notebook 05"
""")

md(r"""
## Step 2 — Per-user lists aur metrics

Har model ke liye har user ki top-100 list (candidates ko score se sort). Per-user Recall/NDCG nikaalte hain taaki
segments bana sakein. **Check:** per-user ka average = `evaluate()` ka number (same users, same formula).
""")

code(r"""
def top_lists(scores, k=100):
    L = np.full((len(c["users"]), k), -1, dtype=np.int64)
    for r, (s0, s1) in enumerate(zip(starts, ends)):
        order = np.argsort(-scores[s0:s1], kind="stable")[:k]
        L[r, :len(order)] = c["i"][s0:s1][order]
    return L

LISTS = {name: top_lists(s) for name, s in SCORES.items()}

N_HIST = np.diff(c["X"].indptr)
IS_NEW = np.bincount(train_val["iid"].to_numpy(), minlength=N_ITEMS) == 0      # history mein kabhi nahi aaya
TKEYS = np.sort(c["target_u"] * N_ITEMS + c["target_i"])
N_T = np.bincount(c["target_u"], minlength=N_USERS)[c["users"]]
N_T_COLD = np.bincount(c["target_u"][IS_NEW[c["target_i"]]], minlength=N_USERS)[c["users"]]
DISC = 1.0 / np.log2(np.arange(2, 12))

def per_user(L):
    hits = np.isin(c["users"][:, None] * N_ITEMS + L, TKEYS) & (L >= 0)
    idcg = np.cumsum(DISC)[np.minimum(N_T, 10) - 1]
    cold_hits = hits & IS_NEW[np.clip(L, 0, None)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({
            "recall@10": hits[:, :10].sum(1) / N_T, "recall@100": hits.sum(1) / N_T,
            "ndcg@10": (hits[:, :10] * DISC).sum(1) / idcg,
            "cold_recall@10": np.where(N_T_COLD > 0, cold_hits[:, :10].sum(1) / N_T_COLD, np.nan),
            "cold_recall@100": np.where(N_T_COLD > 0, cold_hits.sum(1) / N_T_COLD, np.nan)})

PU = {name: per_user(L) for name, L in LISTS.items()}
print(f"ranker per-user mean Recall@100 = {PU['ranker']['recall@100'].mean():.4f} | evaluate() = {r_rank['all/recall@100']:.4f}")
assert abs(PU["ranker"]["recall@100"].mean() - r_rank["all/recall@100"]) < 2e-3
""")

md(r"""
## Step 3 — Beyond-accuracy metrics (top-10, jo user ko dikhta hai)

- **Gini** formula: exposure counts ko chhote se bade sort karo; `G = 2·Σ(i·xᵢ)/(n·Σx) − (n+1)/n`
- **Novelty** = recommended items ka average `−log₂((pop+1)/(total+N))` — bada = kam popular, zyada "discovery"
""")

code(r"""
HIST_POP = np.bincount(train_val["iid"].to_numpy(), minlength=N_ITEMS).astype(np.float64)

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n) if x.sum() > 0 else 0.0

def beyond(L10):
    items = L10[L10 >= 0]
    exposure = np.bincount(items, minlength=N_ITEMS)
    pubs = [np.unique(STORE[row[row >= 0]]) for row in L10]
    top_pub = [np.bincount(STORE[row[(row >= 0)]][STORE[row[(row >= 0)]] > 1]).max() / K_SHOW
               if (STORE[row[row >= 0]] > 1).any() else 0.0 for row in L10]
    return {"coverage@10": len(np.unique(items)) / N_ITEMS, "gini@10": gini(exposure),
            "novelty": float(np.mean(-np.log2((HIST_POP[items] + 1) / (HIST_POP.sum() + N_ITEMS)))),
            "new_item_share@10": float(IS_NEW[items].mean()),
            "distinct_publishers@10": float(np.mean([len(p) for p in pubs])),
            "top_publisher_share@10": float(np.mean(top_pub))}

rows = []
for name, L in LISTS.items():
    acc = PU[name][["recall@10", "ndcg@10", "recall@100", "cold_recall@100"]].mean().to_dict()
    rows.append({"model": name, **acc, **beyond(L[:, :K_SHOW])})
BEYOND = pd.DataFrame(rows).set_index("model")
BEYOND.round(4)
""")

code(r"""
fig, ax = plt.subplots(figsize=(6, 4.2))
for name, L in LISTS.items():
    x = np.sort(np.bincount(L[:, :K_SHOW][L[:, :K_SHOW] >= 0], minlength=N_ITEMS))
    ax.plot(np.linspace(0, 1, len(x)), np.cumsum(x) / x.sum(), label=f"{name} (Gini {gini(x):.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect equality")
ax.set_xlabel("share of catalog (least → most exposed)"); ax.set_ylabel("share of top-10 exposure")
ax.set_title("Lorenz curve: exposure kahan simat raha hai"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""
## Step 4 — Segments: kis tarah ke users ke liye system fail hota hai?

User ki **history length** (test cutoff tak kitne games) ke buckets. Light users ke liye two-tower ke paas kam signal hota hai —
kya ranker / popularity wahan better hai?
""")

code(r"""
BINS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]
hl = N_HIST[c["users"]]
seg_rows = []
for lo, hi in BINS:
    m = (hl >= lo) & (hl <= hi)
    row = {"history_len": f"{lo}-{hi}" if hi < 10**9 else f"{lo}+", "users": int(m.sum()),
           "share_of_users": float(m.mean())}
    for name in LISTS:
        row[f"{name}/recall@100"] = float(PU[name]["recall@100"][m].mean()) if m.any() else np.nan
        row[f"{name}/ndcg@10"] = float(PU[name]["ndcg@10"][m].mean()) if m.any() else np.nan
    seg_rows.append(row)
SEG = pd.DataFrame(seg_rows).set_index("history_len")
display(SEG.round(4))

ax = SEG[[f"{n}/recall@100" for n in LISTS]].rename(columns=lambda s: s.split("/")[0]).plot.bar(figsize=(8, 3.6), rot=0)
ax.set_ylabel("Recall@100"); ax.set_xlabel("user history length"); ax.set_title("Segments: history length vs Recall@100")
ax.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.show()
""")

md(r"""
## Step 5 — Cold-user fallback (bina history wale users)

Ab tak ke saare metrics sirf un users ke the jinki history hai. Test ke baaki users ke liye koi personal signal nahi —
unhe kya dikhayein? Teen non-personal options compare karte hain. **Choice val par** (history = train), **report test par** —
test dekh ke choose karna tuning-on-test hota.
""")

code(r"""
def fallback_eval(hist_df, win_df, t_ref):
    X = build_matrix(hist_df, N_USERS, N_ITEMS)
    no_hist = np.diff(X.indptr) == 0
    w = win_df.filter(pl.col("uid").is_in(np.flatnonzero(no_hist).tolist()))
    users = np.unique(w["uid"].to_numpy())
    if len(users) == 0:
        return 0, {k: {"recall@100": np.nan, "ndcg@10": np.nan} for k in
                   ("all_time_popularity", "recent_popularity_10pct", "last_30_days", "last_90_days")}
    tk = np.sort(w["uid"].to_numpy() * N_ITEMS + w["iid"].to_numpy())
    n_t = np.bincount(w.unique(["uid", "iid"])["uid"].to_numpy(), minlength=N_USERS)[users]
    hi, ht = hist_df["iid"].to_numpy(), hist_df["ts"].to_numpy()
    options = {
        "all_time_popularity": np.bincount(hi, minlength=N_ITEMS),
        "recent_popularity_10pct": recent_popularity(CTX, hist_df),
        "last_30_days": np.bincount(hi[ht >= t_ref - 30 * 86400], minlength=N_ITEMS),
        "last_90_days": np.bincount(hi[ht >= t_ref - 90 * 86400], minlength=N_ITEMS),
    }
    out = {}
    for name, cnt in options.items():
        top = np.argsort(-cnt, kind="stable")[:100]
        hits = np.isin(users[:, None] * N_ITEMS + top[None, :], tk)
        idcg = np.cumsum(DISC)[np.minimum(n_t, 10) - 1]
        out[name] = {"recall@100": float((hits.sum(1) / n_t).mean()), "ndcg@10": float(((hits[:, :10] * DISC).sum(1) / idcg).mean())}
    return len(users), out

n_val_cold, fb_val = fallback_eval(train, val, T_VAL)
n_test_cold, fb_test = fallback_eval(train_val, test, T_TEST)
if n_val_cold > 0:
    FALLBACK = max(fb_val, key=lambda k: fb_val[k]["recall@100"])
else:
    FALLBACK = "recent_popularity_10pct"
    print("NOTE: val mein koi bina-history user nahi -> default fallback")
FB = pd.DataFrame({f"val ({n_val_cold:,} users)": {k: v["recall@100"] for k, v in fb_val.items()},
                   f"test ({n_test_cold:,} users)": {k: v["recall@100"] for k, v in fb_test.items()}})
print(f"chosen on VAL: {FALLBACK}  -> test Recall@100 = {fb_test[FALLBACK]['recall@100']:.4f}, NDCG@10 = {fb_test[FALLBACK]['ndcg@10']:.4f}")
FB.round(4).rename_axis("fallback (Recall@100)")
""")

md(r"""
## Step 6 — Re-rank rules: variety vs accuracy

Ranker ki top-100 se final top-10 banate waqt simple business rules:
- **Publisher cap 3** — ek publisher ke max 3 games (unknown/other publishers par cap nahi)
- **New-item slots 2** — top-10 mein kam se kam 2 aise games jo history mein kabhi nahi aaye (agar top-100 mein hain);
  sabse neeche wale purane games ki jagah aate hain
- Dono saath

Har rule ke liye: accuracy kitni giri, aur coverage / Gini / new-item share / cold recall kitna badla.
""")

code(r"""
def rerank(L, cap=None, min_new=0, k=K_SHOW):
    out = np.full((len(L), k), -1, dtype=np.int64)
    for r, row in enumerate(L):
        row = row[row >= 0]; chosen, cnt = [], {}
        for it in row:
            st = int(STORE[it])
            if cap and st > 1 and cnt.get(st, 0) >= cap:
                continue
            chosen.append(int(it)); cnt[st] = cnt.get(st, 0) + 1
            if len(chosen) == k:
                break
        if min_new:
            need = min_new - int(IS_NEW[chosen].sum()) if chosen else min_new
            extra = [int(it) for it in row if IS_NEW[it] and it not in chosen][:max(need, 0)]
            for e in extra:
                old = [j for j in range(len(chosen) - 1, -1, -1) if not IS_NEW[chosen[j]]]
                if not old:
                    break
                chosen[old[0]] = e
            pos = {int(x): j for j, x in enumerate(row)}
            chosen.sort(key=pos.get)               # final list keeps the ranker's order (same as serving)
        out[r, :len(chosen)] = chosen
    return out

RULES = {"ranker (no re-rank)": dict(), "publisher cap 3": dict(cap=3), "new-item slots 2": dict(min_new=2),
         "cap 3 + new slots 2": dict(cap=3, min_new=2)}
rr_rows = []
for name, kw in RULES.items():
    L10 = rerank(LISTS["ranker"], **kw)
    pu = per_user(np.pad(L10, ((0, 0), (0, 90)), constant_values=-1))
    rr_rows.append({"rule": name, "recall@10": pu["recall@10"].mean(), "ndcg@10": pu["ndcg@10"].mean(),
                    "cold_recall@10": pu["cold_recall@10"].mean(), **beyond(L10)})
RERANK = pd.DataFrame(rr_rows).set_index("rule")
RERANK.round(4)
""")

md("## Step 7 — Save")

code(r"""
RESULTS.mkdir(parents=True, exist_ok=True)
def to_md(df, title):
    df = df.round(4).reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |\n" + "|---" * len(df.columns) + "|\n"
    return f"### {title}\n\n" + head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values) + "\n\n"
report = (to_md(BEYOND, "Accuracy + beyond-accuracy (test, top-10 shown)") + to_md(SEG, "Segments by history length")
          + to_md(FB, f"Cold-user fallback (Recall@100) — chosen on val: {FALLBACK}") + to_md(RERANK, "Re-rank rules on ranker top-100"))
(RESULTS / "beyond_accuracy.md").write_text(report, encoding="utf-8")
(MODELS / "serving_policy.json").write_text(json.dumps({"cold_user_fallback": FALLBACK, "k_show": K_SHOW}, indent=2))
print("saved ->", RESULTS / "beyond_accuracy.md", "|", MODELS / "serving_policy.json")
""")

md(r"""
## Findings (REPORT.md — apne real numbers ke saath)

- **Popularity bias:** recent-pop vs two-tower vs ranker ka Gini@10 / coverage@10 → ranker zyada "popular-heavy" hai ya kam?
- **New items:** ranker ka new-item share vs two-tower → Phase 4 ka cold-recall drop yahan bhi dikhta hai?
- **Segments:** kis history-length bucket mein system sabse kamzor; wahan kaunsa model best (light users ke liye popularity?)
- **Cold users:** fallback = … (val par chuna), test Recall@100 = … — ye users baaki metrics mein nahi the
- **Re-rank:** "new slots 2" se cold recall@10 / coverage kitna badha, NDCG@10 kitna gira → kaunsa rule ship karoge aur kyun
- **Decision for serving (Phase 6):** history wale users → ranker + chosen re-rank; bina history → fallback

**Next:** Phase 6 — serving (FastAPI + Redis + Docker), yahi modules (`retrieval/`, `ranking/`) use karke.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "06_beyond_accuracy.ipynb")
print("ok", len(C), "cells")
