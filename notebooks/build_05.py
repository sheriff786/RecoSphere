"""Builds 05_ranker.ipynb  (edit this file, then: python build_05.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# RankForge · 05 — Candidates + LightGBM Ranker

**Analogy:** naukri ki hiring. **Retrieval** = HR screening — hazaron resumes mein se tezi se 300 shortlist (two-tower, popularity).
**Ranker** = interview panel — sirf un 300 ko dhyan se, zyada jaankari ke saath, sahi order mein lagata hai.

Ranker ke paas wo features hain jo two-tower ke 64 numbers mein nahi aate: *game kitna naya hai, pichhle 30 din mein kitna bika,
user ne is publisher ke kitne games liye, game ka text user ki history se kitna milta hai.* Phase 2 mein **item age tower mein
daalne se nuksaan hua tha** — yahan wahi signal ranker feature ki tarah aata hai.

**LambdaRank** = LightGBM ka objective jo seedha **order** (NDCG) optimize karta hai, har item ka alag score nahi.

### Protocol (leakage yahin sabse aasaan hai)

| | Retrieval model | History (features) | Window (labels) |
|---|---|---|---|
| **Ranker train** | two-tower trained on **train only** | train | **val** |
| **Ranker test** | two-tower trained on **train+val** (notebook 03) | train+val | **test** |

Agar val ke candidates train+val wale model se banate, to model val dekh chuka hota → ranker ko jhootha confidence.

**Saare features cutoff se pehle ke data se.** 2023 snapshot fields (rating_number, Best Sellers Rank, price) kahin nahi.

### Sanity checks (pipeline sahi hai ya nahi)
- Candidates ko sirf two-tower score se order karo → notebook 03 ka test number aana chahiye
- Sirf recent-popularity order → baseline `popularity_recent10` ka number aana chahiye (bahut chhota farak ok hai — barabar
  popularity wale items ka tie-break order alag ho sakta hai)
""")

md("## Step 0 — Config")

code(r"""
import os
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, sys, time
from pathlib import Path

import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import torch

CATEGORY  = os.environ.get("RANKFORGE_CATEGORY", "Video_Games")
DATA_ROOT = Path(os.environ.get("RANKFORGE_DATA", "../data")).resolve()
REPO_ROOT = Path(os.environ.get("RANKFORGE_REPO", "..")).resolve()
GOLD, MODELS, RESULTS = DATA_ROOT / "gold" / CATEGORY, REPO_ROOT / "models" / CATEGORY, REPO_ROOT / "results" / CATEGORY
sys.path.insert(0, str(REPO_ROOT))
from eval.metrics import evaluate
from ranking.pipeline import CAT_FEATS, FEATS, features, gen_candidates, make_context
from retrieval.two_tower import load_two_tower, save_checkpoint, train_two_tower

N_TT     = int(os.environ.get("N_TT", 200))      # two-tower se candidates
N_POP    = int(os.environ.get("N_POP", 100))     # recent popularity se candidates
RECENT_FRAC = 0.1                                # baseline jaisa: history ke last 10% interactions = "recent"
MAX_TRAIN_USERS = int(os.environ.get("MAX_TRAIN_USERS", 20000))   # RAM bachane ke liye
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")     # naye LightGBM ka warning; purane versions ke saath compatible rehne ke liye eval_set hi use kar rahe
print("device:", DEVICE, "| lightgbm", lgb.__version__)
""")

md("## Step 1 — Data + notebook 03 ka model (train+val)")

code(r"""
train = pl.read_parquet(GOLD / "train.parquet"); val = pl.read_parquet(GOLD / "val.parquet")
test  = pl.read_parquet(GOLD / "test.parquet")
train_val = pl.concat([train, val])
N_USERS = pl.read_parquet(GOLD / "user_map.parquet").height
item_feat = pl.read_parquet(GOLD / "item_features.parquet").sort("iid")
N_ITEMS = item_feat.height
T_VAL, T_TEST = int(val["ts"].min()), int(test["ts"].min())

model_tv, ck, dfa_t = load_two_tower(MODELS / "two_tower.pt", GOLD, DEVICE)
CFG, FLAGS, VARIANT = ck["cfg"], ck["flags"], ck["variant"]
TEXT_EMB = np.load(GOLD / f"item_text_emb_{CFG['text_encoder']}.npy")
CTX = make_context(item_feat, TEXT_EMB, dfa_t, N_USERS, CFG["hist_len"], N_TT, N_POP, RECENT_FRAC)   # ranking/pipeline.py
print(f"variant={VARIANT} | users={N_USERS:,} items={N_ITEMS:,} | text emb {TEXT_EMB.shape}")
""")

md(r"""
## Step 2 — Train-only two-tower (ranker training ke candidates ke liye)

Notebook 03 ke val tuning ka **best epoch** use karte hain (`results/.../two_tower.csv`), same config, sirf **train** par.
Ek baar train hoke `two_tower_trainonly.pt` mein cache — dobara chalane par skip.
""")

code(r"""
TRAINONLY = MODELS / "two_tower_trainonly.pt"
tt_csv = RESULTS / "two_tower.csv"
best_ep = int(os.environ.get("TT_EPOCHS", 0)) or int(
    pl.read_csv(tt_csv).filter(pl.col("model") == f"two_tower_{VARIANT}")["best_epoch"][0])
if TRAINONLY.exists():
    model_tr, ck_tr, _ = load_two_tower(TRAINONLY, GOLD, DEVICE)
    assert ck_tr["variant"] == VARIANT and ck_tr["t_ref"] == T_VAL, "stale train-only model — file delete karke dobara chalao"
    print("loaded cache:", TRAINONLY.name)
else:
    print(f"training {VARIANT} on TRAIN only for {best_ep} epochs ...")
    t0 = time.time()
    model_tr = train_two_tower(FLAGS, CFG, train, best_ep, item_feat, TEXT_EMB, dfa_t, DEVICE)
    save_checkpoint(model_tr, TRAINONLY, VARIANT, FLAGS, CFG, N_USERS, T_VAL)
    print(f"done in {time.time() - t0:.0f}s -> {TRAINONLY.name}")
""")

md(r"""
## Step 3 — Candidate generation

Har user ke liye (window mein jiske naye targets hain aur history hai):
- **two-tower top-200** (unseen items, exact search — notebook 04 ne dikhaya L1 par Flat hi best hai)
- **recent popularity top-100** (unseen) — warm items par sabse strong baseline
- **union** (duplicates hata ke) → ~200-300 candidates

**Candidate recall** = window ke asli targets mein se kitne candidates mein aaye. Ye ranker ki **ceiling** hai —
ranker sirf order badal sakta hai, naye items nahi la sakta.
""")

code(r"""
# implementation: ranking/pipeline.py -> gen_candidates (notebook 06 aur serving bhi wahi use karte hain)
t0 = time.time()
cand_val = gen_candidates(CTX, model_tr, train, val, T_VAL)
print(f"VAL : users={len(cand_val['users']):,} rows={len(cand_val['u']):,} ({time.time() - t0:.0f}s) candidate recall:",
      {k: round(v, 4) for k, v in cand_val["recall"].items()})
t0 = time.time()
cand_test = gen_candidates(CTX, model_tv, train_val, test, T_TEST)
print(f"TEST: users={len(cand_test['users']):,} rows={len(cand_test['u']):,} ({time.time() - t0:.0f}s) candidate recall:",
      {k: round(v, 4) for k, v in cand_test["recall"].items()})
""")

md(r"""
## Step 4 — Features (sab **cutoff time `t_ref` se pehle** ke data se)

| Group | Features |
|---|---|
| Retrieval | `tt_score`, `tt_rank`, `pop_rank`, `in_tt`, `in_pop` |
| Item (history se) | `i_pop_all`, `i_pop_30d`, `i_pop_90d`, `i_pop_365d`, `i_trend` (30d / 365d), `i_age_days`, `i_dfa_missing`, `store`, `type`, `rated` |
| User (history se) | `u_n_hist`, `u_days_since_last`, `u_span_days`, `u_mean_rating` |
| User × item | `ui_store_share` (user ki history mein is publisher ka share), `ui_text_sim_mean` / `ui_text_sim_last` (game ka text vs user ke last-20 / last item) |

`i_age_days` 0 par clip — test window mein release hue games ko "brand new" (static-eval caveat, notebook 03 jaisa).
`rating` sirf history ke reviews se (user ka average) — item ki 2023 wali average rating **nahi**.
""")

code(r"""
# implementation: ranking/pipeline.py -> features (FEATS list wahin hai)
t0 = time.time()
F_val = features(CTX, cand_val, train)
F_test = features(CTX, cand_test, train_val)
print(f"features: val {F_val.shape}, test {F_test.shape} ({time.time() - t0:.0f}s)")
F_val.describe().T[["mean", "min", "max"]].round(3)
""")

md(r"""
## Step 5 — Ranker training (val window par)

- Sirf wo users jinke candidates mein **kam se kam ek** positive hai (bina positive ke list se LambdaRank kuch nahi seekhta)
- RAM ke liye max `MAX_TRAIN_USERS` users; unme se 15% ranker ki apni early-stopping ke liye
- Rows user-wise grouped hain (`group` = har user ki list ki length)
""")

code(r"""
def group_sizes(u):
    _, counts = np.unique(u, return_counts=True)       # rows already user-sorted
    return counts

pos_users = np.unique(cand_val["u"][cand_val["label"] == 1])
chosen = rng.permutation(pos_users)[:MAX_TRAIN_USERS]
n_es = max(1, int(0.15 * len(chosen)))
es_users, fit_users = np.sort(chosen[:n_es]), np.sort(chosen[n_es:])
m_fit, m_es = np.isin(cand_val["u"], fit_users), np.isin(cand_val["u"], es_users)
print(f"val users with >=1 positive in candidates: {len(pos_users):,} | fit {len(fit_users):,} | early-stop {len(es_users):,}")
print(f"positive rate in fit rows: {cand_val['label'][m_fit].mean():.4f}")

ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=1000, learning_rate=0.05, num_leaves=63,
                        min_child_samples=50, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                        random_state=SEED, verbose=-1)
t0 = time.time()
ranker.fit(F_val[m_fit], cand_val["label"][m_fit], group=group_sizes(cand_val["u"][m_fit]),
           eval_set=[(F_val[m_es], cand_val["label"][m_es])], eval_group=[group_sizes(cand_val["u"][m_es])],
           eval_at=[10], categorical_feature=CAT_FEATS,
           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
print(f"trained in {time.time() - t0:.0f}s | best iteration {ranker.best_iteration_}")
""")

md(r"""
## Step 6 — Test evaluation (ek baar)

Same `evaluate()`, same test users, same masking. Chaar orderings same candidates par:
1. **two-tower order** → sanity: notebook 03 ka number
2. **recent popularity order** → sanity: baseline ka number
3. **RRF** (Reciprocal Rank Fusion: `1/(60+rank_tt) + 1/(60+rank_pop)`) — bina training ka simple blend
4. **LightGBM ranker**
""")

code(r"""
SHOW = ["all/recall@50", "all/recall@100", "all/ndcg@10", "warm/recall@100", "cold/recall@100"]
known_tv = train_val["iid"].unique().to_numpy()

def eval_scores(c, scores):
    starts = np.searchsorted(c["u"], c["users"]); ends = np.append(starts[1:], len(c["u"]))
    where = dict(zip(c["users"].tolist(), zip(starts.tolist(), ends.tolist())))
    def score_fn(ub):
        S = np.full((len(ub), N_ITEMS), -np.inf, dtype=np.float32)
        for r, u in enumerate(ub):
            if u in where:
                s0, s1 = where[u]; S[r, c["i"][s0:s1]] = scores[s0:s1]
        return S
    return evaluate(score_fn, c["X"], test, known_tv, ks=(10, 50, 100))

rank_or = lambda x: np.where(np.isnan(x), 0.0, 1.0 / (60.0 + np.nan_to_num(x)))
orders = {
    "two_tower order (sanity)":  cand_test["tt_score"],
    "recent_pop order (sanity)": np.where(np.isnan(cand_test["pop_rank"]), -1e9, -np.nan_to_num(cand_test["pop_rank"])),
    "rrf (two_tower + pop)":     (rank_or(cand_test["tt_rank"]) + rank_or(cand_test["pop_rank"])).astype(np.float32),
    "lightgbm ranker":           ranker.predict(F_test, num_iteration=ranker.best_iteration_).astype(np.float32),
}
res = []
for name, s in orders.items():
    r = eval_scores(cand_test, s)
    res.append({"model": name, **{k: r[k] for k in SHOW}})
    print(f"{name:<28}" + "  ".join(f"{k}={r[k]:.4f}" for k in SHOW))

prev = {}
for f, col in [(RESULTS / "baselines.csv", "popularity_recent10"), (RESULTS / "two_tower.csv", f"two_tower_{VARIANT}")]:
    if f.exists():
        row = pl.read_csv(f).filter(pl.col("model") == col)
        if row.height:
            prev[col] = round(row["all/recall@100"][0], 4)
print("\nearlier notebooks (all/recall@100):", prev)
print("candidate recall (ceiling) on test:", {k: round(v, 4) for k, v in cand_test["recall"].items()})
""")

code(r"""
import matplotlib.pyplot as plt
imp = pd.Series(ranker.booster_.feature_importance("gain"), index=FEATS)
imp = (imp / imp.sum()).sort_values()
ax = imp.plot.barh(figsize=(7, 6), color="#4C78A8"); ax.set_xlabel("share of total gain"); ax.set_title("Ranker feature importance")
plt.tight_layout(); plt.show()
imp.sort_values(ascending=False).round(4).head(10)
""")

md("## Step 7 — Save (Phase 6 serving yahi files load karega)")

code(r"""
RESULTS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
table = pl.DataFrame(res)
table.write_csv(RESULTS / "ranker.csv")
lines = ["| model | " + " | ".join(SHOW) + " |", "|---" * (len(SHOW) + 1) + "|"]
lines += [f"| {r['model']} | " + " | ".join(f"{r[k]:.4f}" for k in SHOW) + " |" for r in res]
cr = cand_test["recall"]
(RESULTS / "ranker.md").write_text(
    f"Candidates: two-tower top-{N_TT} + recent-pop top-{N_POP}. Candidate recall (test): "
    f"two_tower={cr['two_tower']:.4f}, recent_pop={cr['recent_pop']:.4f}, union={cr['union']:.4f}\n\n" + "\n".join(lines) + "\n",
    encoding="utf-8")
ranker.booster_.save_model(str(MODELS / "ranker.txt"), num_iteration=ranker.best_iteration_)
(MODELS / "ranker_meta.json").write_text(json.dumps({
    "features": FEATS, "categorical": CAT_FEATS, "n_tt": N_TT, "n_pop": N_POP, "recent_frac": RECENT_FRAC,
    "best_iteration": int(ranker.best_iteration_), "retrieval_variant": VARIANT,
    "params": {k: v for k, v in ranker.get_params().items() if isinstance(v, (int, float, str, type(None)))}}, indent=2))
with pl.Config(tbl_rows=10, float_precision=4, tbl_width_chars=140):
    print(table)
print("saved ->", RESULTS / "ranker.md", "|", MODELS / "ranker.txt")
""")

md(r"""
## Findings (REPORT.md — apne real numbers ke saath)

- **Sanity:** two-tower order = notebook 03 (…)? recent-pop order = baseline (…)?
- **Candidate recall (ceiling):** two-tower … / recent-pop … / union … → union ne kitna jodaa
- **Ranker vs best single source:** NDCG@10 … vs … ; Recall@100 … vs …
- **RRF vs ranker:** learned ranker ne simple blend se kitna jodaa (agar kam, to features/labels mein kya kami?)
- **Warm vs cold:** ranker ne kis slice mein jeeta / haara
- **Top features:** … (recency features upar hain? Phase 2 age-in-tower failure ke saath connect karo)
- **Caveats:** ranker val window (19 mahine) par train, test (26 mahine) par eval; static eval; ek seed

**Next:** Phase 5 — beyond accuracy (coverage, popularity Gini, heavy vs light users, cold-user fallback).
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "05_ranker.ipynb")
print("ok", len(C), "cells")
