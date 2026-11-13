"""Builds 03_two_tower.ipynb  (edit this file, then: python build_03.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# RankForge · 03 — Two-Tower Retrieval

**Goal:** ek model jo har user aur har item ko 64 numbers ke vector mein badle, taaki dot product se "user ko ye item pasand aayega" ka score mile.

Design pichle findings se aaya hai:

| Finding (notebook 02 + baselines) | Design decision |
|---|---|
| Test ke 46.2% interactions **cold items** par (history mein kabhi nahi aaye) | Item tower mein **text** (title, categories, features) — ID ke bina bhi item ko vector milta hai |
| ID embedding text ko "dabaa" deti hai | Training mein **ID dropout**; eval mein cold items ki ID embedding **zero** |
| Sabse strong baseline = **recent popularity** (test Recall@100 = 0.0916) | Item **age** feature (`Date First Available` se) |
| Metadata 2023 ka snapshot hai | Best Sellers Rank, rating_number, price **nahi** (leakage) |

**Protocol (baselines jaisa hi):** train par fit → val par early stopping → train+val par same epochs se refit → test par **ek baar** eval.
Metrics wahi `eval/metrics.py` se, taaki baselines se seedha comparable hon.

**Ablations** (interview ka sabse strong hissa): `id_only` · `content_only` · `id_content` · `id_content_age`
""")

md("## Step 0 — Config\n\nIs notebook mein Spark nahi hai (sirf Polars + PyTorch), isliye Java/Hadoop setup ki zaroorat nahi.")

code(r"""
import json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

CATEGORY  = os.environ.get("RANKFORGE_CATEGORY", "Video_Games")
DATA_ROOT = Path(os.environ.get("RANKFORGE_DATA", "../data")).resolve()
REPO_ROOT = Path(os.environ.get("RANKFORGE_REPO", "..")).resolve()
GOLD        = DATA_ROOT / "gold" / CATEGORY
SILVER_PROD = DATA_ROOT / "silver" / "products" / f"category={CATEGORY}"
OUT_MODELS  = REPO_ROOT / "models" / CATEGORY
OUT_RESULTS = REPO_ROOT / "results" / CATEGORY
sys.path.insert(0, str(REPO_ROOT))
from eval.metrics import build_matrix, evaluate          # same metric code as baselines

CFG = {
    "seed": 42,
    "text_encoder": os.environ.get("TEXT_ENCODER", "minilm"),   # "minilm" (default) | "tfidf" (no download, weaker)
    "minilm_model": "sentence-transformers/all-MiniLM-L6-v2",
    "desc_chars": 200,          # description ke pehle N characters hi text mein
    "min_cat_count": 5,         # isse kam items wali store/type/rating value -> "other"
    "hist_len": 20,             # user tower: last 20 items
    "dim": 64,                  # final vector size
    "tau": 0.07,                # temperature: chhota = sharper softmax
    "batch_size": 2048,
    "lr": 2e-3,
    "weight_decay": 1e-5,
    "id_dropout": 0.5,          # training mein 50% baar item ID embedding zero
    "max_epochs": int(os.environ.get("MAX_EPOCHS", 15)),
    "min_epochs": 5,            # shuru ke epochs mein loss plateau hota hai -> itne epochs se pehle stop nahi
    "patience": 3,              # val Recall@100 itne epochs tak na badhe -> stop
    "ks": (10, 50, 100),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 4)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

set_seed(CFG["seed"])
print("device:", DEVICE, "| gold:", GOLD)
""")

md(r"""
## Task 0 — Gold load + checks

Split ka time order dobara check karte hain, aur dekhte hain ki cutoffs exact quantile se match karte hain ya nahi
(notebook 02 mein `QUANTILE_REL_ERR = 0.0` laga ke re-run kiya ho to match karna chahiye).
""")

code(r"""
train = pl.read_parquet(GOLD / "train.parquet")
val   = pl.read_parquet(GOLD / "val.parquet")
test  = pl.read_parquet(GOLD / "test.parquet")
user_map = pl.read_parquet(GOLD / "user_map.parquet")
item_map = pl.read_parquet(GOLD / "item_map.parquet").sort("iid")
N_USERS, N_ITEMS = user_map.height, item_map.height
train_val = pl.concat([train, val])

assert train["ts"].max() < val["ts"].min() and val["ts"].max() < test["ts"].min(), "split time order broken"
assert (item_map["iid"].to_numpy() == np.arange(N_ITEMS)).all(), "item_map iid must be 0..N-1"
T_VAL, T_TEST = int(val["ts"].min()), int(test["ts"].min())

all_ts = pl.concat([train, val, test])["ts"]
exact = [int(all_ts.quantile(q, interpolation="lower")) for q in (0.8, 0.9)]
print(f"users={N_USERS:,} items={N_ITEMS:,} | train={train.height:,} val={val.height:,} test={test.height:,}")
print(f"val starts at {T_VAL} (exact quantile {exact[0]}), test starts at {T_TEST} (exact quantile {exact[1]})")
if [T_VAL, T_TEST] != exact:
    print("NOTE: cutoffs exact quantile se match nahi karte -> notebook 02 mein QUANTILE_REL_ERR = 0.0 karke re-run karo")
""")

md(r"""
## Task 1 — Item features table

Har item (`iid`) ke liye:
- `text` = title | categories | features | description ke pehle 200 chars
- `store_id`, `type_id`, `rated_id` — categorical ids. `0` = unknown (missing), `1` = other (rare value)
- `dfa_ts` = **Date First Available** (seconds) — item age ke liye. Ye listing ke waqt tay hota hai, isliye leakage nahi

**Jo nahi liya (snapshot leakage):** Best Sellers Rank, rating_number, average_rating, price, Is Discontinued — ye sab 2023 ki state hai.
""")

code(r"""
prod = pl.read_parquet(SILVER_PROD.as_posix() + "/*.parquet")
jget = lambda key: pl.col("details").str.json_path_match(f"$['{key}']")
lst  = lambda c, sep: pl.col(c).list.join(sep).fill_null("")

items = (item_map.join(prod, on="item_id", how="left")
    .with_columns(
        jget("Date First Available").alias("dfa_raw"),
        jget("Type of item").alias("type_raw"),
        jget("Rated").alias("rated_raw"),
        pl.col("store").str.strip_chars().replace("", None).alias("store_raw"),
    )
    .with_columns(
        pl.col("dfa_raw").str.strptime(pl.Date, "%B %d, %Y", strict=False).alias("dfa"),
        pl.concat_str([pl.col("title").fill_null(""), lst("categories", " > "), lst("features", ". "),
                       lst("description", " ").str.slice(0, CFG["desc_chars"])], separator=" | ").alias("text"),
    ))

def encode(col, min_count):
    vc = items.group_by(col).len().filter(pl.col(col).is_not_null() & (pl.col("len") >= min_count))
    vocab = {v: i + 2 for i, v in enumerate(sorted(vc[col].to_list()))}       # 0 unknown, 1 other
    ids = (pl.when(pl.col(col).is_null()).then(0)
             .otherwise(pl.col(col).replace_strict(vocab, default=1, return_dtype=pl.Int64)))
    return ids, len(vocab) + 2

store_ids, N_STORE = encode("store_raw", CFG["min_cat_count"])
type_ids,  N_TYPE  = encode("type_raw",  CFG["min_cat_count"])
rated_ids, N_RATED = encode("rated_raw", CFG["min_cat_count"])

item_feat = items.select(
    "iid", "item_id", "title", "text",
    store_ids.alias("store_id"), type_ids.alias("type_id"), rated_ids.alias("rated_id"),
    pl.col("dfa").cast(pl.Datetime("ms")).dt.epoch("s").alias("dfa_ts"),
).sort("iid")
assert item_feat.height == N_ITEMS and (item_feat["iid"].to_numpy() == np.arange(N_ITEMS)).all()
item_feat.write_parquet(GOLD / "item_features.parquet")

print(f"vocab sizes: store={N_STORE}, type={N_TYPE}, rated={N_RATED}")
print(item_feat.select(
    (pl.col("title").fill_null("").str.strip_chars() == "").mean().alias("title_blank"),
    pl.col("dfa_ts").is_not_null().mean().alias("dfa_parsed"),
    (pl.col("store_id") == 0).mean().alias("store_unknown"),
    (pl.col("type_id") == 0).mean().alias("type_unknown"),
    (pl.col("rated_id") == 0).mean().alias("rated_unknown"),
))
""")

code(r"""
# DFA parse failures: agar dfa_parsed kam aaya, ye rows dekh ke date format adjust karo
print(items.filter(pl.col("dfa_raw").is_not_null() & pl.col("dfa").is_null()).select("dfa_raw").head(5))

# Cold vs warm test items ka metadata quality (text tower cold items ke liye kaam karega ya nahi?)
cold_iids = test.join(train_val.select("iid").unique(), on="iid", how="anti")["iid"].unique()
warm_iids = test.join(pl.DataFrame({"iid": cold_iids}), on="iid", how="anti")["iid"].unique()
def quality(ids, label):
    return items.filter(pl.col("iid").is_in(ids.implode())).select(
        pl.lit(label).alias("group"), pl.len().alias("items"),
        (pl.col("features").list.len().fill_null(0) == 0).mean().alias("features_empty"),
        (pl.col("description").list.len().fill_null(0) == 0).mean().alias("description_empty"),
        pl.col("dfa").is_not_null().mean().alias("dfa_present"),
        pl.col("text").str.len_chars().median().alias("median_text_chars"),
    )
pl.concat([quality(item_feat["iid"], "all items"), quality(warm_iids, "warm test items"),
           quality(cold_iids, "cold test items")])
""")

md(r"""
## Task 2 — Text embeddings (ek baar compute, phir cache)

Default: **all-MiniLM-L6-v2** (384-d, CPU par chal jaata hai). Pehli baar model download hoga (~90 MB).
Transformer ko fine-tune nahi karte — frozen embeddings + ek chhoti Linear layer model ke andar.

`TEXT_ENCODER=tfidf` = fallback (TF-IDF + SVD, koi download nahi). Kamzor hai; sirf tab use karo jab MiniLM download na ho paaye,
aur REPORT mein likho kaunsa use hua.

**Sanity check:** famous items ke text-nearest-neighbours. Mario ke paas Mario/Nintendo aaye to embeddings theek hain.
""")

code(r"""
texts = item_feat["text"].to_list()
EMB_PATH = GOLD / f"item_text_emb_{CFG['text_encoder']}.npy"

if EMB_PATH.exists() and np.load(EMB_PATH, mmap_mode="r").shape[0] == N_ITEMS:
    TEXT_EMB = np.load(EMB_PATH)
    print("loaded cache:", EMB_PATH.name)
elif CFG["text_encoder"] == "minilm":
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(CFG["minilm_model"], device=DEVICE)
    t0 = time.time()
    TEXT_EMB = enc.encode(texts, batch_size=256, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    print(f"encoded {N_ITEMS:,} items in {time.time() - t0:.0f}s")
    np.save(EMB_PATH, TEXT_EMB)
elif CFG["text_encoder"] == "tfidf":
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(min_df=2, max_features=50_000, ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    TEXT_EMB = TruncatedSVD(n_components=min(256, X.shape[1] - 1), random_state=CFG["seed"]).fit_transform(X)
    TEXT_EMB = (TEXT_EMB / (np.linalg.norm(TEXT_EMB, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    np.save(EMB_PATH, TEXT_EMB)
else:
    raise ValueError(CFG["text_encoder"])
print("text embedding shape:", TEXT_EMB.shape)
""")

code(r"""
train_pop = np.bincount(train["iid"].to_numpy(), minlength=N_ITEMS)
titles = item_feat["title"].fill_null("").to_list()

def neighbours(query, k=5):
    hits = item_feat.filter(pl.col("title").str.to_lowercase().str.contains(query.lower(), literal=True))["iid"].to_numpy()
    if len(hits) == 0:
        print(f"[{query}] no title match"); return
    i = hits[np.argmax(train_pop[hits])]                       # sabse popular matching item
    sims = TEXT_EMB @ TEXT_EMB[i]
    top = [j for j in np.argsort(-sims) if j != i][:k]
    print(f"\n[{query}] {titles[i][:80]}")
    for j in top:
        print(f"   {sims[j]:.3f}  {titles[j][:80]}")

for q in ["mario", "fifa", "controller", "pokemon"]:
    neighbours(q)
""")

md(r"""
## Task 3 — Training examples (leakage yahin hoti hai)

Har user ka data `ts` se sort. Har interaction *j* ke liye:
- **history** = usi user ke usse **pehle** ke last 20 items (`ts` strictly chhota — same-second wale bhi nahi)
- **target** = item *j*, **query time** = `ts_j`

Dry run (ek user, L=3): items `A(t1) B(t2) C(t3) D(t4)` →
`B: [_, _, A]` · `C: [_, A, B]` · `D: [A, B, C]` · `A` skip (koi history nahi).
Slot ka matlab fixed hai: **aakhri slot = sabse recent** item. Eval mein bhi yahi format.

Vectorized trick: rows ko (uid, ts) se sort karo; slot `L−k` mein row `r−k` ka item jaata hai — agar wo same user ka hai aur time mein strictly pehle.
""")

code(r"""
PAD = -1
L = CFG["hist_len"]

def build_examples(df):
    df = df.sort(["uid", "ts", "iid"])
    u, i, t = df["uid"].to_numpy(), df["iid"].to_numpy(), df["ts"].to_numpy()
    n = len(df)
    hist = np.full((n, L), PAD, dtype=np.int64)
    hist_ts = np.full((n, L), -1, dtype=np.int64)
    for k in range(1, L + 1):
        src = np.arange(n) - k
        s = np.clip(src, 0, None)
        ok = (src >= 0) & (u[s] == u) & (t[s] < t)
        hist[ok, L - k] = i[s[ok]]
        hist_ts[ok, L - k] = t[s[ok]]
    keep = (hist != PAD).any(axis=1)
    ex = {"hist": hist[keep], "hist_ts": hist_ts[keep], "target": i[keep], "t": t[keep], "uid": u[keep]}
    # LEAKAGE ASSERT: har history item apne target se strictly pehle
    valid = ex["hist"] != PAD
    assert (np.where(valid, ex["hist_ts"], -1) < ex["t"][:, None]).all(), "history contains future items!"
    return ex

def history_at_cutoff(df):
    # eval ke liye: har user ke last L items (cutoff se pehle), aakhri slot = sabse recent
    H = np.full((N_USERS, L), PAD, dtype=np.int64)
    g = df.sort(["uid", "ts", "iid"]).group_by("uid", maintain_order=True).agg(pl.col("iid").tail(L))
    for uid, items_ in zip(g["uid"].to_list(), g["iid"].to_list()):
        H[uid, L - len(items_):] = items_
    return H

ex_check = build_examples(train)
print(f"train examples: {len(ex_check['target']):,} (dropped {train.height - len(ex_check['target']):,} first-interactions)")
print("history length distribution:", np.bincount((ex_check["hist"] != PAD).sum(1), minlength=L + 1)[1:].tolist())
del ex_check
""")

md(r"""
## Task 4 — Model

**Item tower:** `[text → Linear(128)] + [ID emb(64) × keep] + [store, type, rated embs] + [age emb]` → MLP → 64-d → L2 normalize
- `keep` = 0 agar item fit-history mein kabhi nahi aaya (**cold → ID zero**) ya training mein ID dropout laga
- Age bucket = `1 + min(log2(age_days + 1), 15)`; `17` = DFA missing; `0` = "history slot" (history items ko age nahi chahiye)

**User tower:** history items ke item-tower vectors (**shared weights**) → learned position weights (recent ko zyada) → weighted mean → residual MLP → normalize.
User ID embedding nahi — naya user 1-2 items ke baad hi represent ho jaata hai.

**Loss — in-batch softmax + logQ:** batch ke baaki rows ke targets = negatives. Popular items zyada baar negative bante hain, isliye
`logit − log(q_item)` se correct karte hain. Same item kisi aur row ka bhi target ho to use negative nahi ginte (mask).
""")

code(r"""
DFA_T = torch.tensor(item_feat["dfa_ts"].fill_null(-1).to_numpy(), dtype=torch.float64, device=DEVICE)

def age_bucket(iids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dfa = DFA_T[iids]
    days = ((t.double() - dfa).clamp(min=0) / 86400.0)
    b = 1 + torch.clamp(torch.floor(torch.log2(days + 1)), max=15).long()
    return torch.where(dfa < 0, torch.full_like(b, 17), b)

class TwoTower(nn.Module):
    def __init__(self, use_id, use_content, use_age, known_items, cfg):
        super().__init__()
        self.use_id, self.use_content, self.use_age = use_id, use_content, use_age
        self.id_dropout = cfg["id_dropout"] if (use_id and use_content) else 0.0   # ID-only model mein dropout ka matlab nahi
        known = torch.zeros(N_ITEMS, dtype=torch.bool); known[torch.as_tensor(known_items)] = True
        self.register_buffer("known", known)
        in_dim = 0
        if use_content:
            self.register_buffer("text", torch.tensor(TEXT_EMB))
            self.register_buffer("store", torch.tensor(item_feat["store_id"].to_numpy()))
            self.register_buffer("type_", torch.tensor(item_feat["type_id"].to_numpy()))
            self.register_buffer("rated", torch.tensor(item_feat["rated_id"].to_numpy()))
            self.text_proj = nn.Linear(TEXT_EMB.shape[1], 128)
            self.store_emb, self.type_emb, self.rated_emb = nn.Embedding(N_STORE, 16), nn.Embedding(N_TYPE, 8), nn.Embedding(N_RATED, 8)
            in_dim += 128 + 16 + 8 + 8
        if use_id:
            self.id_emb = nn.Embedding(N_ITEMS, 64)
            nn.init.normal_(self.id_emb.weight, std=0.05)
            in_dim += 64
        if use_age:
            self.age_emb = nn.Embedding(18, 16)
            in_dim += 16
        d = cfg["dim"]
        self.item_mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, d))
        self.pos_w = nn.Parameter(torch.zeros(cfg["hist_len"]))
        self.user_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def item_vec(self, iids, age_b=None):
        parts = []
        if self.use_content:
            parts += [self.text_proj(self.text[iids]), self.store_emb(self.store[iids]),
                      self.type_emb(self.type_[iids]), self.rated_emb(self.rated[iids])]
        if self.use_id:
            keep = self.known[iids]                                         # cold item -> ID zero
            if self.training and self.id_dropout > 0:
                keep = keep & (torch.rand(iids.shape, device=iids.device) >= self.id_dropout)
            parts.append(self.id_emb(iids) * keep.unsqueeze(-1).float())
        if self.use_age:
            parts.append(self.age_emb(age_b if age_b is not None else torch.zeros_like(iids)))
        return F.normalize(self.item_mlp(torch.cat(parts, dim=-1)), dim=-1)

    def user_vec(self, hist):
        mask = hist >= 0
        B, Lh = hist.shape
        v = self.item_vec(hist.clamp(min=0).reshape(-1)).reshape(B, Lh, -1)
        w = self.pos_w.expand(B, Lh).masked_fill(~mask, float("-inf")).softmax(dim=-1)
        w = torch.nan_to_num(w, nan=0.0)                                    # no-history user -> zero vector
        pooled = (w.unsqueeze(-1) * v).sum(dim=1)
        return F.normalize(pooled + self.user_mlp(pooled), dim=-1)

def in_batch_loss(model, hist, target, t, log_q, tau):
    u = model.user_vec(hist)
    v = model.item_vec(target, age_bucket(target, t) if model.use_age else None)
    logits = (u @ v.T) / tau - log_q[target][None, :]
    same = (target[:, None] == target[None, :]) & ~torch.eye(len(target), dtype=torch.bool, device=target.device)
    logits = logits.masked_fill(same, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(target), device=target.device))
""")

md(r"""
## Task 5 — Training + eval functions

- `fit_df` ke items = "known" (inke ID embeddings train hote hain); baaki sab cold → ID zero
- Eval: saare items ke vectors **eval start time** par (age wahan se), user vectors cutoff-history se, phir wahi `evaluate()` jo baselines ne use kiya
- Early stopping: val `all/recall@100`, lekin `min_epochs` (5) se pehle nahi — content models pehle 2-3 epochs plateau mein rehte hain, phir seekhte hain
- Refit mein **exactly best epochs** (test kabhi tuning mein nahi)
""")

code(r"""
@torch.no_grad()
def embed_all(model, hist_df, t_eval, chunk=8192):
    model.eval()
    iids = torch.arange(N_ITEMS, device=DEVICE)
    t = torch.full((N_ITEMS,), t_eval, dtype=torch.long, device=DEVICE)
    V = torch.cat([model.item_vec(iids[s:s + chunk], age_bucket(iids[s:s + chunk], t[s:s + chunk]) if model.use_age else None)
                   for s in range(0, N_ITEMS, chunk)])
    H = torch.tensor(history_at_cutoff(hist_df), device=DEVICE)
    U = torch.cat([model.user_vec(H[s:s + chunk]) for s in range(0, N_USERS, chunk)])
    return U.cpu().numpy(), V.cpu().numpy()

def run_eval(model, hist_df, eval_df, t_eval):
    U, V = embed_all(model, hist_df, t_eval)
    X = build_matrix(hist_df, N_USERS, N_ITEMS)
    return evaluate(lambda ub: U[ub] @ V.T, X, eval_df, hist_df["iid"].unique().to_numpy(), ks=CFG["ks"])

def train_model(flags, fit_df, n_epochs=None, val_df=None, t_val=None, label=""):
    set_seed(CFG["seed"])
    ex = build_examples(fit_df)
    cnt = np.bincount(ex["target"], minlength=N_ITEMS) + 1.0
    log_q = torch.tensor(np.log(cnt / cnt.sum()), dtype=torch.float32, device=DEVICE)
    tens = {k: torch.tensor(v, device=DEVICE) for k, v in ex.items() if k in ("hist", "target", "t")}
    model = TwoTower(**flags, known_items=fit_df["iid"].unique().to_numpy(), cfg=CFG).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n, bs = len(ex["target"]), CFG["batch_size"]
    epochs = n_epochs or CFG["max_epochs"]
    log, best, best_ep, best_state, bad = [], -1.0, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            loss = in_batch_loss(model, tens["hist"][b], tens["target"][b], tens["t"][b], log_q, CFG["tau"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        row = {"epoch": ep, "loss": tot / n, "train_s": round(time.time() - t0, 1)}
        if val_df is not None:
            r = run_eval(model, fit_df, val_df, t_val)
            row.update({"val_recall@100": r["all/recall@100"], "val_ndcg@10": r["all/ndcg@10"],
                        "val_cold_recall@100": r["cold/recall@100"]})
            if r["all/recall@100"] > best:
                best, best_ep, bad = r["all/recall@100"], ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        log.append(row)
        print(f"[{label}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val_df is not None and ep >= CFG["min_epochs"] and bad >= CFG["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log, (best_ep if val_df is not None else epochs)
""")

md(r"""
## Task 6 — Ablations: val par tune, train+val par refit, test par ek baar

Chaar variants, har ek ka same protocol. CPU par har variant kuch minute leta hai — sirf main model chalana ho to
`RUN_VARIANTS = ["id_content_age"]` kar do.
""")

code(r"""
VARIANTS = {
    "id_only":        dict(use_id=True,  use_content=False, use_age=False),
    "content_only":   dict(use_id=False, use_content=True,  use_age=False),
    "id_content":     dict(use_id=True,  use_content=True,  use_age=False),
    "id_content_age": dict(use_id=True,  use_content=True,  use_age=True),
}
RUN_VARIANTS = os.environ.get("RUN_VARIANTS", ",".join(VARIANTS)).split(",")

results, logs, final_models = [], {}, {}
for name in RUN_VARIANTS:
    flags = VARIANTS[name]
    print(f"\n===== {name}: tune on VAL =====")
    _, log, best_ep = train_model(flags, train, val_df=val, t_val=T_VAL, label=f"{name}/val")
    logs[name] = log
    best_row = max(log, key=lambda r: r["val_recall@100"])
    print(f"===== {name}: refit on TRAIN+VAL for {best_ep} epochs, eval TEST once =====")
    model, _, _ = train_model(flags, train_val, n_epochs=best_ep, label=f"{name}/refit")
    r = run_eval(model, train_val, test, T_TEST)
    r.update(model=f"two_tower_{name}", best_epoch=best_ep, val_recall_at_100=best_row["val_recall@100"])
    results.append(r)
    final_models[name] = model
    print(f"TEST  all/recall@100={r['all/recall@100']:.4f}  all/ndcg@10={r['all/ndcg@10']:.4f}  "
          f"warm/recall@100={r['warm/recall@100']:.4f}  cold/recall@100={r['cold/recall@100']:.4f}")
""")

code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for name, log in logs.items():
    ep = [r["epoch"] for r in log]
    ax[0].plot(ep, [r["val_recall@100"] for r in log], marker="o", label=name)
    ax[1].plot(ep, [r["val_cold_recall@100"] for r in log], marker="o", label=name)
ax[0].set_title("val Recall@100 (all)"); ax[1].set_title("val Recall@100 (cold items)")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
## Results table — baselines ke saath ek jagah

Baselines `results/<category>/baselines.csv` se (agar `run_baselines.py` chala hai). Same test set, same metric code.
""")

code(r"""
SHOW = ["all/recall@50", "all/recall@100", "all/ndcg@10", "warm/recall@100", "cold/recall@100"]
rows = []
base_csv = OUT_RESULTS / "baselines.csv"
if base_csv.exists():
    rows += pl.read_csv(base_csv).select(["model"] + SHOW).to_dicts()
rows += [{"model": r["model"], **{k: r[k] for k in SHOW}} for r in results]
table = pl.DataFrame(rows)
with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=140):
    print(table)

OUT_RESULTS.mkdir(parents=True, exist_ok=True)
pl.DataFrame(results).write_csv(OUT_RESULTS / "two_tower.csv")
md_lines = ["| model | " + " | ".join(SHOW) + " |", "|---" * (len(SHOW) + 1) + "|"]
md_lines += [f"| {r['model']} | " + " | ".join(f"{r[k]:.4f}" for k in SHOW) + " |" for r in rows]
(OUT_RESULTS / "two_tower.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print("saved ->", OUT_RESULTS / "two_tower.md")
""")

md(r"""
## Task 7 — Artifacts (Phase 3 FAISS index inhi se banega)

Best variant = **val** Recall@100 par (test dekh ke choose nahi karte). Saved:
- `models/<cat>/two_tower.pt` — weights + config + variant
- `models/<cat>/item_emb.npy` — saare items ke 64-d vectors (test start time par)
""")

code(r"""
best_name = max(results, key=lambda r: r["val_recall_at_100"])["model"].replace("two_tower_", "")
best_model = final_models[best_name]
OUT_MODELS.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": best_model.state_dict(), "variant": best_name, "flags": VARIANTS[best_name],
            "cfg": CFG, "n_items": N_ITEMS, "n_users": N_USERS, "t_ref": T_TEST}, OUT_MODELS / "two_tower.pt")
_, V = embed_all(best_model, train_val, T_TEST)
np.save(OUT_MODELS / "item_emb.npy", V.astype(np.float32))
(OUT_MODELS / "two_tower_config.json").write_text(json.dumps({"variant": best_name, **CFG}, indent=2, default=str))
print(f"best variant (by val): {best_name} | item_emb {V.shape} -> {OUT_MODELS}")
""")

md(r"""
## Findings (REPORT.md mein — apne real numbers ke saath)

- **Text encoder:** minilm / tfidf — …
- **Text NN sanity:** Mario → … (sensible? haan/nahi)
- **Ablation:** id_only vs content_only vs id_content vs id_content_age → …
- **Cold items:** content variants ka cold/recall@100 = … (ID models = 0) → text tower ne … % cold targets pakde
- **vs recent popularity (0.0916):** jeeta / haara → agar haara, kyun? (time drift? val–test cold share mismatch 27% vs 46%?)
- **Caveats:** static eval (test window mein release hue items ko age 0); negatives ki age apne row ke time se (approximation)

**Next:** Phase 3 — `item_emb.npy` se FAISS ANN index + latency/recall trade-off.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "03_two_tower.ipynb")
print("ok", len(C), "cells")
