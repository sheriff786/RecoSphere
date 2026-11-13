"""
Two-tower model — importable version of notebook 03's model.

Notebook 03 trains and saves `models/<cat>/two_tower.pt`. Notebook 04 (ANN) and the
serving API (Phase 6) load it from here, so the architecture lives in ONE place.

IMPORTANT: attribute names here must match notebook 03 exactly (text_proj, id_emb,
item_mlp, ...), otherwise `load_state_dict(strict=True)` fails. That strict load is the
check that training and serving use the same model.
"""
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = -1


def age_bucket(dfa: torch.Tensor, iids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Same formula as notebook 03: 1 + min(log2(age_days + 1), 15); 17 = DFA missing."""
    d = dfa[iids]
    days = (t.double() - d).clamp(min=0) / 86400.0
    b = 1 + torch.clamp(torch.floor(torch.log2(days + 1)), max=15).long()
    return torch.where(d < 0, torch.full_like(b, 17), b)


class TwoTower(nn.Module):
    def __init__(self, n_items, use_id, use_content, use_age, cfg,
                 text_dim=None, n_store=None, n_type=None, n_rated=None):
        super().__init__()
        self.use_id, self.use_content, self.use_age = use_id, use_content, use_age
        self.id_dropout = cfg["id_dropout"] if (use_id and use_content) else 0.0
        # buffers: real values come from the checkpoint (load_state_dict)
        self.register_buffer("known", torch.zeros(n_items, dtype=torch.bool))
        in_dim = 0
        if use_content:
            self.register_buffer("text", torch.zeros(n_items, text_dim))
            self.register_buffer("store", torch.zeros(n_items, dtype=torch.long))
            self.register_buffer("type_", torch.zeros(n_items, dtype=torch.long))
            self.register_buffer("rated", torch.zeros(n_items, dtype=torch.long))
            self.text_proj = nn.Linear(text_dim, 128)
            self.store_emb, self.type_emb, self.rated_emb = (
                nn.Embedding(n_store, 16), nn.Embedding(n_type, 8), nn.Embedding(n_rated, 8))
            in_dim += 128 + 16 + 8 + 8
        if use_id:
            self.id_emb = nn.Embedding(n_items, 64)
            nn.init.normal_(self.id_emb.weight, std=0.05)            # same init as notebook 03
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
            keep = self.known[iids]                                   # cold item -> ID zero
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
        w = torch.nan_to_num(w, nan=0.0)                              # no-history user -> zero vector
        pooled = (w.unsqueeze(-1) * v).sum(dim=1)
        return F.normalize(pooled + self.user_mlp(pooled), dim=-1)


def load_two_tower(ckpt_path, gold_dir, device="cpu"):
    """Rebuild the model from notebook 03's checkpoint. Returns (model, checkpoint_dict, dfa_tensor)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)   # own file, not untrusted
    sd, flags, cfg = ck["state_dict"], ck["flags"], ck["cfg"]
    shapes = {"n_items": sd["known"].shape[0]}
    if flags["use_content"]:
        shapes.update(text_dim=sd["text"].shape[1], n_store=sd["store_emb.weight"].shape[0],
                      n_type=sd["type_emb.weight"].shape[0], n_rated=sd["rated_emb.weight"].shape[0])
    model = TwoTower(**shapes, **flags, cfg=cfg).to(device)
    model.load_state_dict(sd, strict=True)          # fails loudly if architecture drifted from notebook 03
    model.eval()
    feats = pl.read_parquet(Path(gold_dir) / "item_features.parquet").sort("iid")
    dfa = torch.tensor(feats["dfa_ts"].fill_null(-1).to_numpy(), dtype=torch.float64, device=device)
    return model, ck, dfa


def history_at_cutoff(df: pl.DataFrame, n_users: int, hist_len: int) -> np.ndarray:
    """Last `hist_len` items per user (most recent in the last slot) — same as notebook 03."""
    H = np.full((n_users, hist_len), PAD, dtype=np.int64)
    g = df.sort(["uid", "ts", "iid"]).group_by("uid", maintain_order=True).agg(pl.col("iid").tail(hist_len))
    for uid, items in zip(g["uid"].to_list(), g["iid"].to_list()):
        H[uid, hist_len - len(items):] = items
    return H


@torch.no_grad()
def embed_items(model, dfa, t_ref, chunk=8192):
    n = model.known.shape[0]
    dev = model.known.device
    iids = torch.arange(n, device=dev)
    t = torch.full((n,), int(t_ref), dtype=torch.long, device=dev)
    out = [model.item_vec(iids[s:s + chunk], age_bucket(dfa, iids[s:s + chunk], t[s:s + chunk]) if model.use_age else None)
           for s in range(0, n, chunk)]
    return torch.cat(out).cpu().numpy().astype(np.float32)


@torch.no_grad()
def embed_users(model, H: np.ndarray, chunk=8192):
    dev = model.known.device
    Ht = torch.tensor(H, device=dev)
    out = [model.user_vec(Ht[s:s + chunk]) for s in range(0, len(Ht), chunk)]
    return torch.cat(out).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Training (same logic as notebook 03) — used by notebook 05 to train a TRAIN-ONLY
# retrieval model, so ranker training candidates never come from a model that saw val.
# ---------------------------------------------------------------------------
def build_examples(df: pl.DataFrame, hist_len: int) -> dict:
    """history = same user's previous items, strictly earlier ts; target = current item."""
    df = df.sort(["uid", "ts", "iid"])
    u, i, t = df["uid"].to_numpy(), df["iid"].to_numpy(), df["ts"].to_numpy()
    n, L = len(df), hist_len
    hist = np.full((n, L), PAD, dtype=np.int64)
    hist_ts = np.full((n, L), -1, dtype=np.int64)
    for k in range(1, L + 1):
        src = np.arange(n) - k
        s = np.clip(src, 0, None)
        ok = (src >= 0) & (u[s] == u) & (t[s] < t)
        hist[ok, L - k] = i[s[ok]]
        hist_ts[ok, L - k] = t[s[ok]]
    keep = (hist != PAD).any(axis=1)
    ex = {"hist": hist[keep], "hist_ts": hist_ts[keep], "target": i[keep], "t": t[keep]}
    valid = ex["hist"] != PAD
    assert (np.where(valid, ex["hist_ts"], -1) < ex["t"][:, None]).all(), "history contains future items!"
    return ex


def in_batch_loss(model, hist, target, t, log_q, tau, dfa):
    u = model.user_vec(hist)
    v = model.item_vec(target, age_bucket(dfa, target, t) if model.use_age else None)
    logits = (u @ v.T) / tau - log_q[target][None, :]
    same = (target[:, None] == target[None, :]) & ~torch.eye(len(target), dtype=torch.bool, device=target.device)
    logits = logits.masked_fill(same, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(target), device=target.device))


def new_two_tower(flags, cfg, item_feat: pl.DataFrame, text_emb: np.ndarray, known_items, device="cpu"):
    """Fresh model with data buffers filled (text, store, type, rated, known)."""
    n_items = item_feat.height
    shapes = {"n_items": n_items}
    if flags["use_content"]:
        shapes.update(text_dim=text_emb.shape[1], n_store=int(item_feat["store_id"].max()) + 1,
                      n_type=int(item_feat["type_id"].max()) + 1, n_rated=int(item_feat["rated_id"].max()) + 1)
    model = TwoTower(**shapes, **flags, cfg=cfg)
    with torch.no_grad():
        model.known.zero_(); model.known[torch.as_tensor(np.asarray(known_items))] = True
        if flags["use_content"]:
            model.text.copy_(torch.tensor(text_emb))
            model.store.copy_(torch.tensor(item_feat["store_id"].to_numpy()))
            model.type_.copy_(torch.tensor(item_feat["type_id"].to_numpy()))
            model.rated.copy_(torch.tensor(item_feat["rated_id"].to_numpy()))
    return model.to(device)


def train_two_tower(flags, cfg, fit_df, epochs, item_feat, text_emb, dfa, device="cpu", log=print):
    """Fixed-epoch training (no early stopping) — epochs come from notebook 03's val tuning."""
    import random
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    ex = build_examples(fit_df, cfg["hist_len"])
    n_items = item_feat.height
    cnt = np.bincount(ex["target"], minlength=n_items) + 1.0
    log_q = torch.tensor(np.log(cnt / cnt.sum()), dtype=torch.float32, device=device)
    tens = {k: torch.tensor(ex[k], device=device) for k in ("hist", "target", "t")}
    model = new_two_tower(flags, cfg, item_feat, text_emb, fit_df["iid"].unique().to_numpy(), device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = len(ex["target"]), cfg["batch_size"]
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            loss = in_batch_loss(model, tens["hist"][b], tens["target"][b], tens["t"][b], log_q, cfg["tau"], dfa)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        log(f"  epoch {ep}/{epochs}  loss={tot / n:.4f}")
    model.eval()
    return model


def save_checkpoint(model, path, variant, flags, cfg, n_users, t_ref):
    torch.save({"state_dict": model.state_dict(), "variant": variant, "flags": flags, "cfg": cfg,
                "n_items": int(model.known.shape[0]), "n_users": n_users, "t_ref": int(t_ref)}, path)
