"""
Phase 1 baselines. Teeno ka interface same: fit(X) aur score(user_ids) -> dense [n, n_items].

1. Popularity  — sabko same list. Agar tumhara deep model isse nahi jeetta, kuch toota hai.
                 `recent_frac` option: sirf train ke last hisse ki popularity. Temporal split mein
                 "abhi kya trending hai" aksar all-time popularity se bahut strong hota hai.
2. ItemKNN     — cosine item-item similarity, har item ke top-k neighbours.
                 score(u, j) = sum over i in history(u) of sim(i, j)
3. EASE        — Steck 2019. Closed form: B = I - P·diag(1/diag(P)), P = (XᵀX + λI)⁻¹, diag(B)=0.
                 O(items³) -> sirf top `max_items` popular items par. Baaki items -inf score.
                 Ye limitation REPORT mein likhni hai: L3 par EASE seedha nahi chalega.
"""
import numpy as np
import scipy.sparse as sp


class Popularity:
    name = "popularity"

    def __init__(self, recent_frac: float | None = None):
        self.recent_frac = recent_frac
        if recent_frac:
            self.name = f"popularity_recent{int(recent_frac * 100)}"

    def fit(self, X: sp.csr_matrix, recent_X: sp.csr_matrix | None = None):
        M = recent_X if (self.recent_frac and recent_X is not None) else X
        self.pop = np.asarray(M.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.pop, (len(users), len(self.pop))).copy()


class ItemKNN:
    def __init__(self, k: int = 100, shrink: float = 10.0, chunk: int = 2048):
        self.k, self.shrink, self.chunk = k, shrink, chunk
        self.name = f"itemknn_k{k}"

    def fit(self, X: sp.csr_matrix):
        self.X = X.tocsr()
        Xc = X.tocsc().astype(np.float32)
        norms = np.sqrt(np.asarray(Xc.multiply(Xc).sum(axis=0)).ravel())
        n = X.shape[1]
        rows, cols, vals = [], [], []
        XtX = (Xc.T @ Xc).tocsr()  # co-occurrence counts, sparse
        for s in range(0, n, self.chunk):
            block = XtX[s:s + self.chunk].toarray()
            denom = norms[s:s + self.chunk, None] * norms[None, :] + self.shrink + 1e-8
            sim = block / denom
            idx = np.arange(block.shape[0])
            sim[idx, s + idx] = 0.0  # self-similarity hatao
            k = min(self.k, n - 1)
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            v = np.take_along_axis(sim, top, axis=1)
            r = np.repeat(np.arange(s, s + block.shape[0]), k)
            mask = v.ravel() > 0
            rows.append(r[mask]); cols.append(top.ravel()[mask]); vals.append(v.ravel()[mask])
        self.S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n, n))
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.X[users] @ self.S).toarray()


class EASE:
    def __init__(self, lam: float = 500.0, max_items: int = 20000):
        self.lam, self.max_items = lam, max_items
        self.name = f"ease_l{int(lam)}"

    def fit(self, X: sp.csr_matrix):
        self.n_items = X.shape[1]
        pop = np.asarray(X.sum(axis=0)).ravel()
        self.items = np.sort(np.argsort(-pop)[: self.max_items])  # top-N popular
        self.Xs = X[:, self.items].tocsr().astype(np.float32)
        G = (self.Xs.T @ self.Xs).toarray()
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        out = np.full((len(users), self.n_items), -np.inf, dtype=np.float32)
        out[:, self.items] = self.Xs[users] @ self.B
        return out
