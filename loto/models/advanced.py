"""Advanced model family: small GRU sequence model and a clustering-based regime model.

Both are deliberately small: ~2,458 observations cannot support large networks, and the
backtest decides whether any of this complexity earns its keep.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import setup_logging

log = setup_logging("loto.advanced")
POOL, PICK = 42, 6
P0 = PICK / POOL


def walk_forward_gru(ind: np.ndarray, start: int, end: int, refit_every: int = 100, seq_len: int = 20,
                     hidden: int = 32, epochs: int = 30, seed: int = 0) -> tuple[np.ndarray, float, int]:
    import torch
    from torch import nn

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    N = len(ind)
    P = np.full((N, POOL), P0)
    x_all = torch.tensor(ind, dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(POOL, hidden, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(hidden, POOL)
            nn.init.zeros_(self.out.weight)
            nn.init.constant_(self.out.bias, float(np.log(P0 / (1 - P0))))

        def forward(self, x):
            h, _ = self.gru(x)
            return self.out(self.drop(h[:, -1]))

    def windows(lo, hi):
        idx = np.arange(max(lo, seq_len), hi)
        X = torch.stack([x_all[i - seq_len:i] for i in idx]) if len(idx) else torch.empty(0, seq_len, POOL)
        return X, x_all[idx]

    t0, n_fits = time.time(), 0
    for block in range(start, end, refit_every):
        X, Y = windows(0, block)
        n_val = max(50, len(X) // 10)
        Xtr, Ytr, Xva, Yva = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, patience = np.inf, None, 0
        g = torch.Generator().manual_seed(seed + block)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr), generator=g)
            for b in range(0, len(Xtr), 128):
                ii = perm[b:b + 128]
                opt.zero_grad()
                loss = lossf(net(Xtr[ii]), Ytr[ii])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = lossf(net(Xva), Yva).item()
            if vl < best - 1e-5:
                best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 5:
                    break
        net.load_state_dict(best_state)
        # refit-free prediction for the block, using only draws < t for each t
        hi = min(block + refit_every, end)
        Xte = torch.stack([x_all[i - seq_len:i] for i in range(block, hi)])
        net.eval()
        with torch.no_grad():
            P[block:hi] = torch.sigmoid(net(Xte)).numpy()
        n_fits += 1
    return np.clip(P, 1e-4, 1 - 1e-4), time.time() - t0, n_fits


def walk_forward_regime(ind: np.ndarray, start: int, end: int, refit_every: int = 50, k: int = 3,
                        w: int = 50, alpha: float = 20.0, seed: int = 0) -> tuple[np.ndarray, float, int]:
    """Cluster rolling-frequency states with KMeans; forecast with number frequencies observed
    in past draws that followed the same regime as the current state."""
    from sklearn.cluster import KMeans

    N = len(ind)
    cs = np.vstack([np.zeros((1, POOL)), np.cumsum(ind, 0)])
    t = np.arange(N)
    lo = np.maximum(0, t - w)
    state = (cs[t] - cs[lo]) / np.maximum(t - lo, 1)[:, None]   # uses draws < t
    P = np.full((N, POOL), P0)
    t0, n_fits = time.time(), 0
    for block in range(start, end, refit_every):
        tr = np.arange(w, block)
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(state[tr])
        lab_tr = km.labels_
        hi = min(block + refit_every, end)
        lab_te = km.predict(state[block:hi])
        for c in range(k):
            sel = tr[lab_tr == c]
            f = (ind[sel].sum(0) + alpha * P0) / (len(sel) + alpha)
            P[block:hi][lab_te == c] = f
        n_fits += 1
    # within a block the regime model can also use draws inside the block that precede t: keep it
    # simple and conservative - no intra-block updating (never uses future draws)
    return np.clip(P, 1e-4, 1 - 1e-4), time.time() - t0, n_fits
