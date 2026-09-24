"""VQ cell — the repaired vector quantizer (exp5 lineage).

The original implementation had a no-op optimizer and a
scale-mismatched codebook. This version: k-means initialization,
EMA codebook updates, and dead-code revival — prerequisite for
honest capacity-coupling measurements.
"""
import torch
import torch.nn as nn


def _kmeans_init(flat, k, iters=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(flat.shape[0], generator=g)[:k]
    codes = flat[idx].clone()
    for _ in range(iters):
        d = torch.cdist(flat, codes)
        assign = d.argmin(-1)
        for j in range(k):
            m = assign == j
            if m.any():
                codes[j] = flat[m].mean(0)
    return codes


class VQCell(nn.Module):
    """Per-position vector quantizer with EMA codebook."""

    def __init__(self, dim, k, seed=0, ema_decay=0.99, eps=1e-5):
        super().__init__()
        self.k = k
        self.ema_decay = ema_decay
        self.eps = eps
        self.register_buffer("codes", torch.zeros(k, dim))
        self.register_buffer("cluster_size", torch.zeros(k))
        self.register_buffer("ema_w", torch.zeros(k, dim))
        self._seed = seed
        self._initialized = False

    @torch.no_grad()
    def _init(self, flat):
        self.codes.copy_(_kmeans_init(flat, self.k, seed=self._seed))
        self.ema_w.copy_(self.codes * self.cluster_size.unsqueeze(1)
                         .clamp_min(1))
        self._initialized = True

    @torch.no_grad()
    def quantize(self, z):
        """z: (N, T, D) -> codes (N, T) integer, quantized (N, T, D)."""
        flat = z.reshape(-1, z.shape[-1])
        if not self._initialized:
            self._init(flat)
        d = torch.cdist(flat, self.codes)
        idx = d.argmin(-1)
        q = self.codes[idx].reshape_as(z)
        onehot = nn.functional.one_hot(idx, self.k).type(flat.dtype)
        # EMA codebook update
        self.cluster_size.mul_(self.ema_decay).add_(
            onehot.sum(0), alpha=1 - self.ema_decay)
        self.ema_w.mul_(self.ema_decay).add_(
            onehot.t() @ flat, alpha=1 - self.ema_decay)
        n = self.cluster_size.sum()
        cs = (self.cluster_size + self.eps) / (n + self.k * self.eps) * n
        self.codes.copy_(self.ema_w / cs.unsqueeze(1).clamp_min(self.eps))
        # dead-code revival: reassign least-used codes to random inputs
        usage = self.cluster_size
        dead = usage < 1.0
        if dead.any():
            rnd = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),))]
            self.codes[dead] = rnd
        return idx.reshape(z.shape[:2]), q

    def straight_through(self, z):
        """Quantize with straight-through gradient estimator."""
        idx, q = self.quantize(z)
        return z + (q - z).detach(), idx
