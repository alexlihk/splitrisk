"""splitrisk/probes/label_probe.py — R2 標籤洩漏探針

表徵 → 類別標籤的分類器讀數（exp14 產品化）。位元率鎖不死標籤：
9-bit 縫壓 token 恢復但不壓標籤（0.859 vs 0.871）——「類別即敏感」
場景紅燈的依據。輸出 acc 與超額（acc − 隨機基線 1/n_classes）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pool(z, mask):
    """mean-pool 有效位置（mask: bool (N,T)，全 False 行回退首位）。"""
    m = mask.unsqueeze(-1).to(z.dtype)
    pooled = (z * m).sum(1) / m.sum(1).clamp_min(1)
    empty = m.sum(1) == 0
    if empty.any():
        pooled[empty] = z[empty][:, 0]
    return pooled


def train_label_probe(z, labels, mask, n_classes, hidden=256, epochs=5,
                      lr=1e-3, seed=0, device="cpu"):
    """R2 探針：mean-pool 表徵 → 2 層 MLP → 類別。返回 (net, acc, random)。"""
    torch.manual_seed(seed)
    pooled = _pool(z, mask).to(device)
    y = labels.to(device).long()
    net = nn.Sequential(nn.Linear(pooled.shape[-1], hidden), nn.GELU(),
                        nn.Linear(hidden, n_classes)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = pooled.shape[0]
    for _ in range(max(1, epochs)):
        net.train()
        perm = torch.randperm(n)
        for s in range(0, n, 64):
            idx = perm[s:s + 64]
            loss = lossf(net(pooled[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(pooled).argmax(-1) == y).float().mean().item()
    return net, acc, 1.0 / n_classes


def label_audit(z, labels, mask, n_classes, device="cpu", seed=0,
                epochs=5):
    """一次性讀數：返回 dict(acc, random, excess_pp, n_classes)。"""
    _, acc, rnd = train_label_probe(z, labels, mask, n_classes,
                                    device=device, seed=seed, epochs=epochs)
    return {"R2_label_acc": acc, "R2_random": rnd,
            "R2_excess_pp": 100.0 * (acc - rnd), "n_classes": n_classes}
