"""Tier-1 (MSE fresher) and Tier-2 (CE decoder) probe training.

The two probes share one architecture and differ only in loss — that
is the paper's central instrument: the attacker's loss function, not
its architecture, sets the measurement (444x instrument gap).
"""
import torch
import torch.nn as nn
from .probe_common import ReconNet


def train_probe(z, token_ids, wte, mode="ce", hidden=None, epochs=2,
                lr=1e-3, batch_size=32, device="cpu", seed=0):
    """Train a probe on cached split representations.

    z:        (N, T, D) cached split-point representations
    token_ids:(N, T) ground-truth token identities
    mode:     "ce" (Tier 2, primary) or "mse" (Tier 1, instrument)
    Returns the trained ReconNet.
    """
    torch.manual_seed(seed)
    n, t, d = z.shape
    net = ReconNet(d, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    if mode == "ce":
        lossf = nn.CrossEntropyLoss()
    else:
        lossf = nn.MSELoss()
    wte = wte.to(device)

    ds = torch.utils.data.TensorDataset(z, token_ids)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                     shuffle=True)
    for epoch in range(max(1, epochs)):
        for zb, tb in dl:
            zb, tb = zb.to(device), tb.to(device)
            rec = net(zb)
            if mode == "ce":
                logits = rec @ wte.t().float()
                loss = lossf(logits.reshape(-1, logits.shape[-1]),
                             tb.reshape(-1))
            else:
                # MSE fresher: reconstruct the pre-transformation
                # representation itself (standard split-learning eval)
                loss = lossf(rec, zb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def probe_readout(net, z, token_ids, wte, mask=None, batch_size=512,
                  device="cpu"):
    """Cosine retrieval top-1/top-5 against the embedding matrix.

    Recovery is scored by cosine retrieval (not gameable in our
    experiments), over non-padding positions.
    """
    net.eval()
    wte = wte.to(device)
    wte_n = wte / wte.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    top1 = top5 = total = 0
    for i in range(0, z.shape[0], batch_size):
        zb = z[i:i + batch_size].to(device)
        tb = token_ids[i:i + batch_size].to(device)
        rec = net(zb.to(z.dtype))
        rec_n = rec / rec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sim = rec_n @ wte_n.t().float()                    # (B, T, V)
        pred = sim.argmax(-1)
        top5v = sim.topk(5, dim=-1).indices
        m = (tb != -100).to(device)
        if mask is not None:
            m = m & mask[i:i + batch_size].to(device).bool()
        top1 += (pred[m] == tb[m]).sum().item()
        top5 += (top5v[m] == tb[m].unsqueeze(-1)).any(-1).sum().item()
        total += m.sum().item()
    t1 = top1 / max(1, total)
    t5 = top5 / max(1, total)
    return t1, t5
