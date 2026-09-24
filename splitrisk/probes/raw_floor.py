"""Tier-0 raw floor: zero-parameter cosine alignment of z against the
public embedding matrix. Measures linear readability only."""
import torch


@torch.no_grad()
def raw_floor(z, token_ids, wte, mask=None, device="cpu"):
    """Top-1 cosine retrieval of the raw representation against wte.

    No training, no parameters — deterministic; independent sessions
    must agree to all printed digits (reproducibility anchor).
    """
    wte = wte.to(device)
    wte_n = wte / wte.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    top1 = total = 0
    for i in range(z.shape[0]):
        z_n = z[i].to(device)
        z_n = z_n / z_n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sim = z_n @ wte_n.t()
        pred = sim.argmax(-1)
        tb = token_ids[i].to(device)
        m = (tb != -100).to(device)
        if mask is not None:
            m = m & mask[i].to(device).bool()
        top1 += (pred[m] == tb[m]).sum().item()
        total += m.sum().item()
    return top1 / max(1, total)
