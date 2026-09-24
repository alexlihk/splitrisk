"""Tier-3 context-aware reassembler: transformer decoder over the full
(quantized) representation sequence. The collusive-ceiling instrument
for sharding audits and the capacity ladder for seam audits.

Capacity ladder (paper sec. 'the one interior region'): recovery does
not plateau with decoder capacity — the ceiling is open and every
AMBER report carries a freshness label.
"""
import torch
import torch.nn as nn


class SeqDecoder(nn.Module):
    """Small bidirectional transformer over per-position representations
    predicting token identities (context-aware decoding)."""

    def __init__(self, dim, vocab_size, wte=None, n_layers=3, n_heads=4,
                 hidden=None, max_len=128):
        super().__init__()
        hidden = hidden or dim
        self.wte = wte
        if wte is None:
            self.out = nn.Linear(hidden, vocab_size)
        else:
            self.out = None  # project via shared wte
        self.proj_in = nn.Linear(dim, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=4 * hidden,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.max_len = max_len

    def forward(self, z):
        h = self.proj_in(z)
        h = self.encoder(h)
        if self.out is not None:
            return self.out(h)
        return h @ self.wte.t()


@torch.no_grad()
def eval_recovery(net, z, token_ids, device="cpu", batch_size=256):
    net.eval()
    top1 = total = 0
    for i in range(0, z.shape[0], batch_size):
        zb = z[i:i + batch_size].to(device)
        tb = token_ids[i:i + batch_size].to(device)
        logits = net(zb.to(z.dtype))
        pred = logits.argmax(-1)
        m = tb != -100
        top1 += (pred[m] == tb[m]).sum().item()
        total += m.sum().item()
    return top1 / max(1, total)


def train_reassembler(z, token_ids, vocab_size, wte=None, n_layers=3,
                      epochs=2, lr=1e-3, batch_size=64, device="cpu",
                      seed=0):
    """Train the context-aware decoder; returns (net, top1)."""
    torch.manual_seed(seed)
    d = z.shape[-1]
    net = SeqDecoder(d, vocab_size, wte=wte, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    ds = torch.utils.data.TensorDataset(z, token_ids)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(max(1, epochs)):
        for zb, tb in dl:
            zb, tb = zb.to(device), tb.to(device)
            logits = net(zb.to(z.dtype))
            loss = lossf(logits.reshape(-1, logits.shape[-1]),
                         tb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, eval_recovery(net, z, token_ids, device=device)
