"""Shared split skeletons and probe networks.

Engineering rule (from the project's bug ledger): never hand-call
decoder blocks — always use the official model API, then truncate.
Head = official model truncated at the split layer (ln_f bypassed);
Tail = official model over remaining layers via inputs_embeds, with
the position embedding zeroed/frozen where applicable.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------- Head/Tail
class Head(nn.Module):
    """Client side: embeddings + layers [0, split_at) of a GPT-2 model.

    Uses the official GPT2Model forward over a truncated layer list —
    never hand-call GPT2Block (breaks on transformers 5.x; project bug
    ledger rule). Returns pre-ln_f hidden states (ln_f belongs to the
    tail side): GPT2Model.hidden_states[-1] is the last layer output
    BEFORE ln_f."""

    def __init__(self, model, split_at):
        super().__init__()
        from transformers import GPT2Model
        tr = model.transformer
        self.inner = GPT2Model(model.config)
        self.inner.wte = tr.wte
        self.inner.wpe = tr.wpe
        self.inner.h = tr.h[:split_at]
        self.split_at = split_at
        self.hidden_size = model.config.n_embd

    def forward(self, input_ids, attention_mask=None):
        out = self.inner(input_ids=input_ids,
                         attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1]


class Tail(nn.Module):
    """Server side: layers [split_at, N) + ln_f + lm_head, via the
    official GPT2Model forward on inputs_embeds (position embeddings
    were already added on the head side; wpe is zeroed and frozen)."""

    def __init__(self, model, split_at):
        super().__init__()
        from transformers import GPT2Model
        tr = model.transformer
        self.inner = GPT2Model(model.config)
        self.inner.wte = tr.wte
        self.inner.wpe = tr.wpe
        self.inner.h = tr.h[split_at:]
        self.lm_head = model.lm_head
        self.split_at = split_at
        # wpe is zeroed and frozen: positions belong to the head side
        self.inner.wpe.weight.requires_grad = False

    def forward(self, z, attention_mask=None):
        out = self.inner(inputs_embeds=z, attention_mask=attention_mask,
                         use_cache=False)
        return self.lm_head(out.last_hidden_state)   # post-ln_f

    def freeze_position_embedding(self):
        with torch.no_grad():
            self.inner.wpe.weight.zero_()


def load_gpt2(name_or_model, device=None):
    """Accept a model name (HF hub) or an already-loaded model."""
    if isinstance(name_or_model, str):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(name_or_model)
    else:
        model = name_or_model
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------- probes
class ReconNet(nn.Module):
    """Tier-2 CE decoder backbone: dim -> h -> h -> dim, GELU.
    Output dim ALWAYS equals input dim (catches the v1 768/1280 bug)."""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or 2048
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z):
        return self.net(z)


def token_logits(z_recon, wte):
    """Project reconstructed states to vocab via the (shared) wte."""
    return z_recon @ wte.t()
