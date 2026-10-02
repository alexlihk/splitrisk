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
        self.base_model = model                     # 供 custom-head 路徑引用
        self.inner = GPT2Model(model.config)
        self.inner.wte = tr.wte
        self.inner.wpe = tr.wpe
        self.inner.h = tr.h[:split_at]
        # ln_f 屬於 tail 側（bug ledger 約定：Head=截層+ln_f=Identity）
        # 否則新實例的 default LayerNorm(w=1,b=0) 會把 z 中途再標準化一次
        self.inner.ln_f = torch.nn.Identity()
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
        import copy
        from transformers import GPT2Model
        tr = model.transformer
        self.inner = GPT2Model(model.config)
        self.inner.wte = tr.wte
        # wpe 用歸零副本：官方 forward 在 inputs_embeds 路徑仍會加 wpe，
        # 而位置已在 head 側加過——這裡加零，保持語義不變（exp13 約定）
        self.inner.wpe = copy.deepcopy(tr.wpe)
        self.inner.h = tr.h[split_at:]
        self.inner.ln_f = tr.ln_f                   # 訓練過的 final norm 在這側
        self.lm_head = model.lm_head
        self.split_at = split_at
        with torch.no_grad():
            self.inner.wpe.weight.zero_()
        self.inner.wpe.weight.requires_grad = False

    def hidden_states(self, z, attention_mask=None):
        """post-ln_f hidden states（任務頭用，非 vocab logits）。"""
        out = self.inner(inputs_embeds=z, attention_mask=attention_mask,
                         use_cache=False)
        return out.last_hidden_state

    def forward(self, z, attention_mask=None):
        return self.lm_head(self.hidden_states(z, attention_mask))

    def freeze_position_embedding(self):
        pass  # wpe 已是歸零副本且凍結（__init__ 內完成；保留接口相容）


def load_gpt2(name_or_model, device=None):
    """Accept a model name (HF hub) or an already-loaded model."""
    if isinstance(name_or_model, str):
        from transformers import AutoModelForCausalLM
        import os
        # v0.3.1: gated repo（Llama-3.1-8B 等）走 HF_TOKEN 環境變數——
        # token 永不硬編（security red line；h20 S8 的 403 修法）
        model = AutoModelForCausalLM.from_pretrained(
            name_or_model, token=os.environ.get("HF_TOKEN") or None)
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
        return self.net(z.float())


def token_logits(z_recon, wte):
    """Project reconstructed states to vocab via the (shared) wte."""
    return z_recon @ wte.t()
