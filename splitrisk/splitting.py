"""splitrisk/splitting.py — 通用 decoder-only 切分器 (AutoSplitter)

原則（bug ledger 鐵律）：只走官方 model API 截層，不手調 block、不手造 mask。
支持兩族：
  - gpt2 家族：GPT2Model 截層（probe_common.Head/Tail，e2e 已驗證）
  - llama 家族結構（Llama/Qwen2/Mistral/Gemma/Yi/Phi3 等一切暴露
    model.model.layers + model.norm + lm_head 的 HF decoder-only 模型）：
    LlamaModel 模式泛化 —— embed+前 s 層留客戶端（z = pre-norm 末層輸出，
    與 GPT-2 系同口徑），其餘層+final norm+lm_head 為服務端。

強制關卡：parity（head+tail vs full logits）與因果洩漏（prefix-invariance）。
不過閘門，構造函數直接 raise —— 病變骨架不許離開本模塊。
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn

# 走 llama-泛化路徑的 HF model_type（官方 AutoModelForCausalLM 結構同構）
LLAMA_STYLE_TYPES = {"llama", "qwen2", "mistral", "gemma", "gemma2",
                     "cohere", "starcoder2", "olmo", "phi3", "qwen3"}


class _LlamaStyleHead(nn.Module):
    """客戶端: embed + layers[0,s)。官方 forward；z = pre-norm hidden_states[-1]。
    norm=Identity：final norm 屬於 tail 側（與 gpt2 Head 同約定）。"""

    def __init__(self, full, s):
        super().__init__()
        config = copy.deepcopy(full.config)
        config.num_hidden_layers = s
        if getattr(config, "layer_types", None) is not None:
            config.layer_types = config.layer_types[:s]
        self.inner = type(full.model)(config)
        self.inner.embed_tokens = full.model.embed_tokens
        if hasattr(full.model, "rotary_emb"):
            self.inner.rotary_emb = full.model.rotary_emb
        self.inner.layers = full.model.layers[:s]   # 引用共享，不 deepcopy
        self.inner.norm = torch.nn.Identity()
        self.embed_tokens = self.inner.embed_tokens
        self.hidden_size = full.config.hidden_size

    def forward(self, input_ids, attention_mask=None):
        out = self.inner(input_ids=input_ids,
                         attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1]


class _LlamaStyleTail(nn.Module):
    """服務端: layers[s,N) + final norm(訓練過的那個) + lm_head。z 經
    inputs_embeds 進官方 forward —— RoPE 位置與 causal mask 正確。
    注意 LlamaModel.forward 末端自帶 self.norm，故此處不再二次 norm。
    hidden_states(): 任務頭用（post-norm hidden，非 vocab logits）。"""

    def __init__(self, full, s):
        super().__init__()
        config = copy.deepcopy(full.config)
        config.num_hidden_layers = len(full.model.layers) - s
        if getattr(config, "layer_types", None) is not None:
            config.layer_types = config.layer_types[s:]
        self.inner = type(full.model)(config)
        self.inner.embed_tokens = full.model.embed_tokens
        if hasattr(full.model, "rotary_emb"):
            self.inner.rotary_emb = full.model.rotary_emb
        self.inner.layers = full.model.layers[s:]
        self.inner.norm = full.model.norm           # 訓練過的 final norm
        self.lm_head = full.lm_head
        self._gemma = full.config.model_type in {"gemma", "gemma2"}

    def hidden_states(self, z, attention_mask=None):
        # Older Gemma forwards scale inputs_embeds too. At the split boundary
        # z is already a residual-stream activation, not a token embedding.
        # Restore it exactly at the first retained block; this also works with
        # newer versions whose scaling lives inside embed_tokens, and keeps
        # the official RoPE/mask preparation and autograd intact.
        hook = None
        if self._gemma:
            def restore_boundary(module, args, kwargs):
                if args:
                    return (z,) + args[1:], kwargs
                return args, {**kwargs, "hidden_states": z}
            hook = self.inner.layers[0].register_forward_pre_hook(
                restore_boundary, with_kwargs=True)
        try:
            out = self.inner(inputs_embeds=z, attention_mask=attention_mask,
                             use_cache=False)
        finally:
            if hook is not None:
                hook.remove()
        return out.last_hidden_state

    def forward(self, z, attention_mask=None):
        logits = self.lm_head(self.hidden_states(z, attention_mask))
        cap = getattr(self.inner.config, "final_logit_softcapping", None)
        if cap is not None:
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        return logits


class AutoSplitter:
    """load(model_name_or_model, split_at, device) -> (head, tail, full)
    head/tail 引用共享原模型層（零複製）；full 保留供 ambient/parity 用。"""

    @classmethod
    def load(cls, model, split_at: int, device=None):
        from transformers import AutoModelForCausalLM
        if isinstance(model, str):
            full = AutoModelForCausalLM.from_pretrained(model)
        else:
            full = model
        if device is not None:
            full = full.to(device)
        full.eval()
        cfg = full.config
        n_layers = getattr(cfg, "n_layer", None) or \
            getattr(cfg, "num_hidden_layers", None)
        if not 0 < split_at < n_layers:
            raise ValueError(f"split_at={split_at} 越界 (n_layers={n_layers})")
        mtype = getattr(cfg, "model_type", "")
        if mtype == "gpt2":
            from .probes.probe_common import Head, Tail
            head, tail = Head(full, split_at), Tail(full, split_at)
        elif mtype in LLAMA_STYLE_TYPES:
            if not (hasattr(full.model, "layers") and hasattr(full, "lm_head")):
                raise ValueError(
                    f"model_type={mtype} 缺少 model.layers/lm_head 結構——"
                    f"不在此列的架構請走 audit_custom_head")
            head, tail = _LlamaStyleHead(full, split_at), \
                _LlamaStyleTail(full, split_at)
        else:
            raise ValueError(
                f"model_type={mtype!r} 未在 AutoSplitter 支持列表"
                f"（gpt2 / {sorted(LLAMA_STYLE_TYPES)}）。"
                f"per-deployment 校準強制——先跑錨點再申請簽章。")
        return head, tail, full


@torch.no_grad()
def parity_gate(head, tail, full, device=None, tol=2e-2) -> float:
    """head+tail 與 full model 的 logits 必須一致（bf16 容差）。"""
    head.eval()
    tail.eval()
    full.eval()
    dev = device or next(head.parameters()).device
    torch.manual_seed(0)
    vocab = full.get_output_embeddings().weight.shape[0]
    ids = torch.randint(0, vocab, (2, 16), device=dev)
    z = head(ids)
    d = (tail(z) - full(input_ids=ids).logits).abs().max().item()
    if d >= tol:
        raise AssertionError(
            f"PARITY FAIL: Δlogits={d:.4f} ≥ {tol} — 骨架病變，禁止出廠")
    return d


@torch.no_grad()
def causal_gate(head, device=None, tol=1e-3) -> float:
    """換掉後半段 token，前半段 z 必須逐位不變（因果洩漏檢查）。"""
    head.eval()
    dev = device or next(head.parameters()).device
    emb = getattr(head, "embed_tokens", None)
    if emb is None:                     # gpt2 Head：embedding 叫 wte
        emb = head.base_model.get_output_embeddings()
    torch.manual_seed(0)
    vocab = emb.weight.shape[0]
    ids = torch.randint(0, vocab, (2, 16), device=dev)
    z = head(ids)
    ids2 = ids.clone()
    ids2[:, 8:] = torch.randint(0, vocab, (2, 8), device=dev)
    d = (z[:, :8] - head(ids2)[:, :8]).abs().max().item()
    if d >= tol:
        raise AssertionError(f"CAUSAL LEAK: Δprefix={d:.4f} ≥ {tol}")
    return d


def gates(head, tail, full, device=None, tol=2e-2):
    """跑雙閘門，返回 (parity_Δ, causal_Δ)。任何 FAIL 即 raise。"""
    return parity_gate(head, tail, full, device, tol), \
        causal_gate(head, device)
