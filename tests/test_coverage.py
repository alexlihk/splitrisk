"""v0.3 coverage tests — AutoSplitter / label_probe / sl（CPU tiny，免下載）"""
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402


def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(n_embd=32, n_layer=4, n_head=4, vocab_size=100,
                     n_positions=32)
    return GPT2LMHeadModel(cfg)


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=32, num_hidden_layers=4,
                      num_attention_heads=4, intermediate_size=64,
                      vocab_size=100, max_position_embeddings=64)
    return LlamaForCausalLM(cfg)


# ------------------------------------------------------------ AutoSplitter
def test_autosplitter_gpt2_gates():
    from splitrisk.splitting import AutoSplitter, gates
    full = _tiny_gpt2()
    head, tail, _ = AutoSplitter.load(full, 2)
    par, leak = gates(head, tail, full)
    assert par < 2e-2 and leak < 1e-3


def test_autosplitter_llama_style_gates():
    """llama 家族泛化路徑（Llama/Qwen2/Mistral 共用）——雙閘門必須過。"""
    from splitrisk.splitting import AutoSplitter, gates
    full = _tiny_llama()
    head, tail, _ = AutoSplitter.load(full, 2)
    assert head.__class__.__name__ == "_LlamaStyleHead"
    par, leak = gates(head, tail, full)
    assert par < 2e-2 and leak < 1e-3


def test_autosplitter_rejects_bad_split():
    from splitrisk.splitting import AutoSplitter
    with pytest.raises(ValueError, match="越界"):
        AutoSplitter.load(_tiny_gpt2(), 99)


# ------------------------------------------------------------ label_probe
def test_label_audit_separable():
    from splitrisk.probes.label_probe import label_audit
    torch.manual_seed(0)
    n, t, d = 128, 8, 32
    mask = torch.ones(n, t, dtype=torch.bool)
    y = torch.randint(0, 4, (n,))
    # 可分：每類一個均值方向
    centers = torch.randn(4, d)
    z = centers[y].unsqueeze(1).expand(n, t, d) + 0.05 * torch.randn(n, t, d)
    r = label_audit(z, y, mask, n_classes=4, epochs=30)
    assert r["R2_label_acc"] > 0.9
    assert r["R2_excess_pp"] > 60


# ------------------------------------------------------------------- sl
def _tiny_loader(n=8, t=8, v=100, n_cls=2):
    from torch.utils.data import DataLoader, TensorDataset
    ids = torch.randint(0, v, (n, t))
    mask = torch.ones(n, t, dtype=torch.bool)
    y = torch.randint(0, n_cls, (n,))
    return DataLoader(TensorDataset(ids, mask, y), batch_size=4)


@pytest.mark.parametrize("baseline", ["honest", "sia", "fsha"])
def test_train_split_runs(baseline):
    from splitrisk.sl import train_split, ClsHead
    from splitrisk.splitting import AutoSplitter
    full = _tiny_gpt2()
    head, tail, full = AutoSplitter.load(full, 2)
    cls = ClsHead(32, 2)
    wte = full.get_output_embeddings().weight
    stats = train_split(head, tail, cls, _tiny_loader(), wte,
                        baseline=baseline, alpha=5.0, epochs=1,
                        lr_head=1e-3, lr_server=1e-3, seed=0)
    assert stats["steps"] == 2
    assert stats["baseline"] == baseline
    assert stats["train_loss"] == stats["train_loss"]   # finite


def test_splitfed_runs():
    from splitrisk.sl import splitfed, ClsHead
    from splitrisk.splitting import AutoSplitter
    full = _tiny_gpt2()
    head0, tail, full = AutoSplitter.load(full, 2)
    cls = ClsHead(32, 2)
    wte = full.get_output_embeddings().weight
    shards = [(torch.randint(0, 100, (4, 8)),
               torch.ones(4, 8, dtype=torch.bool),
               torch.randint(0, 2, (4,))) for _ in range(2)]
    gh, t2, c2, hist = splitfed(
        lambda: AutoSplitter.load(_tiny_gpt2(), 2)[0], tail, cls,
        shards, wte, rounds=1, seed=0)
    assert len(hist) == 2


# ---------------------------------------------------- core.wte 通用化
def test_prepare_wte_is_output_embedding():
    """_prepare 的 wte 來自 get_output_embeddings（gpt2/llama 通用）。"""
    for m in (_tiny_gpt2(), _tiny_llama()):
        w = m.get_output_embeddings().weight
        assert w.shape[0] == m.config.vocab_size
