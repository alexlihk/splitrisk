"""Dimension assertions — catches the v1 768/1280 bug class."""
import pytest

MODEL_CONFIGS = [
    ("gpt2", 768, 12),
    ("gpt2-medium", 1024, 24),
    ("gpt2-large", 1280, 36),
]


def test_reconnet_output_dim():
    """The v1 bug: ReconNet output 768 while large wte is 1280."""
    pytest.importorskip("torch")
    import torch
    from splitrisk.probes.train_cf_attacker import ReconNet
    for dim in [768, 1024, 1280]:
        net = ReconNet(dim=dim)
        out = net(torch.randn(2, 4, dim))
        assert out.shape[-1] == dim, \
            f"ReconNet output {out.shape[-1]} != input dim {dim}"


def test_wte_alignment():
    """CE probe output must match wte's second dimension."""
    pytest.importorskip("torch")
    import torch
    from splitrisk.probes.train_cf_attacker import ReconNet
    for dim in [768, 1024, 1280]:
        net = ReconNet(dim=dim)
        out = net(torch.randn(2, 4, dim))
        wte_mock = torch.randn(100, dim)
        logits = out @ wte_mock.t()
        assert logits.shape == (2, 4, 100)


@pytest.mark.slow
@pytest.mark.parametrize("model,expected_dim,expected_layers",
                         MODEL_CONFIGS)
def test_dims(model, expected_dim, expected_layers):
    pytest.importorskip("transformers")
    from transformers import GPT2Config
    cfg = GPT2Config.from_pretrained(model)
    assert cfg.n_embd == expected_dim
    assert cfg.n_layer == expected_layers
