"""Layer-1 dry run: package imports, CLI wiring, probe training on
synthetic tensors (no model download). Catches broken plumbing before
any GPU run."""
import pytest


def test_package_import():
    import splitrisk
    assert splitrisk.__version__
    for name in ("audit", "audit_vq", "audit_sharded", "audit_training",
                 "AuditResult", "PRESETS"):
        assert hasattr(splitrisk, name)


def test_cli_help_and_version(capsys):
    from splitrisk.cli import main
    assert main(["--version"]) == 0
    assert "splitrisk" in capsys.readouterr().out
    assert main([]) == 0            # prints help
    assert "audit" in capsys.readouterr().out


def test_cli_presets(capsys):
    from splitrisk.cli import main
    main(["presets"])
    out = capsys.readouterr().out
    for name in ("default", "healthcare", "finance", "cross_border"):
        assert name in out


def test_cli_audit_rejects_unsupported_model():
    from splitrisk.cli import main
    with pytest.raises(ValueError, match="support list"):
        main(["audit", "--model", "llama-3.2-1b"])


def test_probe_training_smoke():
    """CE + MSE probes on synthetic representations; the CE probe must
    beat the MSE probe on a linearly-decodable toy (sanity of the loss
    wiring, not of the science)."""
    pytest.importorskip("torch")
    import torch
    from splitrisk.probes.train_cf_attacker import (train_probe,
                                                    probe_readout)
    from splitrisk.probes.raw_floor import raw_floor
    torch.manual_seed(0)
    V, D, N, T = 50, 32, 64, 8
    wte = torch.randn(V, D)
    ids = torch.randint(0, V, (N, T))
    z = wte[ids] + 0.01 * torch.randn(N, T, D)   # nearly linearly readable
    ce = train_probe(z, ids, wte, mode="ce", hidden=64, epochs=40,
                     lr=3e-3, batch_size=32, seed=0)
    t1, t5 = probe_readout(ce, z, ids, wte)
    mse = train_probe(z, ids, wte, mode="mse", hidden=64, epochs=40,
                      lr=3e-3, batch_size=32, seed=0)
    t1_mse, _ = probe_readout(mse, z, ids, wte)
    fl = raw_floor(z, ids, wte)
    assert t1 > 0.9, f"CE probe should solve the toy, got {t1:.2f}"
    assert t5 >= t1
    assert fl > 0.9   # toy is linearly readable


def test_reassembler_smoke():
    pytest.importorskip("torch")
    import torch
    from splitrisk.probes.reassembler import train_reassembler
    torch.manual_seed(0)
    V, D, N, T = 50, 32, 64, 8
    wte = torch.randn(V, D)
    ids = torch.randint(0, V, (N, T))
    z = wte[ids] + 0.05 * torch.randn(N, T, D)
    net, t1 = train_reassembler(z, ids, V, wte=wte, n_layers=2,
                                epochs=30, lr=3e-3, batch_size=32, seed=0)
    assert t1 > 0.5, f"reassembler should partially solve toy, got {t1:.2f}"


def test_vq_cell_smoke():
    pytest.importorskip("torch")
    import torch
    from splitrisk.vq import VQCell
    torch.manual_seed(0)
    z = torch.randn(64, 8, 32)
    vq = VQCell(32, k=16, seed=0)
    idx, q = vq.quantize(z)
    assert idx.shape == (64, 8)
    assert q.shape == z.shape
    assert idx.max() < 16
    # quantization must reduce distinct values to <= k
    assert len(q.reshape(-1, 32).unique(dim=0)) <= 16


def test_head_tail_shapes():
    """Head/Tail split wiring on a tiny random GPT-2 config (no
    download): output shapes and layer counts."""
    pytest.importorskip("transformers")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from splitrisk.probes.probe_common import Head, Tail
    torch.manual_seed(0)
    cfg = GPT2Config(n_embd=32, n_layer=4, n_head=4, vocab_size=100,
                     n_positions=32)
    full = GPT2LMHeadModel(cfg)
    head, tail = Head(full, 2), Tail(full, 2)
    assert len(head.inner.h) == 2 and len(tail.inner.h) == 2
    ids = torch.randint(0, 100, (2, 16))
    z = head(ids)
    assert z.shape == (2, 16, 32)
    logits = tail(z)
    assert logits.shape == (2, 16, 100)
