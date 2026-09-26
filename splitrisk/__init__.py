"""SplitRisk — audit what your split model leaks.

Open calibration layer (SA-RI L1). Measurement tool, not a defense.
"""
from .report import AuditResult
from .presets import PRESETS

__version__ = "0.2.1"

__all__ = ["audit", "audit_vq", "audit_sharded", "audit_training",
           "audit_sl", "audit_custom_head", "AuditResult", "PRESETS",
           "__version__"]


def audit(model="gpt2", split_at=6, task="classification", data="ag_news",
          n_train=5000, n_test=1000, seed=1006, probe_hidden=None,
          epochs=2, lr=1e-3, device=None):
    """Full audit of a split deployment. See docs/USERGUIDE.md."""
    from .core import audit as _audit
    return _audit(model=model, split_at=split_at, task=task, data=data,
                  n_train=n_train, n_test=n_test, seed=seed,
                  probe_hidden=probe_hidden, epochs=epochs, lr=lr,
                  device=device)


def audit_vq(model="gpt2", split_at=6, task="classification",
             data="ag_news", codebook_sizes=(256, 512, 1024),
             n_train=5000, n_test=1000, seed=1006, device=None):
    """VQ bitrate sweep — find the quantization seam, if any."""
    from .core import audit_vq as _vq
    return _vq(model=model, split_at=split_at, task=task, data=data,
               codebook_sizes=tuple(codebook_sizes), n_train=n_train,
               n_test=n_test, seed=seed, device=device)


def audit_sharded(model="gpt2", split_at=6, data="ag_news", n_shards=3,
                  vq_bits=None, n_train=5000, n_test=1000, seed=1006,
                  device=None):
    """Sharding audit: single-shard view + collusive ceiling max_C R(C)."""
    from .core import audit_sharded as _sh
    return _sh(model=model, split_at=split_at, data=data,
               n_shards=n_shards, vq_bits=vq_bits, n_train=n_train,
               n_test=n_test, seed=seed, device=device)


def audit_training(delivered_model, reference_data, task="classification",
                   n_test=1000, seed=1006, device=None, split_at=6,
                   n_train=None):
    """Training-track audit: CE probe on a delivered model version.

    SIA-style stealth hijacks pass task-metric acceptance; only the
    CE probe detects them (R6 axis)."""
    from .core import audit_training as _tr
    return _tr(delivered_model=delivered_model,
               reference_data=reference_data, task=task, n_test=n_test,
               seed=seed, device=device, split_at=split_at,
               n_train=n_train)


def audit_sl(model="gpt2", split_at=6, data="ag_news", baseline="honest",
             alpha=5.0, sl_epochs=1, n_train=5000, n_test=1000, seed=1006,
             device=None, n_seeds=1):
    """SL 訓練環審計（exp13 產品化）：honest/sia/fsha 三 baseline 對照。
    SIA = 任務正常而表徵滿洩漏（唯 CE 探針可抓，R6 軸）。"""
    from .core import audit_sl as _sl
    return _sl(model=model, split_at=split_at, data=data, baseline=baseline,
               alpha=alpha, sl_epochs=sl_epochs, n_train=n_train,
               n_test=n_test, seed=seed, device=device, n_seeds=n_seeds)


def audit_custom_head(head, split_at, task="classification", data=None,
                      labels=None, n_train=5000, n_test=1000, seed=1006,
                      probe_hidden=None, device=None):
    """Audit a bring-your-own head (see probes.probe_common.Head for
    the interface: forward(input_ids, attention_mask) -> hidden states)."""
    from .core import audit as _audit
    return _audit(model=None, split_at=split_at, task=task, data=data,
                  head=head, labels=labels, n_train=n_train,
                  n_test=n_test, seed=seed, probe_hidden=probe_hidden,
                  device=device)
