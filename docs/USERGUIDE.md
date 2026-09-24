# SplitRisk User Guide

## Table of Contents
1. [Installation](#installation)
2. [Your First Audit](#your-first-audit)
3. [Understanding the Report](#understanding-the-report)
4. [Auditing Your Own Model](#auditing-your-own-model)
5. [VQ Quantization Audit](#vq-quantization-audit)
6. [Sharding / Multi-Server Audit](#sharding--multi-server-audit)
7. [Training-Track Audit (SIA Detection)](#training-track-audit)
8. [Custom Datasets](#custom-datasets)
9. [Verdict Tiers and Presets](#verdict-tiers-and-presets)
10. [FAQ](#faq)

---

## Installation

```bash
pip install splitrisk

# Requirements: PyTorch >=2.0, transformers >=4.40, one GPU
# (>=4GB for GPT-2 small; >=16GB for large; CPU works for smoke tests)
```

---

## Your First Audit

```python
from splitrisk import audit

result = audit(model="gpt2", split_at=6, task="classification",
               data="ag_news", n_train=5000, n_test=1000, seed=1006)
print(result.pretty())
```

This runs the full pipeline (~20 min on a T4):
1. Fine-tunes head+tail on your task (1 epoch default)
2. Caches split-point representations
3. Trains CE probe (Tier 2) + MSE probe (Tier 1) + floor (Tier 0)
4. Measures ambient baseline (P_ctx)
5. Generates the verdict (with attacker budget attached)

Or from the CLI:

```bash
splitrisk audit --model gpt2 --split 6 --task cls --data ag_news \
    --output report.json
```

---

## Understanding the Report

```
==============================================================
  SplitRisk Audit Report
  Model: gpt2 | Split: 6 | Task: classification | Mode: audit | Seed: 1006
--------------------------------------------------------------
  MEASUREMENTS
  +- Token recovery t1 (CE probe):    94.2%  <- primary
  +- Token recovery t5 (CE probe):    95.4%  <- capacity check
  +- Token recovery (MSE probe):       0.7%  <- instrument floor
  +- Linear floor (Tier 0):            0.8%
  +- Ambient (P_ctx):                 31.7%
  +- Task accuracy:                   89.1%
  Excess leakage (t1 - pctx):         +62.5pp
  ATTACKER BUDGET
  +- Probe: MLP 768->2048->2048->768 (GELU) (hidden=2048)
  +- Training: 5000 samples, 2 epochs, lr 0.001
  CAVEATS
  ! Attacker ceiling is open: re-audit quarterly ...
  VERDICT: RED
==============================================================
```

### The verdict tiers

| Tier | Condition | Action |
|---|---|---|
| GREEN | t1 < ambient + 5pp | Safe to egress (rare — only in task-dead zone) |
| AMBER | 0.4 <= t1 <= 0.6 AND task alive | Low-sensitivity classification + quarterly re-audit |
| RED | t1 >= 0.85 | TEE / local / don't egress |
| INCONCLUSIVE | t5 - t1 > 10pp | Probe under-capacity — re-run with larger `probe_hidden` |

A readout without its attacker budget is not interpretable; the budget
is always attached.

---

## Auditing Your Own Model

```python
from splitrisk import audit_custom_head

# Bring your own head (must follow the Head interface:
# forward(input_ids, attention_mask) -> hidden states (N, T, D))
result = audit_custom_head(
    head=my_head,
    split_at=6,
    task="classification",
    data="my_org/my_dataset",
    text_column="content",
    label_column="category",
)
```

Built-in calibrated models: `gpt2`, `gpt2-medium`, `gpt2-large`.
Other architectures: pass a custom head (per-deployment anchor
calibration is mandatory — see below).

---

## VQ Quantization Audit

Test whether quantization creates a "seam" (the one narrow region
where utility survives AND leakage drops):

```python
from splitrisk import audit_vq

result = audit_vq(
    model="gpt2", split_at=6, task="classification", data="ag_news",
    codebook_sizes=[256, 512, 1024, 4096],
)
# K=256 (8 bits): acc_r=0.958, t1=0.309  <- AMBER candidate
# K=512 (9 bits): acc_r=0.974, t1=0.359  <- AMBER
```

**Warning**: the 9-bit seam is threat-model-dependent. The ceiling
rises with attacker capacity (0.36 -> 0.42 -> 0.47 at 3/6/more decoder
layers, not converged). Label leakage is NOT reduced (85.9% vs 87.1%).
PHI is disproportionately suppressed (ratio 0.625) — favorable but
mechanism-level only (synthetic clinical notes).

---

## Sharding / Multi-Server Audit

```python
from splitrisk import audit_sharded

result = audit_sharded(
    model="gpt2", split_at=6, data="ag_news",
    n_shards=3,               # position sharding
    vq_bits=9,                # optional: quantize before sharding
)
```

Key finding: shard count does NOT lower the collusive ceiling.
Bits set the ceiling. N only lowers the single-shard view (and much of
that is attacker-capacity dilution, not privacy). Report
`max_C R(C)` — never the single-shard number alone.

---

## Training-Track Audit

Detect whether a delivered model has been hijacked during training:

```python
from splitrisk import audit_training

result = audit_training(
    delivered_model=my_model,   # the model your supplier gave you
    reference_data=eval_set,    # held-out data you control
)
```

Three signatures:
- **FSHA (double-loss)**: task collapses, recovery drops -> easy to catch
- **SIA (stealth)**: task NORMAL, recovery ~97% -> ONLY the CE probe catches it
- **R-M == S-M**: leakage set by hijack intensity alpha, not training mode

This is the R6 axis: task metrics are blind to SIA-style stealth
hijacks. If task accuracy is normal but t1 is high, refuse the model
version.

---

## Custom Datasets

```python
from splitrisk import audit

result = audit(
    model="gpt2-medium",
    split_at=12,              # medium has 24 layers
    task="classification",
    data="my_org/my_dataset", # any HF dataset id
    text_column="content",
    label_column="category",
)
```

**Important**: the ambient anchor (P_ctx) is corpus- and
model-dependent (0.317 small/AG News, 0.356 medium, 0.329 DBpedia).
Your report includes a per-corpus ambient measurement. For any
(model, corpus) pair not in the calibration table, run
`splitrisk anchors` first and record your own reference values.

---

## Verdict Tiers and Presets

Thresholds are the institution's decision (measurement / decision /
supervision separation). Built-in presets:

```bash
splitrisk presets                 # list all
splitrisk presets --name finance  # one preset
```

- `default` — baseline tiers
- `healthcare` — category-is-sensitive = RED; no AMBER offered
- `finance` — label axis (R2) gated alongside token recovery
- `cross_border` — GDPR TIA / PIPL shape with freshness label

---

## FAQ

**Q: My MSE probe reads 0.5% — am I safe?**
A: No. The MSE probe sits at the measurement floor. Run the CE probe.
If it reads 90%+, your representations are fully recoverable. The
444x instrument gap (measured in our setting) is the entire point.

**Q: I quantized to 4 bits and t1 dropped to 1% — safe?**
A: Check your task metrics first. 4-bit typically kills the task
(41x perplexity on LM). If the task is dead, the "safety" is
irrelevant. Use `audit_vq` to find your seam.

**Q: t1 is 85% but t5 is 95% — is my model safe?**
A: The report will say INCONCLUSIVE. The gap means the probe's top-1
discrimination is insufficient for this model's representation
dimension. The information IS there (t5=95%). Increase probe capacity
(`audit(probe_hidden=4096)`) or report both numbers. Never report t1
alone in this regime.

**Q: How often should I re-audit?**
A: The attacker-capacity ceiling rises over time (more decoder layers
= more extraction; 0.36 -> 0.42 -> 0.47 and not converged). Quarterly
plus event-triggered (new attack papers, model updates) is the current
recommendation. The commercial SplitAudit subscription automates this.

**Q: Can SplitRisk defend my model?**
A: No. Nothing in our 17-experiment audit found a software defense
that preserves utility AND blocks reconstruction. SplitRisk measures
risk; it does not eliminate it. RED means TEE, local, or don't egress.

**Q: What's the difference between SplitRisk and SplitAudit?**
A: SplitRisk is the open calibration layer (BSD, this repo).
SplitAudit is the commercial product: signed reports, subscription
monitoring, compliance templates. Think ZAP -> Burp Suite Pro.

**Q: Why did my audit refuse to give a verdict?**
A: Either the probe is under-capacity (t5 - t1 > 10pp; increase
`probe_hidden`) or the (model, corpus) pair is not calibrated
(run `splitrisk anchors`). Refusing to guess is the product working
as designed.
