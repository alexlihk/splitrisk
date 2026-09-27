# SplitRisk 🔓

**Audit what your split model leaks — before someone else does.**

[![CI](https://github.com/alexlihk/splitrisk/actions/workflows/ci.yml/badge.svg)]
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue)]
[![Paper](https://img.shields.io/badge/Paper-RG.DOI-orange)]

> Product name: **SplitRisk** (code name during development: *splitHack*).
> SplitRisk is the packaged, open calibration layer of the
> [splitaudit-battery](https://github.com/alexlihk/splitaudit-battery)
> research artifact — it is a measurement tool, not new science.

## The SA-RI specification

Formal definitions — Safety(S;budget), six axes R1-R6, three-tier routing, 8×N cloud matrix, compliance mapping, training-track signatures, ten honesty clauses — are in the [Technical White Paper v3.3](docs/WHITEPAPER.md).

## What is this?

When you deploy a model in split mode (early layers on client, rest on
server), the intermediate representations you transmit **leak your input
data**. Our 19-experiment audit shows:

- Utility-preserving representations leak **93–99.9%** of input tokens
  **across families** (GPT-2 small/medium/large and Llama-3.2-1B;
  AG News/DBpedia/clinical notes)
- Standard MSE-based privacy metrics are **gameable** (3.2× inflation
  with zero real protection)
- The only honest metric is **cosine-recovery** under a **CE-trained
  probe** with a **stated attacker budget**

SplitRisk gives you the probe battery to measure this yourself.

## Quick start

```bash
pip install splitrisk
```

```python
from splitrisk import audit

result = audit(
    model="gpt2",           # or "gpt2-medium", "gpt2-large"
    split_at=6,             # client keeps layers 0-5
    task="classification",  # or "lm"
    data="ag_news",         # or your own dataset
)

print(f"Token recovery (strong attacker): {result.t1_ce:.1%}")
print(f"Top-5 recovery:                  {result.t5_ce:.1%}")
print(f"Linear floor:                    {result.floor:.1%}")
print(f"Ambient baseline:                {result.pctx:.1%}")
print(f"Verdict: {result.verdict}")
# → Token recovery: 94.2%
#   Top-5 recovery: 95.4%
#   Linear floor:   0.8%
#   Ambient:        31.7%
#   Verdict: RED — full leakage, do not egress
```

Or from the command line:

```bash
splitrisk audit --model gpt2 --split 6 --task cls --data ag_news
splitrisk anchors --model gpt2     # 4-anchor calibration check
splitrisk vq --model gpt2 --k 256 512 1024
```

## The four numbers that matter

| Number | Meaning | How to read it |
|---|---|---|
| `t1_ce` | Top-1 token recovery under the CE probe | >85% = RED; 40–60% = AMBER; <ambient+5pp = GREEN |
| `t5_ce` | Top-5 recovery — **probe capacity check** | t5 − t1 > 10pp ⇒ the probe is under-capacity; the report is INCONCLUSIVE, increase `probe_hidden` |
| `floor` | Zero-parameter cosine alignment | Low floor + high t1 = info is there but not linear |
| `pctx`  | Ambient (attacker without your deployment) | The zero-point of "excess leakage" |

**Excess leakage = t1_ce − pctx.** This is the number your compliance
committee needs.

## Calibration anchors

Any fork that doesn't reproduce these is not calibrated. Anchors are
**model × corpus** dependent — per-deployment calibration is mandatory;
these are reference points, not universal constants.

| Anchor | Value |
|---|---|
| Full information (GPT-2, AG News, s6 CLS) | 0.94 top-1 |
| Full information (GPT-2-medium) / (GPT-2-large, exp18b 5-config mean) | 0.957 / 0.907 |
| Full information (Llama-3.2-1B, cross-family) | 0.934 |
| Full information (GPT-2, DBpedia) | 0.977 |
| Ambient (AG News / medium / large / DBpedia / Llama) | 0.317 / 0.356 / 0.374 / 0.329 / 0.415 |
| Zero information (collapsed codebook) | 0.030 |
| Two-probe closure (9-bit VQ) | 0.469 ≈ 0.475 |

## Coverage (expanding — see [ROADMAP.md](ROADMAP.md))

| | v0.2.1 (live) | v0.3.0 (validating on H20) |
|---|---|---|
| Inference-track audit | ✅ gpt2 family | ✅ + llama-family models (Llama/Qwen2/Mistral/Gemma — parity-gated `AutoSplitter`) |
| SL training-loop audit (honest / SIA / FSHA injection) | — | ✅ `audit_sl()` |
| SplitFed simulation | — | ✅ `splitfed()` |
| R2 label-leakage probe | — | ✅ `label_audit()` |
| Multi-seed protocol (exp18b ~7pp variance) | manual | ✅ `n_seeds` built-in |

Vision/tabular modality is research, not a feature — we don't ship
recovery metrics without an oracle check.

## What SplitRisk does NOT do

- It does **not** defend. It measures.
- It does **not** promise security. It quantifies risk.
- RED verdict solutions: TEE, local inference, or don't egress.

## Certification (L2 preview)

Any saved audit result can be turned into an SA-RI certification
record (schema v1.1) — cert number (content-addressed), attacker
budget, audit chain, legal notice, freshness label, and the ten
non-severable honesty clauses. Results without a decisive verdict
(INCONCLUSIVE) are **refused** — refusing to guess is the product.

```bash
splitrisk audit --model gpt2 --split 6 --task cls --output audit.json
splitrisk certify --input audit.json --org "ACME" --out-prefix cert
# → cert.cert.json + cert.cert.md   (or [REFUSED] + reason)
```

Production signing is the SplitAudit commercial layer (not in this
repo); the L1 preview marks records `UNSIGN`.

## Commercial version

SplitRisk is the open calibration layer. **SplitAudit** adds:
signed certification reports, subscription monitoring, attacker-capacity
updates, compliance templates (GDPR DPIA / PIPL / DORA / EU AI Act).

→ [Contact for enterprise](mailto:alexlihk@hotmail.com)

## Citation

See [CITATION.cff](CITATION.cff). Companion paper: *"Split-Learning and
Split-Inference Privacy Defenses Fail Under Strong Attackers"*
(DOI: 10.13140/RG.2.2.32481.67681).

## License

BSD-3-Clause. The certification engine, subscription service, and
anchor-update stream are NOT part of this repository.
