# Measurement Methodology

Condensed from the companion paper: *"Split-Learning and Split-Inference
Privacy Defenses Fail Under Strong Attackers: A Metric-Agnostic
Dual-Track Empirical Audit of Representation Leakage"*
(DOI: 10.13140/RG.2.2.32481.67681). 17 experiments + re-audits.

## Probe tiers

All probes share one architecture unless stated: an MLP
dim -> 2048 -> 2048 -> dim with GELU (hidden auto-scaled to
max(2048, 2 x model_hidden) since the exp18 capacity finding). They
differ only in loss and context:

| Tier | Probe | Measures |
|---|---|---|
| 0 | Zero-parameter cosine alignment vs the public embedding matrix | Linear readability (floor) |
| 1 | MSE fresner (reconstruct the pre-transformation representation) | The standard split-learning evaluation instrument — sits at its own floor |
| 2 | CE decoder (cross-entropy on ground-truth token identity via embedding projection) | **Primary instrument, both tracks** |
| 3 | Context-aware bidirectional transformer decoder over the full sequence | Collusive ceiling; capacity ladder |

Adaptive tiers: multi-round co-training (Fresner-style) and
cross-domain retraining (train AG News, evaluate WikiText).

## Metric hierarchy (gameability)

| Metric | Gameable? | Evidence |
|---|---|---|
| Reconstruction MSE | yes | 3.2x inflation, recovery unchanged |
| Attack CE | yes | norm inflation, recovery unchanged |
| Cosine recovery | not in our experiments | oracle-confirmed |

Recovery is scored by cosine retrieval against the embedding matrix:
top-1/top-5 token-identity accuracy over non-padding positions.

## Calibration anchors (four-point fixture)

A probe that cannot separate these anchors is not measuring leakage:

1. **Full information** — unmodified representations (ceiling):
   0.94 (GPT-2/AG News), 0.957 (medium), 0.977 (DBpedia)
2. **Ambient P_ctx** — pretrained LM with no access to z:
   0.317 / 0.356 (medium) / 0.329 (DBpedia). Scale- and
   corpus-dependent: per-deployment recalibration is mandatory.
3. **Zero information** — collapsed 1-2 code VQ codebook: 0.030
   (unigram mode floor)
4. **Two-probe closure** — two independent context-aware decoders
   agree on the 9-bit ceiling: 0.469 vs 0.475

Reproducibility: independent sessions agree to 0.14-0.32pp on probe
top-1 and exactly on deterministic floors.

## Core findings the tool encodes

1. **The attacker's loss sets the measurement**: MSE probe 0.0022 vs
   CE probe 0.9758 on identical representations (layer 8) — the
   instrument gap, measured in our setting.
2. **Two clusters, empty interior**: utility-preserving defenses leak
   (>= 0.93); information-destroying defenses kill the task
   (PPL 29-746x). One narrow reproducible region: classification at
   8-9 bits (acc ratio 0.958-0.974, n=6 seeds) — threat-model
   dependent, ceiling open, label gist NOT compressed.
3. **Depth is obfuscation, not privacy**: floor falls 0.069 -> 0.005
   with depth; the CE probe stays flat.
4. **Sharding is orthogonal**: bits set the collusive ceiling
   (0.469 with 9-bit codes across 3 shards vs 0.475 unsharded);
   shard count only lowers the single-shard view, largely via
   attacker-capacity dilution.
5. **Training track, three signatures**: FSHA double-loss (task
   0.291 / recovery 0.756) — easy; SIA stealth (task 0.873 normal /
   recovery 0.9695) — only the CE probe detects it; R-M == S-M
   (0.9725 vs 0.9717) — leakage set by hijack intensity alpha, not
   training mode.
6. **Probe capacity matters at scale** (exp18): GPT-2 large with a
   2048-hidden probe: t1 0.845 but t5 0.954 — the probe, not the
   representation, is the bottleneck. Hence the t5 field, the
   capacity warning, and auto-scaled probe_hidden.

## Engineering rules (from the bug ledger)

- Never hand-call decoder blocks; always the official model API, then
  truncate the layer list.
- Cosine retrieval, never MSE/CE, as the honest readout.
- Padding convention must match the evaluation filter (the honeypot
  false-positive lesson).
- The attacker never trains on the evaluation set.
- Results are saved as JSON immediately after each run.
- Deterministic seeds: split assets 1000+s (CLS) / 2000+s (LM);
  attacker inits fixed per cell; VQ seeds 9000+K+100*seed.
