# Threat Model

## Two tracks, two adversaries

SplitRisk audits **two distinct deployment patterns** that are
commonly conflated under "split learning":

### Inference track — honest-but-curious server

- Client runs the encoder (embeddings + layers 1..s) locally;
  transmits hidden states z = f_{1:s}(x) to a server holding the rest.
- **Adversary**: the server operator, who observes z and may train
  arbitrary decoders on any public corpus.
- **Defender controls** (offline choices only): split depth, encoder
  fine-tuning, a transformation g() before transmission, quantization
  bitrate, optional sharding of z across N servers (by position,
  channel, or layer).

### Training track — malicious server

- Client outsources training to a server that controls the training
  process (FSHA-style gradient hijacking, SIA-style stealth hijacking
  with a main-task bridge).
- **Adversary**: the server, which may manipulate training.
- **Defender observes** only (i) the delivered model and (ii) any
  task-metric or probe readout from it. No training-time gradients,
  no parameter drift.

## What the attacker is modeled to have

| Asset | Assumed? |
|---|---|
| Public pretrained LM (any size) | Yes — ambient anchor P_ctx measures exactly this prior |
| The transmitted z (or a shard of it) | Yes (inference track) |
| Knowledge of your tokenizer/model family | Yes (public) |
| Labeled plaintext from your domain | Partially — probes train on public/attacker corpora; cross-domain retraining is part of the protocol |
| The defense's secret parameters (projection keys etc.) | No — but known-plaintext attacks recover them (exp4) |

The **attacker budget vector** (probe architecture, hidden size,
training samples, epochs, lr) is recorded with every readout and is a
protocol field, not a footnote.

## Out of scope

- Gradient leakage during training (we audit only what is detectable
  from the delivered model or the transmitted representation).
- Parameter drift / model stealing.
- Membership inference (orthogonal; see ML Privacy Meter).
- Side channels (timing, packet sizes).

## Honest limitations

1. Results are measured at GPT-2 scale on AG News/DBpedia/synthetic
   clinical notes; the impossibility map is empirical, not a proof.
2. The context-aware ceiling above 0.47 is open; unused attacker
   advantages (pretrained decoder backbones, larger plaintext
   budgets) likely raise it.
3. Entity-level recovery is measured on synthetic notes only.
4. The MSE/CE instrument gap magnitude (444x in our setting) is not
   claimed as a universal constant.
