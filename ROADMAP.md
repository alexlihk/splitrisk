# SplitRisk Roadmap — split-training coverage expansion

> **Status (2026-09-25): code complete, H20 validation runs queued.**
> This roadmap is the public preview of v0.3.0. All new modules are in
> `main` behind the same gates as everything else (parity + causal-leak
> checks, multi-seed protocol, refuse-to-guess verdicts).

## Why

Our 19-experiment audit covered split **inference** exhaustively and
split **training** through three hijack signatures (FSHA / SIA / R-M).
But the package could only *measure* the training track on delivered
GPT-2-family models. Real split-training deployments are broader:
classic SL loops, SplitFed, and decoder-only families beyond GPT-2
(Llama, Qwen2, Mistral, Gemma). v0.3 closes that gap.

## Coverage matrix

| Capability | v0.2.1 (live) | v0.3.0 (code done, validating) | Research (not scheduled) |
|---|---|---|---|
| Split inference audit (CE/MSE/floor/ambient) | ✅ gpt2 ×3 | ✅ same | — |
| VQ seam sweep / sharding audit | ✅ | ✅ | — |
| Delivered-model audit (R6, hijack signatures) | ✅ | ✅ + explicit split/budget params | — |
| **SL training-loop audit** (honest/SIA/FSHA injection) | — | ✅ `audit_sl()` — exp13 productized | — |
| **SplitFed simulation** (K clients, head-FedAvg) | — | ✅ `splitfed()` | full SplitFed w/ local models |
| **R2 label-leakage probe** | — | ✅ `label_audit()` | ExPLoit-class attacks |
| **Model families** | gpt2/medium/large | + llama-family via `AutoSplitter` (Llama/Qwen2/Mistral/Gemma/Yi — official-API truncation, parity-gated) | encoder/multi-modal |
| Multi-seed protocol (exp18b ~7pp variance) | manual | ✅ `n_seeds` built-in | — |
| Modality | text | text | **vision/tabular — recovery metric needs redefinition (research)** |

## What runs next (H20 schedule)

Sequential, single-GPU, resumable — see
[docs/H20_RUNBOOK.md](docs/H20_RUNBOOK.md):

| Stage | Content | ~Time |
|---|---|---|
| S1 | AutoSplitter gates on real checkpoints | 10 min |
| S2 | Anchor calibration **Llama-3.2-1B** (3 seeds) | 2 h |
| S5 | SL three-baseline audit (**exp13 replication**) | 2.5 h |
| S3 | Anchor calibration **Qwen2-1.5B** | 2 h |
| S6 | SplitFed simulation (K=4) | 2.5 h |
| S4/S7/S8 | Mistral-7B / Gemma-2-2b / Llama-3.1-8B anchors (ladder width) | 2.5-6 h each |

Every stage writes its own JSON with a prereg header and PASS/FAIL
verdict. Anchors that PASS go into the public calibration table and
become **signature-eligible** (per-deployment calibration is mandatory —
uncalibrated models get readouts but no verdict, by design).

## Non-negotiables (unchanged)

- Cosine recovery under a stated attacker budget is the only honest readout.
- Refuse rather than guess: uncalibrated / under-capacity / reduced-budget → INCONCLUSIVE or refusal.
- Honesty clauses (ten, non-severable) travel with every artifact.
- Vision/tabular modality is research, not a feature: we will not ship a
  recovery metric we have not validated against an oracle.

## Anchor points that don't move

full-info 0.94 (124M) · ambient ladder 0.317→0.356→0.374→0.415 ·
zero-info 0.030 · two-probe closure 0.469≈0.475 · exp18b probe variance
~7pp · exp19 cross-family t1 0.934. Forks that can't reproduce these
aren't calibrated.
