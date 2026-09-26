# Changelog

## 0.3.0 (unreleased — code complete, H20 validation queued)
- **AutoSplitter** (`splitting.py`): generic decoder-only split via the
  official model API — covers llama-family structures (Llama/Qwen2/
  Mistral/Gemma/Yi) alongside gpt2; every split passes mandatory
  parity (Δlogits<2e-2) and causal-leak (Δprefix<1e-3) gates.
- **SL training-loop audit** (`sl.py`, `audit_sl()`): classic SL loop
  (client head → smash → server tail → gradient return, PyTorch 2.x
  explicit detach) with honest / SIA (α·token-CE bridge) / FSHA
  (bridge removed) baselines — exp13 productized. SIA = task normal +
  full recovery → R6.
- **SplitFed** head-FedAvg simulation (`splitfed()`).
- **R2 label-leakage probe** (`label_audit()`): representation→label
  classifier readout with random baseline; label gist survives
  quantization (0.859 vs 0.871) — class-is-sensitive stays RED.
- Multi-seed CE probe averaging (`n_seeds`) per exp18b ~7pp variance.
- Task head normalized: classification head reads tail **hidden
  states** (previous path read vocab logits — task_acc was meaningless).
- Tail wpe zero-copy (official inputs_embeds path re-adds wpe), head
  ln_f/norm moved to tail side (bug-ledger convention enforced by gates).
- Calibration runner for GPU validation: `scripts/h20_stage.py`
  (S1-S8, resumable, per-stage JSON) + docs/H20_RUNBOOK.md.
- `audit_training` accepts explicit split_at/n_train.

## 0.2.1 (2026-09-25)
- gpt2-large anchor corrected to the **exp18b 5-config mean (0.907)**
  (single-seed 0.845 documented as a ~7pp run-to-run outlier); probe
  capacity warning re-attributed to probe training **variance**
  (INCONCLUSIVE refusal mechanism unchanged).
- Added Llama-3.2-1B reference anchor row (exp19 cross-family:
  0.934 / 0.415 / 0.037). `audit()` support list remains gpt2-family;
  the official-API Llama skeleton lands in v0.3.
- LICENSE honesty clauses expanded **8 → 10** (AMBER boundary,
  detection granularity, codebook side channel, closed-source-LLM
  scope, probe variance with mandatory multi-seed averaging).
- README: "across families" claim, 19-experiment count, Llama anchor
  row, ambient ladder 0.317→0.415.

## 0.2.0 (2026-09-18)
- `t5_ce` (top-5 recovery) as the fourth key number; probe-capacity
  warning when t5 - t1 > 10pp (exp18 finding: fixed-capacity probes
  under-discriminate at top-1 on larger models).
- Verdicts refuse instead of guess: INCONCLUSIVE on probe
  under-capacity and on reduced-budget runs (n_train < 500 or
  epochs < 2) — a smoke run must never issue GREEN.
- `probe_hidden` auto-scales to max(2048, 2 x model_hidden);
  768/1024-dim models keep the historical 2048 (legacy readouts
  unchanged by construction).
- Calibration table gains scale/corpus rows (gpt2-medium 0.957/0.356,
  gpt2-large 0.845/0.374 with t5=0.954 annotation, gpt2+dbpedia
  0.977/0.329); `lookup()` raises for uncalibrated (model, corpus).
- Head/Tail rebuilt on the official GPT2Model API over a truncated
  layer list (hand-called GPT2Block breaks on transformers 5.x).
- Honesty clauses (8, non-severable) retained in LICENSE.
- CLI: audit / anchors / vq / sharded / presets.
- Presets: default, healthcare, finance, cross_border.

## 0.1.0
- Initial packaging of the splitaudit-battery probe battery:
  Tier 0-3 probes, VQ cell (k-means init, EMA codebook, dead-code
  revival), four-anchor calibration, report with attacker budget.
