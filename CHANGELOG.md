# Changelog

## 0.3.0 (2026-10-01 — H20 校準完成，錨點表跨四家族)

- **H20 錨點校準完成**（8 stage，3 seeds，JSON 在 results/h20/）：
  錨點表新增 Qwen2-1.5B（0.927/0.440/0.005）與 Gemma-2-2b
  （0.852/0.449/0.765，floor 異常高＝gemma 嵌入幾何，已標註待複核）；
  Llama-3.2-1B 實測行入表（0.870/0.415/0.204；P_ctx 與 exp19 的
  0.4148 逐位對位＝跨協議互證）。S4 Mistral-7B / S8 Llama-3.1-8B
  = REVIEW（探針對 7B+ 級 z 不收斂，z 衛生閘已加，按誠實協議暫緩入表）。
- **探針協議修正（v0.3.2/v0.3.3）**：CE probe 訓練在 TRAIN 側表徵
  （battery demo.py 原始協議）、評估在 test 側；bs 對齊 battery（32）。
  修正前 probe 在 test 側自我訓練 4 步＝欠訓練儀器（讀數 0.04）。
  本地驗證：gpt2 縮樣 t1=0.9573/t5=0.9794 落論文區間 0.94-0.97。
- **AutoSplitter gemma adapter**（社區貢獻）：config 切片建真子模型
  （layer_types 對齊 gemma2 sliding-window）＋Gemma 邊界 forward hook
  （z=殘差流激活直接還原，免疫 embed-scaling 位置）＋final
  logit softcapping。本地 random-Gemma2 parity Δ=0.0000。
- **SL 環穩定性**：joint optimizer（head/tail 共享參數 dedup，單一
  AdamW）＋client lr 2e-5（client-lr 掃描：≤5e-5 無損，1e-4 破壞性）。
- **z 衛生閘**：_cache_split_reprs 對 NaN/Inf/極端尺度直接 raise 並附
  統計（7B+ 模型 fp32 前向溢出的診斷數據源）。
- **Q3 象限儀器預覽**（`splitrisk/embed.py`，`splitrisk embed`）：
  閉源 embedding 端點池檢索攻擊測量——cosine top-1 vs 候選池、
  run 內隨機配對地板、三檔 verdict（provisional v0）、誠實邊界
  六條（endpoint+model 綁定保鮮/長度分桶強制/ToS 自查/UNSIGN）。
  首個閉源端點實測：GLM embedding-3 池檢索超額 +0.203（¥0.2 查詢）。
- 0.3.0a0 預告的全部分項（AutoSplitter/sl/splitfed/label_probe/
  h20_stage）如約落地；本地 38 tests green。
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
