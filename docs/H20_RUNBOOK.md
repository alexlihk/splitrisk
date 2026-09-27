# H20 RUNBOOK — SplitRisk v0.3 校準/驗證排程（代跑指引）
2026-09-25 · 對象：GPU 代跑同學 · 全程可斷點續跑 · 支持單卡順序或三卡並行
總原則：**每個 stage 寫完 JSON 才算數**；錯了貼 traceback 給我們，不要自行改代碼。

---

## 0.5 三卡派工（3×4090，24GB 版）★已驗證的分配

| 卡 | 指令（一行一段，可同時開三個終端） | 預估 | 顯存依據 |
|---|---|---|---|
| GPU0 | `CUDA_VISIBLE_DEVICES=0 python scripts/h20_stage.py --stage S1 && CUDA_VISIBLE_DEVICES=0 python scripts/h20_stage.py --stage S5` | 10min + 2.5h | gpt2 級 <4GB |
| GPU1 | `CUDA_VISIBLE_DEVICES=1 python scripts/h20_stage.py --stage S2` | ~1.5h | Llama-1B full-FT 實測 5.4GB（T4 驗證過） |
| GPU2 | `CUDA_VISIBLE_DEVICES=2 python scripts/h20_stage.py --stage S3` | ~1.5h | Qwen2-1.5B 類似量級 |
| 接力 | GPU0 空出後：`--stage S6`；GPU1 空出後：`--stage S7` | 2.5h / 2.5h | Gemma-2-2b ~14GB ✅ |

**顯存界線（為什麼 S4/S8 不上 4090）**：Mistral-7B 權重 bf16 已 14.4GB，
full-FT（8 層 ≈2B 可訓參數）+ AdamW fp32 動量 ≈ **35GB > 24GB**。
Llama-3.1-8B 同理 ~36GB。
→ **S4 / S8 留給 H20（96GB）**，或等我們加 LoRA 變體再上 4090。

三卡並行總牆鐘：**~3 小時跑完 S1/S2/S3/S5/S6/S7 六段**（含接力）。

前置注意：S2（Llama）是 gated repo——跑前 `huggingface-cli login`，
帳號需已申請 Meta 授權（網頁申請，通常即時批）。S3/S5/S6/S7 無門禁，
可先跑。HF 匿名下載限流時設 `HF_TOKEN`。

---

## 0. 環境準備（一次性，~10 分鐘）

```bash
git clone https://github.com/alexlihk/splitrisk && cd splitrisk
conda create -n sr python=3.11 -y && conda activate sr
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e . datasets
huggingface-cli login        # Llama 為 gated repo，需要 token
```
自檢：`python -m pytest -m "not slow" -q` → 應全綠（~1 分鐘，CPU 即可）。

## 1. 排程總表（單卡順序，可中斷，每段獨立）

| 序 | 指令 | 內容 | 預估 | 產出 JSON |
|---|---|---|---|---|
| S1 | `python scripts/h20_stage.py --stage S1` | AutoSplitter 閘門冒煙（真 gpt2 權重） | 10min | S1_gpt2_ag_news.json |
| S2 | `--stage S2` | **錨點校準 Llama-3.2-1B**（3 seed） | ~2h | S2_llama-3.2-1b_ag_news.json |
| S3 | `--stage S3` | 錨點校準 Qwen2-1.5B（3 seed） | ~2h | S3_Qwen-Qwen2-1.5B_ag_news.json |
| S5 | `--stage S5` | **SL 三 baseline 對照**（exp13 復現，W4 驗收） | ~2.5h | S5_gpt2_ag_news.json |
| S6 | `--stage S6` | SplitFed 模擬（K=4，2 輪） | ~2.5h | S6_gpt2_ag_news.json |
| S4 | `--stage S4` | 錨點校準 Mistral-7B（可後排） | ~5h | S4_mistralai-Mistral-7B-v0.3_ag_news.json |
| S7 | `--stage S7` | 錨點校準 Gemma-2-2b（可選） | ~2.5h | S7_...json |
| S8 | `--stage S8` | 錨點校準 Llama-3.1-8B（可選，大階梯點） | ~6h | S8_...json |

**建議順序：S1 → S2 → S5 → S3 → S6 → S4 → (S7/S8 按卡時餘量)。**
S1/S2/S5 完成＝v0.3 可發（llama 支持成立＋SL 環驗證）；S3+ 加一個家族；S4/S7/S8 加寬階梯。

## 2. 每個 stage 的驗收判準（結果 JSON 裏有 verdict 欄）

- **S1**: `verdict: PASS`（parity Δlogits<2e-2、causal Δ<1e-3）。FAIL＝骨架病變，停止並回報
- **S2/S3/S4/S7/S8**: `verdict: PASS`（t1 mean>0.85）。產出可直接寫入 `calibration/anchors.py` 的錨點行
- **S5**: 三檢查全 True——`SIA_stealth`（任務正常+t1 高）、`FSHA_collapse`（任務崩塌）、`honest_high_recovery`。對照 exp13：honest t1≈0.94/acc≈0.89、SIA t1≈0.97/acc≈0.87、FSHA acc≈0.29
- **S6**: `PASS`（SplitFed 全局 t1>0.8）

## 3. 常見問題

| 症狀 | 處置 |
|---|---|
| OOM | 重跑同一指令（腳本無狀態依賴）；仍 OOM 則回報，我們降 n_train |
| `HF_TOKEN` 401 | `huggingface-cli login` 未做或 token 無 llama 權限（gated repo 需自行申請 meta 授權） |
| 中斷 | 直接重跑同一指令——done 的 stage 自動跳過 |
| 磁碟 | results/h20/*.json 每個 <100KB；模型快取佔 ~30GB |
| 下載慢 | 可設 `HF_HUB_ENABLE_HF_TRANSFER=1`（可選） |

## 4. 跑完交付

把 `results/h20/` 整個資料夾（全部 JSON）打包發回。我們收到的動作：
1. 錨點表寫入＋版本升 v0.3.0
2. S5 對照 exp13 的偏差寫進論文勘誤/附錄
3. 更新 README coverage 矩陣的實測狀態

**不要做**：不要改包內代碼；不要把結果貼進 issues（JSON 直接傳）。
