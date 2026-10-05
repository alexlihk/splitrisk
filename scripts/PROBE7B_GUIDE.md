# 7B 探針再校準 — 執行指引（v1.0，2026-10-02）

趙博：兩個命令，跑完回傳 JSON。不需要改任何代碼。

## 前置

```bash
cd /data/xuguangning/work/splitrisk
git pull origin main
```

## 執行（兩條命令）

```bash
# S4: Mistral-7B（~2-3h，bf16 自動）
python scripts/probe7b_batch.py --model mistralai/Mistral-7B-v0.1

# S8: Llama-3.1-8B（本地路徑，~2-3h）
python scripts/probe7b_batch.py --model /data/xuguangning/work/splitrisk/models/Llama-3.1-8B/
```

**建議先跑縮樣驗證**（~15 分鐘，確認管道通）：

```bash
python scripts/probe7b_batch.py --model mistralai/Mistral-7B-v0.1 \
  --n-train 500 --n-test 200
```

縮樣跑通再跑全量。

## 看結果

每個候選會打印 `PASS` 或 `FAIL`：
- **任一候選 PASS**（t1 > P_ctx + 0.15）= 7B 錨點分離成功 → 回傳 JSON
- **全部 FAIL** = 儀器邊界確認 → 也回傳 JSON（診斷數據）

兩個都跑完後，把這兩個檔案發回來：
```
results/h20/probe7b_mistralai-Mistral-7B-v0.1.json
results/h20/probe7b_-data-xuguangning-work-splitrisk-models-Llama-3.1-8B-.json
```

## 技術細節（不需要動）

- 模型自動 bf16 加載（CUDA）/ fp32（CPU）
- 5 個候選：baseline / per-dim 標準化 / clip-z / sink-token 排除 / 組合
- 每候選 3 seeds，錨點分離判定內建（t1 > P_ctx + 0.15）
- 縮樣模式：`--n-train 500 --n-test 200`
- GPU 顯存：Mistral-7B bf16 ~14.5GB（4090/H20 均可）

## 異常處理

- CUDA OOM → 加 `--n-train 1000`（減半訓練量）
- HF 401/403 → `export HF_TOKEN=<token>`
- 任何其他錯誤 → 截圖發回，不要自己修
