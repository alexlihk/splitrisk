# PROBE7B_GUIDE v2 — 第二期（一次全跑，零互動）

徐博士：老規矩，git pull 之後跑 4 條命令，跑完回傳 4 個 JSON。不需要改任何代碼。

## 前置

```bash
cd /data/xuguangning/work/splitrisk
git pull origin main
```

## 執行（4 條命令，順序不限，可掛著跑）

```bash
python scripts/probe7b_batch2.py --model mistralai/Mistral-7B-v0.1
python scripts/probe7b_batch2.py --model /data/xuguangning/work/splitrisk/models/Llama-3.1-8B/
python scripts/probe7b_batch2.py --model Qwen/Qwen2-1.5B
python scripts/probe7b_batch2.py --model google/gemma-2-2b
```

跑完把 `results/h20/` 下**新生成的 4 個 `probe7b2_*.json`** 發回來即可。

## 說明

- 每條命令跑完會印一行 `★` 開頭的結論，JSON 裡也有 `verdict` 欄位。
- 預計時長：Mistral 約 40 分鐘、Llama-8B 約 60 分鐘、Qwen 約 10 分鐘、Gemma 約 15 分鐘（合計 2–2.5 小時）。
- 每條命令含兩段：註冊預算重跑（5 候選×3 seeds）＋預算階梯（per_dim_norm / clip_z，6/12 epochs），全自動，中間不用管。

## 常見問題

| 症狀 | 處理 |
|---|---|
| 第 3/4 條報 403 或下載失敗 | 把 `--model` 換成你本地已有的對應模型路徑即可 |
| 中途 OOM | 那條命令加 `--n-train 3000` 重跑 |
| 其他報錯 | 直接截圖發來，不要自己修 |

（第一期 probe7b_*.json 的多 seed 欄位有 bug，第二期腳本已修——不用做任何額外操作，直接跑第二期就是對的。）
