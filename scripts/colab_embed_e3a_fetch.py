#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_embed_e3a_fetch.py — E3-a: 真實 embedding 端點查詢落盤（PC 上跑）
v1 (2026-09-29): GLM embedding-3 = 首個公開 Q3 案例（「審計自家生態」）

預註冊：
  targets: ag_news test, rng(1006) 抽 n 條（與 E1 同 500 條），截 160 chars（同桶）
  pool:    ag_news test, rng(1007) 抽 pool 條（排除 target idx）＝攻擊者候選池
  budget:  n_queries = n + pool（全部為攻擊者查詢成本，誠實披露）
  端點:    ZHIPU_API_KEY 環境變數（key 永不進對話/存證）
  --dry-run: 不查 API，確定性偽向量（管道驗證，不是測量）
輸出: <out>.npz（fp16 向量）+ <out>.meta.json（texts/idx/協議）
"""
import argparse
import json
import os
import random
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pool", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out", default="e3a_glm_embeddings")
    ap.add_argument("--model", default="embedding-3")
    ap.add_argument("--endpoint",
                    default="https://open.bigmodel.cn/api/paas/v4/embeddings")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    from datasets import load_dataset
    ds_test = load_dataset("fancyzhx/ag_news")["test"]
    ds_train = load_dataset("fancyzhx/ag_news")["train"]
    rng = random.Random(1006)
    tidx = rng.sample(range(len(ds_test)), args.n)
    targets = [(i, ds_test[i]["text"][:160]) for i in tidx]
    # pool 取自 train split（攻擊者自備公開語料；test 僅 7600 條不夠 1 萬池，
    # 且 train/test 不重叠——attacker 語料與 target 語料天然分離，更乾淨）
    rng2 = random.Random(1007)
    pool_idx = rng2.sample(range(len(ds_train)), args.pool)
    pool = [(i, ds_train[i]["text"][:160]) for i in pool_idx]
    print(f"[data] targets={len(targets)} pool={len(pool)}", flush=True)

    if args.dry_run:
        import hashlib
        import numpy as np

        def pseudo(text, dim=2048):
            h = hashlib.sha256(text.encode()).digest()
            v = np.frombuffer(h * (dim // len(h) + 1),
                              dtype=np.uint8)[:dim].astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-9)

        tvecs = np.stack([pseudo(t) for _, t in targets])
        pvecs = np.stack([pseudo(t) for _, t in pool])
        model_name = "pseudo(dry-run)"
        print("[dry-run] 確定性偽向量（管道驗證，不是測量）", flush=True)
    else:
        key = os.environ.get("ZHIPU_API_KEY")
        if not key:
            print("[!] 未設 ZHIPU_API_KEY。PC: set ZHIPU_API_KEY=你的key 後重跑"
                  "（key 走環境變數，不進對話）", flush=True)
            sys.exit(2)
        import requests
        url = args.endpoint
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json"}
        all_texts = [t for _, t in targets] + [t for _, t in pool]
        vecs = []
        for bi in range(0, len(all_texts), args.batch):
            chunk = all_texts[bi:bi + args.batch]
            for attempt in range(6):
                try:
                    r = requests.post(url, headers=headers, timeout=60,
                                      json={"model": args.model,
                                            "input": chunk})
                    if r.status_code == 429:
                        wait = 15 * (attempt + 1)
                        print(f"  [429 rate-limited] wait {wait}s "
                              f"body={r.text[:200]}", flush=True)
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    data = r.json()["data"]
                    data.sort(key=lambda d: d["index"])
                    vecs.extend([d["embedding"] for d in data])
                    break
                except Exception as e:
                    if attempt == 5:
                        raise
                    print(f"  [retry {attempt + 1}] {e}", flush=True)
                    time.sleep(10 * (attempt + 1))
            time.sleep(1.5)   # 批間降速，防 RPM 限流
            done = min(bi + args.batch, len(all_texts))
            if done % 500 < args.batch or done == len(all_texts):
                print(f"  [fetch] {done}/{len(all_texts)} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        import numpy as np
        V = np.array(vecs, dtype=np.float32)
        tvecs, pvecs = V[:len(targets)], V[len(targets):]
        model_name = args.model
        print(f"[api] dim={tvecs.shape[1]} model={model_name}", flush=True)

    np.savez_compressed(args.out + ".npz",
                        target_vecs=tvecs.astype(np.float16),
                        pool_vecs=pvecs.astype(np.float16),
                        target_idx=np.array([i for i, _ in targets]),
                        pool_idx=np.array([i for i, _ in pool]))
    meta = {"model": model_name,
            "endpoint": "dry-run" if args.dry_run else args.endpoint,
            "n_targets": len(targets), "n_pool": len(pool),
            "bucket": "160 chars (~30-50 tok); targets = E1's same 500 (seed 1006)",
            "budget": {"n_queries": len(targets) + len(pool),
                       "note": "all embedding-API queries = attacker cost, "
                               "honest disclosure"},
            "targets_text": [t for _, t in targets],
            "pool_text": [t for _, t in pool]}
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"[saved] {args.out}.npz + {args.out}.meta.json "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
