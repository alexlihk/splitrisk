#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""phase0_cache_glm.py — GLM 空間 corrector 訓練數據快取（PC 隔夜跑）
ag_news train 抽 n 條（seed 2001，與 E1 test targets 天然分離）→ embedding-3 → npz fp16。
成本：100k × ~40 tok ≈ 4M tok ≈ ¥2-4（一次性；訓練時不再查 API）。
跑法：set ZHIPU_API_KEY=... 後  python phase0_cache_glm.py
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--out", default="phase0_glm_cache")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做 100 條偽向量，管道驗證")
    args = ap.parse_args()
    t0 = time.time()

    from datasets import load_dataset
    dtr = load_dataset("fancyzhx/ag_news")["train"]
    rng = random.Random(2001)
    idx = rng.sample(range(len(dtr)), args.n)
    texts = [dtr[i]["text"][:160] for i in idx]
    print(f"[data] {len(texts)} texts (seed 2001, 160 chars)", flush=True)

    if args.dry_run:
        import hashlib

        def pseudo(t, dim=2048):
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(h * (dim // len(h) + 1),
                              dtype=np.uint8)[:dim].astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-9)

        texts = texts[:100]
        V = np.stack([pseudo(t) for t in texts])
        model_name = "pseudo(dry-run)"
    else:
        key = os.environ.get("ZHIPU_API_KEY")
        if not key:
            print("[!] 設 ZHIPU_API_KEY=... 後重跑（key 不進 argv）")
            sys.exit(2)
        import requests
        url = "https://open.bigmodel.cn/api/paas/v4/embeddings"
        headers = {"Authorization": f"Bearer {key}",
                   "Content-Type": "application/json"}
        vecs = []
        for bi in range(0, len(texts), args.batch):
            chunk = texts[bi:bi + args.batch]
            for attempt in range(6):
                r = requests.post(url, headers=headers, timeout=60,
                                  json={"model": "embedding-3", "input": chunk})
                if r.status_code == 429:
                    w = 15 * (attempt + 1)
                    print(f"  [429] wait {w}s body={r.text[:150]}",
                          flush=True)
                    time.sleep(w)
                    continue
                r.raise_for_status()
                data = r.json()["data"]
                data.sort(key=lambda d: d["index"])
                vecs.extend(d["embedding"] for d in data)
                break
            else:
                raise RuntimeError(f"retries exhausted at row {bi}")
            time.sleep(args.sleep)
            done = min(bi + args.batch, len(texts))
            if done % 2000 < args.batch or done == len(texts):
                print(f"  [fetch] {done}/{len(texts)} "
                      f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        V = np.array(vecs, dtype=np.float32)
        model_name = "embedding-3"

    np.savez_compressed(args.out + ".npz", vecs=V.astype(np.float16))
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "n": len(texts), "dim": int(V.shape[1]),
                   "bucket": "160 chars; ag_news train seed 2001 "
                             "(disjoint from E1 test targets seed 1006)",
                   "cost_note": "one-time query cost for corrector training; "
                                "training itself is API-free (cached pairs)",
                   "texts": texts}, f, ensure_ascii=False)
    print(f"[saved] {args.out}.npz + {args.out}.meta.json "
          f"({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
