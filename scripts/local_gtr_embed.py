#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""local_gtr_embed.py — closure 第二儀器：gtr-base 空間池檢索（PC 本地，零 API）
與 E3 同一檢索儀器、同一語料（500 targets seed1006 + 10k pool seed1007 train），
embedding 換成 gtr-base（本地 sentence-transformers，768 維）。
產出 npz/meta 與 e3a 同格式 → 直接餵 colab_embed_e3b_retrieval.py。
closure 判讀：兩個獨立機制（生成式反演 E1 系 vs 池檢索）各自對自己的地板
 是否都確認 gtr-base embedding 有實質洩漏。
"""
import argparse
import json
import os
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pool", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="closure_gtr")
    args = ap.parse_args()
    t0 = time.time()

    from datasets import load_dataset
    ds_test = load_dataset("fancyzhx/ag_news")["test"]
    ds_train = load_dataset("fancyzhx/ag_news")["train"]
    rng = random.Random(1006)
    tidx = rng.sample(range(len(ds_test)), args.n)
    targets = [(i, ds_test[i]["text"][:160]) for i in tidx]
    rng2 = random.Random(1007)
    pool_idx = rng2.sample(range(len(ds_train)), args.pool)
    pool = [(i, ds_train[i]["text"][:160]) for i in pool_idx]
    print(f"[data] targets={len(targets)} pool={len(pool)}", flush=True)

    from sentence_transformers import SentenceTransformer
    print("[embed] loading sentence-transformers/gtr-t5-base (local)...", flush=True)
    m = SentenceTransformer("sentence-transformers/gtr-t5-base")
    all_texts = [t for _, t in targets] + [t for _, t in pool]
    print(f"[embed] encoding {len(all_texts)} texts on CPU...", flush=True)
    V = m.encode(all_texts, batch_size=args.batch, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=True)
    tvecs, pvecs = V[:len(targets)], V[len(targets):]
    print(f"[embed] dim={tvecs.shape[1]} done "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    import numpy as np
    np.savez_compressed(args.out + ".npz",
                        target_vecs=tvecs.astype(np.float16),
                        pool_vecs=pvecs.astype(np.float16),
                        target_idx=np.array([i for i, _ in targets]),
                        pool_idx=np.array([i for i, _ in pool]))
    meta = {"model": "sentence-transformers/gtr-t5-base (local, closure 2nd instrument)",
            "endpoint": "local",
            "n_targets": len(targets), "n_pool": len(pool),
            "bucket": "160 chars (~30-50 tok); same corpus as E1/E3 (seed 1006/1007)",
            "budget": {"n_queries": 0,
                       "note": "no API queries - local model; attacker cost = compute"},
            "targets_text": [t for _, t in targets],
            "pool_text": [t for _, t in pool]}
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"[saved] {args.out}.npz + {args.out}.meta.json "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
