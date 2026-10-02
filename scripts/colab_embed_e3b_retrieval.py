#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_embed_e3b_retrieval.py — E3-b: pool 檢索攻擊（Colab/PC 離線計算）
攻擊者模型：有查詢預算拿 target+pool 的 embedding，cosine top-1 檢索候選池，
量檢索文本與原文的 token-F1（與 E1 同 set-based 約定、同 gtr tokenizer）。
底線：同 run 內算隨機配對 F1（embedding 無洩漏時 top-1 = 任意池文本的機率地板）。
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="e3b_retrieval.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.npz)
    tv = z["target_vecs"].astype(np.float32)
    pv = z["pool_vecs"].astype(np.float32)
    meta = json.load(open(args.meta, encoding="utf-8"))
    ttexts, ptexts = meta["targets_text"], meta["pool_text"]
    print(f"[load] targets={tv.shape} pool={pv.shape} "
          f"model={meta['model']}", flush=True)

    from transformers import AutoTokenizer
    # jxm/gtr__nq__32 的 config 無 model_type，AutoTokenizer 不認——
    # embedder_tokenizer 的本體就是 sentence-transformers/gtr-t5-base（E1 同款快取）
    tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

    def toks(s):
        return [t for t in tok.tokenize(str(s).lower()) if t.strip("▁")]

    def pair_f1(origs, hyps):
        f1s = []
        for o_text, h_text in zip(origs, hyps):
            o, h = set(toks(o_text)), set(toks(h_text))
            if not o:
                continue
            tp = len(o & h)
            p = tp / len(h) if h else 0.0
            r = tp / len(o) if o else 0.0
            f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        return f1s

    tv /= np.linalg.norm(tv, axis=1, keepdims=True) + 1e-9
    pv /= np.linalg.norm(pv, axis=1, keepdims=True) + 1e-9
    sims = tv @ pv.T                       # n_targets × n_pool
    top = sims.argmax(1)
    hit_texts = [ptexts[j] for j in top]
    f1s = pair_f1(ttexts, hit_texts)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pv))
    rand_hits = [ptexts[j] for j in perm[: len(ttexts)]]
    floor_f1s = pair_f1(ttexts, rand_hits)

    m, sd = float(np.mean(f1s)), float(np.std(f1s))
    fm = float(np.mean(floor_f1s))
    res = {"attacker_model": "pool-retrieval (cosine top-1)",
           "n": len(f1s), "n_pool": len(ptexts),
           "token_f1_mean": m, "token_f1_std": sd,
           "floor_random_pairing_f1": fm,
           "excess_over_floor": m - fm,
           "mean_top1_cosine": float(np.mean(sims.max(1))),
           "budget": meta["budget"],
           "model": meta["model"],
           "note": "retrieval attack = needle-in-haystack attacker; "
                   "floor computed in-run by random pairing (same unit)"}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"★ E3b retrieval token-F1={m:.4f}±{sd:.4f}  "
          f"floor={fm:.4f}  excess={m - fm:+.4f}", flush=True)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
