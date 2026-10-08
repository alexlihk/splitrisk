#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bucket_analysis.py — E1 系列長度分桶離線分析（PC，不需 GPU）
預註冊假設：0.70 缺口來自長度（我們 30-50 token vs 論文 ≤32 token 區間）。
語料 seed 確定性（rng(1006) 同 E1）＋ JSON per_sample_f1 順序＝texts 順序 → join 精確。
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

import random
from datasets import load_dataset
ds = load_dataset("fancyzhx/ag_news")["test"]
rng = random.Random(1006)
texts = [ds[i]["text"][:160] for i in rng.sample(range(len(ds)), 500)]
print(f"[data] {len(texts)} texts regenerated (seed 1006, 160 chars)")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
token_lens = [len(tok.tokenize(t)) for t in texts]
import numpy as np
tl = np.array(token_lens)
print(f"[tokens] min={tl.min()} median={np.median(tl)} max={tl.max()} "
      f"mean={tl.mean():.1f}")

BUCKETS = [(1, 16), (17, 32), (33, 48), (49, 10**9)]
def bucket_of(n):
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else "49+"
    return "?"

RUNS = [("E1 (2/1)", "e1_gtrbase.json"),
        ("E1b (4/2)", "e1b_gtrbase_s4b2.json"),
        ("E1c (4/4)", "e1c_gtrbase_s4b4.json")]

rows = {}
for label, fn in RUNS:
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        print(f"[skip] {fn} not present yet")
        continue
    d = json.load(open(p, encoding="utf-8"))
    f1 = np.array(d["per_sample_f1"])
    assert len(f1) == 500, f"{fn}: {len(f1)} != 500"
    rows[label] = f1

print("\n=== token-F1 by length bucket ===")
hdr = "bucket (tok) | n   | " + " | ".join(rows.keys())
print(hdr)
print("-" * len(hdr))
bucket_means = {}
for lo, hi in BUCKETS:
    name = bucket_of(lo) if hi < 10**9 else "49+"
    mask = (tl >= lo) & (tl <= hi)
    n = int(mask.sum())
    if n == 0:
        continue
    vals = []
    for label, f1 in rows.items():
        m = float(f1[mask].mean())
        vals.append(m)
        bucket_means.setdefault(label, {})[name] = (m, n)
    print(f"{name:12s} | {n:3d} | " + " | ".join(f"{v:.4f}" for v in vals))

# 全體對照
print("\n=== overall (unbucketed) ===")
for label, f1 in rows.items():
    print(f"{label}: {f1.mean():.4f}±{f1.std():.4f}")

# 論文區間對照：≤32 token 桶 vs vec2text 論文 gtr-base 短文區間 0.85-0.92
print("\n=== paper-region check (<=32 tokens) ===")
mask32 = tl <= 32
print(f"n(<=32)={int(mask32.sum())}")
for label, f1 in rows.items():
    m = float(f1[mask32].mean())
    print(f"{label}: {m:.4f}  (paper gtr-base short-text range ~0.85-0.92 at their budget)")

out = {"token_len_stats": {"min": int(tl.min()), "median": float(np.median(tl)),
                           "max": int(tl.max()), "mean": float(tl.mean())},
       "bucket_means": {k: v for k, v in bucket_means.items()},
       "note": "per_sample_f1 joined with deterministic seed-1006 corpus; "
               "token lens via sentence-transformers/gtr-t5-base tokenizer"}
with open(os.path.join(BASE, "bucket_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("\n[saved] bucket_analysis.json")
