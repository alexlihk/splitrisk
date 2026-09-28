#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_embed_e1.py — E1: vec2text 復現驗證（embedding 反演儀器校準）

預註冊：
  模型：gtr-base 配對編碼器 + gtr corrector（vec2text 官方 ckpt）
  語料：ag_news 測試集抽樣（PREREG_N 條，截斷 16 token 桶）
  budget：beam_width=4, num_iters=1（序列長度 16 桶）
  驗收判準：token-F1 ≥ 0.70 → PASS（vec2text 論文區間）
            0.50-0.70 → REVIEW；< 0.50 → FAIL（儀器壞了，禁止出數）
  誠實標註：單儀器讀數——closure（E2）前不可作任何結論
輸出：PREREG_OUT 目錄下 e1_gtrbase.json（寫完即 dump，可重跑覆蓋）
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

PREREG = {
    "purpose": "E1 — vec2text attacker reproduction (instrument calibration)",
    "encoder": "gtr-base (paired)", "corrector": "gtr",
    "bucket": "16 tokens", "beam": 4, "iters": 1,
    "acceptance": "token-F1 >= 0.70 PASS / 0.50-0.70 REVIEW / <0.50 FAIL",
    "honesty": "single-instrument readout; no conclusion before E2 closure",
}


def dump(path, res):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--out", default="/content/drive/MyDrive/Colab/XDataTrust/"
                                     "Results/embed/e1_gtrbase.json")
    args = ap.parse_args()
    t0 = time.time()
    print("[prereg]", json.dumps(PREREG, ensure_ascii=False), flush=True)

    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            if json.load(f).get("done"):
                print("[resume] already done:", args.out)
                return

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""),
          flush=True)

    # --- 語料：ag_news 測試集抽樣 ---
    print("[data] loading ag_news test split...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")["test"]
    rng = random.Random(1006)
    idx = rng.sample(range(len(ds)), min(args.n, len(ds)))
    texts = [ds[i]["text"][:400] for i in idx]

    # --- vec2text 儀器 ---
    print("[vec2text] loading gtr-base encoder + gtr corrector...", flush=True)
    import vec2text
    import transformers
    corpus_model, corpus_tokenizer, embedder = vec2text.models.load_encoder(
        "gtr-base")
    corrector = vec2text.models.load_corrector("gtr")
    corpus_model = corpus_model.to(dev).eval()
    embedder = embedder.to(dev).eval()
    print("[vec2text] loaded", flush=True)

    def encode(batch_texts):
        inputs = corpus_tokenizer(batch_texts, return_tensors="pt",
                                  truncation=True, max_length=64,
                                  padding=True).to(dev)
        with torch.no_grad():
            out = embedder(**inputs)
        emb = out.last_hidden_state[:, 0, :]        # CLS
        return torch.nn.functional.normalize(emb, dim=-1)

    print(f"[encode] {len(texts)} texts...", flush=True)
    embs = []
    for i in range(0, len(texts), 64):
        embs.append(encode(texts[i:i + 64]))
    embs = torch.cat(embs)

    # --- 反演 ---
    print(f"[invert] beam={args.beam} iters={args.iters} "
          f"seq_len={args.seq_len}...", flush=True)
    t1s = time.time()
    rec = vec2text.invert_gtr_embeddings(
        embs, corpus_model, corpus_tokenizer, corrector,
        num_iters=args.iters, beam_width=args.beam,
        sequence_length=args.seq_len)
    if isinstance(rec, tuple):
        rec = rec[0]
    print(f"[invert] done in {(time.time() - t1s) / 60:.1f} min", flush=True)

    # --- 指標：token-F1 / CER / exact ---
    print("[metrics]...", flush=True)
    import evaluate as hf_evaluate
    f1m = hf_evaluate.load("f1")
    cerm = hf_evaluate.load("cer")
    f1s, cers, exact = [], [], []
    for orig, hyp in zip(texts, rec):
        o = corpus_tokenizer.tokenize(orig.lower())[:args.seq_len]
        h = corpus_tokenizer.tokenize(str(hyp).lower())[:args.seq_len]
        if not o:
            continue
        f1 = f1m.compute(predictions=[h if h else ["<empty>"]],
                         references=[o], average="macro")["f1"]
        f1s.append(f1)
        cers.append(cerm.compute(predictions=[str(hyp)],
                                 references=[orig]))
        exact.append(1.0 if [t.lower() for t in o] ==
                     [t.lower() for t in corpus_tokenizer.tokenize(
                         str(hyp).lower())] else 0.0)

    import numpy as np
    res = {"prereg": PREREG, "n": len(f1s),
           "result": {"token_f1_mean": float(np.mean(f1s)),
                      "token_f1_std": float(np.std(f1s)),
                      "cer_mean": float(np.mean(cers)),
                      "exact_ratio": float(np.mean(exact)),
                      "minutes": (time.time() - t0) / 60,
                      "device": dev},
           "per_sample_f1": f1s}
    lo, hi = 0.70, 1.0
    res["verdict"] = ("PASS" if res["result"]["token_f1_mean"] >= lo else
                      "REVIEW" if res["result"]["token_f1_mean"] >= 0.50
                      else "FAIL")
    res["done"] = True
    dump(args.out, res)
    print(f"★ E1 token-F1={np.mean(f1s):.4f} CER={np.mean(cers):.4f} "
          f"exact={np.mean(exact):.3f} → {res['verdict']}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
