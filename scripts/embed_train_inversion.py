#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""embed_train_inversion.py — GLM 空間 Phase 1：inversion model（自研簡化架構）
架構：T5 seq2seq；encoder 輸入＝K 個 virtual tokens（2048 維向量經線性投影），
decoder 從向量生成文本。與 vec2text 的 inversion model 同構，但訓練管道完全
自持（只依賴 HF Trainer 標準件），本機 CPU 可迷你驗證後才上 GPU。
Phase 2（corrector 迭代校正）用本檔產出的權重，下一 session。

本地冒煙：  --dry-run --base-model t5-small --n-train 64 --epochs 1 --batch 8 --eval-n 8
Colab 正式：--base-model t5-base --n-train 50000 --epochs 2 --eval-n 500 --eval-beam 4
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", default="")
    ap.add_argument("--train-meta", default="")
    ap.add_argument("--eval-npz", default="", help="e3a_glm.npz（含真 GLM 500 targets）")
    ap.add_argument("--eval-meta", default="")
    ap.add_argument("--base-model", default="t5-small")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--virtual-tokens", type=int, default=8)
    ap.add_argument("--max-target-len", type=int, default=96)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--eval-beam", type=int, default=1)
    ap.add_argument("--out-dir", default="inv_ckpt")
    ap.add_argument("--dry-run", action="store_true",
                    help="本機冒煙：內生偽向量+短文本，不讀 npz")
    return ap.parse_args()


class Emb2Text(nn.Module):
    """embedding → text：T5 encoder 只吃 K 個 virtual tokens。"""

    def __init__(self, base_model, emb_dim, virtual_tokens):
        super().__init__()
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(base_model)
        self.t5 = T5ForConditionalGeneration.from_pretrained(base_model)
        self.d = self.t5.config.d_model
        self.proj = nn.Linear(emb_dim, virtual_tokens * self.d)
        self.vt = virtual_tokens

    def _enc(self, emb):
        B = emb.shape[0]
        ve = self.proj(emb.float()).view(B, self.vt, self.d)
        mask = torch.ones(B, self.vt, dtype=torch.long, device=emb.device)
        return ve, mask

    def forward(self, emb, labels=None):
        ve, mask = self._enc(emb)
        return self.t5(inputs_embeds=ve, attention_mask=mask, labels=labels)

    def generate(self, emb, **kw):
        ve, mask = self._enc(emb)
        return self.t5.generate(inputs_embeds=ve, attention_mask=mask, **kw)


def pseudo_data(n, dim=2048, seed=0):
    """dry-run 用：hash 偽向量 + 短新聞腔文本（管道驗證，非測量）。"""
    rng = np.random.default_rng(seed)
    texts = [f"news report number {i} about market growth and policy"
             for i in range(n)]
    V = np.zeros((n, dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h * (dim // len(h) + 1), dtype=np.uint8)[:dim]
        V[i] = v.astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V, texts


def load_pairs(args):
    if args.dry_run:
        V, texts = pseudo_data(max(args.n_train, 256))
        return V[:args.n_train], texts[:args.n_train]
    z = np.load(args.train_npz)
    meta = json.load(open(args.train_meta, encoding="utf-8"))
    # 雙格式：phase0 cache = vecs/texts；e3a 型 = target_vecs+pool_vecs / targets_text+pool_text
    if "vecs" in z:
        vecs = z["vecs"].astype(np.float32)
        texts = meta["texts"]
    else:
        vecs = np.concatenate([z["target_vecs"], z["pool_vecs"]]).astype(np.float32)
        texts = meta["targets_text"] + meta["pool_text"]
    assert len(vecs) == len(texts), f"{len(vecs)} vs {len(texts)}"
    n = min(args.n_train, len(texts))
    return vecs[:n], texts[:n]


def load_eval(args):
    if args.dry_run:
        V, texts = pseudo_data(64, seed=1)
        return V[:args.eval_n], texts[:args.eval_n]
    z = np.load(args.eval_npz)
    meta = json.load(open(args.eval_meta, encoding="utf-8"))
    if "target_vecs" in z:
        tv = z["target_vecs"].astype(np.float32)
        tt = meta["targets_text"]
    else:
        tv = z["vecs"].astype(np.float32)
        tt = meta["texts"]
    n = min(args.eval_n, len(tt))
    return tv[:n], tt[:n]


def main():
    args = build_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev} | base={args.base_model}", flush=True)

    from transformers import AutoTokenizer

    gtr_tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

    def tok_f1(origs, hyps):
        def toks(s):
            return {t for t in gtr_tok.tokenize(str(s).lower()) if t.strip("▁")}
        f1s = []
        for o, h in zip(origs, hyps):
            so, sh = toks(o), toks(h)
            if not so:
                continue
            tp = len(so & sh)
            p = tp / len(sh) if sh else 0.0
            r = tp / len(so) if so else 0.0
            f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        return float(np.mean(f1s)), float(np.std(f1s))

    model = Emb2Text(args.base_model, 2048, args.virtual_tokens).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")

    train_V, train_T = load_pairs(args)
    print(f"[data] train pairs={len(train_T)}", flush=True)

    model.train()
    step = 0
    losses = []
    for ep in range(args.epochs):
        perm = np.random.default_rng(1000 + ep).permutation(len(train_T))
        for bi in range(0, len(perm) - args.batch + 1, args.batch):
            bidx = perm[bi:bi + args.batch]
            emb = torch.from_numpy(train_V[bidx]).to(dev)
            batch_t = [train_T[i] for i in bidx]
            enc = model.tok(batch_t, return_tensors="pt", padding=True,
                            truncation=True,
                            max_length=args.max_target_len).to(dev)
            labels = enc.input_ids.masked_fill(
                enc.input_ids == model.tok.pad_token_id, -100)
            with torch.cuda.amp.autocast(enabled=dev.type == "cuda"):
                out = model(emb, labels=labels)
                loss = out.loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss))
            step += 1
            if step % 50 == 0:
                print(f"  [train] ep{ep} step{step} "
                      f"loss={np.mean(losses[-50:]):.4f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "base_model": args.base_model,
                "virtual_tokens": args.virtual_tokens, "emb_dim": 2048},
               os.path.join(args.out_dir, "inv_model.pt"))

    # --- eval ---
    model.eval()
    eval_V, eval_T = load_eval(args)
    hyps = []
    with torch.no_grad():
        for bi in range(0, len(eval_T), 16):
            emb = torch.from_numpy(eval_V[bi:bi + 16]).to(dev)
            out = model.generate(emb, max_new_tokens=args.max_target_len,
                                 num_beams=args.eval_beam)
            hyps.extend(model.tok.batch_decode(out, skip_special_tokens=True))
    m, sd = tok_f1(eval_T, hyps)
    print(f"★ inversion eval token-F1={m:.4f}±{sd:.4f} "
          f"(n={len(hyps)}, beam={args.eval_beam})", flush=True)
    res = {"base_model": args.base_model, "n_train": len(train_T),
           "epochs": args.epochs, "eval_n": len(hyps),
           "eval_beam": args.eval_beam, "token_f1_mean": m,
           "token_f1_std": sd,
           "final_loss": float(np.mean(losses[-50:])) if losses else None,
           "minutes": (time.time() - t0) / 60,
           "note": "Phase 1 inversion model (no corrector); "
                   "dry-run = pseudo vectors, NOT a measurement"}
    with open(os.path.join(args.out_dir, "train_result.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out_dir}/inv_model.pt + train_result.json "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
