#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""embed_corrector.py — GLM 空間 Phase 2：迭代校正器（純 GPU，零 API）
架構：Corr2Text（T5-base）——encoder = [假設文本 tokens（T5 詞嵌入）] + [K 個
virtual tokens（目標 2048 維向量投影）]，decoder = 原文。
與 vec2text corrector 同構但其 encoder 吃向量需重嵌假設（API 依賴）；
我們的 encoder 直接讀文本 → 校正環路零查詢。

modes:
  gen_hyps  用 inv_v1 對 50k 訓練文本生成假設（greedy）→ phase2_hyps.json
  train     (假設文本, 原文, 目標向量) 訓 corrector → corr_v1/corr_model.pt
  eval      預算階梯 steps=0(inv)/1/2 × beam，per-sample F1 + 長度分桶
            + corrector 地板（隨機目標向量+隨機池文本假設）→ ladder JSON

本地冒煙：--dry-run（t5-small、偽數據、全模式管道驗證）
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
    ap.add_argument("mode", choices=["gen_hyps", "train", "eval"])
    ap.add_argument("--inv-ckpt", default="")
    ap.add_argument("--train-npz", default="")
    ap.add_argument("--train-meta", default="")
    ap.add_argument("--hyps-json", default="")
    ap.add_argument("--eval-npz", default="")
    ap.add_argument("--eval-meta", default="")
    ap.add_argument("--base-model", default="t5-base")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--virtual-tokens", type=int, default=8)
    ap.add_argument("--max-target-len", type=int, default=96)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--eval-beam", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2,
                    help="eval: corrector 套用次數（steps=0 即純 inv）")
    ap.add_argument("--out-dir", default="corr_v1")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


class Emb2Text(nn.Module):
    """Phase 1 反演模型（與 embed_train_inversion.py 同構，載 ckpt 用）。"""

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


class Corr2Text(nn.Module):
    """校正器：encoder = 假設文本 tokens + 目標向量 virtual tokens。"""

    def __init__(self, base_model, emb_dim, virtual_tokens):
        super().__init__()
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(base_model)
        self.t5 = T5ForConditionalGeneration.from_pretrained(base_model)
        self.d = self.t5.config.d_model
        self.proj = nn.Linear(emb_dim, virtual_tokens * self.d)
        self.vt = virtual_tokens

    def _enc(self, hyp_ids, hyp_mask, emb):
        B = emb.shape[0]
        we = self.t5.shared(hyp_ids)
        ve = self.proj(emb.float()).view(B, self.vt, self.d)
        inputs = torch.cat([we, ve], dim=1)
        mask = torch.cat([hyp_mask,
                          torch.ones(B, self.vt, dtype=torch.long,
                                     device=emb.device)], dim=1)
        return inputs, mask

    def forward(self, hyp_ids, hyp_mask, emb, labels=None):
        inputs, mask = self._enc(hyp_ids, hyp_mask, emb)
        return self.t5(inputs_embeds=inputs, attention_mask=mask, labels=labels)

    def generate(self, hyp_ids, hyp_mask, emb, **kw):
        inputs, mask = self._enc(hyp_ids, hyp_mask, emb)
        return self.t5.generate(inputs_embeds=inputs, attention_mask=mask, **kw)


def pseudo_texts(n, seed=0):
    return [f"news report number {i} about market growth and policy"
            for i in range(n)]


def pseudo_vecs(n, dim=2048, seed=0):
    V = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        h = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        v = np.frombuffer(h * (dim // len(h) + 1), dtype=np.uint8)[:dim]
        V[i] = v.astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V


def gtr_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")


def f1_stats(origs, hyps, gtr):
    def toks(s):
        return {t for t in gtr.tokenize(str(s).lower()) if t.strip("▁")}
    f1s = []
    for o, h in zip(origs, hyps):
        so, sh = toks(o), toks(h)
        if not so:
            continue
        tp = len(so & sh)
        p = tp / len(sh) if sh else 0.0
        r = tp / len(so) if so else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return f1s


def load_train_texts(args):
    if args.dry_run:
        return pseudo_texts(max(args.n, 64))[:args.n]
    z = np.load(args.train_npz)
    meta = json.load(open(args.train_meta, encoding="utf-8"))
    texts = meta["texts"] if "vecs" in z else meta["targets_text"] + meta["pool_text"]
    return texts[:args.n]


def main():
    args = build_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mode={args.mode}] device={dev} base={args.base_model}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "gen_hyps":
        if args.dry_run:
            inv = Emb2Text(args.base_model, 2048,
                           args.virtual_tokens).to(dev)
            inv.eval()
        else:
            ck = torch.load(args.inv_ckpt, map_location=dev, weights_only=False)
            inv = Emb2Text(ck["base_model"], ck["emb_dim"],
                           ck["virtual_tokens"]).to(dev)
            inv.load_state_dict(ck["state_dict"]); inv.eval()
        texts = load_train_texts(args)
        if args.dry_run:
            vecs_all = pseudo_vecs(len(texts))
        else:
            z = np.load(args.train_npz)
            vecs_all = (z["vecs"] if "vecs" in z else
                        np.concatenate([z["target_vecs"], z["pool_vecs"]])
                        ).astype(np.float32)[:len(texts)]
        print(f"[gen] {len(texts)} texts, greedy...", flush=True)
        hyps = []
        with torch.no_grad():
            bs = 32
            for bi in range(0, len(texts), bs):
                emb = torch.from_numpy(vecs_all[bi:bi + bs]).to(dev)
                out = inv.generate(emb, max_new_tokens=args.max_target_len,
                                   num_beams=1)
                hyps.extend(inv.tok.batch_decode(out, skip_special_tokens=True))
                if (bi // bs) % 100 == 0:
                    print(f"  [gen] {min(bi + bs, len(texts))}/{len(texts)} "
                          f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        with open(os.path.join(args.out_dir, "phase2_hyps.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"texts": texts, "hyps": hyps,
                       "inv_ckpt": args.inv_ckpt}, f, ensure_ascii=False)
        print(f"[saved] phase2_hyps.json ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
        return

    if args.mode == "train":
        if args.dry_run:
            texts, hyps = pseudo_texts(64), pseudo_texts(64, seed=1)
            vecs = pseudo_vecs(64)
        else:
            data = json.load(open(args.hyps_json, encoding="utf-8"))
            texts, hyps = data["texts"][:args.n], data["hyps"][:args.n]
            z = np.load(args.train_npz)
            if "vecs" in z:
                vecs = z["vecs"].astype(np.float32)
            else:
                vecs = np.concatenate([z["target_vecs"],
                                       z["pool_vecs"]]).astype(np.float32)
            vecs = vecs[:len(texts)]
        print(f"[train] pairs={len(texts)}", flush=True)
        corr = Corr2Text(args.base_model, 2048,
                         args.virtual_tokens).to(dev)
        opt = torch.optim.AdamW(corr.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
        corr.train()
        step = 0
        skipped = 0
        losses = []
        for ep in range(args.epochs):
            perm = np.random.default_rng(2000 + ep).permutation(len(texts))
            for bi in range(0, len(perm) - args.batch + 1, args.batch):
                bidx = perm[bi:bi + args.batch]
                emb = torch.from_numpy(vecs[bidx]).to(dev)
                enc = corr.tok([hyps[i] for i in bidx], return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=args.max_target_len).to(dev)
                lab = corr.tok([texts[i] for i in bidx], return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=args.max_target_len).to(dev)
                labels = lab.input_ids.masked_fill(
                    lab.input_ids == corr.tok.pad_token_id, -100)
                with torch.cuda.amp.autocast(enabled=dev.type == "cuda"):
                    out = corr(enc.input_ids, enc.attention_mask, emb,
                               labels=labels)
                if not torch.isfinite(out.loss):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad()
                scaler.scale(out.loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                losses.append(float(out.loss.detach()))
                step += 1
                if step % 50 == 0:
                    print(f"  [train] ep{ep} step{step} "
                          f"loss={np.mean(losses[-50:]):.4f} "
                          f"skip={skipped} "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        torch.save({"state_dict": corr.state_dict(),
                    "base_model": args.base_model,
                    "virtual_tokens": args.virtual_tokens, "emb_dim": 2048,
                    "epochs": args.epochs, "lr": args.lr,
                    "batch": args.batch,
                    "nan_windows_total": sum(
                        1 for v in losses if not np.isfinite(v))},
                   os.path.join(args.out_dir, "corr_model.pt"))
        print(f"[saved] corr_model.pt ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
        return

    if args.mode == "eval":
        gtr = gtr_tok()
        if args.dry_run:
            tt = pseudo_texts(max(args.eval_n, 8), seed=2)[:args.eval_n]
            tv = torch.from_numpy(pseudo_vecs(len(tt), seed=2)).to(dev)
        else:
            z = np.load(args.eval_npz)
            meta = json.load(open(args.eval_meta, encoding="utf-8"))
            tv_np = (z["target_vecs"] if "target_vecs" in z else z["vecs"])
            tt = (meta["targets_text"] if "target_vecs" in z else meta["texts"])
            n = min(args.eval_n, len(tt))
            tv = torch.from_numpy(tv_np[:n].astype(np.float32)).to(dev)
            tt = tt[:n]
        n = len(tt)
        ick = torch.load(args.inv_ckpt, map_location=dev, weights_only=False)
        inv = Emb2Text(ick["base_model"], ick["emb_dim"],
                       ick["virtual_tokens"]).to(dev)
        inv.load_state_dict(ick["state_dict"]); inv.eval()
        cck = torch.load(os.path.join(args.out_dir, "corr_model.pt"),
                         map_location=dev, weights_only=False)
        corr = Corr2Text(cck["base_model"], cck["emb_dim"],
                         cck["virtual_tokens"]).to(dev)
        corr.load_state_dict(cck["state_dict"]); corr.eval()

        def gen_batch(model, embs, hyp_texts, beam):
            out_hyps = []
            with torch.no_grad():
                for bi in range(0, len(embs), 16):
                    emb = embs[bi:bi + 16]
                    if hyp_texts is None:
                        out = model.generate(emb, max_new_tokens=args.max_target_len,
                                             num_beams=beam)
                    else:
                        enc = model.tok(hyp_texts[bi:bi + 16],
                                        return_tensors="pt", padding=True,
                                        truncation=True,
                                        max_length=args.max_target_len).to(dev)
                        out = model.generate(enc.input_ids, enc.attention_mask,
                                             emb, max_new_tokens=args.max_target_len,
                                             num_beams=beam)
                    out_hyps.extend(model.tok.batch_decode(
                        out, skip_special_tokens=True))
            return out_hyps

        ladder = {}
        hyps = None
        for s in range(args.steps + 1):
            if s == 0:
                hyps = gen_batch(inv, tv, None, args.eval_beam)
                tag = "step0_inv"
            else:
                hyps = gen_batch(corr, tv, hyps, args.eval_beam)
                tag = f"step{s}_corr"
            f1s = f1_stats(tt, hyps, gtr)
            m, sd = float(np.mean(f1s)), float(np.std(f1s))
            ladder[tag] = {"token_f1_mean": m, "token_f1_std": sd,
                           "n": len(f1s)}
            print(f"★ {tag}: token-F1={m:.4f}±{sd:.4f}", flush=True)

        # 分桶（用 step 最高層的 per-sample）
        lens = np.array([len(gtr.tokenize(t)) for t in tt])
        f1_last = f1_stats(tt, hyps, gtr)
        buckets = {}
        for lo, hi, name in [(1, 16, "1-16"), (17, 32, "17-32"),
                             (33, 48, "33-48"), (49, 10 ** 9, "49+")]:
            mask = (lens >= lo) & (lens <= hi)
            if mask.sum():
                buckets[name] = {"n": int(mask.sum()),
                                 "f1_mean": float(np.array(f1_last)[mask].mean())}
        # corrector 地板：隨機目標向量 + 隨機假設（池文本）
        rng = np.random.default_rng(0)
        rand_t = torch.from_numpy(pseudo_vecs(min(64, n), seed=9)).to(dev)
        pool_for_hyp = json.load(open(args.hyps_json, encoding="utf-8"))["hyps"][:64] \
            if args.hyps_json and os.path.exists(args.hyps_json) else \
            pseudo_texts(64, seed=3)
        rand_hyps = [pool_for_hyp[i] for i in
                     rng.permutation(len(pool_for_hyp))[:len(rand_t)]]
        fh = gen_batch(corr, rand_t, rand_hyps, 1)
        floor_pair = [(tt[i] if i < len(tt) else pseudo_texts(1, seed=9)[0],
                       fh[i]) for i in range(len(fh))]
        floor_f1s = f1_stats([p[0] for p in floor_pair],
                             [p[1] for p in floor_pair], gtr)
        res = {"ladder": ladder, "length_buckets_last_step": buckets,
               "corr_floor_random_target_random_hyp_f1":
                   float(np.mean(floor_f1s)),
               "steps": args.steps, "eval_beam": args.eval_beam,
               "note": "dry-run = plumbing only" if args.dry_run else
                       "Phase 2 corrector ladder; floor protocol = random "
                       "target vector + random pool hypothesis"}
        with open(os.path.join(args.out_dir, "ladder_result.json"), "w",
                  encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[saved] ladder_result.json ({(time.time() - t0) / 60:.1f} min)",
              flush=True)


if __name__ == "__main__":
    main()
