#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_embed_b3.py — Batch 3: multi-seed + second corpus (TIFS v6 彈藥)

=== Colab setup cell（先跑）===
!pip install -q vec2text jiwer
!pip install -q transformers==4.46.3
# from google.colab import drive; drive.mount('/content/drive')

=== 跑法（T4；單 run 約 5-25 分）===
%run colab_embed_b3.py --corpus agnews --seed 2006 --num-steps 2 --beam 1
%run colab_embed_b3.py --corpus agnews --seed 2006 --num-steps 4 --beam 2
%run colab_embed_b3.py --corpus agnews --seed 2006 --num-steps 4 --beam 4
%run colab_embed_b3.py --corpus agnews --seed 3006 --num-steps 2 --beam 1
%run colab_embed_b3.py --corpus agnews --seed 3006 --num-steps 4 --beam 2
%run colab_embed_b3.py --corpus agnews --seed 3006 --num-steps 4 --beam 4
%run colab_embed_b3.py --corpus enron  --seed 1006 --num-steps 2 --beam 1
%run colab_embed_b3.py --corpus enron  --seed 1006 --num-steps 4 --beam 2
%run colab_embed_b3.py --corpus enron  --seed 1006 --num-steps 4 --beam 4

目的（預註冊）：
  A. 多 seed：seed 2006/3006 為兩個新的 500 條語料抽樣（1006＝論文原始 run，
     不重跑）→ 桶級 mean±sd 回應「n=52 single-run」攻擊。
  B. 第二語料：Enron 電郵（SetFit/enron_spam，真實郵件文本，域差異最大）
     → 證明幻覺地板/校準協議不是 AG News 特有（同一 160 字符截斷，隔離域效應）。
輸出：JSON 含 per_sample_f1 ＋ per_text_token_lengths（本 run 內直出，
      免去 bucket_analysis 的離線重算）＋corpus/seed 標籤。
"""
import argparse, json, os, random, shutil, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_embed_e1 import (dump, selfheal_vec2text, selfinstall_deps,
                            check_transformers_version)

PREREG = {
    "purpose": "Batch 3 — multi-seed + second-corpus calibration "
               "(TIFS v6 ammunition)",
    "protocol": "same as E1 (vec2text gtr-base corrector, 160-char "
                "truncation, set-based token-F1)",
    "seeds": {"1006": "original paper run (not rerun here)",
              "2006/3006": "new corpus samples",
              "enron": "SetFit/enron_spam, seed as passed"},
    "acceptance": "descriptive (no pass/fail — multi-seed/variance readout)",
}


def load_corpus(name, seed, n):
    from datasets import load_dataset
    if name == "agnews":
        ds = load_dataset("fancyzhx/ag_news")["test"]
        rng = random.Random(seed)
        idx = rng.sample(range(len(ds)), min(n, len(ds)))
        return [ds[i]["text"][:160] for i in idx]
    if name == "enron":
        ds = load_dataset("SetFit/enron_spam")["train"]
        rng = random.Random(seed)
        idx = rng.sample(range(len(ds)), min(n, len(ds)))
        texts = []
        for i in idx:
            body = ds[i].get("text") or ds[i].get("email") or ""
            body = " ".join(str(body).split())[:160]
            texts.append(body)
        return [x for x in texts if len(x) >= 40]  # drop empties
    raise SystemExit(f"unknown corpus {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["agnews", "enron"], default="agnews")
    ap.add_argument("--seed", type=int, default=2006)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="")
    ap.add_argument("--copy-to", default="",
                    help="copy finished JSON here (e.g. Drive path) — /content "
                         "is ephemeral; always copy to Drive")
    args = ap.parse_args()
    if not args.out:
        args.out = (f"/content/e1_{args.corpus}_s{args.seed}"
                    f"_b{args.num_steps}{args.beam}.json")
    PREREG["corpus"] = args.corpus
    PREREG["corpus_seed"] = args.seed
    PREREG["budget"] = {"num_steps": args.num_steps,
                        "sequence_beam_width": args.beam}
    PREREG["n"] = args.n
    # beam-width memory guard (T4 14.5GB; bug ledger #2): effective
    # sequences = batch × beam — keep ≤ 32 (beam-4 OOMs at batch 32)
    args.batch = max(8, min(args.batch, 32 // max(1, args.beam)))
    print(f"[batch] effective batch={args.batch} (beam={args.beam})",
          flush=True)
    t0 = time.time()
    print("[prereg]", json.dumps(PREREG, ensure_ascii=False), flush=True)

    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            if json.load(f).get("done"):
                print("[resume] already done:", args.out)
                return

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {dev}", flush=True)

    print(f"[data] corpus={args.corpus} seed={args.seed} n={args.n}", flush=True)
    texts = load_corpus(args.corpus, args.seed, args.n)
    print(f"[data] {len(texts)} texts loaded", flush=True)

    selfheal_vec2text()
    selfinstall_deps()
    import vec2text
    check_transformers_version()
    try:
        corrector = vec2text.load_pretrained_corrector("gtr-base")
    except AttributeError:
        from vec2text.api import load_pretrained_corrector
        corrector = load_pretrained_corrector("gtr-base")
    tok = corrector.embedder_tokenizer
    print("[vec2text] corrector loaded", flush=True)

    f1s, cers, exact, tlens = [], [], [], []
    t_inv = time.time()
    import jiwer
    for bi in range(0, len(texts), args.batch):
        batch = texts[bi:bi + args.batch]
        rec = vec2text.invert_strings(
            batch, corrector=corrector,
            num_steps=args.num_steps, sequence_beam_width=args.beam)
        for orig, hyp in zip(batch, rec):
            tlens.append(len(tok.tokenize(orig)))
            o = [t for t in tok.tokenize(orig.lower()) if t.strip("▁")]
            h = [t for t in tok.tokenize(str(hyp).lower()) if t.strip("▁")]
            if not o:
                continue
            o_set, h_set = set(o), set(h)
            tp = len(o_set & h_set)
            prec = tp / len(h_set) if h_set else 0.0
            rec_ = tp / len(o_set) if o_set else 0.0
            f1s.append(2 * prec * rec_ / (prec + rec_)
                       if (prec + rec_) > 0 else 0.0)
            cers.append(jiwer.cer(orig, str(hyp)))
            exact.append(1.0 if o == h else 0.0)
        done = min(bi + args.batch, len(texts))
        print(f"    [invert] {done}/{len(texts)}  "
              f"({(time.time() - t_inv) / 60:.1f} min)", flush=True)
        import numpy as np
        dump(args.out, {"prereg": PREREG, "n": len(f1s),
                        "result": {"token_f1_mean": float(np.mean(f1s)),
                                   "cer_mean": float(np.mean(cers)),
                                   "exact_ratio": float(np.mean(exact))},
                        "done": False})

    import numpy as np
    res = {"prereg": PREREG, "n": len(f1s),
           "result": {"token_f1_mean": float(np.mean(f1s)),
                      "token_f1_std": float(np.std(f1s)),
                      "cer_mean": float(np.mean(cers)),
                      "exact_ratio": float(np.mean(exact))},
           "per_sample_f1": [round(float(x), 6) for x in f1s],
           "per_text_token_lengths": [int(x) for x in tlens],
           "minutes": round((time.time() - t0) / 60, 1),
           "done": True}
    dump(args.out, res)
    print(f"★ B3 {args.corpus} s{args.seed} "
          f"token-F1={res['result']['token_f1_mean']:.4f}"
          f"±{res['result']['token_f1_std']:.4f} → [saved] {args.out}"
          f" ({res['minutes']} min)")

    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        dst = os.path.join(args.copy_to, os.path.basename(args.out))
        shutil.copyfile(args.out, dst)
        print(f"[copied] {dst}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
