#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_embed_floor.py — Batch 4: zero-info floors + matched controls (v6.1 彈藥)

=== Colab setup（同 B3）===
!pip install -q vec2text jiwer
!pip install -q transformers==4.46.3
from google.colab import drive; drive.mount('/content/drive')

=== 跑法（子進程迴圈；單 run 約 1-3 分）===
import sys, subprocess
DRIVE = "/content/drive/MyDrive/B3_results"
runs = [
    # Enron 幻覺地板（隨機向量 × 3 預算，對 Enron 語料評分）
    ("enron",  "random", 2, 1), ("enron",  "random", 4, 2), ("enron",  "random", 4, 4),
    # AG News matched controls（高斯單位向量：範數分佈匹配但零文本信息）
    ("agnews", "gauss",  2, 1), ("agnews", "gauss",  4, 2), ("agnews", "gauss",  4, 4),
]
for corpus, mode, steps, beam in runs:
    subprocess.run([sys.executable, "/content/drive/MyDrive/Colab/colab_embed_floor.py",
                    "--corpus", corpus, "--input-mode", mode,
                    "--num-steps", str(steps), "--beam", str(beam),
                    "--n", "120", "--copy-to", "/content/drive/MyDrive/B3_results"])

目的（預註冊）：
  A. Enron zero-info floor：補齊自家協議條款（per-audit 重算）——Enron 梯度
     從 raw score 升級為 calibrated excess。
  B. AG News matched controls：高斯單位向量（分佈匹配、零文本信息）vs
     uniform-random 向量的地板對比——回應「地板是不是向量分佈造成的」。
輸出：JSON 含 floor F1 mean±sd＋逐樣本分數。
"""
import argparse, json, os, random, shutil, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_embed_e1 import (dump, selfheal_vec2text, selfinstall_deps,
                            check_transformers_version)

PREREG = {
    "purpose": "Batch 4 — per-audit zero-info floors + matched non-information "
               "controls (v6.1)",
    "protocol": "same inversion stack; inputs carry NO text information; "
                "scored set-based token-F1 against corpus texts",
    "input_modes": {"random": "uniform-random unit vectors",
                    "gauss": "Gaussian unit vectors (norm-matched, "
                             "distribution-matched non-information control)"},
}


PREREG_DEF = {
    "purpose": "Batch 4 — per-audit zero-info floors + matched non-information "
               "controls (v6.1)",
    "protocol": "same inversion stack; inputs carry NO text information; "
                "scored set-based token-F1 against corpus texts",
    "input_modes": {"random": "uniform-random unit vectors",
                    "gauss": "Gaussian unit vectors (norm/distribution-matched "
                             "non-information control)"},
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
        return [x for x in texts if len(x) >= 40]
    raise SystemExit(f"unknown corpus {name}")


def make_inputs(mode, n, dim, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    if mode == "random":
        V = rng.random((n, dim)).astype(np.float32)
    elif mode == "gauss":
        V = rng.standard_normal((n, dim)).astype(np.float32)
    else:
        raise SystemExit(f"unknown input-mode {mode}")
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["agnews", "enron"], default="enron")
    ap.add_argument("--input-mode", choices=["random", "gauss"], default="random")
    ap.add_argument("--seed", type=int, default=1006,
                    help="corpus scoring seed; input seed = corpus_seed + 5000")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="")
    ap.add_argument("--copy-to", default="",
                    help="copy finished JSON here (Drive) — /content is "
                         "ephemeral")
    args = ap.parse_args()
    args.batch = max(8, min(args.batch, 32 // max(1, args.beam)))

    if not args.out:
        args.out = (f"/content/e2_{args.corpus}_{args.input_mode}"
                    f"_b{args.num_steps}{args.beam}.json")
    PREREG = dict(PREREG_DEF)
    PREREG.update({"corpus": args.corpus, "input_mode": args.input_mode,
                   "n": args.n,
                   "budget": {"num_steps": args.num_steps,
                              "sequence_beam_width": args.beam}})
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

    texts = load_corpus(args.corpus, args.seed, args.n)
    print(f"[data] scoring corpus: {len(texts)} texts", flush=True)

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
    try:
        dim = corrector.embedder.config.hidden_size
    except AttributeError:
        dim = 768
    print(f"[vec2text] corrector loaded (embedder dim={dim})", flush=True)

    inputs = make_inputs(args.input_mode, args.n, dim,
                         seed=args.seed + 5000)

    f1s, cers = [], []
    t_inv = time.time()
    import jiwer
    import numpy as np
    per_sample = []
    for bi in range(0, len(inputs), args.batch):
        bvec = torch.tensor(inputs[bi:bi + args.batch]).to(dev)
        btxt = [texts[(bi + k) % len(texts)] for k in range(len(bvec))]
        rec = vec2text.invert_embeddings(
            embeddings=bvec.float(), corrector=corrector,
            num_steps=args.num_steps, sequence_beam_width=args.beam)
        for ref_text, hyp in zip(btxt, rec):
            o = [t for t in tok.tokenize(ref_text.lower()) if t.strip("▁")]
            h = [t for t in tok.tokenize(str(hyp).lower()) if t.strip("▁")]
            if not o:
                continue
            o_set, h_set = set(o), set(h)
            tp = len(o_set & h_set)
            prec = tp / len(h_set) if h_set else 0.0
            rec_ = tp / len(o_set) if o_set else 0.0
            f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
            f1s.append(f1); per_sample.append(round(f1, 6))
            cers.append(jiwer.cer(ref_text, str(hyp)))
        done = min(bi + args.batch, len(inputs))
        print(f"    [invert] {done}/{len(inputs)}  "
              f"({(time.time() - t_inv) / 60:.1f} min)", flush=True)

    res = {"prereg": PREREG, "n": len(f1s),
           "result": {"floor_f1_mean": float(np.mean(f1s)),
                      "floor_f1_std": float(np.std(f1s)),
                      "cer_mean": float(np.mean(cers))},
           "per_sample_floor_f1": per_sample,
           "minutes": round((time.time() - t0) / 60, 1),
           "done": True}
    dump(args.out, res)
    print(f"★ floor {args.corpus}/{args.input_mode} "
          f"F1={res['result']['floor_f1_mean']:.4f}"
          f"±{res['result']['floor_f1_std']:.4f} → [saved] {args.out}")
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
