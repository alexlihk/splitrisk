#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_embed_e1.py — E1: vec2text 復現驗證（embedding 反演儀器校準）
v2: 按上游 master 真實 API 重寫（load_pretrained_corrector + invert_strings；
    舊版誤用 models.load_encoder——PyPI/master 均無此函數，已由 repo 比對確認）

預註冊：
  正確器：load_pretrained_corrector("gtr-base")（HF ckpt jxm/gtr__nq__32(+__correct)）
  語料：ag_news 測試集抽樣（PREREG_N 條，來源文本截斷 ~160 chars ≈ 30-50 token；
        invert_strings 內建 max_length=128 截斷）
  budget：num_steps=2, sequence_beam_width=1（遞歸校正 2 步 + 束寬 1）
  驗收判準：token-F1 ≥ 0.70 → PASS（vec2text 論文區間）
            0.50-0.70 → REVIEW；< 0.50 → FAIL（儀器壞了，禁止出數）
  誠實標註：單儀器讀數——closure（E2 AE）完成前不可作任何結論
輸出：--out 指定的 JSON（寫完即 dump，重跑覆蓋）
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
    "api": "load_pretrained_corrector('gtr-base') + invert_strings "
           "(upstream master, verified against repo source)",
    "bucket": "source texts truncated to ~160 chars; embedder max_length=128",
    "budget": {"num_steps": 2, "sequence_beam_width": 1},
    "acceptance": "token-F1 >= 0.70 PASS / 0.50-0.70 REVIEW / <0.50 FAIL",
    "honesty": "single-instrument readout; no conclusion before E2 closure",
}


def dump(path, res):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)


def selfheal_vec2text():
    """上游 main 的 experiments.py 缺 import platform——import 前自動補。"""
    import importlib.util
    spec = importlib.util.find_spec("vec2text")
    if not spec:
        return
    exp_py = os.path.join(os.path.dirname(spec.origin), "experiments.py")
    if os.path.exists(exp_py):
        src = open(exp_py, encoding="utf-8").read()
        if "import platform" not in src:
            with open(exp_py, "w", encoding="utf-8") as f:
                f.write("import platform\n" + src)
            print("[self-heal] patched experiments.py (missing 'import platform')",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
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

    # --- 語料 ---
    print("[data] loading ag_news test split...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")["test"]
    rng = random.Random(1006)
    idx = rng.sample(range(len(ds)), min(args.n, len(ds)))
    texts = [ds[i]["text"][:160] for i in idx]   # ~30-50 tokens

    # --- 儀器（self-heal 後 import）---
    selfheal_vec2text()
    print("[vec2text] importing...", flush=True)
    import vec2text
    print(f"[vec2text] {getattr(vec2text, '__version__', '?')} "
          f"loading corrector (gtr-base)...", flush=True)
    try:
        corrector = vec2text.load_pretrained_corrector("gtr-base")
    except AttributeError:
        from vec2text.api import load_pretrained_corrector
        corrector = load_pretrained_corrector("gtr-base")
    print("[vec2text] corrector loaded", flush=True)

    # --- 反演（分批；invert_strings 內建 embed + invert）---
    f1s, cers, exact = [], [], []
    t_inv = time.time()
    print(f"[invert] num_steps={args.num_steps} beam={args.beam} "
          f"n={len(texts)} batch={args.batch}...", flush=True)
    import evaluate as hf_evaluate
    f1m = hf_evaluate.load("f1")
    cerm = hf_evaluate.load("cer")
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained("gtr-base")
    for bi in range(0, len(texts), args.batch):
        batch = texts[bi:bi + args.batch]
        rec = vec2text.invert_strings(
            batch, corrector=corrector,
            num_steps=args.num_steps, sequence_beam_width=args.beam)
        for orig, hyp in zip(batch, rec):
            o = tok.tokenize(orig.lower())[:48]
            h = tok.tokenize(str(hyp).lower())[:48]
            if not o:
                continue
            f1s.append(f1m.compute(
                predictions=[h if h else ["<empty>"]], references=[o],
                average="macro")["f1"])
            cers.append(cerm.compute(predictions=[str(hyp)],
                                     references=[orig]))
            exact.append(1.0 if [t.lower() for t in o] == [t.lower() for t in
                         tok.tokenize(str(hyp).lower())] else 0.0)
        done = min(bi + args.batch, len(texts))
        print(f"    [invert] {done}/{len(texts)}  "
              f"({(time.time() - t_inv) / 60:.1f} min)", flush=True)
        # 每批落盤一次（斷點保護）
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
                      "exact_ratio": float(np.mean(exact)),
                      "minutes": (time.time() - t0) / 60,
                      "device": dev},
           "per_sample_f1": f1s}
    m = res["result"]["token_f1_mean"]
    res["verdict"] = ("PASS" if m >= 0.70 else
                      "REVIEW" if m >= 0.50 else "FAIL")
    res["done"] = True
    dump(args.out, res)
    print(f"★ E1 token-F1={m:.4f}±{res['result']['token_f1_std']:.4f} "
          f"CER={res['result']['cer_mean']:.4f} "
          f"exact={res['result']['exact_ratio']:.3f} → {res['verdict']}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
