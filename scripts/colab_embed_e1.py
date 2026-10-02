#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_embed_e1.py — E1: vec2text 復現驗證（embedding 反演儀器校準）
v3: 本機 Python 3.12 + transformers 4.46.3 端到端跑通（2026-09-28）

=== Colab setup cell（跑腳本前先跑這段）===
!pip install -q vec2text jiwer
!pip install -q transformers==4.46.3   # vec2text 拉了 5.x，必須降回（meta-device bug）
# 然後：from google.colab import drive; drive.mount('/content/drive')
# 最後：%run colab_embed_e1.py --n 500 --out /content/drive/MyDrive/.../e1.json

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
    d = os.path.dirname(path)
    if d:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            # Google Drive FUSE on Colab: Errno 95 (Operation not supported)
            # Fallback: try creating via subprocess, or just write directly
            if e.errno == 95:
                import subprocess
                subprocess.run(["mkdir", "-p", d], check=False)
            else:
                raise
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)


def selfheal_vec2text():
    """上游 experiments.py 的兩個 bug：缺 import platform、缺 platform guard。
    本機 + Colab 均需此 patch（Linux 無 platform guard 不炸，Windows 炸）。"""
    import importlib.util
    spec = importlib.util.find_spec("vec2text")
    if not spec:
        return
    exp_py = os.path.join(os.path.dirname(spec.origin), "experiments.py")
    if not os.path.exists(exp_py):
        return
    src = open(exp_py, encoding="utf-8").read()
    patched = False

    # Patch 1: add import platform if missing
    if "import platform" not in src:
        src = "import platform\n" + src
        patched = True

    # Patch 2: guard bare "import resource" with platform check
    # (resource module only exists on Unix; bare import crashes on Windows)
    if "\nimport resource\n" in src and "platform.system()" not in src:
        src = src.replace("\nimport resource\n",
                          '\nif platform.system() != "Windows":\n    import resource\n')
        patched = True

    if patched:
        with open(exp_py, "w", encoding="utf-8") as f:
            f.write(src)
        print("[self-heal] patched vec2text/experiments.py", flush=True)


def selfinstall_deps():
    """metric 依賴自動補裝（jiwer 缺失是 Colab 上的常見炸點）。"""
    import importlib.util, subprocess
    if importlib.util.find_spec("jiwer") is None:
        print("[self-install] installing jiwer...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"],
                       check=False)


def check_transformers_version():
    """transformers ≥5.0 會觸發 meta-device RuntimeError（vec2text 不兼容）。
    Colab setup cell 必須先降回 4.46.3。"""
    import transformers
    v = transformers.__version__
    major = int(v.split(".")[0])
    if major >= 5:
        print(f"[WARN] transformers {v} ≥5.0 — vec2text 會炸（meta-device bug）。"
              f"請跑：pip install transformers==4.46.3",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="/content/e1_gtrbase.json",
                        help="Local output (use /content/ not Drive — Drive FUSE "
                             "breaks makedirs. Use --copy-to for Drive.)")
    ap.add_argument("--copy-to", default="",
                        help="Optional: copy result JSON to this path after done "
                             "(e.g. /content/drive/MyDrive/.../e1.json)")
    args = ap.parse_args()
    # v3.2: prereg budget 必須反映實跑參數（v3.1 寫死 2/1，E1b 升檔後 JSON 標籤錯位）
    PREREG["budget"] = {"num_steps": args.num_steps,
                        "sequence_beam_width": args.beam}
    PREREG["n"] = args.n
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
    selfinstall_deps()
    print("[vec2text] importing...", flush=True)
    import vec2text
    check_transformers_version()
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
    import jiwer
    tok = corrector.embedder_tokenizer
    for bi in range(0, len(texts), args.batch):
        batch = texts[bi:bi + args.batch]
        rec = vec2text.invert_strings(
            batch, corrector=corrector,
            num_steps=args.num_steps, sequence_beam_width=args.beam)
        for orig, hyp in zip(batch, rec):
            # set-based token F1（SentencePiece ▁ 前綴剝除；序列級 CER 用 jiwer）
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

    # Optional: copy to Drive (avoids FUSE makedirs issues during run)
    if args.copy_to:
        import shutil
        try:
            shutil.copy2(args.out, args.copy_to)
            print(f"[copied] {args.copy_to}")
        except Exception as e:
            print(f"[copy-to failed] {e} — result is still at {args.out}",
                  flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
