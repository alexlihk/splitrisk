#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""colab_embed_floor.py — Batch 4: zero-info floors + matched controls (v6.1)
v2 (2026-10-19): full event logging — every stage timestamped, log file
written next to the output (Drive) so failures leave evidence."""
import argparse, json, os, random, shutil, sys, time, traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_embed_e1 import (dump, selfheal_vec2text, selfinstall_deps,
                            check_transformers_version)

PREREG_DEF = {
    "purpose": "Batch 4 — per-audit zero-info floors + matched non-information "
               "controls (v6.1)",
    "protocol": "same inversion stack; inputs carry NO text information; "
                "scored set-based token-F1 against corpus texts",
    "input_modes": {"random": "uniform-random unit vectors",
                    "gauss": "Gaussian unit vectors (norm/distribution-matched "
                             "non-information control)"},
}

LOGF = None


def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts} [{level}] {msg}"
    print(line, flush=True)
    if LOGF:
        try:
            LOGF.write(line + "\n")
            LOGF.flush()
        except Exception:
            pass


def load_corpus(name, seed, n):
    log(f"loading corpus={name} seed={seed} n={n}")
    from datasets import load_dataset
    if name == "agnews":
        ds = load_dataset("fancyzhx/ag_news")["test"]
        rng = random.Random(seed)
        idx = rng.sample(range(len(ds)), min(n, len(ds)))
        texts = [ds[i]["text"][:160] for i in idx]
    elif name == "enron":
        ds = load_dataset("SetFit/enron_spam")["train"]
        rng = random.Random(seed)
        idx = rng.sample(range(len(ds)), min(n, len(ds)))
        texts = []
        for i in idx:
            body = ds[i].get("text") or ds[i].get("email") or ""
            body = " ".join(str(body).split())[:160]
            texts.append(body)
        texts = [x for x in texts if len(x) >= 40]
    else:
        raise SystemExit(f"unknown corpus {name}")
    log(f"corpus loaded: {len(texts)} texts "
        f"(min={min(len(t) for t in texts)} chars)", level="OK")
    return texts


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
    log(f"inputs built: mode={mode} shape={V.shape} "
        f"norm_mean={float(np.linalg.norm(V, axis=1).mean()):.4f}", level="OK")
    return V


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["agnews", "enron"], default="enron")
    ap.add_argument("--input-mode", choices=["random", "gauss"], default="random")
    ap.add_argument("--seed", type=int, default=1006)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="")
    ap.add_argument("--copy-to", default="",
                    help="Drive dir; script log + JSON copied here")
    args = ap.parse_args()
    args.batch = max(8, min(args.batch, 32 // max(1, args.beam)))

    if not args.out:
        args.out = (f"/content/e2_{args.corpus}_{args.input_mode}"
                    f"_b{args.num_steps}{args.beam}.json")
    logdir = args.copy_to or os.path.dirname(args.out) or "/content"
    try:
        os.makedirs(logdir, exist_ok=True)
    except OSError:
        import subprocess
        subprocess.run(["mkdir", "-p", logdir], check=False)
    LOGF = open(os.path.join(logdir, os.path.basename(args.out) + ".log"),
                "a", encoding="utf-8")
    log(f"=== start corpus={args.corpus} mode={args.input_mode} "
        f"budget={args.num_steps}/{args.beam} n={args.n} batch={args.batch} "
        f"out={args.out} copy_to={args.copy_to}")

    PREREG = dict(PREREG_DEF)
    PREREG.update({"corpus": args.corpus, "input_mode": args.input_mode,
                   "n": args.n,
                   "budget": {"num_steps": args.num_steps,
                              "sequence_beam_width": args.beam}})
    t0 = time.time()

    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            if json.load(f).get("done"):
                log("resume: already done, skipping")
                return

    log("importing torch...")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        log(f"device=cuda name={torch.cuda.get_device_name(0)} "
            f"mem={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else:
        log("device=cpu")
    log(f"versions: torch={torch.__version__}")

    texts = load_corpus(args.corpus, args.seed, args.n)

    log("self-healing vec2text + importing...")
    selfheal_vec2text()
    selfinstall_deps()
    import vec2text
    check_transformers_version()
    log(f"vec2text {getattr(vec2text, '__version__', '?')} imported")
    log("loading corrector (gtr-base)...")
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
    log(f"corrector loaded (embedder dim={dim})", level="OK")

    inputs = make_inputs(args.input_mode, args.n, dim, seed=args.seed + 5000)

    f1s, cers, per_sample = [], [], []
    t_inv = time.time()
    import jiwer
    import numpy as np
    for bi in range(0, len(inputs), args.batch):
        bt = time.time()
        bvec = torch.tensor(inputs[bi:bi + args.batch]).to(dev)
        btxt = [texts[(bi + k) % len(texts)] for k in range(len(bvec))]
        rec = vec2text.invert_embeddings(
            embeddings=bvec.float(), corrector=corrector,
            num_steps=args.num_steps, sequence_beam_width=args.beam)
        for ref_text, hyp in zip(btxt, rec):
            o = [t for t in tok.tokenize(ref_text.lower()) if t.strip("▁")]
            h = [t for t in tok.tokenize(str(hyp).lower()) if t.strip("▁")]
            if not o:
                log("skipped empty-token reference", level="WARN")
                continue
            o_set, h_set = set(o), set(h)
            tp = len(o_set & h_set)
            prec = tp / len(h_set) if h_set else 0.0
            rec_ = tp / len(o_set) if o_set else 0.0
            f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
            f1s.append(f1); per_sample.append(round(f1, 6))
            cers.append(jiwer.cer(ref_text, str(hyp)))
        done = min(bi + args.batch, len(inputs))
        dt = time.time() - bt
        warn = "  <-- SLOW" if dt > 120 else ""
        log(f"batch {bi//args.batch + 1}: {done}/{len(inputs)} in {dt:.1f}s{warn}")
    log(f"inversion loop done: n={len(f1s)} "
        f"total {(time.time()-t_inv)/60:.1f} min", level="OK")

    res = {"prereg": PREREG, "n": len(f1s),
           "result": {"floor_f1_mean": float(np.mean(f1s)),
                      "floor_f1_std": float(np.std(f1s)),
                      "cer_mean": float(np.mean(cers))},
           "per_sample_floor_f1": per_sample,
           "minutes": round((time.time() - t0) / 60, 1),
           "done": True}
    dump(args.out, res)
    log(f"result saved: floor F1={res['result']['floor_f1_mean']:.4f}"
        f"±{res['result']['floor_f1_std']:.4f} -> {args.out}", level="OK")
    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        dst = os.path.join(args.copy_to, os.path.basename(args.out))
        shutil.copyfile(args.out, dst)
        shutil.copyfile(os.path.join(
            logdir, os.path.basename(args.out) + ".log"),
            os.path.join(args.copy_to, os.path.basename(args.out) + ".log"))
        log(f"copied JSON+log to {args.copy_to}", level="OK")
    log(f"=== done in {(time.time()-t0)/60:.1f} min", level="OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if LOGF:
            try:
                LOGF.write(tb + "\n")
                LOGF.flush()
            except Exception:
                pass
        sys.exit(1)
