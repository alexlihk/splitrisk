#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""splitrisk.embed — Q3 象限：閉源 embedding 端點洩漏測量（v0.3 預覽）

測什麼：廠商把中間層（向量端點）暴露給你，攻擊者拿向量能撈回多少原文。
攻擊者模型：pool-retrieval（needle-in-haystack）——手握候選語料池 + cosine top-1。
  這是「檢索族」讀數；生成式反演族（vec2text 類）天花板更高、另行測量。

儀器校準（SA-RI 四錨點方法的 embedding 移植）：
  floor = run 內隨機配對 F1（embedding 無洩漏時 top-1=任意池文本的機率地板，
  與讀數同單位，免跨儀器比較）
  verdict 三檔（provisional v0 preset，錨點流成熟前僅供路由參考）：
    RED  excess >= +0.30  短敏感文本不應入此端點
    AMBER +0.10 <= excess < +0.30  有實質洩漏面，附保鮮期
    GREEN excess < +0.10  對池攻擊者魯棒（≠無風險——生成式反演族未測）

誠實邊界（不可關閉）：
  1. 測的是端點不是廠商——報告綁定 endpoint+model，廠商升級模型＝重測觸發
  2. 攻擊者容量＝檢索族；生成式反演族天花板開放
  3. 查詢即費用——n_queries 真實披露
  4. 長度分桶強制——單一數字＝灌水/低估
  5. 逐廠商 ToS 自查（反演/檢索類用法是否被禁）
  6. 本報告 UNSIGN——簽章測量報告是 SplitAudit 商業層
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

# provisional v0 verdict thresholds（錨點流成熟前僅供路由參考）
VERDICT_RED = 0.30
VERDICT_AMBER = 0.10

_TOKENIZER = None


def _gtr_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        # jxm/gtr__nq__32 config 無 model_type，AutoTokenizer 不認——
        # 用 embedder_tokenizer 本體（與 E1/E3 儀器鏈同一約定）
        _TOKENIZER = AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base")
    return _TOKENIZER


def _tok_set(text, tokenizer_fn=None):
    tf = tokenizer_fn or (lambda s: _gtr_tokenizer().tokenize(str(s).lower()))
    return {t for t in tf(text) if t.strip("▁")}


def pair_f1(orig_texts, hyp_texts, tokenizer_fn=None):
    """set-based token-F1，與 E1/E3 儀器鏈同一約定。"""
    f1s = []
    for o_text, h_text in zip(orig_texts, hyp_texts):
        o, h = _tok_set(o_text, tokenizer_fn), _tok_set(h_text, tokenizer_fn)
        if not o:
            continue
        tp = len(o & h)
        p = tp / len(h) if h else 0.0
        r = tp / len(o) if o else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return f1s


def fetch_embeddings(texts, endpoint, model, api_key, batch=25, sleep=1.5):
    """HTTP 查詢向量端點（OpenAI-ish /embeddings 形態；429 指數退避）。"""
    import requests
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    vecs = []
    for bi in range(0, len(texts), batch):
        chunk = texts[bi:bi + batch]
        for attempt in range(6):
            r = requests.post(endpoint, headers=headers, timeout=60,
                              json={"model": model, "input": chunk})
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  [429] wait {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()["data"]
            data.sort(key=lambda d: d["index"])
            vecs.extend([d["embedding"] for d in data])
            break
        else:
            raise RuntimeError(f"rate-limited beyond retries at row {bi}")
        time.sleep(sleep)
        done = min(bi + batch, len(texts))
        if done % 500 < batch or done == len(texts):
            print(f"  [fetch] {done}/{len(texts)}", flush=True)
    return np.array(vecs, dtype=np.float32)


def pseudo_embeddings(texts, dim=2048):
    """--dry-run 用：確定性偽向量（管道驗證，不是測量）。"""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h * (dim // len(h) + 1), dtype=np.uint8)[:dim]
        out[i] = v.astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


def pool_retrieval_attack(target_vecs, pool_vecs, target_texts, pool_texts,
                          seed=0, tokenizer_fn=None):
    """檢索族攻擊器：cosine top-1 + run 內隨機配對地板（同單位）。"""
    tv = target_vecs.astype(np.float32)
    pv = pool_vecs.astype(np.float32)
    tv /= np.linalg.norm(tv, axis=1, keepdims=True) + 1e-9
    pv /= np.linalg.norm(pv, axis=1, keepdims=True) + 1e-9
    sims = tv @ pv.T
    top = sims.argmax(1)
    hit_texts = [pool_texts[j] for j in top]
    f1s = pair_f1(target_texts, hit_texts, tokenizer_fn)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pv))
    floor_f1s = pair_f1(target_texts,
                        [pool_texts[j] for j in perm[:len(tv)]], tokenizer_fn)
    m, sd = float(np.mean(f1s)), float(np.std(f1s))
    fm = float(np.mean(floor_f1s))
    return {"attacker_model": "pool-retrieval (cosine top-1)",
            "n": len(f1s), "n_pool": len(pv),
            "token_f1_mean": m, "token_f1_std": sd,
            "floor_random_pairing_f1": fm,
            "excess_over_floor": m - fm,
            "mean_top1_cosine": float(np.mean(sims.max(1)))}


def verdict_of(excess):
    if excess >= VERDICT_RED:
        return "RED"
    if excess >= VERDICT_AMBER:
        return "AMBER"
    return "GREEN"


def run_embed_audit(endpoint, model, api_key=None, data=None, data_file=None,
                    text_column="text", n=500, pool=10000, batch=25,
                    seed=1006, dry_run=False, tokenizer_fn=None,
                    targets=None, pool_texts=None):
    """端到端：語料 → 向量 → 檢索攻擊 → 地板 → verdict → schema 報告。
    targets/pool_texts 可直接傳入（跳過語料載入；庫用法與離線測試路徑）。"""
    if targets is None or pool_texts is None:
        targets, pool_texts = _load_corpus(data, data_file, text_column,
                                           n, pool, seed)
    n = len(targets)
    if dry_run:
        tv = pseudo_embeddings(targets)
        pv = pseudo_embeddings(pool_texts)
        model_name = f"pseudo(dry-run) dim={tv.shape[1]}"
        n_queries = 0
    else:
        if not api_key:
            raise SystemExit("[!] api key 缺——設 --api-key-env 指到的環境變數")
        all_texts = targets + pool_texts
        V = fetch_embeddings(all_texts, endpoint, model, api_key, batch)
        tv, pv = V[:n], V[n:]
        model_name = model
        n_queries = len(all_texts)

    res = pool_retrieval_attack(tv, pv, targets, pool_texts,
                                tokenizer_fn=tokenizer_fn)
    res.update({
        "endpoint": endpoint, "model": model_name,
        "budget": {"n_queries": n_queries,
                   "note": "all embedding-API queries = attacker cost, "
                           "honest disclosure"},
        "verdict": verdict_of(res["excess_over_floor"]),
        "verdict_preset": f"embed-retrieval v0 (provisional: "
                          f"RED>={VERDICT_RED}, AMBER>={VERDICT_AMBER})",
        "honesty": [
            "measurement, not a defense",
            "endpoint+model bound - vendor model upgrade = re-audit trigger",
            "attacker capacity = retrieval family; generative inversion "
            "(vec2text-class) ceiling open, measured separately",
            "length bucketing mandatory - single unbucketed number inflates "
            "or deflates",
            "check your vendor ToS for retrieval/inversion-style usage",
            "UNSIGN - signed measurement reports are the SplitAudit "
            "commercial layer",
        ],
    })
    return res


def _load_corpus(data, data_file, text_column, n, pool, seed):
    """targets：--data-file(jsonl) 或內建 ag_news test(seed)；pool：ag_news train。"""
    if data_file:
        targets = []
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                targets.append(str(json.loads(line)[text_column])[:160])
                if len(targets) >= n:
                    break
    else:
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news")["test"]
        rng = random.Random(seed)
        targets = [ds[i]["text"][:160]
                   for i in rng.sample(range(len(ds)), min(n, len(ds)))]
    from datasets import load_dataset
    dtr = load_dataset("fancyzhx/ag_news")["train"]
    rng2 = random.Random(seed + 1)
    pool_texts = [dtr[i]["text"][:160]
                  for i in rng2.sample(range(len(dtr)), min(pool, len(dtr)))]
    return targets, pool_texts


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="splitrisk embed",
        description="Measure what a closed-source embedding endpoint leaks "
                    "to a pool-retrieval attacker. Measurement — not a defense.")
    ap.add_argument("--endpoint",
                    default="https://open.bigmodel.cn/api/paas/v4/embeddings")
    ap.add_argument("--model", default="embedding-3")
    ap.add_argument("--api-key-env", default="VENDOR_API_KEY",
                    help="env var holding the vendor key (key never in argv)")
    ap.add_argument("--data-file", default=None,
                    help="jsonl of target texts (uses --text-column); "
                         "default = ag_news test seed 1006")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pool", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1006)
    ap.add_argument("--dry-run", action="store_true",
                    help="pseudo vectors - plumbing check, NOT a measurement")
    ap.add_argument("--out-prefix", default="embed_audit")
    args = ap.parse_args(argv)

    key = None
    if not args.dry_run:
        key = os.environ.get(args.api_key_env)
        if not key:
            print(f"[!] env {args.api_key_env} 未設（key 走環境變數，不進 argv）")
            return 2
    res = run_embed_audit(endpoint=args.endpoint, model=args.model,
                          api_key=key, data_file=args.data_file,
                          text_column=args.text_column, n=args.n,
                          pool=args.pool, batch=args.batch, seed=args.seed,
                          dry_run=args.dry_run)
    print(f"★ embed retrieval F1={res['token_f1_mean']:.4f}"
          f"±{res['token_f1_std']:.4f}  "
          f"floor={res['floor_random_pairing_f1']:.4f}  "
          f"excess={res['excess_over_floor']:+.4f}  → {res['verdict']}")
    jp = args.out_prefix + ".json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[saved] {jp}")
    print("[unsign] signed measurement reports = SplitAudit commercial layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
