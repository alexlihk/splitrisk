#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe7b_batch2.py — 第二期：註冊預算真 3-seed 重跑 ＋ 預算階梯（零互動，一次全跑）

對象（四個模型，命令見 PROBE7B_GUIDE.md）：
  mistralai/Mistral-7B-v0.1
  /data/xuguangning/work/splitrisk/models/Llama-3.1-8B/
  Qwen/Qwen2-1.5B
  google/gemma-2-2b

協議（預註冊）：
  Stage A  5 候選 × 3 真 seeds，註冊預算（2 epochs / 2000 訓練樣本）
           ——第一期 JSON 的 seeds 因腳本 bug 相同，本期為真多 seed 讀數
  Stage B  預算階梯（per_dim_norm、clip_z 兩候選）：
           L2 = 6 epochs / 5000；L3 = 12 epochs / 5000
  判定閾不變：t1_mean > P_ctx + 0.15 = 錨點分離

輸出：results/h20/probe7b2_<model>.json（自動判定，零互動）
"""
import argparse, gc, json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "h20")
os.makedirs(OUT, exist_ok=True)

SEEDS = [1006, 1007, 1008]
MARGIN = 0.15
A_EPOCHS = 2
A_TRAIN = 2000
LADDER = [("L2", 6, 5000), ("L3", 12, 5000)]
LADDER_CANDS = ["per_dim_norm", "clip_z"]


# ============================================================
# 候選 z 變換（與第一期同一套；全用 train 統計）
# ============================================================
def t_baseline(z_tr, z_te, am_tr, am_te):
    return z_tr, z_te, am_tr, am_te

def t_per_dim_norm(z_tr, z_te, am_tr, am_te):
    mu = z_tr.reshape(-1, z_tr.shape[-1]).mean(0)
    sd = z_tr.reshape(-1, z_tr.shape[-1]).std(0) + 1e-8
    return (z_tr - mu) / sd, (z_te - mu) / sd, am_tr, am_te

def t_clip_z(z_tr, z_te, am_tr, am_te):
    zt, ze, _, _ = t_per_dim_norm(z_tr, z_te, am_tr, am_te)
    return zt.clamp(-5, 5), ze.clamp(-5, 5), am_tr, am_te

def t_exclude_pos0(z_tr, z_te, am_tr, am_te):
    am_tr2 = am_tr.clone(); am_tr2[:, 0] = 0
    am_te2 = am_te.clone(); am_te2[:, 0] = 0
    return z_tr, z_te, am_tr2, am_te2

def t_norm_plus_excl(z_tr, z_te, am_tr, am_te):
    zt, ze, _, _ = t_per_dim_norm(z_tr, z_te, am_tr, am_te)
    am_tr2 = am_tr.clone(); am_tr2[:, 0] = 0
    am_te2 = am_te.clone(); am_te2[:, 0] = 0
    return zt, ze, am_tr2, am_te2

CANDIDATES = [
    ("baseline",        t_baseline),
    ("per_dim_norm",    t_per_dim_norm),
    ("clip_z",          t_clip_z),
    ("exclude_pos0",    t_exclude_pos0),
    ("norm_plus_excl",  t_norm_plus_excl),
]


# ============================================================
# 探針
# ============================================================
class Probe7B(nn.Module):
    def __init__(self, dim, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, z):
        return self.net(z)


def train_probe(z_tr, tgt_tr, wte, epochs, lr=1e-3, bs=32, device="cpu",
                seed=1006):
    torch.manual_seed(seed)          # Fix（第一期 bug）：seed 參數化
    dim = z_tr.shape[-1]
    net = Probe7B(dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    wte = wte.to(device).float()
    n = z_tr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            zb = z_tr[idx].to(device).float()
            tb = tgt_tr[idx].to(device)
            rec = net(zb)
            logits = rec @ wte.t()
            loss = lossf(logits.reshape(-1, logits.shape[-1]),
                         tb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def eval_probe(net, z_te, tgt_te, wte, am_te, device="cpu"):
    wte = wte.to(device).float()
    wte_n = F.normalize(wte, dim=-1)
    top1 = top5 = total = 0
    for i in range(0, z_te.shape[0], 64):
        zb = z_te[i:i + 64].to(device).float()
        tb = tgt_te[i:i + 64].to(device)
        m = am_te[i:i + 64].to(device).bool() & (tb != -100)
        rec = net(zb)
        rec_n = F.normalize(rec, dim=-1)
        sim = rec_n @ wte_n.t()
        pred = sim.argmax(-1)
        top5v = sim.topk(5, -1).indices
        top1 += (pred[m] == tb[m]).sum().item()
        top5 += (top5v[m] == tb[m].unsqueeze(-1)).any(-1).sum().item()
        total += m.sum().item()
    return top1 / max(1, total), top5 / max(1, total)


@torch.no_grad()
def raw_floor(z, tgt, wte, am, device="cpu"):
    wte = wte.to(device).float()
    wte_n = F.normalize(wte, dim=-1)
    top1 = total = 0
    for i in range(z.shape[0]):
        zn = z[i].to(device).float()
        zn = F.normalize(zn, dim=-1)
        sim = zn @ wte_n.t()
        pred = sim.argmax(-1)
        tb = tgt[i].to(device)
        m = (tb != -100).to(device)
        if am is not None:
            m = m & am[i].to(device).bool()
        top1 += (pred[m] == tb[m]).sum().item()
        total += m.sum().item()
    return top1 / max(1, total)


def run_candidate(name, fn, z_tr_full, z_te, am_tr_full, am_te,
                  tgt_tr_full, tgt_te, wte, pctx, epochs, n_train_rows, dev):
    """跑一個候選（3 真 seeds），回傳結果 dict（含 gap 與分離判定）"""
    zt, ze, at, ae = fn(
        z_tr_full[:n_train_rows].clone(), z_te.clone(),
        am_tr_full[:n_train_rows].clone(), am_te.clone())
    tt = tgt_te.clone(); tt[ae == 0] = -100
    tt_tr = tgt_tr_full[:n_train_rows].clone(); tt_tr[at == 0] = -100

    t1s, t5s = [], []
    for seed in SEEDS:
        net = train_probe(zt.float(), tt_tr, wte, epochs=epochs,
                          device=dev, seed=seed)
        t1, t5 = eval_probe(net, ze.float(), tt, wte, ae, device=dev)
        t1s.append(t1); t5s.append(t5)
        del net
    fl = raw_floor(ze.float(), tt, wte, ae, device=dev)
    t1m, t5m = float(np.mean(t1s)), float(np.mean(t5s))
    t1sd = float(np.std(t1s))
    sep = t1m > pctx + MARGIN
    return {
        "epochs": epochs, "n_train": int(n_train_rows),
        "t1_mean": round(t1m, 4), "t1_std": round(t1sd, 4),
        "t5_mean": round(t5m, 4), "floor": round(fl, 4),
        "gap_vs_pctx": round(t1m - pctx, 4),
        "anchor_separated": bool(sep),
        "per_seed_t1": [round(t, 4) for t in t1s],
    }, sep


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF id 或本地路徑（均可）")
    ap.add_argument("--split", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=5000,
                    help="Stage B 階梯用 5000；縮樣冒煙可調小")
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    n_train_cache = max(args.n_train, A_TRAIN)   # 快取足夠 Stage A+B 用

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[load] {args.model}", flush=True)
    full = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    full = full.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    def enc(split, n):
        d = ds[split].select(range(n))
        e = tok(list(d["text"]), max_length=64, truncation=True,
                padding="max_length", return_tensors="pt")
        return e["input_ids"], e["attention_mask"], torch.tensor(list(d["label"]))

    ids_tr, am_tr, y_tr = enc("train", n_train_cache)
    ids_te, am_te, y_te = enc("test", args.n_test)

    from splitrisk.splitting import AutoSplitter
    head, tail, full = AutoSplitter.load(full, args.split, device=dev)

    @torch.no_grad()
    def cache_z(ids, am):
        zs = []
        head.eval()
        for i in range(0, ids.shape[0], 64):
            zs.append(head(ids[i:i+64].to(dev),
                          am[i:i+64].to(dev)).cpu())
        return torch.cat(zs)

    print(f"[cache] z for train={ids_tr.shape[0]} test={ids_te.shape[0]}", flush=True)
    z_tr = cache_z(ids_tr, am_tr)
    z_te = cache_z(ids_te, am_te)
    tgt_tr = ids_tr.clone(); tgt_tr[am_tr == 0] = -100
    tgt_te = ids_te.clone(); tgt_te[am_te == 0] = -100

    wte = full.get_output_embeddings().weight.detach().cpu()
    dim = z_te.shape[-1]
    print(f"[data] train={z_tr.shape[0]} test={z_te.shape[0]} "
          f"dim={dim} dev={dev}", flush=True)

    @torch.no_grad()
    def measure_ambient():
        correct = total = 0
        for i in range(0, ids_te.shape[0], 64):
            ids = ids_te[i:i+64].to(dev)
            am = am_te[i:i+64].to(dev)
            logits = full(input_ids=ids, attention_mask=am).logits
            pred = logits[:, :-1, :].argmax(-1)
            tgt = ids[:, 1:]
            m = (tgt != -100) & am[:, 1:].bool()
            correct += (pred[m] == tgt[m]).sum().item()
            total += m.sum().item()
        return correct / max(1, total)

    pctx = measure_ambient()
    print(f"[ambient] P_ctx={pctx:.4f}", flush=True)

    safe_name = args.model.replace("/", "-").replace("\\", "-")
    res = {"model": args.model, "split": args.split, "pctx": pctx,
           "dim": dim, "protocol": {
               "margin": MARGIN, "seeds": SEEDS,
               "stageA": f"{A_EPOCHS}ep/{A_TRAIN}",
               "ladder": {k: f"{e}ep/{n}" for k, e, n in LADDER},
               "ladder_candidates": LADDER_CANDS},
           "stageA": {}, "ladder": {}}
    n_pass_a = 0
    ladder_pass = None   # (cand, level, gap) 第一個過線的

    # ---- Stage A：5 候選 × 3 真 seeds，註冊預算 ----
    print("[stageA] registered budget (2ep/2000), real 3 seeds", flush=True)
    a_train_rows = min(A_TRAIN, z_tr.shape[0])
    for name, fn in CANDIDATES:
        try:
            t0c = time.time()
            r, sep = run_candidate(name, fn, z_tr, z_te, am_tr, am_te,
                                   tgt_tr, tgt_te, wte, pctx,
                                   A_EPOCHS, a_train_rows, dev)
            if sep:
                n_pass_a += 1
            res["stageA"][name] = r
            print(f"  [A:{name}] t1={r['t1_mean']:.4f}±{r['t1_std']:.4f} "
                  f"gap={r['gap_vs_pctx']:+.4f} → {'PASS' if sep else 'FAIL'} "
                  f"({(time.time()-t0c)/60:.1f} min)", flush=True)
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            res["stageA"][name] = {"error": str(e)}
            print(f"  [A:{name}] ERROR: {e}", flush=True)
            traceback.print_exc()

    # ---- Stage B：預算階梯（per_dim_norm / clip_z）----
    print("[stageB] budget ladder", flush=True)
    for name in LADDER_CANDS:
        if name not in res["stageA"] or "error" in res["stageA"][name]:
            continue
        fn = dict(CANDIDATES)[name]
        res["ladder"][name] = {}
        for level, ep, n_rows in LADDER:
            try:
                t0c = time.time()
                rows = min(n_rows, z_tr.shape[0])
                r, sep = run_candidate(name, fn, z_tr, z_te, am_tr, am_te,
                                       tgt_tr, tgt_te, wte, pctx,
                                       ep, rows, dev)
                res["ladder"][name][level] = r
                if sep and ladder_pass is None:
                    ladder_pass = (name, level, r["gap_vs_pctx"])
                print(f"  [B:{name}:{level}] t1={r['t1_mean']:.4f}"
                      f"±{r['t1_std']:.4f} gap={r['gap_vs_pctx']:+.4f} "
                      f"→ {'PASS' if sep else 'FAIL'} "
                      f"({(time.time()-t0c)/60:.1f} min)", flush=True)
                gc.collect()
                if dev == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                res["ladder"][name][level] = {"error": str(e)}
                print(f"  [B:{name}:{level}] ERROR: {e}", flush=True)

    # ---- 自動判定 ----
    if n_pass_a > 0:
        res["verdict"] = (f"Stage A {n_pass_a}/5 分離（註冊預算）→ PASS；"
                          f"ladder 首過＝{ladder_pass}")
    elif ladder_pass:
        res["verdict"] = (f"註冊預算 0/5；階梯分離於 {ladder_pass[0]}@"
                          f"{ladder_pass[1]} gap={ladder_pass[2]:+.4f} → "
                          f"邊界隨預算移動（報告必須帶預算向量）")
    else:
        res["verdict"] = (f"0/5 分離至最大預算（L3={LADDER[-1][1]}ep/"
                          f"{LADDER[-1][2]}）→ 儀器邊界在預註冊階梯內確認，"
                          f"此尺度維持 REVIEW")
    res["done"] = True
    res["minutes"] = round((time.time() - t0) / 60, 1)

    out = os.path.join(OUT, f"probe7b2_{safe_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n★ {res['verdict']}")
    print(f"[saved] {out} ({res['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
