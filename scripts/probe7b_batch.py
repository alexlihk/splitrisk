#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe7b_batch.py — 7B 探針再校準：5 候選×自動錨點分離判定（零互動）

用法：
  # S4 Mistral-7B（HF）
  python scripts/probe7b_batch.py --model mistralai/Mistral-7B-v0.1

  # S8 Llama-3.1-8B（本地路徑）
  python scripts/probe7b_batch.py --model /data/xuguangning/work/splitrisk/models/Llama-3.1-8B/

  # 縮樣驗證（先跑這個確認管道）
  python scripts/probe7b_batch.py --model <model> --n-train 500 --n-test 200

輸出：results/h20/probe7b_<model>.json
判定：full_info_t1 > ambient_pctx + 0.15 = 錨點分離 = PASS
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


# ============================================================
# 候選 z 變換（全用 train 統計，test 同步；每個回傳 z_tr, z_te, am_tr, am_te）
# ============================================================
def t_baseline(z_tr, z_te, am_tr, am_te):
    return z_tr, z_te, am_tr, am_te

def t_per_dim_norm(z_tr, z_te, am_tr, am_te):
    """Fix#2: 每特徵維 z-score（train 統計）"""
    mu = z_tr.reshape(-1, z_tr.shape[-1]).mean(0)
    sd = z_tr.reshape(-1, z_tr.shape[-1]).std(0) + 1e-8
    return (z_tr - mu) / sd, (z_te - mu) / sd, am_tr, am_te

def t_clip_z(z_tr, z_te, am_tr, am_te):
    """per-dim norm 後 clip 到 [-5,5]（更激進）"""
    zt, ze, _, _ = t_per_dim_norm(z_tr, z_te, am_tr, am_te)
    return zt.clamp(-5, 5), ze.clamp(-5, 5), am_tr, am_te

def t_exclude_pos0(z_tr, z_te, am_tr, am_te):
    """排除 position 0（attention sink 位）"""
    am_tr2 = am_tr.clone(); am_tr2[:, 0] = 0
    am_te2 = am_te.clone(); am_te2[:, 0] = 0
    return z_tr, z_te, am_tr2, am_te2

def t_norm_plus_excl(z_tr, z_te, am_tr, am_te):
    """per-dim norm ＋ exclude pos 0（組合）"""
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
# 探針（輕量 CE probe，probe 維度自動適配 z 維度）
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


def train_probe(z_tr, tgt_tr, wte, epochs=2, lr=1e-3, bs=32, device="cpu"):
    torch.manual_seed(1006)
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
    wte_n = F.normalize(wte, dim=-1)   # Fix#2: F.normalize 更穩
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


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF id 或本地路徑（均可）")
    ap.add_argument("--split", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=2000,
                    help="縮樣 2000；全規模 5000")
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 載入模型（本地路徑或 HF id 均可）
    print(f"[load] {args.model}", flush=True)
    full = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
    full = full.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 資料
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    def enc(split, n):
        d = ds[split].select(range(n))
        e = tok(list(d["text"]), max_length=64, truncation=True,
                padding="max_length", return_tensors="pt")
        return e["input_ids"], e["attention_mask"], torch.tensor(list(d["label"]))

    ids_tr, am_tr, y_tr = enc("train", args.n_train)
    ids_te, am_te, y_te = enc("test", args.n_test)

    # AutoSplitter 切分
    from splitrisk.splitting import AutoSplitter
    head, tail, full = AutoSplitter.load(full, args.split, device=dev)

    # 快取 z（train + test）
    @torch.no_grad()
    def cache_z(ids, am):
        zs = []
        head.eval()
        for i in range(0, ids.shape[0], 64):
            zs.append(head(ids[i:i+64].to(dev),
                          am[i:i+64].to(dev)).cpu())
        return torch.cat(zs)

    z_tr = cache_z(ids_tr, am_tr)
    z_te = cache_z(ids_te, am_te)
    tgt_tr = ids_tr.clone(); tgt_tr[am_tr == 0] = -100
    tgt_te = ids_te.clone(); tgt_te[am_te == 0] = -100

    # 取 wte（嵌入矩陣）
    wte = full.get_output_embeddings().weight.detach().cpu()
    dim = z_te.shape[-1]
    print(f"[data] train={z_tr.shape[0]} test={z_te.shape[0]} "
          f"dim={dim} dev={dev}", flush=True)

    # ambient
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

    # 逐候選跑
    safe_name = args.model.replace("/", "-").replace("\\", "-")
    res = {"model": args.model, "split": args.split, "pctx": pctx,
           "dim": dim, "candidates": {}}
    n_pass = 0

    for name, fn in CANDIDATES:
        try:
            t0c = time.time()
            zt, ze, at, ae = fn(
                z_tr.clone(), z_te.clone(), am_tr.clone(), am_te.clone())
            # tgt 也要過濾掉被排除的 position（exclude_pos0 時）
            tt = tgt_te.clone()
            tt[ae == 0] = -100
            tt_tr = tgt_tr.clone()
            tt_tr[at == 0] = -100

            t1s, t5s = [], []
            for seed in range(3):
                torch.manual_seed(1006 + seed)
                net = train_probe(zt.float(), tt_tr, wte,
                                  epochs=args.epochs, device=dev)
                t1, t5 = eval_probe(net, ze.float(), tt, wte, ae, device=dev)
                t1s.append(t1); t5s.append(t5)

            fl = raw_floor(ze.float(), tt, wte, ae, device=dev)
            t1m, t5m = float(np.mean(t1s)), float(np.mean(t5s))
            t1sd = float(np.std(t1s))
            sep = t1m > pctx + 0.15
            if sep:
                n_pass += 1

            res["candidates"][name] = {
                "t1_mean": round(t1m, 4), "t1_std": round(t1sd, 4),
                "t5_mean": round(t5m, 4), "floor": round(fl, 4),
                "anchor_separated": sep, "per_seed_t1": [round(t, 4) for t in t1s],
            }
            print(f"  [{name}] t1={t1m:.4f}±{t1sd:.4f} t5={t5m:.4f} "
                  f"floor={fl:.4f} → {'PASS' if sep else 'FAIL'} "
                  f"({(time.time()-t0c)/60:.1f} min)", flush=True)

            del net, zt, ze
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            res["candidates"][name] = {"error": str(e)}
            print(f"  [{name}] ERROR: {e}", flush=True)
            traceback.print_exc()

    res["n_pass"] = n_pass
    res["verdict"] = (f"{n_pass}/5 candidates achieved anchor separation"
                      if n_pass > 0 else
                      "0/5 — 儀器邊界確認，此尺度維持 REVIEW")
    res["done"] = True
    res["minutes"] = round((time.time() - t0) / 60, 1)

    out = os.path.join(OUT, f"probe7b_{safe_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n★ {res['verdict']}")
    print(f"[saved] {out} ({res['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
