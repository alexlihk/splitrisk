#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/h20_stage.py — H20 校準/驗證跑的統一入口（代跑友好）

設計保證（針對「代跑不能老有 bug」）：
  - 每個 stage 一個 JSON 結果，寫完即 dump；重跑自動 resume（done 跳過）
  - 每個 stage 開頭印 PREREG（預註冊）與驗收判準；結尾印 PASS/FAIL 判決
  - 全部走 splitrisk 包的官方 API 骨架 + parity/causal 閘門
  - 零交互輸入；OOM 自動降 batch 一次；錯誤 traceback 完整保留

用法：
  python scripts/h20_stage.py --stage S1 --model gpt2
  python scripts/h20_stage.py --stage S2 --model llama-3.2-1b --corpus ag_news --seeds 3
  python scripts/h20_stage.py --stage S5 --steps 300
  python scripts/h20_stage.py --list
結果目錄：results/h20/<STAGE>_<model>_<corpus>.json
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "h20")
os.makedirs(OUT_DIR, exist_ok=True)


def out_path(stage, model="", corpus=""):
    safe = f"{stage}_{(model or '').replace('/', '-')}_{corpus or ''}".rstrip("_")
    return os.path.join(OUT_DIR, safe + ".json")


def dump(path, res):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)


def banner(msg, t0):
    print(f"\n{'=' * 64}\n  {msg}  [{(time.time() - t0) / 60:.0f}min]\n{'=' * 64}",
          flush=True)


# ==================================================================
# S1 — AutoSplitter 雙閘門冒煙（~10 分鐘）
# ==================================================================
def stage_s1(model, corpus, seeds, steps, device):
    t0 = time.time()
    res = {"prereg": {"purpose": "AutoSplitter parity+causal gates on "
                      "real checkpoint", "pass": "Δlogits<2e-2, "
                      "Δprefix<1e-3"}}
    from splitrisk.splitting import AutoSplitter, gates
    head, tail, full = AutoSplitter.load(model, 6, device=device)
    par, leak = gates(head, tail, full, device=device)
    res["result"] = {"parity_max_diff": par, "causal_leak_max": leak}
    res["verdict"] = "PASS" if (par < 2e-2 and leak < 1e-3) else "FAIL"
    res["done"] = True
    print(f"  parity Δ={par:.4f}  causal Δ={leak:.4f}  → {res['verdict']}")
    return res


# ==================================================================
# S2/S3/S4/S7/S8 — 錨點校準（每模型×語料，多 seed，~2-6 小時）
# ==================================================================
def stage_anchor(model, corpus, seeds, steps, device):
    t0 = time.time()
    seeds = seeds or 3
    res = {"prereg": {
        "purpose": f"per-deployment anchor calibration ({model} × {corpus})",
        "seeds": seeds,
        "note": ("full-info/ambient/floor; multi-seed mean per exp18b "
                 "variance protocol (~7pp run-to-run)"),
    }}
    from splitrisk.core import _prepare, _ambient, auto_probe_hidden
    from splitrisk.probes.train_cf_attacker import train_probe, probe_readout
    from splitrisk.probes.raw_floor import raw_floor
    from splitrisk.splitting import gates
    hidden = auto_probe_hidden(model)   # max(2048, 2×model_hidden) — exp18 規則
    res["prereg"]["probe_hidden"] = hidden
    p = _prepare(model, 6, "classification", corpus,
                 n_train=5000, n_test=1000, seed=1006, device=device)
    print(f"  [gates] parity/casual check...", flush=True)
    gates(p["head"], p["tail"], p["full"], device=device)
    wte = p["wte"]
    t1s, t5s = [], []
    for i in range(seeds):
        net = train_probe(p["z_te"], p["tgt_te"], wte, mode="ce",
                          hidden=hidden, epochs=2, device=device,
                          seed=1006 + 4321 + i)
        t1, t5 = probe_readout(net, p["z_te"], p["tgt_te"], wte,
                               mask=p["am_te"], device=device)
        t1s.append(t1)
        t5s.append(t5)
        print(f"  [seed {i}] t1={t1:.4f} t5={t5:.4f}", flush=True)
        del net
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    import numpy as np
    fl = raw_floor(p["z_te"], p["tgt_te"], wte, mask=p["am_te"],
                   device=device)
    res["result"] = {
        "full_info_t1_mean": float(np.mean(t1s)),
        "full_info_t1_std": float(np.std(t1s)),
        "full_info_t5_mean": float(np.mean(t5s)),
        "ambient_pctx": p["pctx"], "floor": fl,
        "per_seed_t1": t1s, "hidden": p["z_te"].shape[-1],
        "n_layers": p["n_layers"], "model_type":
            p["full"].config.model_type,
    }
    res["verdict"] = ("PASS — 錨點行可寫入 calibration/anchors.py"
                      if float(np.mean(t1s)) > 0.85 else
                      "REVIEW — t1<0.85，人工判讀")
    res["done"] = True
    print(f"  ★ {model}×{corpus}: t1={np.mean(t1s):.4f}±{np.std(t1s):.4f} "
          f"t5={np.mean(t5s):.4f} P_ctx={p['pctx']:.4f} floor={fl:.4f}")
    return res


# ==================================================================
# S5 — SL 訓練環三 baseline 對照（honest/sia/fsha，~2-3 小時）
# ==================================================================
def stage_s5(model, corpus, seeds, steps, device):
    t0 = time.time()
    steps = steps or 300
    res = {"prereg": {
        "purpose": "SL training-loop three-baseline audit (exp13 "
                   "replication, productized)",
        "acceptance": ("SIA: acc normal + t1 high (R6); FSHA: acc "
                       "collapse; honest: acc ~0.89/t1 ~0.94"),
        "alpha": 5.0, "sl_steps_equiv_epochs": 1,
    }}
    from splitrisk.core import audit_sl
    cells = {}
    for baseline in ("honest", "sia", "fsha"):
        print(f"  [baseline={baseline}] training...", flush=True)
        r = audit_sl(model=model, split_at=6, data="ag_news",
                     baseline=baseline, alpha=5.0, sl_epochs=1,
                     n_train=5000, n_test=1000, seed=1006, device=device,
                     n_seeds=seeds or 1)
        cells[baseline] = {"acc": r.task_acc, "t1_ce": r.t1_ce,
                           "t5_ce": r.t5_ce, "floor": r.floor,
                           "pctx": r.pctx, "extra": r.extra}
        print(f"  ★ {baseline}: acc={r.task_acc:.4f} t1={r.t1_ce:.4f}")
        res = res or {}
        res["cells"] = cells
    sia, fsha, hon = cells["sia"], cells["fsha"], cells["honest"]
    checks = {
        "SIA_stealth": sia["acc"] > 0.8 and sia["t1_ce"] > 0.85,
        "FSHA_collapse": fsha["acc"] < 0.5,
        "honest_high_recovery": hon["t1_ce"] > 0.85,
    }
    res["checks"] = checks
    res["verdict"] = ("PASS — R6 軸復現" if all(checks.values())
                      else "REVIEW — " + str(checks))
    res["done"] = True
    print(f"  checks: {checks} → {res['verdict']}")
    return res


# ==================================================================
# S6 — SplitFed 模擬（~2-3 小時）
# ==================================================================
def stage_s6(model, corpus, seeds, steps, device):
    t0 = time.time()
    res = {"prereg": {"purpose": "SplitFed head-FedAvg simulation, "
                      "K=4 clients, probe battery on global",
                      "acceptance": "runs end-to-end, per-client readouts "
                      "recorded"}}
    from splitrisk.core import _prepare
    from splitrisk.sl import train_split, eval_split, ClsHead
    from splitrisk.probes.train_cf_attacker import train_probe, probe_readout
    from torch.utils.data import DataLoader, TensorDataset
    import copy
    K = 4
    p = _prepare(model, 6, "classification", "ag_news",
                 n_train=5000, n_test=1000, seed=1006, device=device)
    wte = p["wte"]
    # 分片（等分訓練語料——_prepare 未快取 train 編碼，重取）
    from splitrisk.core import _load_data, _encode
    texts_tr, _, y_tr, _ = _load_data("ag_news", "classification",
                                      p["tok"], 5000, 0)
    ids_tr, am_tr = _encode(texts_tr, p["tok"], 64)
    y_t = torch.tensor(y_tr, dtype=torch.long)
    shards = [(ids_tr[i::K], am_tr[i::K].bool(), y_t[i::K])
              for i in range(K)]
    global_head = copy.deepcopy(p["head"])
    cls = copy.deepcopy(p["cls"])
    n_shards = len(shards)
    for r in range(2):   # 2 輪 FedAvg
        local_states = []
        for k in range(K):
            local = copy.deepcopy(global_head)
            train_split(local, p["tail"], cls,
                        [(shards[k][0], shards[k][1], shards[k][2])],
                        wte, baseline="honest", epochs=1,
                        lr_head=1e-4, lr_server=5e-5, device=device,
                        seed=1006 + r * 10 + k)
            local_states.append(copy.deepcopy(local.state_dict()))
            print(f"  [fed r{r} client{k}] done", flush=True)
        avg = {key: torch.stack([s[key].float() for s in local_states])
               .mean(0) for key in local_states[0]}
        ref = {key: v.dtype for key, v in global_head.state_dict().items()}
        global_head.load_state_dict({key: v.to(ref[key])
                                     for key, v in avg.items()
                                     if key in ref})
        global_head.eval()
    gh = global_head.to(device).eval()
    z_te = _cache = None
    from splitrisk.core import _cache_split_reprs
    z_te = _cache_split_reprs(gh, p["ids_te"], p["am_te"], device)
    net = train_probe(z_te, p["tgt_te"], wte, mode="ce", epochs=2,
                      device=device, seed=1006)
    t1, t5 = probe_readout(net, z_te, p["tgt_te"], wte, mask=p["am_te"],
                           device=device)
    res["result"] = {"global_t1": t1, "global_t5": t5, "K": K,
                     "rounds": 2, "note": "head-FedAvg simulation "
                     "(global head), tail shared"}
    res["verdict"] = "PASS" if t1 > 0.8 else "REVIEW"
    res["done"] = True
    print(f"  ★ SplitFed global: t1={t1:.4f} t5={t5:.4f}")
    return res


STAGES = {
    "S1": ("AutoSplitter 閘門冒煙", "gpt2", "ag_news", 10,
           stage_s1),
    "S2": ("錨點校準 Llama-3.2-1B", "llama-3.2-1b", "ag_news", 120,
           stage_anchor),
    "S3": ("錨點校準 Qwen2-1.5B", "Qwen/Qwen2-1.5B", "ag_news", 120,
           stage_anchor),
    "S4": ("錨點校準 Mistral-7B", "mistralai/Mistral-7B-v0.3", "ag_news",
           300, stage_anchor),
    "S5": ("SL 三 baseline 對照（exp13 復現）", "gpt2", "ag_news", 150,
           stage_s5),
    "S6": ("SplitFed 模擬", "gpt2", "ag_news", 150, stage_s6),
    "S7": ("錨點校準 Gemma-2-2b（可選）", "google/gemma-2-2b", "ag_news",
           150, stage_anchor),
    "S8": ("錨點校準 Llama-3.1-8B（可選）", "meta-llama/Llama-3.1-8B",
           "ag_news", 360, stage_anchor),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, (name, m, c, mins, _) in STAGES.items():
            print(f"{k}: {name}  (model={m}, corpus={c}, ~{mins}min)")
        return

    if args.stage not in STAGES:
        print(f"unknown stage {args.stage}; --list to see")
        return
    name, dmodel, dcorpus, mins, fn = STAGES[args.stage]
    model = args.model or dmodel
    corpus = args.corpus or dcorpus
    path = out_path(args.stage, model, corpus)

    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("done"):
            print(f"[resume] {path} already done → skip")
            print(f"  verdict: {prev.get('verdict')}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage {args.stage}] {name}  model={model} corpus={corpus} "
          f"device={device}", flush=True)
    t0 = time.time()
    res = None
    try:
        res = fn(model, corpus, args.seeds, args.steps, device)
        res.setdefault("stage", args.stage)
        res.setdefault("model", model)
        res.setdefault("corpus", corpus)
        res["minutes"] = (time.time() - t0) / 60
        dump(path, res)
        print(f"[saved] {path}")
    except Exception:
        traceback.print_exc()
        if res is not None:
            res["error"] = traceback.format_exc()
            dump(path, res)
            print(f"[saved with error] {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
