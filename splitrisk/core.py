"""Core audit pipelines over the probe battery.

One-call entry points (see __init__.py): audit / audit_vq /
audit_sharded / audit_training. All heavy imports are lazy so the
package imports cleanly everywhere; heavy work needs torch +
transformers at call time.
"""
import numpy as np
import torch

from .report import AuditResult
from .splitting import AutoSplitter, gates
from .probes.probe_common import load_gpt2
from .probes.train_cf_attacker import train_probe, probe_readout
from .probes.raw_floor import raw_floor
from .probes.reassembler import train_reassembler
from .probes.label_probe import label_audit
from .vq.vq_cell import VQCell

# 錨點表已校準、可簽章的模型（audit() 名稱路徑的白名單）
SUPPORTED_MODELS = ("gpt2", "gpt2-medium", "gpt2-large",
                    "llama-3.2-1b")


def _device(device):
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def auto_probe_hidden(model) -> int:
    """Probe capacity rule (exp18): hidden >= max(2048, 2 * model_hidden).

    768/1024-dim models map back to the historical 2048, so legacy
    readouts are unchanged by construction; only larger dims scale up.
    """
    cfg = _config_of(model)
    return max(2048, 2 * cfg.hidden_size)


def _config_of(model):
    if isinstance(model, str):
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model)
    return model.config


# ----------------------------------------------------------------- data
def _load_data(data, task, tokenizer, n_train, n_test, text_column=None,
               label_column=None):
    """Returns (train_texts, test_texts, train_labels, test_labels)."""
    if isinstance(data, (tuple, list)) and len(data) == 2:
        (train_texts, train_labels), (test_texts, test_labels) = data
        return (list(train_texts)[:n_train], list(test_texts)[:n_test],
                list(train_labels)[:n_train], list(test_labels)[:n_test])
    from datasets import load_dataset
    if data == "ag_news":
        ds = load_dataset("fancyzhx/ag_news")
        tc, lc = "text", "label"
    elif data == "dbpedia":
        ds = load_dataset("fancyzhx/dbpedia_14")
        tc, lc = "content", "label"
    elif data == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        tc, lc = "text", None
    else:
        ds = load_dataset(data)
        tc = text_column or "text"
        lc = label_column or ("label" if task == "classification" else None)
    tr = [t for t in ds["train"][tc] if t and t.strip()][:n_train]
    te = [t for t in ds["test" if "test" in ds else "train"][tc]
          if t and t.strip()][:n_test]
    if lc and lc in ds["train"].column_names:
        trl = ds["train"][lc][:len(tr)]
        tel = ds["test" if "test" in ds else "train"][lc][:len(te)]
    else:
        trl = tel = None
    return tr, te, trl, tel


def _encode(texts, tokenizer, max_len=64):
    enc = tokenizer(texts, truncation=True, max_length=max_len,
                    return_tensors="pt", padding="max_length")
    return enc["input_ids"], enc["attention_mask"]


# ------------------------------------------------------- split assets
def _cache_split_reprs(head, input_ids, attention_mask, device,
                       batch_size=64):
    """Cache split-point representations z = f_{1:s}(x)."""
    head = head.to(device).eval()
    zs = []
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], batch_size):
            ids = input_ids[i:i + batch_size].to(device)
            am = attention_mask[i:i + batch_size].to(device)
            zs.append(head(ids, am).cpu())
    return torch.cat(zs)


@torch.no_grad()
def _ambient(full_model, input_ids, attention_mask, device,
             batch_size=64):
    """Ambient anchor P_ctx: the pretrained LM's teacher-forced next-token
    top-1 on the corpus — the leakage an attacker has WITHOUT the
    deployment (no access to z)."""
    full_model = full_model.to(device).eval()
    correct = total = 0
    for i in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[i:i + batch_size].to(device)
        am = attention_mask[i:i + batch_size].to(device)
        logits = full_model(input_ids=ids, attention_mask=am).logits
        pred = logits[:, :-1, :].argmax(-1)
        tgt = ids[:, 1:]
        m = (tgt != -100) & am[:, 1:].bool()
        correct += (pred[m] == tgt[m]).sum().item()
        total += m.sum().item()
    return correct / max(1, total)


def _prepare(model, split_at, task, data, n_train, n_test, seed,
             device, head=None, labels=None, text_column=None,
             label_column=None, max_len=64):
    """Shared setup: model, tokenizer, encoded data, cached z.
    v0.3: AutoSplitter（gpt2 + llama 家族結構）；ambient 在微調「前」測
    （保證是 pretrained prior，不被共享權重微調污染）；分類任務建
    ClsHead 任務橋（hidden states → 類別，非 vocab logits）。"""
    from transformers import AutoTokenizer
    from .sl import ClsHead
    dev = _device(device)
    if model is None and head is not None:
        model = getattr(head, "base_model", None)
        if model is None:
            raise ValueError(
                "custom head 需暴露 .base_model（完整模型，供 wte/"
                "ambient/parity）——probe_common.Head 自帶；自寫 head 請補")
    full = load_gpt2(model, device=dev) if isinstance(model, str) \
        else model.to(dev)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    texts_tr, texts_te, y_tr, y_te = _load_data(
        data, task, tok, n_train, n_test, text_column, label_column)
    if task == "lm":
        y_tr = y_te = None

    ids_tr, am_tr = _encode(texts_tr, tok, max_len)
    ids_te, am_te = _encode(texts_te, tok, max_len)

    # ambient 先測（微調前）——它必須是「不掌握部署」的攻擊者先驗
    pctx = _ambient(full, ids_te, am_te, dev)

    if head is None:
        head, tail, full = AutoSplitter.load(full, split_at, device=dev)
    else:
        head = head.to(dev)
        tail = AutoSplitter.load(full, split_at, device=dev)[1]

    cls = None
    y_tr_t = y_te_t = None
    if task == "classification" and y_tr is not None:
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        y_te_t = torch.tensor(y_te, dtype=torch.long)
        n_cls = int(max(y_tr_t.max().item(), y_te_t.max().item())) + 1
        cls = ClsHead(head.hidden_size, n_cls).to(dev)
        head, tail, cls = _finetune_task(head, tail, cls, ids_tr, am_tr,
                                         y_tr_t, dev, seed=seed)

    z_te = _cache_split_reprs(head, ids_te, am_te, dev)
    tgt_te = ids_te.clone()
    tgt_te[am_te == 0] = -100
    return dict(full=full, tok=tok, head=head, tail=tail, cls=cls, dev=dev,
                ids_te=ids_te, am_te=am_te, z_te=z_te, tgt_te=tgt_te,
                y_te=y_te_t, n_layers=_n_layers(full),
                wte=full.get_output_embeddings().weight, pctx=pctx)


def _n_layers(full):
    cfg = full.config
    return getattr(cfg, "n_layer", None) or getattr(cfg,
                                                    "num_hidden_layers", 0)


def _finetune_task(head, tail, cls, input_ids, attention_mask, labels,
                   device, epochs=1, lr=1e-4, batch_size=32, seed=0):
    """joint head+tail+cls 微調：任務頭吃 tail 的 hidden states。"""
    torch.manual_seed(seed)
    head, tail, cls = head.to(device), tail.to(device), cls.to(device)
    head.train()
    tail.train()
    cls.train()
    seen = {}
    for m in (head, tail, cls):
        for p_ in m.parameters():
            if p_.requires_grad:
                seen.setdefault(id(p_), p_)
    opt = torch.optim.AdamW(list(seen.values()), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    n = input_ids.shape[0]
    for _ in range(max(1, epochs)):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ids = input_ids[idx].to(device)
            am = attention_mask[idx].to(device)
            y = labels[idx].to(device)
            z = head(ids, am)
            h = tail.hidden_states(z, am)
            loss = lossf(cls(h, am), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    tail.eval()
    cls.eval()
    return head, tail, cls


@torch.no_grad()
def _task_acc(tail, cls, z, attention_mask, labels, device, batch_size=128):
    tail = tail.to(device).eval()
    cls = cls.to(device).eval()
    preds = []
    for i in range(0, z.shape[0], batch_size):
        h = tail.hidden_states(z[i:i + batch_size].to(device),
                               attention_mask[i:i + batch_size].to(device))
        preds.append(cls(h, attention_mask[i:i + batch_size].to(device))
                     .argmax(-1).cpu())
    return (torch.cat(preds) == labels).float().mean().item()


# ------------------------------------------------------------ audits
def audit(model="gpt2", split_at=6, task="classification", data="ag_news",
          head=None, labels=None, text_column=None, label_column=None,
          n_train=5000, n_test=1000, seed=1006, probe_hidden=None,
          epochs=2, lr=1e-3, device=None, n_seeds=1):
    """Full audit of a split deployment (inference track).

    v0.3: AutoSplitter 支持 gpt2 家族與 llama 家族結構（Llama/Qwen2/
    Mistral/Gemma 等官方 API 截層）；雙閘門（parity+causal）不過即 raise；
    n_seeds>=2 時 CE 探針多 seed 平均（exp18b 方差協議）；分類任務附 R2
    標籤洩漏讀數。錨點表未列的 (model, corpus) 組合，verdict 會附
    「未校準」提示——簽章前必須完成 per-deployment 校準。"""
    if model is None and head is None:
        raise ValueError("audit needs either a model name or a head")
    if model is not None and not (
            str(model).startswith("gpt2")
            or str(model) in SUPPORTED_MODELS):
        raise ValueError(
            f"model {model!r} 不在錨點表支持列表 {SUPPORTED_MODELS}。"
            f"未校準模型請走 audit_custom_head（讀數可出、verdict/簽章"
            f"需先完成 per-deployment 校準）。")
    p = _prepare(model, split_at, task, data, n_train, n_test, seed,
                 device, head=head, labels=labels,
                 text_column=text_column, label_column=label_column)
    dev = p["dev"]
    dim = p["z_te"].shape[-1]
    if probe_hidden is None:
        probe_hidden = max(2048, 2 * dim)   # 由表徵維度決定（custom-head 亦通）
    wte = p["wte"]

    # 強制關卡：骨架病變不許離開管線（gpt2/llama 皆檢）
    par_d, leak_d = gates(p["head"], p["tail"], p["full"], dev)
    extra = {"parity_max_diff": par_d, "causal_leak_max": leak_d,
             "probe_seeds": n_seeds if n_seeds > 1 else 1}

    t1s, t5s = [], []
    for i in range(max(1, n_seeds)):
        ce = train_probe(p["z_te"], p["tgt_te"], wte, mode="ce",
                         hidden=probe_hidden, epochs=epochs, lr=lr,
                         device=dev, seed=seed + 4321 + i)
        t1, t5 = probe_readout(ce, p["z_te"], p["tgt_te"], wte,
                               mask=p["am_te"], device=dev)
        t1s.append(t1)
        t5s.append(t5)
        del ce
    t1 = float(np.mean(t1s))
    t5 = float(np.mean(t5s))
    if n_seeds > 1:
        extra["t1_std"] = float(np.std(t1s))
        extra["t5_std"] = float(np.std(t5s))

    mse = train_probe(p["z_te"], p["tgt_te"], wte, mode="mse",
                      hidden=probe_hidden, epochs=epochs, lr=lr,
                      device=dev, seed=seed + 5678)
    t1_mse, _ = probe_readout(mse, p["z_te"], p["tgt_te"], wte,
                              mask=p["am_te"], device=dev)
    del mse
    fl = raw_floor(p["z_te"], p["tgt_te"], wte, mask=p["am_te"], device=dev)
    pctx = p["pctx"]
    acc = _task_acc(p["tail"], p["cls"], p["z_te"], p["am_te"], p["y_te"],
                    dev) if (p["y_te"] is not None and p["cls"]
                             is not None) else None
    if task == "classification" and p["y_te"] is not None:
        extra.update(label_audit(p["z_te"], p["y_te"], p["am_te"],
                                 n_classes=int(p["y_te"].max().item()) + 1,
                                 device=dev, seed=seed))

    caveats = ["Attacker ceiling is open: re-audit quarterly and on "
               "new attack publications (freshness label required)."]
    if model is None or not _is_calibrated(str(model), data):
        caveats.append(
            "Uncalibrated (model, corpus) pair: readouts are live "
            "measurements, but verdict/signature requires per-deployment "
            "anchor calibration (splitrisk anchors).")
    return AuditResult(
        model=str(model or "custom-head"), split_at=split_at, task=task,
        seed=seed,
        t1_ce=t1, t5_ce=t5, floor=fl, pctx=pctx, t1_mse=t1_mse,
        task_acc=acc, probe_arch=f"MLP {dim}->{probe_hidden}->{probe_hidden}"
        f"->{dim} (GELU)", probe_hidden=probe_hidden,
        n_train=n_train, epochs=epochs, lr=lr, extra=extra,
        caveats=caveats)


def _is_calibrated(model_name: str, corpus: str) -> bool:
    from .calibration.anchors import CALIBRATION_TABLE
    return (model_name, str(corpus)) in CALIBRATION_TABLE


def audit_vq(model="gpt2", split_at=6, task="classification",
             data="ag_news", codebook_sizes=(256, 512, 1024),
             n_train=5000, n_test=1000, seed=1006, device=None,
             probe_hidden=None, epochs=2, lr=1e-3):
    """VQ bitrate sweep — find the quantization seam, if any.

    The seam is threat-model-dependent: the ceiling rises with attacker
    capacity and label leakage is NOT reduced. Treat results as risk
    reduction, never privacy."""
    p = _prepare(model, split_at, task, data, n_train, n_test, seed,
                 device)
    dev = p["dev"]
    if probe_hidden is None:
        probe_hidden = auto_probe_hidden(model)
    wte = p["wte"]
    cells = {}
    for k in codebook_sizes:
        vq = VQCell(p["z_te"].shape[-1], k, seed=seed)
        _, zq = vq.quantize(p["z_te"])
        zq = zq.detach()
        tgt_q = p["tgt_te"].clone()
        net = train_probe(zq, tgt_q, wte, mode="ce", hidden=probe_hidden,
                          epochs=epochs, lr=lr, device=dev, seed=seed)
        t1, t5 = probe_readout(net, zq, tgt_q, wte, mask=p["am_te"],
                               device=dev)
        cells[k] = {"t1_ce": t1, "t5_ce": t5}
    primary_k = codebook_sizes[len(codebook_sizes) // 2]
    c = cells[primary_k]
    return AuditResult(
        model=str(model), split_at=split_at, task=task, seed=seed,
        t1_ce=c["t1_ce"], t5_ce=c["t5_ce"], mode="vq",
        extra={f"K={k} bits={int(k).bit_length() - 1}":
               f"t1={v['t1_ce']:.3f} t5={v['t5_ce']:.3f}"
               for k, v in cells.items()},
        probe_arch=f"MLP (VQ-quantized input)", probe_hidden=probe_hidden,
        n_train=n_train, epochs=epochs, lr=lr,
        caveats=["Label leakage is NOT reduced by quantization (R2 "
                 "0.859 at 9 bits vs 0.871 unquantized).",
                 "Attacker ceiling at 9 bits is open (0.36->0.42->0.47)."])


def audit_sharded(model="gpt2", split_at=6, data="ag_news", n_shards=3,
                  vq_bits=None, n_train=5000, n_test=1000, seed=1006,
                  device=None, probe_hidden=None, epochs=2, lr=1e-3):
    """Sharding audit: single-shard view + collusive ceiling.

    Shard count buys jurisdiction (and a lower single-shard view via
    attacker-capacity dilution); the bitrate alone sets the collusive
    ceiling max_C R(C)."""
    p = _prepare(model, split_at, "classification", data, n_train,
                 n_test, seed, device)
    dev = p["dev"]
    if probe_hidden is None:
        probe_hidden = auto_probe_hidden(model)
    wte = p["wte"]
    z, tgt, am = p["z_te"], p["tgt_te"], p["am_te"]
    n, t, d = z.shape
    shard_len = t // n_shards
    single = {}
    for s in range(n_shards):
        sl = slice(s * shard_len, (s + 1) * shard_len)
        zs, ts, as_ = z[:, sl], tgt[:, sl], am[:, sl]
        if vq_bits:
            vq = VQCell(d, 2 ** vq_bits, seed=seed)
            _, zs = vq.quantize(zs)
            zs = zs.detach()
        net = train_probe(zs, ts, wte, mode="ce", hidden=probe_hidden,
                          epochs=epochs, lr=lr, device=dev, seed=seed)
        t1, t5 = probe_readout(net, zs, ts, wte, mask=as_, device=dev)
        single[f"shard_{s}"] = {"t1_ce": t1, "t5_ce": t5}
    # collusive ceiling: context-aware reassembler over ALL positions
    net, coll = train_reassembler(z, tgt, wte.shape[0], wte=wte,
                                  n_layers=3, epochs=epochs, device=dev,
                                  seed=seed)
    return AuditResult(
        model=str(model), split_at=split_at, task="classification",
        seed=seed, t1_ce=max(v["t1_ce"] for v in single.values()),
        t5_ce=None, mode="sharded",
        extra={"single_shard": single, "collusive_ceiling": coll,
               "n_shards": n_shards, "vq_bits": vq_bits},
        probe_arch=f"per-shard MLP + 3-layer reassembler",
        probe_hidden=probe_hidden, n_train=n_train, epochs=epochs, lr=lr,
        caveats=["Single-shard decrease is attacker-capacity dilution, "
                 "not privacy; report the collusive ceiling.",
                 "Bits, not shard count, set the collusive ceiling."])


def audit_training(delivered_model, reference_data, task="classification",
                   n_test=1000, seed=1006, device=None, probe_hidden=None,
                   epochs=2, lr=1e-3, split_at=6, n_train=None):
    """Training-track audit: CE probe on a delivered model version.

    Three signatures (exp13/13b): FSHA double-loss (task collapses),
    SIA stealth (task normal, recovery ~0.97 — only the CE probe
    detects it), R-M == S-M (leakage set by hijack intensity alpha).
    v0.3: split_at / n_train 可指定（對齊被審計部署的真實切分與預算）。
    """
    dev = _device(device)
    p = _prepare(delivered_model, split_at=split_at, task=task,
                 data="ag_news", n_train=n_train or n_test, n_test=n_test,
                 seed=seed, device=dev)
    if probe_hidden is None:
        probe_hidden = max(2048, 2 * p["z_te"].shape[-1])
    net = train_probe(p["z_te"], p["tgt_te"], p["wte"], mode="ce",
                      hidden=probe_hidden, epochs=epochs, lr=lr,
                      device=dev, seed=seed)
    t1, t5 = probe_readout(net, p["z_te"], p["tgt_te"], p["wte"],
                           mask=p["am_te"], device=dev)
    acc = _task_acc(p["tail"], p["z_te"], p["y_te"], dev) \
        if p["y_te"] is not None else None
    stealth = acc is not None and acc > 0.8 and t1 > 0.85
    return AuditResult(
        model=str(delivered_model), split_at=split_at, task=task, seed=seed,
        t1_ce=t1, t5_ce=t5, mode="training", task_acc=acc,
        probe_arch="MLP (CE)", probe_hidden=probe_hidden,
        n_train=n_train or n_test, epochs=epochs, lr=lr,
        caveats=["Task metrics are blind to SIA-style stealth hijacks; "
                 "only the CE probe sees it (R6 axis).",
                 "SIA DETECTED: task normal with full recovery — refuse "
                 "this model version." if stealth else
                 "No stealth signature detected in this readout."])


def audit_sl(model="gpt2", split_at=6, data="ag_news", baseline="honest",
             alpha=5.0, sl_epochs=1, sl_lr_head=1e-4, sl_lr_server=1e-4,
             n_train=5000, n_test=1000, seed=1006, probe_hidden=None,
             att_epochs=2, device=None, n_seeds=1):
    """SL 訓練環審計（exp13 產品化）：跑 honest/sia/fsha 訓練環後對傳輸
    表徵出 CE 探針讀數。SIA 隱蔽 signature 驗收：任務正常而 t1 高（R6 軸）；
    FSHA：任務崩塌（易偵測）。對照三 baseline 用同參數各跑一次。"""
    assert baseline in ("honest", "sia", "fsha"), baseline
    from .sl import train_split, eval_split
    from torch.utils.data import DataLoader, TensorDataset
    dev = _device(device)
    p = _prepare(model, split_at, "classification", data, n_train, n_test,
                 seed, dev)
    dim = p["z_te"].shape[-1]
    if probe_hidden is None:
        probe_hidden = max(2048, 2 * dim)
    wte = p["wte"]
    if p["y_te"] is None:
        raise ValueError("audit_sl 需要 classification 任務（帶標籤）")

    # 訓練側編碼（_prepare 只快取 test 側）
    texts_tr, _, y_tr, _ = _load_data(data, "classification", p["tok"],
                                      n_train, 0)
    ids_tr, am_tr = _encode(texts_tr, p["tok"], 64)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    import copy
    head_sl = copy.deepcopy(p["head"])
    tail_sl = copy.deepcopy(p["tail"])
    from .sl import ClsHead
    cls_sl = ClsHead(dim, int(p["y_te"].max().item()) + 1).to(dev)
    loader = DataLoader(TensorDataset(ids_tr, am_tr.bool(), y_tr_t),
                        batch_size=32, shuffle=True)
    stats = train_split(head_sl, tail_sl, cls_sl, loader, wte,
                        baseline=baseline, alpha=alpha, epochs=sl_epochs,
                        lr_head=sl_lr_head, lr_server=sl_lr_server,
                        log_every=100, device=dev, seed=seed)
    head_sl.eval()
    tail_sl.eval()
    cls_sl.eval()
    z_te = _cache_split_reprs(head_sl, p["ids_te"], p["am_te"], dev)
    tgt_te = p["tgt_te"]
    t1s, t5s = [], []
    for i in range(max(1, n_seeds)):
        net = train_probe(z_te, tgt_te, wte, mode="ce",
                          hidden=probe_hidden, epochs=att_epochs,
                          device=dev, seed=seed + 4321 + i)
        t1, t5 = probe_readout(net, z_te, tgt_te, wte, mask=p["am_te"],
                               device=dev)
        t1s.append(t1)
        t5s.append(t5)
    acc = eval_split(head_sl, tail_sl, cls_sl, p["ids_te"], p["am_te"],
                     p["y_te"], dev)
    extra = {"baseline": baseline, "alpha": alpha,
             "sl_train_loss": stats["train_loss"],
             "sl_hijack_loss": stats["hijack_loss"], "sl_steps":
             stats["steps"]}
    if n_seeds > 1:
        extra["t1_std"] = float(np.std(t1s))
    t1, t5 = float(np.mean(t1s)), float(np.mean(t5s))
    r6 = (acc > 0.8 and t1 > 0.85)
    return AuditResult(
        model=str(model), split_at=split_at, task="classification",
        seed=seed, t1_ce=t1, t5_ce=t5,
        floor=raw_floor(z_te, tgt_te, wte, mask=p["am_te"], device=dev),
        pctx=p["pctx"],
        task_acc=acc, mode="sl", extra=extra,
        probe_arch="MLP (CE)", probe_hidden=probe_hidden,
        n_train=n_train, epochs=att_epochs, lr=1e-3,
        caveats=(["SL 訓練環讀數——SIA 隱蔽 signature：任務正常而 t1 高"
                  "（R6 軸），任務指標驗收會通過，唯 CE 探針可抓。"] if r6
                 else ["SL 訓練環讀數——未見任務正常+滿洩漏並存 signature；"
                       "對照三 baseline 請同參數各跑一次。"]))


def _measure_anchors(model="gpt2", corpus="ag_news", device=None,
                     n_test=1000, seed=1006, split_at=6):
    """Measure (full_info, ambient, floor) for the anchor check."""
    p = _prepare(model, split_at, "classification", corpus, n_test,
                 n_test, seed, device)
    dev = p["dev"]
    net = train_probe(p["z_te"], p["tgt_te"], p["wte"], mode="ce",
                      hidden=auto_probe_hidden(model), epochs=2,
                      device=dev, seed=seed)
    t1, _ = probe_readout(net, p["z_te"], p["tgt_te"], p["wte"],
                          mask=p["am_te"], device=dev)
    fl = raw_floor(p["z_te"], p["tgt_te"], p["wte"], mask=p["am_te"],
                   device=dev)
    pctx = _ambient(p["full"], p["ids_te"], p["am_te"], dev)
    return t1, pctx, fl
