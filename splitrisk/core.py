"""Core audit pipelines over the probe battery.

One-call entry points (see __init__.py): audit / audit_vq /
audit_sharded / audit_training. All heavy imports are lazy so the
package imports cleanly everywhere; heavy work needs torch +
transformers at call time.
"""
import numpy as np
import torch

from .report import AuditResult
from .probes.probe_common import load_gpt2, Head, Tail
from .probes.train_cf_attacker import train_probe, probe_readout
from .probes.raw_floor import raw_floor
from .probes.reassembler import train_reassembler
from .vq.vq_cell import VQCell

SUPPORTED_MODELS = ("gpt2", "gpt2-medium", "gpt2-large")
SEED_OFFSETS = {"classification": 1000, "lm": 2000}


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


def _finetune_task(head, tail, input_ids, attention_mask, labels, device,
                   epochs=1, lr=1e-4, batch_size=32, seed=0):
    """Joint head+tail fine-tuning on the main task (server side)."""
    torch.manual_seed(seed)
    head, tail = head.to(device), tail.to(device)
    # head and tail share wte/wpe with the source model — dedupe by
    # parameter id or AdamW double-counts them
    seen = {}
    for m in (head, tail):
        for p in m.parameters():
            if p.requires_grad:
                seen.setdefault(id(p), p)
    params = list(seen.values())
    opt = torch.optim.AdamW(params, lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    n = input_ids.shape[0]
    for _ in range(max(1, epochs)):
        for i in range(0, n, batch_size):
            ids = input_ids[i:i + batch_size].to(device)
            am = attention_mask[i:i + batch_size].to(device)
            y = labels[i:i + batch_size].to(device)
            z = head(ids, am)
            logits = tail(z, am)
            loss = lossf(logits[:, 0, :], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head, tail


@torch.no_grad()
def _task_acc(tail, z, labels, device, batch_size=128):
    tail = tail.to(device).eval()
    correct = 0
    for i in range(0, z.shape[0], batch_size):
        logits = tail(z[i:i + batch_size].to(device))
        pred = logits[:, 0, :].argmax(-1)
        correct += (pred.cpu() == labels[i:i + batch_size]).sum().item()
    return correct / max(1, z.shape[0])


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
    """Shared setup: model, tokenizer, encoded data, cached z."""
    from transformers import AutoTokenizer
    dev = _device(device)
    full = load_gpt2(model, device=dev) if isinstance(model, str) \
        else model.to(dev)
    tok = AutoTokenizer.from_pretrained(
        "gpt2" if isinstance(model, str) else "gpt2")
    tok.pad_token = tok.eos_token

    texts_tr, texts_te, y_tr, y_te = _load_data(
        data, task, tok, n_train, n_test, text_column, label_column)
    if task == "lm":
        y_tr = y_te = None

    ids_tr, am_tr = _encode(texts_tr, tok, max_len)
    ids_te, am_te = _encode(texts_te, tok, max_len)

    head = head if head is not None else Head(full, split_at)
    tail = Tail(full, split_at)
    tail.freeze_position_embedding()

    if task == "classification" and y_tr is not None:
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        y_te_t = torch.tensor(y_te, dtype=torch.long)
        head, tail = _finetune_task(head, tail, ids_tr, am_tr, y_tr_t,
                                    dev, seed=seed)
    else:
        y_tr_t = y_te_t = None

    z_te = _cache_split_reprs(head, ids_te, am_te, dev)
    # CE/MSE probes train on the ground-truth token identity of the
    # transmitted positions (padding positions excluded with -100)
    tgt_te = ids_te.clone()
    tgt_te[am_te == 0] = -100
    return dict(full=full, tok=tok, head=head, tail=tail, dev=dev,
                ids_te=ids_te, am_te=am_te, z_te=z_te, tgt_te=tgt_te,
                y_te=y_te_t, n_layers=full.config.n_layer)


# ------------------------------------------------------------ audits
def audit(model="gpt2", split_at=6, task="classification", data="ag_news",
          head=None, labels=None, text_column=None, label_column=None,
          n_train=5000, n_test=1000, seed=1006, probe_hidden=None,
          epochs=2, lr=1e-3, device=None):
    """Full audit of a split deployment (inference track)."""
    if model is None and head is None:
        raise ValueError("audit needs either a model name or a head")
    if model is not None and not str(model).startswith(("gpt2",)):
        # limited support list; bring-your-own head otherwise
        raise ValueError(
            f"model {model!r} is not in the calibrated support list "
            f"{SUPPORTED_MODELS}; pass a custom `head` instead "
            f"(audit_custom_head).")
    p = _prepare(model, split_at, task, data, n_train, n_test, seed,
                 device, head=head, labels=labels,
                 text_column=text_column, label_column=label_column)
    dev, dim = p["dev"], p["z_te"].shape[-1]
    if probe_hidden is None:
        probe_hidden = auto_probe_hidden(model if isinstance(model, str)
                                         else model)
    wte = p["full"].transformer.wte.weight

    ce = train_probe(p["z_te"], p["tgt_te"], wte, mode="ce",
                     hidden=probe_hidden, epochs=epochs, lr=lr,
                     device=dev, seed=seed)
    t1, t5 = probe_readout(ce, p["z_te"], p["tgt_te"], wte, mask=p["am_te"],
                           device=dev)
    mse = train_probe(p["z_te"], p["tgt_te"], wte, mode="mse",
                      hidden=probe_hidden, epochs=epochs, lr=lr,
                      device=dev, seed=seed)
    t1_mse, _ = probe_readout(mse, p["z_te"], p["tgt_te"], wte,
                              mask=p["am_te"], device=dev)
    fl = raw_floor(p["z_te"], p["tgt_te"], wte, mask=p["am_te"], device=dev)
    pctx = _ambient(p["full"], p["ids_te"], p["am_te"], dev)
    acc = _task_acc(p["tail"], p["z_te"], p["y_te"], dev) \
        if p["y_te"] is not None else None

    return AuditResult(
        model=str(model), split_at=split_at, task=task, seed=seed,
        t1_ce=t1, t5_ce=t5, floor=fl, pctx=pctx, t1_mse=t1_mse,
        task_acc=acc, probe_arch=f"MLP {dim}->{probe_hidden}->{probe_hidden}"
        f"->{dim} (GELU)", probe_hidden=probe_hidden,
        n_train=n_train, epochs=epochs, lr=lr,
        caveats=["Attacker ceiling is open: re-audit quarterly and on "
                 "new attack publications (freshness label required)."])


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
    wte = p["full"].transformer.wte.weight
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
    wte = p["full"].transformer.wte.weight
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
                   epochs=2, lr=1e-3):
    """Training-track audit: CE probe on a delivered model version.

    Three signatures (exp13/13b): FSHA double-loss (task collapses),
    SIA stealth (task normal, recovery ~0.97 — only the CE probe
    detects it), R-M == S-M (leakage set by hijack intensity alpha).
    """
    dev = _device(device)
    p = _prepare(delivered_model, split_at=6, task=task, data="ag_news",
                 n_train=n_test, n_test=n_test, seed=seed, device=dev)
    if probe_hidden is None:
        probe_hidden = auto_probe_hidden(
            delivered_model if isinstance(delivered_model, str)
            else "gpt2")
    wte = p["full"].transformer.wte.weight
    net = train_probe(p["z_te"], p["tgt_te"], wte, mode="ce",
                      hidden=probe_hidden, epochs=epochs, lr=lr,
                      device=dev, seed=seed)
    t1, t5 = probe_readout(net, p["z_te"], p["tgt_te"], wte,
                           mask=p["am_te"], device=dev)
    acc = _task_acc(p["tail"], p["z_te"], p["y_te"], dev) \
        if p["y_te"] is not None else None
    stealth = acc is not None and acc > 0.8 and t1 > 0.85
    return AuditResult(
        model=str(delivered_model), split_at=6, task=task, seed=seed,
        t1_ce=t1, t5_ce=t5, mode="training", task_acc=acc,
        probe_arch="MLP (CE)", probe_hidden=probe_hidden,
        n_train=n_test, epochs=epochs, lr=lr,
        caveats=["Task metrics are blind to SIA-style stealth hijacks; "
                 "only the CE probe sees it (R6 axis).",
                 "SIA DETECTED: task normal with full recovery — refuse "
                 "this model version." if stealth else
                 "No stealth signature detected in this readout."])


def _measure_anchors(model="gpt2", corpus="ag_news", device=None,
                     n_test=1000, seed=1006, split_at=6):
    """Measure (full_info, ambient, floor) for the anchor check."""
    p = _prepare(model, split_at, "classification", corpus, n_test,
                 n_test, seed, device)
    dev = p["dev"]
    wte = p["full"].transformer.wte.weight
    net = train_probe(p["z_te"], p["tgt_te"], wte, mode="ce",
                      hidden=auto_probe_hidden(model), epochs=2,
                      device=dev, seed=seed)
    t1, _ = probe_readout(net, p["z_te"], p["tgt_te"], wte,
                          mask=p["am_te"], device=dev)
    fl = raw_floor(p["z_te"], p["tgt_te"], wte, mask=p["am_te"],
                   device=dev)
    pctx = _ambient(p["full"], p["ids_te"], p["am_te"], dev)
    return t1, pctx, fl
