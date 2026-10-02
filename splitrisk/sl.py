"""splitrisk/sl.py — 拆分訓練環 (train_split) + 劫持注入 + SplitFed 模擬

產品化 exp13 協議，讓客戶能審計「SL 訓練過程本身」而不只是交付模型：

  honest  正常 SL：client head 前向 → smash z → server tail+cls → CE loss
          → server 反向至 z.grad → z.backward(gradient=z.grad) → 雙側步進
  sia     可持續劫持（隱蔽，α=5 慣例）：server loss = 任務CE + α·劫持CE
          （劫持解碼器 z→token 識別）——任務正常、表徵滿洩漏 → 唯 CE 探針可抓
  fsha    雙輪劫持：server 只訓劫持解碼器（任務橋移除）→ 任務崩塌、易偵測

PyTorch 2.x 顯式 detach 分離（exp13 retain_graph 教訓）。
SplitFed：頭部 FedAvg 模擬（K 客戶端分片，輪末平均 head；tail 共享）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINES = ("honest", "sia", "fsha")


class ClsHead(nn.Module):
    """主任務橋：最後有效位置 pooling → Linear（exp13 ClsHead 慣例）。
    接口統一為 cls(h, mask)——tail 輸出直接進來。"""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, h, mask):
        idx = mask.long().sum(1).clamp_min(1) - 1
        rows = torch.arange(h.size(0), device=h.device)
        return self.fc(h[rows, idx].float())


def _token_ce(z, ids, mask, wte):
    """劫持解碼器：z 直接投到詞表（攻擊者無需結構知識的 token CE）。"""
    D = z.shape[-1]
    zf, tgt, m = z.reshape(-1, D), ids.reshape(-1), mask.reshape(-1)
    idx = torch.nonzero(m).squeeze(1)
    if idx.numel() == 0:
        return 0.0 * z.sum()
    return F.cross_entropy(zf[idx] @ wte.t().to(z.dtype), tgt[idx])


def train_split(head, tail, cls, loader, wte, n_classes=None,
                baseline="honest", alpha=5.0, epochs=1, lr_head=2e-5,
                lr_server=1e-4, log_every=0, device="cpu", seed=0,
                microbatch_size=None):
    """跑一個 SL 訓練環。返回 dict(train_loss, hijack_loss, task_acc_proxy,
    steps, baseline)。head/tail/cls 就地訓練（調用方先 deepcopy 要保留的）。"""
    assert baseline in BASELINES, baseline
    torch.manual_seed(seed)
    head, tail, cls = head.to(device), tail.to(device), cls.to(device)
    head.train()
    tail.train()
    cls.train()
    # v0.3.2: 撤銷 wte 凍結——battery demo.py 以 2e-5 全參（含 wte）
    # finetune 後 probe 仍 0.94-0.97，此前「漂移」結論係探針欠訓練所致。
    # 保留 joint optimizer + dedup（=demo 的 dict.fromkeys 模式）。
    tail_params = {id(p): p for p in tail.parameters() if p.requires_grad}
    cls_params = {id(p): p for p in cls.parameters() if p.requires_grad}
    groups = []
    head_only = [p for p in head.parameters()
                 if p.requires_grad and id(p) not in tail_params
                 and id(p) not in cls_params]
    if head_only:
        groups.append({"params": head_only, "lr": lr_head})
    if tail_params:
        groups.append({"params": list(tail_params.values()),
                       "lr": lr_server})
    if cls_params:
        groups.append({"params": list(cls_params.values()), "lr": lr_server})
    joint = torch.optim.AdamW(groups) if groups else None
    opt_hijack = joint   # head 側更新器（honest=server 梯度回傳）
    if microbatch_size is not None:
        if microbatch_size < 1 or baseline != "honest":
            raise ValueError("microbatch_size requires honest baseline and a positive size")
        if any(id(p) in tail_params or id(p) in cls_params
               for p in head.parameters()):
            raise ValueError("microbatch accumulation requires disjoint client/server parameters")
    wte = wte.to(device)
    stats = {"train_loss": 0.0, "hijack_loss": 0.0, "steps": 0}
    cls.eval()
    for ep in range(max(1, epochs)):
        for ids, mask, y in loader:
            ids, mask = ids.to(device), mask.to(device)
            y = y.to(device)
            if microbatch_size is not None:
                # Preserve the original two optimizer steps per whole shard:
                # server first, then client; each loss is weighted by shard size.
                if joint is not None:
                    joint.zero_grad()
                client_grads = {}
                task_sum = hij_sum = 0.0
                token_count = int(mask.count_nonzero())
                for start in range(0, len(ids), microbatch_size):
                    bi = ids[start:start + microbatch_size]
                    bm = mask[start:start + microbatch_size]
                    by = y[start:start + microbatch_size]
                    z = head(bi, bm)
                    z_send = z.detach().requires_grad_(True)
                    # Honest hijack CE is diagnostic only. Bound vocabulary
                    # logits to one sequence, with the same token mean.
                    with torch.no_grad():
                        for row in range(len(bi)):
                            count = int(bm[row].count_nonzero())
                            if count:
                                hij_sum += float(_token_ce(
                                    z_send[row:row + 1], bi[row:row + 1],
                                    bm[row:row + 1], wte)) * count
                    h = tail.hidden_states(z_send, bm)
                    loss = F.cross_entropy(cls(h, bm), by)
                    weight = len(bi) / len(ids)
                    (loss * weight).backward()
                    task_sum += float(loss.detach()) * weight
                    if z_send.grad is not None and joint is not None:
                        z.backward(gradient=z_send.grad)
                        for param in head_only:
                            if param.grad is not None:
                                if param not in client_grads:
                                    client_grads[param] = param.grad
                                else:
                                    client_grads[param].add_(param.grad)
                                param.grad = None
                    del z, z_send, h, loss
                if joint is not None:
                    joint.step()
                    joint.zero_grad()
                    for param, grad in client_grads.items():
                        param.grad = grad
                    if client_grads:
                        joint.step()
                stats["train_loss"] += task_sum
                stats["hijack_loss"] += hij_sum / max(1, token_count)
                stats["steps"] += 1
                if log_every and stats["steps"] % log_every == 0:
                    print(f"    [sl:{baseline}] ep{ep+1} step{stats['steps']} "
                          f"loss={task_sum:.4f} hij={hij_sum / max(1, token_count):.4f}")
                continue
            # --- client: head 前向 ---
            z = head(ids, mask)
            # --- smash: 傳輸張量（detach + 開梯度）---
            z_send = z.detach().requires_grad_(True)
            # --- server ---
            hij = _token_ce(z_send, ids, mask, wte)
            h = tail.hidden_states(z_send, mask)
            logits = cls(h, mask)
            if baseline == "fsha":
                server_loss = hij                                  # 任務橋移除
            else:
                server_loss = F.cross_entropy(logits, y)
                if baseline == "sia":
                    server_loss = server_loss + alpha * hij
            if joint is not None:
                joint.zero_grad()
            server_loss.backward()
            if joint is not None:
                joint.step()
            # --- 梯度回傳 client ---
            if z_send.grad is not None and opt_hijack is not None:
                opt_hijack.zero_grad()
                z.backward(gradient=z_send.grad)
                opt_hijack.step()
            stats["train_loss"] += float(server_loss.detach())
            stats["hijack_loss"] += float(hij.detach())
            stats["steps"] += 1
            if log_every and stats["steps"] % log_every == 0:
                print(f"    [sl:{baseline}] ep{ep+1} step{stats['steps']} "
                      f"loss={float(server_loss):.4f} hij={float(hij):.4f}")
    for k in ("train_loss", "hijack_loss"):
        stats[k] = stats[k] / max(1, stats["steps"])
    stats["baseline"] = baseline
    head.eval()
    tail.eval()
    cls.eval()
    return stats


@torch.no_grad()
def eval_split(head, tail, cls, ids, mask, y, device="cpu", batch=64):
    """任務準確率（cls head 走完整 SL 路徑，hidden_states 介面）。"""
    head, tail, cls = head.to(device), tail.to(device), cls.to(device)
    head.eval()
    tail.eval()
    cls.eval()
    preds = []
    for i in range(0, len(ids), batch):
        z = head(ids[i:i + batch].to(device), mask[i:i + batch].to(device))
        h = tail.hidden_states(z, mask[i:i + batch].to(device))
        preds.append(cls(h, mask[i:i + batch].to(device)).argmax(-1).cpu())
    return (torch.cat(preds) == y).float().mean().item()


def splitfed(head_fn, tail, cls, shards, wte, rounds=2, baseline="honest",
             alpha=5.0, lr_head=1e-4, lr_server=1e-4, device="cpu",
             seed=0):
    """SplitFed 頭部 FedAvg 模擬。shards: list[(ids, mask, y)]。
    每輪：各客戶端在自己分片上跑一個 SL step（本地 head 副本）→ FedAvg 頭部
    → tail 共享步進。返回 (global_head, tail, cls, history)。"""
    import copy
    torch.manual_seed(seed)
    global_head = head_fn().to(device)
    history = []
    for r in range(rounds):
        local_states = []
        for ids, mask, y in shards:
            local = copy.deepcopy(global_head)
            loader = [(ids, mask, y)]
            train_split(local, tail, cls, loader, wte, baseline=baseline,
                        alpha=alpha, epochs=1, lr_head=lr_head,
                        lr_server=lr_server, device=device,
                        seed=seed + r * 100 + len(history))
            local_states.append(copy.deepcopy(local.state_dict()))
            history.append({"round": r, "client": len(history),
                            "shard_size": len(ids)})
        # FedAvg
        avg = {k: torch.stack([s[k].float() for s in local_states]).mean(0)
               for k in local_states[0]}
        ref_dtype = {k: v.dtype for k, v in global_head.state_dict().items()}
        global_head.load_state_dict(
            {k: v.to(ref_dtype[k]) for k, v in avg.items()
             if k in ref_dtype})
    return global_head, tail, cls, history
