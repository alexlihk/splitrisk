"""splitrisk/certify.py — L2 認證引擎（schema v1.1）

把 AuditResult 轉為 SA-RI 認證記錄（JSON）+ 認證報告（Markdown）。
規則（與白皮書 §3.5/§6 一致）：
  1. 只認證有確定 verdict 的結果——INCONCLUSIVE 一律拒絕（拒絕猜測）
  2. 認證編號內容尋址（crc32），同輸入同編號，可重現
  3. audit_record 全字段 + 誠實條款十條強制附錄 + 保鮮期
  4. 預覽版簽章欄位標 UNSIGN——生產簽章屬 SplitAudit 商業引擎，不在 L1
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import zlib

from .report import AuditResult

HONESTY_CLAUSES = [
    "攻擊者預算隨每個讀數輸出；單份下降不等於隱私（容量稀釋）——合謀上限由位元率決定。",
    "天花板標註開放上界：0.36→0.42→0.47 未收斂。任何 AMBER 認證必須附保鮮期與容量階梯版本。",
    "entity 已測（合成，機制級，有利）：PHI 被量化差異化壓制（ratio 0.625）；醫院分佈仍屬部署工作。",
    "語料×規模×家族已在實測點確認；環境基線隨規模、域、家族變化（0.317→0.415），per-deployment 校準強制。",
    "R2 位元率鎖不死：9-bit 量化鎖不死標籤——類別即敏感場景必須直接紅燈。",
    "紅燈的解法是 TEE / 本地：本工具測量風險，不消除風險。",
    "AMBER 適用邊界四條件（label 不敏感/輸出無敏感/codebook 不指紋/量大到本地跑不動）；縫寬有架構依賴，不可跨架構外推。",
    "檢測粒度：部署級認證非 query 級檢測；一次審計約 2 GPU 小時；季測+事件觸發；年成本約 6.5 GPU 小時。",
    "Codebook side-channel：碼本出域不增 token 恢復風險，但統計特徵須評估。",
    "閉源 LLM 不適用（無中間表徵可測）；探針訓練有 ~7pp run-to-run 方差——多 seed 平均為協議要求。",
]

LEGAL_NOTICE = (
    "本報告依 SA-RI schema v1.1 出具，可作為 GDPR TIA Step 4 技術補充措施證據及 "
    "DPIA Art. 35(7)(d) 技術措施記錄之技術層附件；本報告不構成法律合規意見，"
    "不免除資料輸出方對目的地法律環境與政府存取風險的個案評估義務。"
)

FRESHNESS = ("保鮮期：本認證自時間戳起有效期一季；"
             "新攻擊族公開、容量階梯更新或重大供應鏈事件即觸發重測。")


class CertifyRefused(Exception):
    """INCONCLUSIVE / 無 verdict 的結果不可認證。"""


def _cert_number(payload: dict) -> str:
    year = _dt.datetime.now(_dt.timezone.utc).strftime("%Y")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return f"SA-RI-{year}-{digest[:6].upper()}"


def _fingerprint(audit: dict) -> str:
    core = {k: audit.get(k) for k in
            ("model", "split_at", "task", "seed", "t1_ce", "t5_ce",
             "floor", "pctx", "probe_hidden", "n_train", "epochs")}
    return "sha256:" + hashlib.sha256(
        json.dumps(core, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:32]


def certify(audit: AuditResult, org: str = "SAMPLE",
            engagement: str = "DUMMY PIPELINE (unattended sample)",
            tier_note: str = "") -> dict:
    """AuditResult → 認證記錄 dict。INCONCLUSIVE 拒絕。"""
    verdict = audit.verdict
    if verdict.startswith("INCONCLUSIVE"):
        raise CertifyRefused(
            f"refuse to certify: {verdict}. 拒絕猜測是產品行為——"
            f"先修復探針容量/多 seed/校準，再申請認證。")

    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    audit_dict = audit.to_dict()
    record = {
        "schema": "SA-RI v1.1",
        "cert_number": None,          # 填於 payload 之後
        "issued_utc": now.isoformat(),
        "org": org,
        "engagement": engagement,
        "tier": verdict,
        "tier_note": tier_note,
        "measurements": {
            "t1_ce": audit.t1_ce, "t5_ce": audit.t5_ce,
            "floor": audit.floor, "pctx": audit.pctx,
            "t1_mse": audit.t1_mse, "task_acc": audit.task_acc,
            "excess_leakage_pp": audit.excess_leakage,
            "mode": audit.mode, "extra": audit.extra,
        },
        "budget": {
            "probe_arch": audit.probe_arch,
            "probe_hidden": audit.probe_hidden,
            "n_train": audit.n_train, "epochs": audit.epochs,
            "lr": audit.lr, "seed": audit.seed,
        },
        "audit_record": {
            "model": audit.model, "split_at": audit.split_at,
            "fingerprint": _fingerprint(audit_dict),
            "anchor_version": "v3.3 (19-exp table)",
            "engine_version": audit_dict.get("mode") and "splitrisk 0.2.1",
            "timestamp": now.isoformat(),
        },
        "legal_notice": LEGAL_NOTICE,
        "freshness": FRESHNESS,
        "caveats": list(HONESTY_CLAUSES),
        "signature": "UNSIGN (L2 engine preview — production signing is "
                     "SplitAudit commercial layer)",
    }
    record["cert_number"] = _cert_number(record)
    return record


def render_markdown(record: dict) -> str:
    m, b, ar = record["measurements"], record["budget"], record["audit_record"]
    fmt = lambda v: f"{v:.1%}" if isinstance(v, float) and 0 <= v <= 1 else str(v)
    lines = [
        "=" * 64,
        "  SA-RI 認證報告（Certification Report）",
        f"  認證編號: {record['cert_number']}   Tier: {record['tier']}",
        f"  機構: {record['org']}   委託: {record['engagement']}",
        f"  簽發(UTC): {record['issued_utc']}   Schema: {record['schema']}",
        "=" * 64,
        "",
        "  一、測量",
        f"    Token 恢復 t1 (CE):   {fmt(m['t1_ce'])}",
        f"    Token 恢復 t5 (CE):   {fmt(m['t5_ce'])}",
        f"    線性地板 (Tier 0):    {fmt(m['floor'])}",
        f"    環境基線 (P_ctx):     {fmt(m['pctx'])}",
        f"    超額洩漏:             {m['excess_leakage_pp'] and format(m['excess_leakage_pp'], '+.1f') + 'pp'}",
        f"    任務準確率:           {fmt(m['task_acc']) if m['task_acc'] else 'n/a'}",
        "",
        "  二、攻擊者預算（不可分離欄位）",
        f"    探針: {b['probe_arch']} (hidden={b['probe_hidden']})",
        f"    訓練: {b['n_train']} 樣本 × {b['epochs']} epochs, lr {b['lr']}, seed {b['seed']}",
        "",
        "  三、審計鏈 (audit_record)",
        f"    模型/切分: {ar['model']} @ split {ar['split_at']}",
        f"    特徵哈希: {ar['fingerprint']}",
        f"    錨點版本: {ar['anchor_version']}   引擎: {ar['engine_version']}",
        "",
        "  四、法律聲明",
        f"    {record['legal_notice']}",
        "",
        f"  五、{record['freshness']}",
        f"  簽章: {record['signature']}",
        "",
        "  六、誠實條款（十條，不可關閉，寫進 license）",
    ]
    lines += [f"    {i}. {c}" for i, c in enumerate(record["caveats"], 1)]
    lines += ["=" * 64, ""]
    return "\n".join(lines)


def certify_to_files(audit: AuditResult, out_prefix: str, org: str = "SAMPLE",
                     engagement: str = "DUMMY PIPELINE") -> tuple[str, str]:
    record = certify(audit, org=org, engagement=engagement)
    jpath = out_prefix + ".cert.json"
    mpath = out_prefix + ".cert.md"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(render_markdown(record))
    return jpath, mpath
