# FAQ

## Closed-source LLMs: the four quadrants

The boundary question is never "open or closed weights" — it is always:
**does an intermediate representation cross a trust boundary?**

| Quadrant | Scenario | Service |
|---|---|---|
| Q1 | Self-hosted open-weight model, split across trust domains | ✅ Full (signable) |
| Q2 | Training outsourcing; delivered model is open-weight | ✅ Full (R6, signable after calibration) |
| Q3 | Closed vendor exposes an **embedding endpoint** | ⚠️ Limited (inversion measurement, v2) |
| Q4 | Closed chat/completion API only (plaintext in/out) | ❌ Not measurable — by anyone |

**Q1.** `splitrisk anchors --model llama-3.2-1b` then
`splitrisk audit --model llama-3.2-1b --split 8 --n-seeds 3`.

**Q2 (delivered-model acceptance).**
`splitrisk training --model <delivered> --split 6` — R6: task metrics are
blind to SIA-style stealth hijacks (exp13: task 0.873 normal, recovery
0.9695); only the CE probe sees it.

**Q2 (SL training-loop audit).**
`splitrisk sl --model gpt2 --split 6 --baseline sia` — run honest/sia/fsha
with identical parameters: honest acc≈0.89/t1≈0.94 · SIA task normal +
t1≥0.85 (caught) · FSHA acc<0.5 (collapses, self-exposes).

**Q3.** Embeddings ARE an intermediate layer the vendor exposes to you.
Inversion risk is measurable (vec2text lineage + SA-RI anchors). CLI is a
v2 preview — not shipped; honesty first.

**Q4.** The artifact crossing the boundary is plaintext — recovery is
total by definition, so there is nothing to measure (governance =
contracts, egress gateways, DLP). The tool refuses:

```python
>>> audit(model="gpt-4o", split_at=6)
ValueError: model 'gpt-4o' 不在錨點表支持列表
('gpt2', 'gpt2-medium', 'gpt2-large', 'llama-3.2-1b')。
未校準模型請走 audit_custom_head（讀數可出、verdict/簽章
需先完成 per-deployment 校準）。
```

That raise is honesty clause 10 in code: refusing to guess before making
the sale. Signature boundary: signable = calibrated rows only; readings
without verdict = uncalibrated; refusal = Q4 and any reduced-budget /
large-variance run.

See also: 宣傳包/06 for paste-ready samples per quadrant (internal doc).

---

See also [USERGUIDE.md](USERGUIDE.md#faq) for operational questions.

## Science

**Q: Is the 444x MSE/CE gap a universal constant?**
A: No. It is measured in our setting (GPT-2, AG News, layer 8). The
claim we stand behind: *the attacker's loss function, not its
architecture, sets the measurement*, and evaluation-dimension
omission can change rankings by orders of magnitude (cf. MIBench for
membership inference). Never quote 444x as an external universal claim.

**Q: Does quantization give privacy?**
A: At 4 bits the task dies (41x PPL). At 8-9 bits there is a narrow
region where classification utility survives and token recovery drops
(0.47 vs 0.97) — threat-model-dependent risk reduction, not privacy:
the ceiling is open (0.36 -> 0.42 -> 0.47 with decoder capacity) and
label leakage is untouched (0.859 vs 0.871).

**Q: Does depth (splitting later) help?**
A: It removes the *linearly* readable component (floor 0.069 -> 0.005)
and leaves learned decodability intact (CE probe flat at >= 0.93).
Obfuscation, not privacy.

**Q: Does sharding across servers help?**
A: It buys jurisdictional distribution and a lower single-shard view
(much of which is attacker-capacity dilution). The collusive ceiling
is set by bitrate alone. Report max_C R(C) with the budget.

**Q: What is P_ctx and why does it matter?**
A: The ambient anchor — what an attacker with a public LM but NO
access to your deployment already recovers (31.7% on AG News with
GPT-2 small; it rises with model scale). Excess leakage = t1 - P_ctx
is the compliance number; a raw t1 without its zero-point is not
interpretable.

## Product

**Q: Why did my verdict come back INCONCLUSIVE?**
A: t5 - t1 > 10pp means the probe's top-1 discrimination is
insufficient for this model's representation dimension — the
information is there (t5 high) but the probe is too small. Re-run
with a larger `probe_hidden`. SplitRisk refuses to guess rather than
issue a false GREEN/AMBER.

**Q: Which models are supported out of the box?**
A: `gpt2`, `gpt2-medium`, `gpt2-large` (calibrated). Anything else:
bring your own head via `audit_custom_head` and run
`splitrisk anchors` to calibrate per deployment.

**Q: Is SplitRisk a defense?**
A: No. It measures. Across 17 experiments, no software defense
preserved utility while blocking reconstruction. RED means: TEE,
local inference, or don't egress.

**Q: How is this different from privacy scanners / DPIA tooling?**
A: Those are qualitative. SplitRisk outputs a measured number
(cosine-recovery under a stated attacker budget) anchored at four
calibration points, reproducible to +-0.3pp across independent
sessions.

**Q: Commercial use?**
A: BSD-3 for this repository. The certification engine, subscription
service, and anchor-update stream are not part of this repo — see the
[commercial SplitAudit](mailto:alexlihk@hotmail.com).
