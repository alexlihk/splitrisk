# FAQ

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
