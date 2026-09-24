# Contributing

## Development setup

```bash
git clone https://github.com/alexlihk/splitrisk
cd splitrisk
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" datasets
pytest -m "not slow"        # fast tier: synthetic, no downloads
pytest -m slow              # real configs/checkpoints, network needed
```

## Ground rules (non-negotiable — these are the product)

1. **Cosine recovery is the only honest readout.** Never add an MSE-
   or CE-based privacy number to a verdict path.
2. **The attacker budget travels with every readout.** A new code
   path that emits a number without its budget vector is a bug.
3. **Refuse rather than guess.** Uncalibrated (model, corpus), probe
   under-capacity, or reduced-budget runs return INCONCLUSIVE.
4. **Never hand-call decoder blocks** — official model API only, then
   truncate. (transforms 5.x breaks hand-called GPT2Block; this is in
   the bug ledger.)
5. **Deterministic seeds** for anything that touches a number:
   1000+s (CLS) / 2000+s (LM); VQ 9000+K+100*seed.
6. **Honesty clauses stay in LICENSE.** If a change would require
   weakening a disclosure, the change is wrong.

## Adding a model family

Add a Head/Tail pair under `probes/` following probe_common.py, add
dimension tests, then **calibrate**: run `splitrisk anchors` on at
least one corpus and add the (model, corpus) row to
`calibration/anchors.py` with the probe capacity recorded. A model
without anchor rows is not "supported" — it is custom-head only.

## PR checklist

- [ ] `pytest -m "not slow"` green
- [ ] New numbers come with budget vectors and seeds
- [ ] LICENSE honesty clauses untouched
- [ ] No universal external claims (444x is our setting only)
