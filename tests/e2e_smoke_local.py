"""Real end-to-end usability check: audit() on actual gpt2 + AG News.
Small-sample smoke (CPU-friendly): n=60 train / 30 test, 1 epoch.
Verifies plumbing, not science — numbers will NOT match anchors.
"""
from splitrisk import audit

r = audit(model="gpt2", split_at=6, task="classification", data="ag_news",
          n_train=60, n_test=30, epochs=1, seed=1006)
print(r.pretty())
assert r.t1_ce is not None and 0 <= r.t1_ce <= 1
assert r.t5_ce >= r.t1_ce
assert r.pctx is not None
assert r.verdict
print("END-TO-END SMOKE: PASS")
