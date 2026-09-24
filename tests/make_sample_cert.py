"""Generate a REAL dummy certification report (W4 milestone):
run a genuine audit (passes reduced-budget guard) then certify it.
Marked SAMPLE/DUMMY throughout — never presented as a client engagement.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from splitrisk import audit
from splitrisk.certify import certify_to_files

r = audit(model="gpt2", split_at=6, task="classification",
          data="ag_news", n_train=1000, n_test=300, epochs=2, seed=1006)
print(r.pretty())
out = r"C:\Users\admin\.zcode\workspace\default\SA-RI\sample_certification"
os.makedirs(out, exist_ok=True)
jp, mp = certify_to_files(r, out, org="SAMPLE ORG",
                          engagement="DUMMY PIPELINE — engine verification")
print(f"[saved] {jp}\n[saved] {mp}")
