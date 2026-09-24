"""5-line quickstart — your first split audit (~20 min on a T4)."""
from splitrisk import audit

result = audit(model="gpt2", split_at=6, task="classification",
               data="ag_news")
print(result.pretty())
