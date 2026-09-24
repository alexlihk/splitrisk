"""Audit your own head (bring-your-own-head interface).

Your head must satisfy: forward(input_ids, attention_mask) -> hidden
states (N, T, D) at the split point. See probes/probe_common.py Head.
"""
from splitrisk import audit_custom_head

# from my_package import MyCustomHead
# head = MyCustomHead(...)
# result = audit_custom_head(
#     head=head,
#     split_at=6,
#     task="classification",
#     data="my_org/my_dataset",     # HF dataset id
#     text_column="content",
#     label_column="category",
# )
# print(result.pretty())
print(__doc__)
