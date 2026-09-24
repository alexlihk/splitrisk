"""Preset threshold templates for common deployment classes.

The decision threshold is the institution's, not ours (three-way
separation of powers: measurement / decision / supervision). Presets
are defaults, always overridable.
"""

PRESETS = {
    "default": {
        "green": "t1 < pctx + 5pp",
        "amber": "0.40 <= t1 <= 0.60 and task alive",
        "red": "t1 >= 0.85",
        "note": "Baseline tiers from the SA-RI paper.",
    },
    "healthcare": {
        "green": "not available",
        "amber": "not available",
        "red": "any PHI-bearing corpus",
        "note": ("Category-is-sensitive = RED. Label leakage is not "
                 "reduced by bitrate (R2 = 0.859 at 9 bits); AMBER is "
                 "not offered for diagnosis/health classes."),
    },
    "finance": {
        "green": "t1 < pctx + 5pp and label-AUC < 0.35",
        "amber": "0.40 <= t1 <= 0.60 and label-AUC < 0.55 and quarterly re-audit",
        "red": "t1 >= 0.85 or label-AUC >= 0.55",
        "note": ("Credit/risk rating classes are sensitive: the label "
                 "axis (R2) is gated alongside token recovery."),
    },
    "cross_border": {
        "green": "t1 < pctx + 5pp",
        "amber": "0.40 <= t1 <= 0.60 and collusive ceiling < 0.60 "
                 "and freshness < 90 days",
        "red": "t1 >= 0.85 or ceiling open",
        "note": ("GDPR TIA / PIPL Art.38 shape: verdict carries the "
                 "attacker budget vector, anchor versions and a "
                 "freshness label (ceiling is open)."),
    },
}
