"""Anchor calibration tests + report logic (no checkpoints needed)."""
import pytest


# ---------------------------------------------------------------- anchors
def test_anchor_table_contents():
    from splitrisk.calibration.anchors import CALIBRATION_TABLE, lookup
    assert ("gpt2", "ag_news") in CALIBRATION_TABLE
    assert ("gpt2-large", "ag_news") in CALIBRATION_TABLE
    assert ("gpt2", "dbpedia") in CALIBRATION_TABLE
    assert ("llama-3.2-1b", "ag_news") in CALIBRATION_TABLE
    assert lookup("gpt2", "ag_news") == (0.94, 0.317, 0.008)


def test_gpt2_large_anchor_is_exp18b_mean():
    """exp18's single-seed 0.845 was a ~7pp run-to-run outlier; the
    shipped anchor must be the exp18b 5-config mean."""
    from splitrisk.calibration.anchors import lookup
    full, ambient, floor = lookup("gpt2-large", "ag_news")
    assert full == 0.907 and ambient == 0.374 and floor == 0.005


def test_lookup_refuses_uncalibrated():
    from splitrisk.calibration.anchors import lookup
    with pytest.raises(KeyError, match="Per-deployment"):
        lookup("qwen-7b", "ag_news")   # no exp19-equivalent calibration


def test_zero_info_and_closure_anchors():
    from splitrisk.calibration.anchors import (ZERO_INFO_ANCHOR,
                                               TWO_PROBE_CLOSURE)
    assert ZERO_INFO_ANCHOR == 0.030
    lo, hi = TWO_PROBE_CLOSURE
    assert abs(lo - hi) <= 0.0061   # two-probe closure holds (float tol)


# ------------------------------------------------------------ report logic
def _result(t1, t5=None, pctx=0.317):
    from splitrisk.report import AuditResult
    return AuditResult(model="gpt2", split_at=6, task="classification",
                       seed=1006, t1_ce=t1, t5_ce=t5, floor=0.008,
                       pctx=pctx)


def test_verdict_refuses_reduced_budget():
    from splitrisk.report import AuditResult
    r = AuditResult(model="gpt2", split_at=6, task="classification",
                    seed=1006, t1_ce=0.05, floor=0.01, pctx=0.011,
                    n_train=60, epochs=1)
    assert "reduced-budget" in r.verdict   # smoke run must not issue GREEN


def test_verdict_green_amber_red():
    assert _result(0.30).verdict == "GREEN"
    assert _result(0.50).verdict == "AMBER"
    assert _result(0.94).verdict == "RED"


def test_capacity_warning_fires_on_exp18_numbers():
    r = _result(0.845, 0.954)   # exp18 single-seed low draw
    assert r.probe_capacity_warning is not None
    assert "variance" in r.probe_capacity_warning
    assert "INCONCLUSIVE" in r.verdict   # refusal mechanism unchanged


def test_no_warning_when_gap_small():
    r = _result(0.94, 0.96)
    assert r.probe_capacity_warning is None
    assert r.verdict == "RED"


def test_excess_leakage():
    r = _result(0.94, pctx=0.317)
    assert abs(r.excess_leakage - (0.94 - 0.317) * 100) < 1e-6


def test_report_roundtrip_and_pretty(tmp_path):
    import json
    r = _result(0.94, 0.96)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["verdict"] == "RED"
    p = tmp_path / "r.json"
    r.to_json(str(p))
    assert json.loads(p.read_text(encoding="utf-8"))["t1_ce"] == 0.94
    assert "VERDICT" in r.pretty()


# ---------------------------------------------------------------- presets
def test_presets_exist():
    from splitrisk.presets import PRESETS
    for name in ("default", "healthcare", "finance", "cross_border"):
        assert name in PRESETS
    # healthcare: category-is-sensitive = RED, no AMBER offered
    assert PRESETS["healthcare"]["amber"] == "not available"
