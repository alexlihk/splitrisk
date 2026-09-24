"""Certification engine tests (L2 preview)."""
import json

import pytest


def _result(tmp_path, t1=0.55, t5=0.60, n_train=5000, epochs=2):
    from splitrisk.report import AuditResult
    r = AuditResult(model="gpt2", split_at=6, task="classification",
                    seed=1006, t1_ce=t1, t5_ce=t5, floor=0.008,
                    pctx=0.317, probe_arch="MLP 768->2048->2048->768",
                    probe_hidden=2048, n_train=n_train, epochs=epochs,
                    lr=1e-3, mode="audit")
    p = tmp_path / "audit.json"
    r.to_json(str(p))
    return r, str(p)


def test_certify_issues_record_and_is_deterministic(tmp_path):
    from splitrisk.certify import certify
    r, _ = _result(tmp_path)
    rec1 = certify(r, org="ACME")
    rec2 = certify(r, org="ACME")
    assert rec1["cert_number"] == rec2["cert_number"]   # content-addressed
    assert rec1["cert_number"].startswith("SA-RI-")
    assert rec1["tier"] == "AMBER"
    assert len(rec1["caveats"]) == 10
    assert rec1["legal_notice"].startswith("本報告依 SA-RI schema v1.1")
    assert "保鮮期" in rec1["freshness"]


def test_certify_refuses_inconclusive(tmp_path):
    from splitrisk.certify import certify, CertifyRefused
    r, _ = _result(tmp_path, t1=0.845, t5=0.954)   # variance gap → INCONCLUSIVE
    with pytest.raises(CertifyRefused):
        certify(r)


def test_certify_refuses_reduced_budget(tmp_path):
    from splitrisk.certify import certify, CertifyRefused
    r, _ = _result(tmp_path, n_train=60, epochs=1)
    with pytest.raises(CertifyRefused):
        certify(r)


def test_certify_cli_roundtrip(tmp_path, capsys):
    from splitrisk.cli import main
    r, ap = _result(tmp_path)
    out = str(tmp_path / "sample")
    rc = main(["certify", "--input", ap, "--out-prefix", out,
               "--org", "ACME"])
    assert rc == 0
    md = (tmp_path / "sample.cert.md").read_text(encoding="utf-8")
    assert "SA-RI 認證報告" in md and "誠實條款" in md
    rec = json.loads((tmp_path / "sample.cert.json")
                     .read_text(encoding="utf-8"))
    assert rec["org"] == "ACME"


def test_refused_cli_exit_code(tmp_path):
    from splitrisk.cli import main
    r, ap = _result(tmp_path, t1=0.845, t5=0.954)
    out = str(tmp_path / "x")
    assert main(["certify", "--input", ap, "--out-prefix", out]) == 2
