"""AuditResult — measurement record, verdict, and report rendering.

The attacker budget vector is a first-class field: a readout without
its budget is not interpretable (honesty clause 1).
"""
from dataclasses import dataclass, field, asdict


@dataclass
class AuditResult:
    model: str
    split_at: int
    task: str
    seed: int

    # --- the four key numbers ---
    t1_ce: float            # top-1 token recovery, CE probe (primary)
    t5_ce: float = None     # top-5 recovery — probe capacity check
    floor: float = None     # Tier-0 linear readability
    pctx: float = None      # ambient zero-point (attacker w/o deployment)

    t1_mse: float = None    # Tier-1 instrument comparison (MSE floor)
    task_acc: float = None
    mode: str = "audit"     # audit | vq | sharded | training
    extra: dict = field(default_factory=dict)   # mode-specific cells

    # --- attacker budget vector ---
    probe_arch: str = ""
    probe_hidden: int = None
    n_train: int = None
    epochs: int = None
    lr: float = None

    caveats: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def excess_leakage(self):
        """t1_ce − pctx, in percentage points. The compliance number."""
        if self.pctx is None:
            return None
        return 100.0 * (self.t1_ce - self.pctx)

    @property
    def probe_capacity_warning(self):
        """exp18b finding: probe training carries ~7pp run-to-run
        variance (5 configs on identical 774M representations: t1
        0.888-0.914, mean 0.907). A single-seed readout with a large
        t5-t1 gap is a low draw, not evidence about the representation.
        Single-seed verdicts are refused; average multiple seeds."""
        if self.t5_ce is None:
            return None
        if self.t5_ce - self.t1_ce > 0.10:
            return (f"t1={self.t1_ce:.1%} but t5={self.t5_ce:.1%} — "
                    f"probe training variance is ~7pp run-to-run "
                    f"(exp18b); re-run with 3+ seeds and average, or "
                    f"increase probe capacity, before interpreting.")
        return None

    @property
    def verdict(self) -> str:
        w = self.probe_capacity_warning
        if w is not None:
            return ("INCONCLUSIVE (probe capacity) — "
                    "re-run with a larger probe_hidden")
        # a reduced-budget run (smoke / plumbing check) cannot support
        # a tier: task fine-tuning and probe training are both
        # under-trained and the readouts are lower bounds
        if (self.n_train is not None and self.n_train < 500) or \
           (self.epochs is not None and self.epochs < 2):
            return ("INCONCLUSIVE (reduced-budget run — "
                    "plumbing check only; use n_train>=500, epochs>=2)")
        if self.pctx is not None and self.t1_ce < self.pctx + 0.05:
            return "GREEN"
        if 0.4 <= self.t1_ce <= 0.6:
            return "AMBER"
        if self.t1_ce >= 0.85:
            return "RED"
        return "AMBER (unclassified band — see USERGUIDE tier table)"

    # ------------------------------------------------------------------
    def pretty(self) -> str:
        lines = [
            "=" * 62,
            "  SplitRisk Audit Report",
            f"  Model: {self.model} | Split: {self.split_at} | "
            f"Task: {self.task} | Mode: {self.mode} | Seed: {self.seed}",
            "-" * 62,
            "  MEASUREMENTS",
            f"  +- Token recovery t1 (CE probe):  {self.t1_ce:6.1%}  <- primary",
        ]
        if self.t5_ce is not None:
            lines.append(f"  +- Token recovery t5 (CE probe):  "
                         f"{self.t5_ce:6.1%}  <- capacity check")
        if self.t1_mse is not None:
            lines.append(f"  +- Token recovery (MSE probe):    "
                         f"{self.t1_mse:6.1%}  <- instrument floor")
        if self.floor is not None:
            lines.append(f"  +- Linear floor (Tier 0):         {self.floor:6.1%}")
        if self.pctx is not None:
            lines.append(f"  +- Ambient (P_ctx):               {self.pctx:6.1%}")
        if self.task_acc is not None:
            lines.append(f"  +- Task accuracy:                 {self.task_acc:6.1%}")
        if self.excess_leakage is not None:
            lines.append(f"  Excess leakage (t1 - pctx):       "
                         f"{self.excess_leakage:+.1f}pp")
        lines += [
            "  ATTACKER BUDGET",
            f"  +- Probe: {self.probe_arch} (hidden={self.probe_hidden})",
            f"  +- Training: {self.n_train} samples, "
            f"{self.epochs} epochs, lr {self.lr}",
        ]
        for k, v in self.extra.items():
            lines.append(f"  [{self.mode}] {k}: {v}")
        lines.append("  CAVEATS")
        for c in self.caveats:
            lines.append(f"  ! {c}")
        lines.append(f"  VERDICT: {self.verdict}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["excess_leakage_pp"] = self.excess_leakage
        d["probe_capacity_warning"] = self.probe_capacity_warning
        d["verdict"] = self.verdict
        return d

    def to_json(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
