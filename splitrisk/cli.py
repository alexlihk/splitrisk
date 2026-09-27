#!/usr/bin/env python3
"""SplitRisk CLI — audit from the command line.

splitrisk audit    full audit of a split deployment
splitrisk anchors  4-anchor calibration check (mandatory before verdicts)
splitrisk vq       VQ bitrate sweep
splitrisk sharded  sharding audit
splitrisk presets  print threshold presets
"""
import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="splitrisk",
        description="Audit what your split model leaks. "
                    "Measurement tool — not a defense.")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("audit", help="Full audit of a split deployment")
    a.add_argument("--model", default="gpt2")
    a.add_argument("--split", type=int, default=6)
    a.add_argument("--task", choices=["cls", "lm", "classification"],
                   default="cls")
    a.add_argument("--data", default="ag_news")
    a.add_argument("--n-train", type=int, default=5000)
    a.add_argument("--n-test", type=int, default=1000)
    a.add_argument("--seed", type=int, default=1006)
    a.add_argument("--probe-hidden", type=int, default=None,
                   help="Probe hidden size (default: auto, "
                        "max(2048, 2*model_hidden))")
    a.add_argument("--output", default=None, help="Save JSON report")

    an = sub.add_parser("anchors", help="Run 4-anchor calibration check")
    an.add_argument("--model", default="gpt2")
    an.add_argument("--data", default="ag_news")

    v = sub.add_parser("vq", help="VQ bitrate sweep")
    v.add_argument("--model", default="gpt2")
    v.add_argument("--split", type=int, default=6)
    v.add_argument("--k", nargs="+", type=int, default=[256, 512, 1024])
    v.add_argument("--output", default=None)

    sh = sub.add_parser("sharded", help="Sharding audit")
    sh.add_argument("--model", default="gpt2")
    sh.add_argument("--split", type=int, default=6)
    sh.add_argument("--n-shards", type=int, default=3)
    sh.add_argument("--vq-bits", type=int, default=None)
    sh.add_argument("--output", default=None)

    pr = sub.add_parser("presets", help="Print threshold presets")
    pr.add_argument("--name", default=None)

    ce = sub.add_parser("certify", help="Certify a saved audit result (L2 preview)")
    ce.add_argument("--input", required=True, help="AuditResult JSON (from --output)")
    ce.add_argument("--org", default="SAMPLE")
    ce.add_argument("--engagement", default="DUMMY PIPELINE")
    ce.add_argument("--out-prefix", required=True)

    tr = sub.add_parser("training",
                        help="Training-track audit (R6) on a delivered model")
    tr.add_argument("--model", required=True, help="Delivered model (HF id)")
    tr.add_argument("--split", type=int, default=6)
    tr.add_argument("--n-test", type=int, default=1000)
    tr.add_argument("--n-train", type=int, default=None)
    tr.add_argument("--seed", type=int, default=1006)
    tr.add_argument("--output", default=None, help="Save JSON report")

    sla = sub.add_parser("sl", help="SL training-loop audit (honest/sia/fsha)")
    sla.add_argument("--model", default="gpt2")
    sla.add_argument("--split", type=int, default=6)
    sla.add_argument("--data", default="ag_news")
    sla.add_argument("--baseline", choices=["honest", "sia", "fsha"],
                     default="honest")
    sla.add_argument("--alpha", type=float, default=5.0)
    sla.add_argument("--sl-epochs", type=int, default=1)
    sla.add_argument("--n-train", type=int, default=5000)
    sla.add_argument("--n-test", type=int, default=1000)
    sla.add_argument("--seed", type=int, default=1006)
    sla.add_argument("--n-seeds", type=int, default=1)
    sla.add_argument("--output", default=None, help="Save JSON report")

    args = p.parse_args(argv)

    if args.version:
        from splitrisk import __version__
        print(f"splitrisk {__version__}")
        return 0

    if args.cmd == "audit":
        from splitrisk import audit
        task = "classification" if args.task in ("cls",
                                                 "classification") else "lm"
        r = audit(model=args.model, split_at=args.split, task=task,
                  data=args.data, n_train=args.n_train,
                  n_test=args.n_test, seed=args.seed,
                  probe_hidden=args.probe_hidden)
        print(r.pretty())
        if args.output:
            r.to_json(args.output)
            print(f"[saved] {args.output}")
    elif args.cmd == "anchors":
        from splitrisk.calibration import run_anchor_check
        run_anchor_check(model=args.model, corpus=args.data)
    elif args.cmd == "vq":
        from splitrisk import audit_vq
        r = audit_vq(model=args.model, split_at=args.split,
                     codebook_sizes=tuple(args.k))
        print(r.pretty())
        if args.output:
            r.to_json(args.output)
    elif args.cmd == "sharded":
        from splitrisk import audit_sharded
        r = audit_sharded(model=args.model, split_at=args.split,
                          n_shards=args.n_shards, vq_bits=args.vq_bits)
        print(r.pretty())
        if args.output:
            r.to_json(args.output)
    elif args.cmd == "certify":
        from splitrisk.certify import certify_to_files, CertifyRefused
        from splitrisk.report import AuditResult
        with open(args.input, encoding="utf-8") as f:
            r = AuditResult(**{k: v for k, v in json.load(f).items()
                               if k in AuditResult.__dataclass_fields__})
        try:
            jp, mp = certify_to_files(r, args.out_prefix, org=args.org,
                                      engagement=args.engagement)
            print(f"[certified] {jp}\n{mp}")
        except CertifyRefused as e:
            print(f"[REFUSED] {e}")
            return 2
    elif args.cmd == "training":
        from splitrisk import audit_training
        r = audit_training(delivered_model=args.model, split_at=args.split,
                           n_test=args.n_test, n_train=args.n_train,
                           seed=args.seed)
        print(r.pretty())
        if args.output:
            r.to_json(args.output)
            print(f"[saved] {args.output}")
    elif args.cmd == "sl":
        from splitrisk import audit_sl
        r = audit_sl(model=args.model, split_at=args.split, data=args.data,
                     baseline=args.baseline, alpha=args.alpha,
                     sl_epochs=args.sl_epochs, n_train=args.n_train,
                     n_test=args.n_test, seed=args.seed,
                     n_seeds=args.n_seeds)
        print(r.pretty())
        if args.output:
            r.to_json(args.output)
            print(f"[saved] {args.output}")
    elif args.cmd == "presets":
        from splitrisk.presets import PRESETS
        if args.name:
            print(json.dumps(PRESETS[args.name], indent=2))
        else:
            for name, cfg in PRESETS.items():
                print(f"[{name}] {cfg['note']}")
    else:
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
