"""Generate a full JSON + Markdown report for a deployment audit."""
import argparse
from splitrisk import audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--split", type=int, default=6)
    ap.add_argument("--task", default="classification")
    ap.add_argument("--data", default="ag_news")
    ap.add_argument("--out", default="audit_report.json")
    args = ap.parse_args()

    r = audit(model=args.model, split_at=args.split, task=args.task,
              data=args.data)
    r.to_json(args.out)
    print(r.pretty())
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
