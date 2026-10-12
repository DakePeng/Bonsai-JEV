"""eval_text.py - KL, accuracy, ECE of a logits file on a split, optionally with per-kind temperatures.

Works on zero-shot (decide-logits) and trained (infer_logits) outputs alike. Writes <out>.metrics.json
and <out>.jsonl (per-row probabilities) for later side-by-side comparison.

    python eval_text.py --logits results/run1_test30k.logits.jsonl --split test_set_30k --calibration ckpt/run1/calibration.json --out results/run1_test30k
    python eval_text.py --logits results/run1_ood.logits.jsonl --split ood ...
"""
import argparse
import json
import os
import sys

from zeroshot_text import active_probs, load_split, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--split", default="test_set_30k")
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"))
    ap.add_argument("--shuffle", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calibration", default=None, help="calibration.json with per_kind temperatures")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_split(args.data, args.split, True, args.shuffle, args.limit)
    outs = [json.loads(l) for l in open(args.logits)]
    assert len(outs) == len(rows), (len(outs), len(rows))
    for r, o in zip(rows, outs):
        assert "id" not in o or o["id"] == r["id"], (r["id"], o.get("id"))
    temps = json.load(open(args.calibration))["per_kind"] if args.calibration else None

    kept, probs = [], []
    for r, o in zip(rows, outs):
        if "error" in o:
            continue
        kept.append(r)
        probs.append(active_probs(r, o["logits"], temps))
    m = score(kept, probs)
    print(f"{args.split}: {len(kept)} rows scored, {len(rows) - len(kept)} skipped, temps={temps}")
    for g in sorted(m, key=lambda g: (g != "all", g)):
        v = m[g]
        print(f"{g:16s} n={v['n']:6d}  KL={v['kl']:.4f}  acc={v['acc']:.4f}  ECE={v['ece']:.4f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(m, open(args.out + ".metrics.json", "w"), indent=1)
    with open(args.out + ".jsonl", "w") as f:
        for r, p in zip(kept, probs):
            f.write(json.dumps({"id": r["id"], "kind": r["kind"], "source": r["source"], "p": p, "target": r["target"]}) + "\n")
    print(f"-> {args.out}.metrics.json", file=sys.stderr)


if __name__ == "__main__":
    main()
