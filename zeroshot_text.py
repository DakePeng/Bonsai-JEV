"""zeroshot_text.py - zero-shot decision baseline of Bonsai 2 (no LoRA, no head) on the JEV corpus.

Renders JEV's bare-v1 prompt, reads the next-token logits of the verbalizer tokens through the
fork (tools/decide-logits), softmaxes over the kind's active options, and scores against the
target distributions. Per-row probabilities are saved so the trained model can be compared later.

    python zeroshot_text.py --split test_set_30k --limit 300
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

LETTERS = "ABCDEFGHIJKLMNOP"
# JEV judge_config.json: slots 0-1 noul, 2-7 score, 8-23 choice
VERBALIZER_IDS = [3721, 1802, 15, 16, 17, 18, 19, 20] + list(range(32, 48))
RANGES = {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}


def as_text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def render(row):
    kind, opts = row["kind"], row["options"]
    lines = opts if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(opts)]
    return (f"[kind] {kind}\n[state] {as_text(row['state'])}\n[question] {row['question']}\n[options]\n"
            + "\n".join(lines) + "\n[decision]:")


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def kl(t, p, eps=1e-9):
    return sum(ti * (math.log(ti) - math.log(max(pi, eps))) for ti, pi in zip(t, p) if ti > 0)


def argmax(v):
    return max(range(len(v)), key=lambda i: v[i])


def score(rows, probs):
    """Returns dict of metrics; rows/probs aligned. ECE on max-prob confidence vs target-argmax correctness."""
    out = {}
    groups = defaultdict(list)
    for r, p in zip(rows, probs):
        groups["all"].append((r, p))
        groups[r["kind"]].append((r, p))
        groups["src:" + r["source"]].append((r, p))
    for g, items in groups.items():
        kls, acc, conf, corr = [], [], [], []
        for r, p in items:
            t = r["target"]
            kls.append(kl(t, p))
            c = argmax(p) == argmax(t)
            acc.append(c)
            conf.append(max(p))
            corr.append(c)
        bins = [[] for _ in range(10)]
        for cf, cr in zip(conf, corr):
            bins[min(int(cf * 10), 9)].append((cf, cr))
        ece = sum(len(b) / len(items) * abs(sum(c for c, _ in b) / len(b) - sum(r for _, r in b) / len(b))
                  for b in bins if b)
        out[g] = {"n": len(items), "kl": sum(kls) / len(kls), "acc": sum(acc) / len(acc), "ece": ece}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_set_30k")
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", type=int, default=None, help="seed; shuffle rows before --limit")
    ap.add_argument("--skip-placeholder", action="store_true", help="drop yuri_v1 rows (placeholder [0.5,0.5] labels)")
    ap.add_argument("--model", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--tool", default=os.path.expanduser("~/bonsai/llama.cpp/build/bin/llama-decide-logits"))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-logits", default=None, help="rescore an existing tool output instead of running the model")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run: skip rows already in the raw logits file")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.data, args.split + ".jsonl"))]
    if args.skip_placeholder:
        rows = [r for r in rows if r["source"] != "yuri_v1"]
    if args.shuffle is not None:
        import random
        random.Random(args.shuffle).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} rows", file=sys.stderr)

    if args.from_logits:
        outs = [json.loads(l) for l in open(args.from_logits)]
    else:
        raw = (args.out or f"zeroshot_{args.split}") + ".logits.jsonl"
        start = 0
        if args.resume and os.path.exists(raw):
            start = sum(1 for _ in open(raw))
            print(f"resuming after {start} rows", file=sys.stderr)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in rows[start:]:
                f.write(json.dumps({"prompt": render(r)}, ensure_ascii=False) + "\n")
            pin = f.name
        env = dict(os.environ, CANDS=",".join(map(str, VERBALIZER_IDS)))
        if start < len(rows):
            with open(pin) as fi, open(raw, "a" if start else "w") as fo:
                subprocess.run([args.tool, "-m", args.model, "-ngl", "99", "-c", str(args.ctx), "--log-disable"],
                               stdin=fi, stdout=fo, env=env, check=True)
        outs = [json.loads(l) for l in open(raw)]
        print(f"raw logits -> {raw}", file=sys.stderr)
    assert len(outs) == len(rows), (len(outs), len(rows))

    kept, probs, skipped = [], [], 0
    for r, o in zip(rows, outs):
        if "error" in o:
            skipped += 1
            continue
        lo, hi = RANGES[r["kind"]]
        z = o["logits"][lo:lo + len(r["options"])]
        kept.append(r)
        probs.append(softmax(z))
    print(f"scored {len(kept)}, skipped {skipped} (too long)", file=sys.stderr)

    m = score(kept, probs)
    for g in sorted(m, key=lambda g: (not g == "all", g)):
        v = m[g]
        print(f"{g:16s} n={v['n']:6d}  KL={v['kl']:.4f}  acc={v['acc']:.4f}  ECE={v['ece']:.4f}")
    outp = (args.out or f"zeroshot_{args.split}") + ".jsonl"
    with open(outp, "w") as f:
        for r, p in zip(kept, probs):
            f.write(json.dumps({"id": r["id"], "kind": r["kind"], "source": r["source"], "p": p, "target": r["target"]}) + "\n")
    json.dump(m, open((args.out or f"zeroshot_{args.split}") + ".metrics.json", "w"), indent=1)
    print(f"per-row probs -> {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
