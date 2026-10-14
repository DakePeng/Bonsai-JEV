import json, os, sys
from zeroshot_text import load_split, active_probs, score
logits, calib = sys.argv[1], sys.argv[2]
rows = load_split(os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"), "test_set_30k", True)
outs = [json.loads(l) for l in open(logits)]
T = json.load(open(calib))["per_kind"]
sel = [(r, o) for r, o in zip(rows, outs) if r["source"] == "yuri_v3" and "logits" in o]
probs = [active_probs(r, o["logits"], T) for r, o in sel]
m = score([r for r, _ in sel], probs)

def ece_vs_target(pairs, nb=15):
    # reliability against the soft targets: per (row, option), bin the predicted prob, compare with the target prob
    bins = [[0.0, 0.0, 0] for _ in range(nb)]
    for p, t in pairs:
        for pi, ti in zip(p, t):
            b = bins[min(int(pi * nb), nb - 1)]
            b[0] += pi; b[1] += ti; b[2] += 1
    n = sum(b[2] for b in bins)
    return sum(abs(b[0] - b[1]) / max(1, b[2]) * b[2] / n for b in bins if b[2])

for k in ["all", "noul", "score", "choice"]:
    pr = [(p, r["target"]) for (r, _), p in zip(sel, probs) if k == "all" or r["kind"] == k]
    v = m[k]
    print(f"{k:7s} n={v['n']:6d}  KL={v['kl']:.4f}  top-1 vs Jev={v['acc']:.4f}  ECE(vs target probs)={ece_vs_target(pr):.4f}")
