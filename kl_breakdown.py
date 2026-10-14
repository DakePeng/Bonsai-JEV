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

# JEV-27B card metrics (its "Headline results" table and model-index), same rows/definitions as far as the card states them
def auroc(pos, neg):
    # Mann-Whitney: P(score of a true row > score of a false row), ties count half
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    rank_sum, i = 0.0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(lab for _, lab in s[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

alls = [(r, active_probs(r, o["logits"], T)) for r, o in zip(rows, outs) if "logits" in o]   # Jev + openjev rows
def noul_auroc(items):
    items = [(r, p) for r, p in items if r["kind"] == "noul" and r["target"][1] != 0.5]
    return auroc([p[1] for r, p in items if r["target"][1] > 0.5], [p[1] for r, p in items if r["target"][1] < 0.5])
jev = list(zip([r for r, _ in sel], probs))
top = lambda v: max(range(len(v)), key=v.__getitem__)
ch = [(r, p) for r, p in jev if r["kind"] == "choice"]
dec = [(r, p) for r, p in alls if r["kind"] == "choice" and (lambda t: t[0] - t[1])(sorted(r["target"], reverse=True)) >= 0.1]
sc = [(r, p) for r, p in jev if r["kind"] == "score"]
ev = lambda v: sum(i * x for i, x in enumerate(v))
nl = [(r, p) for r, p in alls if r["kind"] == "noul"]
print(f"noul AUROC: Jev-labelled {noul_auroc(jev):.4f}, all non-placeholder {noul_auroc(alls):.4f}   (JEV-27B 0.995 / 0.996)")
print(f"choice top-1 vs Jev, Jev-labelled {sum(top(p) == top(r['target']) for r, p in ch) / len(ch):.4f}   (JEV-27B 0.905)")
print(f"choice top-1, rows with target top-2 gap >= 0.1 {sum(top(p) == top(r['target']) for r, p in dec) / len(dec):.4f} (n={len(dec)})   (JEV-27B 0.958)")
print(f"score expected-value MAE, Jev-labelled {sum(abs(ev(p) - ev(r['target'])) for r, p in sc) / len(sc):.4f}   (JEV-27B 0.098)")
print(f"noul Brier vs target prob, all non-placeholder {sum((p[1] - r['target'][1]) ** 2 for r, p in nl) / len(nl):.4f}   (JEV-27B 0.0013, incl. placeholder rows)")
