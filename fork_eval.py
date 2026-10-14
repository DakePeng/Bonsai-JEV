"""fork_eval.py - 24-slot decision logits through the deployment path (PrismML fork + adapter.gguf), optionally with
the options of choice rows permuted, written in load_split() order with logits mapped back to the original option
order, so calibrate.py / eval_text.py and the order comparison all consume the same format.

  logits  python fork_eval.py --adapter-dir D --split test_set_30k --out results/x.logits.jsonl [--permute reverse]
  order   python fork_eval.py --compare results/x.logits.jsonl results/x_rev.logits.jsonl --split test_set_30k

`--adapter-dir` holds adapter.gguf + decision.json (head bias is added to the verbalizer logits, as the server does).
Without it, the zero-shot base is measured. `--kinds choice` restricts to choice rows (for the order test).
"""
import argparse
import json
import math
import os
import random
import subprocess
import tempfile

from zeroshot_text import LETTERS, RANGES, VERBALIZER_IDS, active_probs, as_text, load_split


def perm_for(r, mode):
    n = len(r["options"])
    if r["kind"] != "choice" or mode == "none":
        return list(range(n))
    if mode == "reverse":
        return list(range(n))[::-1]
    if mode.startswith("shuffle"):
        seed = int(mode.split(":")[1]) if ":" in mode else 0
        p = list(range(n))
        random.Random(f"{seed}-{r['id']}").shuffle(p)
        return p
    raise ValueError(mode)


def render_perm(r, p):
    """bare-v1 prompt with options shown in order p (p[j] = original index shown at position j)."""
    opts = [r["options"][i] for i in p]
    lines = opts if r["kind"] != "choice" else [f"{LETTERS[j]}) {o}" for j, o in enumerate(opts)]
    return (f"[kind] {r['kind']}\n[state] {as_text(r['state'])}\n[question] {r['question']}\n[options]\n"
            + "\n".join(lines) + "\n[decision]:")


def run(args):
    rows = load_split(args.data, args.split, True, None, args.limit)
    if args.kinds:
        rows = [r for r in rows if r["kind"] in args.kinds.split(",")]
    perms = [perm_for(r, args.permute) for r in rows]
    bias = [0.0] * 24
    lora = []
    if args.adapter_dir:
        bias = json.load(open(os.path.join(args.adapter_dir, "decision.json"))).get("head_bias", bias)
        lora = ["--lora", os.path.join(args.adapter_dir, "adapter.gguf")]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r, p in zip(rows, perms):
            f.write(json.dumps({"prompt": render_perm(r, p)}, ensure_ascii=False) + "\n")
        pin = f.name
    raw = args.out + ".raw"
    env = dict(os.environ, CANDS=",".join(map(str, VERBALIZER_IDS)))
    with open(pin) as fi, open(raw, "w") as fo:
        subprocess.run([args.tool, "-m", args.model, *lora, "-ngl", "99", "-c", str(args.ctx), "--log-disable"],
                       stdin=fi, stdout=fo, env=env, check=True)
    outs = [json.loads(l) for l in open(raw)]
    assert len(outs) == len(rows)
    with open(args.out, "w") as f:
        for r, p, o in zip(rows, perms, outs):
            if "error" in o:
                f.write(json.dumps({"id": r["id"], "error": o["error"]}) + "\n")
                continue
            z = [a + b for a, b in zip(o["logits"], bias)]
            lo, _ = RANGES[r["kind"]]
            # slot lo+j holds the logit of the option shown at position j = original option p[j]; map back
            z2 = list(z)
            for j, i in enumerate(p):
                z2[lo + i] = z[lo + j]
            f.write(json.dumps({"id": r["id"], "logits": z2}) + "\n")
    os.remove(raw)
    print(f"{len(rows)} rows -> {args.out} (permute={args.permute}, adapter={'yes' if lora else 'no'})")


def compare(args):
    a_path, b_path = args.compare
    A = {o["id"]: o for o in map(json.loads, open(a_path))}
    B = {o["id"]: o for o in map(json.loads, open(b_path))}
    rows = [r for r in load_split(args.data, args.split, True, None, 0) if r["id"] in A and r["id"] in B
            and "logits" in A[r["id"]] and "logits" in B[r["id"]]]
    agree, tv, acc_a, acc_b = 0, 0.0, 0, 0
    for r in rows:
        pa, pb = active_probs(r, A[r["id"]]["logits"]), active_probs(r, B[r["id"]]["logits"])
        ia, ib = max(range(len(pa)), key=pa.__getitem__), max(range(len(pb)), key=pb.__getitem__)
        it = max(range(len(r["target"])), key=r["target"].__getitem__)
        agree += ia == ib
        acc_a += ia == it
        acc_b += ib == it
        tv += 0.5 * sum(abs(x - y) for x, y in zip(pa, pb))
    n = len(rows)
    print(f"{n} rows | order-agree {100 * agree / n:.1f}% | mean total-variation between orders {tv / n:.4f} | "
          f"top-1 vs target: original {100 * acc_a / n:.1f}%  permuted {100 * acc_b / n:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--split", default="test_set_30k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kinds", default=None)
    ap.add_argument("--permute", default="none", help="none | reverse | shuffle[:seed]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"))
    ap.add_argument("--model", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--tool", default=os.path.expanduser("~/bonsai/llama.cpp/build/bin/llama-decide-logits"))
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()
    compare(args) if args.compare else run(args)


if __name__ == "__main__":
    main()
