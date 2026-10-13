"""check_export_real.py - step 15 with a real trained checkpoint: the fork + exported adapter.gguf vs the PyTorch
model that trained it, on the same validation rows.

PyTorch side: infer_logits.py output on the rental box (head logits incl. bias, 24 per row).
Fork side: decide-logits with --lora adapter.gguf (verbalizer-token logits = (W_init + delta) . h), + head bias
from decision.json. Both softmaxed over the kind's active slots.

    python check_export_real.py --ckpt-dir ~/bonsai/ckpt-smoke --pytorch ~/bonsai/ckpt-smoke/smoke_val.logits.jsonl --limit 1024
"""
import argparse
import json
import math
import os
import subprocess
import tempfile

from zeroshot_text import VERBALIZER_IDS, active_probs, load_split, render, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="dir with adapter.gguf + decision.json")
    ap.add_argument("--pytorch", required=True, help="infer_logits.py output for the same rows")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=1024)
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"))
    ap.add_argument("--model", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--tool", default=os.path.expanduser("~/bonsai/llama.cpp/build/bin/llama-decide-logits"))
    args = ap.parse_args()

    rows = load_split(args.data, args.split, True, None, args.limit)
    pt = [json.loads(l) for l in open(args.pytorch)][:len(rows)]
    assert all(p["id"] == r["id"] for p, r in zip(pt, rows)), "row order mismatch"
    dec = json.load(open(os.path.join(args.ckpt_dir, "decision.json")))
    bias = dec["head_bias"]

    out = os.path.join(args.ckpt_dir, f"fork_{args.split}.logits.jsonl")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps({"prompt": render(r)}, ensure_ascii=False) + "\n")
        pin = f.name
    env = dict(os.environ, CANDS=",".join(map(str, VERBALIZER_IDS)))
    with open(pin) as fi, open(out, "w") as fo:
        subprocess.run([args.tool, "-m", args.model, "--lora", os.path.join(args.ckpt_dir, "adapter.gguf"),
                        "-ngl", "99", "-c", "4096", "--log-disable"], stdin=fi, stdout=fo, env=env, check=True)
    fk = [json.loads(l) for l in open(out)]

    p_pt, p_fk, worst, kls, agree = [], [], 0.0, [], 0
    for r, a, b in zip(rows, pt, fk):
        pa = active_probs(r, a["logits"])
        pb = active_probs(r, [z + c for z, c in zip(b["logits"], bias)])
        p_pt.append(pa)
        p_fk.append(pb)
        worst = max(worst, max(abs(x - y) for x, y in zip(pa, pb)))
        kls.append(sum(x * (math.log(max(x, 1e-12)) - math.log(max(y, 1e-12))) for x, y in zip(pa, pb)))
        agree += max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__)
    n = len(rows)
    print(f"{n} rows: max |p_pytorch - p_fork| = {worst:.4f}, mean KL(pt||fork) = {sum(kls)/n:.5f}, argmax agree {agree/n:.4f}")
    ms, mf = score(rows, p_pt), score(rows, p_fk)
    for g in ["all", "noul", "score", "choice"]:
        if g in ms:
            print(f"{g:7s} KL-to-target  pytorch {ms[g]['kl']:.4f}   fork {mf[g]['kl']:.4f}   acc {ms[g]['acc']:.4f} / {mf[g]['acc']:.4f}")
    ok = sum(kls) / n < 0.01 and agree / n > 0.97
    print("PASS" if ok else "CHECK")


if __name__ == "__main__":
    main()
