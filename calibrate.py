"""calibrate.py - per-kind temperature by L-BFGS on the calibration split (JEV: T ~ 1.00 / 0.98 / 1.01).

Minimises mean KL(target || softmax(z / T)) over the kind's active slots. Writes calibration.json in
JEV's shape ({"per_kind": {...}}) so jev_client and serve_decide can consume it, plus before/after KL and ECE.

    python calibrate.py --logits results/run1_calib.logits.jsonl --split calibration --out ckpt/run1/calibration.json
"""
import argparse
import json
import os
import sys

import torch

from zeroshot_text import KINDS, RANGES, active_probs, load_split, score

KINDS = ["noul", "score", "choice"]


def fit_temperature(zs, ts):
    """zs: list of active-slot logit lists; ts: targets. Returns T minimising mean KL."""
    n = max(len(z) for z in zs)
    Z = torch.full((len(zs), n), -1e9)
    T_ = torch.zeros((len(zs), n))
    for i, (z, t) in enumerate(zip(zs, ts)):
        Z[i, :len(z)] = torch.tensor(z)
        T_[i, :len(t)] = torch.tensor(t)
    logT = torch.zeros((), requires_grad=True)

    def loss():
        logp = torch.log_softmax(Z / logT.exp(), -1)
        return torch.sum(torch.where(T_ > 0, T_ * (torch.log(T_.clamp_min(1e-12)) - logp), torch.zeros_like(T_))) / len(zs)

    opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = loss()
        l.backward()
        return l
    opt.step(closure)
    return logT.exp().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"))
    ap.add_argument("--shuffle", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_split(args.data, args.split, True, args.shuffle, args.limit)
    outs = [json.loads(l) for l in open(args.logits)]
    assert len(outs) == len(rows), (len(outs), len(rows))
    for r, o in zip(rows, outs):
        assert "id" not in o or o["id"] == r["id"], (r["id"], o.get("id"))
    pairs = [(r, o["logits"]) for r, o in zip(rows, outs) if "error" not in o]

    temps = {}
    for k in KINDS:
        lo, _ = RANGES[k]
        zs = [z[lo:lo + len(r["options"])] for r, z in pairs if r["kind"] == k]
        ts = [r["target"] for r, _ in pairs if r["kind"] == k]
        temps[k] = fit_temperature(zs, ts) if zs else 1.0
    before = score([r for r, _ in pairs], [active_probs(r, z) for r, z in pairs])
    after = score([r for r, _ in pairs], [active_probs(r, z, temps) for r, z in pairs])
    for k in ["all"] + KINDS:
        print(f"{k:8s} T={temps.get(k, 1.0):.4f}  KL {before[k]['kl']:.4f} -> {after[k]['kl']:.4f}  "
              f"ECE {before[k]['ece']:.4f} -> {after[k]['ece']:.4f}  n={before[k]['n']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"per_kind": temps, "split": args.split, "n": len(pairs),
               "kl_before": {k: before[k]["kl"] for k in KINDS}, "kl_after": {k: after[k]["kl"] for k in KINDS}},
              open(args.out, "w"), indent=1)
    print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
