"""infer_logits.py - 24-slot decision logits of a trained checkpoint (LoRA + head) on a corpus split.

Output: JSONL {"id", "logits": [24]} in load_split() order, the same format decide-logits produces,
so calibrate.py / eval_text.py score zero-shot and trained runs identically. Runs the PyTorch model
(needs the full BF16 base on the rented card; --layers for smoke tests).

    python infer_logits.py --ckpt ckpt/run1/final --split calibration --out results/run1_calib.logits.jsonl
"""
import argparse
import json
import os
import sys

import torch

from bonsai_torch import Bonsai, build_model, text_config
from train import forward_hidden, micro_batches
from zeroshot_text import load_split


def load_checkpoint(ckpt, b, cfg, device, dtype, n_layers=None):
    from peft import PeftModel
    from safetensors.torch import load_file
    base = build_model(b, cfg, dtype=dtype, device=device, n_layers=n_layers)
    model = PeftModel.from_pretrained(base, ckpt).to(device)
    model.eval()
    hd = load_file(os.path.join(ckpt, "head.safetensors"))
    head = torch.nn.Linear(cfg.hidden_size, 24, bias=True, dtype=torch.float32)
    head.load_state_dict(hd)
    return model, head.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"))
    ap.add_argument("--gguf", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--cfg", default=os.path.expanduser("~/bonsai/models/qwen3.8-27b-cfg"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    device, dtype = args.device, getattr(torch, args.dtype)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.cfg)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    b, cfg = Bonsai(args.gguf), text_config(args.cfg)
    model, head = load_checkpoint(args.ckpt, b, cfg, device, dtype, args.layers)
    rows = load_split(args.data, args.split, True, args.shuffle, args.limit)
    print(f"{len(rows)} rows from {args.split}", file=sys.stderr)

    out = {}
    with torch.no_grad():
        for mrows, ids, mask, last in micro_batches(rows, tok, args.max_tokens, pad_id):
            z = head(forward_hidden(model, ids, mask, last, device, dtype))
            for r, zz in zip(mrows, z.tolist()):
                out[r["id"]] = zz
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "logits": out[r["id"]]}) + "\n")
    print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
