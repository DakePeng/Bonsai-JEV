"""export_gguf_lora.py - peft checkpoint (LoRA + decision head) -> one GGUF LoRA for the PrismML fork.

Format (src/llama-adapter.cpp): general.type=adapter, adapter.type=lora, adapter.lora.alpha; per base tensor
`<gguf name>.lora_a` (numpy (r, n_in)) and `.lora_b` (numpy (n_out, r)); runtime scale = alpha / r.
The fork adds B.A.x on the raw (un-rotated) input, which is how the LoRA was trained.

Two layout fixes on the way out (inverse of what bonsai_torch does on the way in):
  * gdn_v_grouped: per-value-head rows (v part of attn_qkv.lora_b, attn_gate.lora_b) and the
    per-value-head columns of ssm_out.lora_a go back from HF's grouped order to the fork's tiled order.
  * the decision head becomes a rank-24 LoRA on output.weight: A = (W_head - W_init) * 24/alpha (so the
    runtime scale alpha/24 cancels), B = one-hot rows at the verbalizer token ids. The head bias cannot
    live in a LoRA; it goes to decision.json for the serving wrapper, like JEV's decision_head.json.

    python export_gguf_lora.py --ckpt ckpt/run1/final --out ckpt/run1/final/adapter.gguf [--calibration cal.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from bonsai_torch import GGUF_PY, Bonsai, _LAYER_MAP, decision_head_rows, tiled_to_grouped
from zeroshot_text import RANGES, VERBALIZER_IDS

sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402

HF2GGUF = {v.replace(".weight", ""): k.replace(".weight", "") for k, v in _LAYER_MAP.items() if v.endswith(".weight")}


def gguf_name(peft_key: str):
    """'base_model.model.layers.3.self_attn.q_proj.lora_A.weight' -> ('blk.3.attn_q.weight', 'lora_a')"""
    k = peft_key.replace("base_model.model.", "")
    if k.startswith("model."):
        k = k[len("model."):]
    parts = k.split(".")
    assert parts[0] == "layers", peft_key
    li, ab = parts[1], parts[-2]
    hf = ".".join(parts[2:-2])
    return f"blk.{li}.{HF2GGUF[hf]}.weight", {"lora_A": "lora_a", "lora_B": "lora_b"}[ab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gguf", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16"])
    args = ap.parse_args()
    from safetensors.torch import load_file

    b = Bonsai(args.gguf)
    cfg = json.load(open(os.path.join(args.ckpt, "adapter_config.json")))
    alpha, r = float(cfg["lora_alpha"]), int(cfg["r"])
    sd = load_file(os.path.join(args.ckpt, "adapter_model.safetensors"))
    inv = torch.argsort(tiled_to_grouped(b.nk, b.nv))        # grouped (HF) -> tiled (fork): fork = hf[inv]
    kd = b.nk * b.hd

    def to_fork(name, ab, w):
        if not b.gdn_v_grouped:
            return w
        if name.endswith("attn_qkv.weight") and ab == "lora_b":          # rows: [q | k | v], v per head
            v = w[2 * kd:].reshape(b.nv, b.hd, -1)[inv].reshape(-1, w.shape[1])
            return torch.cat([w[:2 * kd], v], 0)
        if name.endswith("attn_gate.weight") and ab == "lora_b":         # rows per v head
            return w.reshape(b.nv, b.hd, -1)[inv].reshape(w.shape)
        if name.endswith("ssm_out.weight") and ab == "lora_a":           # columns per v head
            return w.reshape(w.shape[0], b.nv, b.hd)[:, inv].reshape(w.shape)
        return w

    w = gguf.GGUFWriter(args.out, arch="qwen35")
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", alpha)
    npdt = np.float32 if args.dtype == "f32" else np.float16
    n = 0
    for key in sorted(sd):
        if "lora_" not in key:
            continue
        name, ab = gguf_name(key)
        t = to_fork(name, ab, sd[key].float()).contiguous().numpy().astype(npdt)
        w.add_tensor(f"{name}.{ab}", t)
        n += 1
    meta = {"verbalizer_ids": VERBALIZER_IDS, "ranges": RANGES, "lora_alpha": alpha, "lora_r": r, "head": not args.no_head}
    if not args.no_head:
        hd = load_file(os.path.join(args.ckpt, "head.safetensors"))
        init = decision_head_rows(b, VERBALIZER_IDS)
        delta = (hd["weight"].float() - init) * (len(VERBALIZER_IDS) / alpha)   # runtime scale alpha/24 -> 1
        n_vocab = int(b.t["output.weight"].shape[1])
        onehot = np.zeros((n_vocab, len(VERBALIZER_IDS)), dtype=npdt)
        for i, tid in enumerate(VERBALIZER_IDS):
            onehot[tid, i] = 1.0
        w.add_tensor("output.weight.lora_a", delta.contiguous().numpy().astype(npdt))
        w.add_tensor("output.weight.lora_b", onehot)
        meta["head_bias"] = hd["bias"].float().tolist()
        n += 2
    if args.calibration:
        meta["per_kind_temperature"] = json.load(open(args.calibration))["per_kind"]
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    json.dump(meta, open(os.path.join(os.path.dirname(args.out) or ".", "decision.json"), "w"), indent=1)
    print(f"wrote {args.out}: {n} tensors, alpha {alpha}, r {r}; decision.json alongside", file=sys.stderr)


if __name__ == "__main__":
    main()
