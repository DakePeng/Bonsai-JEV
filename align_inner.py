"""align_inner.py - compare every projection inside one layer against the fork's dumps.

Dump with DUMP_LAYERS=<i> so the per-layer intermediates exist, then:
    python align_inner.py DUMP_DIR <layer> [--dtype float32]
Pairs the fork's cb() names with forward hooks on the HF submodules and reports each.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from align_layer import load_dump, report
from bonsai_torch import Bonsai, build_layer, hadamard_blocks, dequant_pq2_0, text_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("layer", type=int)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gguf", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--cfg", default=os.path.expanduser("~/bonsai/models/qwen3.8-27b-cfg"))
    args = ap.parse_args()
    dtype, dev, i = getattr(torch, args.dtype), "cuda", args.layer

    b, cfg = Bonsai(args.gguf), text_config(args.cfg)
    E = cfg.hidden_size
    tokens = torch.tensor([int(l) for l in open(os.path.join(args.dump, "tokens.txt")) if l.strip()])
    n = len(tokens)
    kind = cfg.layer_types[i]
    print(f"layer {i} ({kind}), {n} tokens, {args.dtype}")

    # layer input = fork's own previous output
    x = load_dump(args.dump, "model.input_embed", E) if i == 0 else load_dump(args.dump, f"l_out-{i - 1}", E)
    layer = build_layer(b, cfg, i, dtype=dtype, device=dev)

    cap = {}
    def hook(name, want_input=False):
        def f(m, inp, out):
            cap[name] = (inp[0] if want_input else out).detach()
        return f
    subs = {
        "input_layernorm": False, "post_attention_layernorm": False, "post_attention_layernorm.in": True,
        "mlp.gate_proj": False, "mlp.up_proj": False, "mlp.down_proj": False, "mlp.down_proj.in": True,
    }
    if kind == "linear_attention":
        subs.update({"linear_attn.in_proj_qkv": False, "linear_attn.in_proj_z": False, "linear_attn.in_proj_a": False,
                     "linear_attn.in_proj_b": False, "linear_attn.out_proj": False, "linear_attn.out_proj.in": True,
                     "linear_attn.norm": False, "linear_attn.norm.in": True})
    else:
        subs.update({"self_attn.q_proj": False, "self_attn.k_proj": False, "self_attn.v_proj": False,
                     "self_attn.o_proj": False, "self_attn.o_proj.in": True})
    for name, want_in in subs.items():
        layer.get_submodule(name[:-3] if want_in else name).register_forward_hook(hook(name, want_in))

    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    pos = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    rotary = M.Qwen3_5TextRotaryEmbedding(cfg, device=dev)
    dummy = torch.zeros(1, n, E, device=dev, dtype=dtype)
    pe = rotary(dummy, pos[1:])
    mk = dict(config=cfg, inputs_embeds=dummy, attention_mask=None, past_key_values=None, position_ids=pos[0])
    mask = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}[kind]
    with torch.no_grad():
        y = layer(x[None].to(dev, dtype), position_embeddings=pe, attention_mask=mask, position_ids=pos[0])

    def cmp(fork_name, hf_name, width=None, fn=lambda t: t):
        got = cap[hf_name]
        got = got.reshape(-1, got.shape[-1]) if got.dim() == 3 else got
        ref = load_dump(args.dump, fork_name, width or got.shape[-1])
        if ref is None:
            print(f"{fork_name:24s} (no dump)")
            return
        report(fork_name, fn(got), ref)

    s = f"-{i}"
    cmp("attn_norm" + s, "input_layernorm")
    if kind == "linear_attention":
        cmp("linear_attn_qkv_mixed" + s, "linear_attn.in_proj_qkv")
        cmp("z" + s, "linear_attn.in_proj_z")
        cmp("alpha" + s, "linear_attn.in_proj_a")
        cmp("beta" + s, "linear_attn.in_proj_b")
        # conv + silu recomputed from HF weights on the HF qkv (causal depthwise, kernel 4, zero left pad)
        la = layer.linear_attn
        qkv = cap["linear_attn.in_proj_qkv"].reshape(1, n, -1).transpose(1, 2)              # (1, C, n)
        w = la.conv1d.weight                                                                   # (C, 1, k)
        conv = F.conv1d(F.pad(qkv, (w.shape[-1] - 1, 0)), w, groups=w.shape[0])
        ref = load_dump(args.dump, "conv_output_silu" + s, w.shape[0])
        report("conv_output_silu" + s, F.silu(conv)[0].T, ref)
        cmp("final_output" + s, "linear_attn.out_proj.in")
        cmp("linear_attn_out" + s, "linear_attn.out_proj")
    else:
        cmp("Qcur_full" + s, "self_attn.q_proj")
        cmp("Kcur" + s, "self_attn.k_proj")
        cmp("Vcur" + s, "self_attn.v_proj")
        cmp("attn_gated" + s, "self_attn.o_proj.in")
        cmp("attn_output" + s, "self_attn.o_proj")
    cmp("attn_residual" + s, "post_attention_layernorm.in")
    cmp("attn_post_norm" + s, "post_attention_layernorm")
    cmp("ffn_gate" + s, "mlp.gate_proj")
    cmp("ffn_up" + s, "mlp.up_proj")
    cmp("ffn_swiglu" + s, "mlp.down_proj.in")
    cmp("ffn_out" + s, "mlp.down_proj")
    report("l_out" + s, y[0], load_dump(args.dump, "l_out" + s, E))


if __name__ == "__main__":
    main()
