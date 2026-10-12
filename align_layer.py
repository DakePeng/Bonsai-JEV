"""align_layer.py - per-layer alignment of bonsai_torch against the fork's dumped tensors.

1. On the fork:  DUMP_DIR=/path ./llama-dump-tensors -m model.gguf -ngl 99 -c 512 -p "..."
2. Here:         python align_layer.py /path [layer ...] [--dtype float32]

For each layer i, feed the fork's own input (inp_embd or l_out-(i-1)) through the PyTorch
layer and compare with l_out-i. Also checks the embedding lookup and the final norm/logits
when those dumps are present. Loads one layer at a time, so it fits a 12 GB GPU in fp32.
"""
import argparse
import os

import numpy as np
import torch

from bonsai_torch import (Bonsai, HadamardLinear, build_layer, dequant_pq2_0, hadamard_blocks,
                          hf_value, text_config)


def load_dump(d, name, width):
    p = os.path.join(d, name + ".f32")
    if not os.path.exists(p):
        return None
    return torch.from_numpy(np.fromfile(p, dtype=np.float32).reshape(-1, width))


def report(tag, got, ref):
    got, ref = got.float().cpu(), ref.float().cpu()
    diff = (got - ref).abs()
    scale = ref.abs().max().item()
    print(f"{tag:14s} max|diff|={diff.max().item():.4e}  rel={diff.max().item() / scale:.2e}  "
          f"mean|diff|={diff.mean().item():.3e}  mean|ref|={ref.abs().mean().item():.3e}")
    return diff.max().item() / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("layers", nargs="*", type=int)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gguf", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--cfg", default=os.path.expanduser("~/bonsai/models/qwen3.8-27b-cfg"))
    args = ap.parse_args()
    dtype, dev = getattr(torch, args.dtype), "cuda"

    b, cfg = Bonsai(args.gguf), text_config(args.cfg)
    E = cfg.hidden_size
    tokens = torch.tensor([int(l) for l in open(os.path.join(args.dump, "tokens.txt")) if l.strip()])
    n = len(tokens)
    print(f"{n} tokens, dtype {args.dtype}")

    # --- embedding: dequantize only the looked-up rows, then inverse-after-lookup s * H(z)
    inp = load_dump(args.dump, "model.input_embed", E)
    if inp is None:
        inp = load_dump(args.dump, "inp_embd", E)
    raw = np.asarray(b.t["token_embd.weight"].data)[tokens.numpy()]
    z = dequant_pq2_0(raw, n, E)
    emb = hadamard_blocks(z, b.H) * b.signs[E]
    if inp is None:
        print("inp_embd not dumped; using the PyTorch embedding as layer-0 input")
        inp = emb
    else:
        report("inp_embd", emb, inp)

    # --- position ids / rope / masks, exactly as Qwen3_5TextModel.forward builds them
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    pos = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    text_pos, rope_pos = pos[0], pos[1:]
    rotary = M.Qwen3_5TextRotaryEmbedding(cfg, device=dev)
    dummy = torch.zeros(1, n, E, device=dev, dtype=dtype)
    position_embeddings = rotary(dummy, rope_pos)
    mk = dict(config=cfg, inputs_embeds=dummy, attention_mask=None, past_key_values=None, position_ids=text_pos)
    masks = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}

    layers = args.layers or list(range(cfg.num_hidden_layers))
    worst = 0.0
    for i in layers:
        x = inp if i == 0 else load_dump(args.dump, f"l_out-{i - 1}", E)
        ref = load_dump(args.dump, f"l_out-{i}", E)
        layer = build_layer(b, cfg, i, dtype=dtype, device=dev)
        with torch.no_grad():
            y = layer(x[None].to(dev, dtype), position_embeddings=position_embeddings,
                      attention_mask=masks[cfg.layer_types[i]], position_ids=text_pos)
        # the fork computes the last layer only for output tokens (ne=[E,1]); compare the matching tail
        worst = max(worst, report(f"l_out-{i} {cfg.layer_types[i][:4]}", y[0, -ref.shape[0]:], ref))
        del layer, y
        torch.cuda.empty_cache()

    # --- final norm + lm_head on the last layer's output (when dumped)
    rn = load_dump(args.dump, "result_norm", E)
    last = load_dump(args.dump, f"l_out-{cfg.num_hidden_layers - 1}", E)
    if rn is not None and last is not None:
        norm = M.Qwen3_5RMSNorm(E, eps=cfg.rms_norm_eps).to(dtype)
        norm.weight.data = hf_value("output_norm.weight", b.tensor("output_norm.weight")).to(dtype)
        report("result_norm", norm(last[-rn.shape[0]:].to(dtype)), rn)   # fork emits only output tokens
        logits_ref = load_dump(args.dump, "result_output", cfg.vocab_size)
        if logits_ref is not None:
            head = HadamardLinear(E, cfg.vocab_size, b.signs[E], b.H, dtype=dtype)
            head.weight.data = b.tensor("output.weight").to(dtype)
            with torch.no_grad():
                logits = head.to(dev)(rn.to(dev, dtype))
            report("result_output", logits, logits_ref)
            print("argmax agree:", (logits.argmax(-1).cpu() == logits_ref.argmax(-1)).float().mean().item())
    print(f"worst layer rel diff: {worst:.2e}")


if __name__ == "__main__":
    main()
