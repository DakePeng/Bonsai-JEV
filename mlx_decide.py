"""mlx_decide.py - Bonsai-JEV on Apple Silicon: Prism's MLX 2-bit pack + our LoRA and decision head, no conversion.

Prism's pack (prism-ml/Ternary-Bonsai-2-27B-mlx-2bit, schema 2) is mlx-vlm's qwen3_5 Model whose language-model
linear layers are `Packed` modules: Hadamard-rotate the input, then a 2-bit quantized matmul (lossless transcode of PQ2_0), with HF's grouped
GDN layout. Our peft LoRA was trained on the raw (un-rotated) layer input in that same layout, so it wraps each
Packed module as base(x) + (x A^T) B^T * alpha/r with no math, and the decision head (fp32 24 x 5120 + bias) reads
the final normed hidden state directly. Temperatures come from decision.json.

    python mlx_decide.py --pack bonsai2-27b-mlx --adapter Bonsai-JEV-weights --validate val.jsonl
"""
import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
from mlx import nn

LETTERS = "ABCDEFGHIJKLMNOP"
RANGES = {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}


class LoRA(nn.Module):
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base, self.a, self.b, self.scale = base, a.T.astype(mx.float16), b.T.astype(mx.float16), scale

    def __call__(self, x):
        return self.base(x) + ((x.astype(mx.float16) @ self.a) @ self.b * self.scale).astype(x.dtype)


def load(pack, adapter):
    sys.path.insert(0, os.path.join(pack, "runtime"))
    from vision_artifact import load_vl_model
    vl, proc, _ = load_vl_model(pack)
    model = vl.language_model          # .model.layers[...] holds the Packed text layers; vl.vision_tower stays as shipped
    cfg = json.load(open(os.path.join(adapter, "peft", "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    w = mx.load(os.path.join(adapter, "peft", "adapter_model.safetensors"))
    n = 0
    for k in w:
        if not k.endswith(".lora_A.weight"):
            continue
        path = k[len("base_model.model."):-len(".lora_A.weight")].split(".")   # layers.0.linear_attn.in_proj_qkv
        parent = model.model
        for p in path[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        setattr(parent, path[-1], LoRA(getattr(parent, path[-1]), w[k], w[k.replace("lora_A", "lora_B")], scale))
        n += 1
    head = mx.load(os.path.join(adapter, "head.safetensors"))
    dec = json.load(open(os.path.join(adapter, "decision.json")))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(pack, "tokenizer.json"))
    mx.eval(model.parameters())
    model.vl, model.proc = vl, proc
    return model, head, dec, tok, n


MARKER = "<__media__>"     # where an image goes in a prompt (the fork's marker); becomes vision_start + patches + vision_end
IMG = "<|vision_start|><|image_pad|><|vision_end|>"


def head_logits_mm(model, head, prompt, images):
    """prompt with one MARKER per image; images: file paths. Native resolution through the pack's Qwen2-VL processor."""
    from PIL import Image
    inp = model.proc(text=[prompt.replace(MARKER, IMG)], images=[Image.open(p).convert("RGB") for p in images],
                     padding=False, return_tensors="np")
    ids = mx.array(inp["input_ids"])
    emb = model.vl.get_input_embeddings(ids, mx.array(inp["pixel_values"]), image_grid_thw=mx.array(inp["image_grid_thw"]))
    out = model(ids, inputs_embeds=emb.inputs_embeds, cache=model.make_cache(), return_hidden=True, skip_logits=True)
    h = out.hidden_states[-1][0, -1].astype(mx.float32)
    z = head["weight"] @ h + head["bias"]
    mx.eval(z)
    return z.tolist(), ids.shape[1]


def as_text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def render(kind, state, question, options):
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    return f"[kind] {kind}\n[state] {as_text(state)}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


def head_logits(model, head, tok, prompt):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    out = model(mx.array([ids]), cache=model.make_cache(), return_hidden=True, skip_logits=True)
    h = out.hidden_states[-1][0, -1].astype(mx.float32)          # final RMSNorm output at [decision]:
    z = head["weight"] @ h + head["bias"]
    mx.eval(z)
    return z.tolist(), len(ids)


def decide(model, head, dec, tok, kind, state, question, options=None):
    options = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}.get(kind, options)
    z, n = head_logits(model, head, tok, render(kind, state, question, options))
    lo, _ = RANGES[kind]
    T = dec.get("per_kind_temperature", {}).get(kind, 1.0)
    z = [v / T for v in z[lo:lo + len(options)]]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e], n


def kl(p, q):
    return sum(a * math.log(a / max(b, 1e-12)) for a, b in zip(p, q) if a > 0)


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit download dir")
    ap.add_argument("--adapter", required=True, help="dir with peft/, head.safetensors, decision.json")
    ap.add_argument("--validate", help="jsonl rows {kind, state, question, options, target, fork_logits} exported from the fork")
    ap.add_argument("--validate-vlrb", help="jsonl rows {prompt, image, fork_ab} (VL-RewardBench, both orders per pair)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    model, head, dec, tok, n = load(a.pack, a.adapter)
    print(f"loaded pack + {n} LoRA modules in {time.time() - t0:.1f}s; peak memory {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
    if a.validate_vlrb:
        rows = [json.loads(l) for l in open(a.validate_vlrb)]
        rows = rows[:a.limit] if a.limit else rows
        T = dec["per_kind_temperature"]["choice"]
        b8, b9 = dec["head_bias"][8], dec["head_bias"][9]
        pa_m, pa_f, secs, toks = [], [], [], []
        for r in rows:
            t = time.time()
            z, ntok = head_logits_mm(model, head, r["prompt"], [os.path.join(os.path.dirname(a.validate_vlrb), r["image"])])
            secs.append(time.time() - t)
            toks.append(ntok)
            pa_m.append(1 / (1 + math.exp((z[9] - z[8]) / T)))
            za, zb = r["fork_ab"]
            pa_f.append(1 / (1 + math.exp(((zb + b9) - (za + b8)) / T)))
        n = len(rows)
        agree = sum((x > 0.5) == (y > 0.5) for x, y in zip(pa_m, pa_f))
        s = sorted(secs)
        print(f"{n} image prompts | P(A) MLX vs fork: mean |dp| {sum(abs(x - y) for x, y in zip(pa_m, pa_f)) / n:.4f}, "
              f"max |dp| {max(abs(x - y) for x, y in zip(pa_m, pa_f)):.4f}, same verdict {100 * agree / n:.1f}%")
        print(f"latency p50 {s[n // 2]:.2f}s p95 {s[int(0.95 * (n - 1))]:.2f}s, median prompt tokens {sorted(toks)[n // 2]}, "
              f"peak memory {mx.get_peak_memory() / 2**30:.2f} GiB")
        return
    if not a.validate:
        print(decide(model, head, dec, tok, "noul", "The robot holds a red cube above an empty bin.", "Should it release the cube now?"))
        return
    rows = [json.loads(l) for l in open(a.validate)]
    rows = rows[:a.limit] if a.limit else rows
    T = dec.get("per_kind_temperature", {})
    agg = {"kl_target_mlx": 0.0, "kl_target_fork": 0.0, "kl_fork_mlx": 0.0, "agree": 0, "secs": [], "toks": []}
    for i, r in enumerate(rows):
        t = time.time()
        p, ntok = decide(model, head, dec, tok, r["kind"], r["state"], r["question"], r.get("options"))
        agg["secs"].append(time.time() - t)
        agg["toks"].append(ntok)
        lo, _ = RANGES[r["kind"]]
        q = softmax([v / T.get(r["kind"], 1.0) for v in r["fork_logits"][lo:lo + len(p)]])
        agg["kl_target_mlx"] += kl(r["target"], p)
        agg["kl_target_fork"] += kl(r["target"], q)
        agg["kl_fork_mlx"] += kl(q, p)
        agg["agree"] += max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__)
        if (i + 1) % 50 == 0:
            print(f"{i + 1} rows", flush=True)
    n = len(rows)
    s = sorted(agg["secs"])
    print(f"{n} rows | KL(target||MLX) {agg['kl_target_mlx'] / n:.4f}  KL(target||fork) {agg['kl_target_fork'] / n:.4f}  "
          f"KL(fork||MLX) {agg['kl_fork_mlx'] / n:.5f}  top-1 agreement {100 * agg['agree'] / n:.1f}%")
    print(f"latency p50 {s[n // 2]:.2f}s p95 {s[int(0.95 * (n - 1))]:.2f}s, median prompt tokens {sorted(agg['toks'])[n // 2]}, "
          f"peak memory {mx.get_peak_memory() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
