"""test_export.py - end-to-end check of export_gguf_lora without a trained 27B: a synthetic adapter.

1. Takes a peft checkpoint layout (the 2-layer CPU smoke works), fills layers 0-1 LoRA A/B and the head
   with random values large enough to matter, saves it as a checkpoint.
2. Exports it to GGUF with export_gguf_lora.py.
3. Runs the fork's dump-tensors with --lora, dumping l_out-0, l_out-1, result_norm, result_output.
4. PyTorch: layer 0 and 1 with the same LoRA (peft on the single decoder layer) on the fork's inputs,
   compared with the fork's LoRA'd l_out-N. Head: result_output[verbalizer ids] must equal
   (W_init + delta) . result_norm.
Passes when every relative diff is at kernel-noise level (~1e-3). Exercises A/B orientation, the runtime
scale, the value-head permutation (layer 0 is linear attention) and the lm_head-LoRA head.

    python test_export.py --template /tmp/smoke/final --work /tmp/export_test
"""
import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import torch

from align_layer import load_dump, report
from bonsai_torch import Bonsai, build_layer, decision_head_rows, text_config
from zeroshot_text import VERBALIZER_IDS

PROMPT = "The quick brown fox jumps over the lazy dog. Paris is the capital of France, and Berlin is the capital of"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="/tmp/smoke/final")
    ap.add_argument("--work", default="/tmp/export_test")
    ap.add_argument("--gguf", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--cfg", default=os.path.expanduser("~/bonsai/models/qwen3.8-27b-cfg"))
    ap.add_argument("--bin", default=os.path.expanduser("~/bonsai/llama.cpp/build/bin"))
    args = ap.parse_args()
    from safetensors.torch import load_file, save_file
    torch.manual_seed(0)

    # 1. synthetic checkpoint
    ck = os.path.join(args.work, "ckpt")
    shutil.rmtree(args.work, ignore_errors=True)
    shutil.copytree(args.template, ck)
    sd = load_file(os.path.join(ck, "adapter_model.safetensors"))
    sd = {k: (torch.randn_like(v.float()) * 0.02) for k, v in sd.items()}
    save_file(sd, os.path.join(ck, "adapter_model.safetensors"))
    hd = load_file(os.path.join(ck, "head.safetensors"))
    b, cfg = Bonsai(args.gguf), text_config(args.cfg)
    init = decision_head_rows(b, VERBALIZER_IDS)
    delta = torch.randn_like(init) * 0.01
    save_file({"weight": (init + delta).contiguous(), "bias": torch.zeros(24)}, os.path.join(ck, "head.safetensors"))
    layers = sorted({int(k.split(".")[3]) for k in sd})
    print("synthetic LoRA on layers", layers, "+ head delta")

    # 2. export
    gg = os.path.join(args.work, "adapter.gguf")
    subprocess.run(["python3", "export_gguf_lora.py", "--ckpt", ck, "--out", gg], check=True)

    # 3. fork dump with the adapter
    dump = os.path.join(args.work, "dump")
    os.makedirs(dump)
    env = dict(os.environ, DUMP_DIR=dump, DUMP_LAYERS=",".join(map(str, layers)))
    subprocess.run([os.path.join(args.bin, "llama-dump-tensors"), "-m", args.gguf, "--lora", gg, "-ngl", "99", "-c", "512",
                    "-p", PROMPT, "--log-disable"], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = sum(1 for l in open(os.path.join(dump, "tokens.txt")) if l.strip())
    E = cfg.hidden_size

    # 4a. layers: PyTorch layer + the same LoRA via peft, on the fork's own input
    from peft import LoraConfig, inject_adapter_in_model
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    pcfg = json.load(open(os.path.join(ck, "adapter_config.json")))
    dev, dtype = "cuda", torch.float32
    pos = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    rotary = M.Qwen3_5TextRotaryEmbedding(cfg, device=dev)
    dummy = torch.zeros(1, n, E, device=dev, dtype=dtype)
    pe = rotary(dummy, pos[1:])
    mk = dict(config=cfg, inputs_embeds=dummy, attention_mask=None, past_key_values=None, position_ids=pos[0])
    masks = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
    worst = 0.0
    for i in layers:
        layer = build_layer(b, cfg, i, dtype=dtype, device=dev)
        inject_adapter_in_model(LoraConfig(r=pcfg["r"], lora_alpha=pcfg["lora_alpha"], lora_dropout=0.0,
                                           target_modules=pcfg["target_modules"]), layer)
        lsd = {k.replace(f"base_model.model.layers.{i}.", "").replace("lora_A.weight", "lora_A.default.weight")
                .replace("lora_B.weight", "lora_B.default.weight"): v for k, v in sd.items() if f".layers.{i}." in k}
        missing, unexpected = layer.load_state_dict(lsd, strict=False)
        assert not unexpected and not [m for m in missing if "lora" in m], (unexpected, [m for m in missing if "lora" in m])
        layer.to(dev)
        x = load_dump(dump, "model.input_embed", E) if i == 0 else load_dump(dump, f"l_out-{i - 1}", E)
        with torch.no_grad():
            y = layer(x[None].to(dev), position_embeddings=pe, attention_mask=masks[cfg.layer_types[i]], position_ids=pos[0])
        worst = max(worst, report(f"l_out-{i} +lora", y[0], load_dump(dump, f"l_out-{i}", E)))
        del layer
        torch.cuda.empty_cache()

    # 4b. head: fork logits at verbalizer ids == (init + delta) . result_norm
    rn = load_dump(dump, "result_norm", E)
    logits = load_dump(dump, "result_output", cfg.vocab_size)[:, VERBALIZER_IDS]
    expect = rn @ (init + delta).T
    worst = max(worst, report("head logits", expect, logits))
    base = rn @ init.T
    print(f"head delta contribution: mean|delta.h|={(expect - base).abs().mean():.4f} (must be >> diff above)")
    print(f"worst rel diff {worst:.2e} ->", "PASS" if worst < 5e-3 else "FAIL")


if __name__ == "__main__":
    main()
