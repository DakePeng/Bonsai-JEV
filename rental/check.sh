#!/bin/bash
# Steps 11 + 8 on the rental box (one GPU, ~40 min). Run: bash rental/check.sh 2>&1 | tee results/check.log
set -u
cd ~/bonsai/jev && . ~/bonsai/venv/bin/activate && mkdir -p results
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "== step 11: full 27B, LoRA, checkpointing, 24k-token micro-batches, 3 steps"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU:-0}; sleep 5; done ) > results/check_vram.txt 2>/dev/null & SP=$!
python3 train.py --out ckpt/kernel-check --steps 3 --rows-per-step 128 --max-tokens 24000 --eval-rows 64 --eval-every 3 --save-every 3 --train-limit 20000 2>&1 | grep -v "transformers\]" | tail -12
kill $SP 2>/dev/null
echo "peak VRAM MiB: $(sort -n results/check_vram.txt | tail -1)"
echo "== step 8: full-model PyTorch vs the fork's dumped final hidden state"
python3 - <<'EOF'
import torch, numpy as np, os
from bonsai_torch import Bonsai, build_model, text_config
b = Bonsai(os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
cfg = text_config(os.path.expanduser("~/bonsai/models/qwen3.8-27b-cfg"))
m = build_model(b, cfg, dtype=torch.bfloat16, device="cuda")
toks = torch.tensor([[int(l) for l in open(os.path.expanduser("~/bonsai/test/dump2/tokens.txt")) if l.strip()]], device="cuda")
with torch.no_grad():
    h = m(input_ids=toks).last_hidden_state[0, -1].float().cpu()
ref = torch.from_numpy(np.fromfile(os.path.expanduser("~/bonsai/test/dump2/result_norm.f32"), dtype=np.float32))
d = (h - ref).abs()
print(f"result_norm: max|diff|={d.max():.4e} rel={d.max()/ref.abs().max():.2e} mean|diff|={d.mean():.3e} mean|ref|={ref.abs().mean():.3e}")
print("PASS" if d.max() / ref.abs().max() < 1e-2 else "FAIL (expected <= 1e-2 for a 64-layer BF16 forward)")
EOF
