#!/bin/bash
# Step 14 on the rental box (one GPU, ~1 h): head-only sanity, then the full recipe on 1% of the data,
# eval on validation, export to GGUF. Run: bash rental/smoke.sh 2>&1 | tee results/smoke_all.log
set -u
cd ~/bonsai/jev && . ~/bonsai/venv/bin/activate && mkdir -p results
export CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "== 14a: head only (LoRA lr 0), 20 steps"
python3 train.py --out ckpt/smoke-head --steps 20 --train-limit 5000 --eval-rows 512 --eval-every 20 --save-every 20 --lr-lora 0 2>&1 | grep -E "^\[.*(step|eval|trainable|done)" | tail -8
echo "== 14b: full recipe, 1% data, 40 steps"
python3 train.py --out ckpt/smoke --steps 40 --train-limit 5000 --eval-rows 512 --eval-every 20 --save-every 20 2>&1 | grep -E "^\[.*(step|eval|trainable|done)" | tail -10
echo "== eval on 1024 validation rows (zero-shot on this split ~0.45)"
python3 infer_logits.py --ckpt ckpt/smoke/final --split validation --limit 1024 --out results/smoke_val.logits.jsonl 2>&1 | grep -v "transformers\]" | tail -1
python3 eval_text.py --logits results/smoke_val.logits.jsonl --split validation --limit 1024 --out results/smoke_val 2>&1 | grep -E "^(validation|all|noul|score|choice)"
echo "== export"
python3 export_gguf_lora.py --ckpt ckpt/smoke/final --out ckpt/smoke/final/adapter.gguf 2>&1 | tail -1
ls -l ckpt/smoke/final/
