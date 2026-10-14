#!/bin/bash
# Steps 19 + 20 on the rental box right after training: 24-slot logits of the best checkpoint (by validation KL) on
# calibration / test_set_30k / ood, both GPUs in parallel, then temperatures, scores, and the GGUF export.
#   bash rental/post.sh [ckpt dir, default ckpt/run1/best] 2>&1 | tee results/post.log
set -u
cd ~/bonsai/jev && . ~/bonsai/venv/bin/activate && mkdir -p results
export GGUF_PY=~/bonsai/llama.cpp/gguf-py
CK=${1:-ckpt/run1/best}
echo "[$(date +%T)] post-training on $CK (step $(python3 -c "import json;print(json.load(open('$CK/head.json'))['step'])"))"

# GPU 0: test_set_30k (27.7k rows). GPU 1: calibration then ood (13.8k + 13.1k rows).
CUDA_VISIBLE_DEVICES=0 python3 infer_logits.py --ckpt $CK --split test_set_30k --out results/run1_test30k.logits.jsonl > results/post_gpu0.log 2>&1 &
P0=$!
( CUDA_VISIBLE_DEVICES=1 python3 infer_logits.py --ckpt $CK --split calibration --out results/run1_calib.logits.jsonl &&
  CUDA_VISIBLE_DEVICES=1 python3 infer_logits.py --ckpt $CK --split ood --out results/run1_ood.logits.jsonl ) > results/post_gpu1.log 2>&1 &
P1=$!
wait $P1 || { echo "GPU1 inference failed"; tail -20 results/post_gpu1.log; }
echo "[$(date +%T)] calibration + ood logits done"
python3 calibrate.py --logits results/run1_calib.logits.jsonl --split calibration --out $CK/calibration.json 2>&1 | grep -v Warning
wait $P0 || { echo "GPU0 inference failed"; tail -20 results/post_gpu0.log; }
echo "[$(date +%T)] test_set_30k logits done"

for split in test_set_30k ood; do
  f=results/run1_${split/test_set_30k/test30k}.logits.jsonl
  echo "== $split, no temperature"
  python3 eval_text.py --logits $f --split $split --out results/run1_${split}_raw 2>&1 | grep -E "^(all|noul|score|choice|src)"
  echo "== $split, with temperature"
  python3 eval_text.py --logits $f --split $split --calibration $CK/calibration.json --out results/run1_${split}_cal 2>&1 | grep -E "^(all|noul|score|choice|src)"
done

echo "== export"
python3 export_gguf_lora.py --ckpt $CK --out $CK/adapter.gguf --calibration $CK/calibration.json 2>&1 | tail -1
ls -la $CK
echo "[$(date +%T)] post done"
