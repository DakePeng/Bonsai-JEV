#!/bin/bash
# Release checks on the 4070, run after run_4070_eval.sh. Picks the checkpoint with the lowest calibrated KL on the
# Jev-labelled test_set_30k rows, then: VL-RewardBench, JevBench-231, latency, System 2, concurrency, PTQ1_0
# (correctness + speed), and an 8 GB-card simulation.      bash run_release_checks.sh > results/release_checks.log 2>&1
set -u
cd ~/bonsai/jev && . ~/bonsai/venv/bin/activate && mkdir -p results
export GGUF_PY=~/bonsai/llama.cpp/gguf-py
M=~/bonsai/models/bonsai2-27b; BIN=~/bonsai/llama.cpp/build/bin
log() { echo; echo "[$(date +%H:%M)] ===== $*"; }
stop_all() { bash run_decide.sh stop > /dev/null 2>&1; }

# ---- 0. pick the checkpoint
read FINAL TAG < <(python3 - <<'EOF'
import json, glob, os
best = None
for p in glob.glob("results/*_test30k_cal.metrics.json"):
    tag = os.path.basename(p)[:-len("_test30k_cal.metrics.json")]
    if not tag.split("_")[0] in ("best", "last"):
        continue
    kl = json.load(open(p))["src:yuri_v3"]["kl"]
    if best is None or kl < best[0]:
        best = (kl, tag)
kl, tag = best
print(os.path.expanduser(f"~/bonsai/ckpt-run1/{tag.split('_')[0]}"), tag)
EOF
)
log "selected $TAG ($FINAL)"
for t in results/*_test30k_cal.metrics.json; do python3 -c "import json;m=json.load(open('$t'));print('$t', 'Jev-labelled KL', round(m['src:yuri_v3']['kl'],5))"; done
rm -rf ~/bonsai/ckpt-run1/final; cp -r "$FINAL" ~/bonsai/ckpt-run1/final; echo "$TAG" > ~/bonsai/ckpt-run1/final/SELECTED

# ---- 1. VL-RewardBench (deployment path, adapter + head bias + choice temperature)
log "VL-RewardBench"
stop_all
python3 zeroshot_vlrb.py --adapter-dir "$FINAL" --out results/final_vlrb 2>&1 | grep -E "acc=|single order|temperature|scored"
echo "zero-shot reference: overall 74.5, order-agree 50.9"

# ---- 2. JevBench-231 + latency through the service
log "JevBench-231 and latency"
CTX=16384 NP=1 bash run_decide.sh start "$FINAL" | tail -1
JEV_URL=http://localhost:8000 python3 jevbench_eval.py --out results/final_jevbench | tail -10
python3 latency_bench.py --out results/final_latency
stop_all

# ---- 3. System 2 with the final adapter
log "System 2"
bash run_system2.sh "$FINAL/adapter.gguf" | tail -4

# ---- 4. Concurrency with the final adapter (needs results/s2_* from step 3)
log "concurrency, 8 slots"
CTX=32768 NP=8 bash run_decide.sh start "$FINAL" | tail -1
python3 test_concurrency.py --n 40 --workers 40 | tail -3
stop_all

# ---- 5. PTQ1_0: same adapter on the denser packing
log "PTQ1_0"
if [ -f $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf ]; then
  ls -la $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf
  $BIN/llama-bench -m $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -p 512 -n 128 -r 2 2>&1 | grep -E "^\|.*(pp512|tg128)"
  python3 fork_eval.py --adapter-dir "$FINAL" --limit 3000 --model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --out results/final_ptq1_test30k.logits.jsonl | tail -1
  echo "PTQ1_0 vs PQ2_0, same adapter, first 3000 rows of test_set_30k:"
  python3 fork_eval.py --compare results/${TAG}_test30k.logits.jsonl results/final_ptq1_test30k.logits.jsonl
else
  echo "PTQ1_0 file missing, skipped"
fi

# ---- 6. 8 GB card simulation: hold VRAM so that ~7.5 GiB stay free, then serve PTQ1_0 + fp16 adapter
log "8 GB simulation"
python3 export_gguf_lora.py --ckpt "$FINAL" --out "$FINAL/adapter_f16.gguf" --dtype f16 --calibration "$FINAL/calibration.json" | tail -1
FREE=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
HOLD=$(( FREE - 7680 ))
echo "free ${FREE} MiB, holding ${HOLD} MiB to leave 7.5 GiB"
setsid nohup python3 -c "
import torch, time
x = torch.empty($HOLD * 1024 * 1024, dtype=torch.uint8, device='cuda')
open('/tmp/hold.ready', 'w').write('ok'); time.sleep(36000)" > /tmp/hold.log 2>&1 < /dev/null &
HP=$!
for i in $(seq 60); do [ -f /tmp/hold.ready ] && break; sleep 1; done
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
for mode in none cpu gpu; do
  echo "--- PTQ1_0 + fp16 adapter, 4k context, mmproj=$mode"
  stop_all
  if BASE_GGUF=Ternary-Bonsai-2-27B-PTQ1_0.gguf ADAPTER_FILE=adapter_f16.gguf MMPROJ=$mode CTX=4096 NP=1 \
       bash run_decide.sh start "$FINAL" | tail -1; then
    ( while true; do /usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > /tmp/vram_$mode.txt 2>/dev/null & SP=$!
    if [ $mode = none ]; then python3 latency_bench.py --no-images --out results/final_8gb_$mode; else python3 latency_bench.py --out results/final_8gb_$mode; fi
    kill $SP; echo "peak used incl. the ${HOLD} MiB hold: $(sort -n /tmp/vram_$mode.txt | tail -1) MiB"
    grep -iE "out of memory|failed to allocate|cudaMalloc" results/llama_server.log | head -3
  else
    echo "server failed to start (likely out of memory):"; grep -iE "memory|alloc|error" results/llama_server.log | tail -5
  fi
done
stop_all; kill $HP 2>/dev/null; rm -f /tmp/hold.ready
log "release checks done"
