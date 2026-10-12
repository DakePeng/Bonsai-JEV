#!/bin/bash
# Step 10b on the 4070 (WSL2, 12 GB): JEV-9B FP8 via vLLM + the demo's serve_decide.py, then text and
# VL-RewardBench evals through the demo's own client. Called by run_step10.sh after the Bonsai baselines.
# Pattern from JEV-27B-DEMO/common/serve_jev27b_mm.sh; JEV-9B's language weights are bit-identical to
# Qwen3.5-9B, so the multimodal base + vl/adapter_vllm is the JEV-9B-VL System 1.
set -u
cd ~/bonsai/jev
mkdir -p results
BASE=~/bonsai/models/qwen3.5-9b
ADAPTER=~/bonsai/models/jev-9b/vl/adapter_vllm
PORT=${PORT:-8000}
LOG=results/jev9b_server.log

if [ ! -f "$BASE/config.json" ] || [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "10b: weights not ready, skipping"; exit 0
fi
. ~/bonsai/venv-vllm/bin/activate || { echo "10b: vllm venv missing"; exit 0; }
python3 -c "import vllm" 2>/dev/null || { echo "10b: vllm not installed"; exit 0; }

echo "[$(date +%T)] starting vLLM (fp8)"
setsid nohup python3 ~/bonsai/JEV-27B-DEMO/common/serve_decide.py --model "$BASE" --served-model-name autotrust/JEV-9B \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision="$ADAPTER" \
  --logprobs-mode processed_logprobs --max-model-len 4096 --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 2}' --max-num-seqs 8 --trust-request-chat-template \
  --quantization fp8 --gpu-memory-utilization 0.92 --enforce-eager --port "$PORT" > "$LOG" 2>&1 < /dev/null &
SRV=$!
for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/health" > /dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "10b: server died, see $LOG"; tail -30 "$LOG"; exit 0; }
  sleep 10
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "10b: server not healthy after 30 min"; kill $SRV; exit 0; }
echo "[$(date +%T)] server up"

export JEV_URL="http://localhost:$PORT" JEV_BACKEND=decide
. ~/bonsai/venv/bin/activate   # eval scripts run in the main venv (pandas etc.); client only needs requests
pip show requests > /dev/null 2>&1 || pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple requests
echo "[$(date +%T)] 10b text"
python3 jev9b_eval.py text --out results/jev9b_text > results/jev9b_text.log 2>&1
grep -E "^(all|noul|score|choice|src)" results/jev9b_text.log
echo "[$(date +%T)] 10b VL-RewardBench"
python3 jev9b_eval.py vlrb --out results/jev9b_vlrb > results/jev9b_vlrb.log 2>&1
grep -E "acc=" results/jev9b_vlrb.log
kill $SRV 2>/dev/null; sleep 5; pkill -f "[s]erve_decide.py" 2>/dev/null
echo "[$(date +%T)] 10b done"
