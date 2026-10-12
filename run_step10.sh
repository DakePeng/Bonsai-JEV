#!/bin/bash
# Step 10 chain on the 4070 box, detached: text baseline (resumable) -> VL-RewardBench -> 10b (JEV-9B) if prepared.
# Start with:  setsid nohup bash ~/bonsai/jev/run_step10.sh > ~/bonsai/jev/results/step10.log 2>&1 &
set -u
cd ~/bonsai/jev
. ~/bonsai/venv/bin/activate
mkdir -p results
echo "[$(date +%T)] text baseline"
python3 zeroshot_text.py --split test_set_30k --skip-placeholder --resume --out results/zs_test30k > results/zs_test30k.log 2>&1
grep -E "^(all|noul|score|choice|src)|skipped" results/zs_test30k.log
echo "[$(date +%T)] VL-RewardBench"
python3 zeroshot_vlrb.py --out results/zs_vlrb > results/zs_vlrb.log 2>&1
grep -E "acc=|scored" results/zs_vlrb.log
if [ -x ./run_10b.sh ]; then
  echo "[$(date +%T)] 10b JEV-9B"
  bash ./run_10b.sh
fi
echo "[$(date +%T)] step 10 chain done"
