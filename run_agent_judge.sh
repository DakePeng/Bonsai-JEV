#!/bin/bash
# Agent-judge benchmarks on the 4070 after run_release_checks.sh (uses the checkpoint it selected).
#   bash run_agent_judge.sh > results/agent_judge.log 2>&1      data: python agent_judge_eval.py prep (elsewhere), copied in
set -u
cd ~/bonsai/jev && . ~/bonsai/venv/bin/activate && mkdir -p results
FINAL=~/bonsai/ckpt-run1/final
echo "[$(date +%H:%M)] agent judge on $(cat $FINAL/SELECTED)"
CTX=16384 NP=1 bash run_decide.sh start "$FINAL" | tail -1
JEV_URL=http://localhost:8000 python3 agent_judge_eval.py run --out results/final_agentjudge
bash run_decide.sh stop > /dev/null 2>&1
echo "[$(date +%H:%M)] agent judge done"
