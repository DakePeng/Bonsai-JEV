#!/bin/bash
# System 2 check on the 4070: base server vs the same server with the decision adapter loaded and switched off.
#   bash run_system2.sh <adapter.gguf>        -> results/s2_{base,off,on}.json + PASS/FAIL
set -u
ADAPTER=${1:?adapter.gguf}
cd ~/bonsai/jev && . ~/bonsai/venv/bin/activate && mkdir -p results
M=~/bonsai/models/bonsai2-27b; B=~/bonsai/llama.cpp/build/bin
start() {
  setsid nohup $B/llama-server -m $M/Ternary-Bonsai-2-27B-PQ2_0.gguf "$@" -ngl 99 -c 4096 -np 1 --port 8080 --host 127.0.0.1 \
    > results/s2_server.log 2>&1 < /dev/null &
  SRV=$!
  for i in $(seq 90); do curl -sf 127.0.0.1:8080/health > /dev/null && return 0; sleep 2; done
  echo "server failed to start"; tail -20 results/s2_server.log; exit 1
}
stop() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 2; }
start;                        python3 test_system2.py --mode base; stop
start --lora "$ADAPTER";      curl -s 127.0.0.1:8080/lora-adapters; echo
python3 test_system2.py --mode off; python3 test_system2.py --mode on; stop
python3 test_system2.py --compare
