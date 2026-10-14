#!/bin/bash
# Start (or stop) the decision service on the 4070: the fork's llama-server + decide_server.py, both detached.
#   bash run_decide.sh start [ADAPTER_DIR]    ADAPTER_DIR holds adapter.gguf + decision.json (omit = zero-shot base);
#                                             the release repo also holds the base GGUF + mmproj and is used as the model dir
#   bash run_decide.sh stop
# env: BASE_GGUF (default Ternary-Bonsai-2-27B-PQ2_0.gguf; PTQ1_0 for tight memory), ADAPTER_FILE (default adapter.gguf),
#      MMPROJ=gpu|cpu|none (default gpu; cpu runs the vision encoder on the CPU to save ~0.9 GB VRAM), CTX, NP,
#      LLAMA_BIN (dir with the fork's llama-server; default ~/bonsai/llama.cpp/build/bin), VENV (optional, activated if set)
set -u
[ -n "${2:-}" ] && [ -d "$2" ] && set -- "$1" "$(cd "$2" && pwd)"   # absolute adapter dir before the cd below
cd "$(dirname "$0")"
M=${MODEL_DIR:-~/bonsai/models/bonsai2-27b}
BIN=${LLAMA_BIN:-~/bonsai/llama.cpp/build/bin}
LPORT=${LPORT:-8080}
PORT=${PORT:-8000}
mkdir -p results

stop() {
  pkill -f "[l]lama-server .*--port $LPORT"; pkill -f "[u]vicorn decide_server"
  # wait until both ports are free, else the next start fails to bind (llama-server takes seconds to release VRAM)
  for i in $(seq 60); do pgrep -f "[l]lama-server .*--port $LPORT|[u]vicorn decide_server" > /dev/null || break; sleep 1; done
  pkill -9 -f "[l]lama-server .*--port $LPORT"; pkill -9 -f "[u]vicorn decide_server"; sleep 1; echo "stopped"
}

case "${1:-start}" in
  stop) stop; exit 0 ;;
  start)
    [ -x "$BIN/llama-server" ] || { echo "llama-server not found in $BIN; set LLAMA_BIN to the fork's build/bin"; exit 1; }
    stop > /dev/null
    ADAPTER=${2:-}
    # all-in-one release repo: the base GGUF and mmproj sit next to adapter.gguf
    BASE=${BASE_GGUF:-Ternary-Bonsai-2-27B-PQ2_0.gguf}
    [ -n "$ADAPTER" ] && [ -f "$ADAPTER/$BASE" ] && M=$ADAPTER
    LORA=""; DJ=""
    if [ -n "$ADAPTER" ]; then LORA="--lora $ADAPTER/${ADAPTER_FILE:-adapter.gguf} --lora-init-without-apply"; DJ="$ADAPTER/decision.json"; fi
    case "${MMPROJ:-gpu}" in
      gpu)  MM="--mmproj $M/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf" ;;
      cpu)  MM="--mmproj $M/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --no-mmproj-offload" ;;
      none) MM="" ;;
    esac
    # -np 2: two slots; the server routes a request to the slot with the longest common prefix.
    # --cache-ram / --cache-reuse: keep evicted prompts in RAM and reuse partial prefixes (KNOWN_ISSUES.md).
    setsid nohup $BIN/llama-server -m $M/$BASE $MM \
      $LORA -ngl 99 -c ${CTX:-8192} -np ${NP:-2} --cache-ram 16384 --cache-reuse 256 --port $LPORT --host 127.0.0.1 \
      > results/llama_server.log 2>&1 < /dev/null &
    for i in $(seq 1 60); do curl -sf http://127.0.0.1:$LPORT/health > /dev/null && break; sleep 2; done
    curl -sf http://127.0.0.1:$LPORT/health > /dev/null || { echo "llama-server failed, see results/llama_server.log"; exit 1; }
    # adapter OFF globally (System 2 = untouched base); decide_server switches it on per decision request.
    # --lora-init-without-apply alone does not take effect in this fork's server, so set it explicitly.
    [ -n "$ADAPTER" ] && curl -sf -X POST http://127.0.0.1:$LPORT/lora-adapters -H "content-type: application/json" \
      -d '[{"id": 0, "scale": 0}]' > /dev/null && echo "adapter loaded, global scale 0: $(curl -s http://127.0.0.1:$LPORT/lora-adapters)"
    VENV=${VENV:-~/bonsai/venv}; [ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"
    LLAMA_URL=http://127.0.0.1:$LPORT DECISION_JSON=$DJ setsid nohup uvicorn decide_server:app --host 0.0.0.0 --port $PORT \
      > results/decide_server.log 2>&1 < /dev/null &
    for i in $(seq 1 30); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 1; done
    curl -s http://127.0.0.1:$PORT/health; echo
    echo "decide service on :$PORT (llama-server :$LPORT, adapter: ${ADAPTER:-none})"
    ;;
esac
