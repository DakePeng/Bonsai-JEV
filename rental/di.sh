#!/bin/bash
# Decision Index 0.3 public suites (text + vision) for Bonsai-JEV on a rented 2x RTX PRO 6000 box (AutoDL).
# GPU 0 serves the text suite (8 runner shards), GPU 1 the vision suite (kit's `decide` engine, 8 workers).
#   bash rental/di.sh setup     build the fork's llama-server, fetch the release weights from ModelScope, venv + DI kit
#   bash rental/di.sh start     both decision services
#   bash rental/di.sh sample    100 text rows + 100 vision rows end to end (pipeline check, ~5 min)
#   bash rental/di.sh full      full public suites in the background -> ~/bonsai/di/runs/{text,vision}
#   bash rental/di.sh score     merge the text shards, score both
# Expects (uploaded from the Mac): ~/bonsai/jev (this repo), ~/bonsai/llama.cpp-src.tgz (fork @ e8fd209, no build/.git),
# ~/bonsai/di/decision-index (kit @ f5a6e4f), ~/bonsai/di/suite-0.3 and ~/bonsai/di/vision (rebuilt suites, private).
set -eu
DATA=${DATA_DISK:-$HOME/autodl-tmp}; [ -d "$DATA" ] || DATA=$HOME
[ -e ~/bonsai ] || { mkdir -p $DATA/bonsai && ln -s $DATA/bonsai ~/bonsai; }
R=~/bonsai; DI=$R/di; W=$R/weights; J=$R/jev
MIRROR=${PIP_MIRROR:-https://repo.huaweicloud.com/repository/pypi/simple}
export PIP_NO_CACHE_DIR=1 LLAMA_BIN=$R/llama.cpp/build/bin VENV=$R/venv-di
log() { echo "[$(date +%H:%M:%S)] $*"; }

case "${1:-}" in
setup)
  [ -d $VENV ] || python3 -m venv $VENV
  $VENV/bin/pip install -q -i $MIRROR -U pip
  $VENV/bin/pip install -q -i $MIRROR fastapi "uvicorn[standard]" requests pydantic pillow httpx pyarrow numpy
  $VENV/bin/pip install -q -i $MIRROR -e "$DI/decision-index[vision]"
  if [ ! -x $LLAMA_BIN/llama-server ]; then
    mkdir -p $R/llama.cpp && tar xzf $R/llama.cpp-src.tgz -C $R/llama.cpp
    cmake -S $R/llama.cpp -B $R/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release > $R/cmake.log 2>&1
    cmake --build $R/llama.cpp/build --target llama-server -j "$(nproc)" > $R/build.log 2>&1
  fi
  mkdir -p $W
  for f in Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf adapter.gguf; do
    bash $J/rental/msdl.sh dakepeng/Bonsai-JEV $f $W/$f 16
  done
  [ -f $W/decision.json ] || curl -sL -o $W/decision.json "https://www.modelscope.cn/api/v1/models/dakepeng/Bonsai-JEV/repo?Revision=master&FilePath=decision.json"
  python3 -c "import json; d=json.load(open('$W/decision.json')); print('temperatures', d['per_kind_temperature'])"
  $LLAMA_BIN/llama-server --version 2>&1 | tail -1
  log "setup done"
  ;;
start)
  cd $J
  CUDA_VISIBLE_DEVICES=0 LPORT=8080 PORT=8000 CTX=262144 NP=8 bash run_decide.sh start $W | tail -1   # 8 slots x 32k (text)
  CUDA_VISIBLE_DEVICES=1 LPORT=8081 PORT=8001 CTX=131072 NP=8 bash run_decide.sh start $W | tail -1   # 8 slots x 16k (vision)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  ;;
sample)
  cd $J; . $VENV/bin/activate; mkdir -p $DI/runs
  python -m decision_index suite sample --dir $DI/suite-0.3 --n 100 --out $DI/sample-100.jsonl.gz
  python -m decision_index run --engine di_engine:BonsaiJEV --option base_url=http://127.0.0.1:8000 \
    --rows $DI/sample-100.jsonl.gz --out $DI/runs/text-sample --fresh | tail -c 600; echo
  python -m decision_index score --results $DI/runs/text-sample/results.jsonl --suite-dir $DI/suite-0.3 --edition 0.3 | head -c 600; echo
  python -m decision_index vision run --dir $DI/vision --engine decide --option base_url=http://127.0.0.1:8001 \
    --workers 8 --limit 100 --out $DI/runs/vision-sample --fresh | tail -c 600; echo
  grep -c '"status": "ok"' $DI/runs/text-sample/results.jsonl $DI/runs/vision-sample/results.jsonl || true
  ;;
full)
  cd $J; . $VENV/bin/activate; mkdir -p $DI/runs/text $DI/runs/vision
  [ -d $DI/shards ] || python di_shard.py --suite-dir $DI/suite-0.3 --n 8 --out $DI/shards
  for i in 0 1 2 3 4 5 6 7; do
    setsid nohup python -m decision_index run --engine di_engine:BonsaiJEV --option base_url=http://127.0.0.1:8000 \
      --rows $DI/shards/rows-$i.jsonl.gz --out $DI/runs/text/$i > $DI/runs/text/$i.log 2>&1 < /dev/null &
  done
  setsid nohup python -m decision_index vision pipeline --dir $DI/vision --engine decide \
    --option base_url=http://127.0.0.1:8001 --workers 8 --out $DI/runs/vision > $DI/runs/vision.log 2>&1 < /dev/null &
  log "started 8 text shards + vision"
  ;;
score)
  cd $J; . $VENV/bin/activate
  cat $DI/runs/text/[0-7]/results.jsonl > $DI/runs/text/results.jsonl
  python -m decision_index score --results $DI/runs/text/results.jsonl --suite-dir $DI/suite-0.3 --edition 0.3 --engine bonsai-jev
  python -m decision_index vision score --dir $DI/vision --results $DI/runs/vision/results.jsonl
  ;;
*) echo "usage: bash rental/di.sh setup|start|sample|full|score"; exit 1 ;;
esac
