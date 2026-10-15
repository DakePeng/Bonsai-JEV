#!/bin/bash
# Decision Index 0.3 public suites (text + vision) for Bonsai-JEV on a rented 2x RTX PRO 6000 box (AutoDL).
# The last GPU serves the vision suite (kit's `decide` engine, 8 workers); every other GPU serves text (8 shards each).
#   bash rental/di.sh setup     build the fork's llama-server, fetch the release weights from ModelScope, venv + DI kit
#   bash rental/di.sh start     decision services (text on GPUs 0..N-2, vision on the last)
#   bash rental/di.sh cord      rebuild the CORD vision set on Linux (macOS Pillow re-encodes 2 images differently)
#   bash rental/di.sh sample    100 text rows + 100 vision rows end to end (pipeline check, ~5 min)
#   bash rental/di.sh full      full public suites in the background -> ~/bonsai/di/runs/{text,vision}
#   bash rental/di.sh score     merge the text shards, score both
# Expects (uploaded from the Mac): ~/bonsai/jev (this repo), ~/bonsai/llama.cpp-src.tgz (fork @ e8fd209, no build/.git),
# ~/bonsai/di/decision-index (kit @ f5a6e4f), ~/bonsai/di/suite-0.3 and ~/bonsai/di/vision (rebuilt suites, private).
set -eu
# BONSAI_ROOT=/root/autodl-fs/bonsai keeps everything on AutoDL's region-wide file storage, so the run can move to
# whichever instance in the region has free GPUs (prepare on a no-GPU instance, run on another). Weights are not kept
# there: each GPU instance pulls them from ModelScope (16 streams, ~7 min), since copying 7 GB into the network storage
# fills a no-GPU instance's small memory with page cache.
R=${BONSAI_ROOT:-$HOME/bonsai}; mkdir -p $R; DI=$R/di; W=${WEIGHTS_DIR:-$HOME/autodl-tmp/bonsai-weights}; J=$R/jev
MIRROR=${PIP_MIRROR:-https://repo.huaweicloud.com/repository/pypi/simple}
L=${LOCAL_ROOT:-$HOME/autodl-tmp/bonsai-local}; mkdir -p $L   # venv + build on the instance's own disk (network storage is slow for many small files)
export PIP_NO_CACHE_DIR=1 LLAMA_BIN=$L/llama.cpp/build/bin VENV=$L/venv-di PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
log() { echo "[$(date +%H:%M:%S)] $*"; }
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 2 ] || NG=2   # GPUs on this box (>= 2: text + vision)

case "${1:-}" in
setup)
  mkdir -p $W
  for f in Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf adapter.gguf decision.json; do   # copy prepared on storage
    [ -e $W/$f ] || [ ! -f $R/weights/$f ] || cp $R/weights/$f $W/$f
  done
  for f in Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf adapter.gguf; do
    [ -e $W/$f ] || bash $J/rental/msdl.sh dakepeng/Bonsai-JEV $f $W/$f 16
  done
  (cd $W && sha256sum Ternary-Bonsai-2-27B-PQ2_0.gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf adapter.gguf)
  [ -f $W/decision.json ] || curl -sL -o $W/decision.json "https://www.modelscope.cn/api/v1/models/dakepeng/Bonsai-JEV/repo?Revision=master&FilePath=decision.json"
  [ -d $VENV ] || python3 -m venv $VENV
  $VENV/bin/pip install -q -i $MIRROR -U pip
  $VENV/bin/pip install -q -i $MIRROR fastapi "uvicorn[standard]" requests pydantic pillow httpx pyarrow numpy
  $VENV/bin/pip install -q -i $MIRROR -e "$DI/decision-index[vision]"
  if [ ! -x $LLAMA_BIN/llama-server ]; then
    mkdir -p $L/llama.cpp && tar xzf $R/llama.cpp-src.tgz -C $L/llama.cpp
    cmake -S $L/llama.cpp -B $L/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release > $L/cmake.log 2>&1
    cmake --build $L/llama.cpp/build --target llama-server -j "$(nproc)" > $L/build.log 2>&1
  fi
  python3 -c "import json; d=json.load(open('$W/decision.json')); print('temperatures', d['per_kind_temperature'])"
  $LLAMA_BIN/llama-server --version 2>&1 | tail -1
  log "setup done"
  ;;
start)
  # last GPU serves the vision suite (port 8001); every other GPU serves text (ports 8010, 8011, ...)
  cd $J
  for g in $(seq 0 $((NG - 2))); do
    CUDA_VISIBLE_DEVICES=$g LPORT=$((8090 + g)) PORT=$((8010 + g)) CTX=262144 NP=8 bash run_decide.sh start $W | tail -1   # 8 slots x 32k
  done
  CUDA_VISIBLE_DEVICES=$((NG - 1)) LPORT=8081 PORT=8001 CTX=131072 NP=8 bash run_decide.sh start $W | tail -1              # 8 slots x 16k
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  ;;
cord)
  # CORD's two oversized images re-encode differently with macOS Pillow; rebuild that one set here (Linux) and verify
  cd $DI; . $VENV/bin/activate; [ -f /etc/network_turbo ] && . /etc/network_turbo
  python -m decision_index suite rebuild --vision --sets full_cord --work $DI/vision-work > $DI/cord.log 2>&1
  rm -rf $DI/vision/full/full_cord && cp -r $DI/vision-work/vision/full/full_cord $DI/vision/full/
  python -c "from decision_index.vision import rows; rows.write_rows('$DI/vision')"
  python -m decision_index vision verify --dir $DI/vision | tail -3
  ;;
sample)
  cd $J; . $VENV/bin/activate; mkdir -p $DI/runs
  python -m decision_index suite sample --dir $DI/suite-0.3 --n 100 --out $DI/sample-100.jsonl.gz
  python -m decision_index run --engine di_engine:BonsaiJEV --option base_url=http://127.0.0.1:8010 \
    --rows $DI/sample-100.jsonl.gz --out $DI/runs/text-sample --fresh | tail -c 600; echo
  python -m decision_index score --results $DI/runs/text-sample/results.jsonl --suite-dir $DI/suite-0.3 --edition 0.3 | head -c 600; echo
  python -m decision_index vision run --dir $DI/vision --engine decide --option base_url=http://127.0.0.1:8001 \
    --workers 8 --limit 100 --out $DI/runs/vision-sample --fresh | tail -c 600; echo
  grep -c '"status": "ok"' $DI/runs/text-sample/results.jsonl $DI/runs/vision-sample/results.jsonl || true
  ;;
full)
  # 8 runner shards per text GPU, shard i -> text server i mod (NG - 1)
  cd $J; . $VENV/bin/activate; mkdir -p $DI/runs/text $DI/runs/vision
  NT=$((NG - 1)); NS=$((8 * NT))
  [ -d $DI/shards ] || python di_shard.py --suite-dir $DI/suite-0.3 --n $NS --out $DI/shards
  for i in $(seq 0 $((NS - 1))); do
    setsid nohup python -m decision_index run --engine di_engine:BonsaiJEV --option base_url=http://127.0.0.1:$((8010 + i % NT)) \
      --rows $DI/shards/rows-$i.jsonl.gz --out $DI/runs/text/$i > $DI/runs/text/$i.log 2>&1 < /dev/null &
  done
  setsid nohup python -m decision_index vision pipeline --dir $DI/vision --engine decide \
    --option base_url=http://127.0.0.1:8001 --workers 8 --out $DI/runs/vision > $DI/runs/vision.log 2>&1 < /dev/null &
  log "started $NS text shards on $NT GPU(s) + vision"
  ;;
score)
  cd $J; . $VENV/bin/activate
  cat $DI/runs/text/[0-9]*/results.jsonl > $DI/runs/text/results.jsonl
  python -m decision_index score --results $DI/runs/text/results.jsonl --suite-dir $DI/suite-0.3 --edition 0.3 --engine bonsai-jev
  python -m decision_index vision score --dir $DI/vision --results $DI/runs/vision/results.jsonl
  ;;
*) echo "usage: bash rental/di.sh setup|start|sample|full|score"; exit 1 ;;
esac
