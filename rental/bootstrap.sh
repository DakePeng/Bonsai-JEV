#!/bin/bash
# One-shot environment for a rented GPU box (Ubuntu, CUDA >= 12.8, Blackwell/Hopper). Idempotent.
# Usage:  bash rental/bootstrap.sh            (run from the repo root on the box; ~30-40 min, mostly downloads)
# Layout mirrors the 4070 box so every script's defaults work: ~/bonsai/{jev,venv,venv-vllm,models,data}
set -eu
# AutoDL: home (~) is the 30 GB system disk, the data disk is ~/autodl-tmp. Everything big goes there;
# ~/bonsai is a symlink so all scripts' defaults keep working. pip runs without a cache (it would fill the system disk).
DATA=${DATA_DISK:-$HOME/autodl-tmp}; [ -d "$DATA" ] || DATA=$HOME
[ -e ~/bonsai ] || { mkdir -p $DATA/bonsai && ln -s $DATA/bonsai ~/bonsai; }
R=~/bonsai; mkdir -p $R/models $R/data
export PIP_NO_CACHE_DIR=1
MIRROR=${PIP_MIRROR:-https://repo.huaweicloud.com/repository/pypi/simple}   # fastest from AutoDL west-D (measured 2026-10-13)
[ -e $R/jev ] || ln -s "$(cd "$(dirname "$0")/.." && pwd)" $R/jev

echo "== disks"; df -h $HOME $DATA | tail -2
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; nvcc --version 2>/dev/null | tail -1 || true
free -g | head -2; df -h $R | tail -1

echo "== main venv"
# Reuse the image's CUDA torch when it has one (AutoDL images ship torch+cu13x matching the driver): the venv
# inherits system site-packages, so only the small packages are downloaded. Otherwise install torch from the mirror.
TORCH_PKG="torch"
if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "system torch $(python3 -c 'import torch; print(torch.__version__)') has CUDA; venv will inherit it"
  [ -d $R/venv ] || python3 -m venv --system-site-packages $R/venv
  TORCH_PKG=""
else
  [ -d $R/venv ] || python3 -m venv $R/venv
fi
. $R/venv/bin/activate
pip install -q -i $MIRROR -U pip
pip install -q -i $MIRROR $TORCH_PKG transformers peft safetensors accelerate sentencepiece modelscope pandas pyarrow pillow numpy pyyaml requests fastapi "uvicorn[standard]"
# fast GDN + causal-conv1d kernels (Triton); training is several times slower on the torch fallback
pip install -q -i $MIRROR flash-linear-attention causal-conv1d 2>&1 | tail -1 || echo "WARN: fla/causal-conv1d install failed; torch fallback will be used"
python3 - <<'EOF'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("transformers", transformers.__version__, "peft", peft.__version__)
try:
    import fla; print("flash-linear-attention OK")
except Exception as e:
    print("fla missing:", e)
EOF

echo "== models & data (ModelScope)"
M=$R/models
# big files via 16 parallel range requests (ModelScope gives ~1.6 MB/s per stream from AutoDL, ~18 MB/s with 16)
bash $R/jev/rental/msdl.sh prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf $M/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf 16
bash $R/jev/rental/msdl.sh prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf $M/bonsai2-27b/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf 8
[ -f $M/bonsai2-27b/KNOWN_ISSUES.md ] || modelscope download --model prism-ml/Ternary-Bonsai-2-27B-gguf --include "README.md" "KNOWN_ISSUES.md" --local_dir $M/bonsai2-27b
[ -f $M/qwen3.8-27b-cfg/tokenizer.json ] || modelscope download --model Qwen/Qwen3.8-27B --include "config.json" "generation_config.json" "tokenizer*" "vocab.json" "merges.txt" --local_dir $M/qwen3.8-27b-cfg
[ -f $R/data/jev-distill-corpus-v3/train.jsonl ] || modelscope download --dataset SargeDev/jev-distill-corpus-v3 --local_dir $R/data/jev-distill-corpus-v3
[ -f $R/data/vl-rewardbench/data/test-00000-of-00001.parquet ] || modelscope download --dataset MMInstruction/VL-RewardBench --include "data/*" "README.md" --local_dir $R/data/vl-rewardbench
# 10b (JEV-9B baseline): only the vision adapter (~100 MB) + the multimodal Qwen3.5-9B base (19 GB); the 18 GB
# JEV-9B text shards are not needed. Delete $M/qwen3.5-9b and $R/venv-vllm after 10b on a 50 GB data disk.
if [ "${WITH_10B:-0}" = "1" ]; then   # 10b skipped by default (decision 2026-10-13)
  [ -f $M/jev-9b/vl/adapter_vllm/adapter_model.safetensors ] || modelscope download --model autotrust/JEV-9B --include "vl/*" "calibration.json" "judge_config.json" "README.md" --local_dir $M/jev-9b
  [ -f $M/qwen3.5-9b/config.json ] || modelscope download --model Qwen/Qwen3.5-9B --local_dir $M/qwen3.5-9b
  [ -d $R/venv-vllm ] || { python3 -m venv $R/venv-vllm && $R/venv-vllm/bin/pip install -q -i $MIRROR -U pip vllm; }
  [ -d $R/JEV-27B-DEMO ] || git clone -q --depth 1 https://github.com/yuhai-china/JEV-27B-DEMO.git $R/JEV-27B-DEMO || echo "WARN: demo repo clone failed (needed for jev_client in 10b)"
fi
# gguf-py for bonsai_torch (reads the GGUF); only the python package is needed, not the fork build
[ -d $R/llama.cpp/gguf-py ] || { mkdir -p $R/llama.cpp && pip download -q -i $MIRROR --no-deps -d /tmp/ggufpy gguf 2>/dev/null; pip install -q -i $MIRROR gguf; python3 -c "import gguf, os; print('gguf-py', gguf.__file__)"; }
export GGUF_PY=$(python3 -c "import gguf, os; print(os.path.dirname(os.path.dirname(gguf.__file__)))")

echo "== self-check"
cd $R/jev && GGUF_PY=$GGUF_PY python3 bonsai_torch.py
echo "bootstrap done. export GGUF_PY=$GGUF_PY in every shell (or add to ~/.bashrc)"
echo "export GGUF_PY=$GGUF_PY" >> ~/.bashrc
