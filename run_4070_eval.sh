#!/bin/bash
# Deployment-path evaluation of trained checkpoints on the 4070 (PrismML fork + adapter.gguf).
#   bash run_4070_eval.sh <ckpt dir> [<ckpt dir> ...]     each dir: adapter_model.safetensors, adapter_config.json,
#                                                         head.safetensors, head.json (state.pt not needed)
# Per checkpoint: export -> calibration logits -> temperatures -> test_set_30k logits -> scores (raw / calibrated)
# -> choice rows with options reversed -> order agreement. Once: the zero-shot base's order agreement, for contrast.
set -u
cd ~/bonsai/jev && . ~/bonsai/venv/bin/activate && mkdir -p results
export GGUF_PY=~/bonsai/llama.cpp/gguf-py
log() { echo "[$(date +%H:%M)] $*"; }

if [ ! -f results/base_choice_rev.logits.jsonl ]; then
  log "zero-shot base: original-order logits with ids, and choice rows reversed"
  python3 - <<'EOF'
import json, os
from zeroshot_text import load_split
rows = load_split(os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"), "test_set_30k", True)
outs = [json.loads(l) for l in open("results/zs_test30k.logits.jsonl")]
assert len(rows) == len(outs)
with open("results/base_test30k.logits.jsonl", "w") as f:
    for r, o in zip(rows, outs):
        f.write(json.dumps({"id": r["id"], "logits": o["logits"]} if "logits" in o else {"id": r["id"], "error": o.get("error")}) + "\n")
EOF
  python3 fork_eval.py --kinds choice --permute reverse --out results/base_choice_rev.logits.jsonl | tail -1
fi
echo "== zero-shot base, choice rows, original vs reversed options"
python3 fork_eval.py --compare results/base_test30k.logits.jsonl results/base_choice_rev.logits.jsonl

for CK in "$@"; do
  T=$(basename "$CK")_$(python3 -c "import json;print(json.load(open('$CK/head.json'))['step'])")
  log "== $T: export"
  python3 export_gguf_lora.py --ckpt "$CK" --out "$CK/adapter.gguf" 2>&1 | tail -1
  log "$T: calibration logits"
  python3 fork_eval.py --adapter-dir "$CK" --split calibration --out results/${T}_calib.logits.jsonl | tail -1
  python3 calibrate.py --logits results/${T}_calib.logits.jsonl --split calibration --out "$CK/calibration.json" 2>&1 | grep -v Warning
  log "$T: test_set_30k logits"
  python3 fork_eval.py --adapter-dir "$CK" --split test_set_30k --out results/${T}_test30k.logits.jsonl | tail -1
  echo "== $T test_set_30k, no temperature"
  python3 eval_text.py --logits results/${T}_test30k.logits.jsonl --split test_set_30k --out results/${T}_test30k_raw 2>&1 | grep -E "^(all|noul|score|choice|src)"
  echo "== $T test_set_30k, with temperature"
  python3 eval_text.py --logits results/${T}_test30k.logits.jsonl --split test_set_30k --calibration "$CK/calibration.json" --out results/${T}_test30k_cal 2>&1 | grep -E "^(all|noul|score|choice|src)"
  log "$T: choice rows with options reversed"
  python3 fork_eval.py --adapter-dir "$CK" --kinds choice --permute reverse --out results/${T}_choice_rev.logits.jsonl | tail -1
  echo "== $T choice rows, original vs reversed options"
  python3 fork_eval.py --compare results/${T}_test30k.logits.jsonl results/${T}_choice_rev.logits.jsonl
  # re-export with the fitted temperatures inside decision.json (what the server reads)
  python3 export_gguf_lora.py --ckpt "$CK" --out "$CK/adapter.gguf" --calibration "$CK/calibration.json" 2>&1 | tail -1
done
log "all done"
