#!/bin/bash
# Runs on the Mac: every 2 h copy ckpt/run1/{last,best} from the rental box to rental/stage/ckpt/run1/ (newest only).
# The remote side swaps checkpoints in atomically, so a copy that races a swap just fails and is retried shortly.
# Start: setsid nohup bash rental/sync_ckpt.sh > rental/stage/sync.log 2>&1 &     Stop: pkill -f sync_ckpt.sh
cd "$(dirname "$0")/stage" || exit 1
mkdir -p ckpt/run1
while true; do
  for tag in last best; do
    for try in 1 2 3; do
      rm -rf "ckpt/run1/$tag.new"; mkdir -p "ckpt/run1/$tag.new"
      if ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 autodl "cd /root/Bonsai-JEV/ckpt/run1 && test -d $tag && tar cf - -C $tag ." \
           | tar xf - -C "ckpt/run1/$tag.new" && [ -f "ckpt/run1/$tag.new/state.pt" ]; then
        rm -rf "ckpt/run1/$tag"; mv "ckpt/run1/$tag.new" "ckpt/run1/$tag"
        echo "$(date '+%m-%d %H:%M') $tag synced: step $(python3 -c "import json;print(json.load(open('ckpt/run1/$tag/head.json'))['step'])" 2>/dev/null) $(du -sh ckpt/run1/$tag | cut -f1)"
        break
      fi
      rm -rf "ckpt/run1/$tag.new"; echo "$(date '+%m-%d %H:%M') $tag not available yet (try $try)"; sleep 120
    done
  done
  ssh -o ConnectTimeout=30 autodl 'cd /root/Bonsai-JEV && tail -1 ckpt/run1/log.csv; ls ckpt/run1/eval_*.json 2>/dev/null | tail -1' 2>/dev/null | sed "s/^/$(date '+%m-%d %H:%M') remote: /"
  scp -q -o ConnectTimeout=30 "autodl:/root/Bonsai-JEV/ckpt/run1/eval_*.json" "autodl:/root/Bonsai-JEV/ckpt/run1/log.csv" ckpt/run1/ 2>/dev/null
  sleep 7200
done
