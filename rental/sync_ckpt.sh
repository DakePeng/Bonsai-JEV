#!/bin/bash
# Runs on the Mac: every 10 min, pull ckpt/run1/{last,best} from the rental box when its step changed (newest only),
# plus log.csv and eval_*.json. Remote saves are atomic, so a copy racing a swap just fails and is retried next round.
# Start: nohup bash rental/sync_ckpt.sh > rental/stage/sync.log 2>&1 &     Stop: pkill -f sync_ckpt.sh
cd "$(dirname "$0")/stage" || exit 1
mkdir -p ckpt/run1
RUN=${RUN:-run1}
step_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['step'])" "$1" 2>/dev/null; }
while true; do
  for tag in best last; do
    remote=""
    for try in 1 2 3; do   # the AutoDL gateway / local proxy drops a connection now and then
      remote=$(ssh -o ConnectTimeout=30 autodl "cat /root/Bonsai-JEV/ckpt/$RUN/$tag/head.json 2>/dev/null" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['step'])" 2>/dev/null)
      [ -n "$remote" ] && break; sleep 20
    done
    [ -z "$remote" ] && echo "$(date '+%m-%d %H:%M') $tag: rental box unreachable this round"
    local_=$(step_of "ckpt/$RUN/$tag/head.json")
    [ -z "$remote" ] || [ "$remote" = "$local_" ] && continue
    rm -rf "ckpt/$RUN/$tag.new"; mkdir -p "ckpt/$RUN/$tag.new"
    if ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 autodl "cd /root/Bonsai-JEV/ckpt/$RUN && tar cf - -C $tag ." \
         | tar xf - -C "ckpt/$RUN/$tag.new" && [ -f "ckpt/$RUN/$tag.new/state.pt" ] && [ "$(step_of ckpt/$RUN/$tag.new/head.json)" = "$remote" ]; then
      rm -rf "ckpt/$RUN/$tag"; mv "ckpt/$RUN/$tag.new" "ckpt/$RUN/$tag"
      echo "$(date '+%m-%d %H:%M') $tag synced: step $remote $(du -sh ckpt/$RUN/$tag | cut -f1)"
    else
      rm -rf "ckpt/$RUN/$tag.new"; echo "$(date '+%m-%d %H:%M') $tag sync failed (remote step $remote), retry next round"
    fi
  done
  scp -q -o ConnectTimeout=30 "autodl:/root/Bonsai-JEV/ckpt/$RUN/eval_*.json" "autodl:/root/Bonsai-JEV/ckpt/$RUN/log.csv" "ckpt/$RUN/" 2>/dev/null
  sleep 600
done
