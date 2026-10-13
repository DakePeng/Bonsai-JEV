#!/bin/bash
# Parallel ModelScope file download (HTTP ranges). ModelScope serves ~1.6 MB/s per stream from AutoDL but
# accepts range requests, so 16 streams give ~10x.   Usage: msdl.sh <repo> <file-in-repo> <out-path> [parts=16]
set -eu
REPO=$1; FILE=$2; OUT=$3; N=${4:-16}
U="https://www.modelscope.cn/api/v1/models/$REPO/repo?Revision=master&FilePath=$FILE"
SIZE=$(curl -s -L -I "$U" | tr -d "\r" | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
[ -n "$SIZE" ] && [ "$SIZE" -gt 0 ] || { echo "cannot get size for $FILE"; exit 1; }
if [ -f "$OUT" ] && [ "$(stat -c %s "$OUT")" = "$SIZE" ]; then echo "$OUT already complete"; exit 0; fi
mkdir -p "$(dirname "$OUT")"; TMP="$OUT.parts"; mkdir -p "$TMP"
CHUNK=$(( (SIZE + N - 1) / N ))
echo "$FILE: $SIZE bytes in $N parts of $CHUNK"
for i in $(seq 0 $((N-1))); do
  s=$((i*CHUNK)); e=$((s+CHUNK-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  [ $s -gt $e ] && continue
  want=$((e-s+1)); p="$TMP/$i"
  if [ -f "$p" ] && [ "$(stat -c %s "$p")" = "$want" ]; then continue; fi
  ( for try in 1 2 3 4 5; do curl -s -L -r $s-$e -o "$p" "$U" && [ "$(stat -c %s "$p")" = "$want" ] && exit 0; sleep 2; done; echo "part $i failed"; exit 1 ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do [ -f "$TMP/$i" ] && echo "$TMP/$i"; done) > "$OUT"
got=$(stat -c %s "$OUT")
[ "$got" = "$SIZE" ] || { echo "size mismatch: $got != $SIZE"; exit 1; }
rm -rf "$TMP"; echo "$OUT OK ($got bytes)"
