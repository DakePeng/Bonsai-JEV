"""di_shard.py - split the Decision Index text suite into N row files, so N runner processes can share one server.

The kit's text runner sends one request at a time. Rows come from the kit's own reader (edition filter included:
rebuilt GSM8K, ForecastBench retired), split by linked-case group so multi-request cases stay in one shard.
Afterwards concatenate the shards' results.jsonl and score them once:

    python di_shard.py --suite-dir suite-0.3 --n 8 --out shards
    python -m decision_index run --engine di_engine:BonsaiJEV --rows shards/rows-0.jsonl.gz --out runs/text/0   (x N)
    cat runs/text/*/results.jsonl > runs/text/results.jsonl
    python -m decision_index score --results runs/text/results.jsonl --suite-dir suite-0.3 --edition 0.3
"""
import argparse
import gzip
import json
import zlib
from pathlib import Path

from decision_index.runner import iter_rows
from decision_index.suite.io import Suite

ap = argparse.ArgumentParser()
ap.add_argument("--suite-dir", default="suite-0.3")
ap.add_argument("--edition", default="0.3")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--out", default="shards")
a = ap.parse_args()
suite = Suite(Path(a.suite_dir), a.edition)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
fs = [gzip.open(out / f"rows-{i}.jsonl.gz", "wt") for i in range(a.n)]
counts = [0] * a.n
for row in iter_rows(suite.row_paths, suite.in_edition):
    ev = row["_evaluation"]
    i = zlib.crc32(str(ev.get("group_id") or ev["run_id"]).encode()) % a.n
    fs[i].write(json.dumps(row, ensure_ascii=False) + "\n")
    counts[i] += 1
for f in fs:
    f.close()
print(json.dumps({"rows": sum(counts), "per_shard": counts}))
