"""latency_bench.py - end-to-end /v1/decide latency (client-side wall clock, one request at a time).

  text       cold decisions on corpus rows (distinct states, so no prefix reuse)
  image      cold VL-RewardBench decisions (distinct images, 768 px, two answers)
  follow-up  one image scene warmed with /v1/warm, then three yes/no questions about it

    bash run_decide.sh start <adapter dir>;  python latency_bench.py --out results/final_latency
"""
import argparse
import json
import os
import statistics
import time

import requests

from zeroshot_text import load_split
from zeroshot_vlrb import Q, cut, load

URL = os.environ.get("JEV_URL", "http://localhost:8000").rstrip("/")


def timed(path, body):
    t0 = time.time()
    r = requests.post(f"{URL}{path}", json=body, timeout=600)
    r.raise_for_status()
    return time.time() - t0, r.json()


def summary(name, xs, toks=None):
    xs = sorted(xs)
    p95 = xs[int(0.95 * (len(xs) - 1))]
    line = f"{name:24s} n={len(xs):3d}  p50 {statistics.median(xs):.3f}s  p95 {p95:.3f}s"
    if toks:
        line += f"  median prompt tokens {int(statistics.median(toks))}"
    print(line)
    return {"n": len(xs), "p50": statistics.median(xs), "p95": p95}


def body_of(r):
    b = {"kind": r["kind"], "state": r["state"], "question": r["question"]}
    if r["kind"] == "choice":
        b["options"] = r["options"]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/latency")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    out = {}

    rows = [r for r in load_split(os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"), "test_set_30k", True, 7, 400)
            if len(json.dumps(r["state"])) < 2000][:60]
    for r in rows[:3]:   # warm-up
        timed("/v1/decide", body_of(r))
    ts, tk = [], []
    for r in rows[3:]:
        dt, d = timed("/v1/decide", body_of(r))
        ts.append(dt)
        tk.append(d.get("n_prompt_tokens") or 0)
    out["text_cold"] = summary("text, cold", ts, tk)

    if not args.no_images:
        df = load(os.path.expanduser("~/bonsai/data/vl-rewardbench/data/test-00000-of-00001.parquet"),
                  os.path.expanduser("~/bonsai/data/vl-rewardbench/img"))
        df = df.drop_duplicates("path").head(25)
        ts = []
        for r in df.itertuples():
            a, b = r.response[0], r.response[1]
            parts = ["User's question about the image: " + r.query + "\nImage: ", {"image": r.path},
                     "\n\n[Response A]\n" + cut(a, 1500) + "\n\n[Response B]\n" + cut(b, 1500)]
            dt, _ = timed("/v1/decide", {"kind": "choice", "state": parts, "question": Q, "options": ["Response A", "Response B"]})
            ts.append(dt)
        out["image_cold"] = summary("image + 2 answers, cold", ts[2:])

        qs = ["Is there a person in the image?", "Is the image taken outdoors?", "Is it safe for a robot to move forward?"]
        warm, follow = [], []
        for r in df.tail(6).itertuples():
            scene = ["Camera frame: ", {"image": r.path}, "\nThe robot is idle."]
            dt, _ = timed("/v1/warm", {"state": scene, "kinds": ["noul"]})
            warm.append(dt)
            for q in qs:
                dt, _ = timed("/v1/decide", {"kind": "noul", "state": scene, "question": q})
                follow.append(dt)
        out["scene_warm"] = summary("scene warm-up", warm)
        out["follow_up"] = summary("follow-up question", follow)
    json.dump(out, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
