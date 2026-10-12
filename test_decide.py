"""test_decide.py - checks for the /v1/decide service against the offline tools, plus prefix-cache latency.

1. text: 50 rows of the zero-shot sample through /v1/decide vs the decide-logits probabilities (same engine,
   must agree to ~1e-3; differences come from batch-size-dependent kernels).
2. image: 5 VL-RewardBench pairs vs the decide-logits-mm run.
3. latency: one image + scene, three questions in a row -> the 2nd/3rd must hit the prefix cache.

    JEV_URL=http://localhost:8000 python test_decide.py
"""
import json
import os
import sys
import time

import requests

from zeroshot_text import active_probs, load_split
from zeroshot_vlrb import Q, load, cut

URL = os.environ.get("JEV_URL", "http://localhost:8000").rstrip("/")


def decide(kind, state, question, options=None):
    body = {"kind": kind, "state": state, "question": question}
    if options:
        body["options"] = options
    r = requests.post(f"{URL}/v1/decide", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def main():
    worst = 0.0
    # 1. text vs decide-logits
    rows = load_split(os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"), "test_set_30k", True, 0, 600)[:50]
    raw = [json.loads(l) for l in open("results/zs_sample600.logits.jsonl")][:50]
    for r, o in zip(rows, raw):
        ref = active_probs(r, o["logits"])
        got = decide(r["kind"], r["state"], r["question"], r["options"] if r["kind"] == "choice" else None)["probabilities"]
        worst = max(worst, max(abs(a - b) for a, b in zip(got, ref)))
    print(f"text: 50 rows, max |p_server - p_tool| = {worst:.2e}")

    # 2. image vs decide-logits-mm
    df = load(os.path.expanduser("~/bonsai/data/vl-rewardbench/data/test-00000-of-00001.parquet"),
              os.path.expanduser("~/bonsai/data/vl-rewardbench/img")).head(5)
    raw = [json.loads(l) for l in open("results/zs_vlrb.logits.jsonl")][:10]
    wi = 0.0
    for r, o in zip(df.itertuples(), raw[0::2]):
        a, b = r.response[0], r.response[1]
        parts = ["User's question about the image: " + r.query + "\nImage: ", {"image": r.path},
                 "\n\n[Response A]\n" + cut(a) + "\n\n[Response B]\n" + cut(b)]
        got = decide("choice", parts, Q, ["Response A", "Response B"])["probabilities"][0]
        za, zb = o["logits"]
        ref = 1 / (1 + 2.718281828 ** (zb - za))
        wi = max(wi, abs(got - ref))
    print(f"image: 5 pairs, max |p_server - p_tool| = {wi:.2e}")

    # 3. prefix cache latency
    r0 = df.iloc[0]
    scene = ["Scene description: a photo from a camera.\nImage: ", {"image": r0.path}, "\nThe robot is idle."]
    qs = ["Is there a person in the image?", "Is the image taken outdoors?", "Is it safe for the robot to move forward?"]
    w = requests.post(f"{URL}/v1/warm", json={"state": scene, "kinds": ["noul"]}, timeout=600).json()
    print("warm:", w)
    ts = []
    for q in qs:
        t0 = time.time()
        d = decide("noul", scene, q)
        ts.append((time.time() - t0, d.get("n_prompt_tokens"), d.get("n_cached")))
    print("latency (s, prompt tokens evaluated, cached):", [(round(t, 3), n, c) for t, n, c in ts])
    ok = worst < 5e-3 and wi < 5e-3 and all(n < 60 for _, n, _ in ts)   # every question only evaluates its own tokens
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
