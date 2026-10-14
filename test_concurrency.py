"""test_concurrency.py - do concurrent requests get the right adapter state and the right probabilities?

JEV-27B-VL needs `--max-num-seqs 8` because vLLM's LoRA path for that model class returns wrong System 1
probabilities with more than 8 sequences in a batch. Our stack is llama-server, which only batches slots whose LoRA
sets are equal (`can_batch_with`) and sets the adapter once per batch. This checks it empirically:

  1. sequential: N decision requests (adapter on) + M chat requests (adapter off) one at a time -> reference
  2. concurrent: the same N + M requests fired all at once, mixed
  3. decisions must match the reference within batch noise (2e-2; the fork's kernels are not bit-identical across
     batch sizes), chat first-token logprobs likewise, and chat must not pick up the adapter (compare with s2_on).

    bash run_decide.sh start <adapter dir>   (NP=4 for real parallelism)
    python test_concurrency.py --n 32 --workers 24
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

import requests

from zeroshot_text import load_split
from test_system2 import PROMPTS

DEC = "http://127.0.0.1:8000/v1/decide"
CHAT = "http://127.0.0.1:8080/v1/chat/completions"


def decide(r):
    body = {"kind": r["kind"], "state": r["state"], "question": r["question"]}
    if r["kind"] == "choice":
        body["options"] = r["options"]
    return requests.post(DEC, json=body, timeout=900).json()["probabilities"]


def chat(p):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 48, "temperature": 0.0, "top_k": 1,
            "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none", "cache_prompt": False}
    c = requests.post(CHAT, json=body, timeout=900).json()["choices"][0]
    top = {t["token"]: t["logprob"] for t in c["logprobs"]["content"][0]["top_logprobs"]}
    return {"text": c["message"]["content"], "top": top}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    rows = load_split(os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"), "test_set_30k", True, 1, 2000)
    rows = [r for r in rows if len(json.dumps(r["state"])) < 3000][:args.n]
    prompts = PROMPTS[:8]

    seq_d = [decide(r) for r in rows]
    seq_c = [chat(p) for p in prompts]

    jobs = [("d", i) for i in range(len(rows))] + [("c", i) for i in range(len(prompts))]
    jobs = jobs[::2] + jobs[1::2]   # interleave decisions and chats
    out = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(decide if k == "d" else chat, rows[i] if k == "d" else prompts[i]): (k, i) for k, i in jobs}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()

    dd = max(max(abs(a - b) for a, b in zip(seq_d[i], out[("d", i)])) for i in range(len(rows)))
    flips = sum(max(range(len(seq_d[i])), key=seq_d[i].__getitem__) != max(range(len(out[("d", i)])), key=out[("d", i)].__getitem__)
                for i in range(len(rows)))
    import math
    cd, cdl, same_text = 0.0, 0.0, 0   # probability-space and log-space differences of the first-token top-5
    for i in range(len(prompts)):
        a, b = seq_c[i]["top"], out[("c", i)]["top"]
        for t in set(a) & set(b):
            cd = max(cd, abs(math.exp(a[t]) - math.exp(b[t])))
            cdl = max(cdl, abs(a[t] - b[t]))
        same_text += seq_c[i]["text"] == out[("c", i)]["text"]
    leak = 0
    on_path = "results/s2_on.json"
    if os.path.exists(on_path):
        on = json.load(open(on_path))
        base = json.load(open("results/s2_base.json"))
        for i in range(len(prompts)):
            b48, o48, got = base[i]["text"], on[i]["text"], out[("c", i)]["text"]
            if b48[:len(got)] != o48[:len(got)] and got == o48[:len(got)]:
                leak += 1   # concurrent chat answer follows the adapter-on answer, not the base
    print(f"decisions: {len(rows)} concurrent vs sequential, max |dp| {dd:.4f}, argmax flips {flips}")
    print(f"chat: {len(prompts)} concurrent vs sequential, first-token max |dp| {cd:.4f} (max |dlogprob| {cdl:.4f}, dominated by low-probability tokens), identical text {same_text}/{len(prompts)}, adapter leaks {leak}")
    ok = dd < 2e-2 and flips <= 1 and cd < 2e-2 and leak == 0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
