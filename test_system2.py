"""test_system2.py - is System 2 (chat / reasoning) untouched when the decision adapter is loaded?

System 2 = the same llama-server with the adapter switched off per request ("lora": [{"id": 0, "scale": 0}]),
exactly like JEV serves its unmodified base next to System 1. Two runs, then a diff:

  1. server WITHOUT --lora      python test_system2.py --mode base   (writes results/s2_base.json)
  2. server WITH    --lora      python test_system2.py --mode off    (adapter scale 0, writes s2_off.json)
                                python test_system2.py --mode on     (adapter scale 1, for contrast)
  3.                            python test_system2.py --compare     (off must equal base byte for byte)

Greedy decoding, thinking disabled, 256 tokens. PASS = every `off` answer identical to `base`, first-token
top-5 logprobs within 1e-3. `on` is expected to differ (the backbone LoRA changes every hidden state).
"""
import argparse
import json
import os
import sys

import requests

PROMPTS = [
    "What is 17 * 23? Answer with the number only.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes? Show the calculation briefly.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in two sentences why the sky is blue.",
    "把下面这句话翻译成英文：今天的会议推迟到下午三点。",
    "列出三种常见的排序算法，并各用一句话说明时间复杂度。",
    "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Explain.",
    "Summarise the plot of Romeo and Juliet in one sentence.",
    "Give a JSON object with keys name and age for a 30-year-old named Lin.",
    "A robot arm must pick a red cube next to a blue one. List the steps in 3 bullet points.",
    "What is the capital of Australia?",
    "Solve for x: 3x + 7 = 22.",
]


def ask(url, prompt, scale):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0.0, "top_k": 1,
            "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none", "cache_prompt": False}
    if scale is not None:
        body["lora"] = [{"id": 0, "scale": scale}]
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    c = r.json()["choices"][0]
    first = c.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
    return {"text": c["message"].get("content", ""), "first_top5": {t["token"]: t["logprob"] for t in first}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("LLAMA_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--mode", choices=["base", "off", "on"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    if args.mode:
        scale = {"base": None, "off": 0.0, "on": 1.0}[args.mode]
        out = [ask(args.url, p, scale) for p in PROMPTS]
        json.dump(out, open(os.path.join(args.dir, f"s2_{args.mode}.json"), "w"), ensure_ascii=False, indent=1)
        print(f"{args.mode}: {len(out)} answers -> {args.dir}/s2_{args.mode}.json")
        return

    load = lambda m: json.load(open(os.path.join(args.dir, f"s2_{m}.json")))
    base, off = load("base"), load("off")
    on = load("on") if os.path.exists(os.path.join(args.dir, "s2_on.json")) else None
    same, worst = 0, 0.0
    for i, (b, o) in enumerate(zip(base, off)):
        same += b["text"] == o["text"]
        shared = set(b["first_top5"]) & set(o["first_top5"])
        worst = max([worst] + [abs(b["first_top5"][t] - o["first_top5"][t]) for t in shared])
        if b["text"] != o["text"]:
            print(f"DIFF prompt {i}: {PROMPTS[i][:50]!r}\n  base: {b['text'][:120]!r}\n  off : {o['text'][:120]!r}")
    print(f"adapter off vs base: {same}/{len(base)} answers identical, max first-token logprob diff {worst:.2e}")
    if on:
        print(f"adapter on  vs base: {sum(b['text'] == o['text'] for b, o in zip(base, on))}/{len(base)} identical (expected to differ)")
    ok = same == len(base) and worst < 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
