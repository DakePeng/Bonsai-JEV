"""zeroshot_vlrb.py - zero-shot VL-RewardBench accuracy of Bonsai 2 (no LoRA, no head) via the fork.

Same protocol as JEV-27B-DEMO/09-multimodal-judge/demo.py: image + user question + two responses,
"choice" over ["Response A", "Response B"], asked in both orders and averaged. Images are saved as
768px JPEGs exactly as the demo does. Reads A/B letter logits (ids 32, 33) through decide-logits-mm.

    python zeroshot_vlrb.py --limit 100
"""
import argparse
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile

Q = ("Which response answers the user's question about the image better: more accurate to what is actually in the image, "
     "correct, helpful, and free of hallucinated details?")
MARKER = "<__media__>"
cut = lambda s, n=4000: s if len(s) <= n else s[:n] + " …[truncated]"


def render(question, a, b):
    state = f"User's question about the image: {question}\nImage: {MARKER}\n\n[Response A]\n{cut(a)}\n\n[Response B]\n{cut(b)}"
    return f"[kind] choice\n[state] {state}\n[question] {Q}\n[options]\nA) Response A\nB) Response B\n[decision]:"


def domain_of(id_):
    pre = re.split(r"[_\-:/0-9]", str(id_))[0].lower()
    return "hallucination" if pre in ("hallucination", "rlaif", "rlhf") else "reasoning" if pre in ("mathverse", "mmmu") else "general"


def load(parquet, img_dir):
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(parquet)
    os.makedirs(img_dir, exist_ok=True)
    paths = []
    for r in df.itertuples():
        p = os.path.join(img_dir, f"{r.id}.jpg")
        if not os.path.exists(p):
            im = Image.open(io.BytesIO(r.image["bytes"])).convert("RGB")
            im.thumbnail((768, 768))
            im.save(p, quality=90)
        paths.append(p)
    df["path"] = paths
    df["domain"] = df.id.map(domain_of)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.expanduser("~/bonsai/data/vl-rewardbench/data/test-00000-of-00001.parquet"))
    ap.add_argument("--img-dir", default=os.path.expanduser("~/bonsai/data/vl-rewardbench/img"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--mmproj", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"))
    ap.add_argument("--tool", default=os.path.expanduser("~/bonsai/llama.cpp/build/bin/llama-decide-logits-mm"))
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--extra", default="", help="extra tool args, e.g. '--image-min-tokens 1024'")
    ap.add_argument("--out", default="results/zs_vlrb")
    ap.add_argument("--from-logits", default=None)
    ap.add_argument("--adapter-dir", default=None, help="dir with adapter.gguf + decision.json (head bias, temperatures)")
    args = ap.parse_args()

    global BIAS, T_CHOICE
    BIAS, T_CHOICE = [0.0] * 24, 1.0
    if args.adapter_dir:
        dec = json.load(open(os.path.join(args.adapter_dir, "decision.json")))
        BIAS = dec.get("head_bias", BIAS)
        T_CHOICE = dec.get("per_kind_temperature", {}).get("choice", 1.0)
        print(f"adapter {args.adapter_dir}: choice temperature {T_CHOICE:.4f}", file=sys.stderr)
    df = load(args.parquet, args.img_dir)
    if args.limit:
        df = df.head(args.limit)
    rows = list(df.itertuples())
    print(f"{len(rows)} pairs", file=sys.stderr)

    if args.from_logits:
        outs = [json.loads(l) for l in open(args.from_logits)]
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in rows:
                a, b = r.response[0], r.response[1]
                f.write(json.dumps({"prompt": render(r.query, a, b), "images": [r.path]}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"prompt": render(r.query, b, a), "images": [r.path]}, ensure_ascii=False) + "\n")
            pin = f.name
        raw = args.out + ".logits.jsonl"
        env = dict(os.environ, CANDS="32,33")
        cmd = [args.tool, "-m", args.model, "--mmproj", args.mmproj, "-ngl", "99", "-c", str(args.ctx), "--log-disable"] + args.extra.split()
        if args.adapter_dir:
            cmd += ["--lora", os.path.join(args.adapter_dir, "adapter.gguf")]
        with open(pin) as fi, open(raw, "w") as fo:
            subprocess.run(cmd, stdin=fi, stdout=fo, env=env, check=True)
        outs = [json.loads(l) for l in open(raw)]
    assert len(outs) == 2 * len(rows), (len(outs), len(rows))

    def p_a(o):
        if "error" in o:
            return None
        za, zb = o["logits"]
        # adapter: the head bias of choice slots A/B (slots 8, 9) and the choice temperature, as the server applies them
        za, zb = (za + BIAS[8]) / T_CHOICE, (zb + BIAS[9]) / T_CHOICE
        return 1 / (1 + math.exp(zb - za))

    res, skipped = [], 0
    for r, o1, o2 in zip(rows, outs[0::2], outs[1::2]):
        p1, p2 = p_a(o1), p_a(o2)
        if p1 is None or p2 is None:
            skipped += 1
            continue
        p = (p1 + (1 - p2)) / 2                     # P(first response better), both orders
        hr = list(r.human_ranking)
        res.append({"id": r.id, "domain": r.domain, "first_is_better": bool(hr[0] < hr[1]),
                    "p_first_better": p, "p_order_ab": p1, "p_order_ba": 1 - p2,
                    "order_agree": bool((p1 > 0.5) == ((1 - p2) > 0.5))})
    print(f"scored {len(res)}, skipped {skipped}", file=sys.stderr)

    def acc(items):
        return 100 * sum((x["p_first_better"] > 0.5) == x["first_is_better"] for x in items) / max(1, len(items))
    doms = ["general", "hallucination", "reasoning"]
    per = {d: acc([x for x in res if x["domain"] == d]) for d in doms}
    overall = acc(res)
    macro = sum(per.values()) / len(doms)
    agree = 100 * sum(x["order_agree"] for x in res) / max(1, len(res))
    for d in doms:
        print(f"{d:14s} n={sum(x['domain'] == d for x in res):5d} acc={per[d]:.1f}")
    single_ab = 100 * sum((x["p_order_ab"] > 0.5) == x["first_is_better"] for x in res) / max(1, len(res))
    single_ba = 100 * sum((x["p_order_ba"] > 0.5) == x["first_is_better"] for x in res) / max(1, len(res))
    print(f"{'overall':14s} n={len(res):5d} acc={overall:.1f}  macro={macro:.1f}  order-agree={agree:.1f}")
    print(f"{'single order':14s} as-given {single_ab:.1f}  swapped {single_ba:.1f}  (one forward pass, no averaging)")
    json.dump({"per_domain": per, "overall": overall, "macro": macro, "order_agree": agree,
               "single_order": {"ab": single_ab, "ba": single_ba}, "rows": res},
              open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
