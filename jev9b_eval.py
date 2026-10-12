"""jev9b_eval.py - JEV-9B (served by vLLM + serve_decide.py) on the same two baselines as zeroshot_*.py.

Uses the demo repo's jev_client unchanged (JEV_BACKEND=decide -> POST /v1/decide), so this doubles as the
harness validation: if it reproduces JEV-9B's published VL-RewardBench 74.3% and KL ~0.019, the eval code
is trusted and JEV-27B-VL's published numbers can be used directly.

    JEV_URL=http://localhost:8000 python jev9b_eval.py text [--limit N]
    JEV_URL=http://localhost:8000 python jev9b_eval.py vlrb [--limit N]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/bonsai/JEV-27B-DEMO/common"))
os.environ.setdefault("JEV_BACKEND", "decide")
import jev_client as j  # noqa: E402

from zeroshot_text import score  # noqa: E402
from zeroshot_vlrb import Q, load  # noqa: E402


def text(args):
    rows = [json.loads(l) for l in open(os.path.join(args.data, args.split + ".jsonl"))]
    rows = [r for r in rows if r["source"] != "yuri_v1"]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} rows", file=sys.stderr)

    def one(r):
        d = j.decide(r["kind"], r["state"], r["question"], r["options"] if r["kind"] == "choice" else None)
        return [d[o] for o in (r["options"] if r["kind"] == "choice" else sorted(d, key=list(d).index))]

    with cf.ThreadPoolExecutor(args.workers) as ex:
        probs = list(ex.map(one, rows))
    # noul/score come back keyed by option text in server order; make sure it matches the row's option order
    for r, p in zip(rows, probs):
        assert len(p) == len(r["options"]), (r["id"], p)
    m = score(rows, probs)
    for g in sorted(m, key=lambda g: (not g == "all", g)):
        v = m[g]
        print(f"{g:16s} n={v['n']:6d}  KL={v['kl']:.4f}  acc={v['acc']:.4f}  ECE={v['ece']:.4f}")
    with open(args.out + ".jsonl", "w") as f:
        for r, p in zip(rows, probs):
            f.write(json.dumps({"id": r["id"], "kind": r["kind"], "source": r["source"], "p": p, "target": r["target"]}) + "\n")
    json.dump(m, open(args.out + ".metrics.json", "w"), indent=1)


def vlrb(args):
    df = load(args.parquet, args.img_dir)
    if args.limit:
        df = df.head(args.limit)
    rows = list(df.itertuples())
    OPTS = ["Response A", "Response B"]
    cut = lambda s, n=4000: s if len(s) <= n else s[:n] + " …[truncated]"

    def parts(path, question, a, b):
        return ["User's question about the image: " + question + "\nImage: ", {"image": path},
                "\n\n[Response A]\n" + cut(a) + "\n\n[Response B]\n" + cut(b)]

    def judge(r):
        a, b = r.response[0], r.response[1]
        p1 = j.decide_mm("choice", parts(r.path, r.query, a, b), Q, OPTS)["Response A"]
        p2 = j.decide_mm("choice", parts(r.path, r.query, b, a), Q, OPTS)["Response B"]
        return (p1 + p2) / 2, p1, p2

    with cf.ThreadPoolExecutor(args.workers) as ex:
        out = list(ex.map(judge, rows))
    res = []
    for r, (p, a, b) in zip(rows, out):
        hr = list(r.human_ranking)
        res.append({"id": r.id, "domain": r.domain, "first_is_better": bool(hr[0] < hr[1]), "p_first_better": p,
                    "p_order_ab": a, "p_order_ba": b, "order_agree": bool((a > 0.5) == (b > 0.5))})

    def acc(items):
        return 100 * sum((x["p_first_better"] > 0.5) == x["first_is_better"] for x in items) / max(1, len(items))
    doms = ["general", "hallucination", "reasoning"]
    per = {d: acc([x for x in res if x["domain"] == d]) for d in doms}
    overall, macro = acc(res), sum(per.values()) / 3
    agree = 100 * sum(x["order_agree"] for x in res) / max(1, len(res))
    for d in doms:
        print(f"{d:14s} n={sum(x['domain'] == d for x in res):5d} acc={per[d]:.1f}")
    print(f"{'overall':14s} n={len(res):5d} acc={overall:.1f}  macro={macro:.1f}  order-agree={agree:.1f}")
    json.dump({"per_domain": per, "overall": overall, "macro": macro, "order_agree": agree, "rows": res},
              open(args.out + ".json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["text", "vlrb"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="test_set_30k")
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"))
    ap.add_argument("--parquet", default=os.path.expanduser("~/bonsai/data/vl-rewardbench/data/test-00000-of-00001.parquet"))
    ap.add_argument("--img-dir", default=os.path.expanduser("~/bonsai/data/vl-rewardbench/img"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"results/jev9b_{args.task}"
    (text if args.task == "text" else vlrb)(args)


if __name__ == "__main__":
    main()
