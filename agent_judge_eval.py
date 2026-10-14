"""agent_judge_eval.py - agent-judge benchmarks through /v1/decide, same prompts as JEV-27B-DEMO/10-agent-judge.

  AgentRewardBench (McGill-NLP, 1,302 web-agent trajectories, expert labels): P(task completed) from goal + actions +
      final message + final screenshot, one noul pass. JEV-27B-VL: precision 78.4 / recall 70.2 at 0.5, AUROC 0.91.
      Also run without the screenshot (`text` mode) to see what the image adds.
  Plan-RewardBench (1,171 trajectory pairs, 7 splits): choice "Trajectory A/B", both orders averaged, macro accuracy.
      JEV-27B-VL: 73.2.

  prep (any machine that reaches huggingface.co; the 4070 does not):
      python agent_judge_eval.py prep --data DIR        # ~1302 judgment files + final screenshots + labels, ~0.3 GB
  run (on the box serving /v1/decide):
      JEV_URL=http://localhost:8000 python agent_judge_eval.py run --data DIR --out results/final_agentjudge
"""
import argparse
import json
import math
import os
import re
import time

ARB = "McGill-NLP/agent-reward-bench"
TXT_JUDGE = "gpt-4o-mini-noscreen-noaxtree"   # its prompt files carry goal, action history and final message as text
Q_ARB = ("Is this scenario one where: the agent successfully completed the user's task (the goal was fully achieved, "
         "or the requested information was correctly given to the user)?")
PRB = "wyy1112/Plan-RewardBench"
SPLITS = ["planning_multi_easy", "planning_multi_hard", "planning_single_easy", "planning_single_hard",
          "robust_recovery", "safety_refusal", "tool_irrelevance"]
Q_PRB = ("Which trajectory handles the user's request better: correct and efficient tool use, a correct and complete final "
         "answer, recovering sensibly from tool errors, refusing unsafe requests, and recognising when the available tools "
         "cannot help?")
OPTS = ["Trajectory A", "Trajectory B"]


def prep(data):
    import concurrent.futures as cf
    import shutil
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    os.makedirs(os.path.join(data, "shots"), exist_ok=True)
    os.makedirs(os.path.join(data, "shots_full"), exist_ok=True)
    files = [s.rfilename for s in HfApi().dataset_info(ARB).siblings]
    jfiles = [f for f in files if f.startswith("judgments/") and f"/{TXT_JUDGE}/" in f and f.endswith(".json")]
    last = {}
    for f in files:
        m = re.match(r"screenshots/([^/]+)/([^/]+)/([^/]+)/screenshot_step_(\d+)\.png$", f)
        if m and int(m.group(4)) > last.get(m.groups()[:3], (-1, ""))[0]:
            last[m.groups()[:3]] = (int(m.group(4)), f)

    def one(i_f):
        i, f = i_f
        _, bench, agent, _, name = f.split("/")
        task = name[:-5]
        d = json.load(open(hf_hub_download(ARB, f, repo_type="dataset")))
        user = [m for m in d["chat_messages"]["regular"] if m["role"] == "user"][-1]["content"]
        text = user if isinstance(user, str) else " ".join(c.get("text", "") for c in user if isinstance(c, dict))
        key = f"{bench}_{agent}_{task}"
        png = hf_hub_download(ARB, last[(bench, agent, task)][1], repo_type="dataset")
        im = Image.open(png).convert("RGB")
        if i < 30:   # a few untouched originals for the full-resolution latency check
            shutil.copy(png, os.path.join(data, "shots_full", key + ".png"))
        im.thumbnail((1024, 1024))
        im.save(os.path.join(data, "shots", key + ".jpg"), quality=88)
        return {"benchmark": bench, "agent": agent, "task_id": task, "text": text, "shot": f"shots/{key}.jpg"}

    with cf.ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(one, enumerate(jfiles)))
    with open(os.path.join(data, "arb_rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(hf_hub_download(ARB, "data/annotations.csv", repo_type="dataset"), os.path.join(data, "annotations.csv"))
    shutil.copy(hf_hub_download("McGill-NLP/agent-reward-bench-leaderboard", "results.jsonl", repo_type="space"),
                os.path.join(data, "leaderboard.jsonl"))
    for sp in SPLITS:
        shutil.copy(hf_hub_download(PRB, f"data/{sp}.jsonl", repo_type="dataset"), os.path.join(data, f"planrb_{sp}.jsonl"))
    print(f"{len(rows)} trajectories, {len(SPLITS)} Plan-RewardBench splits -> {data}")


def decide(url, kind, state, question, options=None):
    import requests
    body = {"kind": kind, "state": state, "question": question, **({"options": options} if options else {})}
    r = requests.post(f"{url}/v1/decide", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def clip(t, head=1500, tail=7000):
    return t if len(t) <= head + tail else t[:head] + "\n[... middle steps omitted ...]\n" + t[-tail:]


def labels(data):
    import csv
    from collections import Counter, defaultdict
    votes = defaultdict(Counter)
    for a in csv.DictReader(open(os.path.join(data, "annotations.csv"))):
        if a["trajectory_success"] != "Unsure":
            votes[(a["benchmark"], a["task_id"], a["model_name"])][a["trajectory_success"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def auroc(p, y):
    # rank-sum (Mann-Whitney) with average ranks for ties
    order = sorted(range(len(p)), key=p.__getitem__)
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(y)
    nneg = len(y) - npos
    return (sum(r for r, t in zip(ranks, y) if t) - npos * (npos + 1) / 2) / (npos * nneg)


def arb_summary(res, board):
    p = [o["p"] for o in res]
    y = [o["y"] for o in res]
    pred = [x > 0.5 for x in p]
    tp = sum(a and b for a, b in zip(pred, y))
    order = sorted(range(len(p)), key=lambda i: -p[i])
    cum, c = [], 0
    for i in order:
        c += y[i]
        cum.append(c)

    def prec_at(rec):   # same as the demo: smallest prefix reaching the judge's recall
        k = math.ceil(rec / 100 * sum(y))
        n = next(i for i, v in enumerate(cum) if v >= k) + 1
        return 100 * k / n
    out = {"n": len(res), "success_rate": 100 * sum(y) / len(y), "auroc": auroc(p, y),
           "precision@0.5": 100 * tp / max(1, sum(pred)), "recall@0.5": 100 * tp / sum(y),
           "vs_board": [{"judge": b["Judge"], "precision": b["Overall"], "recall": b["Recall"],
                         "ours_at_recall": prec_at(b["Recall"])} for b in sorted(board, key=lambda b: -b["Overall"])]}
    return out


def run(args):
    url = os.environ.get("JEV_URL", "http://localhost:8000").rstrip("/")
    d = args.data
    lab = labels(d)
    rows = [json.loads(l) for l in open(os.path.join(d, "arb_rows.jsonl"))]
    rows = [r | {"label": lab.get((r["benchmark"], r["task_id"], r["agent"]))} for r in rows]
    rows = [r for r in rows if r["label"]][: args.limit or None]
    board = [json.loads(l) for l in open(os.path.join(d, "leaderboard.jsonl")) if l.strip()]
    report = {}

    for mode in args.modes.split(","):
        t0 = time.time()
        res = []
        for r in rows:
            text = "Evaluating a web-browsing AI agent.\n" + clip(r["text"])
            state = [text, "\n\nFinal screenshot of the web page: ", {"image": os.path.join(d, r["shot"])}] if mode == "screenshot" \
                else text
            try:
                o = decide(url, "noul", state, Q_ARB)
            except Exception:   # rare over-long prompts: retry with a shorter history, as the demo does
                state = [text[:6000]] + state[1:] if mode == "screenshot" else text[:6000]
                o = decide(url, "noul", state, Q_ARB)
            res.append({"benchmark": r["benchmark"], "agent": r["agent"], "task_id": r["task_id"], "y": r["label"] == "Successful",
                        "p": o["probabilities"][1], "seconds": o["elapsed_seconds"], "prompt_tokens": o.get("n_prompt_tokens")})
        s = arb_summary(res, board)
        secs = sorted(x["seconds"] for x in res)
        toks = sorted(x["prompt_tokens"] or 0 for x in res)
        s["latency_p50"], s["latency_p95"], s["median_prompt_tokens"] = secs[len(secs) // 2], secs[int(0.95 * (len(secs) - 1))], toks[len(toks) // 2]
        s["wall_minutes"] = (time.time() - t0) / 60
        report[f"arb_{mode}"] = s
        json.dump(res, open(f"{args.out}_arb_{mode}.rows.json", "w"))
        print(f"AgentRewardBench [{mode}] n={s['n']}  precision {s['precision@0.5']:.1f}  recall {s['recall@0.5']:.1f}  "
              f"AUROC {s['auroc']:.3f}  p50 {s['latency_p50']:.2f}s  ({s['median_prompt_tokens']} tokens)  "
              f"| JEV-27B-VL 78.4 / 70.2 / 0.91", flush=True)
        wins = sum(b["ours_at_recall"] > b["precision"] for b in s["vs_board"])
        print(f"  ahead of {wins}/{len(s['vs_board'])} leaderboard judges at their own recall", flush=True)

    full = sorted(os.listdir(os.path.join(d, "shots_full")))
    if full and "screenshot" in args.modes:
        ts = []
        for f in full:
            o = decide(url, "noul", ["Evaluating a web-browsing AI agent.\nGoal: (latency check)\n\nFinal screenshot of the web page: ",
                                     {"image": os.path.join(d, "shots_full", f)}], Q_ARB)
            ts.append((o["elapsed_seconds"], o.get("n_prompt_tokens")))
        ts.sort()
        report["full_res_latency"] = {"n": len(ts), "p50": ts[len(ts) // 2][0], "median_tokens": ts[len(ts) // 2][1]}
        print(f"full-resolution screenshots (cold, no history): p50 {ts[len(ts) // 2][0]:.2f}s, {ts[len(ts) // 2][1]} tokens", flush=True)

    if not args.skip_planrb:
        t0 = time.time()
        acc, allp = {}, []
        for sp in SPLITS:
            ps = []
            for line in open(os.path.join(d, f"planrb_{sp}.jsonl")):
                r = json.loads(line)
                try:
                    p1 = decide(url, "choice", prb_state(r, r["chosen"], r["reject"]), Q_PRB, OPTS)["probabilities"][0]
                    p2 = decide(url, "choice", prb_state(r, r["reject"], r["chosen"]), Q_PRB, OPTS)["probabilities"][1]
                    ps.append(((p1 + p2) / 2, p1 > 0.5, p2 > 0.5))
                except Exception:
                    ps.append((0.5, None, None))
            acc[sp] = 100 * sum(p > 0.5 for p, _, _ in ps) / len(ps)
            single = 100 * sum(bool(a) for _, a, _ in ps) / len(ps)
            allp += [{"split": sp, "p_chosen": p} for p, _, _ in ps]
            print(f"Plan-RewardBench {sp}: {len(ps)} pairs, accuracy {acc[sp]:.1f} (chosen-first order alone {single:.1f})", flush=True)
        macro = sum(acc.values()) / len(acc)
        report["planrb"] = {"splits": acc, "macro": macro, "wall_minutes": (time.time() - t0) / 60}
        print(f"Plan-RewardBench macro {macro:.1f} | JEV-27B-VL 73.2, GPT-5 68.5", flush=True)
        json.dump(allp, open(f"{args.out}_planrb.rows.json", "w"))
    json.dump(report, open(args.out + ".json", "w"), indent=1)


def prb_render(traj, limit=6000):
    out = []
    for m in traj["messages"][1:]:
        c = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"], ensure_ascii=False)
        out.append(f"[{m['role']}] {c[:1500]}")
    s = "\n".join(out)
    return s if len(s) <= limit else s[:limit // 2] + "\n[...]\n" + s[-limit // 2:]


def prb_state(r, a, b):
    return {"user request": r["query"][:3000], "available tools": json.dumps(r["tools"], ensure_ascii=False)[:3000],
            "trajectory A": prb_render(a), "trajectory B": prb_render(b)}


if __name__ == "__main__":
    assert abs(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9 and auroc([1, 1], [0, 1]) == 0.5
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run"])
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/agent-judge"))
    ap.add_argument("--out", default="results/final_agentjudge")
    ap.add_argument("--modes", default="screenshot,text")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-planrb", action="store_true")
    a = ap.parse_args()
    prep(a.data) if a.cmd == "prep" else run(a)
