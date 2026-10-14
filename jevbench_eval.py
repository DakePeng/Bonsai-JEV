"""jevbench_eval.py - JevBench-231 through /v1/decide, graded exactly like bonjev's benchmarks/jevbench_exam.py.

Same task files (fstandhartinger/jevbench @ 3749b4fc, the commit bonjev pins), same top-1 grading, so the result
lines up with bonjev's table (27B base 194/231, best adapter `vega_clef_plumb` 207/231).

JevBench questions are in TypeSafe's native format ({type, instructions, criteria}); our model was trained on JEV's
bare-v1 rows ({kind, question, options}). Translation, kept as close to the training corpus as possible:
  noul   question = instructions + the true/false criteria; options false/true; "yes" <-> P(true) > 0.5
  choice options = "<label>: <criterion>" in the task's `labels` order (the corpus writes choice options as
         "key: description"); prediction = label of the argmax
  score  question = instructions + "Levels:" + "k: <criterion>" lines; read the first N of the 0-5 score slots,
         renormalise; prediction = argmax level
The translation is part of what is measured, so always run the zero-shot base through it too and compare with
bonjev's base 194: that separates "our prompt rendering" from "our adapter".

    JEV_URL=http://localhost:8000 python jevbench_eval.py --out results/jevbench_base
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

import requests

COMMIT = "3749b4fc1b88e4f5f02a3c0b9766c4ffa57891c0"
BASE = f"https://raw.githubusercontent.com/fstandhartinger/jevbench/{COMMIT}/datasets/public"
SPLITS = ["easy", "original", "hard"]


def load(cache):
    os.makedirs(cache, exist_ok=True)
    rows = []
    for s in SPLITS:
        p = os.path.join(cache, f"{s}.jsonl")
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"{BASE}/{s}.jsonl", p)
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                r["_split"] = s
                rows.append(r)
    assert len(rows) == 231, len(rows)
    return rows


def translate(r):
    q = r["question"]
    t, ins, cr = q["type"], q.get("instructions", ""), q.get("criteria")
    if t == "noul":
        text = f"{ins}\ntrue means: {cr['true']}\nfalse means: {cr['false']}" if cr else ins
        return {"kind": "noul", "state": r["state"], "question": text}, None
    if t == "choice":
        labels = r.get("labels") or list(cr)
        return {"kind": "choice", "state": r["state"], "question": ins,
                "options": [f"{k}: {cr[k]}" for k in labels]}, labels
    if t == "score":
        levels = cr if isinstance(cr, list) else [cr[k] for k in sorted(cr, key=int)]
        assert len(levels) <= 6, "score with more than 6 levels"
        text = ins + "\nLevels:\n" + "\n".join(f"{i}: {c}" for i, c in enumerate(levels))
        return {"kind": "score", "state": r["state"], "question": text}, len(levels)
    raise ValueError(t)


def predict(r, probs, aux):
    t = r["question"]["type"]
    if t == "noul":
        return "yes" if probs[1] > 0.5 else "no", probs[1]
    if t == "choice":
        i = max(range(len(probs)), key=probs.__getitem__)
        return aux[i], probs[i]
    p = probs[:aux]
    s = sum(p)
    p = [x / s for x in p]
    i = max(range(len(p)), key=p.__getitem__)
    return str(i), p[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("JEV_URL", "http://localhost:8000"))
    ap.add_argument("--cache", default=os.path.expanduser("~/bonsai/data/jevbench"))
    ap.add_argument("--out", default="results/jevbench")
    args = ap.parse_args()
    rows = load(args.cache)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    res = []
    with open(args.out + ".jsonl", "w") as f:
        for r in rows:
            body, aux = translate(r)
            t0 = time.time()
            resp = requests.post(f"{args.url.rstrip('/')}/v1/decide", json=body, timeout=600)
            sec = time.time() - t0
            if not resp.ok:   # count as wrong, keep going
                print(f"{r['id']}: HTTP {resp.status_code} {resp.text[:200]}")
                row = {"id": r["id"], "split": r["_split"], "kind": r["question"]["type"], "ok": False, "error": resp.text[:300]}
                res.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            d = resp.json()
            got, conf = predict(r, d["probabilities"], aux)
            exp = str(r["expected"]).lower() if r["question"]["type"] == "noul" else str(r["expected"])
            row = {"id": r["id"], "split": r["_split"], "kind": r["question"]["type"], "expected": exp, "got": got,
                   "ok": got == exp, "conf": round(conf, 4), "sec": round(sec, 3), "probabilities": d["probabilities"]}
            res.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def line(name, items):
        return f"{name:10s} {sum(x['ok'] for x in items):3d}/{len(items):3d}  ({100 * sum(x['ok'] for x in items) / len(items):5.1f}%)"
    print(line("all", res))
    for k in ["choice", "noul", "score"]:
        print(line(k, [x for x in res if x["kind"] == k]))
    for s in SPLITS:
        print(line(s, [x for x in res if x["split"] == s]))
    lat = sorted(x["sec"] for x in res if "sec" in x)
    print(f"latency p50 {statistics.median(lat):.3f}s  p95 {lat[int(0.95 * len(lat))]:.3f}s  total {sum(lat):.0f}s")
    print("bonjev reference (27B, its own prompt): base 194/231, vega_clef_plumb 207/231")
    json.dump({"correct": sum(x["ok"] for x in res), "n": len(res)}, open(args.out + ".summary.json", "w"))


if __name__ == "__main__":
    main()
