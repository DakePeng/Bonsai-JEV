"""erqa_eval.py - ERQA (Google DeepMind's embodied-reasoning QA, 400 questions, 1-6 images) through /v1/decide.

Same protocol as the JEV-35B card: each question is one `choice` decision, options A-D as given, answer = most
probable option, System 1 only ("one pass"). Images go into the state in order ("Image 1: <img>" ...), capped at
1.6 MP like the Decision Index vision rule. The option list is parsed from the question text (three formats occur:
"Choices: A. ..", "A: ..", "A) .."); the stem without the trailing answer instruction is the question.

    python erqa_eval.py prep --parquet test.parquet --out ~/bonsai/data/erqa      # any machine
    JEV_URL=http://localhost:8000 python erqa_eval.py run --data ~/bonsai/data/erqa --out results/final_erqa
"""
import argparse
import collections
import io
import json
import os
import re
import statistics
import time

INSTR = re.compile(r"\s*Please answer directly with only the letter of the correct option and nothing else\.?\s*$")
LABEL = re.compile(r"(?<![A-Za-z0-9])([A-F])[\.\):](?=\s|$)")
MAX_PIXELS = 1_638_400


def parse(q):
    """-> (stem, [option texts]) or None. Takes the first A-label that starts an in-order A, B, (C, D, ...) run."""
    q = INSTR.sub("", q)
    marks = [(m.start(), m.end(), m.group(1)) for m in LABEL.finditer(q)]
    for i, (s, e, lab) in enumerate(marks):
        if lab != "A":
            continue
        run, want = [(s, e)], "B"
        for s2, e2, lab2 in marks[i + 1:]:
            if lab2 == want:
                run.append((s2, e2))
                want = chr(ord(want) + 1)
        if len(run) < 2:
            continue
        opts = [q[run[k][1]:(run[k + 1][0] if k + 1 < len(run) else len(q))].strip().rstrip(".").strip() for k in range(len(run))]
        stem = q[:s].strip()
        stem = re.sub(r"\s*(Choices:|You have the following options\.)\s*$", "", stem).strip()
        return stem, opts
    return None


def prep(a):
    import pyarrow.parquet as pq
    from PIL import Image
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    rows, failed = [], []
    for r in pq.read_table(a.parquet).to_pylist():
        p = parse(r["question"])
        if p is None or r["answer"] not in "ABCDEF"[:len(p[1])]:
            failed.append(r["question_id"])
            continue
        paths = []
        for k, im in enumerate(r["images"]):
            img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
            if img.width * img.height > MAX_PIXELS:
                f = (MAX_PIXELS / (img.width * img.height)) ** 0.5
                img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
            path = f"img/{r['question_id']}_{k}.png"
            img.save(os.path.join(a.out, path))
            paths.append(path)
        rows.append({"id": r["question_id"], "type": r["question_type"], "answer": r["answer"], "stem": p[0],
                     "options": p[1], "images": paths})
    json.dump(rows, open(os.path.join(a.out, "erqa.json"), "w"), ensure_ascii=False, indent=0)
    print(f"{len(rows)} questions, {len(failed)} unparsed: {failed}")


def run(a):
    import requests
    url = os.environ.get("JEV_URL", "http://localhost:8000").rstrip("/")
    rows = json.load(open(os.path.join(a.data, "erqa.json")))
    res = []
    with open(a.out + ".jsonl", "w") as f:
        for r in rows:
            state = []
            for k, p in enumerate(r["images"]):
                state += [f"Image {k + 1}: " if len(r["images"]) > 1 else "Image: ", {"image": os.path.join(a.data, p)}, "\n"]
            body = {"kind": "choice", "state": state, "question": r["stem"], "options": r["options"]}
            t0 = time.time()
            resp = requests.post(f"{url}/v1/decide", json=body, timeout=600)
            sec = time.time() - t0
            if not resp.ok:
                row = {"id": r["id"], "type": r["type"], "ok": False, "error": resp.text[:300]}
            else:
                d = resp.json()
                got = "ABCDEF"[d["choice_index"]]
                row = {"id": r["id"], "type": r["type"], "answer": r["answer"], "got": got, "ok": got == r["answer"],
                       "probabilities": d["probabilities"], "sec": round(sec, 3), "tokens": d.get("n_prompt_tokens"),
                       "n_images": len(r["images"])}
            res.append(row)
            f.write(json.dumps(row) + "\n")
    by = collections.defaultdict(list)
    for x in res:
        by[x["type"]].append(x["ok"])
    print(f"{'All':22s} {len(res):4d}  {100 * sum(x['ok'] for x in res) / len(res):5.1f}%")
    for t, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"{t:22s} {len(v):4d}  {100 * sum(v) / len(v):5.1f}%")
    secs = sorted(x["sec"] for x in res if "sec" in x)
    print(f"latency p50 {statistics.median(secs):.2f}s p95 {secs[int(0.95 * (len(secs) - 1))]:.2f}s; errors {sum('error' in x for x in res)}")
    print("JEV-27B-VL one pass 53.8, thinking 61.5; JEV-35B one pass 51.3")
    json.dump({"accuracy": sum(x["ok"] for x in res) / len(res), "n": len(res),
               "by_type": {t: sum(v) / len(v) for t, v in by.items()}}, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--parquet", required=True); p.add_argument("--out", required=True)
    r = sp.add_parser("run"); r.add_argument("--data", required=True); r.add_argument("--out", default="results/final_erqa")
    a = ap.parse_args()
    prep(a) if a.cmd == "prep" else run(a)
