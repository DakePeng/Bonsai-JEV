"""di_engine.py - Decision Index (apolinario/decision-index, edition 0.3) engine over our /v1/decide.

The kit sends {state, questions} in TypeSafe's format; each question becomes one /v1/decide call, translated the same
way as jevbench_eval.py (choice options "key: description" in the given order; noul instructions + true/false
criteria). Multi-question rows warm the shared state once (/v1/warm), so each question only evaluates its own tokens.
Choices above 16 options use the server's wide labels (Q-Z, then one-token letter pairs), as JEV does.

    python -m decision_index pipeline --engine di_engine:BonsaiJEV --option base_url=http://127.0.0.1:8000 --out runs/bonsai-jev
"""
import json

import httpx
from decision_index.engines.base import Engine, Unsupported

CAPACITY = ("exceeds the available context", "context size", "context length", "choice needs")


def text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def translate(state, q):
    ins, cr = text(q.get("instructions", "")), q.get("criteria")
    if q["type"] == "noul":
        body = f"{ins}\ntrue means: {text(cr['true'])}\nfalse means: {text(cr['false'])}" if cr else ins
        return {"kind": "noul", "state": state, "question": body}, None
    keys = list(cr)
    return {"kind": "choice", "state": state, "question": ins, "options": [f"{k}: {text(cr[k])}" for k in keys]}, keys


class BonsaiJEV(Engine):
    name = "bonsai-jev"
    latency = "Client wall time of the /v1/decide calls for one request (one per question, plus /v1/warm for multi-question rows)."

    def __init__(self, base_url="http://127.0.0.1:8000", timeout="600", **options):
        super().__init__(base_url=base_url, timeout=timeout, **options)
        self.url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=float(timeout))
        self.provenance = {"model": "Bonsai-JEV (step 4,000) on Ternary-Bonsai-2-27B-PQ2_0", "server": "decide_server.py over the PrismML llama.cpp fork",
                           "protocol": "JEV bare-v1 via /v1/decide"}

    def post(self, path, body):
        r = self.http.post(self.url + path, json=body)
        if r.status_code >= 400:
            if any(m in r.text for m in CAPACITY):
                raise Unsupported(r.text[:200])
            raise RuntimeError(f"{path}: HTTP {r.status_code} {r.text[:300]}")
        return r.json()

    def __call__(self, state, questions):
        state = "" if state is None else state   # rendered by the server exactly as the training corpus (as_text)
        if len(questions) > 1:
            self.post("/v1/warm", {"state": state, "kinds": sorted({q["type"] for q in questions.values()})})
        answers, raw, n_in = {}, {}, 0
        for k, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported(f"question type {q['type']}")
            body, keys = translate(state, q)
            d = self.post("/v1/decide", body)
            raw[k] = {"probabilities": d["probabilities"], "adaptation": d.get("adaptation")}
            n_in += d.get("n_prompt_tokens") or 0
            if q["type"] == "noul":
                answers[k] = {"type": "noul", "noul": d["probabilities"][1]}
            else:
                p = dict(zip(keys, d["probabilities"]))
                answers[k] = {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}
        return {"model": "bonsai-jev", "answers": answers, "usage": {"input_tokens": n_in}}, raw

    def close(self):
        self.http.close()


if __name__ == "__main__":   # translation self-check, no server needed
    b, keys = translate("s", {"type": "choice", "instructions": {"task": "pick"}, "criteria": {"x": "first", "y": {"d": 1}}})
    assert keys == ["x", "y"] and b["options"] == ["x: first", 'y: {"d":1}'] and b["question"] == '{"task":"pick"}'
    b, _ = translate({"a": 1}, {"type": "noul", "instructions": "ok?", "criteria": {"true": "yes", "false": "no"}})
    assert b["question"].endswith("false means: no") and b["kind"] == "noul"
    print("ok")
