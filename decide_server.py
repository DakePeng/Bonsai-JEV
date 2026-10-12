"""decide_server.py - POST /v1/decide for the Bonsai 2 decision model, on top of the fork's llama-server.

Same request/response shape as JEV's serve_decide.py, so JEV-27B-DEMO/common/jev_client.py (JEV_BACKEND=decide)
and jev9b_eval.py work against it unchanged.

  POST /v1/decide  {kind: noul|score|choice, state: str | [str | {"image": path|url|data-url}], question, options?}
               ->  {kind, options, probabilities, choice_index, choice, elapsed_seconds, n_prompt_tokens, n_cached, model}

How: render JEV's bare-v1 prompt (state first, question last, images as <__media__> markers), ask llama-server
/completion for the top-N full-softmax logprobs of the next token, keep the kind's verbalizer ids, add the head
bias and divide by the per-kind temperature from decision.json (both 0/1 without a trained adapter), softmax.
llama-server does the prefix caching: a second question about the same state only processes the new tokens.

  LLAMA_URL=http://localhost:8080 DECISION_JSON=ckpt/run1/final/decision.json uvicorn decide_server:app --port 8000
"""
import base64
import json
import math
import mimetypes
import os
import time

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from zeroshot_text import RANGES, VERBALIZER_IDS, as_text

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080").rstrip("/")
_http = requests.Session()
_marker = None


def marker():
    """This fork's llama-server randomises its media marker per run (see /props, or pin LLAMA_MEDIA_MARKER)."""
    global _marker
    if _marker is None:
        _marker = _http.get(f"{LLAMA_URL}/props", timeout=10).json().get("media_marker", "<__media__>")
    return _marker


LETTERS = "ABCDEFGHIJKLMNOP"
MODEL_NAME = os.environ.get("DECIDE_MODEL_NAME", "bonsai2-27b-decision")

_dj = os.environ.get("DECISION_JSON")
DEC = json.load(open(_dj)) if _dj else {}
BIAS = DEC.get("head_bias", [0.0] * 24)
TEMPS = DEC.get("per_kind_temperature", {})
app = FastAPI()


class DecideRequest(BaseModel):
    kind: str
    state: str | dict | list
    question: str
    options: list[str] | None = None
    debug: bool = False


def _image_b64(src: str) -> str:
    if src.startswith("data:"):
        return src.split(",", 1)[1]
    if src.startswith(("http://", "https://")):
        return base64.b64encode(_http.get(src, timeout=60).content).decode()
    return base64.b64encode(open(src, "rb").read()).decode()


def render(kind, state, question, options):
    """-> (prompt string with markers, [base64 images])"""
    images, text = [], ""
    for p in (state if isinstance(state, list) else [state]):
        if isinstance(p, dict) and "image" in p:
            images.append(_image_b64(p["image"]))
            text += marker()
        else:
            text += as_text(p)
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    prompt = f"[kind] {kind}\n[state] {text}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    return prompt, images


@app.get("/health")
def health():
    r = _http.get(f"{LLAMA_URL}/health", timeout=10)
    return {"status": "ok" if r.ok else "backend down", "backend": r.json() if r.ok else None}


@app.post("/v1/decide")
def decide(req: DecideRequest):
    t0 = time.time()
    kind = req.kind
    if kind not in RANGES:
        raise HTTPException(400, f"kind must be one of {list(RANGES)}")
    options = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}.get(kind, req.options)
    if not options or (kind == "choice" and not 2 <= len(options) <= 16):
        raise HTTPException(400, "choice needs 2-16 options")
    lo, _ = RANGES[kind]
    ids = VERBALIZER_IDS[lo:lo + len(options)]
    prompt, images = render(kind, req.state, req.question, options)
    body = {
        "prompt": {"prompt_string": prompt, "multimodal_data": images} if images else prompt,
        "n_predict": 1, "n_probs": 1000, "post_sampling_probs": False, "cache_prompt": True,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "samplers": [],
    }
    r = _http.post(f"{LLAMA_URL}/completion", json=body, timeout=600)
    if not r.ok:
        raise HTTPException(502, f"llama-server: {r.status_code} {r.text[:300]}")
    j = r.json()
    top = {t["id"]: t["logprob"] for t in j["completion_probabilities"][0]["top_logprobs"]}
    floor = min(top.values()) - 1.0                      # a verbalizer outside the top-N is below the N-th
    z = [top.get(t, floor) + BIAS[lo + i] for i, t in enumerate(ids)]
    T = TEMPS.get(kind, 1.0)
    m = max(z)
    e = [math.exp((v - m) / T) for v in z]
    probs = [v / sum(e) for v in e]
    ci = max(range(len(probs)), key=lambda i: probs[i])
    out = {"kind": kind, "options": options, "probabilities": probs, "choice_index": ci, "choice": options[ci],
           "elapsed_seconds": round(time.time() - t0, 4), "n_prompt_tokens": j.get("tokens_evaluated"),
           "n_cached": j.get("tokens_cached"), "model": MODEL_NAME, "protocol": "jev27-bare-v1"}
    if req.debug:
        out["debug"] = {"logprobs": [top.get(t) for t in ids], "bias": BIAS[lo:lo + len(ids)], "temperature": T,
                        "missing_from_topn": [t for t in ids if t not in top], "timings": j.get("timings")}
    return out
