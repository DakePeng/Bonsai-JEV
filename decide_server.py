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
# The server starts with --lora-init-without-apply, so the adapter is OFF by default (any chat / System 2 request
# sees the untouched base). Decision and warm-up calls switch it on explicitly. Same scale on both, so warm-up
# prefixes stay reusable (llama-server only reuses cache across requests with identical LoRA sets).
LORA_ON = {"lora": [{"id": 0, "scale": 1.0}]} if DEC else {}
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


class WarmRequest(BaseModel):
    state: str | dict | list
    kinds: list[str] = ["noul"]


@app.post("/v1/warm")
def warm(req: WarmRequest):
    """Pre-process a state (scene text + images) so that following /v1/decide calls on the same state only
    evaluate their question tokens. This model has recurrent layers, so the server can only resume from a
    saved boundary: the end of a warm-up prompt is exactly such a boundary. JEV's template starts with
    [kind], so warm once per kind you intend to ask (default: noul)."""
    t0 = time.time()
    out = {}
    for kind in req.kinds:
        if kind not in RANGES:
            raise HTTPException(400, f"kind must be one of {list(RANGES)}")
        prompt, images = render(kind, req.state, "", ["x"])
        prefix = prompt.split("\n[question]")[0]          # "[kind] k\n[state] <state>"
        body = {"prompt": {"prompt_string": prefix, "multimodal_data": images} if images else prefix,
                "n_predict": 1, "n_probs": 0, "cache_prompt": True, **LORA_ON}
        r = _http.post(f"{LLAMA_URL}/completion", json=body, timeout=600)
        if not r.ok:
            raise HTTPException(502, f"llama-server: {r.status_code} {r.text[:300]}")
        tm = r.json().get("timings", {})
        out[kind] = {"n_tokens": tm.get("prompt_n", 0) + tm.get("cache_n", 0), "n_cached": tm.get("cache_n")}
    return {"warmed": out, "elapsed_seconds": round(time.time() - t0, 4)}


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
        "n_predict": 1, "n_probs": 1000, "post_sampling_probs": False, "cache_prompt": True, **LORA_ON,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "samplers": [],
    }
    for attempt in range(2):   # ponytail: one retry. Rarely (~1/230 rows) the sampled token is a partial UTF-8 byte; llama-server then holds it back and returns no tokens and no probabilities. Re-sampling fixes it
        r = _http.post(f"{LLAMA_URL}/completion", json=body, timeout=600)
        if not r.ok:
            raise HTTPException(502, f"llama-server: {r.status_code} {r.text[:300]}")
        j = r.json()
        if j.get("completion_probabilities"):
            break
        print(f"llama-server returned no probabilities (attempt {attempt + 1}): {str(j)[:600]}", flush=True)
    else:
        raise HTTPException(502, f"llama-server returned no probabilities: {str(j)[:300]}")
    top = {t["id"]: t["logprob"] for t in j["completion_probabilities"][0]["top_logprobs"]}
    floor = min(top.values()) - 1.0                      # a verbalizer outside the top-N is below the N-th
    z = [top.get(t, floor) + BIAS[lo + i] for i, t in enumerate(ids)]
    T = TEMPS.get(kind, 1.0)
    m = max(z)
    e = [math.exp((v - m) / T) for v in z]
    probs = [v / sum(e) for v in e]
    ci = max(range(len(probs)), key=lambda i: probs[i])
    out = {"kind": kind, "options": options, "probabilities": probs, "choice_index": ci, "choice": options[ci],
           "elapsed_seconds": round(time.time() - t0, 4), "n_prompt_tokens": j.get("timings", {}).get("prompt_n"),
           "n_cached": j.get("timings", {}).get("cache_n"), "model": MODEL_NAME, "protocol": "jev27-bare-v1"}
    if req.debug:
        out["debug"] = {"logprobs": [top.get(t) for t in ids], "bias": BIAS[lo:lo + len(ids)], "temperature": T,
                        "missing_from_topn": [t for t in ids if t not in top], "timings": j.get("timings")}
    return out
