"""mlx_serve.py - /v1/decide on Apple Silicon (MLX 2-bit pack + LoRA + decision head), same request/response shape
as decide_server.py. One request at a time (one model in unified memory).

    python mlx_serve.py --pack bonsai2-27b-mlx --adapter Bonsai-JEV-weights --port 8000
    curl -s localhost:8000/v1/decide -H 'content-type: application/json' \
      -d '{"kind": "noul", "state": "The gripper holds a cup above the sink.", "question": "Should it let go?"}'
`state` is a string, an object, or a list mixing text and {"image": path | URL | data-URL}.
"""
import argparse
import base64
import math
import os
import tempfile
import threading
import time
import urllib.request

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mlx_decide as M

app = FastAPI()
LOCK = threading.Lock()
S = {}


class DecideRequest(BaseModel):
    kind: str
    state: str | dict | list
    question: str
    options: list[str] | None = None


def image_file(src, tmp):
    if src.startswith("data:"):
        p = os.path.join(tmp, f"img{len(os.listdir(tmp))}")
        open(p, "wb").write(base64.b64decode(src.split(",", 1)[1]))
        return p
    if src.startswith(("http://", "https://")):
        p = os.path.join(tmp, f"img{len(os.listdir(tmp))}")
        urllib.request.urlretrieve(src, p)
        return p
    return os.path.expanduser(src)


@app.post("/v1/decide")
async def decide(req: DecideRequest):   # async: runs on the event-loop thread, where MLX's streams live (they are per-thread)
    t0 = time.time()
    if req.kind not in M.RANGES:
        raise HTTPException(400, f"kind must be one of {list(M.RANGES)}")
    options = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}.get(req.kind, req.options)
    if not options or (req.kind == "choice" and not 2 <= len(options) <= 16):
        raise HTTPException(400, "choice needs 2-16 options")
    with tempfile.TemporaryDirectory() as tmp:
        text, images = "", []
        for p in (req.state if isinstance(req.state, list) else [req.state]):
            if isinstance(p, dict) and "image" in p:
                images.append(image_file(p["image"], tmp))
                text += M.MARKER
            else:
                text += M.as_text(p)
        prompt = M.render(req.kind, text, req.question, options)
        with LOCK:
            if images:
                z, n = M.head_logits_mm(S["model"], S["head"], prompt, images)
            else:
                z, n = M.head_logits(S["model"], S["head"], S["tok"], prompt)
    lo, _ = M.RANGES[req.kind]
    T = S["dec"].get("per_kind_temperature", {}).get(req.kind, 1.0)
    probs = M.softmax([v / T for v in z[lo:lo + len(options)]])
    ci = max(range(len(probs)), key=probs.__getitem__)
    return {"kind": req.kind, "options": options, "probabilities": probs, "choice_index": ci, "choice": options[ci],
            "elapsed_seconds": round(time.time() - t0, 4), "n_prompt_tokens": n, "model": "bonsai-jev-mlx-2bit",
            "protocol": "jev27-bare-v1"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    S["model"], S["head"], S["dec"], S["tok"], _ = M.load(a.pack, a.adapter)
    uvicorn.run(app, host=a.host, port=a.port)
