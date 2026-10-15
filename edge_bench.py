"""edge_bench.py - latency and memory of a running /v1/decide service on an edge box, from a small sample folder.

  first      the first text and image request right after the service came up
  text       new-state text decisions (no prefix to reuse)
  image      new image + two answers (VL-RewardBench pairs)
  warm       one /v1/warm per scene, then three yes/no follow-ups on the warmed scene
Memory is sampled every 0.5 s in a thread: system RAM used, the llama-server RSS, and on AMD/Intel iGPUs the DRM
VRAM/GTT counters (/sys/class/drm/card*/device/mem_info_*); on NVIDIA, nvidia-smi.

    python edge_bench.py --samples ~/bonsai/edge_samples --out results/edge_guinevere
Samples: text.jsonl (decide bodies), images.jsonl ({image, query, a, b}) + img/, meta.json ({question}).
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import threading
import time

import requests

URL = os.environ.get("JEV_URL", "http://localhost:8000").rstrip("/")


def mem_now():
    m = {}
    info = dict(l.split(":", 1) for l in open("/proc/meminfo"))
    kb = lambda k: int(info[k].split()[0])
    m["ram_used_mib"] = (kb("MemTotal") - kb("MemAvailable")) // 1024
    pid = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True).stdout.split()
    if pid:
        try:
            m["server_rss_mib"] = int(open(f"/proc/{pid[0]}/status").read().split("VmRSS:")[1].split()[0]) // 1024
        except Exception:
            pass
    for d in glob.glob("/sys/class/drm/card*/device"):
        for k in ("vram", "gtt"):
            p = f"{d}/mem_info_{k}_used"
            if os.path.exists(p):
                m[f"{k}_used_mib"] = int(open(p).read()) // 2**20
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        if out:
            m["nvidia_used_mib"] = int(out[0])
    except Exception:
        pass
    return m


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = {}, False

    def run(self):
        while not self.stop:
            for k, v in mem_now().items():
                self.peak[k] = max(self.peak.get(k, 0), v)
            time.sleep(0.5)


def timed(path, body):
    t0 = time.time()
    r = requests.post(f"{URL}{path}", json=body, timeout=1200)
    r.raise_for_status()
    return time.time() - t0, r.json()


def summary(name, xs, toks=None):
    xs = sorted(xs)
    out = {"n": len(xs), "p50": statistics.median(xs), "p95": xs[int(0.95 * (len(xs) - 1))]}
    line = f"{name:28s} n={len(xs):3d}  p50 {out['p50']:.2f}s  p95 {out['p95']:.2f}s"
    if toks:
        out["median_prompt_tokens"] = int(statistics.median(toks))
        line += f"  median prompt tokens {out['median_prompt_tokens']}"
    print(line, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.expanduser("~/bonsai/edge_samples"))
    ap.add_argument("--out", default="results/edge")
    ap.add_argument("--n-text", type=int, default=30)
    ap.add_argument("--n-image", type=int, default=12)
    ap.add_argument("--n-scenes", type=int, default=4)
    a = ap.parse_args()
    S = a.samples
    text = [json.loads(l) for l in open(f"{S}/text.jsonl")]
    imgs = [json.loads(l) for l in open(f"{S}/images.jsonl")]
    Q = json.load(open(f"{S}/meta.json"))["question"]
    img_body = lambda r: {"kind": "choice", "question": Q, "options": ["Response A", "Response B"], "state": [
        "User's question about the image: " + r["query"] + "\nImage: ", {"image": f"{S}/{r['image']}"},
        "\n\n[Response A]\n" + r["a"] + "\n\n[Response B]\n" + r["b"]]}
    idle = mem_now()
    print("memory with the service loaded, idle:", idle, flush=True)
    samp = Sampler()
    samp.start()
    res = {"idle_memory_mib": idle}
    dt, d = timed("/v1/decide", text[0])
    res["first_text"] = {"seconds": dt, "prompt_tokens": d.get("n_prompt_tokens")}
    dt, d = timed("/v1/decide", img_body(imgs[0]))
    res["first_image"] = {"seconds": dt, "prompt_tokens": d.get("n_prompt_tokens")}
    print(f"first requests after start: text {res['first_text']['seconds']:.2f}s, image {res['first_image']['seconds']:.2f}s", flush=True)
    ts, tk = [], []
    for r in text[1:1 + a.n_text]:
        dt, d = timed("/v1/decide", r)
        ts.append(dt)
        tk.append(d.get("n_prompt_tokens") or 0)
    res["text_new_state"] = summary("text, new state", ts, tk)
    ts, tk = [], []
    for r in imgs[1:1 + a.n_image]:
        dt, d = timed("/v1/decide", img_body(r))
        ts.append(dt)
        tk.append(d.get("n_prompt_tokens") or 0)
    res["image_new"] = summary("image + 2 answers, new image", ts, tk)
    qs = ["Is there a person in the image?", "Is the image taken outdoors?", "Is it safe for a robot to move forward?"]
    warm, follow = [], []
    for r in imgs[-a.n_scenes:]:
        scene = ["Camera frame: ", {"image": f"{S}/{r['image']}"}, "\nThe robot is idle."]
        dt, _ = timed("/v1/warm", {"state": scene, "kinds": ["noul"]})
        warm.append(dt)
        for q in qs:
            dt, _ = timed("/v1/decide", {"kind": "noul", "state": scene, "question": q})
            follow.append(dt)
    res["scene_warm"] = summary("warming a scene (one image)", warm)
    res["follow_up"] = summary("follow-up on a warmed scene", follow)
    samp.stop = True
    samp.join()
    res["peak_memory_mib"] = samp.peak
    print("peak memory during the run:", samp.peak, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
