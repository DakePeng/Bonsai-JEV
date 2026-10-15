---
license: apache-2.0
library_name: mlx
pipeline_tag: text-classification
base_model:
- prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
base_model_relation: adapter
datasets:
- SargeDev/jev-distill-corpus-v3
language:
- en
tags:
- system-one
- decision
- calibrated-probabilities
- jev
- bonsai
- ternary
- mlx
- apple-silicon
- lora
- vision
- on-device
---

# Bonsai-JEV (MLX)

**Bonsai-JEV on Apple Silicon.** A calibrated, image-capable 27B decision model: ask a typed question (yes/no, 0–5 score, or 2–16-way choice) about a state that can include images, and one forward pass returns calibrated probabilities over the options. This repo is the MLX build: Prism ML's MLX 2-bit pack of Ternary Bonsai 2 27B plus the Bonsai-JEV LoRA and decision head, ready to serve with one command on a 16 GB Mac.

**Code and training:** [github.com/DakePeng/Bonsai-JEV](https://github.com/DakePeng/Bonsai-JEV) · GGUF build (CUDA / Vulkan / Metal, 8 GB GPUs): [dakebakery/Bonsai-JEV](https://huggingface.co/dakebakery/Bonsai-JEV) · Mirror: [ModelScope](https://modelscope.cn/models/dakepeng/Bonsai-JEV-mlx)

> **Independent project.** Bonsai-JEV is built by an independent developer. It is not affiliated with, endorsed by, or sponsored by AutoTrust (autotrust), TypeSafe, or Prism ML. Their names appear only to credit their work and to identify the models and data this project builds on; all trademarks belong to their owners.

## Built on

- **[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) by [Prism ML](https://huggingface.co/prism-ml)** (Apache 2.0). *Created using Bonsai by Prism ML.* Its MLX 2-bit pack and bundled runtime are included here unchanged (SHA-256 checked against Prism's `files.json`). Bonsai 2 is built from Qwen3.8-27B by Alibaba Cloud (Apache 2.0).
- **[JEV-9B](https://huggingface.co/autotrust/JEV-9B) and [JEV-27B](https://huggingface.co/autotrust/JEV-27B) by [autotrust](https://huggingface.co/autotrust)** (Apache 2.0). Training recipe, prompt template, verbalizers and the `/v1/decide` API are theirs; their published numbers are our targets.

## Quickstart

Apple Silicon with 16 GB of unified memory or more.

```bash
hf download dakebakery/Bonsai-JEV-mlx --local-dir Bonsai-JEV-mlx
cd Bonsai-JEV-mlx
pip install -r runtime/requirements.txt fastapi uvicorn
python mlx_serve.py --pack . --adapter . --port 8000
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "kind": "choice",
  "state": ["Camera frame: ", {"image": "frame.jpg"}, "\nThe gripper is open above the table."],
  "question": "Which object should the arm pick first?",
  "options": ["red cube", "blue cube", "nothing, the scene is unsafe"]}'
```

- `kind`: `noul` (options `false`/`true`), `score` (`0`–`5`), `choice` (2–16 options, labelled A–P).
- `state`: a string, a JSON object, or a list mixing text and `{"image": path | URL | data-URL}`.
- Same request and response shape as the GGUF build and as JEV's `serve_decide.py`. One request at a time.
- Without a server: `python mlx_decide.py --pack . --adapter .` runs one example decision; import `mlx_decide` to call `decide(...)` from Python.

## How it works

Prism's runtime (`runtime/`) loads the pack as mlx-vlm's Qwen3.5 model whose linear layers rotate their input with the block-1024 Hadamard transform and run a 2-bit quantized matmul. The Bonsai-JEV LoRA (rank 16, all 64 layers) was trained on the raw, un-rotated layer input in the same value-head layout, so `mlx_decide.py` wraps each layer as `base(x) + B·A·x · α/r` with no conversion, and the decision head (24 × 5120 + bias, fp32) reads the final normed hidden state at `[decision]:`. Per-kind temperatures come from `decision.json`. The vision tower is Prism's, unchanged; images run at native resolution through the pack's Qwen2-VL processor.

## Results

Same weights and adapter as [dakebakery/Bonsai-JEV](https://huggingface.co/dakebakery/Bonsai-JEV); quality numbers are in that card (text KL to Jev 1.13 on `test_set_30k` 0.0163, JEV-27B 0.017; VL-RewardBench 78.0, JEV-27B-VL 78.3; ERQA 52.0). This build reproduces them:

| Check against the CUDA build (llama.cpp fork on an RTX 4070) | Result |
|---|---|
| Text, 300 random `test_set_30k` rows | KL(CUDA ‖ MLX) 0.00000, top-1 agreement 100%, KL to Jev 0.0195 on both |
| Images, 82 VL-RewardBench prompts (41 pairs × 2 orders) | same verdict on 96.3%, mean \|ΔP(A)\| 0.014 (the two runtimes resize images differently) |

**Speed and memory, Mac mini M4, 16 GB:**

| | |
|---|---|
| Load (pack + LoRA + head) | 5–6 s |
| Peak memory | 8.6 GiB text, 9.7 GiB with images |
| Text decision, ~84 prompt tokens | p50 2.1 s, p95 4.3 s |
| Image + two answers, ~620 prompt tokens | p50 13.8 s, p95 26.7 s |

A decision is one forward pass over the prompt; nothing is generated. Larger Apple chips (Pro / Max / Ultra) have proportionally more GPU cores and should be correspondingly faster (not measured).

## Files

| File | What |
|---|---|
| `model.safetensors`, `config.json`, `hadamard.json`, `tokenizer*.json`, `chat_template.jinja`, `generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `files.json`, `runtime/`, `PACK-RUNTIME.md`, `*-validation.json` | Prism ML's MLX 2-bit pack and loader, unchanged (8.6 GB); `runtime/` is MIT, © Apple, see `runtime/LICENSE` |
| `peft/` | the Bonsai-JEV LoRA (peft format, fp32, 0.44 GB) |
| `head.safetensors`, `head.json`, `decision.json`, `calibration.json` | decision head, verbalizer ids, per-kind temperatures |
| `mlx_decide.py`, `mlx_serve.py` | the MLX decision code and the `/v1/decide` server |
| `LICENSE`, `NOTICE` | Apache 2.0 and attributions for Bonsai-JEV |
| `LICENSE-Ternary-Bonsai-2`, `NOTICE-Ternary-Bonsai-2.txt` | Prism ML's license and notice for the base, unchanged |

The adapter is bound to this exact base (Hadamard metadata, value-head layout); keep the bundled pack.

## Limitations

- **Quality ceiling is JEV level**; same teacher as JEV, so on text it matches JEV-27B rather than exceeding it. Transfer to unseen task families is weaker (KL 0.128 vs 0.104).
- **Memory:** the MLX pack is 2-bit (there is no MLX build of the denser PTQ1_0 packing), so it needs a 16 GB Mac; an 8 GB Mac does not fit. For 8 GB GPUs use the GGUF build with PTQ1_0.
- **Images are zero-shot**, as in JEV-27B-VL; multi-image reasoning is weaker than single-image (ERQA 41.6 vs 56.1).
- **No System 2 server here.** For chat, run Prism's `quickstart.py` on the same pack without the adapter.
- **English prompts.** The training data is English.

## License and acknowledgements

Apache 2.0 for the adapter, head and scripts. *Created using Bonsai by Prism ML*; Ternary Bonsai 2 27B is © Prism ML, Inc., Apache 2.0, built from Qwen3.8-27B (© Alibaba Cloud, Apache 2.0), redistributed unchanged with its license and NOTICE. Recipe and serving design from autotrust's JEV-9B / JEV-27B (Apache 2.0). Trained on [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) (Apache 2.0; `openjev_v2` CC0), the same corpus JEV-9B and JEV-27B were trained on; its `yuri_v3` labels are outputs of TypeSafe's Jev 1.13, so commercial users may want to review TypeSafe's API terms. Evaluation sets are used for evaluation only.

Thanks to Prism ML, autotrust, SargeDev and the Qwen team.
