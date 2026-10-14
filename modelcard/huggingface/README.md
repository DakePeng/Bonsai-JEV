---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-classification
base_model:
- prism-ml/Ternary-Bonsai-2-27B-gguf
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
- gguf
- llama-cpp
- lora
- vision
- on-device
---

# Bonsai-JEV

**A calibrated, image-capable 27B decision model in ~9 GB.** The [JEV](https://huggingface.co/autotrust/JEV-27B) recipe (a LoRA plus a 24-slot decision head, distilled to soft labels) trained on [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), Prism ML's ternary compression of Qwen3.8-27B. Ask a typed question (yes/no, 0–5 score, or 2–16-way choice) about a state that can include images; one forward pass returns calibrated probabilities over the options. No text is generated.

The same weights also serve **System 2**: with the adapter switched off, the server is Ternary Bonsai 2 itself, byte for byte.

> **Status (2026-10-14): training in progress.** This card is a draft; *TBD* entries are filled in when the run and its evaluation finish.

## Built on

- **[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) by [Prism ML](https://huggingface.co/prism-ml)** (Apache 2.0). *Created using Bonsai by Prism ML.* Its ternary weights are the frozen base, unchanged and included here byte-identical; its [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) is the runtime. Bonsai 2 is built from Qwen3.8-27B by Alibaba Cloud (Apache 2.0).
- **[JEV-9B](https://huggingface.co/autotrust/JEV-9B) and [JEV-27B](https://huggingface.co/autotrust/JEV-27B) by [autotrust](https://huggingface.co/autotrust)** (Apache 2.0). Training recipe, prompt template, verbalizers, the decision-head-as-lm_head-LoRA serving approach and the `/v1/decide` API are theirs; their published numbers are our targets.

## At a glance

| | JEV-27B / JEV-27B-VL | **Bonsai-JEV** |
|---|---|---|
| Base | Qwen3.8-27B, BF16 | Ternary Bonsai 2 27B, PQ2_0 (2.13 bits/weight) |
| Download | ~54 / ~56 GB | **~8.7 GB** |
| Runs on | one 80 GB+ GPU (vLLM) | 12 GB GPU (RTX 4070 tested); Jetson Orin NX 16 GB and Apple Silicon via the fork (not yet tested with the adapter) |
| Trainable parameters | 108.8M LoRA + 123k head | same recipe, same shapes |
| Images | zero-shot | zero-shot |
| System 2 | unmodified Qwen3.8-27B | unmodified Ternary Bonsai 2 27B |

## Results

**Text decisions**, `test_set_30k` of [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3), KL(target ‖ model) after per-kind temperature, lower is better. "Jev-labelled" = the 25,376 rows labelled by TypeSafe Jev 1.13.

| Model | Jev-labelled | noul | score | choice |
|---|---:|---:|---:|---:|
| Ternary Bonsai 2 27B, zero-shot (letter logits) | 0.443 | 0.167 | 0.724 | 0.595 |
| JEV-9B (published) | 0.019 | 0.005 | 0.023 | 0.028 |
| JEV-27B (published) | 0.017 | | | |
| **Bonsai-JEV** | *TBD* | *TBD* | *TBD* | *TBD* |

Validation KL during training (1,024 rows, no temperature): 0.070 at step 250, 0.035 at 1,000, 0.025 at 2,000, **0.019 at 3,250** (noul 0.004, score 0.021, choice 0.033); final *TBD*.

**Image decisions**, [VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench) (1,247 pairs, both orders averaged, the JEV demo protocol):

| Model | overall | hallucination | reasoning | general | order-agree |
|---|---:|---:|---:|---:|---:|
| Ternary Bonsai 2 27B, zero-shot | 74.5 | 78.9 | 72.2 | 60.2 | 50.9% |
| JEV-27B-VL (published) | 78.3 | 83.2 | 78.2 | 58.0 | 89% |
| **Bonsai-JEV** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

Order-agree is the share of pairs whose verdict survives swapping the two answers; it shows whether a single forward pass can be trusted.

**Speed and memory**, RTX 4070 12 GB, PrismML llama.cpp fork:

| | |
|---|---|
| Prompt processing / generation | 1,226 / 49.6 tokens/s |
| Peak VRAM (model + vision + 4k context) | 9.17 GB |
| Text decision, 100–300 tokens | 0.1–0.35 s zero-shot; with adapter *TBD* |
| Image decision, 640×480, cold / follow-up on a warmed scene | 0.5–0.9 s / ~0.2 s |
| Max context on 12 GB | ~24–32k tokens (f16 KV), ~48–60k (q8_0 KV) |

## Quickstart

Build the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) from source with CUDA (or Metal). Stock llama.cpp cannot load these files; on AVX-512 CPUs (e.g. AMD Zen 4) use a source build, the release binaries crash on load.

```bash
huggingface-cli download dakebakery/Bonsai-JEV --local-dir Bonsai-JEV-weights
git clone https://github.com/DakePeng/Bonsai-JEV && cd Bonsai-JEV
bash run_decide.sh start ../Bonsai-JEV-weights      # llama-server on :8080 + /v1/decide on :8000
```

### System 1: typed decisions

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "kind": "choice",
  "state": ["Camera frame: ", {"image": "frame.jpg"}, "\nThe gripper is open above the table."],
  "question": "Which object should the arm pick first?",
  "options": ["red cube", "blue cube", "nothing, the scene is unsafe"]}'
# {"kind":"choice","options":[...],"probabilities":[0.71,0.22,0.07],"choice_index":0,"choice":"red cube",...}
```

- `kind`: `noul` (options `false`/`true`), `score` (`0`–`5`), `choice` (2–16 options, labelled A–P).
- `state`: a string, a JSON object, or a list mixing text and `{"image": path | URL | data-URL}`.
- The response shape matches JEV's `serve_decide.py`, so JEV's [`jev_client.py`](https://github.com/yuhai-china/JEV-27B-DEMO/blob/main/common/jev_client.py) works with `JEV_URL=http://localhost:8000 JEV_BACKEND=decide`.
- Several questions about one scene: `POST /v1/warm {"state": ..., "kinds": ["noul"]}` once, then each `/v1/decide` evaluates only its question tokens (~0.2 s on the 4070). The model has recurrent layers, so prefix reuse needs this explicit boundary, one per `kind`.

### System 2: chat and reasoning

Send ordinary OpenAI-style requests to `llama-server` on port 8080. `run_decide.sh` loads the adapter with global scale 0, so these requests see the untouched Ternary Bonsai 2: with the adapter off, 12/12 test answers are byte-identical to a server without the adapter. Do not set the adapter's scale to 1 for chat; with it on, answers change.

### Prompt format

The wrapper renders JEV's `bare-v1` template; you only pass the fields:

```
[kind] choice
[state] <state, images inline>
[question] <question>
[options]
A) <option 1>
B) <option 2>
[decision]:
```

The probabilities are read from the verbalizer tokens at `[decision]:` (`false/true`, `0`–`5`, `A`–`P`), plus the head bias and per-kind temperature from `decision.json`. Put the scene first and the question last so warmed prefixes are reused.

## Files

| File | Size | What |
|---|---:|---|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7.2 GB | Prism ML's base, byte-identical to the validated file |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.6 GB | vision projector |
| `adapter.gguf` | 0.46 GB | LoRA on all 64 layers + the decision head as a rank-24 LoRA on `output.weight` |
| `decision.json` | — | head bias, verbalizer ids, slot ranges, per-kind temperatures |
| `peft/`, `head.safetensors` | 0.44 GB | the same adapter for PyTorch / further training |
| `decide_server.py`, `run_decide.sh` | — | `/v1/decide` and `/v1/warm` over `llama-server` |
| `LICENSE`, `NOTICE` | — | Apache 2.0; attributions incl. Prism ML's NOTICE |

The adapter is bound to this exact base file (Hadamard rotation metadata, value-head layout). Use the bundled GGUF rather than a newer or older Bonsai 2 file.

## Training

LoRA r=16, α=32, dropout 0.05 on `in_proj_qkv, in_proj_z, out_proj, q/k/v/o_proj, gate/up/down_proj` of all 64 layers, applied to the raw (un-rotated) layer input exactly as the fork applies it; fp32 decision head 5120→24 initialised from the lm_head rows of the verbalizer tokens. Loss KL(target ‖ model) over active slots + 0.5·RPS for scores; 30% of choice rows shuffled. 128 rows/step stratified by kind, micro-batches ≤ 24k padded tokens, gradient checkpointing, AdamW (0.9, 0.98), LR 2e-4 head / 1e-4 LoRA, 3% warmup, cosine to 0.02×, 4,000 steps (JEV-27B's 5,000 steps minus its 21% placeholder rows). 518,603 training rows of [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) (the 137,203 placeholder `yuri_v1` rows dropped). Base frozen in BF16, dequantised from the ternary file. 2× RTX PRO 6000 96 GB, ~24 h. Release checkpoint chosen by validation KL; per-kind temperatures fitted by L-BFGS on the `calibration` split.

Code, alignment checks and the full log of decisions: [GitHub](https://github.com/DakePeng/Bonsai-JEV).

## Limitations

- **Quality ceiling is JEV level.** The ternary base can only take a LoRA. Full fine-tunes of Qwen3.8-27B score higher on the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index); this model is about memory, not about beating them.
- **Same teacher as JEV**, so on text it approaches JEV-27B rather than exceeding it. `choice` is the hardest kind.
- **Ternarisation costs more on vision than on text** (Bonsai's card: MMMU-Pro 81.7 → 75.5). Image decisions are zero-shot, as in JEV-27B-VL.
- **English prompts.** The training data is English; decisions on other languages are untested.
- **Latency:** each decision is a 27B forward pass; on Jetson Orin NX expect seconds per image decision.

## License and acknowledgements

Apache 2.0 for the adapter, head and scripts. *Created using Bonsai by Prism ML*; Ternary Bonsai 2 27B is © Prism ML, Inc., Apache 2.0, built from Qwen3.8-27B (© Alibaba Cloud, Apache 2.0) and redistributed unchanged with its license and NOTICE. Recipe and serving design from autotrust's JEV-9B / JEV-27B (Apache 2.0). Trained on [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) (Apache 2.0; `openjev_v2` CC0), the same corpus JEV-9B and JEV-27B were trained on; its `yuri_v3` labels are outputs of TypeSafe's Jev 1.13, so commercial users may want to review TypeSafe's API terms. VL-RewardBench is used for evaluation only.

Thanks to Prism ML, autotrust, SargeDev, the Qwen team and the maintainers of the Jev Decision Index.

## Also on ModelScope

[modelscope.cn/models/dakepeng/Bonsai-JEV](https://modelscope.cn/models/dakepeng/Bonsai-JEV), same files, faster from mainland China.
