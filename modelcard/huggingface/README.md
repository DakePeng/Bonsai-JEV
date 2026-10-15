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

**A calibrated, image-capable 27B decision model that fits on an 8 GB GPU.** The [JEV](https://huggingface.co/autotrust/JEV-27B) recipe (a LoRA plus a 24-slot decision head, distilled to soft labels) trained on [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), Prism ML's ternary compression of Qwen3.8-27B. Ask a typed question (yes/no, 0–5 score, or 2–16-way choice) about a state that can include images; one forward pass returns calibrated probabilities over the options. No text is generated.

**Code, training recipe and evaluation scripts:** [github.com/DakePeng/Bonsai-JEV](https://github.com/DakePeng/Bonsai-JEV) · Mirror: [ModelScope](https://modelscope.cn/models/dakepeng/Bonsai-JEV)

> **About the "GGUF" panel on this page:** the Hub summarises the repo from the 230 MB `adapter_f16.gguf`, so it shows "0.1B params" and an "F16" variant. That file is the LoRA adapter (115M parameters). The model is the 27B ternary base, `Ternary-Bonsai-2-27B-PQ2_0.gguf` (listed as "Q2_0") or `-PTQ1_0.gguf` ("TQ1_0"); the Hub does not know Prism ML's PQ2_0/PTQ1_0 types and shortens their names.

The same weights also serve **System 2**: with the adapter switched off, the server is Ternary Bonsai 2 itself, byte for byte.

> **Status (2026-10-15):** trained; all numbers below are measured through the deployment path (the fork running `adapter.gguf` on the ternary file). The Decision Index run is pending.

> **Independent project.** Bonsai-JEV is built by an independent developer. It is not affiliated with, endorsed by, or sponsored by AutoTrust (autotrust), TypeSafe, or Prism ML. Their names appear only to credit their work and to identify the models and data this project builds on; all trademarks belong to their owners.

## Built on

- **[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) by [Prism ML](https://huggingface.co/prism-ml)** (Apache 2.0). *Created using Bonsai by Prism ML.* Its ternary weights are the frozen base, unchanged and included here byte-identical; its [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) is the runtime. Bonsai 2 is built from Qwen3.8-27B by Alibaba Cloud (Apache 2.0).
- **[JEV-9B](https://huggingface.co/autotrust/JEV-9B) and [JEV-27B](https://huggingface.co/autotrust/JEV-27B) by [autotrust](https://huggingface.co/autotrust)** (Apache 2.0). Training recipe, prompt template, verbalizers, the decision-head-as-lm_head-LoRA serving approach and the `/v1/decide` API are theirs; their published numbers are our targets.

## At a glance

| | JEV-27B / JEV-27B-VL | **Bonsai-JEV** |
|---|---|---|
| Base | Qwen3.8-27B, BF16 | Ternary Bonsai 2 27B, PQ2_0 (2.13 bits/weight) |
| Download | ~54 / ~56 GB | **~8.7 GB** (PQ2_0) or ~7.4 GB (PTQ1_0) |
| Runs on | one 80 GB+ GPU (vLLM) | **8 GB GPU** with PTQ1_0 (~7.1 GiB peak incl. vision, 4k context; simulated on an RTX 4070 capped at 7.5 GiB); 12 GB GPU (RTX 4070 tested); Jetson Orin NX 16 GB and Apple Silicon via the fork (not yet tested with the adapter) |
| Trainable parameters | 108.8M LoRA + 123k head | same recipe, same shapes |
| Images | zero-shot | zero-shot |
| System 2 | unmodified Qwen3.8-27B | unmodified Ternary Bonsai 2 27B |

## Results

**Text decisions**, `test_set_30k` of [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3), KL(target ‖ model) after per-kind temperature, lower is better. "Jev-labelled" = the 25,376 rows labelled by TypeSafe Jev 1.13.

| Model | Jev-labelled | noul | score | choice |
|---|---:|---:|---:|---:|
| Ternary Bonsai 2 27B, zero-shot (letter logits) | 0.443 | 0.167 | 0.724 | 0.595 |
| JEV-9B (published) | 0.019 | 0.005 | 0.023 | 0.028 |
| JEV-27B (published) | 0.017 | 0.004 | 0.021 | 0.025 |
| **Bonsai-JEV** | **0.0163** | 0.0041 | 0.0205 | 0.0245 |

The other JEV-27B card metrics, same rows and definitions:

| | JEV-27B (published) | **Bonsai-JEV** |
|---|---:|---:|
| Yes/no AUROC, Jev-labelled / all rows | 0.995 / 0.996 | 0.995 / 0.996 |
| Choice top-1 agreement with Jev, Jev-labelled | 90.5% | 91.1% |
| Choice top-1, rows with a clear top option (top-2 gap ≥ 0.1) | 95.8% | 96.2% |
| Rating error (MAE of the expected 0–5 rating) | 0.098 | 0.097 |
| KL to ground truth, unseen task families | **0.104** | 0.128 |
| Top-1 accuracy, unseen task families | 0.942 | 0.948 |
| Choice top-1 flips when the options are shuffled / reversed | 2.9% / — | 3.4% / 4.3% |

Fitted temperatures 1.014 / 1.014 / 1.012 (noul / score / choice): the model is calibrated by training. Validation KL fell from 0.070 at step 250 to 0.018 at step 4,000, flat over the last 250 steps.

**Image decisions**, [VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench) (1,247 pairs, both orders averaged, the JEV demo protocol):

| Model | overall | hallucination | reasoning | general | order-agree |
|---|---:|---:|---:|---:|---:|
| Ternary Bonsai 2 27B, zero-shot | 74.5 | 78.9 | 72.2 | 60.2 | 50.9% |
| JEV-27B-VL (published) | 78.3 | 83.2 | 78.2 | 58.0 | 89% |
| **Bonsai-JEV** | **78.0** | **84.9** | 74.1 | 56.4 | 88.4% |

A tie with JEV-27B-VL (±2.4 at 1,247 pairs). One forward pass without averaging: 77.5 / 79.1 (as given / swapped).

Order-agree is the share of pairs whose verdict survives swapping the two answers; it shows whether a single forward pass can be trusted.

**Agent judges** (zero-shot, JEV-27B-DEMO prompts): AgentRewardBench AUROC **0.930** (JEV-27B-VL 0.91), precision / recall at 0.5 74.7 / 74.7 (78.4 / 70.2); Plan-RewardBench macro accuracy **74.7** (73.2). **[JevBench-231](https://github.com/fstandhartinger/jevbench)**: 194 / 231 with JEV's `bare-v1` prompt, +28 over the zero-shot base with the same prompt.

**Speed and memory**, RTX 4070 12 GB, PrismML llama.cpp fork:

| | |
|---|---|
| Prompt processing / generation, PQ2_0 / PTQ1_0 | 1,226 / 49.6 and 1,315 / 63.1 tokens/s |
| Peak VRAM (PQ2_0 + vision + 4k context) | 9.17 GB |
| `/v1/decide` end to end: text / image + two answers, cold (p50) | 0.62 s / 0.99 s |
| Follow-up question on a warmed scene (p50) | 0.23 s |
| 8 GB card: PTQ1_0 + fp16 adapter + vision, 4k context, VRAM capped at 7.5 GiB | text 0.42 s, image 1.04 s (p50), ~7.1 GiB used; vision on the CPU (`MMPROJ=cpu`): ~6.3 GiB, image 3.4 s |
| Max context on 12 GB | ~24–32k tokens (f16 KV), ~48–60k (q8_0 KV) |

**PTQ1_0** runs the same adapter with identical decisions (100% top-1 agreement, zero probability difference on 3,000 rows) and is faster. **Concurrency:** 40 concurrent decisions over 8 slots match sequential ones (max |Δp| 0.0035, no flips; mixed chat unaffected); the fork batches only requests with the same LoRA set.

## Quickstart

Build the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) from source with CUDA (or Metal). Stock llama.cpp cannot load these files; on AVX-512 CPUs (e.g. AMD Zen 4) use a source build, the release binaries crash on load.

```bash
pip install fastapi uvicorn requests pydantic pillow
# 12 GB+ GPU: PQ2_0 base (~8.3 GB). Either source; ModelScope is faster from mainland China.
hf download dakebakery/Bonsai-JEV --exclude "*PTQ1_0*" --exclude "adapter_f16.gguf" --exclude "peft/*" --exclude "head.*" --local-dir Bonsai-JEV-weights
modelscope download --model dakepeng/Bonsai-JEV --exclude "*PTQ1_0*" "adapter_f16.gguf" "peft/*" "head.*" --local_dir Bonsai-JEV-weights
# start llama-server (:8080) + the /v1/decide wrapper (:8000); LLAMA_BIN = the fork's build/bin
LLAMA_BIN=~/llama.cpp/build/bin bash Bonsai-JEV-weights/run_decide.sh start Bonsai-JEV-weights

# 8 GB GPU: PTQ1_0 base + fp16 adapter (~6.8 GB download)
hf download dakebakery/Bonsai-JEV --exclude "*PQ2_0*" --exclude "adapter.gguf" --exclude "peft/*" --exclude "head.*" --local-dir Bonsai-JEV-weights
BASE_GGUF=Ternary-Bonsai-2-27B-PTQ1_0.gguf ADAPTER_FILE=adapter_f16.gguf CTX=4096 NP=1 \
  LLAMA_BIN=~/llama.cpp/build/bin bash Bonsai-JEV-weights/run_decide.sh start Bonsai-JEV-weights
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
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5.95 GB | the same weights, denser packing; for 8 GB cards (`BASE_GGUF=Ternary-Bonsai-2-27B-PTQ1_0.gguf`) |
| `adapter_f16.gguf` | 0.23 GB | the adapter in fp16, for 8 GB cards (`ADAPTER_FILE=adapter_f16.gguf`) |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.6 GB | vision projector |
| `adapter.gguf` | 0.46 GB | LoRA on all 64 layers + the decision head as a rank-24 LoRA on `output.weight` |
| `decision.json` | — | head bias, verbalizer ids, slot ranges, per-kind temperatures |
| `peft/`, `head.safetensors` | 0.44 GB | the same adapter for PyTorch / further training |
| `calibration.json`, `head.json` | — | fitted temperatures and fit statistics; head metadata (step 4,000) |
| `decide_server.py`, `run_decide.sh`, `zeroshot_text.py` | — | `/v1/decide` and `/v1/warm` over `llama-server` |
| `LICENSE`, `NOTICE` | — | Apache 2.0; attributions |
| `LICENSE-Ternary-Bonsai-2`, `NOTICE-Ternary-Bonsai-2.txt`, `KNOWN_ISSUES-Ternary-Bonsai-2.md` | — | Prism ML's license, notice and known issues for the base files, unchanged |

The adapter is bound to this exact base file (Hadamard rotation metadata, value-head layout). Use the bundled GGUF rather than a newer or older Bonsai 2 file.

## Training

LoRA r=16, α=32, dropout 0.05 on `in_proj_qkv, in_proj_z, out_proj, q/k/v/o_proj, gate/up/down_proj` of all 64 layers, applied to the raw (un-rotated) layer input exactly as the fork applies it; fp32 decision head 5120→24 initialised from the lm_head rows of the verbalizer tokens. Loss KL(target ‖ model) over active slots + 0.5·RPS for scores; 30% of choice rows shuffled. 128 rows/step stratified by kind, micro-batches ≤ 24k padded tokens, gradient checkpointing, AdamW (0.9, 0.98), LR 2e-4 head / 1e-4 LoRA, 3% warmup, cosine to 0.02×, 4,000 steps (JEV-27B's 5,000 steps minus its 21% placeholder rows). 518,603 training rows of [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) (the 137,203 `yuri_v1` rows with placeholder `[0.5, 0.5]` labels dropped rather than weighted ×0.05 as JEV-27B does). Dropping them costs nothing measurable: noul KL 0.0041 vs JEV-27B's 0.004, fitted noul temperature 1.014 on both. Base frozen in BF16, dequantised from the ternary file. 2× RTX PRO 6000 96 GB, ~24 h. Release checkpoint: validation KL tied at steps 3,750 and 4,000 (0.0180), so the tie was broken on the `calibration` split, where step 4,000 is lower on all three kinds; the test set was not used to choose; per-kind temperatures fitted by L-BFGS on the `calibration` split.

Code, alignment checks and the full log of decisions: [GitHub](https://github.com/DakePeng/Bonsai-JEV).

## Limitations

- **Quality ceiling is JEV level.** The ternary base can only take a LoRA. Full fine-tunes of Qwen3.8-27B score higher on the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index); this model is about memory, not about beating them.
- **Same teacher as JEV**, so on text it matches JEV-27B rather than exceeding it. Transfer to unseen task families is weaker (KL 0.128 vs 0.104).
- **Ternarisation costs more on vision than on text** (Bonsai's card: MMMU-Pro 81.7 → 75.5). Image decisions are zero-shot, as in JEV-27B-VL.
- **English prompts.** The training data is English; decisions on other languages are untested.
- **Latency:** each decision is a 27B forward pass; on Jetson Orin NX expect seconds per image decision.

## License and acknowledgements

Apache 2.0 for the adapter, head and scripts. *Created using Bonsai by Prism ML*; Ternary Bonsai 2 27B is © Prism ML, Inc., Apache 2.0, built from Qwen3.8-27B (© Alibaba Cloud, Apache 2.0) and redistributed unchanged with its license and NOTICE. Recipe and serving design from autotrust's JEV-9B / JEV-27B (Apache 2.0). Trained on [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) (Apache 2.0; `openjev_v2` CC0), the same corpus JEV-9B and JEV-27B were trained on; its `yuri_v3` labels are outputs of TypeSafe's Jev 1.13, so commercial users may want to review TypeSafe's API terms. VL-RewardBench is used for evaluation only.

Thanks to Prism ML, autotrust, SargeDev, the Qwen team and the maintainers of the Jev Decision Index.

## Also on ModelScope

[modelscope.cn/models/dakepeng/Bonsai-JEV](https://modelscope.cn/models/dakepeng/Bonsai-JEV), same files, faster from mainland China.
