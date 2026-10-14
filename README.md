# Bonsai-JEV

**A calibrated, image-capable 27B decision model that runs in ~9 GB.** The [JEV](https://huggingface.co/autotrust/JEV-27B) recipe (LoRA + 24-slot decision head, distilled to soft labels) trained on top of [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), PrismML's ternary (−1/0/+1) compression of Qwen3.8-27B. The base stays ternary and frozen; the adapter is a separate file. One forward pass returns calibrated probabilities over the options of a typed question (yes/no, 0–5 score, or 2–16-way choice), with or without images.

[中文说明](README.zh.md)

| | JEV-27B / JEV-27B-VL | **Bonsai-JEV** |
|---|---|---|
| Base | Qwen3.8-27B, BF16 | Ternary Bonsai 2 27B, PQ2_0 (2.13 bpw) |
| Weights in memory | ~54 GB (vLLM, one 80 GB GPU) | **7.2 GB** base + 0.6 GB vision + 0.46 GB adapter |
| Runs on | H100 / B200 | RTX 4070 12 GB (tested); Jetson Orin NX 16 GB and Apple Silicon via the fork (not yet tested with the adapter) |
| Trainable | 108.8M LoRA + 123k head | same recipe, same shapes |

## Built on

This project stands on two releases and would not exist without them:

- **[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) by [Prism ML](https://huggingface.co/prism-ml)** (Apache 2.0). *Created using Bonsai by Prism ML.* We use its ternary weights unchanged as the frozen base, its Hadamard-rotated format and `prism.hadamard.*` metadata, its vision projector, and the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (MIT) as the runtime. Bonsai 2 is itself built from Qwen3.8-27B by Alibaba Cloud (Apache 2.0).
- **[JEV-9B](https://huggingface.co/autotrust/JEV-9B) and [JEV-27B](https://huggingface.co/autotrust/JEV-27B) by [autotrust](https://huggingface.co/autotrust)** (Apache 2.0). We reproduce their training recipe (LoRA targets and rank, 24-slot decision head and its initialisation, KL + RPS loss, option shuffling, stratified batching, optimiser and schedule), their `bare-v1` prompt template and verbalizer token ids, their "decision head as an lm_head LoRA" serving trick, the `/v1/decide` API shape of their `serve_decide.py`, and their published numbers as our targets. Their [JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) card and the [JEV-27B-DEMO](https://github.com/yuhai-china/JEV-27B-DEMO) repository supplied the VL-RewardBench protocol and the client we evaluate through.

Bonsai-JEV is the JEV recipe applied to the Bonsai 2 base; credit for both ideas belongs to their authors.

Also thanks to **[bonjev](https://github.com/ChezRD/bonjev) by ChezRD** (MIT), an earlier, training-free route to decisions on Ternary Bonsai: a Rust engine over the Prism llama.cpp libraries that combines existing decision-LoRA deltas. Its published [JevBench-231](https://github.com/fstandhartinger/jevbench) and Decision Index protocol and numbers are the external comparison we report against (`jevbench_eval.py` grades exactly like its exam).

> **Status (2026-10-15): trained and evaluated through the deployment path** (fork + `adapter.gguf` on the ternary base). JevBench-231, the agent-judge benchmarks and the Decision Index run are still pending (*TBD*).

## Results

### Text decisions (`test_set_30k` of [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3))

KL(target ‖ model), lower is better, after per-kind temperature. "Jev-labelled" = the 25,376 `yuri_v3` rows, the fairest comparison with the JEV model cards. Bonsai-JEV numbers are measured through the deployment path: the PrismML fork running `adapter.gguf` on the ternary PQ2_0 file.

| Model | Jev-labelled rows | All rows* | noul | score | choice |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 27B zero-shot (no adapter, letter logits) | 0.443 | 0.481 | 0.167 | 0.724 | 0.595 |
| JEV-9B (published) | 0.019 | 0.021 | 0.005 | 0.023 | 0.028 |
| JEV-27B (published) | 0.017 | 0.019 | 0.004 | 0.021 | 0.025 |
| **Bonsai-JEV** (step 4,000, released) | **0.0163** | 0.0202 | 0.0041 | 0.0205 | 0.0245 |

\* Our "all" excludes the 2,260 `yuri_v1` rows whose labels are placeholder `[0.5, 0.5]`; JEV's includes them.

The other metrics of the JEV-27B card, same rows and definitions (`kl_breakdown.py`):

| | JEV-27B (published) | **Bonsai-JEV** |
|---|---:|---:|
| Yes/no AUROC (`noul`), Jev-labelled / all rows | 0.995 / 0.996 | **0.995 / 0.996** |
| Choice top-1 agreement with Jev, Jev-labelled | 90.5% | **91.1%** |
| Choice top-1, rows whose target's top two differ by ≥ 0.1 | 95.8% | **96.2%** |
| Rating error (`score`, MAE of the expected rating, 0–5) | 0.098 | **0.097** |
| ECE, 15 bins, after temperature | 0.0009 | 0.0006 † |
| Fitted temperatures noul / score / choice | 1.014 / 1.004 / 1.016 | 1.014 / 1.014 / 1.012 |
| KL to ground-truth labels, unseen task families (`ood`) | **0.104** | 0.128 |
| Top-1 accuracy, unseen task families | 0.942 | **0.948** |
| Choice top-1 flips when the options are reversed | — | 4.3% (zero-shot base: 65.7%) |

† Ours bins each predicted option probability against the target probability; the JEV card does not define its ECE further, so treat the two as indicative.

On Jev's own distributions the two models are indistinguishable. The one gap is transfer to unseen task families (KL 0.128 vs 0.104, between JEV-9B's 0.234 and JEV-27B), where top-1 accuracy is still on par.

Validation KL during training (1,024 rows of the `validation` split, no temperature, ±0.003 noise):

| step | 250 | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,250 | 4,000 |
|---|---|---|---|---|---|---|---|---|---|
| all | 0.070 | 0.047 | 0.035 | 0.045 | 0.025 | 0.023 | 0.020 | 0.019 | **0.018** |
| noul / score / choice | | | | | | | | 0.004 / 0.021 / 0.033 | 0.003 / 0.021 / 0.032 |

### Image decisions ([VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench), 1,247 pairs)

Same protocol as the [JEV demo](https://github.com/yuhai-china/JEV-27B-DEMO/tree/main/09-multimodal-judge): both presentation orders, averaged. The adapter was trained on text only, so image decisions are zero-shot for both models.

| Model | general | hallucination | reasoning | overall | order-agree |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 27B zero-shot | 60.2 | 78.9 | 72.2 | 74.5 | 50.9% |
| JEV-9B (published) | | | | 74.3 | |
| JEV-27B-VL (published) | 58.0 | 83.2 | 78.2 | 78.3 | 89% |
| **Bonsai-JEV** | 56.4 | **84.9** | 74.1 | **78.0** | 88.4% |

One forward pass without averaging: 77.5 (as given) / 79.1 (swapped). 78.0 vs 78.3 is inside the ±2.4 confidence interval of 1,247 pairs: a tie. The ternary base costs reasoning (−4.1) and gains hallucination (+1.7).

### JevBench-231 (top-1 accuracy, bonjev's grading)

| Model | prompt | correct |
|---|---|---:|
| Ternary Bonsai 2 27B, zero-shot | bonjev's own template | 194 / 231 (published) |
| bonjev `vega_clef_plumb` (training-free LoRA merge) | bonjev's own template | 207 / 231 (published) |
| Ternary Bonsai 2 27B, zero-shot | JEV `bare-v1` via `jevbench_eval.py` | 166 / 231 (easy 47/48, original 58/72, hard 61/111) |
| **Bonsai-JEV** | JEV `bare-v1` via `jevbench_eval.py` | *TBD* |

The two zero-shot rows differ by 28 tasks on the same base: the prompt rendering matters, so compare each adapter against the zero-shot row that uses its own prompt as well as on the absolute count. 231 tasks: ±6 is noise.

Order-agree is the share of pairs whose verdict survives swapping the two answers. The ternary base alone is near a coin flip per order; the averaged score hides that. We report it for every model.

### Speed and memory (RTX 4070 12 GB, WSL2, PrismML llama.cpp fork, CUDA 12)

| | measured |
|---|---|
| Prompt processing / generation, PQ2_0 | 1,226 / 49.6 tok/s (`llama-bench` pp512 / tg128) |
| Prompt processing / generation, PTQ1_0 | 1,315 / 63.1 tok/s |
| Peak VRAM, PQ2_0 + vision + 4k context | 9.17 GB |
| Text decision, cold, with adapter (`/v1/decide`, end to end) | p50 0.62 s, p95 0.68 s |
| Image + two answers, cold | p50 0.99 s, p95 1.22 s |
| Follow-up question on a warmed scene | p50 0.23 s |
| **8 GB card** (PTQ1_0 + fp16 adapter + vision, 4k context, VRAM capped at 7.5 GiB) | text p50 0.42 s, image p50 1.04 s, ~7.1 GiB used |

**PTQ1_0** (5.95 GB, the denser packing of the same ternary weights) runs the same adapter with identical decisions: 100% top-1 agreement and zero probability difference with PQ2_0 on 3,000 test rows, and it is faster. Use it on 8 GB cards.

**Concurrency:** 40 concurrent decisions over 8 slots match sequential ones (max |Δp| 0.0035, no flips), with 8 chat requests mixed in (identical text, no adapter leakage). The fork batches only requests with the same LoRA set, so JEV-27B-VL's vLLM limit of 8 sequences does not apply.

Max context on 12 GB: ~24–32k tokens with f16 KV, ~48–60k with q8_0 KV (only 16 of 64 layers keep a KV cache; the 48 linear-attention layers hold a fixed ~150 MB state per slot).

## Quickstart (serving)

Requires the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) built from source with CUDA (stock llama.cpp cannot load PQ2_0; on AVX-512 CPUs such as Zen 4 the release binaries crash on load, see the model's KNOWN_ISSUES).

Everything needed to serve (base GGUF, vision projector, adapter, calibration) is in one model repo, mirrored on [Hugging Face](https://huggingface.co/dakebakery/Bonsai-JEV) and [ModelScope](https://modelscope.cn/models/dakepeng/Bonsai-JEV) (uploaded at release). The base is Prism ML's file, byte-identical to the one the adapter was validated against.

```bash
# ≈ 8.7 GB. Either source:
huggingface-cli download dakebakery/Bonsai-JEV --local-dir ~/bonsai/Bonsai-JEV-weights
modelscope download --model dakepeng/Bonsai-JEV --local_dir ~/bonsai/Bonsai-JEV-weights     # faster from mainland China
# start llama-server + the /v1/decide wrapper
bash run_decide.sh start ~/bonsai/Bonsai-JEV-weights
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "kind": "choice",
  "state": ["Camera frame: ", {"image": "/path/frame.jpg"}, "\nThe gripper is open above the table."],
  "question": "Which object should the arm pick first?",
  "options": ["red cube", "blue cube", "nothing, the scene is unsafe"]}'
```

`/v1/decide` has the same request/response shape as JEV's `serve_decide.py`, so JEV's [`jev_client.py`](https://github.com/yuhai-china/JEV-27B-DEMO/blob/main/common/jev_client.py) works unchanged (`JEV_URL=http://localhost:8000 JEV_BACKEND=decide`). For several questions about one scene, `POST /v1/warm {"state": ..., "kinds": ["noul"]}` first: the model has recurrent layers, so prefix reuse needs an explicit boundary, and follow-ups then evaluate only their question tokens.

**System 2 is the untouched base.** `run_decide.sh` loads the adapter with global scale 0 and the wrapper switches it on per decision request, so ordinary chat/completion requests to `llama-server` (port 8080) get Ternary Bonsai 2 exactly: with the adapter switched off, 12/12 test answers are byte-identical to a server without the adapter (`run_system2.sh`). With the adapter on, 9/12 change, so do not route chat through it.

## Reproducing the training

Total for one run: ~27 h on 2× RTX PRO 6000 96 GB (≈ ¥400 on AutoDL), plus a few hours of checks on a 12 GB card.

```bash
bash rental/bootstrap.sh            # env + downloads on the rented box (reuses the image's CUDA torch)
bash rental/check.sh                # 27B forward+backward (48–59 s/step on one card, 87 GB peak), full-model alignment
bash rental/smoke.sh                # head-only and full-recipe smoke on 1% of the data, export
torchrun --nproc_per_node 2 train.py --out ckpt/run1 --steps 4000
python infer_logits.py --ckpt ckpt/run1/best --split calibration --out results/calib.logits.jsonl
python calibrate.py   --logits results/calib.logits.jsonl --out ckpt/run1/best/calibration.json
python export_gguf_lora.py --ckpt ckpt/run1/best --out ckpt/run1/best/adapter.gguf --calibration ckpt/run1/best/calibration.json
```

`rental/session1.md` is the full runbook. `execution-plan.md` records every step, measurement and decision.

### Recipe (matched to the JEV-27B model card)

LoRA r=16, α=32, dropout 0.05 on `in_proj_qkv, in_proj_z, out_proj, q/k/v/o_proj, gate/up/down_proj` of all 64 layers (108.8M params). fp32 decision head 5120→24 initialised from the lm_head rows of the verbalizer tokens (`false/true`, `0`–`5`, `A`–`P`). Loss: KL(target ‖ model) over the kind's active slots + 0.5·RPS for scores; 30% of choice rows get their options shuffled. 128 rows/step stratified by kind, length-bucketed micro-batches ≤ 24k padded tokens, gradient checkpointing. AdamW (0.9, 0.98), LR 2e-4 head / 1e-4 LoRA, 3% warmup, cosine to 0.02×, clip 1.0. 4,000 steps ≈ JEV-27B's 5,000 steps minus its 21% placeholder rows (we drop the 137,203 `yuri_v1` placeholder rows instead of weighting them ×0.05). Dropping them costs nothing measurable: noul KL 0.0041 vs JEV-27B's 0.004, fitted noul temperature 1.014 on both. Release checkpoint chosen by validation KL.

### What was hard (and is solved in this repo)

- **Rebuilding the ternary model in PyTorch** (`bonsai_torch.py`): PQ2_0 blocks are unpacked straight from the GGUF; each folded weight consumes `H(s ⊙ x)` (block-1024 normalized Walsh–Hadamard with per-width sign vectors from `prism.hadamard.*` metadata); the embedding is inverse-after-lookup. Per-layer alignment against the fork's own tensors (`tools/dump-tensors`, `align_layer.py`, `align_inner.py`): all 64 layers within 1.8e-3 relative in fp32; full 64-layer BF16 forward within 4.5e-3. Three bugs only visible at this granularity: HF's RMSNorm applies `(1 + w)` while the GGUF stores `w + 1`; linear-attention value heads are stored in the fork's tiled order (`gdn_v_grouped`) and must be regrouped for HF; standalone HF layers attend non-causally unless `_attn_implementation` is pinned.
- **Where the LoRA sits.** The fork adds `B·A·x` on the *raw* input while the ternary matmul sees the rotated one (`build_lora_mm`). The PyTorch model mirrors that, so the trained LoRA exports with no math.
- **Decision head without forking the runtime.** `export_gguf_lora.py` writes the head as a rank-24 LoRA on `output.weight` (A = head delta, B = one-hot at the 24 verbalizer rows), so the fork's verbalizer logits *are* the head logits; bias and temperatures go to `decision.json`. Real-checkpoint check: fork vs PyTorch on 1,024 rows, mean KL 0.0002, argmax agreement 99.0%.
- **Two-GPU DDP over a frozen 54 GB base:** the frozen base and Hadamard buffers are excluded from DDP sync; only LoRA and head gradients are reduced.

## Limitations

- **Quality ceiling is JEV level.** The ternary base can only take a LoRA; full fine-tunes of the same Qwen3.8-27B reach 62–66 on the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) where JEV-27B scores 55.6. This model is about *memory*, not about beating them.
- **Distilled from the same teacher as JEV**, so on text it matches JEV-27B rather than exceeding it. Transfer to unseen task families is weaker (OOD KL 0.128 vs 0.104).
- **Ternarisation costs more on vision than on text** (Bonsai's own card: MMMU-Pro 81.7 → 75.5). The image head is zero-shot, as in JEV-27B-VL; an image-decision training round is planned.
- **Latency:** one decision costs the compute of a 27B forward pass. On Jetson Orin NX expect several seconds per image decision.

## Repository layout

| path | what |
|---|---|
| `bonsai_torch.py` | PQ2_0 unpacker, Hadamard modules, GGUF → HF loader, `build_model`, decision-head init |
| `train.py` | the JEV recipe; DDP, atomic checkpoints, `best/` by validation KL |
| `infer_logits.py`, `calibrate.py`, `eval_text.py` | 24-slot logits, per-kind temperature (L-BFGS), KL/accuracy/ECE |
| `export_gguf_lora.py` | peft checkpoint → one GGUF LoRA for the fork + `decision.json` |
| `decide_server.py`, `run_decide.sh` | `/v1/decide` and `/v1/warm` over `llama-server` |
| `zeroshot_text.py`, `zeroshot_vlrb.py` | zero-shot baselines; JEV's bare-v1 prompt and verbalizer ids |
| `tools/dump-tensors`, `tools/decide-logits`, `tools/decide-logits-mm` | small llama.cpp examples (copy into the fork's `examples/`) |
| `align_layer.py`, `align_inner.py`, `test_export.py`, `check_export_real.py`, `test_decide.py`, `test_system2.py` | numerical checks |
| `rental/` | rented-GPU bootstrap, runbook, parallel ModelScope fetch, checkpoint sync |
| `execution-plan.md` / `.zh.md` | the full plan, measurements and decisions |

## License and attribution

Code and adapter weights in this repository: **Apache 2.0** (see `LICENSE` and `NOTICE`).

- **Created using Bonsai by Prism ML.** Ternary Bonsai 2 27B is Apache 2.0, © Prism ML, Inc., built from Qwen3.8-27B (© Alibaba Cloud, Apache 2.0). Base weights are not redistributed here.
- [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp): MIT, © the ggml authors. The files under `tools/` are written against it.
- Training recipe, prompt template and verbalizers follow [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) and [JEV-27B](https://huggingface.co/autotrust/JEV-27B) (Apache 2.0).
- Trained on [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) (Apache 2.0; `openjev_v2` stream CC0). This is the same corpus JEV-9B and JEV-27B were trained on and released under Apache 2.0. Its `yuri_v3` labels (2/3 of the corpus) are output distributions of TypeSafe's Jev 1.13; as with JEV, users planning commercial use may want to review TypeSafe's API terms themselves.
- VL-RewardBench is used for evaluation only (its card states research use only); it is not redistributed or trained on.

## Acknowledgements

PrismML for Ternary Bonsai and the llama.cpp fork; autotrust for publishing the JEV recipe, adapters and demos; SargeDev for the distillation corpus; the Qwen team for Qwen3.8; the maintainers of the Jev Decision Index.
