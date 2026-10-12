# Bonsai 2 27B edge decision model — evaluation and execution plan

Date: 2026-10-10. Based on `bonsai-edge-plan.html` v3.1, adapted to the hardware actually on hand.

## 1. Verdict

**Viable, with three changes to the original plan.**

1. **No H20.** All training is rented. Cost is small: JEV-9B took ~3 B200-hours, so 27B is ~9–10 B200-hours per run. On the chosen rental (2× RTX PRO 6000 96GB at ¥13.96/h) a run is ¥310–380, and the whole project is **¥450–1700 ($65–240)** depending on how many runs it takes.
2. **The 4070 12GB is the single reference runtime and deployment stand-in.** It runs the same PrismML llama.cpp fork (CUDA) that the Orin will run, so it is a better reference for the Orin line than MLX. PQ2_0 27B is 7.21 GB + 0.63 GB vision + ~0.3 GB LoRA, about 9.5 GB total, with room for KV cache. It also replaces the "20.97" box (not available) for the JEV-9B FP8 baseline. The M5 is reduced to an optional one-day cross-check at the end. Orin latency cannot be measured until an Orin exists.
3. **Skip the 8B smoke test.** `Ternary-Bonsai-8B-unpacked` is Qwen3-based, not Qwen3.8. It has no linear-attention layers, so it does not exercise the `in_proj_qkv/in_proj_z/out_proj` LoRA targets or the hybrid model code path. Smoke-test on the 27B itself with 1% of data on a rented GPU (~$10). Same code, no second model to debug.

### Hardware map

| Machine | Role | Fits? |
|---|---|---|
| 4070 12GB + 7700X | Reference runtime (PrismML llama.cpp fork, CUDA), deploy stand-in, all local evals, JEV-9B FP8 baseline, per-layer PyTorch checks | Yes for inference. **No** for any 27B or 8B BF16 training |
| M5 MacBook 32GB | Optional: one-day MLX cross-check at the very end (step 25). Not on the critical path | Yes, ~11 GB |
| M1 Air 8GB (this Mac) | Editing, scripting, orchestration only | Cannot run any model in this plan |
| Rented RTX PRO 6000 96GB, 1 or 2 cards (see §1.1) | P0 full-model alignment, P1 smoke, P2 training (2 cards), P3 teacher eval | — |
| Orin NX 16GB | Final target | Not in hand. Procure or defer P4b final step |

### Packing: PQ2_0 vs PTQ1_0

Both files hold the **same** ternary weights (−1/0/+1, fp16 scale per 128 weights, same Hadamard-rotated basis, same sign vectors in `config.json`). Dequantized values are bit-identical, so model quality, KL, ECE, and decision probabilities do not depend on the choice. The Hadamard rotation is applied the same way in both; the LoRA-placement check in step 9 covers both.

| | PTQ1_0 | PQ2_0 |
|---|---|---|
| Encoding | 5 ternary digits per byte (base 3), ~1.75 bpw | 2 bits per weight, one of four slots unused, ~2.13 bpw |
| File | 5.95 GB | 7.21 GB |
| Decode in kernel | Division/modulo or lookup, slow on GPU | Shift and mask, nearly free |
| Prompt processing | 1× | 1.7–2.3× faster on every CUDA GPU in the model card |
| Generation | Slightly faster (bandwidth bound) | About equal |

One decision generates one token, so prompt processing is the whole cost. **Use PQ2_0 wherever it fits** (4070: ~9.5 GB total; Orin NX 16 GB: ~10.7 GB total). Run `llama-bench` with both on the Orin and keep whichever measures faster there. Files convert between packings without requantizing.

### 1.1 Rental: RTX PRO 6000 Blackwell 96GB

Chosen listing: domestic platform, RTX PRO 6000 96GB, ¥6.98/h per GPU, 22 cores + 110 GB RAM per GPU, 30 GB system disk, 50 GB data disk expandable to 103 GB, driver 595, CUDA ≤ 13.2, 2 cards free on one host.

**Why it works.** 96 GB holds the 54 GB BF16 frozen base plus the recipe's 24k-token micro-batch without the squeeze an H100 80GB would need. Raw BF16 throughput is roughly half an H100 and a quarter of a B200, so a run is slower but the hourly price makes the cost a wash.

**Cards per phase.** LoRA on a frozen base is data-parallel: each card holds the full base, only the 40M LoRA params' gradients are all-reduced, PCIe is enough. Two cards give ~1.9× throughput.

| Phase | Cards | Wall clock | Cost |
|---|---|---|---|
| P0 alignment + kernel check | 1 | 5–6 h | ~¥40 |
| P1 smoke | 1 | 3–4 h | ~¥30 |
| P2 training, per run | 2 (`torchrun --nproc_per_node 2`) | 22–27 h | ¥310–380 |
| P3 teacher rerun (only if harness check fails) | 1 | 8–12 h | ¥60–85 |

**Three constraints of this host.**

- **Disk is tight.** Do not write the 54 GB BF16 base to disk. Keep the 7 GB ternary file and dequantize into GPU memory at load time; `bonsai_torch.py` already does this. Then P3's 52 GB JEV-27B-VL fits alongside the dataset on the 103 GB data disk.
- **Blackwell is sm_120.** PyTorch needs a CUDA 12.8+ build. `flash-linear-attention` depends on Triton; only recent Triton supports sm_120. Step 11 verifies this before anything else.
- **Rent by phase, not continuously.** Persistent data disk, instance off between phases. Machine is actually on for 5–9 days across the 5 weeks.

Fallbacks if this listing disappears: B200 (~10 h/run, $4–9/h) or H100 80GB (~22 h/run, micro-batch cut to 8–12k tokens, $2–4/h).

### 1.2 Prior art (searched 2026-10-10)

Nobody has published this combination: a gradient-trained LoRA plus decision head, distilled from JEV, on Ternary Bonsai 2 27B. What exists:

| Work | What it is | Useful to us |
|---|---|---|
| `ChezRD/ternary-bonsai-2-27b-decision-lora` | Rank-128 adapter made by **summing three existing deltas**, no training, no data. JevBench 194→207 of 231. Runs on bonjev. | Proves the fork loads an unmerged LoRA on the 27B. Not a quality reference. |
| `AtomicChat/Ternary-Bonsai-2-27B-Abliterate-LoRA-GGUF` | Rank-1 inference-time LoRA, base pack untouched. | Tiny file. Read how the fork applies it to settle the Hadamard-placement question in step 9. |
| `OS-Software/...-Heretic-GGUF` | Abliteration merged into the ternary weights, 34 matrices changed. | Shows merge-and-requantize is possible. We do not need it. |
| `ternary-qat` (PyPI, third party) | `swap_linear` + PEFT LoRA on a ternary frozen base, re-ternarizes on save. Does not say whether it handles the Hadamard basis. | Check its GitHub in P1 before writing `bonsai_torch.py` from scratch. |
| arXiv 2608.24469, Low-Rank Ternary Adaptation | Academic, LLaMA-3 1B/3B, keeps the merged result ternary. | Different goal. Our LoRA stays separate in BF16. |

Nothing above measures KL to JEV, calibration, or vision. The plan's P0 zero-shot baseline will be the first such number.

## 2. Risks the original plan missed

| Risk | Why it matters | Check |
|---|---|---|
| **LoRA basis mismatch. CONFIRMED 2026-10-12.** | In the fork's `build_lora_mm` (`src/llama-graph.cpp`), the base matmul consumes the rotated input `cur_mm = H(s⊙Px)` but the LoRA branch consumes the raw `cur`. A LoRA trained as B·A·(H·x) would be applied to the wrong input, silently. | **Decision:** the PyTorch training model mirrors the fork exactly: frozen base on the rotated input, LoRA on the raw input. A is then already in the plain basis and export needs no math. The original "train in the rotated basis" design is dropped for the LoRA path. |
| **No JEV training code is released.** | The whole trainer (head, KL+RPS loss, stratified batching, shuffle augmentation) is written from the README. | P1 smoke must reproduce the README's loss curve shape. Budget 3–5 days, not 3. |
| **Qwen3.8 hybrid attention needs training-grade kernels.** | Linear attention (Gated DeltaNet style) needs `flash-linear-attention` or equivalent in the HF implementation for backward to be fast. | Verify on the first rented box in P0 before anything else. |
| **bonjev is not a real fallback.** | 16 commits, no training, "LoRA" there means recombined deltas. Only its logit-readout approach is reusable. | Treat as reference code, not a dependency. |
| **No Orin in hand.** | Orin latency and PTQ1_0 on sm_87 are unverifiable. The plan's 6 s target stays unconfirmed. | Decide early: buy an Orin NX 16GB (~$600–900) in week 1, or declare P4b out of scope for this round. |
| **The "already completed" JEV-9B FP8 baseline** was measured on the 20.97 box, which you don't have. | Without those numbers there is no "better than JEV-9B or stop" gate. | Get the `eval/bf16.jsonl` and result files from whoever ran it. Otherwise rerun on the 4070: vLLM FP8 works on Ada (sm_89), 9B FP8 ≈ 9 GB, use `--max-model-len 8192`. |
| **Fork binary must match the GGUF's rotation generation.** | The model card says earlier builds were "pre-rotation". An old fork binary loads the current GGUF, outputs garbage, and does not error. Same for stock llama.cpp. | Pin the fork to the commit the Bonsai-demo repo names. Record the commit hash in `README.md`. The step 7/8 logits alignment is the detector. |
| **7700X is Zen 4 (AVX-512): PQ2_0 segfaults on load with the fork's release binaries**, even with all layers on the GPU. | KNOWN_ISSUES: fixed in source 2026-09-23 (PR #245), not yet in a binary release at time of writing. | Always build the fork from source on this box (done 2026-10-12, commit e8fd209, CUDA 12.0, gcc-12). Never use release binaries here. Also avoid CUDA 13.3 builds (separate open crash). |
| **Only one local reference runtime.** | With MLX dropped, a bug in the llama.cpp fork's logits has nothing local to disagree with. | The rented PyTorch model (step 8) is the second opinion. Read the MLX `runtime/` Python source for Hadamard details; reading it costs no M5 time. |

## 3. Step-by-step plan

One engineer, ~5 weeks. Rental hours in brackets.

### Week 0 — Setup (2 days)

1. Create a Git repo for this project. Add a `requirements.txt`, `Makefile`, and a `configs/` folder.
2. 4070: build `PrismML-Eng/llama.cpp` with `-DGGML_CUDA=ON`. Pull the **PQ2_0** GGUF (7.21 GB, ~2× faster prompt processing than PTQ1_0 on every CUDA GPU in the model card, still fits 12 GB) + Q8_0 mmproj. Keep PTQ1_0 for the Orin only. Run text and image. Record prompt tok/s and VRAM. Also build the `llama-eval-callback` example; it dumps per-layer tensors and replaces the MLX layer dumps.
3. Download the MLX pack's `runtime/` folder only (a few KB of Python, no weights). It is the readable spec of the Hadamard transform and sign vectors. Read, do not run.
4. Pull `SargeDev/jev-distill-corpus-v3` (490 MB) and `fraserprice/Ternary-Bonsai-2-27B-vllm` (codes + scales + sign vectors) to the 7700X box.
5. Pick a rental provider. Create a 150 GB persistent volume. Put the dataset and the vllm-format weights on it. Write a `bootstrap.sh` that installs deps and mounts the volume so every rented box starts in under 10 minutes.

**Gate:** both local runtimes produce sane text; numbers recorded.

**Measured 2026-10-12, 4070 12GB, PQ2_0, fork commit e8fd209, CUDA 12.0, WSL2:** `llama-bench` pp512 = **1226 tok/s**, tg128 = **49.6 tok/s**. Matches the pre-run estimate (1000–1300 / ~45). Implied single-decision latency: ~0.25 s for 300 text tokens, ~0.4 s for a 640×480 image + question before vision-encoder time. Image run: vision encoder 119–125 ms for 640×480, **peak VRAM 9.17 GB** (model + mmproj + 4k context), answer correct on a synthetic test image. Text completion sane.

**GGUF metadata (PQ2_0 file), what `bonsai_torch.py` must replicate:**
- Architecture `qwen35`: 64 blocks, hidden 5120, FFN 17408, full attention every 4th block (blk 3, 7, …), 24 heads × head_dim 256, 4 KV heads; linear-attention blocks use `attn_qkv` [5120→10240], `attn_gate` [5120→6144], `ssm_out` [6144→5120], plus BF16 `ssm_alpha`/`ssm_beta` [5120→48] and F32 conv1d/norms. 402 PQ2_0 tensors, 96 BF16, 353 F32.
- `prism.hadamard.version=1`, `block_size=1024`, `transform=normalized-sylvester-walsh-hadamard`, `axis=input-last-dimension`, `sign_mode=explicit` with sign vectors for input widths 5120, 6144, 17408. 401 folded weights listed in `weight_names`. `gdn_v_grouped=true` → `ssm_out` gets the `perm_rep` feature permutation before signs and Hadamard.
- **`output.weight` is PQ2_0 and folded (rotated).** The decision head is initialized from its 24 verbalizer rows, so either (a) the head consumes `H(s⊙h)` like the fork does for `output.weight`, or (b) un-rotate the 24 rows once at init (`row_plain = row_rot · H · diag(s)`) so the head consumes raw `h`. Pick (b): the head then exports as a plain 24×5120 matrix with no runtime transform.
- `token_embd.weight` is PQ2_0 with `inverse-after-lookup` (version-1 explicit head, not tied).
- Source of truth for ternary codes and scales: **unpack the PQ2_0 GGUF directly** (bit-identical to what the runtime sees) rather than downloading the separate vllm-format repo. PQ2_0 block layout (`ggml-common.h`, `ggml-quants.c`): group of 128 weights = 1 fp16 scale `d` + 32 bytes of codes, 2 bits per weight little-endian within each byte (`q = (byte >> 2*(j%4)) & 3`), value `(q-1)*d`, code 3 unused. A 20-line numpy unpacker.
- 640×480 image = **300 image tokens** in this runtime, as the original plan assumed.
- Local assets on the 4070 box: fork at `~/bonsai/llama.cpp` (build in `build/bin`), models in `~/bonsai/models/bonsai2-27b`, dataset in `~/bonsai/data/jev-distill-corpus-v3` (from ModelScope), venv at `~/bonsai/venv`, project clone at `~/bonsai/jev`.

### Week 1 — P0 feasibility

6. **Dequant + Hadamard PyTorch model.** Write `bonsai_torch.py`: load ternary codes and scales, dequant to BF16 `(code-1)*scale`, wrap each linear so that `y = W_rot · H(s⊙Px) + B·A·x` with the per-weight permutation, signs, and block-1024 normalized Sylvester Walsh-Hadamard read from the GGUF `prism.hadamard.*` metadata. Base path rotated, LoRA path raw, exactly as the fork does it. Reuse the HF Qwen3.8 model class; only swap the linear layers.
7. **Per-layer alignment on the 4070, no rental. DONE 2026-10-12, all 64 layers.** Embedding lookup exact (0 diff); 63/64 layer outputs within 1.8e-3 relative in fp32, median ~5e-4; the final layer is compared on the output token only (the fork prunes it). Final norm 1e-7, logits 2e-3, argmax 100%. **Gate passed.** In BF16 (the training dtype; fp16 group scales round to BF16) layer outputs sit at 2–6e-3 relative and logits at 3e-3, argmax still 100%: the frozen base deviates from the runtime by ~0.5% per layer, well inside the plan's 0.01 probability tolerance. Tooling: `tools/dump-tensors` (fork example, dumps `model.input_embed`, `l_out-N`, `result_norm`, `result_output` and, with `DUMP_LAYERS=1,3`, every intermediate of those layers as raw f32), `align_layer.py` (one layer at a time, fp32 fits the 12 GB card), `align_inner.py` (projection-by-projection inside a layer). Three bugs found and fixed this way, each invisible from the logits alone:
   - **Zero-centered RMSNorm.** HF `Qwen3_5RMSNorm` applies `(1 + w)`; the GGUF stores `w + 1` already. Subtract 1 for `attn_norm`, `post_attention_norm`, `q_norm`, `k_norm`, `output_norm`. The gated `ssm_norm` is a plain weight.
   - **Value-head order in linear attention.** `gdn_v_grouped=true`: the fork's native v-head order is tiled (`h = k + nk·r`), HF's is grouped (`h = r + rep·k`). Per-head tensors in the GGUF (`v` rows of `attn_qkv`, `attn_gate`/z, `ssm_alpha`, `ssm_beta`, `ssm_dt`, `ssm_a`, the v channels of `ssm_conv1d`) are tiled and must be un-permuted for HF. `ssm_out` columns are already grouped (the fork permutes the activation instead), so `perm_for()` stays identity.
   - **Standalone HF layers need `_attn_implementation` pinned** (sdpa). Unset, the mask builder returns nothing and the eager fallback attends non-causally.
   Result: every projection matches the fork to ~1e-3 relative (the fork's reduced-precision kernels), norms to 1e-7, layer outputs to 3.5e-4 (linear) and 5.9e-4 (full attention), lm_head logits to 2e-3 with 100% argmax agreement.
8. **Full-model alignment** [PRO 6000 ×1, 3–4 h]: run the full PyTorch model, compare final logits to the fork on the same 20 prompts.
9. **LoRA placement check in llama.cpp. DONE 2026-10-12.** LoRA input is **pre-Hadamard** (raw `cur`); base input is post-Hadamard. Also noted: the fork registers rotations per weight tensor from `prism.hadamard.*` GGUF keys, some weights get a feature permutation (`perm_rep`, GDN v grouped) before signs and Hadamard, and signs are applied before the transform. `bonsai_torch.py` must read these keys and replicate the per-weight pipeline P → ⊙s → H for the base path only.
10. **Zero-shot baselines on the 4070.** No LoRA, read letter logits directly. Text KL on `test_set_30k` (the Jev-labelled subset) and VL-RewardBench via the fork. These are the floor the trained model must beat.
   **Text, IN PROGRESS 2026-10-12.** Method pinned to JEV's `bare-v1` template from `common/jev_client.py` and `judge_config.json`: `[kind] k\n[state] s\n[question] q\n[options]\n<lines>\n[decision]:`, no special tokens, option lines `false`/`true`, `0`–`5`, `A) …`; verbalizer ids 3721, 1802, 15–20, 32–47 (verified identical in the Qwen3.8 tokenizer); softmax over the kind's active ids, no bias, temperature 1. Tooling: `tools/decide-logits` (fork example, next-token logits for given ids, ~0.18 s/row on the 4070) + `zeroshot_text.py` (KL, accuracy, ECE per kind and source; per-row probs saved for the later trained-vs-base comparison). `yuri_v1` rows carry placeholder `[0.5,0.5]` labels and are excluded. 600-row shuffled sample: KL 0.46 overall (noul 0.13 / choice 0.60 / score 0.72), accuracy noul 82%, choice 48%, score 13%. Full run on the remaining ~27.7k rows is running.
10b. **Prepared 2026-10-12, queued after 10 in `run_step10.sh`.** `run_10b.sh` serves the multimodal Qwen3.5-9B base + `vl/adapter_vllm` through JEV-9B's own `vl/serve_decide.py` (vLLM 0.31, FP8, 4k context, eager, 8 seqs) and `jev9b_eval.py` runs text + VL-RewardBench through the demo's unmodified client (`JEV_BACKEND=decide`). Weights from ModelScope (JEV-9B 18 GB, Qwen3.5-9B 19 GB; note the 9B serve script pins base revision c202236, ModelScope gave latest). Risk: 12 GB may not fit; fallback is ~¥15 on the rented card.
    **JEV-9B FP8 on the 4070, two jobs in one.** vLLM FP8, `--max-model-len 8192`, using the JEV-27B-DEMO eval code unchanged. (a) Lower-bound gate: KL, ECE, robot-arm demo success. (b) **Harness validation:** if it reproduces the published 74.3% VL-RewardBench and ~0.019 KL, the eval code is trusted and JEV-27B-VL's published 78.3% / AUROC 0.91 / KL 0.017 are used directly as targets. No teacher rerun needed.
11. **Hybrid-attention training kernels** [same session]: confirm PyTorch and Triton see sm_120, then run one forward+backward on the PyTorch model with LoRA attached and gradient checkpointing, micro-batch 24k tokens. Record step time and peak memory. This confirms the P2 time estimate.

**Gate:** logits match; LoRA placement understood; step time measured. **Stop if logits cannot be aligned after the fallback (unrotate weights with H, train plain, re-rotate A on export).**

### Week 2 — P1 trainer

12. **DONE 2026-10-12 (code + CPU smoke).** `train.py`: frozen Bonsai base via `bonsai_torch.build_model` (text model without lm_head, ~51 GB BF16), peft LoRA r16/α32/dropout 0.05 on the 10 JEV target modules (fp32 LoRA params, bf16 autocast), fp32 24-slot head with bias initialised from the un-rotated lm_head rows (`decision_head_rows`), KL + 0.5·RPS, 30% choice shuffle, stratified 128-row steps, length-bucketed micro-batches under a token budget, gradient checkpointing, AdamW (0.9, 0.98) with the two LRs, cosine + 3% warmup, clip 1, `torchrun` DDP for 2 cards, checkpoint/resume (`last/`, `final/`: peft adapter + `head.safetensors` + `head.json`), CSV log, periodic validation KL by kind. Smoke-tested on the 4070 box **on CPU** with a 2-layer truncated model (`--layers 2 --device cpu`): loss falls, eval/save/resume work. 507k usable train rows → ~3,700 steps for 0.93 epoch.
    Original spec: write `train.py` on top of `peft`: 24-slot fp32 decision head initialised from `lm_head` rows; KL(target‖model) over active slots + 0.5·RPS for score; 30% option-shuffle augmentation for choice; stratified sampler (≥1/6 per kind); length bucketing; AdamW (0.9, 0.98), head LR 2e-4, LoRA 1e-4, cosine, 3% warmup, clip 1.0. Drop `yuri_v1` rows with [0.5, 0.5] placeholders.
13. **DONE 2026-10-12.** `infer_logits.py` (checkpoint → 24-slot logits per row, same JSONL as `decide-logits`), `calibrate.py` (per-kind temperature by L-BFGS minimising KL on `calibration`, clamped to [0.1, 10], writes JEV-shaped `calibration.json` with before/after KL and ECE), `eval_text.py` (KL, accuracy, ECE per kind and source, optional temperatures, works on zero-shot and trained logits alike). Shared `load_split()` keeps row filtering and order identical across all scripts. Tested on the 600-row zero-shot sample (fitted T: noul 1.15, score 2.22, choice 1.62 → the base is overconfident; KL 0.463 → 0.441) and on CPU with the 2-layer smoke checkpoint end to end.
    Original spec: `calibrate.py` (L-BFGS temperature per kind on `calibration` split) and `eval_text.py` (KL, ECE, accuracy on `test_set_30k` and `ood`).
14. **Smoke test on 27B, 1% of train** [PRO 6000 ×1, 3–4 h, ~¥30]. Loss must decrease; head-only variant first (LoRA frozen) as a sanity check, then full. Save a checkpoint and run `eval_text.py` on it.
15. Convert that checkpoint to GGUF LoRA plus a head matrix. Load in the fork on the 4070. Compare PyTorch vs fork probabilities on 200 rows. This is the earliest possible catch of the LoRA-placement bug.

**Gate:** KL falls on smoke; PyTorch and the fork agree within 0.01 on the smoke checkpoint.

### Week 3 — P2 training + P4 deploy on the 4070 in parallel

16. **Full run** [PRO 6000 ×2, 22–27 h, ¥310–380]: 0.93 epoch, ~4,750 steps, data-parallel over 2 cards. Checkpoint every 250 steps to the volume. Log loss per kind and LR to a CSV; plot against the JEV-9B README curve.
17. While it trains: on the 4070, build `/v1/decide` (FastAPI over the fork's server or `llama-cpp-python`) with LoRA + head readout of the 24 letter logits. Prefix caching for image + scene, question appended. Use the smoke checkpoint until the real one lands. If the fork lacks unmerged LoRA or head readout, add the 24-row matmul in the wrapper; bonjev's readout code is the reference.
18. If the first run's KL on `test_set_30k` is above 0.03, do one more run with longer schedule or LR tweak. Budget allows three runs total.

**Gate:** training KL converges; 4070 `/v1/decide` matches training-side probabilities within 0.01.

### Week 4 — P3 calibration and eval

19. Fit temperatures. If any is far from 1.0, treat as a training bug, not a calibration fix.
20. Text evals on the 4070: KL, ECE, accuracy on `test_set_30k` and `ood`. Compare against JEV-27B (0.017) and JEV-9B (0.019) and the zero-shot floor from step 10.
21. **Vision evals on the 4070:** run our model on VL-RewardBench, AgentRewardBench, MMRB2, QuickDraw with the validated harness from 10b. Compare to JEV-27B-VL's published numbers. Add ECE on images; no published reference exists, report it as an absolute number. **Teacher rerun only if 10b failed to reproduce the 9B numbers** [PRO 6000 ×1, 8–12 h, ¥60–85]: JEV-27B-VL BF16 via `serve.sh --max-model-len 131072` on the same sets.
22. `text_parity`: same text question through VL path and text path, diff < 0.02.
23. Confirm the "better than JEV-9B or stop" gate against the 10b numbers.

**Gate:** the section-1 targets. If vision scores fall well short of 95% of JEV-27B-VL, that is the signal to pull P5 (image distillation) forward.

### Week 5 — Hardening, then Orin when available

24. Load test `/v1/decide` on the 4070 with concurrent requests to find the batching ceiling (the fork and bonjev both serve one request at a time). Measure 640×480 and prefix-cached follow-up latency. These are the only latency numbers this round produces.
25. **Optional M5 cross-check, one day:** convert the final LoRA to MLX format, load with the bundled `runtime/` loader, diff 200 rows against the 4070. Only do this if the Mac mini edge-server target is still wanted. Skip otherwise.
26. **Orin NX** (only if procured): cross-compile the fork for aarch64 / sm_87 under JetPack, copy the same GGUF files, repeat the 200-row diff against the 4070, measure 640×480 latency. The 6 s target is checked here and nowhere else.

## 4. Budget

RTX PRO 6000 96GB at ¥6.98/h per card.

| Item | Cards × hours | Cost |
|---|---|---|
| P0 full alignment + kernel check | 1 × 5–6 | ~¥40 |
| P1 smoke | 1 × 3–4 | ~¥30 |
| P2 training, per run | 2 × 22–27 | ¥310–380 |
| P3 teacher rerun (conditional) | 1 × 8–12 | ¥0 or ¥60–85 |
| Data disk expansion to 103 GB, 2 months | — | small |
| **Total, 1 training run** | | **~¥450–550** |
| **Total, 3 training runs** | | **~¥1100–1300** |
| Plus 30% debugging margin | | **≤ ¥1700 (~$240)** |
| Orin NX 16GB dev kit (optional) | — | $600–900 |

Listing valid to 2027-07-01; 2 cards free at time of writing.

## 5. Decisions you need to make before week 1

1. Buy an Orin NX now, or ship P4b as "4070-verified, Orin pending"?
2. Acceptable latency ceiling for one decision (drives Orin NX vs AGX Orin).
3. Do you have the original JEV-9B FP8 baseline result files? If not, step 23 reruns it on the 4070.
4. Is the Mac mini edge-server target still wanted? If not, delete step 25 and the M5 leaves the plan entirely.
