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

### 1.3 Positioning against the Jev Decision Index (checked 2026-10-13)

The community leaderboard (149 text entrants, 33 vision entrants, one RTX PRO 6000, chance-corrected, 20/50/30 public/private weighting) changes two things in how this project should be framed.

- **JEV-27B is not the quality frontier.** Text board: vLLM-SR d3 27B 66.3, Decider 31B (NVFP4) 64.6, Perplexity Decider 62.8, Darwin-27B 62.5; JEV-27B 55.6 (rank 27), JEV API 60.1. Vision board: d3 27B 71.3, Perplexity 70.6, **JEV-27B-VL 69.6 (rank 3)**. The top four text entries are full fine-tunes trained on public benchmark train/dev splits; the best LoRA+head entry (Sieve 27B, 60.2) ties JEV. **A frozen ternary base can only take a LoRA, so this project's quality ceiling is JEV level, not d3 level.** Say so in every write-up.
- **The ≤8 GB × 27B-class cell is empty.** The board's params/score front is d3-mini 4B (52.6 text / 63.2 vision, ~9 GB BF16) and d3-flash 9B (55.8 / 66.9, ~17 GB). Reproducing JEV-27B-VL at ~7 GB beats the 4B on both boards and the 9B on vision; a 4-bit 9B (~6 GB) would tie on text. Latency is not our axis (3× the compute of a 9B). The pitch is "27B decision quality in 8 GB", nothing else.
- **Quantized and llama.cpp-served entries are accepted**: Decider NVFP4 passed the ≥96% decision-agreement bar vs BF16 (96.99%); Winnow entries run Q8_0 GGUF on llama.cpp. No ternary or sub-10 GB 27B entry exists. Step 21 should also run the public DI suite so the model can be submitted; expect the agreement check against a BF16 reference.

**Four things borrowed from the board, now in the steps below:** conditional second reading with reversed options (Decider 31B), option-count-dependent temperature (Decider), public benchmark train splits as extra data (Darwin/Decider/Torchcast, leakage-filtered), and a DI public-suite run for submission. Rotation averaging + label-free prior (AnyJev L0, 60.2 with no training) is the fallback if training underdelivers on order consistency.

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
   **Text, IN PROGRESS 2026-10-12.** Method pinned to JEV's `bare-v1` template from `common/jev_client.py` and `judge_config.json`: `[kind] k\n[state] s\n[question] q\n[options]\n<lines>\n[decision]:`, no special tokens, option lines `false`/`true`, `0`–`5`, `A) …`; verbalizer ids 3721, 1802, 15–20, 32–47 (verified identical in the Qwen3.8 tokenizer); softmax over the kind's active ids, no bias, temperature 1. Tooling: `tools/decide-logits` (fork example, next-token logits for given ids, ~0.18 s/row on the 4070) + `zeroshot_text.py` (KL, accuracy, ECE per kind and source; per-row probs saved for the later trained-vs-base comparison). `yuri_v1` rows carry placeholder `[0.5,0.5]` labels and are excluded. **Full result, 27,695 rows (all non-placeholder rows of `test_set_30k`), 2026-10-12:**

   | subset | n | KL | acc | ECE |
   |---|---:|---:|---:|---:|
   | all | 27,695 | **0.481** | 49.0% | 0.036 |
   | noul | 9,969 | 0.167 | 76.6% | 0.058 |
   | choice | 9,199 | 0.595 | 49.4% | 0.088 |
   | score | 8,527 | 0.724 | 16.4% | 0.074 |
   | src: yuri_v3 (Jev-labelled) | 25,376 | 0.443 | 48.5% | 0.042 |
   | src: openjev_v2 | 2,319 | 0.888 | 55.1% | 0.119 |

   This is the zero-shot floor: JEV-27B's trained KL is 0.017, JEV-9B's 0.019, i.e. 25× lower. The 600-row sample predicted it within noise. Scores are near chance (6-way), so the head and LoRA carry nearly all of the score-task skill.

   **VL-RewardBench zero-shot (no LoRA, no head), demo protocol (both orders averaged), 1,247 pairs, 2026-10-12:**

   | | general | hallucination | reasoning | **overall** | macro | order-agree |
   |---|---:|---:|---:|---:|---:|---:|
   | Bonsai 2 27B zero-shot | 60.2 | 78.9 | 72.2 | **74.5** | 70.5 | 50.9 |
   | JEV-9B (published) | | | | 74.3 | | |
   | JEV-27B-VL (published) | 58.0 | 83.2 | 78.2 | 78.3 | 73.1 | 89 |

   Reading: the ternary base's vision understanding is intact (the plan's "三值化损伤了视觉理解" risk is not showing), and the averaged protocol already lands at JEV-9B level. But order-agree 50.9% vs JEV's 89% says the raw base has a dominant position bias: each single order is nearly a coin flip, and the accuracy comes from the small residual that survives averaging. Training the head/LoRA should mostly fix the bias; single-order accuracy and order-agree are the numbers to watch in P3, not just the averaged overall. ~0.7 s per decision on the 4070 (300 image tokens + ~250 text tokens).
10b. **Prepared 2026-10-12, queued after 10 in `run_step10.sh`.** `run_10b.sh` serves the multimodal Qwen3.5-9B base + `vl/adapter_vllm` through JEV-9B's own `vl/serve_decide.py` (vLLM 0.31, FP8, 4k context, eager, 8 seqs) and `jev9b_eval.py` runs text + VL-RewardBench through the demo's unmodified client (`JEV_BACKEND=decide`). Weights from ModelScope (JEV-9B 18 GB, Qwen3.5-9B 19 GB; note the 9B serve script pins base revision c202236, ModelScope gave latest). **Outcome 2026-10-12: does not fit the 4070.** vLLM 0.31 with `--quantization fp8` still loaded 11.03 GiB (embeddings, lm_head and the vision tower stay BF16), leaving −2.2 GiB for KV cache. Moved to the first rental session: `QUANT=fp8 bash run_10b.sh` and `QUANT=none` for a BF16 reference, ~¥15 total, bundled with steps 8, 11 and 14. The 4070 result files stay zero-shot only.
    **JEV-9B FP8 on the 4070, two jobs in one.** vLLM FP8, `--max-model-len 8192`, using the JEV-27B-DEMO eval code unchanged. (a) Lower-bound gate: KL, ECE, robot-arm demo success. (b) **Harness validation:** if it reproduces the published 74.3% VL-RewardBench and ~0.019 KL, the eval code is trusted and JEV-27B-VL's published 78.3% / AUROC 0.91 / KL 0.017 are used directly as targets. No teacher rerun needed.
11. **Hybrid-attention training kernels** [same session]: confirm PyTorch and Triton see sm_120, then run one forward+backward on the PyTorch model with LoRA attached and gradient checkpointing, micro-batch 24k tokens. Record step time and peak memory. This confirms the P2 time estimate.

**Gate:** logits match; LoRA placement understood; step time measured. **Stop if logits cannot be aligned after the fallback (unrotate weights with H, train plain, re-rotate A on export).**

### Week 2 — P1 trainer

12. **DONE 2026-10-12 (code + CPU smoke).** `train.py`: frozen Bonsai base via `bonsai_torch.build_model` (text model without lm_head, ~51 GB BF16), peft LoRA r16/α32/dropout 0.05 on the 10 JEV target modules (fp32 LoRA params, bf16 autocast), fp32 24-slot head with bias initialised from the un-rotated lm_head rows (`decision_head_rows`), KL + 0.5·RPS, 30% choice shuffle, stratified 128-row steps, length-bucketed micro-batches under a token budget, gradient checkpointing, AdamW (0.9, 0.98) with the two LRs, cosine + 3% warmup, clip 1, `torchrun` DDP for 2 cards, checkpoint/resume (`last/`, `final/`: peft adapter + `head.safetensors` + `head.json`), CSV log, periodic validation KL by kind. Smoke-tested on the 4070 box **on CPU** with a 2-layer truncated model (`--layers 2 --device cpu`): loss falls, eval/save/resume work. 507k usable train rows → ~3,700 steps for 0.93 epoch.
    Original spec: write `train.py` on top of `peft`: 24-slot fp32 decision head initialised from `lm_head` rows; KL(target‖model) over active slots + 0.5·RPS for score; 30% option-shuffle augmentation for choice; stratified sampler (≥1/6 per kind); length bucketing; AdamW (0.9, 0.98), head LR 2e-4, LoRA 1e-4, cosine, 3% warmup, clip 1.0. Drop `yuri_v1` rows with [0.5, 0.5] placeholders.
13. **DONE 2026-10-12.** `infer_logits.py` (checkpoint → 24-slot logits per row, same JSONL as `decide-logits`), `calibrate.py` (per-kind temperature by L-BFGS minimising KL on `calibration`, clamped to [0.1, 10], writes JEV-shaped `calibration.json` with before/after KL and ECE), `eval_text.py` (KL, accuracy, ECE per kind and source, optional temperatures, works on zero-shot and trained logits alike). Shared `load_split()` keeps row filtering and order identical across all scripts. Tested on the 600-row zero-shot sample (fitted T: noul 1.15, score 2.22, choice 1.62 → the base is overconfident; KL 0.463 → 0.441) and on CPU with the 2-layer smoke checkpoint end to end.
    Original spec: `calibrate.py` (L-BFGS temperature per kind on `calibration` split) and `eval_text.py` (KL, ECE, accuracy on `test_set_30k` and `ood`).
14. **Smoke test on 27B, 1% of train** [PRO 6000 ×1, 3–4 h, ~¥30]. Loss must decrease; head-only variant first (LoRA frozen) as a sanity check, then full. Save a checkpoint and run `eval_text.py` on it.
15. **Export written and validated 2026-10-12 (synthetic end-to-end test PASSED on the 4070: LoRA'd layer 0 (linear attn) 6.1e-4, layer 1 4.2e-4, head logits 1.5e-3 with the head delta contributing 0.91 on average, i.e. 600× the residual).** The remaining part of this step, PyTorch-vs-fork on 200 real rows with a real checkpoint, runs after the first rented smoke (step 14). `export_gguf_lora.py` writes one GGUF LoRA the fork loads with `--lora`: metadata `general.type=adapter`, `adapter.type=lora`, `adapter.lora.alpha`; tensors `<gguf name>.lora_a` (r × n_in) / `.lora_b` (n_out × r), runtime scale α/r. Layout fixes on the way out: v-head rows of `attn_qkv.lora_b` and `attn_gate.lora_b` and v-head columns of `ssm_out.lora_a` go back from HF grouped to the fork's tiled order. The decision head becomes a rank-24 LoRA on `output.weight` (A = (W_head − W_init)·24/α so the runtime scale cancels, B = one-hot at the 24 verbalizer rows), exactly JEV's "head as lm_head LoRA" trick, so the fork's logits at the verbalizer ids are the head's logits; the head bias and temperatures go to `decision.json` for the serving wrapper. The fork's loader was read to confirm shapes and that `output.weight` accepts a LoRA. `test_export.py` validates all of it without a trained model: random LoRA on layers 0–1 + random head delta → export → fork dump with `--lora` → compare against PyTorch layers with the same LoRA and against (W_init+Δ)·result_norm.
    Original spec: convert that checkpoint to GGUF LoRA plus a head matrix, load in the fork on the 4070, compare PyTorch vs fork probabilities on 200 rows.

**Gate:** KL falls on smoke; PyTorch and the fork agree within 0.01 on the smoke checkpoint.

### Week 3 — P2 training + P4 deploy on the 4070 in parallel

16. **Full run** [PRO 6000 ×2, 22–27 h, ¥310–380]: 0.93 epoch, ~4,750 steps, data-parallel over 2 cards. Checkpoint every 250 steps to the volume. Log loss per kind and LR to a CSV; plot against the JEV-9B README curve.
17. **DONE 2026-10-12 (zero-shot mode; the trained adapter is a `--lora` flag away).** `decide_server.py` (FastAPI) over the fork's `llama-server`: same `POST /v1/decide` request/response as JEV's `serve_decide.py`, so `jev_client.py` and `jev9b_eval.py` evaluate our model unchanged (verified: demo client vs offline tool agree on 300 rows). Reads the kind's verbalizer logprobs from `/completion` `n_probs=1000` (full-vocab softmax, same semantics as JEV's `allowed_token_ids`), adds the head bias and per-kind temperature from `decision.json`. The head itself rides inside the LoRA file (step 15), so no custom readout in the fork. `run_decide.sh start [ADAPTER_DIR]` launches both processes detached (`-np 2 --cache-ram 16384 --cache-reuse 256`, this fork randomises its media marker per run and the wrapper reads it from `/props`). `test_decide.py` PASSES: text 50 rows max |Δp| 1.25e-2 and image 5 pairs 5.2e-3 vs the offline tools (the fork's kernels differ across batch sizes; repeated requests are bit-identical), and prefix reuse works.
    **Prefix caching needs a warm-up.** The model has recurrent layers, so the server can only resume from saved boundaries (checkpoints, default spacing 8192 tokens): a cold `state+question` prompt is not partially reusable for the next question. `POST /v1/warm {state, kinds}` pre-processes `[kind] k\n[state] …`; afterwards each `/v1/decide` on that state evaluates only its question (22–26 tokens, 189 image+scene tokens cached), **~0.2 s per follow-up on the 4070** vs 0.5–0.9 s cold. Caveat from JEV's template: `[kind]` is the first token, so a state warmed for `noul` is not a prefix of a `choice` prompt; warm per kind. Stop with `run_decide.sh stop` before other GPU work (it holds ~9 GB).
18. If the first run's KL on `test_set_30k` is above 0.03, do one more run with longer schedule or LR tweak. Budget allows three runs total. **Optional extra data for a later run (borrowed from the board's top entries):** train/dev splits of public decision benchmarks (CLINC150, GSM8K, ARC, BoolQ, MMLU, NLI, iSarcasmEval, VAST…) rendered in the bare-v1 format with gold labels, leakage-filtered against every test suite we report on. It lifts the public DI score; for the robot/edge product the JEV corpus plus business data matter more, so it stays behind them in priority.

**Gate:** training KL converges; 4070 `/v1/decide` matches training-side probabilities within 0.01.

### Week 4 — P3 calibration and eval

19. Fit temperatures. If any is far from 1.0, treat as a training bug, not a calibration fix. **Borrowed from Decider 31B:** fit the temperature as T(n) = a + b·ln n per kind (n = number of options; three parameters per kind, same L-BFGS in `calibrate.py`) and report whether it beats the constant; and add a **conditional second reading** to `/v1/decide`: if the top probability is below 0.7, reverse the options, read again, average the two log-softmaxes, then apply T2(n). Decider reports this touches ~28% of questions, so cost rises ~30%, not 2×. Measure order-agree with and without it; this is the cheap fix for the zero-shot position bias (51%).
20. Text evals on the 4070: KL, ECE, accuracy on `test_set_30k` and `ood`. Compare against JEV-27B (0.017) and JEV-9B (0.019) and the zero-shot floor from step 10. **Also run the Jev Decision Index public suite** (its kit, same `/v1/decide` protocol) so the model has a board score next to JEV-27B's 55.6 / JEV-27B-VL's 69.6, and keep the BF16 PyTorch model's decisions on the same rows for the ≥96% agreement check the board applies to quantized entries.
21. **Vision evals on the 4070:** run our model on VL-RewardBench, AgentRewardBench, MMRB2, QuickDraw with the validated harness from 10b. Compare to JEV-27B-VL's published numbers. Add ECE on images; no published reference exists, report it as an absolute number. **Teacher rerun only if 10b failed to reproduce the 9B numbers** [PRO 6000 ×1, 8–12 h, ¥60–85]: JEV-27B-VL BF16 via `serve.sh --max-model-len 131072` on the same sets.
22. `text_parity`: same text question through VL path and text path, diff < 0.02.
23. Confirm the "better than JEV-9B or stop" gate against the 10b numbers.

**Gate:** the section-1 targets. If vision scores fall well short of 95% of JEV-27B-VL, that is the signal to pull P5 (image distillation) forward.

### Week 5 — Hardening, then Orin when available

24. Load test `/v1/decide` on the 4070 with concurrent requests to find the batching ceiling (the fork and bonjev both serve one request at a time). Measure 640×480 and prefix-cached follow-up latency. These are the only latency numbers this round produces.
25. **Optional M5 cross-check, one day:** convert the final LoRA to MLX format, load with the bundled `runtime/` loader, diff 200 rows against the 4070. Only do this if the Mac mini edge-server target is still wanted. Skip otherwise.
26. **Orin NX** (only if procured): cross-compile the fork for aarch64 / sm_87 under JetPack, copy the same GGUF files, repeat the 200-row diff against the 4070, measure 640×480 latency. The 6 s target is checked here and nowhere else.

### P5 — Image decision round (after P4, needs the text-trained model as the control)

**Why.** JEV-27B-VL's image decisions are zero-shot: its LoRA and head saw text only. Our zero-shot base already scores 74.5% on VL-RewardBench but with a 50.9% order-agree rate (position bias). Image decision data is the one lever JEV never pulled, and the only realistic way to beat it. Text cannot exceed JEV-27B (same teacher labels); the P5 target is the vision side.

**Data, three sources that stack.**
1. Public human preference sets over images: RLAIF-V, RLHF-V, VLFeedback, POVID (tens of thousands of pairs, hallucination-centred, matching VL-RewardBench's largest subset). Never train on VL-RewardBench or MMRB2 themselves.
2. Teacher labels from JEV-27B-VL at its *best effort*: both orders averaged, System 2 thinking enabled when System 1 is uncertain, then read the distribution. The student learns the teacher's ceiling, so a single pass can beat the teacher's single pass. One H100/PRO 6000 session, ~¥100 for ~50k images.
3. Business images (the original P5), labelled the same way.

**Recipe changes (small).**
- `bonsai_torch.build_model` gains the frozen Qwen3.8-27B vision tower (BF16, Bonsai leaves it unquantized), run under `no_grad`; `train.py` renders image rows with the same bare-v1 template and `<|vision_start|><|image_pad|><|vision_end|>` placeholders, mixing image and text rows with the same stratified sampler so text KL does not regress. Both presentation orders of each pair are training rows (that is what removes the position bias).
- Memory: +0.9 GB weights, no optimizer state, image tokens cost the same per-token activation as text → ~81 GB, still fits the PRO 6000. A 640×480 image is ~300 tokens, so a 24k-token micro-batch holds ~48 image rows instead of ~100 text rows. Optional: precompute image embeddings once (3 MB per image) if the set is under ~20k images and disk allows.
- Cost: 50k image rows × ~500 tokens ≈ 25M tokens per epoch, under half of a text epoch → ~¥100–150 per run on 2× PRO 6000.
- Eval: VL-RewardBench and MMRB2 with the step-10 harness, reporting **single-order accuracy and order-agree**, not only the averaged overall; text KL on `test_set_30k` must stay within 0.005 of the P3 result.

**Targets.** VL-RewardBench overall ≥ 80 (JEV-27B-VL 78.3; the 1,247-pair CI is ±2.4, so claim a win only with ≥3 points and MMRB2 moving the same way), order-agree ≥ 90%, hallucination ≥ 85, text KL unchanged.

**Gate before starting.** P3 finished with a text-trained model and its numbers recorded, so P5's gains are attributable.

**Joint training is the end state.** P5 itself is a mixed run (text and image rows in one stratified sampler). The text-only run before it exists only the first time, as the control and as the trainer's validation against JEV's published curve. Every later retrain (business images, new data mix) is a single joint run from scratch, one round, no separate text stage.

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
