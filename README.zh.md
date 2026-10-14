# Bonsai-JEV-27B

**一个约 9 GB 就能跑、能看图、输出校准概率的 27B 决策模型。** 把 [JEV](https://huggingface.co/autotrust/JEV-27B) 的配方（LoRA + 24 位决策头，用软标签蒸馏）训练在 [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) 上。后者是 PrismML 把 Qwen3.8-27B 压缩成三值（−1/0/+1）的版本。底座保持三值、冻结不动，适配器是单独一个文件。对一个有类型的问题（是/否、0–5 打分、2–16 选 1），带图或不带图，一次前向就输出各选项的校准概率。

[English](README.md)

| | JEV-27B / JEV-27B-VL | **Bonsai-JEV-27B** |
|---|---|---|
| 底座 | Qwen3.8-27B，BF16 | Ternary Bonsai 2 27B，PQ2_0（2.13 bit/权重） |
| 显存中的权重 | 约 54 GB（vLLM，一张 80 GB 卡） | **7.2 GB** 底座 + 0.6 GB 视觉 + 0.46 GB 适配器 |
| 能跑在 | H100 / B200 | RTX 4070 12 GB（已测）；Jetson Orin NX 16 GB 和 Apple Silicon 走分支的后端（尚未带适配器测试） |
| 训练参数 | 1.088 亿 LoRA + 12.3 万决策头 | 同一配方、同样的形状 |

> **状态（2026-10-14）：训练进行中。** 下面是验证集 KL；测试集数字、发布的适配器和看图评测在训练结束后补上。标 *待测* 的还没有测。

## 结果

### 文本决策（[jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) 的 `test_set_30k`）

KL(目标 ‖ 模型)，越低越好，按类型温度校准后。"Jev 标注的行"指 25,376 行 `yuri_v3`，和 JEV 模型卡比最公平。

| 模型 | Jev 标注的行 | 全部行* | noul | score | choice |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 27B 零样本（无适配器，直接读字母 logits） | 0.443 | 0.481 | 0.167 | 0.724 | 0.595 |
| JEV-9B（公开） | 0.019 | 0.021 | 0.005 | 0.023 | 0.028 |
| JEV-27B（公开） | 0.017 | 0.019 | | | |
| **Bonsai-JEV-27B** | *待测* | *待测* | *待测* | *待测* | *待测* |

\* 我们的"全部"去掉了 2,260 行标签为占位 `[0.5, 0.5]` 的 `yuri_v1`；JEV 的包含它们。

训练过程中的验证 KL（`validation` 划分的 1,024 行，未做温度校准，噪声约 ±0.003）：

| 步数 | 250 | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,250 | 4,000 |
|---|---|---|---|---|---|---|---|---|---|
| 全部 | 0.070 | 0.047 | 0.035 | 0.045 | 0.025 | 0.023 | 0.020 | **0.019** | *待测* |
| 第 3,250 步的 noul / score / choice | | | | | | | | 0.004 / 0.021 / 0.033 | |

### 看图决策（[VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench)，1,247 对）

协议和 [JEV demo](https://github.com/yuhai-china/JEV-27B-DEMO/tree/main/09-multimodal-judge) 相同：两种呈现顺序各问一次取平均。

| 模型 | general | hallucination | reasoning | 总分 | 顺序一致率 |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 27B 零样本 | 60.2 | 78.9 | 72.2 | 74.5 | 50.9% |
| JEV-9B（公开） | | | | 74.3 | |
| JEV-27B-VL（公开） | 58.0 | 83.2 | 78.2 | 78.3 | 89% |
| **Bonsai-JEV-27B** | *待测* | *待测* | *待测* | *待测* | *待测* |

顺序一致率是把两个回答换位后结论不变的比例。三值底座单看任一种顺序接近抛硬币，平均分把这一点掩盖了，所以每个模型都报这一列。

### 速度和显存（RTX 4070 12 GB，WSL2，PrismML 的 llama.cpp 分支，CUDA 12）

| | 实测 |
|---|---|
| 处理输入 / 生成 | 1,226 / 49.6 token/秒（`llama-bench` pp512 / tg128） |
| 峰值显存，模型 + 视觉 + 4k 上下文 | 9.17 GB |
| 文本决策（100–300 token） | 零样本 0.1–0.35 秒；挂适配器预计慢 10–30%（*待测*） |
| 一张 640×480 图的决策，冷启动 | 0.5–0.9 秒（视觉编码约 0.12 秒） |
| 同一场景预热后的追问 | 约 0.2 秒 |

12 GB 能开的上下文：f16 KV 约 2.4–3.2 万 token，q8_0 KV 约 4.8–6 万（64 层里只有 16 层有 KV cache；48 层线性注意力每个并发槽位固定约 150 MB 状态）。

## 快速上手（推理服务）

需要从源码带 CUDA 编译 [PrismML 的 llama.cpp 分支](https://github.com/PrismML-Eng/llama.cpp)（原版 llama.cpp 加载不了 PQ2_0；在 Zen 4 等 AVX-512 CPU 上，发布的二进制一加载就崩，见模型的 KNOWN_ISSUES）。

```bash
# 1. 底座（约 8 GB），Hugging Face 或 ModelScope
modelscope download --model prism-ml/Ternary-Bonsai-2-27B-gguf \
  --include "Ternary-Bonsai-2-27B-PQ2_0.gguf" "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf" --local_dir ~/bonsai/models/bonsai2-27b
# 2. 适配器：adapter.gguf + decision.json（链接待补）放到 ~/bonsai/adapter
# 3. 启动 llama-server 和 /v1/decide 封装
bash run_decide.sh start ~/bonsai/adapter
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "kind": "choice",
  "state": ["摄像头画面：", {"image": "/path/frame.jpg"}, "\n夹爪张开，位于桌面上方。"],
  "question": "机械臂应该先抓哪个物体？",
  "options": ["红色方块", "蓝色方块", "都不抓，场景不安全"]}'
```

`/v1/decide` 的请求和返回格式和 JEV 的 `serve_decide.py` 一致，JEV 的 [`jev_client.py`](https://github.com/yuhai-china/JEV-27B-DEMO/blob/main/common/jev_client.py) 不用改就能用（`JEV_URL=http://localhost:8000 JEV_BACKEND=decide`）。对同一个场景问多个问题时，先 `POST /v1/warm {"state": ..., "kinds": ["noul"]}`：模型有循环层，前缀复用需要一个显式边界，之后每次追问只计算问题本身的 token。

**System 2 就是未改动的底座。** `run_decide.sh` 加载适配器时把全局 scale 设为 0，封装层在每个决策请求上再打开它，所以发给 `llama-server`（8080 端口）的普通聊天/补全请求得到的就是 Ternary Bonsai 2 本身：适配器关闭时 12 个测试回答和不加载适配器的服务逐字节一致（`run_system2.sh`）。适配器打开时有 5 个变了，所以不要把聊天走决策路径。

## 复现训练

一次训练：2× RTX PRO 6000 96 GB 约 27 小时（AutoDL 上约 ¥400），外加在 12 GB 卡上几小时的检查。

```bash
bash rental/bootstrap.sh            # 租卡机器上装环境、下载（复用镜像自带的 CUDA torch）
bash rental/check.sh                # 27B 前向加反向（单卡每步 48–59 秒，峰值 87 GB）、全模型对齐
bash rental/smoke.sh                # 只训决策头和全配方各在 1% 数据上冒烟，导出
torchrun --nproc_per_node 2 train.py --out ckpt/run1 --steps 4000
python infer_logits.py --ckpt ckpt/run1/best --split calibration --out results/calib.logits.jsonl
python calibrate.py   --logits results/calib.logits.jsonl --out ckpt/run1/best/calibration.json
python export_gguf_lora.py --ckpt ckpt/run1/best --out ckpt/run1/best/adapter.gguf --calibration ckpt/run1/best/calibration.json
```

完整操作手册在 `rental/session1.md`，每一步的测量和决定都记在 `execution-plan.zh.md`。

### 配方（按 JEV-27B 模型卡对齐）

LoRA r=16、α=32、dropout 0.05，挂在全部 64 层的 `in_proj_qkv, in_proj_z, out_proj, q/k/v/o_proj, gate/up/down_proj` 上（1.088 亿参数）。fp32 决策头 5120→24，用 verbalizer token（`false/true`、`0`–`5`、`A`–`P`）在 lm_head 里的行初始化。损失：该类型有效槽位上的 KL(目标 ‖ 模型)，打分题再加 0.5·RPS；30% 的选择题打乱选项顺序。每步 128 行按类型分层，按长度分桶，micro-batch ≤ 2.4 万个填充 token，开梯度检查点。AdamW (0.9, 0.98)，学习率决策头 2e-4 / LoRA 1e-4，3% 预热，余弦退火到 0.02 倍，梯度裁剪 1.0。4,000 步 ≈ JEV-27B 的 5,000 步扣掉它 21% 的占位行（我们直接丢掉 137,203 行 `yuri_v1` 占位数据，而不是给 0.05 权重）。发布的 checkpoint 按验证 KL 选。

### 难点（本仓库已解决）

- **在 PyTorch 里重建三值模型**（`bonsai_torch.py`）：直接从 GGUF 解包 PQ2_0 块；每个折叠权重吃的是 `H(s ⊙ x)`（按 1024 分块的归一化 Walsh–Hadamard，符号向量从 `prism.hadamard.*` 元数据读）；嵌入是查表后逆变换。用分支自己的张量逐层对齐（`tools/dump-tensors`、`align_layer.py`、`align_inner.py`）：64 层在 fp32 下相对差都在 1.8e-3 以内；64 层 BF16 全模型前向在 4.5e-3 以内。只有在这个粒度才看得见的三个 bug：HF 的 RMSNorm 乘的是 `(1 + w)` 而 GGUF 存的已是 `w + 1`；线性注意力的 value 头在 GGUF 里是分支的 tiled 顺序（`gdn_v_grouped`），给 HF 要重排；单独使用的 HF 层不钉死 `_attn_implementation` 会做非因果注意力。
- **LoRA 加在哪个坐标系。** 分支把 `B·A·x` 加在*原始*输入上，而三值矩阵乘法用的是旋转后的输入（`build_lora_mm`）。PyTorch 模型照搬这一点，所以训练出的 LoRA 导出时不需要任何数学变换。
- **不改运行时就能用决策头。** `export_gguf_lora.py` 把决策头写成 `output.weight` 上的 rank-24 LoRA（A = 决策头增量，B = 落到 24 个 verbalizer 行的 one-hot），于是分支在 verbalizer 上的 logits *就是*决策头的 logits；偏置和温度放进 `decision.json`。真实 checkpoint 检查：1,024 行上分支对 PyTorch 平均 KL 0.0002，argmax 一致率 99.0%。
- **冻结的 54 GB 底座上做两卡 DDP：** 冻结底座和 Hadamard 缓冲区不参与 DDP 同步，只同步 LoRA 和决策头的梯度。

## 局限

- **质量天花板是 JEV 水平。** 三值底座只能挂 LoRA；同一个 Qwen3.8-27B 做全量微调的模型在 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) 上到 62–66 分，JEV-27B 是 55.6。这个模型的意义在*内存*，不在超过它们。
- **和 JEV 用同一个教师蒸馏**，文本上只能逼近 JEV-27B，不能超过。`choice` 最难（第 3,250 步 0.033，JEV-9B 是 0.028）。
- **三值化对视觉的伤害大于文本**（Bonsai 自己的模型卡：MMMU-Pro 81.7 → 75.5）。看图决策和 JEV-27B-VL 一样是零样本；计划做一轮图片决策训练。
- **延迟：** 一次决策要做一次 27B 的前向。Jetson Orin NX 上一次看图决策预计几秒。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `bonsai_torch.py` | PQ2_0 解包、Hadamard 模块、GGUF → HF 加载、`build_model`、决策头初始化 |
| `train.py` | JEV 配方；DDP、原子化 checkpoint、按验证 KL 保留 `best/` |
| `infer_logits.py`、`calibrate.py`、`eval_text.py` | 24 位 logits、按类型温度（L-BFGS）、KL/准确率/ECE |
| `export_gguf_lora.py` | peft checkpoint → 分支用的单个 GGUF LoRA + `decision.json` |
| `decide_server.py`、`run_decide.sh` | 套在 `llama-server` 上的 `/v1/decide` 和 `/v1/warm` |
| `zeroshot_text.py`、`zeroshot_vlrb.py` | 零样本基线；JEV 的 bare-v1 提示词和 verbalizer id |
| `tools/dump-tensors`、`tools/decide-logits`、`tools/decide-logits-mm` | llama.cpp 小工具（拷进分支的 `examples/`） |
| `align_layer.py`、`align_inner.py`、`test_export.py`、`check_export_real.py`、`test_decide.py`、`test_system2.py` | 数值检查 |
| `rental/` | 租卡环境脚本、操作手册、ModelScope 并行下载、checkpoint 同步 |
| `execution-plan.md` / `.zh.md` | 完整计划、测量和决定 |

## 许可证和署名

本仓库的代码和适配器权重：**Apache 2.0**（见 `LICENSE` 和 `NOTICE`）。

- **Created using Bonsai by Prism ML.** Ternary Bonsai 2 27B 为 Apache 2.0，© Prism ML, Inc.，基于 Qwen3.8-27B（© 阿里云，Apache 2.0）。本仓库不再分发底座权重。
- [PrismML 的 llama.cpp 分支](https://github.com/PrismML-Eng/llama.cpp)：MIT，© the ggml authors。`tools/` 下的文件针对它编写。
- 训练配方、提示词模板和 verbalizer 沿用 [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) 和 [JEV-27B](https://huggingface.co/autotrust/JEV-27B)（Apache 2.0）。
- 训练数据为 [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)（Apache 2.0；`openjev_v2` 部分源头为 CC0）。JEV-9B 和 JEV-27B 用的也是这份语料，并以 Apache 2.0 发布。其中 `yuri_v3` 的标签（占语料三分之二）是 TypeSafe 的 Jev 1.13 的输出分布；和 JEV 一样，计划商用的使用者可以自行查看 TypeSafe 的 API 条款。
- VL-RewardBench 只用于评测（其卡片声明仅限研究用途），既不再分发，也不用于训练。

## 致谢

PrismML 提供了 Ternary Bonsai 和 llama.cpp 分支；autotrust 公开了 JEV 的配方、适配器和 demo；SargeDev 提供了蒸馏语料；Qwen 团队提供了 Qwen3.8；以及 Jev Decision Index 的维护者。
