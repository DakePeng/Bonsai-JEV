# Bonsai-JEV

**一个约 9 GB 就能跑、能看图、输出校准概率的 27B 决策模型。** 把 [JEV](https://huggingface.co/autotrust/JEV-27B) 的配方（LoRA + 24 位决策头，用软标签蒸馏）训练在 [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) 上。后者是 PrismML 把 Qwen3.8-27B 压缩成三值（−1/0/+1）的版本。底座保持三值、冻结不动，适配器是单独一个文件。对一个有类型的问题（是/否、0–5 打分、2–16 选 1），带图或不带图，一次前向就输出各选项的校准概率。

[English](README.md)

| | JEV-27B / JEV-27B-VL | **Bonsai-JEV** |
|---|---|---|
| 底座 | Qwen3.8-27B，BF16 | Ternary Bonsai 2 27B，PQ2_0（2.13 bit/权重） |
| 显存中的权重 | 约 54 GB（vLLM，一张 80 GB 卡） | **7.2 GB** 底座 + 0.6 GB 视觉 + 0.46 GB 适配器 |
| 能跑在 | H100 / B200 | RTX 4070 12 GB（已测）；Jetson Orin NX 16 GB 和 Apple Silicon 走分支的后端（尚未带适配器测试） |
| 训练参数 | 1.088 亿 LoRA + 12.3 万决策头 | 同一配方、同样的形状 |

> **独立项目。** Bonsai-JEV 由一位独立开发者完成，与 AutoTrust（autotrust）、TypeSafe、Prism ML 均无隶属关系，也未获得它们的认可或赞助。文中提到它们的名字，只是为了致谢和注明本项目所基于的模型与数据；相关商标归各自所有者。

## 基于

本项目建立在两个发布之上，没有它们就不会有这个项目：

- **[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)，作者 [Prism ML](https://huggingface.co/prism-ml)**（Apache 2.0）。*Created using Bonsai by Prism ML.* 我们原样使用它的三值权重作为冻结底座，使用它的 Hadamard 旋转格式和 `prism.hadamard.*` 元数据、它的视觉投影，以及 [PrismML 的 llama.cpp 分支](https://github.com/PrismML-Eng/llama.cpp)（MIT）作为运行时。Bonsai 2 本身基于阿里云的 Qwen3.8-27B（Apache 2.0）。
- **[JEV-9B](https://huggingface.co/autotrust/JEV-9B) 和 [JEV-27B](https://huggingface.co/autotrust/JEV-27B)，作者 [autotrust](https://huggingface.co/autotrust)**（Apache 2.0）。我们复现了它们的训练配方（LoRA 的挂载位置和秩、24 位决策头及其初始化、KL + RPS 损失、选项打乱、分层批次、优化器和学习率调度），沿用它们的 `bare-v1` 提示词模板和 verbalizer token id、"把决策头表示成 lm_head 上的 LoRA"的部署做法、它们 `serve_decide.py` 的 `/v1/decide` 接口格式，并以它们公开的数字作为目标。[JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL) 的模型卡和 [JEV-27B-DEMO](https://github.com/yuhai-china/JEV-27B-DEMO) 仓库提供了 VL-RewardBench 的评测协议和我们评测时使用的客户端。

Bonsai-JEV 就是把 JEV 的配方用在 Bonsai 2 底座上；两部分思路的功劳都属于各自的作者。

另外感谢 **ChezRD 的 [bonjev](https://github.com/ChezRD/bonjev)**（MIT），一条更早的、不训练的 Ternary Bonsai 决策路线：一个基于 Prism llama.cpp 库的 Rust 引擎，把现成的决策 LoRA 增量组合起来。它公开的 [JevBench-231](https://github.com/fstandhartinger/jevbench) 和 Decision Index 评测协议与成绩，是我们对照的外部基准（`jevbench_eval.py` 的判分方式和它的考试完全一致）。

> **状态（2026-10-15）：已训练完成，并在部署路径（分支 + 三值底座上的 `adapter.gguf`）上评测完毕。** JevBench-231、agent-judge 两项和 Decision Index 还没跑完（标 *待测*）。

## 结果

### 文本决策（[jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) 的 `test_set_30k`）

KL(目标 ‖ 模型)，越低越好，按类型温度校准后。"Jev 标注的行"指 25,376 行 `yuri_v3`，和 JEV 模型卡比最公平。Bonsai-JEV 的数字是在部署路径上测的：PrismML 分支在三值 PQ2_0 文件上跑 `adapter.gguf`。

| 模型 | Jev 标注的行 | 全部行* | noul | score | choice |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 27B 零样本（无适配器，直接读字母 logits） | 0.443 | 0.481 | 0.167 | 0.724 | 0.595 |
| JEV-9B（公开） | 0.019 | 0.021 | 0.005 | 0.023 | 0.028 |
| JEV-27B（公开） | 0.017 | 0.019 | 0.004 | 0.021 | 0.025 |
| **Bonsai-JEV**（第 4,000 步，发布版） | **0.0163** | 0.0202 | 0.0041 | 0.0205 | 0.0245 |

\* 我们的"全部"去掉了 2,260 行标签为占位 `[0.5, 0.5]` 的 `yuri_v1`；JEV 的包含它们。

JEV-27B 模型卡上的其他指标，同样的行、同样的定义（`kl_breakdown.py`）：

| | JEV-27B（公开） | **Bonsai-JEV** |
|---|---:|---:|
| 是/否 AUROC（`noul`），Jev 标注的行 / 全部行 | 0.995 / 0.996 | **0.995 / 0.996** |
| 选择题和 Jev 的 top-1 一致率，Jev 标注的行 | 90.5% | **91.1%** |
| 选择题 top-1，目标前两名相差 ≥ 0.1 的行 | 95.8% | **96.2%** |
| 打分误差（`score`，期望分的 MAE，0–5 分制） | 0.098 | **0.097** |
| ECE，15 个分箱，温度校准后 | 0.0009 | 0.0006 † |
| 拟合温度 noul / score / choice | 1.014 / 1.004 / 1.016 | 1.014 / 1.014 / 1.012 |
| 对真实标签的 KL，没见过的任务族（`ood`） | **0.104** | 0.128 |
| top-1 准确率，没见过的任务族 | 0.942 | **0.948** |
| 选项倒序后选择题 top-1 翻转的比例 | — | 4.3%（零样本底座 65.7%） |

† 我们的 ECE 是把每个选项的预测概率分箱，和目标概率比；JEV 卡没有进一步说明它的 ECE 定义，两者只作参考。

在 Jev 自己的分布上，两个模型分不出差别。唯一的差距是迁移到没见过的任务族（KL 0.128 对 0.104，介于 JEV-9B 的 0.234 和 JEV-27B 之间），top-1 准确率仍然持平。

训练过程中的验证 KL（`validation` 划分的 1,024 行，未做温度校准，噪声约 ±0.003）：

| 步数 | 250 | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,250 | 4,000 |
|---|---|---|---|---|---|---|---|---|---|
| 全部 | 0.070 | 0.047 | 0.035 | 0.045 | 0.025 | 0.023 | 0.020 | 0.019 | **0.018** |
| noul / score / choice | | | | | | | | 0.004 / 0.021 / 0.033 | 0.003 / 0.021 / 0.032 |

### 看图决策（[VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench)，1,247 对）

协议和 [JEV demo](https://github.com/yuhai-china/JEV-27B-DEMO/tree/main/09-multimodal-judge) 相同：两种呈现顺序各问一次取平均。适配器只用文本训练，所以两个模型看图决策都是零样本。

| 模型 | general | hallucination | reasoning | 总分 | 顺序一致率 |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 27B 零样本 | 60.2 | 78.9 | 72.2 | 74.5 | 50.9% |
| JEV-9B（公开） | | | | 74.3 | |
| JEV-27B-VL（公开） | 58.0 | 83.2 | 78.2 | 78.3 | 89% |
| **Bonsai-JEV** | 56.4 | **84.9** | 74.1 | **78.0** | 88.4% |

不取平均、只做一次前向：77.5（原顺序）/ 79.1（换位）。78.0 对 78.3 落在 1,247 对的 ±2.4 置信区间以内，算持平。三值底座在 reasoning 上掉了 4.1 分，hallucination 上高 1.7 分。

### JevBench-231（top-1 准确率，按 bonjev 的判分方式）

| 模型 | 提示词 | 答对 |
|---|---|---:|
| Ternary Bonsai 2 27B 零样本 | bonjev 自己的模板 | 194 / 231（公开） |
| bonjev `vega_clef_plumb`（不训练、合并 LoRA 增量） | bonjev 自己的模板 | 207 / 231（公开） |
| Ternary Bonsai 2 27B 零样本 | JEV `bare-v1`，经 `jevbench_eval.py` 转换 | 166 / 231（easy 47/48、original 58/72、hard 61/111） |
| **Bonsai-JEV** | JEV `bare-v1`，经 `jevbench_eval.py` 转换 | *待测* |

同一个底座，两种提示词的零样本差了 28 题：提示词的渲染方式影响很大。所以比较时既看绝对题数，也看每个适配器相对"同一提示词下零样本"的提升。231 题里 ±6 题以内是噪声。

顺序一致率是把两个回答换位后结论不变的比例。三值底座单看任一种顺序接近抛硬币，平均分把这一点掩盖了，所以每个模型都报这一列。

### 速度和显存（RTX 4070 12 GB，WSL2，PrismML 的 llama.cpp 分支，CUDA 12）

| | 实测 |
|---|---|
| 处理输入 / 生成，PQ2_0 | 1,226 / 49.6 token/秒（`llama-bench` pp512 / tg128） |
| 处理输入 / 生成，PTQ1_0 | 1,315 / 63.1 token/秒 |
| 峰值显存，PQ2_0 + 视觉 + 4k 上下文 | 9.17 GB |
| 文本决策，冷启动，挂适配器（`/v1/decide` 端到端） | p50 0.62 秒，p95 0.68 秒 |
| 一张图 + 两个回答，冷启动 | p50 0.99 秒，p95 1.22 秒 |
| 同一场景预热后的追问 | p50 0.23 秒 |
| **8 GB 显卡**（PTQ1_0 + fp16 适配器 + 视觉，4k 上下文，显存限制在 7.5 GiB） | 文本 p50 0.42 秒，看图 p50 1.04 秒，占用约 7.1 GiB |

**PTQ1_0**（5.95 GB，同一套三值权重的更紧凑打包）挂同一个适配器，决策完全一致：在 3,000 行测试数据上和 PQ2_0 的 top-1 一致率 100%，概率差为 0，而且更快。8 GB 显卡用它。

**并发：** 8 个槽位上 40 个并发决策和逐个发的结果一致（最大概率差 0.0035，没有翻转）；同时混入 8 个聊天请求，文本一致，适配器没有串到聊天。分支只把 LoRA 配置相同的请求放进同一批，所以 JEV-27B-VL 在 vLLM 上"最多 8 个序列"的限制对这里不适用。

12 GB 能开的上下文：f16 KV 约 2.4–3.2 万 token，q8_0 KV 约 4.8–6 万（64 层里只有 16 层有 KV cache；48 层线性注意力每个并发槽位固定约 150 MB 状态）。

## 快速上手（推理服务）

需要从源码带 CUDA 编译 [PrismML 的 llama.cpp 分支](https://github.com/PrismML-Eng/llama.cpp)（原版 llama.cpp 加载不了 PQ2_0；在 Zen 4 等 AVX-512 CPU 上，发布的二进制一加载就崩，见模型的 KNOWN_ISSUES）。

推理所需的全部文件（底座 GGUF、视觉投影、适配器、校准）都在同一个模型仓库里，[Hugging Face](https://huggingface.co/dakebakery/Bonsai-JEV) 和 [ModelScope](https://modelscope.cn/models/dakepeng/Bonsai-JEV) 同步发布（发布时上传）。底座就是 Prism ML 的原文件，和适配器验证时用的那份逐字节一致。

```bash
# 约 8.7 GB，两个来源任选：
modelscope download --model dakepeng/Bonsai-JEV --local_dir ~/bonsai/Bonsai-JEV-weights     # 国内推荐
huggingface-cli download dakebakery/Bonsai-JEV --local-dir ~/bonsai/Bonsai-JEV-weights
# 启动 llama-server 和 /v1/decide 封装
bash run_decide.sh start ~/bonsai/Bonsai-JEV-weights
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "kind": "choice",
  "state": ["摄像头画面：", {"image": "/path/frame.jpg"}, "\n夹爪张开，位于桌面上方。"],
  "question": "机械臂应该先抓哪个物体？",
  "options": ["红色方块", "蓝色方块", "都不抓，场景不安全"]}'
```

`/v1/decide` 的请求和返回格式和 JEV 的 `serve_decide.py` 一致，JEV 的 [`jev_client.py`](https://github.com/yuhai-china/JEV-27B-DEMO/blob/main/common/jev_client.py) 不用改就能用（`JEV_URL=http://localhost:8000 JEV_BACKEND=decide`）。对同一个场景问多个问题时，先 `POST /v1/warm {"state": ..., "kinds": ["noul"]}`：模型有循环层，前缀复用需要一个显式边界，之后每次追问只计算问题本身的 token。

**System 2 就是未改动的底座。** `run_decide.sh` 加载适配器时把全局 scale 设为 0，封装层在每个决策请求上再打开它，所以发给 `llama-server`（8080 端口）的普通聊天/补全请求得到的就是 Ternary Bonsai 2 本身：适配器关闭时 12 个测试回答和不加载适配器的服务逐字节一致（`run_system2.sh`）。适配器打开时有 9 个变了，所以不要把聊天走决策路径。

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

LoRA r=16、α=32、dropout 0.05，挂在全部 64 层的 `in_proj_qkv, in_proj_z, out_proj, q/k/v/o_proj, gate/up/down_proj` 上（1.088 亿参数）。fp32 决策头 5120→24，用 verbalizer token（`false/true`、`0`–`5`、`A`–`P`）在 lm_head 里的行初始化。损失：该类型有效槽位上的 KL(目标 ‖ 模型)，打分题再加 0.5·RPS；30% 的选择题打乱选项顺序。每步 128 行按类型分层，按长度分桶，micro-batch ≤ 2.4 万个填充 token，开梯度检查点。AdamW (0.9, 0.98)，学习率决策头 2e-4 / LoRA 1e-4，3% 预热，余弦退火到 0.02 倍，梯度裁剪 1.0。4,000 步 ≈ JEV-27B 的 5,000 步扣掉它 21% 的占位行（我们直接丢掉 137,203 行 `yuri_v1` 占位数据，而不是给 0.05 权重）。丢掉它们没有可测的代价：noul KL 0.0041，JEV-27B 是 0.004；拟合出的 noul 温度两边都是 1.014。发布的 checkpoint 按验证 KL 选。

### 难点（本仓库已解决）

- **在 PyTorch 里重建三值模型**（`bonsai_torch.py`）：直接从 GGUF 解包 PQ2_0 块；每个折叠权重吃的是 `H(s ⊙ x)`（按 1024 分块的归一化 Walsh–Hadamard，符号向量从 `prism.hadamard.*` 元数据读）；嵌入是查表后逆变换。用分支自己的张量逐层对齐（`tools/dump-tensors`、`align_layer.py`、`align_inner.py`）：64 层在 fp32 下相对差都在 1.8e-3 以内；64 层 BF16 全模型前向在 4.5e-3 以内。只有在这个粒度才看得见的三个 bug：HF 的 RMSNorm 乘的是 `(1 + w)` 而 GGUF 存的已是 `w + 1`；线性注意力的 value 头在 GGUF 里是分支的 tiled 顺序（`gdn_v_grouped`），给 HF 要重排；单独使用的 HF 层不钉死 `_attn_implementation` 会做非因果注意力。
- **LoRA 加在哪个坐标系。** 分支把 `B·A·x` 加在*原始*输入上，而三值矩阵乘法用的是旋转后的输入（`build_lora_mm`）。PyTorch 模型照搬这一点，所以训练出的 LoRA 导出时不需要任何数学变换。
- **不改运行时就能用决策头。** `export_gguf_lora.py` 把决策头写成 `output.weight` 上的 rank-24 LoRA（A = 决策头增量，B = 落到 24 个 verbalizer 行的 one-hot），于是分支在 verbalizer 上的 logits *就是*决策头的 logits；偏置和温度放进 `decision.json`。真实 checkpoint 检查：1,024 行上分支对 PyTorch 平均 KL 0.0002，argmax 一致率 99.0%。
- **冻结的 54 GB 底座上做两卡 DDP：** 冻结底座和 Hadamard 缓冲区不参与 DDP 同步，只同步 LoRA 和决策头的梯度。

## 局限

- **质量天花板是 JEV 水平。** 三值底座只能挂 LoRA；同一个 Qwen3.8-27B 做全量微调的模型在 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) 上到 62–66 分，JEV-27B 是 55.6。这个模型的意义在*内存*，不在超过它们。
- **和 JEV 用同一个教师蒸馏**，文本上和 JEV-27B 持平，不会超过它。迁移到没见过的任务族更弱（OOD KL 0.128 对 0.104）。
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
