---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-classification
base_model:
- prism-ml/Ternary-Bonsai-2-27B-gguf
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

**8 GB 显卡就能跑、能看图、输出校准概率的 27B 决策模型。** 把 [JEV](https://modelscope.cn/models/autotrust/JEV-27B) 的配方（LoRA + 24 位决策头，用软标签蒸馏）训练在 [Ternary Bonsai 2 27B](https://modelscope.cn/models/prism-ml/Ternary-Bonsai-2-27B-gguf) 上。后者是 Prism ML 把 Qwen3.8-27B 压缩成三值的版本。对一个有类型的问题（是/否、0–5 打分、2–16 选 1）提问，状态里可以带图，一次前向就输出各选项的校准概率，不生成文字。

**代码、训练配方和评测脚本：** [github.com/DakePeng/Bonsai-JEV](https://github.com/DakePeng/Bonsai-JEV) · 英文模型卡：[Hugging Face](https://huggingface.co/dakebakery/Bonsai-JEV)

同一套权重也提供 **System 2**：把适配器关掉，服务就是 Ternary Bonsai 2 本身，逐字节一致。

> **状态（2026-10-15）：** 已训练完成；下面的数字都是在部署路径上测的（分支在三值文件上跑 `adapter.gguf`）。Decision Index 还没跑。

[English model card on Hugging Face](https://huggingface.co/dakebakery/Bonsai-JEV)

> **独立项目。** Bonsai-JEV 由一位独立开发者完成，与 AutoTrust（autotrust）、TypeSafe、Prism ML 均无隶属关系，也未获得它们的认可或赞助。文中提到它们的名字，只是为了致谢和注明本项目所基于的模型与数据；相关商标归各自所有者。

## 基于

- **[Ternary Bonsai 2 27B](https://modelscope.cn/models/prism-ml/Ternary-Bonsai-2-27B-gguf)，作者 Prism ML**（Apache 2.0）。*Created using Bonsai by Prism ML.* 它的三值权重是冻结底座，原样不动，本仓库附带一份逐字节一致的副本；它的 [llama.cpp 分支](https://github.com/PrismML-Eng/llama.cpp)是运行时。Bonsai 2 基于阿里云的 Qwen3.8-27B（Apache 2.0）。
- **[JEV-9B](https://huggingface.co/autotrust/JEV-9B) 和 [JEV-27B](https://huggingface.co/autotrust/JEV-27B)，作者 autotrust**（Apache 2.0）。训练配方、提示词模板、verbalizer、"把决策头表示成 lm_head 上的 LoRA"的部署方式和 `/v1/decide` 接口都来自它们；它们公开的数字是我们的目标。

## 一览

| | JEV-27B / JEV-27B-VL | **Bonsai-JEV** |
|---|---|---|
| 底座 | Qwen3.8-27B，BF16 | Ternary Bonsai 2 27B，PQ2_0（2.13 bit/权重） |
| 下载大小 | 约 54 / 56 GB | **约 8.7 GB**（PQ2_0）或约 7.4 GB（PTQ1_0） |
| 能跑在 | 一张 80 GB 以上的卡（vLLM） | **8 GB 显卡**，用 PTQ1_0（带视觉和 4k 上下文峰值约 7.1 GiB；在 RTX 4070 上把显存限制到 7.5 GiB 模拟）；12 GB 显卡（RTX 4070 已测）；16 GB 的 Apple Silicon 走 MLX（Mac mini M4 已测，用 Prism 的 MLX 2-bit 包）；AMD Radeon 780M 核显走 Vulkan（已测）；Jetson Orin NX 16 GB 尚未测试 |
| 训练参数 | 1.088 亿 LoRA + 12.3 万决策头 | 同一配方、同样的形状 |
| 看图 | 零样本 | 零样本 |
| System 2 | 未改动的 Qwen3.8-27B | 未改动的 Ternary Bonsai 2 27B |

## 结果

**文本决策**，[jev-distill-corpus-v3](https://modelscope.cn/datasets/SargeDev/jev-distill-corpus-v3) 的 `test_set_30k`，按类型温度校准后的 KL(目标 ‖ 模型)，越低越好。"Jev 标注"指由 TypeSafe Jev 1.13 打标签的 25,376 行。

| 模型 | Jev 标注 | noul | score | choice |
|---|---:|---:|---:|---:|
| Ternary Bonsai 2 27B 零样本（读字母 logits） | 0.443 | 0.167 | 0.724 | 0.595 |
| JEV-9B（公开） | 0.019 | 0.005 | 0.023 | 0.028 |
| JEV-27B（公开） | 0.017 | 0.004 | 0.021 | 0.025 |
| **Bonsai-JEV** | **0.0163** | 0.0041 | 0.0205 | 0.0245 |

JEV-27B 模型卡上的其他指标，同样的行、同样的定义：

| | JEV-27B（公开） | **Bonsai-JEV** |
|---|---:|---:|
| 是/否 AUROC，Jev 标注的行 / 全部行 | 0.995 / 0.996 | 0.995 / 0.996 |
| 选择题和 Jev 的 top-1 一致率，Jev 标注的行 | 90.5% | 91.1% |
| 选择题 top-1，目标有明确最优项的行（前两名相差 ≥ 0.1） | 95.8% | 96.2% |
| 打分误差（期望分的 MAE，0–5 分制） | 0.098 | 0.097 |
| 对真实标签的 KL，没见过的任务族 | **0.104** | 0.128 |
| top-1 准确率，没见过的任务族 | 0.942 | 0.948 |
| 选项打乱 / 倒序后选择题 top-1 翻转的比例 | 2.9% / — | 3.4% / 4.3% |

拟合温度 1.014 / 1.014 / 1.012（noul / score / choice）：训练本身就校准好了。验证 KL 从第 250 步的 0.070 降到第 4,000 步的 0.018，最后 250 步已经持平。

**看图决策**，[VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench)（1,247 对，两种顺序取平均，与 JEV demo 协议相同）：

| 模型 | 总分 | hallucination | reasoning | general | 顺序一致率 |
|---|---:|---:|---:|---:|---:|
| Ternary Bonsai 2 27B 零样本 | 74.5 | 78.9 | 72.2 | 60.2 | 50.9% |
| JEV-27B-VL（公开） | 78.3 | 83.2 | 78.2 | 58.0 | 89% |
| **Bonsai-JEV** | **78.0** | **84.9** | 74.1 | 56.4 | 88.4% |

和 JEV-27B-VL 持平（1,247 对的置信区间约 ±2.4）。不取平均、只做一次前向：77.5 / 79.1（原顺序 / 换位）。

顺序一致率是把两个回答换位后结论不变的比例，反映单次前向的结果是否可信。

**具身推理**，[ERQA](https://github.com/embodiedreasoning/ERQA)（400 道机器人场景题，单次前向，JEV-35B 的协议）：**52.0**（JEV-27B-VL 53.8，JEV-35B 51.3；随机约 26，400 题置信区间约 ±5）；单图题 56.1，多图题 41.6。

**Agent 评判**（零样本，JEV-27B-DEMO 的提示词）：AgentRewardBench AUROC **0.930**（JEV-27B-VL 0.91），阈值 0.5 时精确率 / 召回率 74.7 / 74.7（78.4 / 70.2）；Plan-RewardBench 宏平均准确率 **74.7**（73.2）。**[JevBench-231](https://github.com/fstandhartinger/jevbench)**：用 JEV 的 `bare-v1` 提示词答对 194 / 231，比同一提示词下的零样本底座多 28 题。

**速度和显存**，RTX 4070 12 GB，PrismML 的 llama.cpp 分支：

| | |
|---|---|
| 处理输入 / 生成，PQ2_0 / PTQ1_0 | 1,226 / 49.6 和 1,315 / 63.1 token/秒 |
| 峰值显存（PQ2_0 + 视觉 + 4k 上下文） | 9.17 GB |
| 服务启动 / 启动后第一个请求（文本、看图） | 12 秒 / 0.54 秒、0.98 秒 |
| `/v1/decide` 端到端，新状态：文本 / 一张图 + 两个回答（p50） | 0.62 秒 / 0.99 秒 |
| 用 `/v1/warm` 预热一个场景（一张图）/ 之后每次追问（p50） | 一次 1.19 秒 / 0.23 秒 |
| 8 GB 显卡：PTQ1_0 + fp16 适配器 + 视觉，4k 上下文，显存限制在 7.5 GiB | 文本 0.42 秒，看图 1.04 秒（p50），占用约 7.1 GiB；视觉放 CPU（`MMPROJ=cpu`）约 6.3 GiB，看图 3.4 秒 |
| 12 GB 上的最大上下文 | 约 2.4–3.2 万 token（f16 KV），约 4.8–6 万（q8_0 KV） |

**PTQ1_0** 挂同一个适配器，决策完全一致（3,000 行上 top-1 一致率 100%，概率差为 0），而且更快。**并发：** 8 个槽位上 40 个并发决策和逐个发的结果一致（最大概率差 0.0035，没有翻转；混入的聊天不受影响）；分支只把 LoRA 配置相同的请求放进同一批。

**其他设备。** *Mac mini M4 16 GB（MLX）：* Prism 的 [MLX 2-bit 包](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) 加本仓库的 `peft/` 和 `head.safetensors`，不需要转换（GitHub 上的 `mlx_serve.py`）；文本和 CUDA 路径完全一致（300 行上 KL 0.00000），看图 96.3% 判断一致；内存 8.6–9.7 GiB；文本 p50 2.1 秒，看图 p50 13.8 秒。*Radeon 780M 核显（Vulkan）：* PTQ1_0 + fp16 适配器，核显内存约 7.7 GB，预填充 54 token/秒；文本 p50 4.3 秒，看图 p50 13.3 秒，场景预热后追问 2.9 秒。

## 快速上手

从源码编译 [PrismML 的 llama.cpp 分支](https://github.com/PrismML-Eng/llama.cpp)（CUDA 或 Metal）。原版 llama.cpp 加载不了这些文件；AVX-512 CPU（比如 AMD Zen 4）上一定要自己编译，发布的二进制一加载就崩。

```bash
pip install fastapi uvicorn requests pydantic pillow
# 12 GB 以上显卡：PQ2_0 底座（约 8.3 GB）。两个来源任选，国内推荐 ModelScope
modelscope download --model dakepeng/Bonsai-JEV --exclude "*PTQ1_0*" "adapter_f16.gguf" "peft/*" "head.*" --local_dir Bonsai-JEV-weights
hf download dakebakery/Bonsai-JEV --exclude "*PTQ1_0*" --exclude "adapter_f16.gguf" --exclude "peft/*" --exclude "head.*" --local-dir Bonsai-JEV-weights
# 启动 llama-server（:8080）和 /v1/decide 封装（:8000）；LLAMA_BIN 指向分支的 build/bin
LLAMA_BIN=~/llama.cpp/build/bin bash Bonsai-JEV-weights/run_decide.sh start Bonsai-JEV-weights

# 8 GB 显卡：PTQ1_0 底座 + fp16 适配器（下载约 6.8 GB）
modelscope download --model dakepeng/Bonsai-JEV --exclude "*PQ2_0*" "adapter.gguf" "peft/*" "head.*" --local_dir Bonsai-JEV-weights
BASE_GGUF=Ternary-Bonsai-2-27B-PTQ1_0.gguf ADAPTER_FILE=adapter_f16.gguf CTX=4096 NP=1 \
  LLAMA_BIN=~/llama.cpp/build/bin bash Bonsai-JEV-weights/run_decide.sh start Bonsai-JEV-weights
```

### System 1：类型化决策

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "kind": "choice",
  "state": ["摄像头画面：", {"image": "frame.jpg"}, "\n夹爪张开，位于桌面上方。"],
  "question": "Which object should the arm pick first?",
  "options": ["red cube", "blue cube", "nothing, the scene is unsafe"]}'
# {"kind":"choice","options":[...],"probabilities":[0.71,0.22,0.07],"choice_index":0,"choice":"red cube",...}
```

- `kind`：`noul`（选项 `false`/`true`）、`score`（`0`–`5`）、`choice`（2–16 个选项，标为 A–P）。
- `state`：字符串、JSON 对象，或文本和 `{"image": 路径 | URL | data-URL}` 混合的列表。
- 返回格式和 JEV 的 `serve_decide.py` 一致，JEV 的 [`jev_client.py`](https://github.com/yuhai-china/JEV-27B-DEMO/blob/main/common/jev_client.py) 设置 `JEV_URL=http://localhost:8000 JEV_BACKEND=decide` 即可直接用。
- 同一个场景问多个问题：先 `POST /v1/warm {"state": ..., "kinds": ["noul"]}` 一次，之后每次 `/v1/decide` 只计算问题本身的 token（4070 上约 0.2 秒）。模型有循环层，前缀复用需要这个显式边界，每种 `kind` 预热一次。

### System 2：聊天和推理

直接向 8080 端口的 `llama-server` 发普通的 OpenAI 格式请求。`run_decide.sh` 加载适配器时把全局 scale 设为 0，所以这些请求看到的是未改动的 Ternary Bonsai 2：适配器关闭时，12 个测试回答和不加载适配器的服务逐字节一致。聊天时不要把适配器的 scale 设成 1，打开后回答会变。

### 提示词格式

封装层会渲染 JEV 的 `bare-v1` 模板，你只需传字段：

```
[kind] choice
[state] <状态，图片内嵌>
[question] <问题>
[options]
A) <选项 1>
B) <选项 2>
[decision]:
```

概率从 `[decision]:` 位置的 verbalizer token（`false/true`、`0`–`5`、`A`–`P`）读出，再加上 `decision.json` 里的决策头偏置和按类型温度。场景放前、问题放后，预热的前缀才能复用。

## 文件

| 文件 | 大小 | 内容 |
|---|---:|---|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7.2 GB | Prism ML 的底座，和验证时用的文件逐字节一致 |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5.95 GB | 同一套权重的更紧凑打包，给 8 GB 显卡用（`BASE_GGUF=Ternary-Bonsai-2-27B-PTQ1_0.gguf`） |
| `adapter_f16.gguf` | 0.23 GB | fp16 版适配器，给 8 GB 显卡用（`ADAPTER_FILE=adapter_f16.gguf`） |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.6 GB | 视觉投影 |
| `adapter.gguf` | 0.46 GB | 64 层上的 LoRA + 作为 `output.weight` 上 rank-24 LoRA 的决策头 |
| `decision.json` | — | 决策头偏置、verbalizer id、槽位范围、按类型温度 |
| `peft/`、`head.safetensors` | 0.44 GB | 同一个适配器的 PyTorch 版本，可继续训练 |
| `calibration.json`、`head.json` | — | 拟合的温度和拟合统计；决策头元数据（第 4,000 步） |
| `decide_server.py`、`run_decide.sh`、`zeroshot_text.py` | — | 套在 `llama-server` 上的 `/v1/decide` 和 `/v1/warm` |
| `configuration.json` | — | ModelScope 仓库配置 |
| `LICENSE`、`NOTICE` | — | Apache 2.0；署名 |
| `LICENSE-Ternary-Bonsai-2`、`NOTICE-Ternary-Bonsai-2.txt`、`KNOWN_ISSUES-Ternary-Bonsai-2.md` | — | Prism ML 为底座文件提供的许可证、NOTICE 和已知问题，原样附上 |

适配器和这个底座文件是绑定的（Hadamard 旋转元数据、value 头排布）。请使用仓库附带的 GGUF，不要换成更新或更旧的 Bonsai 2 文件。

## 训练

LoRA r=16、α=32、dropout 0.05，挂在全部 64 层的 `in_proj_qkv, in_proj_z, out_proj, q/k/v/o_proj, gate/up/down_proj` 上，作用于层的原始（未旋转）输入，和分支的做法完全一致；fp32 决策头 5120→24，用 verbalizer token 在 lm_head 中的行初始化。损失为有效槽位上的 KL(目标 ‖ 模型)，打分题加 0.5·RPS；30% 的选择题打乱选项。每步 128 行按类型分层，micro-batch ≤ 2.4 万填充 token，梯度检查点，AdamW (0.9, 0.98)，学习率决策头 2e-4 / LoRA 1e-4，3% 预热，余弦退火到 0.02 倍，4,000 步（JEV-27B 的 5,000 步扣掉其中 21% 的占位行）。训练数据为 [jev-distill-corpus-v3](https://modelscope.cn/datasets/SargeDev/jev-distill-corpus-v3) 的 518,603 行（137,203 行标签为占位 `[0.5, 0.5]` 的 `yuri_v1` 直接丢掉，而不是像 JEV-27B 那样给 0.05 权重）。丢掉它们没有可测的代价：noul KL 0.0041，JEV-27B 是 0.004；拟合出的 noul 温度两边都是 1.014。底座在 BF16 下冻结，从三值文件解包。2× RTX PRO 6000 96 GB，约 24 小时。发布的 checkpoint：第 3,750 步和第 4,000 步的验证 KL 打平（都是 0.0180），于是用 `calibration` 划分来分胜负，第 4,000 步在三种题型上都更低；挑选时没有用测试集；按类型温度在 `calibration` 划分上用 L-BFGS 拟合。

代码、对齐检查和完整的决策记录：[GitHub](https://github.com/DakePeng/Bonsai-JEV)。

## 局限

- **质量天花板是 JEV 水平。** 三值底座只能挂 LoRA。Qwen3.8-27B 的全量微调模型在 [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) 上分数更高；这个模型的意义在内存，不在超过它们。
- **和 JEV 同一个教师**，文本上和 JEV-27B 持平，不会超过它。迁移到没见过的任务族更弱（KL 0.128 对 0.104）。
- **三值化对视觉的伤害大于文本**（Bonsai 的模型卡：MMMU-Pro 81.7 → 75.5）。看图决策和 JEV-27B-VL 一样是零样本。
- **英文提示词。** 训练数据是英文；其他语言（包括中文）的决策质量未测。
- **延迟：** 每次决策是一次 27B 的前向；Jetson Orin NX 上一次看图决策预计几秒。

## 许可证和致谢

适配器、决策头和脚本为 Apache 2.0。*Created using Bonsai by Prism ML*；Ternary Bonsai 2 27B © Prism ML, Inc.，Apache 2.0，基于 Qwen3.8-27B（© 阿里云，Apache 2.0），本仓库原样再分发并附带其许可证和 NOTICE。配方和部署设计来自 autotrust 的 JEV-9B / JEV-27B（Apache 2.0）。训练数据为 [SargeDev/jev-distill-corpus-v3](https://modelscope.cn/datasets/SargeDev/jev-distill-corpus-v3)（Apache 2.0；`openjev_v2` 为 CC0），与 JEV-9B 和 JEV-27B 使用的语料相同；其中 `yuri_v3` 的标签是 TypeSafe 的 Jev 1.13 的输出，商用者可自行查看 TypeSafe 的 API 条款。VL-RewardBench 仅用于评测。

感谢 Prism ML、autotrust、SargeDev、Qwen 团队和 Jev Decision Index 的维护者。
