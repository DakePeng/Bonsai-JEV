# Bonsai 2 27B 边缘决策模型 — 评估与执行计划

日期：2026-10-10。基于 `bonsai-edge-plan.html` v3.1，按实际手上的硬件调整。

## 1. 结论

**可行，但原方案要改三处。**

1. **没有 H20。** 所有训练都租卡。费用不高：JEV-9B 用了约 3 个 B200 小时，27B 一次约 9–10 个 B200 小时。按选定的租卡方案（2× RTX PRO 6000 96GB，每小时 ¥13.96）一次训练 ¥310–380，整个项目 **¥450–1700（$65–240）**，取决于训练几次。
2. **4070 12GB 是唯一的参考运行时和部署替身。** 它跑的就是 Orin 将来要跑的 PrismML llama.cpp 分支（CUDA），所以对 Orin 主线来说它比 MLX 更合适做参考。PQ2_0 27B 7.21 GB + 视觉 0.63 GB + LoRA 约 0.3 GB，总占用约 9.5 GB，KV cache 有余量。它同时顶替没有的"20.97"机器来跑 JEV-9B FP8 对照基线。M5 降为最后可选的一天交叉验证。Orin 的延迟在拿到 Orin 之前无法测。
3. **跳过 8B 冒烟测试。** `Ternary-Bonsai-8B-unpacked` 是 Qwen3 架构，不是 Qwen3.8。它没有线性注意力层，测不到 `in_proj_qkv/in_proj_z/out_proj` 这些 LoRA 目标，也走不到混合架构的代码路径。直接用 27B 本身取 1% 数据在租的卡上冒烟（约 $10）。同一套代码，不用多调一个模型。

### 硬件分工

| 机器 | 用途 | 放得下？ |
|---|---|---|
| 4070 12GB + 7700X | 参考运行时（PrismML llama.cpp 分支，CUDA）、部署替身、全部本地评测、JEV-9B FP8 基线、逐层 PyTorch 对比 | 推理可以。**27B 或 8B 的 BF16 训练都不行** |
| M5 MacBook 32GB | 可选：最后一天的 MLX 交叉验证（第 25 步）。不在关键路径上 | 可以，约 11 GB |
| M1 Air 8GB（这台 Mac） | 只做编辑、写脚本、调度 | 跑不了本方案里任何模型 |
| 租用 RTX PRO 6000 96GB，1 或 2 张（见 1.1 节） | P0 全模型对齐、P1 冒烟、P2 训练（2 张）、P3 教师模型评测 | — |
| Orin NX 16GB | 最终目标 | 手上没有。采购或推迟 P4b 最后一步 |

### 打包格式：PQ2_0 与 PTQ1_0

两种文件装的是**同一组**三值权重（−1/0/+1，每 128 个权重一个 fp16 缩放因子，同一个 Hadamard 旋转坐标系，`config.json` 里同一组符号向量）。解包后的数值逐位相同，所以模型质量、KL、ECE、决策概率都和选哪个无关。Hadamard 旋转在两种格式里的处理方式一样；第 9 步的 LoRA 位置检查对两者都适用。

| | PTQ1_0 | PQ2_0 |
|---|---|---|
| 编码 | 一个字节装 5 个三进制位，约 1.75 bit/权重 | 每个权重 2 bit，四个格子空一个，约 2.13 bit/权重 |
| 文件 | 5.95 GB | 7.21 GB |
| 内核里的解包 | 除法取余或查表，GPU 上慢 | 移位加掩码，几乎免费 |
| 处理输入速度 | 1× | 模型卡上每张 CUDA 卡都快 1.7–2.3 倍 |
| 生成速度 | 略快（受带宽限制） | 基本持平 |

一次决策只生成 1 个 token，成本全在处理输入上。**放得下就用 PQ2_0**（4070 总占用约 9.5 GB；Orin NX 16 GB 约 10.7 GB）。到了 Orin 上两种都跑一遍 `llama-bench`，留实测更快的那个。两种打包之间可以互相转换，不需要重新量化。

### 1.1 租卡方案：RTX PRO 6000 Blackwell 96GB

选定机型：国内平台，RTX PRO 6000 96GB，每卡每小时 ¥6.98，每卡配 22 核 + 110 GB 内存，系统盘 30 GB，数据盘 50 GB 可扩到 103 GB，驱动 595，CUDA ≤ 13.2，同一台机器有 2 张空闲。

**为什么合适。** 96 GB 能装下 54 GB 的 BF16 冻结底座，还能按原配方跑 24k token 的 micro-batch，不用像 H100 80GB 那样压缩。BF16 算力大约是 H100 的一半、B200 的四分之一，单次训练慢一些，但按小时单价算总费用差不多。

**每个阶段用几张卡。** 冻结底座上训 LoRA 是数据并行：每张卡各放一份完整底座，只同步 4000 万 LoRA 参数的梯度，PCIe 够用。两卡吞吐约 1.9 倍。

| 阶段 | 卡数 | 时长 | 费用 |
|---|---|---|---|
| P0 对齐 + 内核检查 | 1 | 5–6 小时 | 约 ¥40 |
| P1 冒烟 | 1 | 3–4 小时 | 约 ¥30 |
| P2 正式训练，每次 | 2（`torchrun --nproc_per_node 2`） | 22–27 小时 | ¥310–380 |
| P3 教师模型重跑（仅评测代码校验失败时） | 1 | 8–12 小时 | ¥60–85 |

**这台机器的三个限制。**

- **硬盘紧。** 不要把 54 GB 的 BF16 底座写到盘上。盘上只存 7 GB 的三值文件，加载时在显存里解包；`bonsai_torch.py` 本来就是这么做的。这样 P3 的 52 GB JEV-27B-VL 也能和数据集一起放进 103 GB 的数据盘。
- **Blackwell 是 sm_120。** PyTorch 要 CUDA 12.8 以上的构建。`flash-linear-attention` 依赖 Triton，只有最近的 Triton 支持 sm_120。第 11 步先验证这个。
- **按阶段租，不要连续包。** 数据盘持久化，阶段之间关机。五周里机器实际开着 5–9 天。

这个机型租不到时的备选：B200（每次约 10 小时，每小时 $4–9）或 H100 80GB（每次约 22 小时，micro-batch 压到 8–12k token，每小时 $2–4）。

### 1.2 前人工作（2026-10-10 检索）

没有人公开做过这个组合：在 Ternary Bonsai 2 27B 上用梯度训练 LoRA 加决策头，从 JEV 蒸馏。已有的相关工作：

| 工作 | 是什么 | 对我们的用处 |
|---|---|---|
| `ChezRD/ternary-bonsai-2-27b-decision-lora` | rank-128 适配器，**把三个现成的 delta 相加**得到，没有训练、没有数据。JevBench 194→207（共 231 题）。跑在 bonjev 上。 | 证明分支能在 27B 上加载不合并的 LoRA。不是质量参照。 |
| `AtomicChat/Ternary-Bonsai-2-27B-Abliterate-LoRA-GGUF` | rank-1 的推理时 LoRA，底座不动。 | 文件很小。读分支怎么应用它，就能解决第 9 步的 Hadamard 位置问题。 |
| `OS-Software/...-Heretic-GGUF` | 把去审查的修改合并进三值权重，改了 34 个矩阵。 | 说明合并后重新三值化可行。我们用不到。 |
| `ternary-qat`（PyPI，第三方） | `swap_linear` 加 PEFT LoRA 在三值冻结底座上训练，保存时重新三值化。没说是否处理 Hadamard 坐标系。 | P1 写 `bonsai_torch.py` 之前先看它的 GitHub，别从零写。 |
| arXiv 2608.24469，Low-Rank Ternary Adaptation | 学术工作，LLaMA-3 1B/3B，合并结果保持三值。 | 目标不同。我们的 LoRA 单独以 BF16 存放。 |

上面没有任何一个测过对 JEV 的 KL、校准或看图能力。方案里 P0 的零样本基线会是第一个这样的数字。

### 1.3 对照 Jev Decision Index 的定位（2026-10-13 查）

社区榜单（149 个文本条目、33 个视觉条目，统一在一张 RTX PRO 6000 上跑，去随机基线，公开/私有同技能/私有新领域按 20/50/30 加权）改变了这个项目的两点定位。

- **JEV-27B 不是质量前沿。** 文本榜：vLLM-SR d3 27B 66.3、Decider 31B（NVFP4）64.6、Perplexity Decider 62.8、Darwin-27B 62.5；JEV-27B 55.6（第 27 名），JEV 官方 API 60.1。视觉榜：d3 27B 71.3、Perplexity 70.6、**JEV-27B-VL 69.6（第 3 名）**。文本榜前四全是用公开基准 train/dev 划分训练的全量微调；LoRA 加决策头做得最好的 Sieve 27B 是 60.2，和 JEV 持平。**冻结的三值底座只能挂 LoRA，所以本项目的质量天花板是 JEV 水平，不是 d3 水平。** 每份汇报都要写明。
- **"≤ 8 GB × 27B 级"这个格子是空的。** 榜上按参数量的前沿是 d3-mini 4B（文本 52.6 / 视觉 63.2，BF16 约 9 GB）和 d3-flash 9B（55.8 / 66.9，约 17 GB）。在约 7 GB 里复现 JEV-27B-VL，两个榜都压过 4B，视觉上压过 9B；9B 做 4-bit（约 6 GB）文本能持平。延迟不是我们的轴（计算量是 9B 的 3 倍）。卖点只有一句："8 GB 里的 27B 级决策质量"。
- **量化和 llama.cpp 服务的条目榜上都收**：Decider NVFP4 通过了和 BF16 决策一致率 ≥ 96% 的门槛（96.99%）；Winnow 用 llama.cpp 跑 Q8_0 GGUF 上榜。榜上没有三值或 10 GB 以下的 27B 条目。第 21 步顺便跑它的公开套件，以便提交；预期要做和 BF16 参照的一致率检查。

**从榜上借的四条，已写进下面的步骤：** 倒序二次读取（Decider 31B）、随选项数变化的温度（Decider）、公开基准 train 划分作为补充数据（Darwin/Decider/Torchcast，做泄漏过滤）、跑 DI 公开套件用于提交。轮换平均加无标签先验（AnyJev L0，不训练拿到 60.2）是训练后顺序一致率不达标时的兜底。

### 1.4 许可证（2026-10-13 核对）

| 组件 | 许可证 | 对本项目的影响 |
|---|---|---|
| Ternary Bonsai 2 27B（权重、mmproj） | Apache 2.0（Prism ML 的 NOTICE；基于 Qwen3.8-27B，Apache 2.0） | 微调、LoRA、再分发、商用都允许。要求署名："Created using Bonsai by Prism ML"。 |
| PrismML 的 llama.cpp 分支 | MIT | 可以随产品发布；保留版权声明。 |
| Qwen3.8-27B 的配置/分词器/视觉塔 | Apache 2.0 | 没问题。 |
| `SargeDev/jev-distill-corpus-v3` | 数据集层面 Apache 2.0；`openjev_v2` 源头 CC0 | 允许训练；要署名数据集。**待确认点：** `yuri_v3`（49.8 万行，占语料三分之二）的标签是用 TypeSafe 的 Jev 1.13 通过 OpenRouter 生成的。数据集卡片没提 TypeSafe 的 API 条款，TypeSafe 网站条款不覆盖 API，OpenRouter 条款要求用户自行遵守各提供方的模型条款。没找到任何禁止用 Jev 输出训练的公开条款，也没找到允许的。商用发布前要读 TypeSafe 的 API 协议；如果禁止蒸馏，退到 `openjev_v2`（CC0）+ `yuri_v1` + 用 JEV-27B（Apache 2.0，见下一行）打标签的自有合成数据。 |
| JEV-9B / JEV-27B / JEV-27B-VL（适配器、决策头）、JEV-27B-DEMO | Apache 2.0 | 当基线、评测代码和 **P5 的教师**都允许。万一 Jev-1.13 标注的那部分要弃用，它也是干净的教师。 |
| VL-RewardBench | YAML 写 MIT，卡片写"仅限研究用途"，源自 GPT-4o/Claude 的输出 | 只用于评测，绝不训练。只报数字，不再分发。 |
| RLAIF-V、RLHF-V（P5 候选） | CC BY-NC 4.0 | **商用模型不能用。** 只做研究性消融，或换成用 JEV-27B 打标签的公开图片和业务图片。 |
| VLFeedback | 未声明许可证 | 澄清前视为不可用。 |
| Open-Jev（`ZefanCai/Open-Jev`） | CC0 | 随意用。 |

规则：发布模型的训练数据必须是 Apache/MIT/CC0/自有；NC 和仅研究用途的数据集只用于评测和消融。

### 1.5 开源方案

**许可证：Apache 2.0**，覆盖我们产出的全部（LoRA、决策头、`decision.json`、校准文件、代码、GGUF 打包）。整条链都是 Apache 2.0（Bonsai、Qwen3.8、JEV 适配器、语料），Apache 带专利授权，打包里一个许可证省去逐文件标注。MIT 用在我们自己的文件上也可以，但打包 GGUF 时要双重声明。

**发布内容（2026-10-14 定）：按 JEV 的方式做一个"全家桶"模型仓库，Hugging Face 和 ModelScope 同步发布。**
JEV-9B/27B/27B-VL 都是把完整底座权重和适配器、决策头、校准、服务脚本放在同一个仓库（"一个仓库、一条命令"；JEV-27B 约 54 GB，JEV-27B-VL 约 56 GB）。我们也这样：

| 文件 | 大小 |
|---|---:|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf`（Prism ML 的原文件，逐字节一致，就是我们验证过的那份） | 7.2 GB |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.6 GB |
| `adapter.gguf`（LoRA + 作为 lm_head LoRA 的决策头，第 15 步）+ `decision.json` + `calibration.json` | 0.46 GB |
| `peft/`（adapter_model.safetensors + 配置）+ `head.safetensors`，给 PyTorch 和继续训练用 | 0.44 GB |
| `decide_server.py`、`run_decide.sh`、模型卡 README、`LICENSE`、`NOTICE`（含 Prism ML 的 NOTICE） | 很小 |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf`（5.95 GB），适配器在它上面验证通过后再加 | 可选 |

为什么打包底座而不是只给链接：适配器和这个 GGUF 文件是绑死的（v 头顺序、`prism.hadamard.*` 元数据、旋转版本）；Bonsai 的已知问题里已经记录过新旧文件不匹配会静默输出乱码。放一份验证过的副本就把版本锁住了。Apache 2.0 允许再分发，只要保留 Prism ML 的许可证和 NOTICE。
- **Hugging Face** [`dakebakery/Bonsai-JEV`](https://huggingface.co/dakebakery/Bonsai-JEV) 和 **ModelScope** [`dakepeng/Bonsai-JEV`](https://modelscope.cn/models/dakepeng/Bonsai-JEV)：文件相同。ModelScope 对国内用户很重要（我们的 4070 和 AutoDL 机器访问 Hugging Face 都很慢或连不上，所有下载都是 ModelScope 完成的）。用 `huggingface-cli upload` / `modelscope upload` 上传；ModelScope 的模型卡需要 YAML 头（license `apache-2.0`、tags、base_model）。
- **GitHub**：训练、导出、评测和服务的代码（就是本仓库），README 指向两个模型仓库。
- `LICENSE`（Apache 2.0）和 `NOTICE`："Created using Bonsai by Prism ML"；基于 Qwen3.8-27B（阿里云，Apache 2.0）；JEV（autotrust，Apache 2.0）用作基线/教师；"trained on SargeDev/jev-distill-corpus-v3"并附作者要求的链接。

**哪个 checkpoint 可以发布**
| 训练数据 | 可发布？ |
|---|---|
| 干净集：`openjev_v2`（CC0）+ `yuri_v1` + 用 JEV-27B/JEV-27B-VL（Apache 2.0）打标签的自有合成数据 + 自有业务图片 | **可以，无条件。** 这是发布版。 |
| 完整的 `jev-distill-corpus-v3` 含 `yuri_v3`（标签来自 TypeSafe 的 Jev 1.13） | 要先确认 TypeSafe 的 API 条款不禁止蒸馏。开源比内部使用更显眼，先问清楚。 |
| 碰过 RLAIF-V / RLHF-V（CC BY-NC）或 VLFeedback（无许可证）的 | **不行。** 这些只作内部消融。 |

**对计划的影响。** 第一次训练（P2）仍用完整语料，作为 JEV 配方的对照。再加一次干净集的训练（同配方，约 ¥350）；如果它的 KL 和完整语料那次相差在 0.005 以内，干净集就成为默认，TypeSafe 的问题对发布版不再构成影响。模型卡必须写明：冻结三值底座上的 LoRA，质量天花板是 JEV 水平；平均后的 VL-RewardBench 分数旁边并列顺序一致率和单顺序准确率；DI 榜分和 ≥ 96% 的 BF16 一致率数字。

## 2. 原方案漏掉的风险

| 风险 | 为什么重要 | 怎么查 |
|---|---|---|
| **LoRA 坐标系不匹配。2026-10-12 已确认。** | 分支的 `build_lora_mm`（`src/llama-graph.cpp`）里，底座矩阵乘法用的是旋转后的输入 `cur_mm = H(s⊙Px)`，而 LoRA 分支用的是原始 `cur`。按 B·A·(H·x) 训练的 LoRA 会被套在错误的输入上，静默出错。 | **决定：** PyTorch 训练模型完全照搬分支：冻结底座用旋转后的输入，LoRA 用原始输入。这样 A 天生就在普通坐标系里，导出不需要任何数学变换。原方案"在旋转坐标系里训练"对 LoRA 路径不再适用。 |
| **JEV 没有公开训练代码。** | 整个训练器（决策头、KL+RPS 损失、分层采样、打乱增强）都是照 README 写的。 | P1 冒烟必须复现 README 里损失曲线的形状。预留 3–5 天，不是 3 天。 |
| **Qwen3.8 混合注意力需要训练级内核。** | 线性注意力（Gated DeltaNet 一类）的反向传播要快，HF 实现里得有 `flash-linear-attention` 或等价物。 | P0 第一次租卡时先验证这个，别的都往后放。 |
| **bonjev 不是真正的备选。** | 16 个 commit，不做训练，它说的"LoRA"是把现成的 delta 重新组合。只有读 logit 的做法可以借鉴。 | 当参考代码看，不当依赖。 |
| **手上没有 Orin。** | Orin 的延迟和 sm_87 上的 PTQ1_0 都验证不了。方案里 6 秒的目标一直悬着。 | 尽早定：第 1 周买一台 Orin NX 16GB（约 $600–900），或者宣布 P4b 本期不做。 |
| **"已完成"的 JEV-9B FP8 基线**是在 20.97 那台机器上测的，你没有这台机器。 | 没有这些数字，就没有"比 JEV-9B 好否则停"这道关。 | 找跑过的人要 `eval/bf16.jsonl` 和结果文件。要不到就在 4070 上重跑：vLLM FP8 在 Ada（sm_89）上可用，9B FP8 约 9 GB，用 `--max-model-len 8192`。 |
| **分支二进制必须和 GGUF 的旋转版本匹配。** | 模型卡说早期版本是"pre-rotation"。旧的分支二进制加载现在的 GGUF 会输出乱码而且不报错。原版 llama.cpp 也一样。 | 把分支锁定到 Bonsai-demo 仓库指定的 commit。把哈希记进 `README.md`。第 7、8 步的 logits 对齐就是检测手段。 |
| **7700X 是 Zen 4（AVX-512）：用分支的预编译二进制加载 PQ2_0 会段错误**，即使所有层都在 GPU 上。 | KNOWN_ISSUES：源码已于 2026-09-23 修复（PR #245），写作时还没进二进制发布。 | 这台机器上永远从源码编译分支（2026-10-12 已完成，commit e8fd209，CUDA 12.0，gcc-12）。不要在这台机器上用发布的二进制。另外避开 CUDA 13.3 构建（另一个未修复的崩溃）。 |
| **只有一个本地参考运行时。** | 去掉 MLX 之后，llama.cpp 分支的 logits 如果有 bug，本地没有东西能和它对不上。 | 租卡上的 PyTorch 模型（第 8 步）就是第二意见。读 MLX `runtime/` 的 Python 源码了解 Hadamard 细节，读代码不占 M5 时间。 |

## 3. 分步计划

一个工程师，约 5 周。方括号里是租卡时长。

### 第 0 周 — 环境（2 天）

1. 给项目建 Git 仓库。加 `requirements.txt`、`Makefile` 和 `configs/` 目录。
2. 4070：用 `-DGGML_CUDA=ON` 编译 `PrismML-Eng/llama.cpp`。拉 **PQ2_0** GGUF（7.21 GB，模型卡上每张 CUDA 卡处理输入都比 PTQ1_0 快约 2 倍，12 GB 放得下）加 Q8_0 mmproj。PTQ1_0 只留给 Orin。跑一次纯文本和一次带图。记录处理输入的 tok/s 和显存占用。同时编译 `llama-eval-callback` 示例，它能导出每层张量，替代原来的 MLX 逐层导出。
3. 只下载 MLX 包里的 `runtime/` 目录（几 KB 的 Python，不含权重）。这是 Hadamard 变换和符号向量的可读规范。只读，不跑。
4. 把 `SargeDev/jev-distill-corpus-v3`（490 MB）和 `fraserprice/Ternary-Bonsai-2-27B-vllm`（三值码 + 缩放因子 + 符号向量）拉到 7700X 机器上。
5. 选一家租卡平台。建一个 150 GB 的持久卷。把数据集和 vllm 格式的权重放上去。写一个 `bootstrap.sh` 装依赖、挂卷，让每台租来的机器 10 分钟内能开工。

**过关标准：** 本地运行时能输出正常文本；数字记下来了。

**2026-10-12 实测，4070 12GB，PQ2_0，分支 commit e8fd209，CUDA 12.0，WSL2：** `llama-bench` pp512 = **1226 tok/s**，tg128 = **49.6 tok/s**。和事前估算（1000–1300 / 约 45）一致。推算单次决策延迟：300 token 纯文本约 0.25 秒，640×480 图加问题约 0.4 秒，不含视觉编码器时间。带图运行：640×480 视觉编码 119–125 毫秒，**峰值显存 9.17 GB**（模型 + mmproj + 4k 上下文），合成测试图上答案正确。纯文本补全正常。

**GGUF 元数据（PQ2_0 文件），`bonsai_torch.py` 必须复现的内容：**
- 架构 `qwen35`：64 层，隐藏维 5120，FFN 17408，每 4 层一个全注意力层（第 3、7、… 层），24 头 × head_dim 256，4 个 KV 头；线性注意力层用 `attn_qkv` [5120→10240]、`attn_gate` [5120→6144]、`ssm_out` [6144→5120]，加 BF16 的 `ssm_alpha`/`ssm_beta` [5120→48] 和 F32 的 conv1d、归一化。402 个 PQ2_0 张量，96 个 BF16，353 个 F32。
- `prism.hadamard.version=1`，`block_size=1024`，`transform=normalized-sylvester-walsh-hadamard`，`axis=input-last-dimension`，`sign_mode=explicit`，符号向量覆盖输入宽度 5120、6144、17408。`weight_names` 列出 401 个折叠权重。`gdn_v_grouped=true` → `ssm_out` 在符号和 Hadamard 之前有 `perm_rep` 特征排列。
- **`output.weight` 是 PQ2_0 且已折叠（旋转）。** 决策头从它的 24 行初始化，所以要么 (a) 决策头像分支处理 `output.weight` 那样吃 `H(s⊙h)`，要么 (b) 初始化时把这 24 行旋转回去一次（`row_plain = row_rot · H · diag(s)`），让决策头吃原始 `h`。选 (b)：导出就是一个普通的 24×5120 矩阵，运行时不需要任何变换。
- `token_embd.weight` 是 PQ2_0，带 `inverse-after-lookup`（版本 1，显式输出头，非共享）。
- 三值码和缩放因子的来源：**直接解包 PQ2_0 GGUF**（和运行时看到的逐位一致），不另外下载 vllm 格式的仓库。PQ2_0 块布局（`ggml-common.h`、`ggml-quants.c`）：每 128 个权重一组 = 1 个 fp16 缩放因子 `d` + 32 字节码，每个权重 2 bit，字节内小端（`q = (byte >> 2*(j%4)) & 3`），数值 `(q-1)*d`，码 3 不用。numpy 解包 20 行。
- 640×480 的图在这个运行时里是 **300 个图片 token**，和原方案假设一致。
- 4070 机器上的本地资源：分支在 `~/bonsai/llama.cpp`（二进制在 `build/bin`），模型在 `~/bonsai/models/bonsai2-27b`，数据集在 `~/bonsai/data/jev-distill-corpus-v3`（来自 ModelScope），venv 在 `~/bonsai/venv`，项目克隆在 `~/bonsai/jev`。

### 第 1 周 — P0 可行性

6. **解包 + Hadamard 的 PyTorch 模型。** 写 `bonsai_torch.py`：加载三值码和缩放因子，按 `(code-1)*scale` 解包成 BF16，给每个线性层包成 `y = W_rot · H(s⊙Px) + B·A·x`，其中每个权重的排列、符号、按 1024 分块的归一化 Sylvester Walsh-Hadamard 都从 GGUF 的 `prism.hadamard.*` 元数据读。底座路径旋转，LoRA 路径不旋转，和分支完全一致。复用 HF 的 Qwen3.8 模型类，只换线性层。
7. **在 4070 上逐层对齐，不租卡。2026-10-12 全部 64 层已完成。** 嵌入查表完全一致（差为 0）；64 层中 63 层输出的 fp32 相对差在 1.8e-3 以内，中位数约 5e-4；最后一层只在输出 token 上比（分支对它做了裁剪）。最终归一化 1e-7，logits 2e-3，argmax 100%。**过关。** BF16（训练用的精度；fp16 的分组缩放因子舍入到 BF16）下层输出相对差 2–6e-3，logits 3e-3，argmax 仍 100%：冻结底座和运行时每层差约 0.5%，远在方案 0.01 概率容差之内。 工具：`tools/dump-tensors`（分支的 example，导出 `model.input_embed`、`l_out-N`、`result_norm`、`result_output`，加 `DUMP_LAYERS=1,3` 时还导出这些层的所有中间张量，raw f32）、`align_layer.py`（一次一层，fp32 能放进 12 GB）、`align_inner.py`（层内逐投影对比）。用这套工具找到并修了三个只看 logits 看不出来的 bug：
   - **零中心 RMSNorm。** HF 的 `Qwen3_5RMSNorm` 乘的是 `(1 + w)`，GGUF 里存的已经是 `w + 1`。`attn_norm`、`post_attention_norm`、`q_norm`、`k_norm`、`output_norm` 都要减 1。带门控的 `ssm_norm` 是普通权重。
   - **线性注意力的 value 头顺序。** `gdn_v_grouped=true`：分支原生的 v 头顺序是 tiled（`h = k + nk·r`），HF 是 grouped（`h = r + rep·k`）。GGUF 里按 v 头存的张量（`attn_qkv` 的 v 行、`attn_gate`/z、`ssm_alpha`、`ssm_beta`、`ssm_dt`、`ssm_a`、`ssm_conv1d` 的 v 通道）是 tiled 顺序，给 HF 用要重排。`ssm_out` 的列已经是 grouped（分支是对激活做排列），所以 `perm_for()` 保持恒等。
   - **单独使用的 HF 层必须钉死 `_attn_implementation`**（sdpa）。不设的话 mask 构造器返回空，eager 回退路径会做非因果注意力。
   结果：每个投影和分支的差约 1e-3（分支内核是降精度计算），归一化层 1e-7，层输出 3.5e-4（线性注意力）和 5.9e-4（全注意力），lm_head logits 2e-3，argmax 100% 一致。
8. **2026-10-13 在租的机器上完成。** 64 层 BF16 全模型前向对分支导出的 `result_norm`（23 token 的 prompt）：最大相对差 4.45e-3，平均差为平均 |参照| 的 1.2%。通过（门槛 ≤ 1e-2；逐层 fp32 是 3.5e-4，64 层 BF16 累积到这个量级符合预期）。
9. **检查 llama.cpp 里的 LoRA 位置。2026-10-12 已完成。** LoRA 的输入在 **Hadamard 之前**（原始 `cur`），底座的输入在 Hadamard 之后。另外发现：分支按权重张量逐个从 GGUF 的 `prism.hadamard.*` 键注册旋转，部分权重在符号和 Hadamard 之前还有一个特征排列（`perm_rep`，GDN v 分组），符号在变换之前乘。`bonsai_torch.py` 必须读这些键，只在底座路径上复现 P → ⊙s → H 这条流水线。
10. **在 4070 上测零样本基线。** 不挂 LoRA，直接读字母 logits。用分支跑 `test_set_30k`（原版 Jev 标注的部分）的文本 KL 和 VL-RewardBench。这是训练后模型必须超过的下限。
   **文本部分，2026-10-12 进行中。** 方法按 JEV 的 `bare-v1` 模板钉死（来自 `common/jev_client.py` 和 `judge_config.json`）：`[kind] k\n[state] s\n[question] q\n[options]\n<行>\n[decision]:`，不加特殊 token，选项行为 `false`/`true`、`0`–`5`、`A) …`；verbalizer id 3721、1802、15–20、32–47（已验证在 Qwen3.8 分词器里完全相同）；在该类型的有效 id 上做 softmax，不加偏置，温度 1。工具：`tools/decide-logits`（分支 example，输出指定 id 的下一 token logits，4070 上约 0.18 秒/行）加 `zeroshot_text.py`（按类型和来源算 KL、准确率、ECE；逐行概率存下来，供之后训练后模型和底座对比）。`yuri_v1` 的行是 `[0.5,0.5]` 占位标签，已排除。**全量结果，27,695 行（`test_set_30k` 去掉占位行后的全部），2026-10-12：**

   | 子集 | n | KL | 准确率 | ECE |
   |---|---:|---:|---:|---:|
   | 全部 | 27,695 | **0.481** | 49.0% | 0.036 |
   | noul | 9,969 | 0.167 | 76.6% | 0.058 |
   | choice | 9,199 | 0.595 | 49.4% | 0.088 |
   | score | 8,527 | 0.724 | 16.4% | 0.074 |
   | 来源 yuri_v3（Jev 标注） | 25,376 | 0.443 | 48.5% | 0.042 |
   | 来源 openjev_v2 | 2,319 | 0.888 | 55.1% | 0.119 |

   这就是零样本下限：JEV-27B 训练后的 KL 是 0.017，JEV-9B 是 0.019，低 25 倍。600 行样本的预测在噪声范围内。打分题接近随机（6 选 1），说明打分能力几乎全靠决策头和 LoRA。

   **VL-RewardBench 零样本（不挂 LoRA、无决策头），demo 协议（两种顺序取平均），1,247 对，2026-10-12：**

   | | general | hallucination | reasoning | **overall** | macro | 顺序一致率 |
   |---|---:|---:|---:|---:|---:|---:|
   | Bonsai 2 27B 零样本 | 60.2 | 78.9 | 72.2 | **74.5** | 70.5 | 50.9 |
   | JEV-9B（公开） | | | | 74.3 | | |
   | JEV-27B-VL（公开） | 58.0 | 83.2 | 78.2 | 78.3 | 73.1 | 89 |

   解读：三值底座的视觉理解没有损伤（方案里"三值化损伤了视觉理解"的风险没有出现），取平均的协议下已经达到 JEV-9B 的水平。但顺序一致率 50.9% 对比 JEV 的 89%，说明原始底座有很强的位置偏好：单看任一种顺序接近抛硬币，准确率来自取平均后残留的小信号。训练决策头和 LoRA 应该能大幅消掉这个偏好；P3 要盯的是单顺序准确率和顺序一致率，不只是平均后的总分。4070 上每次决策约 0.7 秒（300 个图片 token 加约 250 个文本 token）。
10b. **2026-10-12 已准备好，排在 `run_step10.sh` 里第 10 步之后自动跑。** `run_10b.sh` 用 JEV-9B 自带的 `vl/serve_decide.py`（vLLM 0.31，FP8，4k 上下文，eager，8 并发）服务多模态 Qwen3.5-9B 底座 + `vl/adapter_vllm`，`jev9b_eval.py` 通过 demo 仓库原封不动的客户端（`JEV_BACKEND=decide`）跑文本和 VL-RewardBench。权重来自 ModelScope（JEV-9B 18 GB，Qwen3.5-9B 19 GB；注意 9B 的 serve 脚本钉死了底座版本 c202236，ModelScope 给的是最新版）。**2026-10-13 决定：本轮不做 10b**（租卡的磁盘和时间；门槛改用 JEV-9B 的公开数字 VL-RewardBench 74.3 和 KL 约 0.019，评测代码的可信度以 demo 客户端复现我们离线工具的结果为据）。P3 的数字有歧义时再补。**此前 2026-10-12 的结果：4070 放不下。** vLLM 0.31 加 `--quantization fp8` 之后模型仍占 11.03 GiB（嵌入、lm_head 和视觉塔保持 BF16），KV cache 剩 −2.2 GiB。改到第一次租卡会话：`QUANT=fp8 bash run_10b.sh`，再加 `QUANT=none` 跑一个 BF16 参照，合计约 ¥15，和第 8、11、14 步一起做。4070 上的结果文件只有零样本的。
    **在 4070 上跑 JEV-9B FP8，一次干两件事。** vLLM FP8，`--max-model-len 8192`，原封不动用 JEV-27B-DEMO 的评测代码。(a) 下限门槛：KL、ECE、机械臂 demo 成功率。(b) **评测代码校验：** 如果复现出公开的 VL-RewardBench 74.3% 和 KL 约 0.019，评测代码就可信，JEV-27B-VL 公开的 78.3% / AUROC 0.91 / KL 0.017 直接拿来当目标。不需要重跑教师模型。
11. **2026-10-13 在租的 2× RTX PRO 6000 上完成（AutoDL，torch 2.12.1+cu130，flash-linear-attention 0.5.2，无 causal-conv1d）。** 27B 全模型 BF16 + LoRA r16（1.088 亿参数）+ 决策头，开梯度检查点，24k token 的 micro-batch，单卡每步 128 行：加载 5 分钟，**预热后每步 48–59 秒（首步 96 秒），峰值显存 87.2 GB**。→ 单卡 3,700 步约 49 小时；两卡（各 64 行）约 26–29 小时。80 GB 的 H100 放不下这个 micro-batch；96 GB 余 9 GB。

**过关标准：** logits 对齐；LoRA 位置搞清楚；单步时间测出来。**如果用了备选方案（用 H 把权重旋转回去、按普通模型训练、导出时再把 A 旋转回来）之后 logits 还是对不齐，就停。**

### 第 2 周 — P1 训练器

12. **2026-10-12 已完成（代码 + CPU 冒烟）。** `train.py`：用 `bonsai_torch.build_model` 加载冻结底座（不含 lm_head 的文本模型，BF16 约 51 GB），peft LoRA r16/α32/dropout 0.05 挂在 JEV 的 10 个目标模块上（LoRA 参数 fp32，bf16 autocast），fp32 的 24 位决策头带偏置，用旋转回去的 lm_head 行初始化（`decision_head_rows`），KL + 0.5·RPS，30% 选择题打乱，分层的 128 行步，按长度分桶、受 token 预算限制的 micro-batch，梯度检查点，AdamW (0.9, 0.98) 两个学习率，余弦 + 3% 预热，裁剪 1，`torchrun` DDP 两卡，checkpoint/续跑（`last/`、`final/`：peft 适配器 + `head.safetensors` + `head.json`），CSV 日志，定期按类型算验证 KL。在 4070 机器上**用 CPU** 和截断到 2 层的模型冒烟（`--layers 2 --device cpu`）：损失下降，评估/保存/续跑都正常。可用训练行约 50.7 万 → 0.93 个 epoch 约 3,700 步。
    原规格：基于 `peft` 写 `train.py`：24 位 fp32 决策头，用 `lm_head` 对应行初始化；在有效位上算 KL(target‖model)，打分题加 0.5·RPS；选择题 30% 打乱选项增强；分层采样器（每类 ≥ 1/6）；按长度分桶；AdamW (0.9, 0.98)，决策头学习率 2e-4、LoRA 1e-4，余弦退火，3% 预热，梯度裁剪 1.0。去掉 `yuri_v1` 里 [0.5, 0.5] 占位标签的行。
13. **2026-10-12 已完成。** `infer_logits.py`（checkpoint → 每行 24 位 logits，和 `decide-logits` 同格式的 JSONL）、`calibrate.py`（在 `calibration` 上按类型用 L-BFGS 最小化 KL 拟合温度，限制在 [0.1, 10]，输出 JEV 格式的 `calibration.json` 并带拟合前后的 KL 和 ECE）、`eval_text.py`（按类型和来源算 KL、准确率、ECE，可选温度，零样本和训练后的 logits 通用）。共享的 `load_split()` 保证所有脚本的行筛选和顺序一致。在 600 行零样本样本上测过（拟合温度：noul 1.15、score 2.22、choice 1.62 → 底座过度自信；KL 0.463 → 0.441），也在 CPU 上用 2 层冒烟 checkpoint 走通了全流程。
    原规格：`calibrate.py`（在 `calibration` 划分上对每类用 L-BFGS 拟合温度）和 `eval_text.py`（在 `test_set_30k` 和 `ood` 上算 KL、ECE、准确率）。
14. **在 27B 上用 1% 训练数据冒烟** [PRO 6000 ×1，3–4 小时，约 ¥30]。损失必须下降；先跑只训决策头的版本（LoRA 冻结）做 sanity check，再跑完整版。存一个 checkpoint，对它跑 `eval_text.py`。
15. **导出已于 2026-10-12 写好并验证（合成的端到端测试在 4070 上通过：挂 LoRA 的第 0 层（线性注意力）6.1e-4、第 1 层 4.2e-4，决策头 logits 1.5e-3，决策头增量平均贡献 0.91，是残差的 600 倍）。** **真实 checkpoint 检查 2026-10-13 通过**（`check_export_real.py`，冒烟 checkpoint，1,024 行验证集）：分支 + 导出的 `adapter.gguf` 对训练它的 PyTorch 模型，平均 KL(pt‖fork) 0.0002，argmax 一致率 99.0%，对目标的 KL 0.1650 对 0.1647，接近平局的行上最大 |Δp| 0.09。
14. **冒烟 2026-10-13 在租的机器上完成：** 只训决策头 20 步验证 KL 0.388；全配方 1% 数据 40 步验证 KL **0.165**（零样本约 0.45），打分题准确率 16% → 68%。两卡 DDP 修了两处（rotary 的 `inv_freq` 留在 CPU；冻结底座和 Hadamard 缓冲区不参与 DDP 同步）。 `export_gguf_lora.py` 输出一个分支能用 `--lora` 加载的 GGUF LoRA：元数据 `general.type=adapter`、`adapter.type=lora`、`adapter.lora.alpha`；张量 `<gguf 名>.lora_a`（r × n_in）/ `.lora_b`（n_out × r），运行时缩放 α/r。导出时的两处布局修正：`attn_qkv.lora_b`、`attn_gate.lora_b` 的 v 头行和 `ssm_out.lora_a` 的 v 头列从 HF 的 grouped 顺序换回分支的 tiled 顺序。决策头表示成 `output.weight` 上的 rank-24 LoRA（A = (W_head − W_init)·24/α 抵消运行时缩放，B = 落到 24 个 verbalizer 行的 one-hot），正是 JEV"决策头当 lm_head LoRA"的做法，这样分支在 verbalizer id 上的 logits 就是决策头的 logits；决策头偏置和温度放进 `decision.json` 给服务封装层。已读分支的加载代码确认形状要求和 `output.weight` 可以挂 LoRA。`test_export.py` 不需要训练好的模型就能验证全部：第 0–1 层随机 LoRA + 随机决策头增量 → 导出 → 分支带 `--lora` 导出张量 → 和挂同样 LoRA 的 PyTorch 层比，以及和 (W_init+Δ)·result_norm 比。
    原规格：把这个 checkpoint 转成 GGUF LoRA 加决策头矩阵，在 4070 上用分支加载，在 200 行上比较 PyTorch 和分支的概率。

**过关标准：** 冒烟时 KL 下降；冒烟 checkpoint 上 PyTorch 和分支的差在 0.01 以内。

### 第 3 周 — P2 训练，同时在 4070 上做 P4 部署

16. **正式训练** [PRO 6000 ×2，约 28–33 小时]：**4,000 步**，两卡数据并行。按 JEV-27B 对齐（2026-10-13 对照其模型卡核实）：它训了 5,000 步 × 128 = 64 万行（0.98 个 epoch，单一余弦调度总长 5,124 步，3% 预热，退火到峰值的 0.02 倍，按验证 KL 选发布 checkpoint），其中 79% 是非占位行 → 50.6 万有效行 ≈ 我们的 4,000 步（我们直接丢掉 137,203 行占位数据而不是给 0.05 权重；保留 518,603 行）。`train.py` 现在默认 4,000 步、`--lr-floor 0.02`，并在 `last/` 之外按验证 KL 保留 `best/`。（之前写的 3,700 步 / 0.93 epoch / 4,750 步是 JEV-9B 两阶段配方的数字。）每 250 步存一次 checkpoint 到持久卷。把每类的损失和学习率记成 CSV；和 JEV-9B README 的曲线画在一起对比。
17. **2026-10-12 已完成（零样本模式；训练后的适配器只差一个 `--lora` 参数）。** `decide_server.py`（FastAPI）套在分支的 `llama-server` 上：`POST /v1/decide` 的请求和返回和 JEV 的 `serve_decide.py` 一致，所以 `jev_client.py` 和 `jev9b_eval.py` 不改一行就能评测我们的模型（已验证：demo 客户端和离线工具在 300 行上一致）。从 `/completion` 的 `n_probs=1000` 读该类型 verbalizer 的 logprob（全词表 softmax，语义和 JEV 的 `allowed_token_ids` 相同），加上 `decision.json` 里的决策头偏置和按类型温度。决策头本身在 LoRA 文件里（第 15 步），分支不需要任何自定义读取。`run_decide.sh start [适配器目录]` 以 detached 方式启动两个进程（`-np 2 --cache-ram 16384 --cache-reuse 256`；这个分支的 server 每次启动随机化媒体标记，封装层从 `/props` 读）。`test_decide.py` 通过：文本 50 行最大 |Δp| 1.25e-2、图片 5 对 5.2e-3（分支内核在不同 batch 大小下不是逐位一致；重复请求逐位相同），前缀复用正常。
    **前缀缓存需要预热。** 模型有循环层，server 只能从保存的边界（checkpoint，默认间隔 8192 token）恢复：冷的"状态+问题"提示词不能为下一个问题部分复用。`POST /v1/warm {state, kinds}` 先处理 `[kind] k\n[state] …`；之后对同一状态的每个 `/v1/decide` 只算问题本身（22–26 个 token，189 个图片+场景 token 已缓存），**4070 上每次追问约 0.2 秒**，冷启动 0.5–0.9 秒。JEV 模板带来的限制：`[kind]` 是第一个 token，所以为 `noul` 预热的状态不是 `choice` 提示词的前缀，要按类型分别预热。做其他 GPU 工作前先 `run_decide.sh stop`（占约 9 GB）。
18. 如果第一次训练在 `test_set_30k` 上的 KL 高于 0.03，加长训练或调学习率再跑一次。预算够跑三次。**之后某次训练可选的补充数据（借自榜首几家）：** 公开决策基准的 train/dev 划分（CLINC150、GSM8K、ARC、BoolQ、MMLU、NLI、iSarcasmEval、VAST 等）按 bare-v1 格式渲染、用金标签，对我们汇报的所有测试集做泄漏过滤。它提升 DI 公开分；对机器人/边缘产品来说 JEV 语料加业务数据更重要，优先级排在它们之后。

**过关标准：** 训练 KL 收敛；4070 上的 `/v1/decide` 和训练端概率差在 0.01 以内。

### 第 4 周 — P3 校准与评测

19. 拟合温度。如果有哪个明显偏离 1.0，当训练 bug 处理，不要靠校准遮掩。**借自 Decider 31B：** 温度按类型拟合成 T(n) = a + b·ln n（n 为选项数；每类三个参数，`calibrate.py` 里同一个 L-BFGS），报告是否优于常数；并在 `/v1/decide` 里加**条件二次读取**：最高概率低于 0.7 时把选项倒序再读一次，两次 log-softmax 取平均后套 T2(n)。Decider 报告约 28% 的问题触发，成本增约 30%，不是 2 倍。带与不带各测一次顺序一致率；这是修零样本位置偏好（51%）最便宜的办法。
20. 4070 上的文本评测：`test_set_30k` 和 `ood` 上的 KL、ECE、准确率。和 JEV-27B（0.017）、JEV-9B（0.019）以及第 10 步的零样本下限比。**同时跑 Jev Decision Index 的公开套件**（用它的评测包，走同一个 `/v1/decide` 协议），让模型有一个能和 JEV-27B 的 55.6 / JEV-27B-VL 的 69.6 并排的榜分；同一批行保留 BF16 PyTorch 模型的决策，供榜单对量化条目要求的 ≥ 96% 一致率检查。
21. **在 4070 上跑视觉评测：** 用 10b 校验过的评测代码，跑我们的模型在 VL-RewardBench、AgentRewardBench、MMRB2、QuickDraw 上的成绩，和 JEV-27B-VL 的公开数字比。补上图片任务的 ECE；没有公开参照，作为绝对指标报告。**只有 10b 复现不出 9B 的公开数字时才重跑教师模型** [PRO 6000 ×1，8–12 小时，¥60–85]：JEV-27B-VL BF16，`serve.sh --max-model-len 131072`，跑同样的评测集。
22. `text_parity`：同一个纯文本问题分别走视觉版和纯文本版，差 < 0.02。
23. 用 10b 的数字确认"比 JEV-9B 好否则停"这道关。

**过关标准：** 第 1 节的指标。如果视觉成绩明显低于 JEV-27B-VL 的 95%，就是提前做 P5（图片蒸馏）的信号。

### 第 5 周 — 加固，有 Orin 再上 Orin

23b. **System 2 检查（`run_system2.sh <adapter.gguf>`，4070 上约 10 分钟）。** System 2 就是同一个 `llama-server`，按请求把决策适配器关掉（`"lora": [{"id": 0, "scale": 0}]`），和 JEV 在 System 1 旁边提供未改动底座的做法一样。12 个贪心解码的提示词（算术、代码、翻译、逻辑、JSON、机器人步骤；关闭思考）：适配器关闭时必须和不加 `--lora` 启动的服务逐字节一致。**2026-10-14 用冒烟适配器验证：12/12 一致，首 token 的 logprob 差为 0。** 适配器打开时 12 个里有 5 个回答变了，所以 System 2 的请求绝不能继承适配器。**服务现在默认就是 System 2 安全的：** `run_decide.sh` 在 `llama-server` 启动后立即把适配器的全局 scale 设为 0（这个分支里单用 `--lora-init-without-apply` 不生效），`decide_server.py` 在 `/v1/decide` 和 `/v1/warm` 上按请求打开它。已用冒烟适配器验证：普通聊天请求在决策调用前后都和底座回答一致；决策仍然用到适配器。最终适配器出来后再跑一次。这样 System 2 的能力就严格等于 Bonsai 2 本身（厂商报告的相对 Qwen3.8 BF16 的保持率），我们的训练改变不了它。
24a. **并发检查 2026-10-14 已完成（冒烟适配器，`test_concurrency.py`）。** JEV-27B-VL 需要 `--max-num-seqs 8`，因为 vLLM 在那类模型上的 LoRA 路径超过 8 个序列时 System 1 概率会算错。我们的栈不受影响：分支的 `llama-server` 只把 LoRA 配置相同的槽位放进同一批（`can_batch_with`），每批设一次适配器。决策请求（适配器开）和聊天请求（适配器关）混着同时发实测：4 槽位 32 个决策、8 槽位 40 个决策 + 8 个聊天，和逐个发相比最大概率差 0.005，选择没有翻转，聊天首 token 概率差 0.008，8/8 聊天文本一致，0 次适配器串到聊天。最终适配器出来后再跑一次。
24. 在 4070 上对 `/v1/decide` 做并发压测，找到批处理的上限（分支和 bonjev 都是一次只服务一个请求）。测 640×480 和前缀缓存追问的延迟。这是本期能拿到的唯一延迟数据。
25. **可选的 M5 交叉验证，一天：** 把最终 LoRA 转成 MLX 格式，用自带的 `runtime/` 加载器加载，和 4070 对比 200 行。只有还想要 Mac mini 边缘服务器这个目标时才做。否则跳过。
26. **Orin NX**（只在采购了的情况下）：在 JetPack 下为 aarch64 / sm_87 交叉编译分支，复制同一套 GGUF 文件，重复和 4070 的 200 行对比，测 640×480 延迟。6 秒的目标只在这里验证。

### P5 — 图片决策训练轮（P4 之后做，以文本训练后的模型为对照）

**为什么做。** JEV-27B-VL 的看图决策全是零样本：它的 LoRA 和决策头只见过文本。我们的底座零样本已有 74.5%，但顺序一致率只有 50.9%（位置偏好）。图片决策数据是 JEV 没拉过的杠杆，也是超过它的唯一现实路径。文本侧超不过 JEV-27B（教师标签相同），P5 的目标在视觉侧。

**数据，三个来源可以叠加。**
1. 公开的图片回答人工偏好集：RLAIF-V、RLHF-V、VLFeedback、POVID（几万对，以幻觉为主，正对应 VL-RewardBench 最大的子集）。VL-RewardBench 和 MMRB2 本身绝不能拿来训练。
2. 用 JEV-27B-VL 在*最好状态*下打标签：两种顺序取平均，System 1 不确定时开 System 2 思考后再读分布。学生学到的是教师的上限，所以单次前向能超过教师的单次前向。一次 H100/PRO 6000 会话，约 5 万张图约 ¥100。
3. 业务图片（原 P5 的内容），用同样方法打标签。

**配方改动（不大）。**
- `bonsai_torch.build_model` 加上冻结的 Qwen3.8-27B 视觉塔（BF16，Bonsai 没有量化它），在 `no_grad` 下跑；`train.py` 用同一个 bare-v1 模板渲染图片行，占位符 `<|vision_start|><|image_pad|><|vision_end|>`，图片行和文本行用同一个分层采样器混合，避免文本 KL 退化。每一对的两种呈现顺序都作为训练行（这就是消除位置偏好的办法）。
- 显存：权重 +0.9 GB，没有优化器状态，图片 token 的激活开销和文本 token 一样 → 约 81 GB，PRO 6000 放得下。一张 640×480 约 300 个 token，24k token 的 micro-batch 放约 48 行图片行而不是约 100 行文本行。可选：图片集在约 2 万张以内且磁盘够用时，提前离线算好图片嵌入（每张 3 MB）。
- 费用：5 万行图片 × 约 500 token ≈ 2500 万 token 一轮，不到文本一轮的一半 → 2× PRO 6000 上每次约 ¥100–150。
- 评测：用第 10 步的评测代码跑 VL-RewardBench 和 MMRB2，报告**单顺序准确率和顺序一致率**，不只报平均后的总分；`test_set_30k` 的文本 KL 和 P3 结果相差不能超过 0.005。

**目标。** VL-RewardBench 总分 ≥ 80（JEV-27B-VL 为 78.3；1,247 对的置信区间约 ±2.4，所以要超 3 个点以上并且 MMRB2 同向才算赢），顺序一致率 ≥ 90%，hallucination ≥ 85，文本 KL 不变。

**开始前的门槛。** P3 已完成，文本训练后的模型和它的数字已记录，这样 P5 的收益才能归因。

**P5 能不能超过 JEV-27B？判断（2026-10-14）。**
- *文本：不能。* 同一个教师（Jev 1.13）的标签；JEV-27B 的 LoRA 挂在 BF16 底座上，我们的挂在三值底座上。蒸馏只能逼近教师。文本目标仍是"接近 0.017"，图片数据不能把它推高。
- *视觉：有一个真实杠杆，两个逆风。* 杠杆：JEV-27B-VL 的看图决策是零样本，从没见过图片决策数据。逆风 1：三值化伤视觉推理比伤文本多（Bonsai 自己的模型卡：MMMU-Pro 81.7 → 75.5，OCR Bench v2 61.0 → 56.9）；我们零样本的 74.5 靠两种顺序取平均撑着。逆风 2：能商用的图片偏好数据稀缺（RLAIF-V/RLHF-V 是非商用），发布版只能学 JEV-27B-VL 自己的标签，而它的 78.3 已经是两种顺序平均的结果，超过教师的余量很小。参照：DI 视觉榜第一的 d3 27B（全量微调、能看图）也只比 JEV-27B-VL 高 1.7 分（71.3 对 69.6）。

| 目标 | 超过 JEV-27B(-VL) 的可能性 |
|---|---|
| 文本 KL | < 5% |
| VL-RewardBench 超 3 分以上，研究版（可用非商用数据） | 40–50% |
| 同上，发布版（只用 JEV 标签和自有数据） | 约 25% |
| DI 视觉榜（覆盖面广，含私有集） | 15–20% |

**因此 P5 的首要目标是顺序一致性，不是总分：** 顺序一致率从 51% 到 ≥ 90%，单顺序准确率接近两序平均的准确率，让一次前向就可信（机器人场景只问一次）。产品说法是"约 8 GB 内与 JEV-27B-VL 相差 1–2 分以内"；超过它是额外收获。

**第一次训练（纯文本）后的决策规则，在 4070 上用第 10 步的评测代码测：**
| run1 的 VL-RewardBench（两序平均） | 解读 | P5 怎么做 |
|---|---|---|
| ≥ 77 | 文本训练已接近 JEV-27B-VL，三值化损失小 | 做 P5，目标超过 78.3（考虑 ±2.4 置信区间，≥ 81 才算赢），外加顺序一致率 ≥ 90% |
| 75–77 | 部分追回 | 做 P5，目标追平 78.3 并修好顺序一致性；对外说"持平"，不说"超过" |
| ≤ 75 | 三值化损失占上风 | P5 只为顺序一致性而做；放弃"超过 JEV"的目标；重新评估相对 4-bit 9B（DI 视觉榜 66.9）的边缘卖点是否还成立 |
每一档都同时记录单顺序准确率、顺序一致率、幻觉子集，并确认文本 KL 变化不超过 0.005。

**混合训练是最终形态。** P5 本身就是混合训练（文本行和图片行在同一个分层采样器里）。前面那轮纯文本只在第一次存在，作用是对照和用 JEV 公开曲线验证训练器。之后每次重训（加业务图片、换数据配比）都是一次从头的混合训练，一轮出结果，不再单独跑文本。

### 第一次训练结果（2026-10-14）

训练 21:48 结束（4,000 步，2× PRO 6000，约 23.8 小时）。验证 KL 已平：第 3,750 步和第 4,000 步都是 0.0180；`best` 为第 3,750 步。`test_set_30k`（PyTorch 路径，温度校准后）：

| Jev 标注的行（yuri_v3，25,376 行） | KL | 与 Jev 的 top-1 一致率 | 相对目标概率的 ECE* |
|---|---:|---:|---:|
| 全部 | **0.0165** | 91.9% | 0.0005 |
| noul | 0.0041 | 95.8% | 0.0007 |
| score | 0.0207 | 89.0% | 0.0007 |
| choice | 0.0248 | 90.7% | 0.0010 |
| JEV-27B（公开） | 0.017（noul 0.004 / score 0.021 / choice 0.025） | noul 95.9 / score 89.1 / choice 90.5 | 0.0009 |
| JEV-9B（公开） | 0.019（0.005 / 0.023 / 0.028） | | |

全部非占位行 0.0204（openjev_v2 为 0.063）。拟合温度 noul 1.014、score 1.000、choice 1.017（JEV-27B：1.014 / 1.004 / 1.016）：训练本身已经校准好了。**第一次训练就在 7.2 GB 的三值底座上复现了 JEV-27B 的文本成绩，不需要第二阶段。**
\* 相对软标签的可靠性（把每个选项的预测概率分箱，和目标概率比）。`eval_text.py` 里的 "ECE"（最大概率当置信度、对比 argmax 是否一致）对蒸馏到软标签的模型没有意义，不报告。

## 4. 预算

RTX PRO 6000 96GB，每卡每小时 ¥6.98。

| 项目 | 卡数 × 小时 | 费用 |
|---|---|---|
| P0 全模型对齐 + 内核检查 | 1 × 5–6 | 约 ¥40 |
| P1 冒烟 | 1 × 3–4 | 约 ¥30 |
| P2 训练，每次 | 2 × 22–27 | ¥310–380 |
| P3 教师模型重跑（条件执行） | 1 × 8–12 | ¥0 或 ¥60–85 |
| 数据盘扩到 103 GB，2 个月 | — | 很少 |
| **合计，训练 1 次** | | **约 ¥450–550** |
| **合计，训练 3 次** | | **约 ¥1100–1300** |
| 加 30% 调试余量 | | **≤ ¥1700（约 $240）** |
| Orin NX 16GB 开发套件（可选） | — | $600–900 |

机型可租到 2027-07-01；写作时有 2 张空闲。

## 5. 第 1 周前要定的事

1. 现在买 Orin NX，还是 P4b 以"4070 验证通过，Orin 待定"交付？
2. 单次决策可接受的延迟上限是多少？决定 Orin NX 还是 AGX Orin。
3. 有没有 JEV-9B FP8 基线的原始结果文件？没有的话第 23 步在 4070 上重跑。
4. 还要不要 Mac mini 边缘服务器这个目标？不要的话删掉第 25 步，M5 完全退出计划。
