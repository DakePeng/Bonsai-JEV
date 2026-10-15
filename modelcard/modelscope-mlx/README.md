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

# Bonsai-JEV（MLX）

**跑在 Apple Silicon 上的 Bonsai-JEV。** 一个能看图、输出校准概率的 27B 决策模型：对一个有类型的问题（是/否、0–5 打分、2–16 选 1）提问，状态里可以带图，一次前向就输出各选项的校准概率。这个仓库是 MLX 版：Prism ML 的 Ternary Bonsai 2 27B MLX 2-bit 包，加上 Bonsai-JEV 的 LoRA 和决策头，在 16 GB 的 Mac 上一条命令就能起服务。

**代码和训练：** [github.com/DakePeng/Bonsai-JEV](https://github.com/DakePeng/Bonsai-JEV) · GGUF 版（CUDA / Vulkan / Metal，8 GB 显卡可用）：[dakepeng/Bonsai-JEV](https://modelscope.cn/models/dakepeng/Bonsai-JEV) · 英文模型卡：[Hugging Face](https://huggingface.co/dakebakery/Bonsai-JEV-mlx)

> **独立项目。** Bonsai-JEV 由一位独立开发者完成，与 AutoTrust（autotrust）、TypeSafe、Prism ML 均无隶属关系，也未获得它们的认可或赞助。文中提到它们的名字，只是为了致谢和注明本项目所基于的模型与数据；相关商标归各自所有者。

## 基于

- **[Ternary Bonsai 2 27B](https://modelscope.cn/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)，作者 Prism ML**（Apache 2.0）。*Created using Bonsai by Prism ML.* 它的 MLX 2-bit 包和自带的运行时原样收录在本仓库（按 Prism 的 `files.json` 核对过 SHA-256）。Bonsai 2 基于阿里云的 Qwen3.8-27B（Apache 2.0）。
- **[JEV-9B](https://huggingface.co/autotrust/JEV-9B) 和 [JEV-27B](https://huggingface.co/autotrust/JEV-27B)，作者 autotrust**（Apache 2.0）。训练配方、提示词模板、verbalizer 和 `/v1/decide` 接口都来自它们；它们公开的数字是我们的目标。

## 快速上手

需要 16 GB 及以上统一内存的 Apple Silicon。

```bash
modelscope download --model dakepeng/Bonsai-JEV-mlx --local_dir Bonsai-JEV-mlx
cd Bonsai-JEV-mlx
pip install -r runtime/requirements.txt fastapi uvicorn
python mlx_serve.py --pack . --adapter . --port 8000
```

```bash
curl -s localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "kind": "choice",
  "state": ["摄像头画面：", {"image": "frame.jpg"}, "\n夹爪张开，位于桌面上方。"],
  "question": "机械臂应该先抓哪个物体？",
  "options": ["红色方块", "蓝色方块", "都不抓，场景不安全"]}'
```

- `kind`：`noul`（选项 `false`/`true`）、`score`（`0`–`5`）、`choice`（2–16 个选项，标为 A–P）。
- `state`：字符串、JSON 对象，或文本和 `{"image": 路径 | 网址 | data-URL}` 混合的列表。
- 请求和返回格式和 GGUF 版、JEV 的 `serve_decide.py` 一致。一次处理一个请求。
- 不起服务也能用：`python mlx_decide.py --pack . --adapter .` 会跑一个示例决策；在 Python 里 `import mlx_decide` 后调用 `decide(...)`。

## 原理

Prism 的运行时（`runtime/`）把这个包加载成 mlx-vlm 的 Qwen3.5 模型，每个线性层先对输入做 block-1024 的 Hadamard 旋转，再做 2-bit 量化矩阵乘法。Bonsai-JEV 的 LoRA（秩 16，全部 64 层）训练时作用在未旋转的层输入上，值头的排列顺序也相同，所以 `mlx_decide.py` 直接把每层包成 `base(x) + B·A·x · α/r`，不需要任何转换；决策头（24 × 5120 + 偏置，fp32）读取 `[decision]:` 处最后一层归一化后的隐状态。按类型的温度来自 `decision.json`。视觉塔是 Prism 原装的；图片按原生分辨率经包里的 Qwen2-VL 预处理器处理。

## 结果

和 [dakepeng/Bonsai-JEV](https://modelscope.cn/models/dakepeng/Bonsai-JEV) 是同一套权重和适配器，效果数字见那张卡（`test_set_30k` 上对 Jev 1.13 的文本 KL 0.0163，JEV-27B 0.017；VL-RewardBench 78.0，JEV-27B-VL 78.3；ERQA 52.0）。这个版本能复现出同样的结果：

| 和 CUDA 版（RTX 4070 上的 llama.cpp 分支）对比 | 结果 |
|---|---|
| 文本，随机 300 行 `test_set_30k` | KL(CUDA ‖ MLX) 0.00000，top-1 一致率 100%，两边对 Jev 的 KL 都是 0.0195 |
| 看图，82 条 VL-RewardBench 提示（41 对 × 2 种顺序） | 96.3% 判断一致，平均 \|ΔP(A)\| 0.014（两种运行时缩放图片的方式不同） |

**速度和内存，Mac mini M4，16 GB：**

| | |
|---|---|
| 加载（底座 + LoRA + 决策头） | 5–6 秒 |
| 内存峰值 | 文本 8.6 GiB，看图 9.7 GiB |
| 文本决策，约 84 个输入 token | p50 2.1 秒，p95 4.3 秒 |
| 一张图 + 两段回答，约 620 个输入 token | p50 13.8 秒，p95 26.7 秒 |

一次决策就是对提示词做一次前向，不生成文字。更大的 Apple 芯片（Pro / Max / Ultra）GPU 核心更多，应该相应更快（未实测）。

## 文件

| 文件 | 内容 |
|---|---|
| `model.safetensors`、`config.json`、`hadamard.json`、`tokenizer*.json`、`chat_template.jinja`、`generation_config.json`、`preprocessor_config.json`、`video_preprocessor_config.json`、`files.json`、`runtime/`、`PACK-RUNTIME.md`、`*-validation.json` | Prism ML 的 MLX 2-bit 包和加载器，原样收录（8.6 GB）；`runtime/` 为 MIT 许可，© Apple，见 `runtime/LICENSE` |
| `peft/` | Bonsai-JEV 的 LoRA（peft 格式，fp32，0.44 GB） |
| `head.safetensors`、`head.json`、`decision.json`、`calibration.json` | 决策头、verbalizer id、按类型温度 |
| `mlx_decide.py`、`mlx_serve.py` | MLX 决策代码和 `/v1/decide` 服务 |
| `configuration.json` | ModelScope 仓库配置 |
| `LICENSE`、`NOTICE` | Bonsai-JEV 的 Apache 2.0 许可证和署名 |
| `LICENSE-Ternary-Bonsai-2`、`NOTICE-Ternary-Bonsai-2.txt` | Prism ML 为底座提供的许可证和 NOTICE，原样附上 |

适配器和这个底座绑定（Hadamard 元数据、值头排列），请使用本仓库附带的包。

## 局限

- **质量天花板是 JEV 水平**；和 JEV 同一个教师，文本上和 JEV-27B 持平，不会超过它。迁移到没见过的任务族更弱（KL 0.128 对 0.104）。
- **内存：** MLX 包是 2-bit（PTQ1_0 那种更紧凑的打包没有 MLX 版），需要 16 GB 的 Mac，8 GB 的 Mac 放不下。8 GB 显卡请用 GGUF 版的 PTQ1_0。
- **看图是零样本**，和 JEV-27B-VL 一样；多图推理比单图弱（ERQA 41.6 对 56.1）。
- **这里没有 System 2 服务。** 要聊天，在同一个包上不挂适配器跑 Prism 的 `quickstart.py`。
- **英文提示词。** 训练数据是英文的。

## 许可证和致谢

适配器、决策头和脚本采用 Apache 2.0。*Created using Bonsai by Prism ML*；Ternary Bonsai 2 27B © Prism ML, Inc.，Apache 2.0，基于 Qwen3.8-27B（© 阿里云，Apache 2.0），原样再分发并附带其许可证和 NOTICE。配方和部署设计来自 autotrust 的 JEV-9B / JEV-27B（Apache 2.0）。训练数据为 [SargeDev/jev-distill-corpus-v3](https://modelscope.cn/datasets/SargeDev/jev-distill-corpus-v3)（Apache 2.0；`openjev_v2` 为 CC0），和 JEV-9B、JEV-27B 是同一份语料；其中 `yuri_v3` 的标签是 TypeSafe 的 Jev 1.13 的输出，商用前建议自行查阅 TypeSafe 的 API 条款。评测数据只用于评测。

感谢 Prism ML、autotrust、SargeDev 和 Qwen 团队。
