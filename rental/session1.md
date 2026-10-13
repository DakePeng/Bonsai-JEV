# 租卡会话 1：步骤 5、8、10b、11、14、16、19、20

**单卡版（1× PRO 6000 96GB，约 50 小时，约 ¥350）是默认。** 双卡不需要 NVLink：LoRA 训练只同步 4000 万参数的梯度，同机两张 PCIe 卡即可，时间减半、费用不变；租到了就按文末"双卡差异"改两处。所有长任务都用 `setsid nohup` 挂起，SSH 断了不受影响。

## 单卡时间表

| 时刻 | 任务 | 时长 |
|---|---|---|
| T+0:00 | 步骤 5 环境 + 下载 | 0.7 h |
| T+0:40 | 步骤 11 训练内核检查 + 步骤 8 整模型检查 | 0.7 h |
| T+1:20 | 步骤 10b JEV-9B 基线，fp8 和 bf16 各一遍 | 1 h |
| T+2:20 | 步骤 14 冒烟（决策头 sanity + 1% 数据） | 1 h |
| T+3:20 | 步骤 16 正式训练，3,700 步 | 40–48 h |
| T+48 | 步骤 19、20 校准和评测，导出 | 1.5 h |
| T+50 | 打包、拷出、释放 | 0.3 h |

下面各段的命令里 `CUDA_VISIBLE_DEVICES=0/1` 单卡时一律去掉，顺序按上表。

## 租之前

- 数据盘扩到 100 GB（Bonsai 8 GB + JEV-9B 18 GB + Qwen3.5-9B 19 GB + 语料 0.5 GB + checkpoint 若干个 0.5 GB）。不要把 54 GB 的 BF16 底座落盘。
- 4070 开机，关掉 Windows 睡眠（第 15 步的真实检查和 10b 的离线对照数据都在它上面）。
- 把仓库推到租的机器：`git clone` 不通就 `scp -r` 整个 `Bonsai-JEV`。

## T+0:00 — 步骤 5：环境（30–40 分钟，两张卡都空着）

```bash
cd ~/Bonsai-JEV && bash rental/bootstrap.sh 2>&1 | tee bootstrap.log
```
过关：末尾打印 `selfcheck ok: 851 tensors, 401 folded, block 1024`，以及 `flash-linear-attention OK`。fla 装不上也能继续，但第 11 步的时间要重新估。

## T+0:40 — 卡 0：步骤 11 训练内核 + 步骤 8 整模型检查（40 分钟）

```bash
cd ~/bonsai/jev && . ~/bonsai/venv/bin/activate
# 11: 全 27B，挂 LoRA，开检查点，24k token 的 micro-batch 跑 3 步；看单步时间和峰值显存
CUDA_VISIBLE_DEVICES=0 python3 train.py --out ckpt/kernel-check --steps 3 --rows-per-step 128 --max-tokens 24000 --eval-rows 64 --eval-every 3 --save-every 3 --train-limit 20000 2>&1 | tee results/kernel-check.log
```
记录：`step N ... tok ... Xs` 里的秒数和 `nvidia-smi` 峰值。按 3,700 步换算总时长；如果一步超过 40 秒（两卡折算），训练时间估算要上调。

```bash
# 8: 全模型 PyTorch 对分支的 logits（分支的导出文件从 4070 拷过来：~/bonsai/test/dump2，约 30 MB）
scp -r <4070>:~/bonsai/test/dump2 ~/bonsai/test/
CUDA_VISIBLE_DEVICES=0 python3 - <<'EOF'
import torch, numpy as np, os
from bonsai_torch import Bonsai, build_model, decision_head_rows, text_config
b, cfg = Bonsai(os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf")), text_config(os.path.expanduser("~/bonsai/models/qwen3.8-27b-cfg"))
m = build_model(b, cfg, dtype=torch.bfloat16, device="cuda")
toks = torch.tensor([[int(l) for l in open(os.path.expanduser("~/bonsai/test/dump2/tokens.txt")) if l.strip()]], device="cuda")
with torch.no_grad(): h = m(input_ids=toks).last_hidden_state[0, -1].float()
ref = torch.from_numpy(np.fromfile(os.path.expanduser("~/bonsai/test/dump2/result_norm.f32"), dtype=np.float32))
print("result_norm rel diff:", ((h.cpu() - ref).abs().max() / ref.abs().max()).item())
EOF
```
过关：相对差 ≤ 1e-2（BF16 全模型 64 层累积；第 7 步逐层是 3.5e-4）。

## T+0:40 — 卡 1（并行）：步骤 10b JEV-9B 基线（约 1 小时）

```bash
cd ~/bonsai/jev && CUDA_VISIBLE_DEVICES=1 QUANT=fp8 GPU_UTIL=0.9 bash run_10b.sh 2>&1 | tee results/10b-fp8.log
CUDA_VISIBLE_DEVICES=1 QUANT=none GPU_UTIL=0.9 bash run_10b.sh 2>&1 | tee results/10b-bf16.log
```
过关：VL-RewardBench overall 接近公开的 74.3，文本 KL 接近 0.019。对上了，评测代码可信，JEV-27B-VL 的公开数字直接当目标，第 21 步的教师重跑取消。对不上，先查 `jev_client` 的协议差异再说。

## T+1:20 — 卡 0：步骤 14 冒烟（约 1 小时）

```bash
# 先只训决策头（LoRA 冻结）的 sanity：
CUDA_VISIBLE_DEVICES=0 python3 train.py --out ckpt/smoke-head --steps 20 --train-limit 5000 --eval-rows 512 --eval-every 20 --save-every 20 --lr-lora 0 2>&1 | tee results/smoke-head.log
# 再全量配方 1% 数据：
CUDA_VISIBLE_DEVICES=0 python3 train.py --out ckpt/smoke --steps 40 --train-limit 5000 --eval-rows 512 --eval-every 20 --save-every 20 2>&1 | tee results/smoke.log
python3 infer_logits.py --ckpt ckpt/smoke/final --split validation --limit 1024 --out results/smoke_val.logits.jsonl
python3 eval_text.py --logits results/smoke_val.logits.jsonl --split validation --limit 1024 --out results/smoke_val
python3 export_gguf_lora.py --ckpt ckpt/smoke/final --out ckpt/smoke/final/adapter.gguf
scp -r ckpt/smoke/final <4070>:~/bonsai/ckpt-smoke
```
过关：两条曲线都在降，`eval` 的 KL 低于零样本（验证集零样本约 0.45）。导出的适配器送回 4070 后做第 15 步的真实检查（200 行 PyTorch 对分支，由 4070 那边跑）。

## T+3:20 — 步骤 16 正式训练（单卡 40–48 小时；双卡 22–27 小时）

```bash
cd ~/bonsai/jev && setsid nohup python3 train.py --out ckpt/run1 --steps 3700 > results/run1.log 2>&1 < /dev/null &
# 双卡改为：setsid nohup torchrun --nproc_per_node 2 train.py --out ckpt/run1 --steps 3700 > results/run1.log 2>&1 < /dev/null &
tail -f results/run1.log        # 每 10 步一行；每 250 步一次验证 KL 和一个 checkpoint（ckpt/run1/last）
```
看什么：`kl n/s/c` 三类都要降；验证 KL 的目标是 0.03 以内；梯度范数不应该长期 > 10。中途断了：同一命令加 `--resume`。
**每隔几小时把 `ckpt/run1/last` 拷出去一份**（scp 到 4070 或下载到本地），租的盘不是永久的。

## T+26:00 — 卡 0：步骤 19、20 校准和评测（约 1.5 小时）

```bash
python3 infer_logits.py --ckpt ckpt/run1/final --split calibration --out results/run1_calib.logits.jsonl
python3 calibrate.py --logits results/run1_calib.logits.jsonl --split calibration --out ckpt/run1/final/calibration.json
python3 infer_logits.py --ckpt ckpt/run1/final --split test_set_30k --out results/run1_test30k.logits.jsonl
python3 eval_text.py --logits results/run1_test30k.logits.jsonl --split test_set_30k --calibration ckpt/run1/final/calibration.json --out results/run1_test30k
python3 infer_logits.py --ckpt ckpt/run1/final --split ood --out results/run1_ood.logits.jsonl
python3 eval_text.py --logits results/run1_ood.logits.jsonl --split ood --calibration ckpt/run1/final/calibration.json --out results/run1_ood
python3 export_gguf_lora.py --ckpt ckpt/run1/final --out ckpt/run1/final/adapter.gguf --calibration ckpt/run1/final/calibration.json
```
过关：`test_set_30k` KL ≤ 0.03，三个温度在 0.9–1.1 之间。拟合出的温度明显偏离 1 就是训练有问题，先别校准。

## 收尾（15 分钟）

```bash
tar czf run1-release.tgz ckpt/run1/final results/run1_* results/smoke* results/10b-* results/kernel-check.log results/jev9b_*
scp run1-release.tgz <4070>:~/bonsai/   # 或下载到本机
```
然后释放实例。看图评测（第 21 步）和服务对比（第 15、17 步）都在 4070 上做，不占租卡时间。

## 时间和费用

| 段 | 单卡时长 | 单卡费用（¥6.98/卡时） | 双卡时长 |
|---|---|---|---|
| 环境 + 下载 | 0.7 h | ¥5 | 0.7 h |
| 11 + 8 | 0.7 h | ¥5 | 0.7 h（与 10b 并行） |
| 10b 两遍 | 1 h | ¥7 | 并行 |
| 14 冒烟 | 1 h | ¥7 | 1 h |
| 16 正式训练 | 40–48 h | ¥280–335 | 22–27 h |
| 19 + 20 | 1.5 h | ¥10 | 1.5 h |
| **合计** | **约 50 h** | **约 ¥320–370** | **约 28 h，¥350–420** |

## 双卡差异（租到同机两张时）

1. 第 11 步和第 8 步放卡 0，10b 同时放卡 1（命令前加 `CUDA_VISIBLE_DEVICES=0` / `=1`）。
2. 第 16 步用 `torchrun --nproc_per_node 2`。
其余不变。
