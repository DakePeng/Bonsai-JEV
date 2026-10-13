# 租卡会话 1：步骤 5、8、11、14、16、19、20（约 27 小时，2× RTX PRO 6000 96GB，同机两卡）

双卡不需要 NVLink：LoRA 训练只同步 4000 万参数的梯度，PCIe 够用。所有长任务都用 `setsid nohup` 挂起，SSH 断了不受影响。JEV-9B 基线（10b）本轮不做。

## 时间表

| 时刻 | 卡 | 任务 | 时长 |
|---|---|---|---|
| T+0:00 | — | 步骤 5 环境 + 下载 | 0.7 h |
| T+0:40 | 卡 0 | 步骤 11 训练内核检查 + 步骤 8 整模型检查 | 0.7 h |
| T+1:20 | 卡 0 | 步骤 14 冒烟（决策头 sanity + 1% 数据）、导出 | 1 h |
| T+2:30 | 两卡 | 步骤 16 正式训练，4,000 步 | 22–27 h |
| T+26 | 卡 0 | 步骤 19、20 校准和评测，导出 | 1.5 h |
| T+27 | — | 打包、拷出、释放 | 0.3 h |

## 租之前

- 镜像：基础镜像 `PyTorch / 2.12.1 / 3.12 (ubuntu22.04) / 13.0`（CUDA ≥ 12.8 即可，脚本会建自己的 venv）。GPU 数量 2。
- 磁盘：A32 机数据盘 50 GB 不可扩容，系统盘 30 GB。`bootstrap.sh` 把环境和模型都放数据盘（`~/autodl-tmp`，`~/bonsai` 是软链），pip 不留缓存。占用：Bonsai 8 GB + 语料 0.5 GB + venv 约 6 GB + checkpoint 约 4 GB ≈ 19 GB。不要把 54 GB 的 BF16 底座落盘。
- 4070 开机，关掉 Windows 睡眠（第 8 步要从它拷分支的导出张量，第 15 步的真实检查在它上面做）。
- 把仓库推到租的机器：`git clone` 不通就 `scp -r` 整个 `Bonsai-JEV`。
- 余额 ≥ ¥500，扣到 0 实例会被关掉。

## T+0:00 — 步骤 5：环境（30–40 分钟）

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
记录：`step N ... tok ... Xs` 里的秒数和 `nvidia-smi` 峰值。按 3,700 步换算总时长；如果单卡一步超过 50 秒，训练时间估算要上调。

```bash
# 8: 全模型 PyTorch 对分支的最终隐藏状态（分支的导出文件从 4070 拷过来：~/bonsai/test/dump2，约 30 MB）
scp -r <4070>:~/bonsai/test/dump2 ~/bonsai/test/
CUDA_VISIBLE_DEVICES=0 python3 - <<'EOF'
import torch, numpy as np, os
from bonsai_torch import Bonsai, build_model, text_config
b, cfg = Bonsai(os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf")), text_config(os.path.expanduser("~/bonsai/models/qwen3.8-27b-cfg"))
m = build_model(b, cfg, dtype=torch.bfloat16, device="cuda")
toks = torch.tensor([[int(l) for l in open(os.path.expanduser("~/bonsai/test/dump2/tokens.txt")) if l.strip()]], device="cuda")
with torch.no_grad(): h = m(input_ids=toks).last_hidden_state[0, -1].float()
ref = torch.from_numpy(np.fromfile(os.path.expanduser("~/bonsai/test/dump2/result_norm.f32"), dtype=np.float32))
print("result_norm rel diff:", ((h.cpu() - ref).abs().max() / ref.abs().max()).item())
EOF
```
过关：相对差 ≤ 1e-2（BF16 全模型 64 层累积；第 7 步逐层是 3.5e-4）。

## T+1:20 — 卡 0：步骤 14 冒烟（约 1 小时）

```bash
# 先只训决策头（LoRA 学习率 0）的 sanity：
CUDA_VISIBLE_DEVICES=0 python3 train.py --out ckpt/smoke-head --steps 20 --train-limit 5000 --eval-rows 512 --eval-every 20 --save-every 20 --lr-lora 0 2>&1 | tee results/smoke-head.log
# 再全量配方 1% 数据：
CUDA_VISIBLE_DEVICES=0 python3 train.py --out ckpt/smoke --steps 40 --train-limit 5000 --eval-rows 512 --eval-every 20 --save-every 20 2>&1 | tee results/smoke.log
python3 infer_logits.py --ckpt ckpt/smoke/final --split validation --limit 1024 --out results/smoke_val.logits.jsonl
python3 eval_text.py --logits results/smoke_val.logits.jsonl --split validation --limit 1024 --out results/smoke_val
python3 export_gguf_lora.py --ckpt ckpt/smoke/final --out ckpt/smoke/final/adapter.gguf
scp -r ckpt/smoke/final <4070>:~/bonsai/ckpt-smoke
```
过关：两条曲线都在降，`eval` 的 KL 低于零样本（验证集零样本约 0.45）。导出的适配器送回 4070 后做第 15 步的真实检查（200 行 PyTorch 对分支，在 4070 上跑，不占租卡时间）。

## T+2:30 — 两张卡：步骤 16 正式训练（22–27 小时）

```bash
cd ~/bonsai/jev && setsid nohup torchrun --nproc_per_node 2 train.py --out ckpt/run1 --steps 4000 > results/run1.log 2>&1 < /dev/null &
tail -f results/run1.log        # 每 10 步一行；每 250 步一次验证 KL 和一个 checkpoint（ckpt/run1/last）
```
看什么：`kl n/s/c` 三类都要降；验证 KL 的目标是 0.03 以内；梯度范数不应该长期 > 10。中途断了：同一命令加 `--resume`。
**每 2 小时把 `ckpt/run1/last`（1.8 GB）拉到 Mac 的 `rental/stage/ckpt/`**（只留最新一份；Mac 不要休眠），租的盘不是永久的。训练中**不要关机/重启**：关机会释放 GPU，这台只有两张空闲卡，再开机可能拿不到。

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
tar czf run1-release.tgz ckpt/run1/final results/run1_* results/smoke* results/kernel-check.log
scp run1-release.tgz <4070>:~/bonsai/   # 或下载到本机
```
**释放前先重启一次**挂上文件存储（`/root/autodl-fs`，西北B区已初始化，前 20 GB 免费，实例释放后保留），把 Bonsai 权重 + 语料 + 适配器 + 结果拷进去（约 12 GB），下次租卡一开机就全在，免下载。此时有没有卡已无所谓。然后释放实例。看图评测（第 21 步）和服务对比（第 15、17 步）都在 4070 上做，不占租卡时间。

环境注意（2026-10-13 实测）：pip 用华为源（阿里 0.17 MB/s、清华 0.28、华为 1.1）；venv 继承镜像自带的 torch 2.12.1+cu130，不另装 torch；`causal-conv1d` 要 nvcc 从源码编，跳过，只装 `flash-linear-attention`；ModelScope 下载约 3 MB/s，8.6 GB 权重约 45 分钟，下次从文件存储拷。

## 时间和费用

| 段 | 卡 | 时长 | 费用（¥6.98/卡时） |
|---|---|---|---|
| 环境 + 下载 | 2（空转） | 0.7 h | ¥10 |
| 11 + 8 | 卡 0 | 0.7 h | ¥10 |
| 14 冒烟 | 卡 0 | 1 h | ¥14 |
| 16 正式训练 | 2 | 22–27 h | ¥310–380 |
| 19 + 20 | 卡 0 | 1.5 h | ¥20 |
| **合计** | | **约 27 h** | **约 ¥360–430** |

两张卡从开机起一起计费，前 2.5 小时空着的那张约 ¥17，不值得为此分两次租。
