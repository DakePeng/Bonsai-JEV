"""train.py - JEV recipe on the frozen Ternary Bonsai 2 27B: LoRA r16 + 24-slot decision head.

Recipe (JEV-9B README): LoRA r=16 a=32 dropout 0.05 on in_proj_qkv/in_proj_z/out_proj/q/k/v/o/gate/up/down;
fp32 head (hidden -> 24 slots) initialised from the lm_head rows of the verbalizer tokens; loss =
KL(target||model) over the kind's active slots + 0.5*RPS for score; 30% of choice rows get their
options shuffled; 128 rows/step stratified by kind (>= 1/6 each), length-bucketed micro-batches under a
token budget, gradient checkpointing; AdamW (0.9, 0.98), head 2e-4 / LoRA 1e-4, cosine, 3% warmup, clip 1.

Single GPU:   python train.py --out ckpt/run1
Two GPUs:     torchrun --nproc_per_node 2 train.py --out ckpt/run1
Local smoke:  python train.py --layers 2 --device cpu --max-tokens 512 --rows-per-step 8 --steps 3 --out /tmp/smoke
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from bonsai_torch import Bonsai, build_model, decision_head_rows, text_config
from zeroshot_text import RANGES, VERBALIZER_IDS, render

LORA_TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
KINDS = ["noul", "score", "choice"]


# ----------------------------------------------------------------------------- data
def load_rows(path, drop_placeholder=True):
    rows = [json.loads(l) for l in open(path)]
    if drop_placeholder:
        rows = [r for r in rows if not (r["source"] == "yuri_v1" and r["target"] == [0.5, 0.5])]
    return rows


def maybe_shuffle_choice(r, rng, p=0.3):
    if r["kind"] != "choice" or rng.random() >= p:
        return r
    perm = list(range(len(r["options"])))
    rng.shuffle(perm)
    return {**r, "options": [r["options"][i] for i in perm], "target": [r["target"][i] for i in perm]}


class Sampler:
    """Yields steps of `rows_per_step` rows: >= 1/6 from each kind, rest uniform; ~0.93 epoch total."""

    def __init__(self, rows, rows_per_step, seed, rank=0, world=1):
        self.by_kind = {k: [r for r in rows if r["kind"] == k] for k in KINDS}
        self.rows, self.n, self.rng = rows, rows_per_step, random.Random(seed + rank)
        self.rank, self.world = rank, world

    def step(self):
        per_kind = self.n // 6
        pick = [self.rng.choice(self.by_kind[k]) for k in KINDS for _ in range(per_kind)]
        pick += self.rng.sample(self.rows, self.n - len(pick))
        self.rng.shuffle(pick)
        return [maybe_shuffle_choice(r, self.rng) for r in pick]


def micro_batches(rows, tok, max_tokens, pad_id):
    """Length-bucketed micro-batches whose padded token count stays under max_tokens."""
    enc = [tok(render(r), add_special_tokens=False)["input_ids"] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(enc[i]))
    cur, out = [], []
    for i in order:
        L = max([len(enc[j]) for j in cur] + [len(enc[i])])
        if cur and L * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    for idx in out:
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for k, i in enumerate(idx):
            ids[k, :len(enc[i])] = torch.tensor(enc[i])
            mask[k, :len(enc[i])] = 1
        last = torch.tensor([len(enc[i]) - 1 for i in idx])
        yield [rows[i] for i in idx], ids, mask, last


# ----------------------------------------------------------------------------- loss
def decision_loss(logits24, batch_rows, device):
    """KL(target||model) over active slots (+0.5*RPS for score). Returns (loss_sum, per-kind sums, counts)."""
    total = logits24.new_zeros(())
    sums = {k: 0.0 for k in KINDS}
    cnt = {k: 0 for k in KINDS}
    for i, r in enumerate(batch_rows):
        lo, _ = RANGES[r["kind"]]
        n = len(r["options"])
        z = logits24[i, lo:lo + n].float()
        t = torch.tensor(r["target"], device=device, dtype=torch.float32)
        logp = F.log_softmax(z, -1)
        kl = torch.sum(torch.where(t > 0, t * (torch.log(t.clamp_min(1e-12)) - logp), torch.zeros_like(t)))
        loss = kl
        if r["kind"] == "score":
            cdf_p, cdf_t = logp.exp().cumsum(-1), t.cumsum(-1)
            loss = loss + 0.5 * torch.sum((cdf_p - cdf_t) ** 2) / (n - 1)
        total = total + loss
        sums[r["kind"]] += kl.item()
        cnt[r["kind"]] += 1
    return total, sums, cnt


# ----------------------------------------------------------------------------- model
def build(args, b, cfg, device, dtype):
    from peft import LoraConfig, get_peft_model
    model = build_model(b, cfg, dtype=dtype, device=device, n_layers=args.layers)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, lcfg)
    for n, p in model.named_parameters():        # LoRA params in fp32 for AdamW; base stays bf16
        if p.requires_grad:
            p.data = p.data.float()
    head = nn.Linear(cfg.hidden_size, 24, bias=True, dtype=torch.float32).to(device)
    with torch.no_grad():
        head.weight.copy_(decision_head_rows(b, VERBALIZER_IDS).to(device))
        head.bias.zero_()
    return model, head


def forward_hidden(model, ids, mask, last, device, dtype):
    with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    h = out.last_hidden_state
    return h[torch.arange(h.shape[0], device=device), last.to(device)].float()   # final-norm state of "[decision]:"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--cfg", default=os.path.expanduser("~/bonsai/models/qwen3.8-27b-cfg"))
    ap.add_argument("--data", default=os.path.expanduser("~/bonsai/data/jev-distill-corpus-v3"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, default=3700, help="~0.93 epoch of ~507k rows at 128/step")
    ap.add_argument("--rows-per-step", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=24000, help="padded tokens per micro-batch")
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--train-limit", type=int, default=0, help="use only the first N train rows (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = args.device
    dtype = getattr(torch, args.dtype)
    is_main = rank == 0
    log = (lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)) if is_main else (lambda *a: None)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.cfg)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    b, cfg = Bonsai(args.gguf), text_config(args.cfg)
    log("loading model", "layers", args.layers or cfg.num_hidden_layers)
    model, head = build(args, b, cfg, device, dtype)
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])
        head = DDP(head, device_ids=[rank])
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable: lora {n_lora/1e6:.1f}M, head {sum(p.numel() for p in head.parameters())/1e3:.1f}k")

    train = load_rows(os.path.join(args.data, "train.jsonl"))
    if args.train_limit:
        train = train[:args.train_limit]
    val = load_rows(os.path.join(args.data, "validation.jsonl"))
    random.Random(1).shuffle(val)
    val = val[:args.eval_rows]
    log(f"train rows {len(train)}, val rows {len(val)}")
    sampler = Sampler(train, args.rows_per_step // world, args.seed, rank, world)

    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_lora},
        {"params": list(head.parameters()), "lr": args.lr_head},
    ], betas=(0.9, 0.98), weight_decay=0.0, fused=device.startswith("cuda"))
    warm = int(args.warmup * args.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))

    step = 0
    ck = os.path.join(args.out, "last")
    if args.resume and os.path.exists(os.path.join(ck, "state.pt")):
        st = torch.load(os.path.join(ck, "state.pt"), map_location=device)
        (model.module if world > 1 else model).load_state_dict(st["lora"], strict=False)
        (head.module if world > 1 else head).load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        sampler.rng.setstate(st["rng"])
        log(f"resumed at step {step}")

    def evaluate():
        model.eval()
        sums, cnt = {k: 0.0 for k in KINDS}, {k: 0 for k in KINDS}
        with torch.no_grad():
            for rows, ids, mask, last in micro_batches(val, tok, args.max_tokens, pad_id):
                h = forward_hidden(model, ids, mask, last, device, dtype)
                _, s, c = decision_loss(head(h), rows, device)
                for k in KINDS:
                    sums[k] += s[k]
                    cnt[k] += c[k]
        model.train()
        return {k: sums[k] / max(1, cnt[k]) for k in KINDS}, sum(sums.values()) / max(1, sum(cnt.values()))

    def save(tag):
        """Write to <tag>.tmp, then swap in atomically; the previous <tag> is kept as <tag>.prev. A crash or
        power cut mid-write can never leave the only checkpoint half-written."""
        if not is_main:
            return
        import shutil
        d = os.path.join(args.out, tag)
        tmp = d + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        m, hd = (model.module if world > 1 else model), (head.module if world > 1 else head)
        lora_sd = {k: v for k, v in m.state_dict().items() if "lora_" in k}
        torch.save({"lora": lora_sd, "head": hd.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "rng": sampler.rng.getstate(), "args": vars(args)}, os.path.join(tmp, "state.pt"))
        m.save_pretrained(tmp)                                 # peft adapter (adapter_model.safetensors + config)
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in hd.state_dict().items()}, os.path.join(tmp, "head.safetensors"))
        json.dump({"verbalizer_ids": VERBALIZER_IDS, "ranges": RANGES, "step": step}, open(os.path.join(tmp, "head.json"), "w"))
        prev = d + ".prev"
        shutil.rmtree(prev, ignore_errors=True)
        if os.path.exists(d):
            os.replace(d, prev)
        os.replace(tmp, d)

    logf = open(os.path.join(args.out, "log.csv"), "a", newline="")
    logw = csv.writer(logf)
    if is_main and step == 0:
        logw.writerow(["step", "loss", "kl_noul", "kl_score", "kl_choice", "lr_lora", "lr_head", "tokens", "sec"])
    model.train()
    while step < args.steps:
        t0 = time.time()
        rows = sampler.step()
        opt.zero_grad(set_to_none=True)
        sums, cnt, ntok, loss_acc = {k: 0.0 for k in KINDS}, {k: 0 for k in KINDS}, 0, 0.0
        mbs = list(micro_batches(rows, tok, args.max_tokens, pad_id))
        for mi, (mrows, ids, mask, last) in enumerate(mbs):
            if world > 1:   # only sync grads on the last micro-batch
                model.require_backward_grad_sync = head.require_backward_grad_sync = (mi == len(mbs) - 1)
            h = forward_hidden(model, ids, mask, last, device, dtype)
            loss, s, c = decision_loss(head(h), mrows, device)
            (loss / len(rows)).backward()
            loss_acc += loss.item()
            ntok += ids.numel()
            for k in KINDS:
                sums[k] += s[k]
                cnt[k] += c[k]
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], args.clip)
        opt.step()
        sched.step()
        step += 1
        if is_main:
            kls = [sums[k] / max(1, cnt[k]) for k in KINDS]
            logw.writerow([step, f"{loss_acc/len(rows):.5f}"] + [f"{v:.5f}" for v in kls]
                          + [f"{opt.param_groups[0]['lr']:.2e}", f"{opt.param_groups[1]['lr']:.2e}", ntok, f"{time.time()-t0:.1f}"])
            logf.flush()
            if step % 10 == 0 or step <= 5:
                log(f"step {step} loss {loss_acc/len(rows):.4f} kl n/s/c {kls[0]:.3f}/{kls[1]:.3f}/{kls[2]:.3f} "
                    f"gn {gn:.2f} tok {ntok} {time.time()-t0:.1f}s")
        if step % args.eval_every == 0 or step == args.steps:
            per, allkl = evaluate()
            log(f"eval step {step}: KL all {allkl:.4f} " + " ".join(f"{k} {v:.4f}" for k, v in per.items()))
            if is_main:
                json.dump({"step": step, "kl": allkl, **per}, open(os.path.join(args.out, f"eval_{step}.json"), "w"))
        if step % args.save_every == 0 or step == args.steps:
            save("last")
            if step == args.steps:
                save("final")
    log("done")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
