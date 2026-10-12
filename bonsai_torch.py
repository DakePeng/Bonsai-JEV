"""bonsai_torch.py - PyTorch reconstruction of Ternary Bonsai 2 27B from the PQ2_0 GGUF.

Mirrors the PrismML llama.cpp fork (src/llama-graph.cpp build_lora_mm) exactly:
  base path : y = W_rot . H(s * P x)   W_rot = frozen ternary weight, dequantized
  LoRA path : + B.A.x                  raw x; peft wraps HadamardLinear as base_layer
  embedding : e = s * H(z_rot)         inverse-after-lookup
  lm_head   : same as base path (output.weight is folded too)

H = normalized Sylvester Walsh-Hadamard, block 1024 on the last dim; s = +-1 sign vector
per input width; P = feature permutation (identity in HF layout, see perm_for()).
All of this is read from the GGUF's prism.hadamard.* metadata, nothing hard-coded.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GGUF_PY = os.environ.get("GGUF_PY", os.path.expanduser("~/bonsai/llama.cpp/gguf-py"))
sys.path.insert(0, GGUF_PY)
from gguf import GGUFReader  # noqa: E402

# ----------------------------------------------------------------------------- PQ2_0
QK = 128            # weights per block
BLOCK_BYTES = 34    # fp16 scale + 32 bytes of 2-bit codes


def dequant_pq2_0(raw: np.ndarray, out_features: int, in_features: int) -> torch.Tensor:
    """raw: uint8 (out, in*34/128). Code j of block: byte j//4, bits 2*(j%4). value = (q-1)*d."""
    raw = np.ascontiguousarray(raw).reshape(out_features, -1)
    nb = in_features // QK
    assert raw.shape[1] == nb * BLOCK_BYTES, (raw.shape, in_features)
    blk = raw.reshape(out_features, nb, BLOCK_BYTES)
    d = blk[:, :, :2].copy().view(np.float16).astype(np.float32)          # (out, nb, 1)
    shifts = (np.arange(4, dtype=np.uint8) * 2)
    q = (blk[:, :, 2:, None] >> shifts) & 3                              # (out, nb, 32, 4)
    q = q.reshape(out_features, nb, QK).astype(np.int8) - 1               # -1 / 0 / +1 (2 = unused)
    w = q.astype(np.float32) * d
    return torch.from_numpy(w.reshape(out_features, in_features))         # fp32; caller casts


# ----------------------------------------------------------------------------- Hadamard
def hadamard_matrix(n: int) -> torch.Tensor:
    """Normalized Sylvester Walsh-Hadamard H_n (symmetric, orthonormal, its own inverse)."""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def hadamard_blocks(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Apply H to each consecutive block of H.shape[0] features on the last dim."""
    # ponytail: dense 1024x1024 matmul, ~20% of the main matmul's FLOPs. FWHT butterfly if it matters.
    b = H.shape[0]
    shp = x.shape
    return (x.reshape(-1, shp[-1] // b, b) @ H).reshape(shp)


class HadamardLinear(nn.Linear):
    """nn.Linear whose weight lives in the rotated basis: forward = W_rot . H(s * x[perm]).
    Subclasses nn.Linear so peft's LoRA treats it as base_layer and adds B.A.x on the raw x,
    which is exactly what the fork's build_lora_mm does."""

    def __init__(self, in_features, out_features, signs, H, perm=None, dtype=torch.bfloat16):
        super().__init__(in_features, out_features, bias=False, dtype=dtype)
        self.register_buffer("signs", signs.to(dtype), persistent=False)
        self.register_buffer("H", H.to(dtype), persistent=False)
        self.register_buffer("perm", perm, persistent=False)

    def rotate_input(self, x):
        if self.perm is not None:
            x = x[..., self.perm]
        return hadamard_blocks(x * self.signs, self.H)

    def forward(self, x):
        return F.linear(self.rotate_input(x), self.weight)


class HadamardEmbedding(nn.Embedding):
    """Rows stored rotated; lookup reconstructs s * H(z)."""

    def __init__(self, num, dim, signs, H, dtype=torch.bfloat16):
        super().__init__(num, dim, dtype=dtype)
        self.register_buffer("signs", signs.to(dtype), persistent=False)
        self.register_buffer("H", H.to(dtype), persistent=False)

    def forward(self, ids):
        return hadamard_blocks(super().forward(ids), self.H) * self.signs


# ----------------------------------------------------------------------------- GGUF
def _field(r: GGUFReader, key: str):
    f = r.fields[key]
    if f.types[-1].name == "STRING":
        vals = [bytes(f.parts[i]).decode() for i in f.data]
    else:
        vals = [f.parts[i].tolist() for i in f.data]
        vals = [v[0] if isinstance(v, list) and len(v) == 1 else v for v in vals]
    return vals[0] if len(vals) == 1 and not key.endswith(("_names", "_widths", "_values")) else vals


class Bonsai:
    """Reads the PQ2_0 GGUF and hands out HF-named tensors plus the rotation metadata."""

    def __init__(self, gguf_path: str):
        self.r = GGUFReader(gguf_path)
        self.t = {t.name: t for t in self.r.tensors}
        md = lambda k: _field(self.r, "prism.hadamard." + k)
        assert md("version") == 1 and md("transform") == "normalized-sylvester-walsh-hadamard"
        assert md("axis") == "input-last-dimension" and md("sign_mode") == "explicit"
        self.block = md("block_size")
        self.H = hadamard_matrix(self.block)
        self.folded = set(md("weight_names"))
        self.inverse = set(md("inverse_weight_names"))
        widths, values = md("sign_widths"), torch.tensor(md("sign_values"), dtype=torch.float32)
        self.signs, off = {}, 0
        for w in widths:
            self.signs[w] = values[off:off + w]
            off += w
        assert off == len(values)
        self.gdn_v_grouped = bool(_field(self.r, "prism.hadamard.gdn_v_grouped"))
        self.n_layer = _field(self.r, "qwen35.block_count")
        self.n_embd = _field(self.r, "qwen35.embedding_length")

    # --- raw tensor -> torch, HF layout (out, in) ---
    def tensor(self, name: str) -> torch.Tensor:
        t = self.t[name]
        ty = t.tensor_type.name
        shape = [int(s) for s in t.shape][::-1]          # ggml ne is (in, out, ...); HF is (out, in)
        d = np.asarray(t.data)
        if ty == "PQ2_0":
            out, inn = shape
            return dequant_pq2_0(d, out, inn)
        if ty == "F32":
            return torch.from_numpy(np.array(d, dtype=np.float32)).reshape(shape)
        if ty == "BF16":
            if d.dtype == np.uint8:
                d = d.reshape(-1).view(np.int16)
            return torch.from_numpy(np.array(d, dtype=np.int16)).view(torch.bfloat16).reshape(shape)
        if ty == "F16":
            return torch.from_numpy(np.array(d, dtype=np.float16)).reshape(shape)
        raise NotImplementedError(ty)

    def layer_names(self, li: int):
        return [n for n in self.t if n.startswith(f"blk.{li}.")]


# GGUF -> HF (transformers qwen3_5 text model) names
_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def hf_name(gguf_name: str) -> str:
    if gguf_name in _GLOBAL_MAP:
        return _GLOBAL_MAP[gguf_name]
    _, li, rest = gguf_name.split(".", 2)
    return f"model.layers.{li}.{_LAYER_MAP[rest]}"


_ZERO_CENTERED_NORMS = ("attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight",
                        "attn_k_norm.weight", "output_norm.weight")


def hf_value(gguf_name: str, w: torch.Tensor) -> torch.Tensor:
    """GGUF storage conventions -> HF conventions."""
    if gguf_name.endswith("ssm_a"):
        return torch.log(-w)                 # GGUF stores -exp(A_log)
    if gguf_name.endswith("ssm_conv1d.weight"):
        return w.unsqueeze(1)                # (conv_dim, k) -> (conv_dim, 1, k)
    if gguf_name.endswith(_ZERO_CENTERED_NORMS):
        return w - 1                         # HF Qwen3_5RMSNorm applies (1 + w); GGUF has the +1 baked in
    return w                                 # ssm_norm (RMSNormGated) is a plain weight


def perm_for(b: Bonsai, gguf_name: str):
    """Feature permutation before signs+Hadamard. The fork permutes its own tiled GDN layout
    [hd, nk, rep] into the grouped [hd, rep, nk] order the folded ssm_out weight expects.
    HF's GatedDeltaNet already emits the grouped order (v head j uses k head j // rep), so in
    HF layout the permutation is the identity."""
    # ponytail: identity assumed for HF layout; step 7 alignment on a linear-attention layer proves it.
    return None


# ----------------------------------------------------------------------------- model assembly
def swap_linear(module: nn.Module, child: str, b: Bonsai, gguf_name: str, dtype):
    lin = getattr(module, child)
    new = HadamardLinear(lin.in_features, lin.out_features, b.signs[lin.in_features], b.H,
                         perm=perm_for(b, gguf_name), dtype=dtype)
    setattr(module, child, new)
    return new


def load_into(model_or_layer: nn.Module, b: Bonsai, names, strip_prefix: str, dtype, device):
    """Load the given GGUF tensor names into an HF module. Folded Linear children are swapped for
    HadamardLinear first. Returns the list of HF keys loaded."""
    sd, loaded = {}, []
    for g in names:
        key = hf_name(g)
        assert key.startswith(strip_prefix), (key, strip_prefix)
        key = key[len(strip_prefix):]
        w = hf_value(g, b.tensor(g))
        if g in b.folded:
            parent_path, child = key.rsplit(".", 2)[0], key.rsplit(".", 2)[1]
            parent = model_or_layer.get_submodule(parent_path) if parent_path else model_or_layer
            swap_linear(parent, child, b, g, dtype)
        sd[key] = w.to(dtype)
        loaded.append(key)
    missing, unexpected = model_or_layer.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    model_or_layer.to(device)
    return loaded, missing


def text_config(cfg_dir: str):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(cfg_dir)
    return getattr(cfg, "text_config", cfg)


def build_layer(b: Bonsai, cfg, li: int, dtype=torch.bfloat16, device="cuda"):
    """One decoder layer with Bonsai weights, for per-layer alignment on a 12 GB GPU."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    layer = M.Qwen3_5DecoderLayer(cfg, li)
    loaded, missing = load_into(layer, b, b.layer_names(li), f"model.layers.{li}.", dtype, device)
    assert not missing, missing
    return layer


def build_model(b: Bonsai, cfg, dtype=torch.bfloat16, device="cuda"):
    """Full text model (54 GB in BF16). Embedding and lm_head are Hadamard modules too."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    with torch.device("meta"):
        model = M.Qwen3_5ForCausalLM(cfg)
    model = model.to_empty(device="cpu")
    emb = HadamardEmbedding(cfg.vocab_size, cfg.hidden_size, b.signs[cfg.hidden_size], b.H, dtype)
    model.model.embed_tokens = emb
    names = [n for n in b.t if n not in ("output.weight",)]  # lm_head handled with the rest via folded set
    names.append("output.weight")
    loaded, missing = load_into(model, b, names, "", dtype, device)
    # rotary buffers etc. are non-persistent and were created on meta; rebuild them
    model.model.rotary_emb = type(model.model.rotary_emb)(cfg, device=device)
    assert all(m.startswith("model.rotary_emb") or "inv_freq" in m for m in missing), missing
    return model


# ----------------------------------------------------------------------------- self-check
def _selfcheck(gguf_path: str):
    torch.manual_seed(0)
    # 1. Hadamard matrix is orthonormal and symmetric
    H = hadamard_matrix(1024)
    assert torch.allclose(H @ H, torch.eye(1024), atol=1e-5) and torch.equal(H, H.T)
    # 2. HadamardLinear == explicit formula; LoRA-style add on raw x is independent of rotation
    lin = HadamardLinear(2048, 64, torch.sign(torch.randn(2048)), H, dtype=torch.float32)
    x = torch.randn(3, 2048)
    ref = (x * lin.signs).reshape(3, 2, 1024) @ H
    assert torch.allclose(lin(x), ref.reshape(3, 2048) @ lin.weight.T, atol=1e-4)
    # 3. real tensor: codes in {-1,0,1}, groups of 128 share one scale, no code 3
    b = Bonsai(gguf_path)
    w = b.tensor("blk.0.ffn_gate.weight")
    assert w.shape == (17408, 5120)
    g = w.reshape(17408, 40, 128)
    d = g.abs().amax(-1, keepdim=True)
    assert torch.all((g == 0) | (g.abs() == d)), "values must be 0 or +-scale per group"
    raw = np.asarray(b.t["blk.0.ffn_gate.weight"].data).reshape(17408, 40, 34)[:, :, 2:]
    assert not np.any(((raw[..., None] >> (np.arange(4) * 2)) & 3) == 3), "code 3 present"
    # 4. storage conventions
    assert abs(hf_value("output_norm.weight", b.tensor("output_norm.weight")).mean() + 1
               - b.tensor("output_norm.weight").mean()) < 1e-6
    a = b.tensor("blk.0.ssm_a")
    assert torch.all(a < 0) and hf_value("blk.0.ssm_a", a).isfinite().all()
    assert hf_value("blk.0.ssm_conv1d.weight", b.tensor("blk.0.ssm_conv1d.weight")).shape == (10240, 1, 4)
    assert "output.weight" in b.folded and "token_embd.weight" in b.inverse
    assert set(b.signs) == {5120, 6144, 17408}
    # 5. every tensor name maps
    for n in b.t:
        hf_name(n)
    print("selfcheck ok:", len(b.t), "tensors,", len(b.folded), "folded, block", b.block)


if __name__ == "__main__":
    _selfcheck(sys.argv[1] if len(sys.argv) > 1
               else os.path.expanduser("~/bonsai/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
