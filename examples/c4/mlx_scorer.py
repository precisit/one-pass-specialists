"""An MLX twin of the toolkit's `TinyTransformerScorer`, for *training speed only*.

PyTorch-MPS reaches ~0.7 TFLOPs on this model's matmuls on an M1 Max; MLX does the same work ~3.5x
faster. The architecture is unchanged: this module re-implements the exact forward pass of
`vendor/cua_s1/model.py` (norm-first encoder layers with ReLU MLPs, torch MultiheadAttention's packed
in-projection, mean-pooled option encoder, AttentionHead) with **the same parameter names and
shapes**, so a trained state dict is saved as a toolkit checkpoint and loaded by the toolkit's own
class for evaluation, games and ONNX export. `parity()` is the gate: the toolkit model and this twin
must produce the same logits on real boards (max |delta| < 1e-3) before any MLX-trained weights
are trusted.

Only the v2 input shape is supported: context = 44 bytes with no padding, options = "column k"
(8 bytes, no padding), so every attention/pooling mask except the legal-option mask is all-true.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

import encode_c4 as E


def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + eps) * w + b


def encoder_layer(p, prefix, x, heads):
    # torch.nn.TransformerEncoderLayer(norm_first=True, activation=relu, dropout=0)
    b, t, w = x.shape
    h = layer_norm(x, p[f"{prefix}.norm1.weight"], p[f"{prefix}.norm1.bias"])
    qkv = h @ p[f"{prefix}.self_attn.in_proj_weight"].T + p[f"{prefix}.self_attn.in_proj_bias"]
    q, k, v = mx.split(qkv, 3, axis=-1)
    d = w // heads
    q = q.reshape(b, t, heads, d).transpose(0, 2, 1, 3)
    k = k.reshape(b, t, heads, d).transpose(0, 2, 1, 3)
    v = v.reshape(b, t, heads, d).transpose(0, 2, 1, 3)
    a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(d))
    a = a.transpose(0, 2, 1, 3).reshape(b, t, w)
    x = x + a @ p[f"{prefix}.self_attn.out_proj.weight"].T + p[f"{prefix}.self_attn.out_proj.bias"]
    h = layer_norm(x, p[f"{prefix}.norm2.weight"], p[f"{prefix}.norm2.bias"])
    h = mx.maximum(h @ p[f"{prefix}.linear1.weight"].T + p[f"{prefix}.linear1.bias"], 0)
    return x + h @ p[f"{prefix}.linear2.weight"].T + p[f"{prefix}.linear2.bias"]


def forward(p, config, ctx_ids, option_ids, legal):
    """ctx_ids [B, 44] int, option_ids [7, 8] int (shared), legal [B, 7] bool -> logits [B, 7]."""
    layers, heads, rank = config["layers"], config["heads"], config["rank"]
    emb, pos = p["embedding.weight"], p["position.weight"]
    x = emb[ctx_ids] + pos[: ctx_ids.shape[1]]
    for i in range(layers):
        x = encoder_layer(p, f"encoder.layers.{i}", x, heads)
    o = emb[option_ids] + pos[: option_ids.shape[1]]           # [7, 8, w] - same for every row
    o = encoder_layer(p, "option_encoder.layers.0", o, heads)
    o = o.mean(1)                                             # [7, w]
    c = layer_norm(x, p["head.context_norm.weight"], p["head.context_norm.bias"])
    o = layer_norm(o, p["head.option_norm.weight"], p["head.option_norm.bias"])
    q = o @ p["head.query.weight"].T                          # [7, r]
    k = c @ p["head.key.weight"].T                            # [B, L, r]
    v = c @ p["head.value.weight"].T
    scores = mx.einsum("nr,blr->bnl", q, k) / math.sqrt(rank)
    attended = mx.softmax(scores, axis=-1) @ v                # [B, 7, r]
    logits = (q[None] * attended).sum(-1) / math.sqrt(rank)
    return mx.where(legal, logits, -1e9)


def from_torch(state: dict) -> dict:
    return {k: mx.array(v.detach().cpu().numpy()) for k, v in state.items()}


def to_numpy(p: dict) -> dict:
    return {k: np.array(v) for k, v in p.items()}


def inputs(boards: np.ndarray):
    boards = boards.reshape(-1, 42)
    ctx = E.contexts(boards).astype(np.int32) + 1
    return mx.array(ctx), mx.array(E.option_ids_table().astype(np.int32)), mx.array(E.legal_mask(boards))


def parity(torch_model, p, config, boards: np.ndarray) -> float:
    import torch

    from model_c4 import Batcher

    batcher = Batcher(torch.device("cpu"))
    torch_model = torch_model.to("cpu").eval()
    with torch.no_grad():
        want = torch_model(batcher(torch.from_numpy(boards))).float().numpy()
    got = np.array(forward(p, config, *inputs(boards)))
    legal = E.legal_mask(boards)
    return float(np.abs(want - got)[legal].max())
