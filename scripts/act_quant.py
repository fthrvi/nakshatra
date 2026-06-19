"""
act_quant.py — activation (hidden-state) quantization on the wire (speed-stack finding #16).

In the distributed chain, each stage ships a hidden-state tensor [n_tokens, n_embd] to the next
stage. Today it goes as fp32 (4 bytes/elem). Over WAN, bandwidth is the cost center, and
activations tolerate quantization far better than weights — so int8 on the wire cuts ~4× off
every hop with negligible quality loss. (The fabric packet already reserves DTYPE_INT8=0x04.)

This is the codec only — pure + numpy, no transport. The worker plugs it in at the hidden-state
boundary (quantize before sending, dequantize before the daemon) behind NAKSHATRA_ACT_QUANT.

Scheme: **per-token symmetric int8**. One fp32 scale per token (row): scale = max(|row|)/127,
q = round(x/scale) clipped to [-127,127]. Per-token (not per-tensor) keeps accuracy high when
token magnitudes vary, and the n_tokens scales are negligible overhead vs the n_tokens·n_embd data.

Wire blob layout (so a receiver with (n_tokens, n_embd) can split it):
    [ n_tokens × float32 scales ][ n_tokens·n_embd × int8 data ]
Size = n_tokens·4 + n_tokens·n_embd  vs fp32's n_tokens·n_embd·4  → ~4× smaller for n_embd ≫ 1.
"""
from __future__ import annotations

import numpy as np

DTYPE_F32 = "f32"
DTYPE_INT8 = "int8"


def quantize_int8(hidden_f32: bytes, n_tokens: int, n_embd: int) -> bytes:
    """fp32 hidden bytes [n_tokens, n_embd] → wire blob (scales ++ int8). Per-token symmetric."""
    x = np.frombuffer(hidden_f32, dtype=np.float32).reshape(n_tokens, n_embd)
    scale = np.maximum(np.abs(x).max(axis=1), 1e-8).astype(np.float32) / 127.0   # per-token
    q = np.round(x / scale[:, None]).clip(-127, 127).astype(np.int8)
    return scale.tobytes() + q.tobytes()


def dequantize_int8(blob: bytes, n_tokens: int, n_embd: int) -> bytes:
    """Wire blob (scales ++ int8) → fp32 hidden bytes [n_tokens, n_embd]."""
    scale = np.frombuffer(blob, dtype=np.float32, count=n_tokens).reshape(n_tokens, 1)
    q = np.frombuffer(blob, dtype=np.int8, offset=n_tokens * 4).reshape(n_tokens, n_embd).astype(np.float32)
    return (q * scale).astype(np.float32).tobytes()


def quant_blob_size(n_tokens: int, n_embd: int) -> int:
    """Bytes of the int8 wire blob (for size validation on the receive side)."""
    return n_tokens * 4 + n_tokens * n_embd


def f32_size(n_tokens: int, n_embd: int) -> int:
    return n_tokens * n_embd * 4
