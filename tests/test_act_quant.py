"""
Tests for the activation-quant wire codec (scripts/act_quant.py). Pure.

Proves: int8 round-trip stays within the per-token quantization bound, the blob is ~4× smaller
than fp32, the per-token scaling actually helps when token magnitudes differ, and the sizes line
up for the receiver's validation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from act_quant import (  # noqa: E402
    quantize_int8, dequantize_int8, quant_blob_size, f32_size,
)


def _roundtrip(x):
    n_tokens, n_embd = x.shape
    blob = quantize_int8(x.astype(np.float32).tobytes(), n_tokens, n_embd)
    out = dequantize_int8(blob, n_tokens, n_embd)
    return np.frombuffer(out, dtype=np.float32).reshape(n_tokens, n_embd), blob


def test_roundtrip_within_quant_bound():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((5, 4096)) * 3.0).astype(np.float32)   # realistic hidden-ish
    y, _ = _roundtrip(x)
    # per-token error ≤ half a quantization step = (max|row|/127)/2
    step = np.abs(x).max(axis=1) / 127.0
    err = np.abs(y - x).max(axis=1)
    assert np.all(err <= step / 2 + 1e-4)
    # and the overall reconstruction is tight (relative L2 well under 1%)
    rel = np.linalg.norm(y - x) / np.linalg.norm(x)
    assert rel < 0.01


def test_blob_is_about_4x_smaller():
    n_tokens, n_embd = 8, 4096
    blob = quantize_int8(np.zeros((n_tokens, n_embd), np.float32).tobytes(), n_tokens, n_embd)
    assert len(blob) == quant_blob_size(n_tokens, n_embd)
    ratio = f32_size(n_tokens, n_embd) / len(blob)
    assert 3.9 < ratio < 4.01            # ~4×; the per-token scales are the tiny overhead


def test_per_token_scaling_handles_magnitude_spread():
    # token 0 tiny, token 1 huge — a per-TENSOR scale would crush token 0; per-token keeps both.
    x = np.zeros((2, 1024), np.float32)
    x[0] = np.linspace(-0.01, 0.01, 1024)
    x[1] = np.linspace(-50, 50, 1024)
    y, _ = _roundtrip(x)
    rel0 = np.linalg.norm(y[0] - x[0]) / np.linalg.norm(x[0])
    rel1 = np.linalg.norm(y[1] - x[1]) / np.linalg.norm(x[1])
    assert rel0 < 0.02 and rel1 < 0.02   # BOTH tokens reconstruct well


def test_single_token_and_zeros():
    # decode step is n_tokens=1; and an all-zero row must not divide by zero
    y, blob = _roundtrip(np.zeros((1, 256), np.float32))
    assert np.all(y == 0)
    assert len(blob) == quant_blob_size(1, 256)
    y2, _ = _roundtrip(np.array([[1.0, -1.0, 0.5, -0.5]], np.float32))
    assert y2.shape == (1, 4)


def test_sizes_match_helpers():
    assert quant_blob_size(3, 100) == 3 * 4 + 3 * 100
    assert f32_size(3, 100) == 3 * 100 * 4


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
