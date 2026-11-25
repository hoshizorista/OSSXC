"""Exact YaRN RoPE cos/sin table builder for openai/gpt-oss-20b (up to 64K context)."""
from __future__ import annotations

import math

import torch

from .config import (
    HEAD_DIM,
    INITIAL_CONTEXT_LENGTH,
    ROPE_NTK_ALPHA,
    ROPE_NTK_BETA,
    ROPE_SCALING_FACTOR,
    ROPE_THETA,
)


def build_yarn_rope_cos_sin(
    max_seq_len: int = 65536,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact FP32 YaRN RoPE cos/sin tables [max_seq_len, 32] matching official HF config."""
    freq = ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM)
    concentration = 0.1 * math.log(ROPE_SCALING_FACTOR) + 1.0
    d_half = HEAD_DIM / 2.0
    low = d_half * math.log(INITIAL_CONTEXT_LENGTH / (ROPE_NTK_BETA * 2.0 * math.pi)) / math.log(ROPE_THETA)
    high = d_half * math.log(INITIAL_CONTEXT_LENGTH / (ROPE_NTK_ALPHA * 2.0 * math.pi)) / math.log(ROPE_THETA)
    interpolation = 1.0 / (ROPE_SCALING_FACTOR * freq)
    extrapolation = 1.0 / freq
    ramp = (torch.arange(int(d_half), dtype=torch.float32, device=device) - low) / (high - low)
    mask = 1.0 - ramp.clamp(0.0, 1.0)
    inv_freq = interpolation * (1.0 - mask) + extrapolation * mask

    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos() * concentration
    sin = freqs.sin() * concentration
    return cos, sin
