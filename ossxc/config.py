"""Portable configuration and model constants for OSSXC (openai/gpt-oss-20b)."""
from __future__ import annotations

import os
from pathlib import Path

# Portable paths relative to repository root (overridable via environment variables)
REPO_ROOT = Path(os.environ.get("OSSXC_HOME", Path(__file__).resolve().parent.parent)).resolve()
HF_MODEL_DIR = str(Path(os.environ.get("OSSXC_MODEL_DIR", REPO_ROOT / "hf_model")).resolve())
WEIGHTS_DIR = str(Path(os.environ.get("OSSXC_WEIGHTS_DIR", REPO_ROOT / "compressed")).resolve())
ENGINE_DIR = str((REPO_ROOT / "engine").resolve())

# Official Hugging Face model reference
HF_MODEL_REPO = "openai/gpt-oss-20b"

# Architecture constants (openai/gpt-oss-20b)
NUM_LAYERS = 24
NUM_EXPERTS = 32
TOP_K = 4
D_MODEL = 2880
D_FF = 2880
VOCAB_SIZE = 201088
HEAD_DIM = 64
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
GQA_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS  # 8
SLIDING_WINDOW = 128
MAX_CONTEXT = 65536
INITIAL_CONTEXT_LENGTH = 4096
RMS_NORM_EPS = 1e-5
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
ROPE_THETA = 150000.0
ROPE_SCALING_FACTOR = 32.0
ROPE_NTK_ALPHA = 1.0
ROPE_NTK_BETA = 32.0

EXPERT_BYTES = 13_219_200  # exact 16-byte aligned native MXFP4 expert payload
LAYER_BYTES = NUM_EXPERTS * EXPERT_BYTES  # 423,014,400 B per layer


def bytes_to_gb(n: int) -> float:
    return n / 1e9


def bytes_to_gib(n: int) -> float:
    return n / (1024.0 ** 3)
