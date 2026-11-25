"""Portable Python ctypes wrapper for the OSSXC Native C++/CUDA Engine (ossxc_engine.dll / libossxc_engine.so).
Loads resident_gpu.pt strictly on CPU (never initializes PyTorch CUDA!), keeps embed_tokens in RAM,
and streams untouched native 4-bit MXFP4 experts from per-layer pinned RAM with zero-copy lookahead prefetch.
"""
from __future__ import annotations

import ctypes
import gc
import glob
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch

from .config import (
    D_FF,
    D_MODEL,
    ENGINE_DIR,
    NUM_EXPERTS,
    NUM_LAYERS,
    TOP_K,
    VOCAB_SIZE,
    WEIGHTS_DIR,
    bytes_to_gb,
    bytes_to_gib,
)
from .rope import build_yarn_rope_cos_sin

# Cache policy constants matching ossxc_engine.cu
CACHE_NONE = 0
CACHE_LRU = 1
CACHE_LFU_AGING = 2
CACHE_ADAPTIVE_HYBRID = 3


class ProfileStats(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_uint64),
        ("pcie_bytes", ctypes.c_uint64),
        ("cache_hits", ctypes.c_uint64),
        ("cache_misses", ctypes.c_uint64),
        ("total_wall_ms", ctypes.c_double),
        ("embed_lookup_h2d_sync_ms", ctypes.c_double),
        ("cpu_staging_memcpy_ms", ctypes.c_double),
        ("pcie_dma_ms", ctypes.c_double),
        ("gpu_attn_router_ms", ctypes.c_double),
        ("gpu_moe_ms", ctypes.c_double),
        ("gpu_lmhead_ms", ctypes.c_double),
        ("gpu_wait_dma_ms", ctypes.c_double),
        ("cpu_wait_router_ms", ctypes.c_double),
        ("cpu_wait_end_ms", ctypes.c_double),
        ("layer_hits", ctypes.c_uint64 * NUM_LAYERS),
        ("layer_misses", ctypes.c_uint64 * NUM_LAYERS),
        ("prefetch_attempts", ctypes.c_uint64),
        ("prefetch_hits", ctypes.c_uint64),
        ("prefetch_dma_ms", ctypes.c_double),
        ("prefetch_hidden_ms", ctypes.c_double),
        ("lookahead_overhead_ms", ctypes.c_double),
    ]


class PrefillProfileStats(ctypes.Structure):
    _fields_ = [
        ("chunks", ctypes.c_uint64),
        ("tokens", ctypes.c_uint64),
        ("experts_loaded", ctypes.c_uint64),
        ("total_wall_ms", ctypes.c_double),
        ("attn_local_ring128_ms", ctypes.c_double),
        ("attn_global_causal_ms", ctypes.c_double),
        ("qkv_o_router_ms", ctypes.c_double),
        ("h2d_dma_ms", ctypes.c_double),
        ("cpu_staging_ms", ctypes.c_double),
        ("dequant_mxfp4_ms", ctypes.c_double),
        ("gather_scatter_swiglu_ms", ctypes.c_double),
        ("expert_gemm_ms", ctypes.c_double),
        ("gpu_wait_dma_ms", ctypes.c_double),
        ("sync_cpu_gpu_ms", ctypes.c_double),
    ]


class MemReport(ctypes.Structure):
    _fields_ = [
        ("resident_weights_bytes", ctypes.c_size_t),
        ("kv_local_ring128_bytes", ctypes.c_size_t),
        ("kv_global_bytes", ctypes.c_size_t),
        ("ring_shadow_bytes", ctypes.c_size_t),
        ("stream_bufs_2x_bytes", ctypes.c_size_t),
        ("scratch_workspace_bytes", ctypes.c_size_t),
        ("arena_bytes", ctypes.c_size_t),
        ("total_engine_vram_bytes", ctypes.c_size_t),
        ("cuda_free_bytes", ctypes.c_size_t),
        ("cuda_total_bytes", ctypes.c_size_t),
        ("kv_cap_global", ctypes.c_int),
        ("cache_capacity_slots", ctypes.c_int),
        ("cache_active_entries", ctypes.c_int),
        ("cache_policy", ctypes.c_int),
        ("num_copy_threads", ctypes.c_int),
        ("kv_global_hot_bytes", ctypes.c_size_t),
        ("kv_global_cold_data_bytes", ctypes.c_size_t),
        ("kv_global_cold_scale_bytes", ctypes.c_size_t),
        ("kv_ram_backup_bytes", ctypes.c_size_t),
        ("hybrid_kv_mode", ctypes.c_int),
    ]


def _resolve_engine_library_path() -> str:
    env_lib = os.environ.get("OSSXC_ENGINE_LIB")
    if env_lib and os.path.isfile(env_lib):
        return env_lib
    is_win = platform.system().lower() == "windows"
    fname = "ossxc_engine.dll" if is_win else "libossxc_engine.so"
    cand = os.path.join(ENGINE_DIR, fname)
    if os.path.isfile(cand):
        return cand
    build_hint = (
        r"powershell -ExecutionPolicy Bypass -File engine\build_windows.ps1"
        if is_win
        else "bash engine/build_linux.sh"
    )
    raise FileNotFoundError(f"Compiled OSSXC engine not found at {cand}. Build it first with:\n  {build_hint}")


def _add_windows_cuda_dll_dirs() -> None:
    if platform.system().lower() != "windows" or not hasattr(os, "add_dll_directory"):
        return
    cands: list[str] = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin"))
    cands.extend(sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"), reverse=True))
    seen = set()
    for d in cands:
        if d not in seen and os.path.isdir(d):
            os.add_dll_directory(d)
            seen.add(d)


def _load_dll() -> ctypes.CDLL:
    _add_windows_cuda_dll_dirs()
    lib_path = _resolve_engine_library_path()
    lib = ctypes.CDLL(lib_path)

    c_vp = ctypes.c_void_p
    c_i = ctypes.c_int
    c_sz = ctypes.c_size_t
    c_i8p = ctypes.POINTER(ctypes.c_int8)
    c_u8p = ctypes.POINTER(ctypes.c_uint8)
    c_u16p = ctypes.POINTER(ctypes.c_uint16)
    c_i32p = ctypes.POINTER(ctypes.c_int)
    c_u32p = ctypes.POINTER(ctypes.c_uint32)
    c_f32p = ctypes.POINTER(ctypes.c_float)
    c_f64p = ctypes.POINTER(ctypes.c_double)
    c_u64p = ctypes.POINTER(ctypes.c_uint64)

    lib.engine_create.argtypes = [c_i]
    lib.engine_create.restype = c_vp
    lib.engine_create_ex.argtypes = [c_i, c_i]
    lib.engine_create_ex.restype = c_vp
    lib.engine_set_global_weights.argtypes = [c_vp, c_u16p, c_u16p, c_f32p, c_f32p, c_f32p]
    lib.engine_set_global_weights.restype = c_i
    lib.engine_set_layer_weights.argtypes = [
        c_vp, c_i, c_f32p, c_f32p, c_u16p, c_u16p, c_u16p, c_u16p,
        c_f32p, c_u16p, c_u16p, c_u16p, c_u16p, c_u8p,
    ]
    lib.engine_set_layer_weights.restype = c_i
    lib.engine_init_arena.argtypes = [c_vp, c_sz, c_sz]
    lib.engine_init_arena.restype = c_i
    lib.engine_set_cache_policy.argtypes = [c_vp, c_i, c_i]
    lib.engine_set_cache_policy.restype = None
    lib.engine_set_copy_threads.argtypes = [c_vp, c_i]
    lib.engine_set_copy_threads.restype = None
    lib.engine_set_staging_mode.argtypes = [c_vp, c_i, c_i]
    lib.engine_set_staging_mode.restype = None
    lib.engine_set_profile_enabled.argtypes = [c_vp, c_i]
    lib.engine_set_profile_enabled.restype = None
    lib.engine_reset_profile.argtypes = [c_vp]
    lib.engine_reset_profile.restype = None
    lib.engine_get_profile.argtypes = [c_vp, ctypes.POINTER(ProfileStats)]
    lib.engine_get_profile.restype = None
    lib.engine_set_prefill_profile_enabled.argtypes = [c_vp, c_i]
    lib.engine_set_prefill_profile_enabled.restype = None
    lib.engine_reset_prefill_profile.argtypes = [c_vp]
    lib.engine_reset_prefill_profile.restype = None
    lib.engine_get_prefill_profile.argtypes = [c_vp, ctypes.POINTER(PrefillProfileStats)]
    lib.engine_get_prefill_profile.restype = None
    lib.engine_reset_cache.argtypes = [c_vp]
    lib.engine_reset_cache.restype = None
    lib.engine_clear_kv.argtypes = [c_vp]
    lib.engine_clear_kv.restype = None
    lib.engine_set_lookahead_prefetch.argtypes = [c_vp, c_i, ctypes.c_float]
    lib.engine_set_lookahead_prefetch.restype = None
    lib.engine_set_fast_gemv.argtypes = [c_vp, c_i]
    lib.engine_set_fast_gemv.restype = None
    lib.engine_set_verify_readonly_cache.argtypes = [c_vp, c_i]
    lib.engine_set_verify_readonly_cache.restype = None
    lib.engine_commit_verified_cache_accesses.argtypes = [c_vp, c_i]
    lib.engine_commit_verified_cache_accesses.restype = None
    lib.engine_set_force_splits.argtypes = [c_vp, c_i]
    lib.engine_set_force_splits.restype = None
    lib.engine_set_quest_prune_margin.argtypes = [c_vp, ctypes.c_float, c_i]
    lib.engine_set_quest_prune_margin.restype = None
    lib.engine_reset_prune_stats.argtypes = [c_vp]
    lib.engine_reset_prune_stats.restype = None
    lib.engine_get_prune_stats.argtypes = [c_vp, c_u64p]
    lib.engine_get_prune_stats.restype = c_i
    lib.engine_get_last_hidden.argtypes = [c_vp, c_f32p, c_i]
    lib.engine_get_last_hidden.restype = c_i
    lib.engine_alloc_pinned_layer.argtypes = [c_vp, c_i, c_sz]
    lib.engine_alloc_pinned_layer.restype = c_vp
    lib.engine_set_layer_pinned_flag.argtypes = [c_vp, c_i, c_i]
    lib.engine_set_layer_pinned_flag.restype = None
    lib.engine_get_cache_resident_mask.argtypes = [c_vp, c_u32p]
    lib.engine_get_cache_resident_mask.restype = None
    lib.engine_benchmark_lmhead.argtypes = [c_vp, c_i, c_i, c_f64p]
    lib.engine_benchmark_lmhead.restype = c_i
    lib.engine_get_mem_info.argtypes = [c_vp, ctypes.POINTER(MemReport)]
    lib.engine_get_mem_info.restype = None
    lib.engine_forward_token.argtypes = [c_vp, c_i, c_i, c_f32p, c_i, c_u8p]
    lib.engine_forward_token.restype = c_i
    lib.engine_prefill_chunk.argtypes = [c_vp, c_i32p, c_i, c_i, c_i, c_f32p, c_i]
    lib.engine_prefill_chunk.restype = c_i
    lib.engine_ring_snapshot.argtypes = [c_vp, c_i, c_i]
    lib.engine_ring_snapshot.restype = c_i
    lib.engine_ring_rollback.argtypes = [c_vp, c_i, c_i, c_i]
    lib.engine_ring_rollback.restype = c_i
    lib.engine_restore_global_hot_from_ram.argtypes = [c_vp, c_i]
    lib.engine_restore_global_hot_from_ram.restype = c_i
    lib.engine_verify_tokens.argtypes = [c_vp, c_i32p, c_i, c_i, c_f32p, c_i, c_u64p, c_i32p, c_i32p]
    lib.engine_verify_tokens.restype = c_i
    lib.engine_read_ring_kv.argtypes = [c_vp, c_u16p, c_u16p]
    lib.engine_read_ring_kv.restype = c_i
    lib.engine_write_ring_kv.argtypes = [c_vp, c_u16p, c_u16p]
    lib.engine_write_ring_kv.restype = c_i
    lib.engine_verify_tree_tokens.argtypes = [
        c_vp, c_i32p, c_i32p, c_u16p, c_i, c_i, c_f32p, c_i, c_u64p, c_i32p, c_i32p,
    ]
    lib.engine_verify_tree_tokens.restype = c_i
    lib.engine_commit_tree_path.argtypes = [c_vp, c_i, c_i, c_i32p, c_i]
    lib.engine_commit_tree_path.restype = c_i
    lib.engine_eagle3_load_int8.argtypes = [
        c_vp, c_i,
        c_f32p, c_f32p, c_f32p, c_f32p, c_f32p,
        c_i8p, c_f32p, c_i8p, c_f32p, c_i8p, c_f32p, c_i8p, c_f32p, c_i8p, c_f32p, c_f32p,
        c_f32p, c_i8p, c_f32p, c_i8p, c_f32p, c_f32p, c_i8p, c_f32p, c_i8p, c_f32p,
        c_f32p, c_i8p, c_f32p, c_i8p, c_f32p, c_f32p, c_i8p, c_f32p, c_i8p, c_f32p,
        c_f32p, c_i8p, c_f32p, c_f32p,
        c_i8p, c_f32p, c_f32p,
        c_i32p, c_i,
    ]
    lib.engine_eagle3_load_int8.restype = c_i
    lib.engine_eagle3_set_shortlist.argtypes = [c_vp, c_i32p, c_i]
    lib.engine_eagle3_set_shortlist.restype = c_i
    lib.engine_eagle3_reset_kv.argtypes = [c_vp]
    lib.engine_eagle3_reset_kv.restype = None
    lib.engine_eagle3_commit_prefix.argtypes = [c_vp, c_i, c_i32p, c_i]
    lib.engine_eagle3_commit_prefix.restype = c_i
    lib.engine_eagle3_draft_tree_v2.argtypes = [
        c_vp, c_i, c_i, c_i, c_i, c_i, ctypes.c_float, c_i, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        c_i32p, c_i32p, c_i32p, c_u16p, c_i32p, c_i32p, c_f64p,
    ]
    lib.engine_eagle3_draft_tree_v2.restype = c_i
    lib.engine_destroy.argtypes = [c_vp]
    lib.engine_destroy.restype = None

    return lib


def query_nvidia_smi_vram_mib() -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        u, t = [float(x.strip()) for x in out.splitlines()[0].split(",")]
        return u, t
    except Exception:
        return -1.0, -1.0


def _u16_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _f32_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _u8_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _i32_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _to_np_u16(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.to(torch.float16).contiguous().view(torch.uint16).numpy())


def _to_np_f32(t: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(t.float().contiguous().numpy())


class OSSXCEngine:
    """High-level wrapper around the native OSSXC C++/CUDA engine (zero PyTorch CUDA allocations)."""

    def __init__(
        self,
        weights_dir: str = WEIGHTS_DIR,
        kv_cap_global: int = 65536,
        max_vram_ceiling_bytes: int = 5_190_000_000,
        device_safety_margin_bytes: int = 32 * 1024 * 1024,
        cache_policy: int = CACHE_ADAPTIVE_HYBRID,
        num_copy_threads: int = 2,
        hybrid_kv: bool = True,
        pinned_experts: bool = True,
    ):
        self.weights_dir = str(Path(weights_dir).resolve())
        self.lib = _load_dll()
        self.kv_cap_global = kv_cap_global
        self.hybrid_kv = bool(hybrid_kv) and (kv_cap_global > 4096)
        self.pinned_experts = bool(pinned_experts)
        self.handle = self.lib.engine_create_ex(kv_cap_global, 1 if self.hybrid_kv else 0)
        if not self.handle:
            raise RuntimeError("engine_create_ex failed (CUDA/cuBLAS initialization error)")

        self.embed_ram_u16: np.ndarray | None = None
        self.ram_layers: list[np.ndarray] = []

        self._load_weights()

        rc = self.lib.engine_init_arena(
            self.handle,
            ctypes.c_size_t(max_vram_ceiling_bytes),
            ctypes.c_size_t(device_safety_margin_bytes),
        )
        if rc != 0:
            self.close()
            raise RuntimeError(f"engine_init_arena failed with CUDA error {rc}")

        # WDDM anti-paging self-check: ensure local VRAM stays full-speed (lm_head <= 8.5 ms) and < 5950 MiB
        cur_margin = device_safety_margin_bytes
        for _ in range(3):
            lm_ms = self.benchmark_lmhead(iters=10, use_fast=True)
            smi_u, smi_t = query_nvidia_smi_vram_mib()
            smi_ok = (smi_u < 0) or (smi_t > 7000) or (smi_u <= 5950.0)
            if lm_ms <= 8.5 and smi_ok:
                break
            cur_margin += 96 * 1024 * 1024
            rc2 = self.lib.engine_init_arena(
                self.handle,
                ctypes.c_size_t(max_vram_ceiling_bytes),
                ctypes.c_size_t(cur_margin),
            )
            if rc2 != 0:
                break

        self.set_cache_policy(cache_policy)
        self.set_copy_threads(num_copy_threads)

        self._logits_1 = np.empty((VOCAB_SIZE,), dtype=np.float32)
        self._top4_24x4 = np.empty((NUM_LAYERS, TOP_K), dtype=np.uint8)

    def _load_weights(self) -> None:
        res_path = os.path.join(self.weights_dir, "resident_gpu.pt")
        if not os.path.isfile(res_path):
            raise FileNotFoundError(
                f"Missing {res_path}. Run `python scripts/prepare_weights.py --download` first."
            )
        res = torch.load(res_path, map_location="cpu", weights_only=False)

        self.embed_ram_u16 = _to_np_u16(res["embed_tokens"])
        lm_head_u16 = _to_np_u16(res["lm_head"])
        final_norm_f32 = _to_np_f32(res["final_norm"])

        cos_t, sin_t = build_yarn_rope_cos_sin(65536, device="cpu")
        cos_np = _to_np_f32(cos_t)
        sin_np = _to_np_f32(sin_t)

        rc = self.lib.engine_set_global_weights(
            self.handle,
            _u16_ptr(self.embed_ram_u16),
            _u16_ptr(lm_head_u16),
            _f32_ptr(final_norm_f32),
            _f32_ptr(cos_np),
            _f32_ptr(sin_np),
        )
        if rc != 0:
            raise RuntimeError(f"engine_set_global_weights failed: {rc}")
        del lm_head_u16, final_norm_f32, cos_t, sin_t, cos_np, sin_np

        mxfp4_dir = os.path.join(self.weights_dir, "mxfp4_experts")
        for l in range(NUM_LAYERS):
            rl = res["layers"][l]
            in_norm = _to_np_f32(rl["in_norm"])
            post_norm = _to_np_f32(rl["post_norm"])
            w_qkv = _to_np_u16(torch.cat([rl["q_w"], rl["k_w"], rl["v_w"]], dim=0))
            b_qkv = _to_np_u16(torch.cat([rl["q_b"], rl["k_b"], rl["v_b"]], dim=0))
            w_o = _to_np_u16(rl["o_w"])
            b_o = _to_np_u16(rl["o_b"])
            sinks = _to_np_f32(rl["sinks"])
            r_w = _to_np_u16(rl["router_w"])
            r_b = _to_np_u16(rl["router_b"])

            gu_b_t = torch.empty((NUM_EXPERTS, 2 * D_FF), dtype=torch.float16)
            gu_b_t[:, 0::2] = rl["gate_bias"].to(torch.float16)
            gu_b_t[:, 1::2] = rl["up_bias"].to(torch.float16)
            gu_b = _to_np_u16(gu_b_t)
            dn_b = _to_np_u16(rl["down_bias"])

            bin_path = os.path.join(mxfp4_dir, f"layer_{l:02d}.bin")
            if not os.path.isfile(bin_path):
                raise FileNotFoundError(
                    f"Missing {bin_path}. Run `python scripts/prepare_weights.py --download` first."
                )
            fsize = os.path.getsize(bin_path)
            if self.pinned_experts:
                p_addr = self.lib.engine_alloc_pinned_layer(self.handle, l, ctypes.c_size_t(fsize))
                if not p_addr:
                    raise RuntimeError(f"engine_alloc_pinned_layer({l}, {fsize}) failed (check available system RAM)")
                buf = (ctypes.c_ubyte * fsize).from_address(p_addr)
                with open(bin_path, "rb") as bf:
                    nread = bf.readinto(buf)
                if nread != fsize:
                    raise IOError(f"Short read {bin_path}: {nread} != {fsize}")
                ram_ptr = ctypes.cast(p_addr, ctypes.POINTER(ctypes.c_uint8))
            else:
                arr = np.fromfile(bin_path, dtype=np.uint8)
                self.ram_layers.append(arr)
                ram_ptr = _u8_ptr(arr)

            rc = self.lib.engine_set_layer_weights(
                self.handle, l,
                _f32_ptr(in_norm), _f32_ptr(post_norm),
                _u16_ptr(w_qkv), _u16_ptr(b_qkv),
                _u16_ptr(w_o), _u16_ptr(b_o),
                _f32_ptr(sinks),
                _u16_ptr(r_w), _u16_ptr(r_b),
                _u16_ptr(gu_b), _u16_ptr(dn_b),
                ram_ptr,
            )
            if rc != 0:
                raise RuntimeError(f"engine_set_layer_weights({l}) failed: {rc}")

        del res
        gc.collect()

    def set_cache_policy(self, policy: int, max_slots_override: int = -1) -> None:
        self.lib.engine_set_cache_policy(self.handle, int(policy), int(max_slots_override))

    def set_copy_threads(self, num_threads: int) -> None:
        self.lib.engine_set_copy_threads(self.handle, int(num_threads))

    def set_profile_enabled(self, enabled: bool) -> None:
        self.lib.engine_set_profile_enabled(self.handle, 1 if enabled else 0)

    def reset_profile(self) -> None:
        self.lib.engine_reset_profile(self.handle)

    def reset_cache(self) -> None:
        self.lib.engine_reset_cache(self.handle)

    def clear_kv(self) -> None:
        self.lib.engine_clear_kv(self.handle)

    def set_lookahead_prefetch(self, mode: int = 1, tau_prob32: float = 0.040) -> None:
        self.lib.engine_set_lookahead_prefetch(self.handle, int(mode), ctypes.c_float(float(tau_prob32)))

    def set_verify_readonly_cache(self, enabled: bool) -> None:
        self.lib.engine_set_verify_readonly_cache(self.handle, 1 if enabled else 0)

    def commit_verified_cache_accesses(self, n_accepted: int) -> None:
        self.lib.engine_commit_verified_cache_accesses(self.handle, int(n_accepted))

    def set_quest_prune_margin(self, margin_nats: float = 14.0, collect_stats: bool = False) -> None:
        self.lib.engine_set_quest_prune_margin(self.handle, ctypes.c_float(float(margin_nats)), 1 if collect_stats else 0)

    def benchmark_lmhead(self, iters: int = 20, use_fast: bool = True) -> float:
        out_ms = ctypes.c_double(0.0)
        rc = self.lib.engine_benchmark_lmhead(self.handle, int(iters), 1 if use_fast else 0, ctypes.byref(out_ms))
        if rc != 0:
            raise RuntimeError(f"engine_benchmark_lmhead failed: {rc}")
        return float(out_ms.value)

    def get_mem_info(self) -> dict:
        m = MemReport()
        self.lib.engine_get_mem_info(self.handle, ctypes.byref(m))
        smi_used_mib, smi_tot_mib = query_nvidia_smi_vram_mib()
        rss_bytes = psutil.Process().memory_info().rss
        cuda_used_bytes = m.cuda_total_bytes - m.cuda_free_bytes
        return {
            "resident_weights_gb": round(bytes_to_gb(m.resident_weights_bytes), 4),
            "kv_total_gb": round(bytes_to_gb(m.kv_local_ring128_bytes + m.kv_global_bytes), 4),
            "arena_gb": round(bytes_to_gb(m.arena_bytes), 4),
            "scratch_workspace_mb": round(m.scratch_workspace_bytes / 1e6, 2),
            "total_engine_vram_gb": round(bytes_to_gb(m.total_engine_vram_bytes), 4),
            "total_engine_vram_gib": round(bytes_to_gib(m.total_engine_vram_bytes), 4),
            "cuda_device_used_gib": round(bytes_to_gib(cuda_used_bytes), 4),
            "nvidia_smi_used_mib": round(smi_used_mib, 1),
            "nvidia_smi_used_gib": round(smi_used_mib / 1024.0, 4) if smi_used_mib >= 0 else -1.0,
            "nvidia_smi_total_mib": round(smi_tot_mib, 1),
            "process_ram_rss_gb": round(bytes_to_gb(rss_bytes), 4),
            "kv_cap_global": int(m.kv_cap_global),
            "cache_capacity_slots": int(m.cache_capacity_slots),
            "cache_active_entries": int(m.cache_active_entries),
            "cache_policy": int(m.cache_policy),
            "hybrid_kv_mode": bool(m.hybrid_kv_mode),
            "kv_ram_backup_gb": round(bytes_to_gb(m.kv_ram_backup_bytes), 4),
        }

    def get_profile(self) -> dict:
        p = ProfileStats()
        self.lib.engine_get_profile(self.handle, ctypes.byref(p))
        n = max(1, int(p.tokens))
        tot_wall = p.total_wall_ms / n
        dma = p.pcie_dma_ms / n
        gpu_tot = (p.gpu_attn_router_ms + p.gpu_moe_ms + p.gpu_lmhead_ms) / n
        hits, misses = int(p.cache_hits), int(p.cache_misses)
        pcie_gb = (p.pcie_bytes / n) / 1e9
        return {
            "n_tokens": int(p.tokens),
            "mean_total_ms": round(tot_wall, 3),
            "tokens_per_sec": round(1000.0 / max(1e-9, tot_wall), 3),
            "pcie_h2d_dma_ms": round(dma, 3),
            "pcie_effective_bw_gbps": round(pcie_gb / (dma / 1000.0), 3) if dma > 0 else 0.0,
            "gpu_compute_total_ms": round(gpu_tot, 3),
            "gpu_wait_dma_ms": round(p.gpu_wait_dma_ms / n, 3),
            "cache_hit_rate": round(hits / max(1, hits + misses), 4),
            "prefetch_hit_rate_pct": round(100.0 * int(p.prefetch_hits) / max(1, int(p.prefetch_attempts)), 2),
            "prefetch_hidden_ms": round(p.prefetch_hidden_ms / n, 3),
            "lookahead_overhead_ms": round(p.lookahead_overhead_ms / n, 3),
        }

    def forward_token(self, token_id: int, pos: int, return_top4: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        top_ptr = _u8_ptr(self._top4_24x4) if return_top4 else None
        rc = self.lib.engine_forward_token(
            self.handle, int(token_id), int(pos), _f32_ptr(self._logits_1), 0, top_ptr,
        )
        if rc != 0:
            raise RuntimeError(f"engine_forward_token(pos={pos}) failed: CUDA err {rc}")
        return self._logits_1.copy(), (self._top4_24x4.copy() if return_top4 else None)

    def prefill_chunk(
        self, token_ids: np.ndarray | list[int], start_pos: int, compute_last_logits: bool = False,
    ) -> np.ndarray | None:
        arr = np.ascontiguousarray(token_ids, dtype=np.int32)
        out_ptr = _f32_ptr(self._logits_1) if compute_last_logits else None
        rc = self.lib.engine_prefill_chunk(
            self.handle, _i32_ptr(arr), int(len(arr)), int(start_pos),
            1 if compute_last_logits else 0, out_ptr, 0,
        )
        if rc != 0:
            raise RuntimeError(f"engine_prefill_chunk(start={start_pos}, len={len(arr)}) failed: CUDA err {rc}")
        return self._logits_1.copy() if compute_last_logits else None

    def prefill_sequence(self, token_ids: np.ndarray | list[int], chunk_size: int = 512, start_pos: int = 0) -> np.ndarray:
        arr = np.ascontiguousarray(token_ids, dtype=np.int32)
        n = len(arr)
        logits = None
        for off in range(0, n, chunk_size):
            end = min(n, off + chunk_size)
            logits = self.prefill_chunk(arr[off:end], start_pos + off, compute_last_logits=(end == n))
        return logits

    def ring_snapshot(self, start_pos: int, K: int) -> None:
        rc = self.lib.engine_ring_snapshot(self.handle, int(start_pos), int(K))
        if rc != 0:
            raise RuntimeError(f"engine_ring_snapshot failed: {rc}")

    def ring_rollback(self, start_pos: int, first_rejected_i: int, K: int) -> None:
        rc = self.lib.engine_ring_rollback(self.handle, int(start_pos), int(first_rejected_i), int(K))
        if rc != 0:
            raise RuntimeError(f"engine_ring_rollback failed: {rc}")

    def restore_global_hot_from_ram(self, new_L: int) -> None:
        rc = self.lib.engine_restore_global_hot_from_ram(self.handle, int(new_L))
        if rc != 0:
            raise RuntimeError(f"engine_restore_global_hot_from_ram failed: {rc}")

    def read_ring_kv(self) -> tuple[np.ndarray, np.ndarray]:
        k = np.empty((12, 128, 512), dtype=np.uint16)
        v = np.empty((12, 128, 512), dtype=np.uint16)
        rc = self.lib.engine_read_ring_kv(self.handle, _u16_ptr(k), _u16_ptr(v))
        if rc != 0:
            raise RuntimeError(f"engine_read_ring_kv failed: {rc}")
        return k, v

    def write_ring_kv(self, k: np.ndarray, v: np.ndarray) -> None:
        kc = np.ascontiguousarray(k, dtype=np.uint16)
        vc = np.ascontiguousarray(v, dtype=np.uint16)
        rc = self.lib.engine_write_ring_kv(self.handle, _u16_ptr(kc), _u16_ptr(vc))
        if rc != 0:
            raise RuntimeError(f"engine_write_ring_kv failed: {rc}")

    def verify_tokens(self, token_ids: list[int] | np.ndarray, start_pos: int) -> tuple[np.ndarray, int, int, int]:
        arr = np.ascontiguousarray(token_ids, dtype=np.int32)
        K = len(arr)
        out_logits = np.empty((K, VOCAB_SIZE), dtype=np.float32)
        pcie_b = ctypes.c_uint64(0)
        uniq_e = ctypes.c_int(0)
        hits_e = ctypes.c_int(0)
        rc = self.lib.engine_verify_tokens(
            self.handle, _i32_ptr(arr), K, int(start_pos),
            _f32_ptr(out_logits), 0,
            ctypes.byref(pcie_b), ctypes.byref(uniq_e), ctypes.byref(hits_e),
        )
        if rc != 0:
            raise RuntimeError(f"engine_verify_tokens failed: {rc}")
        return out_logits, int(pcie_b.value), int(uniq_e.value), int(hits_e.value)

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.engine_destroy(self.handle)
            self.handle = None
        self.ram_layers.clear()
        self.embed_ram_u16 = None
        gc.collect()

    def __del__(self):
        self.close()
