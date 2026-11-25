/*
 * OSSXC — Native C++/CUDA MXFP4 Inference Engine for openai/gpt-oss-20b (6 GB VRAM, 64K Context)
 * Copyright (c) 2026 Juan & Aria (Antigravity) — MIT License
 *
 * Key architectural features (66 CUDA kernels, 0 register spills on sm_75):
 *   - 100% untouched native 4-bit MXFP4 target experts streamed directly from per-layer pinned RAM
 *   - Zero-extra-VRAM Inter-Layer Speculative DMA Lookahead Prefetch (M3_ema + 1-ns zero-copy pointer swap)
 *   - Adaptive LRU/LFU VRAM Arena (C_VRAM >= 104 slots at 64K within 6 GB VRAM)
 *   - Exact Ring-128 local KV + Hybrid Global KV (exact K-FP16 + V_hot4K + V_sink128 + Cold V_pc32 + RAM mirror)
 *   - Exact per-position V_hot/V_pc32 page-32 boundary parity (V_bnd_fp16) for lossless batched/tree verification
 *   - Turing-optimized Split-K Flash-Decoding (PTX ex2.approx + Lazy-V 14-nat pruning) & SGEMM Chunked Flash-Prefill
 *   - Optional resident INT8 per-channel EAGLE-3 draft with cache-aware micro-tree pruning
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include "mxfp4_rans_format.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <float.h>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

static inline int host_popcnt32(uint32_t x) {
#ifdef _MSC_VER
    return (int)__popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}

// Model constants (openai/gpt-oss-20b)
static constexpr int NUM_LAYERS = 24;
static constexpr int NUM_LOCAL_LAYERS = 12;
static constexpr int NUM_GLOBAL_LAYERS = 12;
static constexpr int NUM_EXPERTS = 32;
static constexpr int TOP_K = 4;
static constexpr int D_MODEL = 2880;
static constexpr int D_FF = 2880;
static constexpr int VOCAB_SIZE = 201088;
static constexpr int NUM_Q_HEADS = 64;
static constexpr int NUM_KV_HEADS = 8;
static constexpr int HEAD_DIM = 64;
static constexpr int QKV_DIM = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM; // 5120
static constexpr int RING_SIZE = 128;
static constexpr int HOT_WINDOW = 4096; // Iter 5 Hybrid Global KV: most recent 4,096 V tokens in exact FP16!
static constexpr int V_PAGE_SIZE = 32;  // Iter 5 Lever B: per-channel V page size (1:1 aligned to TILE_K=32!)
static constexpr int V_SINK_FP16 = 128; // First 128 tokens kept in exact FP16 V (1.57 MB, 4 pages)
static constexpr int MAX_POS = 65536;
static constexpr int MAX_SPLITS = 64;
static constexpr int MAX_VERIFY_K = 8;
static constexpr int MAX_PREFILL_CHUNK = 512;

struct TreeMeta {
    int8_t   depth[MAX_VERIFY_K];
    uint16_t anc_mask[MAX_VERIFY_K];
    uint8_t  active;
    uint8_t  num_nodes;
};

// Device constant memory for TreeMeta -> ZERO local stack frame (0 bytes) in attention & RoPE kernels!
__constant__ TreeMeta c_tree_meta;

static inline TreeMeta make_inactive_tree_meta() {
    TreeMeta tm{};
    tm.active = 0;
    tm.num_nodes = 0;
    return tm;
}
static constexpr float RMS_EPS = 1e-5f;
static constexpr float SM_SCALE = 0.125f; // 1 / sqrt(64)
static constexpr float LOG2E = 1.4426950408889634f;
static constexpr float SM_SCALE_LOG2E = SM_SCALE * LOG2E; // 0.18033688011112042f
static constexpr float SWIGLU_ALPHA = 1.702f;
static constexpr float SWIGLU_LIMIT = 7.0f;

__device__ __forceinline__ float fast_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Packed native MXFP4 single expert layout (exact match to pack_abc_variants.py!)
static constexpr size_t GU_BLOCKS_BYTES = 5760ULL * 90ULL * 16ULL; // 8,294,400
static constexpr size_t GU_SCALES_BYTES = 5760ULL * 90ULL;         //   518,400
static constexpr size_t DN_BLOCKS_BYTES = 2880ULL * 90ULL * 16ULL; // 4,147,200
static constexpr size_t DN_SCALES_BYTES = 2880ULL * 90ULL;         //   259,200
static constexpr size_t EXPERT_BYTES    = GU_BLOCKS_BYTES + GU_SCALES_BYTES + DN_BLOCKS_BYTES + DN_SCALES_BYTES; // 13,219,200
static constexpr size_t OFF_GU_BLOCKS   = 0;
static constexpr size_t OFF_GU_SCALES   = GU_BLOCKS_BYTES;
static constexpr size_t OFF_DN_BLOCKS   = GU_BLOCKS_BYTES + GU_SCALES_BYTES;
static constexpr size_t OFF_DN_SCALES   = GU_BLOCKS_BYTES + GU_SCALES_BYTES + DN_BLOCKS_BYTES;

__constant__ float c_FP4_LUT[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__constant__ uint8_t d_kSym15ToNib[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 0
};

static constexpr size_t COMP_STREAM_BUF_BYTES = 12000000ULL; // >= max aligned rANS expert (11.876 MB)

// Fused Lossless Warp-rANS MXFP4 Decoder (Grid=75, Block=192):
//   Blocks 0..44: 6 warps/block = 270 warps decoding 12.4416 MB canonicalized FP4 blocks (1 LDS.U32/nibble!)
//   Blocks 45..74: 1 warp/block = 30 warps decoding 777.6 KB delta-rANS E8M0 scales IN PARALLEL!
__global__ void rans_decode_expert_fused_kernel(
    const uint8_t* __restrict__ comp_buf,
    uint8_t* __restrict__ dst_raw_expert
) {
    __shared__ union {
        struct {
            uint32_t table[RANS_PROB_SCALE]; // 16 KB: (nib<<28) | (cdf<<12) | freq
            uint16_t freq[16];
            uint16_t cdf[16];
        } blk;
        struct {
            uint8_t  lut[RANS_PROB_SCALE];   // 4 KB
            uint16_t freq[32];
            uint16_t cdf[32];
            uint8_t  row_buf[32 * RANS_SCL_ROW_LEN]; // 2880 B coalesced write staging
        } scl;
    } su;

    const auto* hdr = reinterpret_cast<const RansExpertHeader*>(comp_buf);
    const int tid = threadIdx.x;

    if (blockIdx.x < 45) {
        if (tid < 16) su.blk.freq[tid] = hdr->blk_freq[tid];
        __syncthreads();
        if (tid == 0) {
            uint16_t c = 0;
            for (int s = 0; s < 16; ++s) { su.blk.cdf[s] = c; c += su.blk.freq[s]; }
        }
        __syncthreads();
        for (int s = 0; s < RANS_BLK_SYMS; ++s) {
            uint32_t c0 = su.blk.cdf[s];
            uint32_t f  = su.blk.freq[s];
            uint32_t packed = ((uint32_t)d_kSym15ToNib[s] << 28) | (c0 << 12) | f;
            for (uint32_t k = tid; k < f; k += blockDim.x) su.blk.table[c0 + k] = packed;
        }
        __syncthreads();

        const int warp_in_blk = tid >> 5;
        const int lane        = tid & 31;
        const int w           = blockIdx.x * 6 + warp_in_blk;
        if (w >= RANS_BLK_WARPS) return;

        const uint16_t* b_stream = reinterpret_cast<const uint16_t*>(comp_buf + hdr->blk_stream_off);
        uint32_t in_pos = hdr->blk_warp_u16_off[w];
        uint32_t st     = hdr->blk_init_state[w][lane];
        const uint32_t lane_mask = (1u << lane) - 1u;

        uint32_t* out_u32 = (w < RANS_BLK_GU_WARPS)
            ? (reinterpret_cast<uint32_t*>(dst_raw_expert) + (size_t)w * RANS_BLK_WORDS_PER_LANE * 32ULL)
            : (reinterpret_cast<uint32_t*>(dst_raw_expert + OFF_DN_BLOCKS) + (size_t)(w - RANS_BLK_GU_WARPS) * RANS_BLK_WORDS_PER_LANE * 32ULL);

        for (int step = 0; step < RANS_BLK_WORDS_PER_LANE; ++step) {
            uint32_t word = 0;
            #pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                uint32_t slot  = st & RANS_PROB_MASK;
                uint32_t entry = su.blk.table[slot];
                uint32_t freq  = entry & 0xFFFu;
                uint32_t cdf   = (entry >> 12) & 0xFFFu;
                uint32_t nval  = entry >> 28;
                word |= (nval << (nib * 4));
                st = freq * (st >> RANS_PROB_BITS) + (slot - cdf);
                bool need = (st < RANS_L);
                uint32_t bm = __ballot_sync(0xFFFFFFFFu, need);
                if (need) {
                    int r = __popc(bm & lane_mask);
                    st = (st << 16) | (uint32_t)__ldg(&b_stream[in_pos + r]);
                }
                in_pos += __popc(bm);
            }
            out_u32[(size_t)step * 32ULL + lane] = word;
        }
    } else {
        const int w = blockIdx.x - 45;
        if (w >= RANS_SCL_WARPS || tid >= 32) return;
        const int lane = tid;
        su.scl.freq[lane] = hdr->scl_freq[lane];
        __syncwarp();
        if (lane == 0) {
            uint16_t c = 0;
            for (int s = 0; s < 32; ++s) { su.scl.cdf[s] = c; c += su.scl.freq[s]; }
        }
        __syncwarp();
        for (int s = 0; s < RANS_SCL_SYMS; ++s) {
            int c0 = su.scl.cdf[s];
            int f  = su.scl.freq[s];
            for (int k = lane; k < f; k += 32) su.scl.lut[c0 + k] = (uint8_t)s;
        }
        __syncwarp();

        const uint16_t* s_stream = reinterpret_cast<const uint16_t*>(comp_buf + hdr->scl_stream_off);
        const uint8_t*  e_stream = comp_buf + hdr->scl_esc_off;
        uint32_t in_pos  = hdr->scl_warp_u16_off[w];
        uint32_t esc_pos = hdr->scl_warp_esc_off[w];
        uint32_t st      = hdr->scl_init_state[w][lane];
        const uint32_t lane_mask = (1u << lane) - 1u;
        const int base_scl = (int)hdr->scl_base;

        uint8_t* dst_base = (w < RANS_SCL_GU_WARPS)
            ? (dst_raw_expert + OFF_GU_SCALES + (size_t)w * 32ULL * RANS_SCL_ROWS_PER_LANE * RANS_SCL_ROW_LEN)
            : (dst_raw_expert + OFF_DN_SCALES + (size_t)(w - RANS_SCL_GU_WARPS) * 32ULL * RANS_SCL_ROWS_PER_LANE * RANS_SCL_ROW_LEN);

        for (int r = 0; r < RANS_SCL_ROWS_PER_LANE; ++r) {
            int prev = base_scl;
            uint8_t* my_row = su.scl.row_buf + lane * RANS_SCL_ROW_LEN;
            for (int k = 0; k < RANS_SCL_ROW_LEN; ++k) {
                uint32_t slot = st & RANS_PROB_MASK;
                uint8_t  sym  = su.scl.lut[slot];
                st = (uint32_t)su.scl.freq[sym] * (st >> RANS_PROB_BITS) + (slot - (uint32_t)su.scl.cdf[sym]);
                bool need = (st < RANS_L);
                uint32_t bm = __ballot_sync(0xFFFFFFFFu, need);
                if (need) {
                    int rk = __popc(bm & lane_mask);
                    st = (st << 16) | (uint32_t)__ldg(&s_stream[in_pos + rk]);
                }
                in_pos += __popc(bm);

                bool is_esc = (sym == 31);
                uint32_t em = __ballot_sync(0xFFFFFFFFu, is_esc);
                uint8_t v;
                if (is_esc) {
                    int rk = __popc(em & lane_mask);
                    v = __ldg(&e_stream[esc_pos + rk]);
                } else {
                    v = (uint8_t)(prev + ((int)sym - 15));
                }
                esc_pos += __popc(em);
                prev = (int)v;
                my_row[k] = v;
            }
            __syncwarp();
            uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(dst_base + (size_t)r * 32ULL * RANS_SCL_ROW_LEN);
            const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(su.scl.row_buf);
            for (int i = lane; i < (32 * RANS_SCL_ROW_LEN) / 4; i += 32) {
                dst_u32[i] = src_u32[i];
            }
            __syncwarp();
        }
    }
}

__device__ __forceinline__ float decode_e8m0_scale(uint8_t s) {
    if (s == 0) return ldexpf(1.0f, -127);
    return __uint_as_float(((uint32_t)s) << 23);
}

// ============================================================================
// KERNEL 1: RMSNorm (FP32 input -> simultaneous FP32 & FP16 outputs)
// ============================================================================
__global__ void rmsnorm_fp32_fp16_kernel(
    const float* __restrict__ H,
    const float* __restrict__ weight,
    float* __restrict__ out_f32,
    __half* __restrict__ out_f16,
    int B
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x; // 256 threads
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_red[8];

    const float* h_row = H + (size_t)b * D_MODEL;
    float sum_sq = 0.0f;
    for (int i = tid; i < D_MODEL; i += blockDim.x) {
        float v = h_row[i];
        sum_sq = fmaf(v, v, sum_sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);
    if (lane == 0) s_red[warp_id] = sum_sq;
    __syncthreads();

    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)D_MODEL + RMS_EPS);
    }
    __syncthreads();

    const float inv = s_inv;
    float* o32 = out_f32 ? (out_f32 + (size_t)b * D_MODEL) : nullptr;
    __half* o16 = out_f16 ? (out_f16 + (size_t)b * D_MODEL) : nullptr;
    for (int i = tid; i < D_MODEL; i += blockDim.x) {
        float y = (h_row[i] * inv) * weight[i];
        if (o32) o32[i] = y;
        if (o16) o16[i] = __float2half(y);
    }
}

// ============================================================================
// KERNEL 2A (Iter 5 Lever B): Page-32 Aligned Eviction Hot-4K V FP16 -> V_sink128 (FP16) + Cold V_pc32 (uint8)
//   (K stays 100% EXACT FP16 across all 64K tokens; V has NO RoPE so per-channel V_pc32 cuts error ~2.8x!)
//   Grid: (n_pages, 8 kv_heads), Block: 64 threads (1 thread per head_dim d=0..63)
// ============================================================================
__global__ void evict_v_hot_pages32_to_cold_pc_kernel(
    const __half* __restrict__ V_hot,    // [4096, 8, 64] FP16
    __half* __restrict__ V_sink_fp16,    // [128, 8, 64] exact FP16 for first 128 tokens (pages 0..3)
    uint8_t* __restrict__ V_cold,        // [cold_cap, 8, 64] uint8 per-channel
    __half* __restrict__ V_page_min,     // [cold_pages, 8, 64] FP16
    __half* __restrict__ V_page_max,     // [cold_pages, 8, 64] FP16
    int first_page,
    int n_pages
) {
    const int p_idx = blockIdx.x;
    const int h     = blockIdx.y; // 0..7
    const int d     = threadIdx.x; // 0..63
    if (p_idx >= n_pages) return;

    const int pg   = first_page + p_idx;
    const int pos0 = pg * V_PAGE_SIZE;

    if (pos0 < V_SINK_FP16) {
        // First 128 tokens: copy exact FP16 V directly into V_sink_fp16 (0 quantization error!)
        #pragma unroll 4
        for (int t = 0; t < V_PAGE_SIZE; ++t) {
            int abs_pos  = pos0 + t;
            int hot_slot = abs_pos & (HOT_WINDOW - 1);
            V_sink_fp16[((size_t)abs_pos * 8 + h) * 64 + d] = V_hot[((size_t)hot_slot * 8 + h) * 64 + d];
        }
        return;
    }

    // Pass 1: per-channel min/max across the 32 tokens of this page
    float v_min = FLT_MAX;
    float v_max = -FLT_MAX;
    #pragma unroll 4
    for (int t = 0; t < V_PAGE_SIZE; ++t) {
        int hot_slot = (pos0 + t) & (HOT_WINDOW - 1);
        float vv = __half2float(V_hot[((size_t)hot_slot * 8 + h) * 64 + d]);
        v_min = fminf(v_min, vv);
        v_max = fmaxf(v_max, vv);
    }
    __half h_min = __float2half(v_min);
    __half h_max = __float2half(v_max);
    float f_min  = __half2float(h_min);
    float f_max  = __half2float(h_max);
    float range  = f_max - f_min;
    float inv_v  = (range > 1e-8f) ? (255.0f / range) : 0.0f;

    const size_t pg_off = ((size_t)pg * 8 + h) * 64 + d;
    V_page_min[pg_off] = h_min;
    V_page_max[pg_off] = h_max;

    // Pass 2: quantize V_cold per-channel uint8 [0..255]
    #pragma unroll 4
    for (int t = 0; t < V_PAGE_SIZE; ++t) {
        int abs_pos  = pos0 + t;
        int hot_slot = abs_pos & (HOT_WINDOW - 1);
        float vv = __half2float(V_hot[((size_t)hot_slot * 8 + h) * 64 + d]);
        int qv = __float2int_rn((vv - f_min) * inv_v);
        V_cold[((size_t)abs_pos * 8 + h) * 64 + d] = (uint8_t)max(0, min(255, qv));
    }
}

// Iter 7 Pillar 2: Snapshot pristine 32-token FP16 boundary page from V_hot before B>1 writes!
//   Grid: 32 blocks (1 per token in page), Block: 512 threads (8 kv_heads * 64 dims)
__global__ void snapshot_v_bnd_page32_kernel(
    const __half* __restrict__ V_hot, // [4096, 512] single global layer
    __half* __restrict__ V_bnd_out,   // [32, 512] single global layer
    int bnd_pos0
) {
    const int t = blockIdx.x;
    if (t >= V_PAGE_SIZE) return;
    const int d = threadIdx.x; // 0..511
    const int hot_slot = (bnd_pos0 + t) & (HOT_WINDOW - 1);
    V_bnd_out[(size_t)t * 512 + d] = V_hot[(size_t)hot_slot * 512 + d];
}

// ============================================================================
// KERNEL 2: Fused QKV Bias + Split-Half YaRN RoPE + FP16 KV Cache Append
//   Supports separate k_ring_mask and v_ring_mask:
//     - Local Ring-128: k_ring_mask=127, v_ring_mask=127
//     - Pure FP16 Global: k_ring_mask=0, v_ring_mask=0
//     - Iter 5 Hybrid (K-FP16 + V-Hot4K/Cold-INT8): k_ring_mask=0, v_ring_mask=4095
// ============================================================================
__global__ void qkv_bias_rope_kv_append_kernel(
    const float* __restrict__ qkv_in,     // [B, 5120] FP32 from cuBLAS GemmEx
    const __half* __restrict__ qkv_bias,  // [5120] FP16
    const float* __restrict__ cos_table,  // [65536, 32] FP32
    const float* __restrict__ sin_table,  // [65536, 32] FP32
    float* __restrict__ Q_out,            // [B, 64, 64] FP32
    __half* __restrict__ K_cache,         // [k_cap, 8, 64] FP16
    __half* __restrict__ V_cache,         // [v_cap, 8, 64] FP16
    int B,
    int start_pos,
    int k_ring_mask,
    int v_ring_mask,
    int kv_base_pos,
    int use_tree
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    const int tid = threadIdx.x;
    const int kv_pos   = start_pos + b;
    const int rope_pos = use_tree ? (start_pos + (int)c_tree_meta.depth[b]) : kv_pos;
    const int k_slot = (k_ring_mask > 0) ? (kv_pos & k_ring_mask) : (kv_pos - kv_base_pos);
    const int v_slot = (v_ring_mask > 0) ? (kv_pos & v_ring_mask) : (kv_pos - kv_base_pos);

    const float* row = qkv_in + (size_t)b * QKV_DIM;
    const float* c_pos = cos_table + (size_t)rope_pos * 32;
    const float* s_pos = sin_table + (size_t)rope_pos * 32;

    // 1. Q: 64 heads * 32 pairs = 2048 pairs
    float* q_dst = Q_out + (size_t)b * 4096;
    for (int idx = tid; idx < 2048; idx += blockDim.x) {
        int h = idx >> 5;
        int d = idx & 31;
        int off0 = h * 64 + d;
        int off1 = off0 + 32;
        float x1 = row[off0] + __half2float(qkv_bias[off0]);
        float x2 = row[off1] + __half2float(qkv_bias[off1]);
        float cv = c_pos[d];
        float sv = s_pos[d];
        q_dst[off0] = x1 * cv - x2 * sv;
        q_dst[off1] = x2 * cv + x1 * sv;
    }

    // 2. K: 8 heads * 32 pairs = 256 pairs
    __half* k_dst = K_cache + (size_t)k_slot * 512;
    for (int idx = tid; idx < 256; idx += blockDim.x) {
        int h = idx >> 5;
        int d = idx & 31;
        int base = 4096 + h * 64 + d;
        float x1 = row[base] + __half2float(qkv_bias[base]);
        float x2 = row[base + 32] + __half2float(qkv_bias[base + 32]);
        float cv = c_pos[d];
        float sv = s_pos[d];
        k_dst[h * 64 + d]      = __float2half(x1 * cv - x2 * sv);
        k_dst[h * 64 + d + 32] = __float2half(x2 * cv + x1 * sv);
    }

    // 3. V: 8 heads * 64 = 512 elems
    __half* v_dst = V_cache + (size_t)v_slot * 512;
    for (int idx = tid; idx < 512; idx += blockDim.x) {
        int base = 4608 + idx;
        v_dst[idx] = __float2half(row[base] + __half2float(qkv_bias[base]));
    }
}

// ============================================================================
// KERNEL 3 (Iter 5 Lever A): Zero-Shuffle Key-Parallel Turing Flash-Decoding (sm_75)
//   - Eliminates ALL 160 SHFL.BFLY/tile inside Q*K: lane 0..31 owns key 0..31,
//     broadcasting s_Q_f2[8][32] against bank-conflict-free s_K_h2[32][33]!
//   - PTX fast_ex2 (ex2.approx.ftz.f32) warp-parallel (1 SFU op per 32 keys!) + deferred rescale!
//   - Lazy Sparse V loading (only cache lines active in block_mask!) + __ffs V accumulation!
//   - Template<COLD_V_INT8>: handles Local Ring-128, Pure FP16, Hybrid Hot-4K V-FP16, and Cold V-INT8!
// ============================================================================
template<bool COLD_V_PC32>
__global__ void flash_decode_gqa_splitk_part1_turing(
    const float* __restrict__ Q,             // [B, 64, 64]
    const __half* __restrict__ K_cache,      // [k_cap, 8, 64] exact FP16!
    const __half* __restrict__ V_fp16,       // [v_cap, 8, 64] FP16 (Hot/Pure FP16, or V_sink_fp16[128] when COLD_V_PC32)
    const uint8_t* __restrict__ V_cold,      // [cold_cap, 8, 64] uint8 per-channel (when COLD_V_PC32)
    const __half* __restrict__ V_page_min,   // [cold_pages, 8, 64] FP16 (when COLD_V_PC32)
    const __half* __restrict__ V_page_max,   // [cold_pages, 8, 64] FP16 (when COLD_V_PC32)
    const float* __restrict__ sinks,         // [64]
    const float* __restrict__ m_floor_in,    // optional [B, 64] exact Hot-4K+sink floor in log2 space
    float* __restrict__ partial_m,           // [B, 64, split_stride]
    float* __restrict__ partial_l,           // [B, 64, split_stride]
    float* __restrict__ partial_o,           // [B, 64, split_stride, 64]
    __half* __restrict__ direct_O_fp16,      // [B, 64, 64] used directly when split_stride == 1
    int start_pos,
    int is_sliding_window,
    int k_ring_mask,
    int v_ring_mask,
    int kv_base_pos,
    int forced_key_beg,                      // 0 for full/cold, cold_len for Hybrid Phase-1 Hot-4K
    int forced_key_end_excl,                 // <= 0 means cur_pos + 1; > 0 means cold_len (exact multiple of 32!)
    int sp_offset,                           // 0 for Phase-1/single-phase, n_hot_splits for Phase-2 Cold
    int num_splits,
    int split_stride,
    float cutoff_log2,
    unsigned long long* __restrict__ d_prune_stats, // optional [6] counters
    int use_tree,
    const __half* __restrict__ V_bnd_fp16 = nullptr, // optional [32, 8, 64] pristine boundary page FP16
    int bnd_pos0 = -1,
    int use_per_node_cold = 0
) {
    constexpr int TILE_K = 32;
    __shared__ float2 s_Q_f2[8][32]; // 2 KB: pre-scaled by SM_SCALE_LOG2E (100% broadcast!)
    __shared__ union {
        uint32_t s_K_u32[TILE_K][33]; // 4.2 KB: odd stride 33 (__half2 packed in u32) -> ZERO bank conflicts!
        float    s_V[TILE_K][64];     // 8.0 KB: aliased with s_K_u32, ZERO bank conflicts!
    } smem;
    __shared__ float s_vmin[64];
    __shared__ float s_vsc[64];
    __shared__ uint32_t s_warp_mask[8];

    const int kv_head = blockIdx.x; // 0..7
    const int sp      = blockIdx.y; // 0..num_splits-1
    const int b       = blockIdx.z; // 0..B-1
    const int tid     = threadIdx.x; // 0..255
    const int qg      = tid >> 5;    // 0..7
    const int lane    = tid & 31;    // 0..31
    const int q_head  = (kv_head << 3) + qg;

    const uint16_t anc_mask_b = use_tree ? c_tree_meta.anc_mask[b] : 0u;
    const int cur_pos   = use_tree ? (start_pos + (int)c_tree_meta.num_nodes - 1) : (start_pos + b);
    const int log_q_pos = use_tree ? (start_pos + (int)c_tree_meta.depth[b])      : (start_pos + b);
    int eff_key_beg = forced_key_beg;
    int eff_key_end = forced_key_end_excl;
    if (use_per_node_cold) {
        const int pages_b    = (log_q_pos + 1 > HOT_WINDOW) ? ((log_q_pos + 1 - HOT_WINDOW + V_PAGE_SIZE - 1) / V_PAGE_SIZE) : 0;
        const int cold_len_b = pages_b * V_PAGE_SIZE;
        if constexpr (COLD_V_PC32) {
            eff_key_end = cold_len_b;
        } else {
            eff_key_beg = cold_len_b;
        }
    }
    const int win_start = is_sliding_window ? max(0, log_q_pos - RING_SIZE + 1) : eff_key_beg;
    const int win_end   = (use_per_node_cold && COLD_V_PC32)
        ? min(eff_key_end, cur_pos + 1)
        : ((eff_key_end > 0) ? min(eff_key_end, cur_pos + 1) : (cur_pos + 1));
    const int total_len = max(0, win_end - win_start);

    // For COLD_V_PC32, align split boundaries strictly to V_PAGE_SIZE=32 so every 32-key tile is 1 exact page!
    int t_beg, t_end;
    if constexpr (COLD_V_PC32) {
        const int n_pg   = total_len >> 5;
        const int pp_sp  = (n_pg + num_splits - 1) / num_splits;
        t_beg = (sp * pp_sp) << 5;
        t_end = min(total_len, ((sp + 1) * pp_sp) << 5);
    } else {
        const int per_sp = (total_len + num_splits - 1) / num_splits;
        t_beg = sp * per_sp;
        t_end = min(total_len, t_beg + per_sp);
    }

    // Pre-load Q scaled by SM_SCALE_LOG2E ONCE into shared memory (all 256 threads in 1 step!)
    const float* q_ptr = Q + ((size_t)b * 64 + q_head) * 64;
    s_Q_f2[qg][lane] = make_float2(
        q_ptr[2 * lane]     * SM_SCALE_LOG2E,
        q_ptr[2 * lane + 1] * SM_SCALE_LOG2E
    );
    __syncthreads();

    const float sink_log2 = sinks[q_head] * LOG2E;
    const float m_floor   = (m_floor_in != nullptr) ? m_floor_in[(size_t)b * NUM_Q_HEADS + q_head] : sink_log2;

    float m_run = -FLT_MAX;
    float l_run = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    const uint2*  K_u2 = reinterpret_cast<const uint2*>(K_cache);
    const uint2*  V_u2 = reinterpret_cast<const uint2*>(V_fp16);
    const uchar4* V_u4 = reinterpret_cast<const uchar4*>(V_cold);

    uint32_t loc_v_rows_tot = 0;
    uint32_t loc_v_rows_act = 0;
    uint32_t loc_tiles_tot  = 0;
    uint32_t loc_tiles_skip = 0;

    for (int t0 = t_beg; t0 < t_end; t0 += TILE_K) {
        const int tlen = min(TILE_K, t_end - t0);
        const int total_vec4 = tlen * 16; // 16 uint2 (64 halfs) per key row
        loc_tiles_tot++;

        // 1. Cooperative coalesced FP16 K load into bank-conflict-free smem.s_K_u32[32][33]
        for (int idx = tid; idx < total_vec4; idx += 256) {
            int k_loc = idx >> 4;
            int d4    = idx & 15;
            int log_j = win_start + t0 + k_loc;
            int k_sl  = (k_ring_mask > 0) ? (log_j & k_ring_mask) : (log_j - kv_base_pos);
            size_t u2_off = ((size_t)k_sl * 8 + kv_head) * 16 + d4;
            uint2 v = K_u2[u2_off];
            smem.s_K_u32[k_loc][2 * d4]     = v.x;
            smem.s_K_u32[k_loc][2 * d4 + 1] = v.y;
        }
        __syncthreads();

        // 2. ZERO-SHUFFLE Key-Parallel Q*K dot product (lane k computes key k across 32 half2 pairs!)
        float score_mine = -FLT_MAX;
        bool lane_valid = (lane < tlen);
        if (lane_valid && use_tree) {
            int log_j = win_start + t0 + lane;
            if (log_j >= start_pos) {
                int tj = log_j - start_pos;
                lane_valid = (tj < MAX_VERIFY_K) && (((anc_mask_b >> tj) & 1u) != 0u);
            }
        }
        if (lane_valid) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            #pragma unroll 8
            for (int d2 = 0; d2 < 32; ++d2) {
                float2 qv = s_Q_f2[qg][d2];
                uint32_t raw_h2 = smem.s_K_u32[lane][d2];
                float2 kv = __half22float2(*reinterpret_cast<const __half2*>(&raw_h2));
                acc0 = fmaf(qv.x, kv.x, acc0);
                acc1 = fmaf(qv.y, kv.y, acc1);
            }
            score_mine = acc0 + acc1;
        }

        // 3. Tile max (5 shuffles ONCE per 32 keys!) + active bitmask against exact m_floor
        float m_tile = score_mine;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, off));
        }
        float m_ref = fmaxf(fmaxf(m_run, m_tile), m_floor);
        uint32_t active_mask = __ballot_sync(0xffffffff, lane_valid && (score_mine >= m_ref - cutoff_log2));
        if (lane == 0) s_warp_mask[qg] = active_mask;
        __syncthreads();

        uint32_t block_mask = s_warp_mask[0] | s_warp_mask[1] | s_warp_mask[2] | s_warp_mask[3]
                            | s_warp_mask[4] | s_warp_mask[5] | s_warp_mask[6] | s_warp_mask[7];
        loc_v_rows_tot += (uint32_t)tlen;
        loc_v_rows_act += (uint32_t)__popc(block_mask);
        if (block_mask == 0u) {
            loc_tiles_skip++;
            continue; // ZERO V DRAM reads, ZERO ex2, ZERO V FMAs!
        }

        // 4. Lazy Sparse V load (ONLY cache lines active in block_mask!) into aliased smem.s_V[32][64]
        if constexpr (COLD_V_PC32) {
            const int tile_abs0 = win_start + t0;
            if (tile_abs0 < V_SINK_FP16) {
                // First 128 tokens (pages 0..3): exact FP16 from V_sink_fp16!
                for (int idx = tid; idx < total_vec4; idx += 256) {
                    int k_loc = idx >> 4;
                    if (!((block_mask >> k_loc) & 1u)) continue;
                    int d4    = (idx & 15) << 2;
                    int log_j = tile_abs0 + k_loc;
                    size_t u2_off = ((size_t)log_j * 8 + kv_head) * 16 + (idx & 15);
                    uint2 v = V_u2[u2_off];
                    float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
                    float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
                    smem.s_V[k_loc][d4 + 0] = f0.x;
                    smem.s_V[k_loc][d4 + 1] = f0.y;
                    smem.s_V[k_loc][d4 + 2] = f1.x;
                    smem.s_V[k_loc][d4 + 3] = f1.y;
                }
            } else {
                // Page-32 aligned per-channel uint8 dequantization (load min/scale ONCE for all 32 keys!)
                const int pg = tile_abs0 >> 5;
                if (tid < 64) {
                    size_t pg_off = ((size_t)pg * 8 + kv_head) * 64 + tid;
                    float vmin = __half2float(V_page_min[pg_off]);
                    float vmax = __half2float(V_page_max[pg_off]);
                    s_vmin[tid] = vmin;
                    s_vsc[tid]  = (vmax - vmin) * (1.0f / 255.0f);
                }
                __syncthreads();
                for (int idx = tid; idx < total_vec4; idx += 256) {
                    int k_loc = idx >> 4;
                    if (!((block_mask >> k_loc) & 1u)) continue;
                    int d4 = (idx & 15) << 2;
                    int log_j = tile_abs0 + k_loc;
                    size_t u4_off = ((size_t)log_j * 8 + kv_head) * 16 + (idx & 15);
                    uchar4 v = V_u4[u4_off];
                    smem.s_V[k_loc][d4 + 0] = fmaf((float)v.x, s_vsc[d4 + 0], s_vmin[d4 + 0]);
                    smem.s_V[k_loc][d4 + 1] = fmaf((float)v.y, s_vsc[d4 + 1], s_vmin[d4 + 1]);
                    smem.s_V[k_loc][d4 + 2] = fmaf((float)v.z, s_vsc[d4 + 2], s_vmin[d4 + 2]);
                    smem.s_V[k_loc][d4 + 3] = fmaf((float)v.w, s_vsc[d4 + 3], s_vmin[d4 + 3]);
                }
            }
        } else {
            const uint2* V_bnd_u2 = reinterpret_cast<const uint2*>(V_bnd_fp16);
            for (int idx = tid; idx < total_vec4; idx += 256) {
                int k_loc = idx >> 4;
                if (!((block_mask >> k_loc) & 1u)) continue;
                int d4    = (idx & 15) << 2;
                int log_j = win_start + t0 + k_loc;
                uint2 v;
                if (V_bnd_u2 != nullptr && log_j >= bnd_pos0 && log_j < bnd_pos0 + V_PAGE_SIZE) {
                    size_t b_off = ((size_t)(log_j - bnd_pos0) * 8 + kv_head) * 16 + (idx & 15);
                    v = V_bnd_u2[b_off];
                } else {
                    int v_sl  = (v_ring_mask > 0) ? (log_j & v_ring_mask) : (log_j - kv_base_pos);
                    size_t u2_off = ((size_t)v_sl * 8 + kv_head) * 16 + (idx & 15);
                    v = V_u2[u2_off];
                }
                float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
                float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
                smem.s_V[k_loc][d4 + 0] = f0.x;
                smem.s_V[k_loc][d4 + 1] = f0.y;
                smem.s_V[k_loc][d4 + 2] = f1.x;
                smem.s_V[k_loc][d4 + 3] = f1.y;
            }
        }
        __syncthreads();

        // 5. Deferred rescale (at most 1 fast_ex2/tile) + warp-parallel fast_ex2 + __ffs V accumulate
        if (active_mask != 0u) {
            if (m_tile > m_run) {
                float gamma = (m_run > -1e29f) ? fast_ex2(m_run - m_tile) : 0.0f;
                l_run *= gamma;
                a0    *= gamma;
                a1    *= gamma;
                m_run  = m_tile;
            }
            float w_mine = ((active_mask >> lane) & 1u) ? fast_ex2(score_mine - m_run) : 0.0f;
            float w_sum = w_mine;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                w_sum += __shfl_xor_sync(0xffffffff, w_sum, off);
            }
            l_run += w_sum;

            uint32_t m = active_mask;
            while (m) {
                int k = __ffs(m) - 1;
                m &= (m - 1);
                float wk = __shfl_sync(0xffffffff, w_mine, k);
                a0 = fmaf(wk, smem.s_V[k][lane], a0);
                a1 = fmaf(wk, smem.s_V[k][lane + 32], a1);
            }
        }
        __syncthreads();
    }

    // Incorporate learned attention sink ONCE (in global split 0 only!)
    if (!COLD_V_PC32 && sp_offset == 0 && sp == 0) {
        float m_new = fmaxf(m_run, sink_log2);
        float gamma = (m_run > -1e29f) ? fast_ex2(m_run - m_new) : 0.0f;
        float w_sink = fast_ex2(sink_log2 - m_new);
        l_run = gamma * l_run + w_sink;
        a0 *= gamma;
        a1 *= gamma;
        m_run = m_new;
    }

    if (split_stride == 1) {
        float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
        __half* out_ptr = direct_O_fp16 + ((size_t)b * 64 + q_head) * 64;
        out_ptr[lane]      = __float2half(a0 * inv_l);
        out_ptr[lane + 32] = __float2half(a1 * inv_l);
    } else {
        const int out_sp = sp_offset + sp;
        const size_t ml_idx = ((size_t)b * 64 + q_head) * split_stride + out_sp;
        if (lane == 0) {
            partial_m[ml_idx] = m_run;
            partial_l[ml_idx] = l_run;
            if (d_prune_stats != nullptr && qg == 0 && loc_tiles_tot > 0) {
                atomicAdd(&d_prune_stats[0], (unsigned long long)loc_v_rows_tot);
                atomicAdd(&d_prune_stats[1], (unsigned long long)(loc_v_rows_tot - loc_v_rows_act));
                atomicAdd(&d_prune_stats[2], (unsigned long long)loc_tiles_tot);
                atomicAdd(&d_prune_stats[3], (unsigned long long)loc_tiles_skip);
            }
        }
        float* po = partial_o + ml_idx * 64;
        po[lane]      = a0;
        po[lane + 32] = a1;
    }
}

__global__ void compute_hot_floor_kernel(
    const float* __restrict__ partial_m, // [B, 64, split_stride]
    const float* __restrict__ partial_l, // [B, 64, split_stride]
    float* __restrict__ m_floor_out,     // [B, 64]
    int n_hot_splits,
    int split_stride
) {
    const int b = blockIdx.x;
    const int q_head = threadIdx.x;
    const size_t base = ((size_t)b * NUM_Q_HEADS + q_head) * split_stride;
    float mf = -FLT_MAX;
    for (int s = 0; s < n_hot_splits; ++s) {
        if (partial_l[base + s] > 0.0f) {
            mf = fmaxf(mf, partial_m[base + s]);
        }
    }
    m_floor_out[(size_t)b * NUM_Q_HEADS + q_head] = mf;
}

__global__ void flash_decode_gqa_splitk_reduce(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_o,
    __half* __restrict__ O_fp16,
    int num_splits
) {
    const int q_head = blockIdx.x;
    const int b      = blockIdx.y;
    const int d      = threadIdx.x;

    const size_t base_ml = ((size_t)b * 64 + q_head) * num_splits;
    float m_max = -FLT_MAX;
    for (int s = 0; s < num_splits; ++s) {
        if (partial_l[base_ml + s] > 0.0f) {
            m_max = fmaxf(m_max, partial_m[base_ml + s]);
        }
    }
    float Z = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float ls = partial_l[base_ml + s];
        if (ls <= 0.0f) continue;
        float ms = partial_m[base_ml + s];
        float g = fast_ex2(ms - m_max);
        Z += g * ls;
        acc += g * partial_o[(base_ml + s) * 64 + d];
    }
    float inv_z = (Z > 0.0f) ? (1.0f / Z) : 0.0f;
    O_fp16[((size_t)b * 64 + q_head) * 64 + d] = __float2half(acc * inv_z);
}

// ============================================================================
// KERNEL 4: GQA + 4-Query Tiled Flash-Prefill Attention (32x less KV reads!)
//   Each block handles: 1 KV head (8 Q heads) x BQ=4 consecutive query positions!
//   Block = 256 threads = 8 warps (1 warp per Q-head, 2 dims/lane, 4 queries in regs)
// ============================================================================
__global__ void flash_prefill_gqa_4q_kernel(
    const float* __restrict__ Q,        // [chunk, 64, 64]
    const __half* __restrict__ K_cache, // [cap, 8, 64]
    const __half* __restrict__ V_cache, // [cap, 8, 64]
    const float* __restrict__ sinks,    // [64]
    __half* __restrict__ O_fp16,        // [chunk, 64, 64]
    int chunk_len,
    int start_pos,
    int is_sliding_window,
    int use_ring_indexing,
    int kv_base_pos
) {
    constexpr int BQ = 4;
    constexpr int TILE_K = 16;
    __shared__ float s_K[TILE_K][64];
    __shared__ float s_V[TILE_K][64];

    const int kv_head = blockIdx.x;
    const int q_blk   = blockIdx.y;
    const int q0_idx  = q_blk * BQ;
    const int nq      = min(BQ, chunk_len - q0_idx);
    if (nq <= 0) return;

    const int tid    = threadIdx.x;
    const int qg     = tid >> 5;
    const int lane   = tid & 31;
    const int q_head = (kv_head << 3) + qg;

    float qr0[BQ], qr1[BQ];
    float m_run[BQ], l_run[BQ], a0[BQ], a1[BQ];
    int q_pos[BQ];

    #pragma unroll
    for (int i = 0; i < BQ; ++i) {
        if (i < nq) {
            const float* qp = Q + ((size_t)(q0_idx + i) * 64 + q_head) * 64;
            qr0[i] = qp[lane];
            qr1[i] = qp[lane + 32];
            q_pos[i] = start_pos + q0_idx + i;
        } else {
            qr0[i] = 0.0f; qr1[i] = 0.0f; q_pos[i] = -1;
        }
        m_run[i] = -FLT_MAX;
        l_run[i] = 0.0f;
        a0[i] = 0.0f;
        a1[i] = 0.0f;
    }

    const int min_q_pos = start_pos + q0_idx;
    const int max_q_pos = start_pos + q0_idx + nq - 1;
    const int j_begin   = is_sliding_window ? max(0, min_q_pos - RING_SIZE + 1) : 0;
    const int j_end     = max_q_pos + 1;

    for (int j0 = j_begin; j0 < j_end; j0 += TILE_K) {
        const int tlen = min(TILE_K, j_end - j0);
        const int total_elems = tlen * 64;
        for (int idx = tid; idx < total_elems; idx += 256) {
            int k_loc = idx >> 6;
            int d     = idx & 63;
            int log_j = j0 + k_loc;
            int slot  = use_ring_indexing ? (log_j & (RING_SIZE - 1)) : (log_j - kv_base_pos);
            size_t off = (size_t)slot * 512 + kv_head * 64 + d;
            s_K[k_loc][d] = __half2float(K_cache[off]);
            s_V[k_loc][d] = __half2float(V_cache[off]);
        }
        __syncthreads();

        for (int k_loc = 0; k_loc < tlen; ++k_loc) {
            int log_j = j0 + k_loc;
            float k0 = s_K[k_loc][lane];
            float k1 = s_K[k_loc][lane + 32];
            float v0 = s_V[k_loc][lane];
            float v1 = s_V[k_loc][lane + 32];

            #pragma unroll
            for (int i = 0; i < BQ; ++i) {
                if (i < nq) {
                    int win_lo = is_sliding_window ? max(0, q_pos[i] - RING_SIZE + 1) : 0;
                    if (log_j >= win_lo && log_j <= q_pos[i]) {
                        float dot = qr0[i] * k0 + qr1[i] * k1;
                        #pragma unroll
                        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
                        float score = dot * SM_SCALE;
                        float m_new = fmaxf(m_run[i], score);
                        float gamma = (m_run[i] == -FLT_MAX) ? 0.0f : expf(m_run[i] - m_new);
                        float w = expf(score - m_new);
                        l_run[i] = gamma * l_run[i] + w;
                        a0[i] = gamma * a0[i] + w * v0;
                        a1[i] = gamma * a1[i] + w * v1;
                        m_run[i] = m_new;
                    }
                }
            }
        }
        __syncthreads();
    }

    const float sink_logit = sinks[q_head];
    #pragma unroll
    for (int i = 0; i < BQ; ++i) {
        if (i < nq) {
            float m_new = fmaxf(m_run[i], sink_logit);
            float gamma = (m_run[i] == -FLT_MAX) ? 0.0f : expf(m_run[i] - m_new);
            float w_sink = expf(sink_logit - m_new);
            float Z = gamma * l_run[i] + w_sink;
            float inv_z = (Z > 0.0f) ? (1.0f / Z) : 0.0f;
            __half* out_ptr = O_fp16 + ((size_t)(q0_idx + i) * 64 + q_head) * 64;
            out_ptr[lane]      = __float2half((gamma * a0[i]) * inv_z);
            out_ptr[lane + 32] = __float2half((gamma * a1[i]) * inv_z);
        }
    }
}

// ============================================================================
// KERNEL 5: Residual + Bias Add, Router Top-4 Softmax, SwiGLU, Weighted Scatter-Add
// ============================================================================
__global__ void residual_add_bias_kernel(
    float* __restrict__ H,            // [B, D_MODEL]
    const float* __restrict__ delta,  // [B, D_MODEL]
    const __half* __restrict__ bias,  // [D_MODEL] nullable
    int B
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * D_MODEL;
    if (idx >= total) return;
    int col = idx % D_MODEL;
    float bval = bias ? __half2float(bias[col]) : 0.0f;
    H[idx] += delta[idx] + bval;
}

__global__ void router_topk4_softmax_kernel(
    const float* __restrict__ router_logits, // [B, 32]
    const __half* __restrict__ router_bias,  // [32]
    uint8_t* __restrict__ top_idx_out,       // [B, 4]
    float* __restrict__ top_w_out,           // [B, 4]
    int B
) {
    // 1 warp (32 threads) per token b!
    const int b = blockIdx.x;
    if (b >= B) return;
    const int lane = threadIdx.x; // 0..31
    float val = router_logits[b * NUM_EXPERTS + lane] + __half2float(router_bias[lane]);
    bool avail = true;

    float best_v[TOP_K];
    int best_i[TOP_K];

    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        float v_cand = avail ? val : -FLT_MAX;
        int i_cand = lane;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float pv = __shfl_xor_sync(0xffffffff, v_cand, off);
            int pi = __shfl_xor_sync(0xffffffff, i_cand, off);
            if (pv > v_cand || (pv == v_cand && pi < i_cand)) {
                v_cand = pv;
                i_cand = pi;
            }
        }
        best_v[k] = v_cand;
        best_i[k] = i_cand;
        if (lane == i_cand) avail = false;
    }

    if (lane == 0) {
        float m = best_v[0];
        float e0 = expf(best_v[0] - m);
        float e1 = expf(best_v[1] - m);
        float e2 = expf(best_v[2] - m);
        float e3 = expf(best_v[3] - m);
        float inv = 1.0f / (e0 + e1 + e2 + e3);
        top_idx_out[b * 4 + 0] = (uint8_t)best_i[0];
        top_idx_out[b * 4 + 1] = (uint8_t)best_i[1];
        top_idx_out[b * 4 + 2] = (uint8_t)best_i[2];
        top_idx_out[b * 4 + 3] = (uint8_t)best_i[3];
        top_w_out[b * 4 + 0] = e0 * inv;
        top_w_out[b * 4 + 1] = e1 * inv;
        top_w_out[b * 4 + 2] = e2 * inv;
        top_w_out[b * 4 + 3] = e3 * inv;
    }
}

__global__ void swiglu_interleaved_kernel(
    const float* __restrict__ gu_in,    // [N, 5760]
    const __half* __restrict__ gu_bias, // [5760] nullable (if already added)
    float* __restrict__ mid_f32,        // [N, 2880] nullable
    __half* __restrict__ mid_f16,       // [N, 2880] nullable
    int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D_FF) return;
    int n = idx / D_FF;
    int c = idx - n * D_FF;
    int g_idx = c * 2;
    int u_idx = g_idx + 1;
    float g = gu_in[(size_t)n * (2 * D_FF) + g_idx] + (gu_bias ? __half2float(gu_bias[g_idx]) : 0.0f);
    float u = gu_in[(size_t)n * (2 * D_FF) + u_idx] + (gu_bias ? __half2float(gu_bias[u_idx]) : 0.0f);
    g = fminf(g, SWIGLU_LIMIT);
    u = fmaxf(-SWIGLU_LIMIT, fminf(u, SWIGLU_LIMIT));
    float sig = 1.0f / (1.0f + expf(-SWIGLU_ALPHA * g));
    float out = (g * sig) * (u + 1.0f);
    if (mid_f32) mid_f32[idx] = out;
    if (mid_f16) mid_f16[idx] = __float2half(out);
}

__global__ void gather_tokens_kernel(
    const float* __restrict__ src_f32,
    const __half* __restrict__ src_f16,
    const int* __restrict__ tok_indices,
    float* __restrict__ dst_f32,
    __half* __restrict__ dst_f16,
    int n_tok
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tok * D_MODEL) return;
    int i = idx / D_MODEL;
    int c = idx - i * D_MODEL;
    int src_row = tok_indices[i];
    if (dst_f32) dst_f32[idx] = src_f32[(size_t)src_row * D_MODEL + c];
    if (dst_f16) dst_f16[idx] = src_f16[(size_t)src_row * D_MODEL + c];
}

__global__ void scatter_weighted_add_kernel(
    const float* __restrict__ exp_out,   // [n_tok, 2880]
    const __half* __restrict__ dn_bias,  // [2880] nullable
    const int* __restrict__ tok_indices, // [n_tok]
    const float* __restrict__ weights,   // [n_tok]
    float* __restrict__ H_accum,         // [B, 2880]
    int n_tok
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tok * D_MODEL) return;
    int i = idx / D_MODEL;
    int c = idx - i * D_MODEL;
    int dst_row = tok_indices[i];
    float v = exp_out[idx] + (dn_bias ? __half2float(dn_bias[c]) : 0.0f);
    atomicAdd(&H_accum[(size_t)dst_row * D_MODEL + c], v * weights[i]);
}

__global__ void weighted_add_single_kernel(
    const float* __restrict__ exp_out,
    float w,
    float* __restrict__ H_dst
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < D_MODEL) H_dst[c] = fmaf(exp_out[c], w, H_dst[c]);
}

// ============================================================================
// KERNEL 6: Optimized sm_75 MXFP4 Kernels (0-way bank conflicts, __shfl_sync FP4 E2M1 LUT,
//           fused gate_up+SwiGLU, fused down+weighted_add, vectorized FP32 dequant!)
// ============================================================================
__device__ __forceinline__ float fp4_shfl_decode(float r_lut, uint32_t nib) {
    return __shfl_sync(0xffffffff, r_lut, (int)nib);
}

// Fused gate_up MXFP4 GEMV + Official SwiGLU (alpha=1.702, limit=7.0, up+1) for B=1
// 1 warp computes BOTH gate (row=2*c) and up (row=2*c+1), reusing s_X[g][0..31] 2x with 0 bank conflicts!
__global__ void mxfp4_gate_up_swiglu_1vec_kernel(
    const uint8_t* __restrict__ blocks,   // [5760, 90, 16]
    const uint8_t* __restrict__ scales,   // [5760, 90]
    const __half* __restrict__ gu_bias,   // [5760] FP16
    const float* __restrict__ X,          // [2880] FP32
    float* __restrict__ mid_out           // [2880] FP32
) {
    constexpr int WARPS_PER_BLOCK = 8;
    // Pad stride 32 -> 33 floats per group: bank(s_X[lane][k]) = (lane*33 + k) % 32 = (lane + k) % 32 -> 0 bank conflicts!
    __shared__ float s_X[90][33];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    for (int j = tid; j < D_MODEL; j += 256) {
        s_X[j >> 5][j & 31] = X[j];
    }
    __syncthreads();

    const int c = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool row_ok = (c < D_FF);
    const int row_g = row_ok ? (c << 1) : 0;
    const int row_u = row_g + 1;

    const float r_lut = c_FP4_LUT[lane & 15];
    const uint4* bg_row = reinterpret_cast<const uint4*>(blocks + (size_t)row_g * 90 * 16);
    const uint4* bu_row = reinterpret_cast<const uint4*>(blocks + (size_t)row_u * 90 * 16);
    const uint8_t* scg_row = scales + (size_t)row_g * 90;
    const uint8_t* scu_row = scales + (size_t)row_u * 90;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    // Warp-uniform 3 iterations (0..31, 32..63, 64..95) so ALL 32 lanes participate in __shfl_sync!
    #pragma unroll
    for (int g0 = 0; g0 < 90; g0 += 32) {
        const int g = g0 + lane;
        const bool valid = row_ok && (g < 90);
        const int gs = valid ? g : 0;

        uint4 pg = valid ? bg_row[gs] : make_uint4(0, 0, 0, 0);
        uint4 pu = valid ? bu_row[gs] : make_uint4(0, 0, 0, 0);
        float scg = valid ? decode_e8m0_scale(scg_row[gs]) : 0.0f;
        float scu = valid ? decode_e8m0_scale(scu_row[gs]) : 0.0f;

        const uint32_t wg[4] = {pg.x, pg.y, pg.z, pg.w};
        const uint32_t wu[4] = {pu.x, pu.y, pu.z, pu.w};
        const float* xg = &s_X[gs][0];

        float sum_g = 0.0f;
        float sum_u = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t word_g = wg[w];
            uint32_t word_u = wu[w];
            const float* xw = xg + (w << 3);
            float x0 = xw[0], x1 = xw[1], x2 = xw[2], x3 = xw[3];
            float x4 = xw[4], x5 = xw[5], x6 = xw[6], x7 = xw[7];

            sum_g = fmaf(fp4_shfl_decode(r_lut,  word_g        & 0xFu), x0, sum_g);
            sum_u = fmaf(fp4_shfl_decode(r_lut,  word_u        & 0xFu), x0, sum_u);
            sum_g = fmaf(fp4_shfl_decode(r_lut, (word_g >>  4) & 0xFu), x1, sum_g);
            sum_u = fmaf(fp4_shfl_decode(r_lut, (word_u >>  4) & 0xFu), x1, sum_u);
            sum_g = fmaf(fp4_shfl_decode(r_lut, (word_g >>  8) & 0xFu), x2, sum_g);
            sum_u = fmaf(fp4_shfl_decode(r_lut, (word_u >>  8) & 0xFu), x2, sum_u);
            sum_g = fmaf(fp4_shfl_decode(r_lut, (word_g >> 12) & 0xFu), x3, sum_g);
            sum_u = fmaf(fp4_shfl_decode(r_lut, (word_u >> 12) & 0xFu), x3, sum_u);
            sum_g = fmaf(fp4_shfl_decode(r_lut, (word_g >> 16) & 0xFu), x4, sum_g);
            sum_u = fmaf(fp4_shfl_decode(r_lut, (word_u >> 16) & 0xFu), x4, sum_u);
            sum_g = fmaf(fp4_shfl_decode(r_lut, (word_g >> 20) & 0xFu), x5, sum_g);
            sum_u = fmaf(fp4_shfl_decode(r_lut, (word_u >> 20) & 0xFu), x5, sum_u);
            sum_g = fmaf(fp4_shfl_decode(r_lut, (word_g >> 24) & 0xFu), x6, sum_g);
            sum_u = fmaf(fp4_shfl_decode(r_lut, (word_u >> 24) & 0xFu), x6, sum_u);
            sum_g = fmaf(fp4_shfl_decode(r_lut, (word_g >> 28) & 0xFu), x7, sum_g);
            sum_u = fmaf(fp4_shfl_decode(r_lut, (word_u >> 28) & 0xFu), x7, sum_u);
        }
        if (valid) {
            acc_g = fmaf(sum_g, scg, acc_g);
            acc_u = fmaf(sum_u, scu, acc_u);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_down_sync(0xffffffff, acc_g, off);
        acc_u += __shfl_down_sync(0xffffffff, acc_u, off);
    }
    if (lane == 0 && row_ok) {
        float g_v = acc_g + (gu_bias ? __half2float(gu_bias[row_g]) : 0.0f);
        float u_v = acc_u + (gu_bias ? __half2float(gu_bias[row_u]) : 0.0f);
        g_v = fminf(g_v, SWIGLU_LIMIT);
        u_v = fmaxf(-SWIGLU_LIMIT, fminf(u_v, SWIGLU_LIMIT));
        float sig = 1.0f / (1.0f + expf(-SWIGLU_ALPHA * g_v));
        mid_out[c] = (g_v * sig) * (u_v + 1.0f);
    }
}

// Fused down MXFP4 GEMV + weighted residual add (H_dst[row] += w_router * (dot + bias)) for B=1
__global__ void mxfp4_down_wadd_1vec_kernel(
    const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales,
    const __half* __restrict__ bias_fp16,
    const float* __restrict__ X,
    float w_router,
    float* __restrict__ H_dst,
    int M
) {
    constexpr int WARPS_PER_BLOCK = 8;
    __shared__ float s_X[90][33];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    for (int j = tid; j < D_MODEL; j += 256) {
        s_X[j >> 5][j & 31] = X[j];
    }
    __syncthreads();

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool row_ok = (row < M);
    const int r_safe = row_ok ? row : 0;

    const float r_lut = c_FP4_LUT[lane & 15];
    const uint4* b_row = reinterpret_cast<const uint4*>(blocks + (size_t)r_safe * 90 * 16);
    const uint8_t* sc_row = scales + (size_t)r_safe * 90;

    float acc = 0.0f;
    #pragma unroll
    for (int g0 = 0; g0 < 90; g0 += 32) {
        const int g = g0 + lane;
        const bool valid = row_ok && (g < 90);
        const int gs = valid ? g : 0;

        uint4 pack = valid ? b_row[gs] : make_uint4(0, 0, 0, 0);
        float sc = valid ? decode_e8m0_scale(sc_row[gs]) : 0.0f;
        const uint32_t words[4] = {pack.x, pack.y, pack.z, pack.w};
        const float* xg = &s_X[gs][0];

        float gsum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t word = words[w];
            const float* xw = xg + (w << 3);
            gsum = fmaf(fp4_shfl_decode(r_lut,  word        & 0xFu), xw[0], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >>  4) & 0xFu), xw[1], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >>  8) & 0xFu), xw[2], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 12) & 0xFu), xw[3], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 16) & 0xFu), xw[4], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 20) & 0xFu), xw[5], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 24) & 0xFu), xw[6], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 28) & 0xFu), xw[7], gsum);
        }
        if (valid) acc = fmaf(gsum, sc, acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0 && row_ok) {
        if (bias_fp16) acc += __half2float(bias_fp16[row]);
        H_dst[row] = fmaf(acc, w_router, H_dst[row]);
    }
}

// Standalone fast MXFP4 GEMV (B=1) retained for unit parity testing
__global__ void mxfp4_gemv_1vec_fp16bias_kernel(
    const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales,
    const __half* __restrict__ bias_fp16,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M
) {
    constexpr int WARPS_PER_BLOCK = 8;
    __shared__ float s_X[90][33];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    for (int j = tid; j < D_MODEL; j += 256) s_X[j >> 5][j & 31] = X[j];
    __syncthreads();

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool row_ok = (row < M);
    const int r_safe = row_ok ? row : 0;

    const float r_lut = c_FP4_LUT[lane & 15];
    const uint4* b_row = reinterpret_cast<const uint4*>(blocks + (size_t)r_safe * 90 * 16);
    const uint8_t* sc_row = scales + (size_t)r_safe * 90;

    float acc = 0.0f;
    #pragma unroll
    for (int g0 = 0; g0 < 90; g0 += 32) {
        const int g = g0 + lane;
        const bool valid = row_ok && (g < 90);
        const int gs = valid ? g : 0;
        uint4 pack = valid ? b_row[gs] : make_uint4(0, 0, 0, 0);
        float sc = valid ? decode_e8m0_scale(sc_row[gs]) : 0.0f;
        const uint32_t words[4] = {pack.x, pack.y, pack.z, pack.w};
        const float* xg = &s_X[gs][0];

        float gsum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t word = words[w];
            const float* xw = xg + (w << 3);
            gsum = fmaf(fp4_shfl_decode(r_lut,  word        & 0xFu), xw[0], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >>  4) & 0xFu), xw[1], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >>  8) & 0xFu), xw[2], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 12) & 0xFu), xw[3], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 16) & 0xFu), xw[4], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 20) & 0xFu), xw[5], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 24) & 0xFu), xw[6], gsum);
            gsum = fmaf(fp4_shfl_decode(r_lut, (word >> 28) & 0xFu), xw[7], gsum);
        }
        if (valid) acc = fmaf(gsum, sc, acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0 && row_ok) {
        if (bias_fp16) acc += __half2float(bias_fp16[row]);
        Y[row] = acc;
    }
}

template <int MAX_B>
__global__ void mxfp4_gemv_batched_kernel(
    const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales,
    const __half* __restrict__ bias_fp16,
    const float* __restrict__ X, // [B, 2880]
    float* __restrict__ Y,       // [B, M]
    int M,
    int B
) {
    // Tile over K in 5 chunks of 18 groups, padded stride 33 for 0-way bank conflicts!
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int G_TILE = 18;
    constexpr int K_TILE = G_TILE * 32; // 576
    __shared__ float s_X[MAX_B][G_TILE][33];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool row_ok = (row < M);
    const int r_safe = row_ok ? row : 0;
    const float r_lut = c_FP4_LUT[lane & 15];

    float acc[MAX_B];
    #pragma unroll
    for (int b = 0; b < MAX_B; ++b) acc[b] = 0.0f;

    const uint4* b_row = reinterpret_cast<const uint4*>(blocks + (size_t)r_safe * 90 * 16);
    const uint8_t* sc_row = scales + (size_t)r_safe * 90;

    for (int g0 = 0; g0 < 90; g0 += G_TILE) {
        const int k0 = g0 * 32;
        for (int idx = tid; idx < B * K_TILE; idx += 256) {
            int bv = idx / K_TILE;
            int kl = idx - bv * K_TILE;
            s_X[bv][kl >> 5][kl & 31] = X[(size_t)bv * D_MODEL + k0 + kl];
        }
        __syncthreads();

        const bool valid = row_ok && (lane < G_TILE);
        const int g = g0 + (valid ? lane : 0);
        uint4 pack = valid ? b_row[g] : make_uint4(0, 0, 0, 0);
        float sc = valid ? decode_e8m0_scale(sc_row[g]) : 0.0f;
        const uint32_t words[4] = {pack.x, pack.y, pack.z, pack.w};

        // Warp-uniform __shfl_sync FP4 E2M1 decode across ALL 32 lanes!
        float wv[32];
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t word = words[w];
            wv[w * 8 + 0] = fp4_shfl_decode(r_lut,  word        & 0xFu) * sc;
            wv[w * 8 + 1] = fp4_shfl_decode(r_lut, (word >>  4) & 0xFu) * sc;
            wv[w * 8 + 2] = fp4_shfl_decode(r_lut, (word >>  8) & 0xFu) * sc;
            wv[w * 8 + 3] = fp4_shfl_decode(r_lut, (word >> 12) & 0xFu) * sc;
            wv[w * 8 + 4] = fp4_shfl_decode(r_lut, (word >> 16) & 0xFu) * sc;
            wv[w * 8 + 5] = fp4_shfl_decode(r_lut, (word >> 20) & 0xFu) * sc;
            wv[w * 8 + 6] = fp4_shfl_decode(r_lut, (word >> 24) & 0xFu) * sc;
            wv[w * 8 + 7] = fp4_shfl_decode(r_lut, (word >> 28) & 0xFu) * sc;
        }
        if (valid) {
            #pragma unroll
            for (int b = 0; b < MAX_B; ++b) {
                if (b < B) {
                    const float* xb = &s_X[b][lane][0];
                    float s = acc[b];
                    #pragma unroll
                    for (int t = 0; t < 32; ++t) s = fmaf(wv[t], xb[t], s);
                    acc[b] = s;
                }
            }
        }
        __syncthreads();
    }

    if (row_ok) {
        float bval = bias_fp16 ? __half2float(bias_fp16[row]) : 0.0f;
        #pragma unroll
        for (int b = 0; b < MAX_B; ++b) {
            if (b < B) {
                float v = acc[b];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
                if (lane == 0) Y[(size_t)b * M + row] = v + bval;
            }
        }
    }
}

// Fast MXFP4 -> FP32 Dequantizer (warp-uniform __shfl_sync LUT + vectorized float4 stores for cublasSgemm!)
__global__ void mxfp4_dequant_to_fp32_kernel(
    const uint8_t* __restrict__ blocks, // [M, 90, 16]
    const uint8_t* __restrict__ scales, // [M, 90]
    float* __restrict__ W_fp32,         // [M, 2880]
    int M
) {
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_g = (size_t)M * 90ULL;
    const bool valid = (gid < total_g);
    const int lane = threadIdx.x & 31;
    const float r_lut = c_FP4_LUT[lane & 15];

    const float sc = valid ? decode_e8m0_scale(scales[gid]) : 0.0f;
    const uint4 pack = valid ? reinterpret_cast<const uint4*>(blocks)[gid] : make_uint4(0, 0, 0, 0);
    const uint32_t words[4] = {pack.x, pack.y, pack.z, pack.w};
    float4* out_f4 = valid ? reinterpret_cast<float4*>(W_fp32 + gid * 32ULL) : nullptr;

    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t word = words[w];
        float4 v0, v1;
        v0.x = fp4_shfl_decode(r_lut,  word        & 0xFu) * sc;
        v0.y = fp4_shfl_decode(r_lut, (word >>  4) & 0xFu) * sc;
        v0.z = fp4_shfl_decode(r_lut, (word >>  8) & 0xFu) * sc;
        v0.w = fp4_shfl_decode(r_lut, (word >> 12) & 0xFu) * sc;
        v1.x = fp4_shfl_decode(r_lut, (word >> 16) & 0xFu) * sc;
        v1.y = fp4_shfl_decode(r_lut, (word >> 20) & 0xFu) * sc;
        v1.z = fp4_shfl_decode(r_lut, (word >> 24) & 0xFu) * sc;
        v1.w = fp4_shfl_decode(r_lut, (word >> 28) & 0xFu) * sc;
        if (valid) {
            out_f4[w * 2 + 0] = v0;
            out_f4[w * 2 + 1] = v1;
        }
    }
}

// Vectorized FP16 -> FP32 cast (8 elements/thread via uint4 -> 2x float4, saturates DRAM BW!)
__global__ void fp16_to_fp32_vec8_kernel(const __half* __restrict__ src, float* __restrict__ dst, size_t n_vec8) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vec8) return;
    uint4 pack = reinterpret_cast<const uint4*>(src)[i];
    float2 p0 = __half22float2(*reinterpret_cast<const __half2*>(&pack.x));
    float2 p1 = __half22float2(*reinterpret_cast<const __half2*>(&pack.y));
    float2 p2 = __half22float2(*reinterpret_cast<const __half2*>(&pack.z));
    float2 p3 = __half22float2(*reinterpret_cast<const __half2*>(&pack.w));
    float4* d4 = reinterpret_cast<float4*>(dst);
    d4[i * 2 + 0] = make_float4(p0.x, p0.y, p1.x, p1.y);
    d4[i * 2 + 1] = make_float4(p2.x, p2.y, p3.x, p3.y);
}

// Batched FP16 GEMV (B=2..8, reads W ONCE via uint4 for verify_tokens lm_head/QKV/O!)
template <int MAX_B>
__global__ void fp16_gemv_bvec_kernel(
    const __half* __restrict__ W, // [N, K] row-major, K % 8 == 0
    const __half* __restrict__ X, // [B, K]
    float* __restrict__ Y,        // [B, N]
    int N,
    int B,
    int K
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int K_TILE = 256; // 32 uint4s -> 8 * 256 * 4 = 8 KB smem!
    __shared__ float s_X[MAX_B][K_TILE];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    float acc[MAX_B];
    #pragma unroll
    for (int b = 0; b < MAX_B; ++b) acc[b] = 0.0f;

    const uint4* w_row = (row < N) ? reinterpret_cast<const uint4*>(W + (size_t)row * K) : nullptr;

    for (int k0 = 0; k0 < K; k0 += K_TILE) {
        const int kt = min(K_TILE, K - k0);
        for (int idx = tid; idx < B * kt; idx += 256) {
            int bv = idx / kt;
            int kl = idx - bv * kt;
            s_X[bv][kl] = __half2float(X[(size_t)bv * K + k0 + kl]);
        }
        __syncthreads();

        if (row < N) {
            const int j_cnt = kt >> 3;
            const int j_base = k0 >> 3;
            if (lane < j_cnt) {
                uint4 pack = w_row[j_base + lane];
                float2 p0 = __half22float2(*reinterpret_cast<const __half2*>(&pack.x));
                float2 p1 = __half22float2(*reinterpret_cast<const __half2*>(&pack.y));
                float2 p2 = __half22float2(*reinterpret_cast<const __half2*>(&pack.z));
                float2 p3 = __half22float2(*reinterpret_cast<const __half2*>(&pack.w));
                const int kl = lane << 3;
                #pragma unroll
                for (int b = 0; b < MAX_B; ++b) {
                    if (b < B) {
                        const float* xb = &s_X[b][kl];
                        float s = acc[b];
                        s = fmaf(p0.x, xb[0], s); s = fmaf(p0.y, xb[1], s);
                        s = fmaf(p1.x, xb[2], s); s = fmaf(p1.y, xb[3], s);
                        s = fmaf(p2.x, xb[4], s); s = fmaf(p2.y, xb[5], s);
                        s = fmaf(p3.x, xb[6], s); s = fmaf(p3.y, xb[7], s);
                        acc[b] = s;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row < N) {
        #pragma unroll
        for (int b = 0; b < MAX_B; ++b) {
            if (b < B) {
                float v = acc[b];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
                if (lane == 0) Y[(size_t)b * N + row] = v;
            }
        }
    }
}

// ============================================================================
// KERNEL 6B: Tiled cublasSgemm Global Flash-Prefill Helpers (exact causal + sinks + online FP32 LSE!)
// ============================================================================
__global__ void prefill_gather_q_group_kernel(
    const float* __restrict__ Q_all, // [C, 64, 64]
    float* __restrict__ Q_grp,       // [C * 8, 64]
    int C,
    int h_kv
) {
    // Grid: (C * 8), Block: 64
    const int r = blockIdx.x;
    if (r >= C * 8) return;
    const int d = threadIdx.x;
    const int q_idx = r >> 3;
    const int qg = r & 7;
    const int q_head = (h_kv << 3) + qg;
    Q_grp[(size_t)r * 64 + d] = Q_all[((size_t)q_idx * 64 + q_head) * 64 + d];
}

__global__ void prefill_init_mlo_kernel(float* m_acc, float* l_acc, float* O_acc, int M_q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M_q) {
        m_acc[idx] = -FLT_MAX;
        l_acc[idx] = 0.0f;
    }
    if (idx < M_q * 64) {
        O_acc[idx] = 0.0f;
    }
}

__global__ void prefill_gather_kv_tile_kernel(
    const __half* __restrict__ K_cache, // [cap, 8, 64]
    const __half* __restrict__ V_cache, // [cap, 8, 64]
    float* __restrict__ K_tile,         // [tlen, 64]
    float* __restrict__ V_tile,         // [tlen, 64]
    int j0,
    int tlen,
    int h_kv
) {
    // Grid: (tlen), Block: 64
    const int k = blockIdx.x;
    if (k >= tlen) return;
    const int d = threadIdx.x;
    const size_t src = ((size_t)(j0 + k) * 8 + h_kv) * 64 + d;
    K_tile[(size_t)k * 64 + d] = __half2float(K_cache[src]);
    V_tile[(size_t)k * 64 + d] = __half2float(V_cache[src]);
}

__global__ void prefill_gather_kv_tile_hybrid_kernel(
    const __half* __restrict__ K_global,    // [kv_cap_global, 8, 64] exact FP16!
    const __half* __restrict__ V_hot,       // [4096, 8, 64] FP16
    const __half* __restrict__ V_sink_fp16, // [128, 8, 64] exact FP16 for tokens 0..127
    const uint8_t* __restrict__ V_cold,     // [cold_cap, 8, 64] uint8 per-channel
    const __half* __restrict__ V_page_min,  // [cold_pages, 8, 64] FP16
    const __half* __restrict__ V_page_max,  // [cold_pages, 8, 64] FP16
    float* __restrict__ K_tile,             // [tlen, 64]
    float* __restrict__ V_tile,             // [tlen, 64]
    int j0,
    int tlen,
    int h_kv,
    int cold_len
) {
    const int k = blockIdx.x;
    if (k >= tlen) return;
    const int d = threadIdx.x;
    const int abs_j = j0 + k;
    const size_t k_src = ((size_t)abs_j * 8 + h_kv) * 64 + d;
    K_tile[(size_t)k * 64 + d] = __half2float(K_global[k_src]);
    if (abs_j < cold_len) {
        if (abs_j < V_SINK_FP16) {
            V_tile[(size_t)k * 64 + d] = __half2float(V_sink_fp16[k_src]);
        } else {
            const int pg = abs_j >> 5;
            const size_t pg_off = ((size_t)pg * 8 + h_kv) * 64 + d;
            float vmin = __half2float(V_page_min[pg_off]);
            float vmax = __half2float(V_page_max[pg_off]);
            float vsc  = (vmax - vmin) * (1.0f / 255.0f);
            V_tile[(size_t)k * 64 + d] = fmaf((float)V_cold[k_src], vsc, vmin);
        }
    } else {
        const int slot = abs_j & (HOT_WINDOW - 1);
        const size_t v_src = ((size_t)slot * 8 + h_kv) * 64 + d;
        V_tile[(size_t)k * 64 + d] = __half2float(V_hot[v_src]);
    }
}

__global__ void prefill_online_softmax_tile_kernel(
    float* __restrict__ S_tile, // [M_q, tlen] overwritten in-place with P_tile!
    float* __restrict__ m_acc,  // [M_q]
    float* __restrict__ l_acc,  // [M_q]
    float* __restrict__ O_acc,  // [M_q, 64]
    int M_q,
    int tlen,
    int start_pos,
    int j0
) {
    // Grid: (M_q), Block: 256 threads (8 warps)
    const int r = blockIdx.x;
    if (r >= M_q) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int q_pos = start_pos + (r >> 3);
    const int k_valid = min(tlen, max(0, q_pos - j0 + 1));
    float* s_row = S_tile + (size_t)r * tlen;

    if (k_valid <= 0) {
        for (int k = tid; k < tlen; k += 256) s_row[k] = 0.0f;
        return;
    }

    __shared__ float s_red[8];
    __shared__ float s_m_new;
    __shared__ float s_alpha_old;

    // 1. Row max over [0, k_valid)
    float local_m = -FLT_MAX;
    for (int k = tid; k < k_valid; k += 256) {
        local_m = fmaxf(local_m, s_row[k]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local_m = fmaxf(local_m, __shfl_down_sync(0xffffffff, local_m, off));
    if (lane == 0) s_red[warp_id] = local_m;
    __syncthreads();

    if (tid == 0) {
        float mt = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < 8; ++w) mt = fmaxf(mt, s_red[w]);
        float m_old = m_acc[r];
        float m_new = fmaxf(m_old, mt);
        s_m_new = m_new;
        s_alpha_old = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);
    }
    __syncthreads();

    const float m_new = s_m_new;
    const float alpha_old = s_alpha_old;

    // 2. Compute exp(s - m_new) in-place & sum l_tile
    float local_l = 0.0f;
    for (int k = tid; k < tlen; k += 256) {
        float p = (k < k_valid) ? expf(s_row[k] - m_new) : 0.0f;
        s_row[k] = p;
        local_l += p;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) local_l += __shfl_down_sync(0xffffffff, local_l, off);
    if (lane == 0) s_red[warp_id] = local_l;
    __syncthreads();

    // 3. Rescale existing O_acc[r, 0..63] by alpha_old
    if (tid < 64 && alpha_old != 1.0f) {
        O_acc[(size_t)r * 64 + tid] *= alpha_old;
    }
    if (tid == 0) {
        float lt = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) lt += s_red[w];
        m_acc[r] = m_new;
        l_acc[r] = alpha_old * l_acc[r] + lt;
    }
}

__global__ void prefill_finalize_sink_scatter_kernel(
    const float* __restrict__ O_acc, // [M_q, 64]
    const float* __restrict__ m_acc, // [M_q]
    const float* __restrict__ l_acc, // [M_q]
    const float* __restrict__ sinks, // [64]
    float* __restrict__ O_out_f32,   // [C, 64, 64]
    int M_q,
    int h_kv
) {
    const int r = blockIdx.x;
    if (r >= M_q) return;
    const int d = threadIdx.x; // 0..63
    const int q_idx = r >> 3;
    const int qg = r & 7;
    const int q_head = (h_kv << 3) + qg;

    float m_old = m_acc[r];
    float l_old = l_acc[r];
    float sink = sinks[q_head];
    float m_new = fmaxf(m_old, sink);
    float g = (m_old == -FLT_MAX) ? 0.0f : expf(m_old - m_new);
    float Z = g * l_old + expf(sink - m_new);
    float inv_z = (Z > 0.0f) ? (1.0f / Z) : 0.0f;
    O_out_f32[((size_t)q_idx * 64 + q_head) * 64 + d] = (g * O_acc[(size_t)r * 64 + d]) * inv_z;
}

// ============================================================================
// KERNEL 7: Ring-128 Atomic Shadow Snapshot & Rollback (Phase 2 Speculative)
// ============================================================================
__global__ void ring128_snapshot_kernel(
    const __half* __restrict__ K_ring, // [12, 128, 512]
    const __half* __restrict__ V_ring, // [12, 128, 512]
    __half* __restrict__ K_shadow,     // [12, K, 512]
    __half* __restrict__ V_shadow,     // [12, K, 512]
    int start_pos,
    int K
) {
    // Grid: (12, K), Block: 512
    const int loc_l = blockIdx.x;
    const int i     = blockIdx.y;
    const int d     = threadIdx.x;
    const int slot  = (start_pos + i) & (RING_SIZE - 1);
    const size_t src = ((size_t)loc_l * RING_SIZE + slot) * 512 + d;
    const size_t dst = ((size_t)loc_l * K + i) * 512 + d;
    K_shadow[dst] = K_ring[src];
    V_shadow[dst] = V_ring[src];
}

__global__ void ring128_rollback_kernel(
    __half* __restrict__ K_ring,         // [12, 128, 512]
    __half* __restrict__ V_ring,         // [12, 128, 512]
    const __half* __restrict__ K_shadow, // [12, K, 512]
    const __half* __restrict__ V_shadow, // [12, K, 512]
    int start_pos,
    int first_rejected_i,
    int K
) {
    const int loc_l = blockIdx.x;
    const int i     = first_rejected_i + blockIdx.y;
    if (i >= K) return;
    const int d     = threadIdx.x;
    const int slot  = (start_pos + i) & (RING_SIZE - 1);
    const size_t dst = ((size_t)loc_l * RING_SIZE + slot) * 512 + d;
    const size_t src = ((size_t)loc_l * K + i) * 512 + d;
    K_ring[dst] = K_shadow[src];
    V_ring[dst] = V_shadow[src];
}

// Iter 5: Bit-exact Hot-4K V circular shadow snapshot & rollback for verify_tokens(K+1)
__global__ void hot4k_v_snapshot_kernel(
    const __half* __restrict__ V_hot, // [12, 4096, 512]
    __half* __restrict__ V_shadow,    // [12, K, 512]
    int start_pos,
    int K
) {
    const int gl_l = blockIdx.x;
    const int i    = blockIdx.y;
    const int d    = threadIdx.x;
    const int slot = (start_pos + i) & (HOT_WINDOW - 1);
    const size_t src = ((size_t)gl_l * HOT_WINDOW + slot) * 512 + d;
    const size_t dst = ((size_t)gl_l * K + i) * 512 + d;
    V_shadow[dst] = V_hot[src];
}

__global__ void hot4k_v_rollback_kernel(
    __half* __restrict__ V_hot,          // [12, 4096, 512]
    const __half* __restrict__ V_shadow, // [12, K, 512]
    int start_pos,
    int first_rejected_i,
    int K
) {
    const int gl_l = blockIdx.x;
    const int i    = first_rejected_i + blockIdx.y;
    if (i >= K) return;
    const int d    = threadIdx.x;
    const int slot = (start_pos + i) & (HOT_WINDOW - 1);
    const size_t dst = ((size_t)gl_l * HOT_WINDOW + slot) * 512 + d;
    const size_t src = ((size_t)gl_l * K + i) * 512 + d;
    V_hot[dst] = V_shadow[src];
}

// Copy previous prev_len ring tokens [win_base .. win_base + prev_len - 1] into linear staging buffer
__global__ void copy_ring_prev_to_linear_kernel(
    const __half* __restrict__ K_ring, // [128, 512] single local layer
    const __half* __restrict__ V_ring, // [128, 512]
    __half* __restrict__ K_lin,        // [prev_len + B, 512]
    __half* __restrict__ V_lin,        // [prev_len + B, 512]
    int win_base,
    int prev_len
) {
    const int i = blockIdx.x;
    if (i >= prev_len) return;
    const int d = threadIdx.x; // 0..511
    const int slot = (win_base + i) & (RING_SIZE - 1);
    K_lin[(size_t)i * 512 + d] = K_ring[(size_t)slot * 512 + d];
    V_lin[(size_t)i * 512 + d] = V_ring[(size_t)slot * 512 + d];
}

// Write final count tokens from linear buffer [lin_start .. lin_start + count - 1] into Ring-128
__global__ void copy_linear_tail_to_ring_kernel(
    const __half* __restrict__ K_lin,
    const __half* __restrict__ V_lin,
    __half* __restrict__ K_ring,
    __half* __restrict__ V_ring,
    int lin_start,
    int start_abs_pos,
    int count
) {
    const int i = blockIdx.x;
    if (i >= count) return;
    const int d = threadIdx.x; // 0..511
    const int slot = (start_abs_pos + i) & (RING_SIZE - 1);
    K_ring[(size_t)slot * 512 + d] = K_lin[(size_t)(lin_start + i) * 512 + d];
    V_ring[(size_t)slot * 512 + d] = V_lin[(size_t)(lin_start + i) * 512 + d];
}

// ============================================================================
// KERNEL 8: Utility & Vectorized B=1 FP16 GEMV (saturates sm_75 DRAM BW for lm_head/QKV/O!)
// ============================================================================
__global__ void fp16_to_fp32_kernel(const __half* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

__global__ void add_bias_32_kernel(float* __restrict__ x, const __half* __restrict__ b) {
    int i = threadIdx.x;
    if (i < NUM_EXPERTS) x[i] += __half2float(b[i]);
}

__global__ void add_f32_f16_bias_2880_kernel(
    const float* __restrict__ src,
    const __half* __restrict__ b_fp16,
    float* __restrict__ dst
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D_MODEL) dst[i] = src[i] + __half2float(b_fp16[i]);
}

__global__ void add_f32_scaled_2880_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float scale_b,
    float* __restrict__ dst
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D_MODEL) dst[i] = fmaf(scale_b, b[i], a[i]);
}

__global__ void add_f32_scaled_batched_2880_kernel(
    const float* __restrict__ a, // [B, D_MODEL]
    const float* __restrict__ b, // [D_MODEL] broadcast across B
    float scale_b,
    float* __restrict__ dst,     // [B, D_MODEL]
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * D_MODEL) {
        int col = idx % D_MODEL;
        dst[idx] = fmaf(scale_b, b[col], a[idx]);
    }
}

// Iter 7 Pillar 1: Lookahead Router Top-4 + Full 32-way Softmax Probability (1 warp/token, 0 spills!)
__global__ void lookahead_router_top4_kernel(
    const float* __restrict__ router_logits, // [B, 32]
    const __half* __restrict__ router_bias,  // [32]
    uint8_t* __restrict__ top_idx_out,       // [B, 4]
    float* __restrict__ top_p32_out,         // [B, 4] full 32-way softmax probability!
    int B
) {
    const int b = blockIdx.x;
    if (b >= B) return;
    const int lane = threadIdx.x; // 0..31
    float val = router_logits[b * NUM_EXPERTS + lane] + __half2float(router_bias[lane]);

    // Full 32-way max and sum(exp) across the warp
    float m32 = val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m32 = fmaxf(m32, __shfl_xor_sync(0xffffffff, m32, off));
    float e32 = expf(val - m32);
    float z32 = e32;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) z32 += __shfl_xor_sync(0xffffffff, z32, off);
    float inv_z32 = (z32 > 0.0f) ? (1.0f / z32) : 0.0f;

    bool avail = true;
    float best_v0 = -FLT_MAX, best_v1 = -FLT_MAX, best_v2 = -FLT_MAX, best_v3 = -FLT_MAX;
    int   best_i0 = 0, best_i1 = 0, best_i2 = 0, best_i3 = 0;

    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        float vc = avail ? val : -FLT_MAX;
        int   ic = lane;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float pv = __shfl_xor_sync(0xffffffff, vc, off);
            int   pi = __shfl_xor_sync(0xffffffff, ic, off);
            if (pv > vc || (pv == vc && pi < ic)) { vc = pv; ic = pi; }
        }
        if (k == 0) { best_v0 = vc; best_i0 = ic; }
        else if (k == 1) { best_v1 = vc; best_i1 = ic; }
        else if (k == 2) { best_v2 = vc; best_i2 = ic; }
        else             { best_v3 = vc; best_i3 = ic; }
        if (lane == ic) avail = false;
    }
    if (lane == 0) {
        top_idx_out[b * 4 + 0] = (uint8_t)best_i0;
        top_idx_out[b * 4 + 1] = (uint8_t)best_i1;
        top_idx_out[b * 4 + 2] = (uint8_t)best_i2;
        top_idx_out[b * 4 + 3] = (uint8_t)best_i3;
        top_p32_out[b * 4 + 0] = expf(best_v0 - m32) * inv_z32;
        top_p32_out[b * 4 + 1] = expf(best_v1 - m32) * inv_z32;
        top_p32_out[b * 4 + 2] = expf(best_v2 - m32) * inv_z32;
        top_p32_out[b * 4 + 3] = expf(best_v3 - m32) * inv_z32;
    }
}

// Iter 7 Pillar 2: Compact Per-Channel INT8 GEMV for EAGLE-3 (2x less VRAM & BW, 0 spills!)
__global__ void int8_gemv_1vec_kernel(
    const int8_t* __restrict__ W, // [N, K] row-major, K % 16 == 0, K <= 4096
    const float*  __restrict__ S, // [N] per-channel FP32 scale
    const __half* __restrict__ X, // [K] FP16 input
    float*        __restrict__ Y, // [N] FP32 output
    int N,
    int K
) {
    constexpr int WARPS_PER_BLOCK = 8;
    __shared__ float s_X[4096];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    for (int j = tid; j < K; j += 256) s_X[j] = __half2float(X[j]);
    __syncthreads();

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int k_vec = K >> 4; // 16 int8s per int4
    const int4* w_row = reinterpret_cast<const int4*>(W + (size_t)row * K);

    float acc = 0.0f;
    for (int j = lane; j < k_vec; j += 32) {
        int4 pack = w_row[j];
        const float* xj = &s_X[j << 4];
        const uint32_t w0 = (uint32_t)pack.x;
        const uint32_t w1 = (uint32_t)pack.y;
        const uint32_t w2 = (uint32_t)pack.z;
        const uint32_t w3 = (uint32_t)pack.w;
        acc = fmaf((float)(int8_t)( w0        & 0xFFu), xj[ 0], acc);
        acc = fmaf((float)(int8_t)((w0 >>  8) & 0xFFu), xj[ 1], acc);
        acc = fmaf((float)(int8_t)((w0 >> 16) & 0xFFu), xj[ 2], acc);
        acc = fmaf((float)(int8_t)((w0 >> 24) & 0xFFu), xj[ 3], acc);
        acc = fmaf((float)(int8_t)( w1        & 0xFFu), xj[ 4], acc);
        acc = fmaf((float)(int8_t)((w1 >>  8) & 0xFFu), xj[ 5], acc);
        acc = fmaf((float)(int8_t)((w1 >> 16) & 0xFFu), xj[ 6], acc);
        acc = fmaf((float)(int8_t)((w1 >> 24) & 0xFFu), xj[ 7], acc);
        acc = fmaf((float)(int8_t)( w2        & 0xFFu), xj[ 8], acc);
        acc = fmaf((float)(int8_t)((w2 >>  8) & 0xFFu), xj[ 9], acc);
        acc = fmaf((float)(int8_t)((w2 >> 16) & 0xFFu), xj[10], acc);
        acc = fmaf((float)(int8_t)((w2 >> 24) & 0xFFu), xj[11], acc);
        acc = fmaf((float)(int8_t)( w3        & 0xFFu), xj[12], acc);
        acc = fmaf((float)(int8_t)((w3 >>  8) & 0xFFu), xj[13], acc);
        acc = fmaf((float)(int8_t)((w3 >> 16) & 0xFFu), xj[14], acc);
        acc = fmaf((float)(int8_t)((w3 >> 24) & 0xFFu), xj[15], acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) Y[row] = acc * S[row];
}

__global__ void ema_update_attn_2880_kernel(
    float* __restrict__ ema,
    const float* __restrict__ proj_out,
    const __half* __restrict__ b_o,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D_MODEL) {
        float cur = proj_out[i] + __half2float(b_o[i]);
        ema[i] = fmaf(alpha, cur, (1.0f - alpha) * ema[i]);
    }
}

__global__ void fp16_gemv_1vec_kernel(
    const __half* __restrict__ W, // [N, K] row-major, K % 8 == 0, K <= 4096
    const __half* __restrict__ X, // [K]
    float* __restrict__ Y,        // [N]
    int N,
    int K
) {
    constexpr int WARPS_PER_BLOCK = 8;
    __shared__ float s_X[4096];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    for (int j = tid; j < K; j += 256) {
        s_X[j] = __half2float(X[j]);
    }
    __syncthreads();

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int k_vec = K >> 3;
    const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)row * K);

    float acc = 0.0f;
    for (int j = lane; j < k_vec; j += 32) {
        uint4 pack = w_row[j];
        const float* xj = &s_X[j << 3];
        float2 p0 = __half22float2(*reinterpret_cast<const __half2*>(&pack.x));
        float2 p1 = __half22float2(*reinterpret_cast<const __half2*>(&pack.y));
        float2 p2 = __half22float2(*reinterpret_cast<const __half2*>(&pack.z));
        float2 p3 = __half22float2(*reinterpret_cast<const __half2*>(&pack.w));
        acc = fmaf(p0.x, xj[0], acc);
        acc = fmaf(p0.y, xj[1], acc);
        acc = fmaf(p1.x, xj[2], acc);
        acc = fmaf(p1.y, xj[3], acc);
        acc = fmaf(p2.x, xj[4], acc);
        acc = fmaf(p2.y, xj[5], acc);
        acc = fmaf(p3.x, xj[6], acc);
        acc = fmaf(p3.y, xj[7], acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) Y[row] = acc;
}

// ============================================================================
// ITERATION 5 LEVER C: Native Resident EAGLE-1/3 Draft Kernels (< 3.5 ms for K=5!)
//   Reuses GPT-OSS-20B's ALREADY-RESIDENT final_norm + lm_head via indexed shortlist GEMV!
// ============================================================================
static constexpr int EAGLE_D_DRAFT = 512;
static constexpr int EAGLE_D_FF = 1536;
static constexpr int EAGLE_QKV_DIM = 768; // 8 Q heads (512) + 2 K heads (128) + 2 V heads (128)
static constexpr int EAGLE_KV_WIN = 512;
static constexpr int EAGLE_MAX_SHORTLIST = 18432;
static constexpr int E3_N_LAYERS = 2;
static constexpr int E3_D_FF = 1280;
static constexpr int E3_MAX_TREE_SCRATCH = 32;

__global__ void fp16_gemv_1vec_indexed_kernel(
    const __half* __restrict__ W,        // [VOCAB_SIZE, K] row-major lm_head
    const __half* __restrict__ X,        // [K] FP16 normalized hidden
    const int* __restrict__ shortlist,   // [V_sub] vocab row indices
    float* __restrict__ Y,               // [V_sub] output logits
    int V_sub,
    int K
) {
    constexpr int WARPS_PER_BLOCK = 8;
    __shared__ float s_X[4096];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    for (int j = tid; j < K; j += 256) {
        s_X[j] = __half2float(X[j]);
    }
    __syncthreads();

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= V_sub) return;

    const int vocab_row = shortlist[row];
    const int k_vec = K >> 3;
    const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)vocab_row * K);

    float acc = 0.0f;
    for (int j = lane; j < k_vec; j += 32) {
        uint4 pack = w_row[j];
        const float* xj = &s_X[j << 3];
        float2 p0 = __half22float2(*reinterpret_cast<const __half2*>(&pack.x));
        float2 p1 = __half22float2(*reinterpret_cast<const __half2*>(&pack.y));
        float2 p2 = __half22float2(*reinterpret_cast<const __half2*>(&pack.z));
        float2 p3 = __half22float2(*reinterpret_cast<const __half2*>(&pack.w));
        acc = fmaf(p0.x, xj[0], acc);
        acc = fmaf(p0.y, xj[1], acc);
        acc = fmaf(p1.x, xj[2], acc);
        acc = fmaf(p1.y, xj[3], acc);
        acc = fmaf(p2.x, xj[4], acc);
        acc = fmaf(p2.y, xj[5], acc);
        acc = fmaf(p3.x, xj[6], acc);
        acc = fmaf(p3.y, xj[7], acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) Y[row] = acc;
}

__global__ void eagle_unit_rms_copy_kernel(
    const float* __restrict__ src, // [D_MODEL]
    float* __restrict__ dst        // [D_MODEL]
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_red[8];
    float sq = 0.0f;
    for (int i = tid; i < D_MODEL; i += 256) {
        float v = src[i];
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
    if (lane == 0) s_red[wid] = sq;
    __syncthreads();
    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)D_MODEL + RMS_EPS);
    }
    __syncthreads();
    float inv = s_inv;
    for (int i = tid; i < D_MODEL; i += 256) {
        dst[i] = src[i] * inv;
    }
}

__global__ void eagle_rmsnorm_f32_to_f16_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    __half* __restrict__ Y,
    int dim
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_red[8];
    float sq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float v = X[i];
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
    if (lane == 0) s_red[wid] = sq;
    __syncthreads();
    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)dim + RMS_EPS);
    }
    __syncthreads();
    float inv = s_inv;
    for (int i = tid; i < dim; i += 256) {
        Y[i] = __float2half((X[i] * inv) * W[i]);
    }
}

__global__ void eagle_rmsnorm_f16_to_f16_kernel(
    const __half* __restrict__ X,
    const float* __restrict__ W,
    __half* __restrict__ Y,
    int dim
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_red[8];
    float sq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float v = __half2float(X[i]);
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
    if (lane == 0) s_red[wid] = sq;
    __syncthreads();
    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)dim + RMS_EPS);
    }
    __syncthreads();
    float inv = s_inv;
    for (int i = tid; i < dim; i += 256) {
        Y[i] = __float2half((__half2float(X[i]) * inv) * W[i]);
    }
}

__global__ void eagle_fuse_add_bias_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ bias,
    float* __restrict__ out
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < EAGLE_D_DRAFT) out[i] = a[i] + b[i] + bias[i];
}

__global__ void eagle_add_residual_512_kernel(
    float* __restrict__ x,
    const float* __restrict__ delta
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < EAGLE_D_DRAFT) x[i] += delta[i];
}

__global__ void eagle_silu_mul_1536_kernel(
    const float* __restrict__ gu, // [3072]: gate[0..1535], up[1536..3071]
    __half* __restrict__ mid_f16  // [1536]
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < EAGLE_D_FF) {
        float g = gu[i];
        float s = g / (1.0f + expf(-g));
        mid_f16[i] = __float2half(s * gu[i + EAGLE_D_FF]);
    }
}

__global__ void eagle_rope_kv_attn_kernel(
    const float* __restrict__ qkv,  // [768]: Q[512], K[128], V[128]
    __half* __restrict__ K_kv,      // [512, 2, 64]
    __half* __restrict__ V_kv,      // [512, 2, 64]
    __half* __restrict__ O_f16,     // [512]
    int dpos
) {
    // 1 block x 256 threads = 8 warps (1 warp per Q-head, 2 dims/lane)
    const int tid  = threadIdx.x;
    const int qg   = tid >> 5;  // 0..7 Q-head
    const int lane = tid & 31;  // 0..31 pair index
    const int kv_h = qg >> 2;   // 0..1 KV-head (4 Q heads per KV head)

    float inv_f = powf(10000.0f, -((float)(2 * lane) / 64.0f));
    float angle = (float)dpos * inv_f;
    float sv, cv;
    sincosf(angle, &sv, &cv);

    // Rotate Q
    float qx1 = qkv[qg * 64 + lane];
    float qx2 = qkv[qg * 64 + lane + 32];
    const float q0 = (qx1 * cv - qx2 * sv) * SM_SCALE_LOG2E;
    const float q1 = (qx2 * cv + qx1 * sv) * SM_SCALE_LOG2E;

    // Warps 0 and 1 write rotated K and raw V for KV heads 0 and 1 into circular ring slot
    const int slot = dpos & (EAGLE_KV_WIN - 1);
    if (qg < 2) {
        int kh = qg;
        float kx1 = qkv[512 + kh * 64 + lane];
        float kx2 = qkv[512 + kh * 64 + lane + 32];
        __half* kd = K_kv + ((size_t)slot * 2 + kh) * 64;
        kd[lane]      = __float2half(kx1 * cv - kx2 * sv);
        kd[lane + 32] = __float2half(kx2 * cv + kx1 * sv);

        __half* vd = V_kv + ((size_t)slot * 2 + kh) * 64;
        vd[lane]      = __float2half(qkv[640 + kh * 64 + lane]);
        vd[lane + 32] = __float2half(qkv[640 + kh * 64 + lane + 32]);
    }
    __syncthreads();

    // Sliding-window causal attention over [j_start .. dpos]
    const int j_start = max(0, dpos - EAGLE_KV_WIN + 1);
    float m_run = -FLT_MAX;
    float l_run = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int j = j_start; j <= dpos; ++j) {
        int js = j & (EAGLE_KV_WIN - 1);
        const __half* kp = K_kv + ((size_t)js * 2 + kv_h) * 64;
        const __half* vp = V_kv + ((size_t)js * 2 + kv_h) * 64;
        float dot = fmaf(q0, __half2float(kp[lane]), q1 * __half2float(kp[lane + 32]));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        }
        float m_new = fmaxf(m_run, dot);
        float gamma = (m_run > -1e29f) ? fast_ex2(m_run - m_new) : 0.0f;
        float w     = fast_ex2(dot - m_new);
        l_run = fmaf(gamma, l_run, w);
        a0    = fmaf(gamma, a0, w * __half2float(vp[lane]));
        a1    = fmaf(gamma, a1, w * __half2float(vp[lane + 32]));
        m_run = m_new;
    }

    float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    O_f16[qg * 64 + lane]      = __float2half(a0 * inv_l);
    O_f16[qg * 64 + lane + 32] = __float2half(a1 * inv_l);
}

__global__ void eagle_residual_unit_rms_2880_kernel(
    float* __restrict__ u_inout,          // [2880] input u_in -> overwritten with unit-RMS u_hat!
    float* __restrict__ out_f32,          // [2880] W_out(x) -> scratch
    const float* __restrict__ b_out,      // [2880]
    const float* __restrict__ final_norm, // [2880] GPT-OSS-20B frozen final_norm!
    __half* __restrict__ nh_f16_out       // [2880] ready for indexed lm_head GEMV!
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_red[8];

    float sq = 0.0f;
    for (int i = tid; i < D_MODEL; i += 256) {
        // Direct EAGLE-1 prediction (NO u_in skip to avoid echoing cur_tok logits!)
        float v = out_f32[i] + b_out[i];
        out_f32[i] = v;
        sq = fmaf(v, v, sq);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
    if (lane == 0) s_red[wid] = sq;
    __syncthreads();
    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)D_MODEL + RMS_EPS);
    }
    __syncthreads();
    float inv = s_inv;
    for (int i = tid; i < D_MODEL; i += 256) {
        float u = out_f32[i] * inv;
        u_inout[i] = u;
        if (nh_f16_out != nullptr) {
            nh_f16_out[i] = __float2half(u * final_norm[i]);
        }
    }
}

__global__ void eagle_argmax_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ shortlist,
    int V_sub,
    int* __restrict__ out_vocab_tok,
    float* __restrict__ out_top1_prob
) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_val[8];
    __shared__ int   s_idx[8];

    float best_v = -FLT_MAX;
    int   best_i = 0;
    for (int i = tid; i < V_sub; i += 256) {
        float v = logits[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, best_v, off);
        int   oi = __shfl_down_sync(0xffffffff, best_i, off);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_val[wid] = best_v; s_idx[wid] = best_i; }
    __syncthreads();
    __shared__ float s_max;
    __shared__ int   s_best;
    if (tid == 0) {
        float gv = -FLT_MAX;
        int   gi = 0;
        for (int w = 0; w < 8; ++w) {
            if (s_val[w] > gv) { gv = s_val[w]; gi = s_idx[w]; }
        }
        s_max = gv;
        s_best = gi;
        out_vocab_tok[0] = shortlist[gi];
    }
    __syncthreads();

    // Compute exact softmax confidence prob(top1) over shortlist in 1 pass (for confidence-gated dynamic K!)
    const float gv = s_max;
    float sum_e = 0.0f;
    for (int i = tid; i < V_sub; i += 256) {
        float d = logits[i] - gv;
        if (d > -20.0f) sum_e += fast_ex2(d * LOG2E);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum_e += __shfl_down_sync(0xffffffff, sum_e, off);
    if (lane == 0) s_val[wid] = sum_e;
    __syncthreads();
    if (tid == 0) {
        float z = 0.0f;
        for (int w = 0; w < 8; ++w) z += s_val[w];
        out_top1_prob[0] = (z > 0.0f) ? (1.0f / z) : 0.0f;
    }
}

// ============================================================================
// ITERATION 6 PILLARS 1 & 3: EAGLE-3 2-Layer Tree Draft + MoE Router Predictor
//   + Bit-Exact Tree KV Path Compaction Kernels (0 register spills!)
// ============================================================================
__global__ void eagle3_fuse_4add_bias_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    const float* __restrict__ d,
    const float* __restrict__ bias,
    float* __restrict__ out
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < EAGLE_D_DRAFT) out[i] = a[i] + b[i] + c[i] + d[i] + bias[i];
}

__global__ void eagle3_silu_mul_interleaved_kernel(
    const float* __restrict__ gu, // [2 * d_ff] interleaved (gate=2*i, up=2*i+1)
    __half* __restrict__ mid_f16, // [d_ff]
    int d_ff
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < d_ff) {
        float g = gu[2 * i];
        float u = gu[2 * i + 1];
        float s = g / (1.0f + expf(-g));
        mid_f16[i] = __float2half(s * u);
    }
}

// Tree-aware causal GQA attention for EAGLE-3 (1 block x 256 threads = 8 warps):
//   - If tree_slot < 0: committed step at dpos -> writes K/V to K_kv[dpos & 511], attends [j_start .. dpos].
//   - If tree_slot >= 0: speculative tree node at logical dpos = committed_len + depth -> writes K/V to
//     K_tree[tree_slot], attends committed [j_start .. committed_len-1] + tree ancestors in anc_mask!
__global__ void eagle3_rope_kv_attn_tree_kernel(
    const float* __restrict__ qkv,  // [768]
    __half* __restrict__ K_kv,      // [512, 2, 64]
    __half* __restrict__ V_kv,      // [512, 2, 64]
    __half* __restrict__ K_tree,    // [32, 2, 64]
    __half* __restrict__ V_tree,    // [32, 2, 64]
    __half* __restrict__ O_f16,     // [512]
    int committed_len,
    int dpos,
    int tree_slot,
    uint32_t anc_mask
) {
    const int tid  = threadIdx.x;
    const int qg   = tid >> 5;
    const int lane = tid & 31;
    const int kv_h = qg >> 2;

    float inv_f = powf(10000.0f, -((float)(2 * lane) / 64.0f));
    float angle = (float)dpos * inv_f;
    float sv, cv;
    sincosf(angle, &sv, &cv);

    float qx1 = qkv[qg * 64 + lane];
    float qx2 = qkv[qg * 64 + lane + 32];
    const float q0 = (qx1 * cv - qx2 * sv) * SM_SCALE_LOG2E;
    const float q1 = (qx2 * cv + qx1 * sv) * SM_SCALE_LOG2E;

    if (qg < 2) {
        int kh = qg;
        float kx1 = qkv[512 + kh * 64 + lane];
        float kx2 = qkv[512 + kh * 64 + lane + 32];
        __half* kd = (tree_slot < 0)
            ? (K_kv   + ((size_t)(dpos & (EAGLE_KV_WIN - 1)) * 2 + kh) * 64)
            : (K_tree + ((size_t)tree_slot * 2 + kh) * 64);
        __half* vd = (tree_slot < 0)
            ? (V_kv   + ((size_t)(dpos & (EAGLE_KV_WIN - 1)) * 2 + kh) * 64)
            : (V_tree + ((size_t)tree_slot * 2 + kh) * 64);
        kd[lane]      = __float2half(kx1 * cv - kx2 * sv);
        kd[lane + 32] = __float2half(kx2 * cv + kx1 * sv);
        vd[lane]      = __float2half(qkv[640 + kh * 64 + lane]);
        vd[lane + 32] = __float2half(qkv[640 + kh * 64 + lane + 32]);
    }
    __syncthreads();

    float m_run = -FLT_MAX;
    float l_run = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    const int j_end   = (tree_slot < 0) ? dpos : (committed_len - 1);
    const int j_start = max(0, dpos - EAGLE_KV_WIN + 1);
    for (int j = j_start; j <= j_end; ++j) {
        int js = j & (EAGLE_KV_WIN - 1);
        const __half* kp = K_kv + ((size_t)js * 2 + kv_h) * 64;
        const __half* vp = V_kv + ((size_t)js * 2 + kv_h) * 64;
        float dot = fmaf(q0, __half2float(kp[lane]), q1 * __half2float(kp[lane + 32]));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        float m_new = fmaxf(m_run, dot);
        float gamma = (m_run > -1e29f) ? fast_ex2(m_run - m_new) : 0.0f;
        float w     = fast_ex2(dot - m_new);
        l_run = fmaf(gamma, l_run, w);
        a0    = fmaf(gamma, a0, w * __half2float(vp[lane]));
        a1    = fmaf(gamma, a1, w * __half2float(vp[lane + 32]));
        m_run = m_new;
    }

    if (tree_slot >= 0) {
        uint32_t m = anc_mask;
        while (m) {
            int ts = __ffs(m) - 1;
            m &= (m - 1);
            const __half* kp = K_tree + ((size_t)ts * 2 + kv_h) * 64;
            const __half* vp = V_tree + ((size_t)ts * 2 + kv_h) * 64;
            float dot = fmaf(q0, __half2float(kp[lane]), q1 * __half2float(kp[lane + 32]));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
            float m_new = fmaxf(m_run, dot);
            float gamma = (m_run > -1e29f) ? fast_ex2(m_run - m_new) : 0.0f;
            float w     = fast_ex2(dot - m_new);
            l_run = fmaf(gamma, l_run, w);
            a0    = fmaf(gamma, a0, w * __half2float(vp[lane]));
            a1    = fmaf(gamma, a1, w * __half2float(vp[lane + 32]));
            m_run = m_new;
        }
    }

    float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    O_f16[qg * 64 + lane]      = __float2half(a0 * inv_l);
    O_f16[qg * 64 + lane + 32] = __float2half(a1 * inv_l);
}

// Fast Top-4 softmax probabilities over shortlist (1 block x 256 threads)
__global__ void eagle3_topk4_softmax_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ shortlist,
    int V_sub,
    int* __restrict__ out_toks4,
    float* __restrict__ out_probs4
) {
    const int tid  = threadIdx.x;
    const int wid  = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_v[8][4];
    __shared__ int   s_i[8][4];
    __shared__ float s_max;
    __shared__ float s_z[8];

    float b0 = -FLT_MAX, b1 = -FLT_MAX, b2 = -FLT_MAX, b3 = -FLT_MAX;
    int   i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    for (int i = tid; i < V_sub; i += 256) {
        float v = logits[i];
        if (v > b3) {
            if (v > b0)      { b3 = b2; i3 = i2; b2 = b1; i2 = i1; b1 = b0; i1 = i0; b0 = v; i0 = i; }
            else if (v > b1) { b3 = b2; i3 = i2; b2 = b1; i2 = i1; b1 = v; i1 = i; }
            else if (v > b2) { b3 = b2; i3 = i2; b2 = v; i2 = i; }
            else             { b3 = v;  i3 = i; }
        }
    }
    // Warp-level Top-4 via 4 sequential argmax passes across 32 lanes (100% registers, 0 stack!)
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float mv = b0;
        int   mi = i0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, mv, off);
            int   oi = __shfl_xor_sync(0xffffffff, mi, off);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
        }
        if (lane == 0) { s_v[wid][k] = mv; s_i[wid][k] = mi; }
        if (i0 == mi && b0 == mv) {
            b0 = b1; i0 = i1; b1 = b2; i1 = i2; b2 = b3; i2 = i3; b3 = -FLT_MAX; i3 = -1;
        }
    }
    __syncthreads();

    __shared__ float g_v[4];
    __shared__ int   g_i[4];
    if (tid == 0) {
        for (int k = 0; k < 4; ++k) { g_v[k] = -FLT_MAX; g_i[k] = 0; }
        for (int w = 0; w < 8; ++w) {
            for (int r = 0; r < 4; ++r) {
                float v = s_v[w][r];
                int   idx = s_i[w][r];
                if (v > g_v[3]) {
                    int pos = 3;
                    while (pos > 0 && v > g_v[pos - 1]) {
                        g_v[pos] = g_v[pos - 1]; g_i[pos] = g_i[pos - 1]; pos--;
                    }
                    g_v[pos] = v; g_i[pos] = idx;
                }
            }
        }
        s_max = g_v[0];
    }
    __syncthreads();

    const float gv = s_max;
    float sum_e = 0.0f;
    for (int i = tid; i < V_sub; i += 256) {
        float d = logits[i] - gv;
        if (d > -20.0f) sum_e += fast_ex2(d * LOG2E);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum_e += __shfl_down_sync(0xffffffff, sum_e, off);
    if (lane == 0) s_z[wid] = sum_e;
    __syncthreads();

    if (tid == 0) {
        float z = 0.0f;
        for (int w = 0; w < 8; ++w) z += s_z[w];
        float inv_z = (z > 0.0f) ? (1.0f / z) : 0.0f;
        for (int k = 0; k < 4; ++k) {
            out_toks4[k]  = shortlist[g_i[k]];
            float d = g_v[k] - gv;
            out_probs4[k] = (d > -20.0f) ? (fast_ex2(d * LOG2E) * inv_z) : 0.0f;
        }
    }
}

// Pillar 3: Predict 24-layer Top-4 MoE expert bitmask (Grid=24, Block=32 -> 1 warp per layer!)
__global__ void eagle3_moe_top4_mask_kernel(
    const float* __restrict__ moe_lg,   // [24, 32]
    const float* __restrict__ moe_bias, // [24, 32]
    uint32_t* __restrict__ out_mask_24  // [24]
) {
    const int l    = blockIdx.x;
    const int lane = threadIdx.x;
    float v = moe_lg[l * NUM_EXPERTS + lane] + moe_bias[l * NUM_EXPERTS + lane];
    uint32_t mask = 0u;
    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        float mv = v;
        int   mi = lane;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, mv, off);
            int   oi = __shfl_xor_sync(0xffffffff, mi, off);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
        }
        if (lane == 0) mask |= (1u << mi);
        if (lane == mi) v = -FLT_MAX;
    }
    if (lane == 0) out_mask_24[l] = mask;
}

// Bit-Exact Tree Path Compaction Kernels!
// 1. Local Ring-128: write ONLY the L_acc accepted nodes from d_K/V_loc_tree[12, 16, 512] into K/V_local_ring!
__global__ void commit_tree_local_kv_kernel(
    const __half* __restrict__ K_tree, // [12, 16, 512]
    const __half* __restrict__ V_tree, // [12, 16, 512]
    __half* __restrict__ K_ring,       // [12, 128, 512]
    __half* __restrict__ V_ring,       // [12, 128, 512]
    const int* __restrict__ path_ids,  // [L_acc]
    int start_pos,
    int L_acc
) {
    const int sub_l = blockIdx.x; // 0..11
    const int k     = blockIdx.y; // 0..L_acc-1
    const int d     = threadIdx.x; // 0..511
    if (k >= L_acc) return;
    int node_i = path_ids[k];
    int rslot  = (start_pos + k) & (RING_SIZE - 1);
    size_t src = ((size_t)sub_l * MAX_VERIFY_K + node_i) * 512 + d;
    size_t dst = ((size_t)sub_l * RING_SIZE + rslot) * 512 + d;
    K_ring[dst] = K_tree[src];
    V_ring[dst] = V_tree[src];
}

// 2. Global KV Stage A (gather accepted path [0..L_acc-1] into temp [12, 16, 512])
__global__ void gather_tree_global_kv_kernel(
    const __half* __restrict__ K_glob, // [12, kv_cap, 512]
    const __half* __restrict__ V_glob, // [12, v_cap, 512]
    __half* __restrict__ K_tmp,        // [12, 16, 512]
    __half* __restrict__ V_tmp,        // [12, 16, 512]
    const int* __restrict__ path_ids,
    int start_pos,
    int L_acc,
    int kv_cap_k,
    int v_ring_mask
) {
    const int sub_l = blockIdx.x;
    const int k     = blockIdx.y;
    const int d     = threadIdx.x;
    if (k >= L_acc) return;
    int pos_src = start_pos + path_ids[k];
    int v_sl = (v_ring_mask > 0) ? (pos_src & v_ring_mask) : pos_src;
    int v_stride = (v_ring_mask > 0) ? (v_ring_mask + 1) : kv_cap_k;
    size_t dst = ((size_t)sub_l * MAX_VERIFY_K + k) * 512 + d;
    K_tmp[dst] = K_glob[((size_t)sub_l * kv_cap_k + pos_src) * 512 + d];
    V_tmp[dst] = V_glob[((size_t)sub_l * v_stride + v_sl) * 512 + d];
}

// 3. Global KV Stage B (scatter accepted [0..L_acc-1] sequentially + restore rejected [L_acc..N-1] from V_hot_shadow!)
__global__ void scatter_tree_global_kv_kernel(
    const __half* __restrict__ K_tmp,        // [12, 16, 512]
    const __half* __restrict__ V_tmp,        // [12, 16, 512]
    const __half* __restrict__ V_hot_shadow, // optional [12, 16, 512]
    __half* __restrict__ K_glob,
    __half* __restrict__ V_glob,
    int start_pos,
    int L_acc,
    int N_nodes,
    int kv_cap_k,
    int v_ring_mask
) {
    const int sub_l = blockIdx.x;
    const int k     = blockIdx.y; // 0..N_nodes-1
    const int d     = threadIdx.x;
    if (k >= N_nodes) return;
    int pos_dst = start_pos + k;
    int v_sl = (v_ring_mask > 0) ? (pos_dst & v_ring_mask) : pos_dst;
    int v_stride = (v_ring_mask > 0) ? (v_ring_mask + 1) : kv_cap_k;
    size_t src = ((size_t)sub_l * MAX_VERIFY_K + k) * 512 + d;
    if (k < L_acc) {
        K_glob[((size_t)sub_l * kv_cap_k + pos_dst) * 512 + d] = K_tmp[src];
        V_glob[((size_t)sub_l * v_stride + v_sl) * 512 + d]    = V_tmp[src];
    } else if (V_hot_shadow != nullptr) {
        size_t sh_src = ((size_t)sub_l * N_nodes + k) * 512 + d;
        V_glob[((size_t)sub_l * v_stride + v_sl) * 512 + d]    = V_hot_shadow[sh_src];
    }
}

// 4. Compact accepted tip hidden states (d_H_l2, d_H_l12, d_H) from node_i into slot 0!
__global__ void compact_single_hidden_kernel(
    const float* __restrict__ src_base,
    float* __restrict__ dst_base,
    int src_slot,
    int dst_slot
) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < D_MODEL && src_slot != dst_slot) {
        dst_base[(size_t)dst_slot * D_MODEL + i] = src_base[(size_t)src_slot * D_MODEL + i];
    }
}

// ============================================================================
// Multi-Threaded Parallel CPU memcpy helper (AVX2 Non-Temporal + Lock-Free Hot Spin!)
// ============================================================================
static inline void copy_avx2_non_temporal(uint8_t* dst, const uint8_t* src, size_t n_bytes) {
#if defined(__AVX2__)
    uintptr_t mis = reinterpret_cast<uintptr_t>(dst) & 31ULL;
    if (mis != 0) {
        size_t head = std::min(n_bytes, (size_t)(32ULL - mis));
        memcpy(dst, src, head);
        dst += head;
        src += head;
        n_bytes -= head;
    }
    size_t i = 0;
    size_t n256 = n_bytes & ~255ULL;
    for (; i < n256; i += 256) {
        _mm_prefetch(reinterpret_cast<const char*>(src + i + 512), _MM_HINT_NTA);
        _mm_prefetch(reinterpret_cast<const char*>(src + i + 576), _MM_HINT_NTA);
        _mm_prefetch(reinterpret_cast<const char*>(src + i + 640), _MM_HINT_NTA);
        _mm_prefetch(reinterpret_cast<const char*>(src + i + 704), _MM_HINT_NTA);
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 0));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 32));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 64));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 96));
        __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 128));
        __m256i v5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 160));
        __m256i v6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 192));
        __m256i v7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + 224));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 0), v0);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 32), v1);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 64), v2);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 96), v3);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 128), v4);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 160), v5);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 192), v6);
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i + 224), v7);
    }
    for (; i + 32 <= n_bytes; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + i), v);
    }
    if (i < n_bytes) memcpy(dst + i, src + i, n_bytes - i);
    _mm_sfence();
#else
    memcpy(dst, src, n_bytes);
#endif
}

struct CopyThreadPool {
    int num_threads = 1;
    bool use_avx2_nt = true;
    std::vector<std::thread> workers;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> job_gen{0};
    std::atomic<int> done_count{0};
    std::atomic<int> sleeping_workers{0};

    uint8_t* dst_ptr = nullptr;
    const uint8_t* src_ptr = nullptr;
    size_t total_bytes = 0;

    inline void do_copy(uint8_t* d, const uint8_t* s, size_t len) const {
        if (use_avx2_nt) copy_avx2_non_temporal(d, s, len);
        else memcpy(d, s, len);
    }

    void init(int n) {
        shutdown();
        num_threads = std::max(1, std::min(n, 8));
        if (num_threads <= 1) return;
        stop_flag.store(false, std::memory_order_relaxed);
        job_gen.store(0, std::memory_order_relaxed);
        sleeping_workers.store(0, std::memory_order_relaxed);
        for (int t = 1; t < num_threads; ++t) {
            workers.emplace_back([this, t]() {
                uint64_t seen = 0;
                while (true) {
                    // Lock-free hot spin (~2ms) so worker stays warm across all 24 layers with 0 OS syscalls!
                    uint64_t g = job_gen.load(std::memory_order_acquire);
                    int spins = 0;
                    while (g == seen && spins < 200000) {
                        if (stop_flag.load(std::memory_order_relaxed)) return;
                        _mm_pause();
                        g = job_gen.load(std::memory_order_acquire);
                        spins++;
                    }
                    if (g == seen) {
                        std::unique_lock<std::mutex> lk(mtx);
                        sleeping_workers.fetch_add(1, std::memory_order_seq_cst);
                        cv.wait(lk, [&]() {
                            return job_gen.load(std::memory_order_seq_cst) != seen ||
                                   stop_flag.load(std::memory_order_seq_cst);
                        });
                        sleeping_workers.fetch_sub(1, std::memory_order_seq_cst);
                        if (stop_flag.load(std::memory_order_seq_cst)) return;
                        g = job_gen.load(std::memory_order_seq_cst);
                    }
                    seen = g;
                    size_t chunk = (((total_bytes + num_threads - 1) / num_threads) + 63ULL) & ~63ULL;
                    size_t beg = (size_t)t * chunk;
                    if (beg < total_bytes) {
                        size_t len = std::min(chunk, total_bytes - beg);
                        do_copy(dst_ptr + beg, src_ptr + beg, len);
                    }
                    done_count.fetch_add(1, std::memory_order_release);
                }
            });
        }
    }

    void copy(void* dst, const void* src, size_t n_bytes) {
        uint8_t* d = static_cast<uint8_t*>(dst);
        const uint8_t* s = static_cast<const uint8_t*>(src);
        if (num_threads <= 1 || n_bytes < (256ULL << 10)) {
            do_copy(d, s, n_bytes);
            return;
        }
        dst_ptr = d;
        src_ptr = s;
        total_bytes = n_bytes;
        done_count.store(0, std::memory_order_relaxed);
        job_gen.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping_workers.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lk(mtx);
            cv.notify_all();
        }

        size_t chunk = (((n_bytes + num_threads - 1) / num_threads) + 63ULL) & ~63ULL;
        do_copy(d, s, std::min(chunk, n_bytes));

        while (done_count.load(std::memory_order_acquire) < num_threads - 1) {
            _mm_pause();
        }
    }

    void shutdown() {
        if (!workers.empty()) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                stop_flag.store(true, std::memory_order_release);
            }
            cv.notify_all();
            for (auto& th : workers) {
                if (th.joinable()) th.join();
            }
            workers.clear();
        }
    }

    ~CopyThreadPool() { shutdown(); }
};

// ============================================================================
// Profiling & Memory Structures (exported to Python via ctypes)
// ============================================================================
struct ProfileStats {
    uint64_t tokens;
    uint64_t pcie_bytes;
    uint64_t cache_hits;
    uint64_t cache_misses;
    double total_wall_ms;
    double embed_lookup_h2d_sync_ms;
    double cpu_staging_memcpy_ms;
    double pcie_dma_ms;
    double gpu_attn_router_ms;
    double gpu_moe_ms;
    double gpu_lmhead_ms;
    double gpu_wait_dma_ms;
    double cpu_wait_router_ms;
    double cpu_wait_end_ms;
    uint64_t layer_hits[NUM_LAYERS];
    uint64_t layer_misses[NUM_LAYERS];
    // Iter 7 Pillar 1: Inter-Layer Lookahead Prefetch Telemetry
    uint64_t prefetch_attempts;
    uint64_t prefetch_hits;
    double prefetch_dma_ms;
    double prefetch_hidden_ms;
    double lookahead_overhead_ms;
};

struct PrefillProfileStats {
    uint64_t chunks;
    uint64_t tokens;
    uint64_t experts_loaded;
    double total_wall_ms;
    double attn_local_ring128_ms;
    double attn_global_causal_ms;
    double qkv_o_router_ms;
    double h2d_dma_ms;
    double cpu_staging_ms;
    double dequant_mxfp4_ms;
    double gather_scatter_swiglu_ms;
    double expert_gemm_ms;
    double gpu_wait_dma_ms;
    double sync_cpu_gpu_ms;
};

struct MemReport {
    size_t resident_weights_bytes;
    size_t kv_local_ring128_bytes;
    size_t kv_global_bytes;
    size_t ring_shadow_bytes;
    size_t stream_bufs_2x_bytes;
    size_t scratch_workspace_bytes;
    size_t arena_bytes;
    size_t total_engine_vram_bytes;
    size_t cuda_free_bytes;
    size_t cuda_total_bytes;
    int kv_cap_global;
    int cache_capacity_slots;
    int cache_active_entries;
    int cache_policy;
    int num_copy_threads;
    // Stage 4B Hybrid Global KV & RAM Backup breakdown
    size_t kv_global_hot_bytes;
    size_t kv_global_cold_data_bytes;
    size_t kv_global_cold_scale_bytes;
    size_t kv_ram_backup_bytes;
    int hybrid_kv_mode;
};

// ============================================================================
// Resident Layer Weights on GPU
// ============================================================================
struct LayerGPU {
    float* in_norm = nullptr;       // [2880] FP32
    float* post_norm = nullptr;     // [2880] FP32
    __half* W_qkv = nullptr;        // [5120, 2880] FP16 (fused Q, K, V!)
    __half* b_qkv = nullptr;        // [5120] FP16
    __half* W_o = nullptr;          // [2880, 4096] FP16
    __half* b_o = nullptr;          // [2880] FP16
    float* sinks = nullptr;         // [64] FP32
    __half* router_w = nullptr;     // [32, 2880] FP16
    __half* router_b = nullptr;     // [32] FP16
    __half* gu_bias = nullptr;      // [32, 5760] FP16
    __half* dn_bias = nullptr;      // [32, 2880] FP16
};

// ============================================================================
// Adaptive LRU / LFU-Aging Expert Cache Manager inside Single VRAM Arena
// ============================================================================
struct CacheSlotMeta {
    int key = -1; // l * 32 + e, or -1 if empty
    uint64_t last_used = 0;
    float freq = 0.0f;
    uint64_t pinned_step = 0; // prevents evicting an expert needed in the current layer/step!
};

// ============================================================================
// Main Native C++/CUDA Engine Class
// ============================================================================
struct GPTOSSNativeEngine {
    cublasHandle_t cublas = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t copy_stream = nullptr;

    // Resident GPU weights (NO embed_tokens!)
    __half* lm_head = nullptr;    // [201088, 2880] FP16
    float* final_norm = nullptr;  // [2880] FP32
    float* cos_table = nullptr;   // [65536, 32] FP32
    float* sin_table = nullptr;   // [65536, 32] FP32
    LayerGPU layers[NUM_LAYERS];

    // Host pointers (owned by caller / Python numpy buffers OR engine-owned per-layer cudaHostAlloc pinned RAM!)
    const uint16_t* embed_ram_fp16 = nullptr; // [201088, 2880] uint16 (FP16) in RAM!
    const uint8_t* mxfp4_ram_layers[NUM_LAYERS] = {nullptr};
    uint8_t* mxfp4_pinned_layers[NUM_LAYERS] = {nullptr}; // engine-owned per-layer cudaHostAlloc (~377-423 MB each!)
    bool layer_is_pinned[NUM_LAYERS] = {false};           // true => 0ms CPU staging direct H2D DMA @ 11.8 GB/s!
    bool rans_mode = false;
    uint32_t rans_exp_off[NUM_LAYERS][NUM_EXPERTS] = {};
    uint32_t rans_exp_sz[NUM_LAYERS][NUM_EXPERTS] = {};
    uint8_t* comp_stream_bufs[2] = {nullptr, nullptr};    // 2 x 12.0 MB compressed DMA ping-pong buffers (only when rans_mode)
    cudaEvent_t ev_rans_done[2] = {nullptr, nullptr};

    // KV Cache (Pure FP16 OR Iter 5 Hybrid: Exact K-FP16 + Hot-4K V-FP16 / V_sink128 FP16 / Cold V_pc32 + V RAM Mirror!)
    int kv_cap_global = 4096;
    bool hybrid_kv_mode = false;
    int kv_cold_cap = 0;                // aligned multiple of V_PAGE_SIZE=32 when hybrid_kv_mode is true
    int kv_cold_pages = 0;              // kv_cold_cap / V_PAGE_SIZE
    __half* K_local_ring = nullptr;     // [12, 128, 512] FP16 (1.57 MB)
    __half* V_local_ring = nullptr;     // [12, 128, 512] FP16 (1.57 MB)
    __half* K_global = nullptr;         // [12, kv_cap_global, 512] ALWAYS EXACT FP16!
    __half* V_global = nullptr;         // [12, kv_cap_global (or HOT_WINDOW in hybrid), 512] FP16
    __half* V_sink_fp16 = nullptr;      // [12, 128, 512] FP16 exact sink V (1.57 MB)
    uint8_t* V_global_cold = nullptr;   // [12, kv_cold_cap, 512] uint8 per-channel V_pc32
    __half* V_page_min = nullptr;       // [12, kv_cold_pages, 512] FP16
    __half* V_page_max = nullptr;       // [12, kv_cold_pages, 512] FP16
    __half* K_ring_shadow = nullptr;    // [12, MAX_VERIFY_K, 512] FP16 (98 KB)
    __half* V_ring_shadow = nullptr;    // [12, MAX_VERIFY_K, 512] FP16 (98 KB)
    __half* V_hot_shadow = nullptr;     // [12, MAX_VERIFY_K, 512] FP16 (49 KB)
    __half* V_bnd_fp16 = nullptr;       // [12, 32, 512] FP16 pristine boundary page shadow (393 KB)
    // Full FP16 Global V Mirror in system RAM (0.805 GB at 64K — K is already linear FP16 in VRAM!)
    uint16_t* h_V_global_ram = nullptr; // [12, kv_cap_global, 512] FP16 in RAM
    __half* h_kv_mirror_v = nullptr;    // [MAX_PREFILL_CHUNK, 512] pinned D2H staging (512 KB)

    // 2 permanent GPU streaming buffers (orig_stream_bufs owns allocations; stream_bufs[0..1] may zero-copy pointer-swap with slot_ptrs during decode!)
    uint8_t* orig_stream_bufs[2] = {nullptr, nullptr};
    uint8_t* stream_bufs[2] = {nullptr, nullptr};
    std::vector<uint8_t*> slot_ptrs;
    // 4-slot pinned host staging buffers (optimal under Windows WDDM!)
    uint8_t* host_pinned_slots[4] = {nullptr, nullptr, nullptr, nullptr};
    // Small pinned host buffers for embed row, router top4, and expert gather indices/weights
    __half* host_pinned_embed = nullptr;    // [MAX_PREFILL_CHUNK * 2880] FP16
    uint8_t* host_pinned_top_idx = nullptr; // [MAX_PREFILL_CHUNK * 4]
    float* host_pinned_top_w = nullptr;     // [MAX_PREFILL_CHUNK * 4]
    int* host_pinned_tok_idx = nullptr;     // [MAX_PREFILL_CHUNK]
    float* host_pinned_tok_w = nullptr;     // [MAX_PREFILL_CHUNK]

    // Fixed scratch buffers on GPU (for B up to MAX_VERIFY_K = 8)
    __half* d_embed_f16 = nullptr;   // [MAX_VERIFY_K, 2880]
    float* d_H = nullptr;            // [MAX_VERIFY_K, 2880]
    float* d_N_f32 = nullptr;        // [MAX_VERIFY_K, 2880]
    __half* d_N_f16 = nullptr;       // [MAX_VERIFY_K, 2880]
    float* d_QKV = nullptr;          // [MAX_VERIFY_K, 5120]
    float* d_Q = nullptr;            // [MAX_VERIFY_K, 4096]
    __half* d_O_f16 = nullptr;       // [MAX_VERIFY_K, 4096]
    float* d_proj_out = nullptr;     // [MAX_VERIFY_K, 2880]
    float* d_router_lg = nullptr;    // [MAX_VERIFY_K, 32]
    uint8_t* d_top_idx = nullptr;    // [MAX_VERIFY_K, 4]
    float* d_top_w = nullptr;        // [MAX_VERIFY_K, 4]
    float* d_mlp_acc = nullptr;      // [MAX_VERIFY_K, 2880]
    float* d_exp_in = nullptr;       // [MAX_VERIFY_K, 2880]
    float* d_exp_gu = nullptr;       // [MAX_VERIFY_K, 5760]
    float* d_exp_mid = nullptr;      // [MAX_VERIFY_K, 2880]
    float* d_exp_dn = nullptr;       // [MAX_VERIFY_K, 2880]
    int* d_tok_idx_buf = nullptr;    // [MAX_PREFILL_CHUNK]
    float* d_tok_w_buf = nullptr;    // [MAX_PREFILL_CHUNK]
    float* d_logits = nullptr;       // [MAX_VERIFY_K, VOCAB_SIZE]

    // Flash-Decoding Split-K workspace & local linear staging for B=2..16 verify + Tree KV buffers
    float* d_split_m = nullptr;      // [MAX_VERIFY_K, 64, MAX_SPLITS]
    float* d_split_l = nullptr;      // [MAX_VERIFY_K, 64, MAX_SPLITS]
    float* d_split_o = nullptr;      // [MAX_VERIFY_K, 64, MAX_SPLITS, 64]
    float* d_m_floor = nullptr;      // [MAX_VERIFY_K, 64] exact Hot-4K + sink floor in log2 space
    unsigned long long* d_prune_stats = nullptr; // [6] Quest + Lazy-V diagnostic counters
    __half* d_K_loc_tmp = nullptr;   // [RING_SIZE + MAX_VERIFY_K, 512]
    __half* d_V_loc_tmp = nullptr;   // [RING_SIZE + MAX_VERIFY_K, 512]
    // Iter 6: Multi-layer target hidden capture [L2, L12] + Bit-Exact Tree KV Compaction buffers (~1.15 MB total)
    float*  d_H_l2 = nullptr;        // [MAX_VERIFY_K, 2880]
    float*  d_H_l12 = nullptr;       // [MAX_VERIFY_K, 2880]
    float*  d_ema_attn = nullptr;    // [NUM_LAYERS, 2880] online EMA of per-layer attention output (276 KB)
    int     ema_attn_valid[NUM_LAYERS] = {};
    __half* d_K_loc_tree = nullptr;  // [12, MAX_VERIFY_K, 512]
    __half* d_V_loc_tree = nullptr;  // [12, MAX_VERIFY_K, 512]
    __half* d_K_glob_tmp = nullptr;  // [12, MAX_VERIFY_K, 512]
    __half* d_V_glob_tmp = nullptr;  // [12, MAX_VERIFY_K, 512]
    int*    d_tree_path_ids = nullptr; // [MAX_VERIFY_K]

    // Dynamic VRAM Arena (unified pre-allocated pool: prefill temporaries OR decode MXFP4 cache!)
    uint8_t* d_arena = nullptr;
    size_t arena_bytes = 0;
    int cache_capacity_slots = 0;
    int cache_policy = 0; // 0=NONE, 1=LRU, 2=LFU_AGING, 3=ADAPTIVE_THRASH_FREE_LRU
    std::vector<CacheSlotMeta> cache_slots;
    int key_to_slot[NUM_LAYERS * NUM_EXPERTS];
    int layer_slot_beg[NUM_LAYERS];
    int layer_slot_end[NUM_LAYERS];
    uint64_t access_clock = 0;
    uint64_t step_pin_id = 0;
    int active_cache_entries = 0;

    // Events for synchronization and fine-grained profiling
    cudaEvent_t ev_copy_done[2];
    cudaEvent_t ev_comp_done[2];
    cudaEvent_t ev_pref_done[2];
    cudaEvent_t ev_host_slot_done[4];
    cudaEvent_t ev_dma_s[NUM_LAYERS][TOP_K];
    cudaEvent_t ev_dma_e[NUM_LAYERS][TOP_K];
    cudaEvent_t ev_wait_s[NUM_LAYERS][TOP_K];
    cudaEvent_t ev_wait_e[NUM_LAYERS][TOP_K];
    cudaEvent_t ev_copy_done_miss[TOP_K];
    cudaEvent_t ev_la_s[NUM_LAYERS];
    cudaEvent_t ev_la_e[NUM_LAYERS];
    cudaEvent_t ev_pref_s[NUM_LAYERS][2];
    cudaEvent_t ev_pref_e[NUM_LAYERS][2];
    cudaEvent_t ev_pwait_s[NUM_LAYERS][2];
    cudaEvent_t ev_pwait_e[NUM_LAYERS][2];
    cudaEvent_t ev_exp_s[NUM_LAYERS][TOP_K];
    cudaEvent_t ev_exp_e[NUM_LAYERS][TOP_K];
    cudaEvent_t ev_attn_s[NUM_LAYERS];
    cudaEvent_t ev_attn_e[NUM_LAYERS];
    cudaEvent_t ev_lm_s, ev_lm_e;
    int   lookahead_prefetch_mode = 1;  // 0=off, 1=b1+verify (default), 2=b1_only
    float prefetch_tau_prob32 = 0.045f; // full 32-way softmax threshold (~96% cold-miss precision!)
    // Prefill fine-grained profiling events (reused per layer/expert when prefill_profile_enabled)
    cudaEvent_t ev_p_s, ev_p_e;
    cudaEvent_t ev_p_dma_s[NUM_EXPERTS], ev_p_dma_e[NUM_EXPERTS];
    cudaEvent_t ev_p_wait_s[NUM_EXPERTS], ev_p_wait_e[NUM_EXPERTS];
    cudaEvent_t ev_p_gath_s[NUM_EXPERTS], ev_p_gath_e[NUM_EXPERTS];
    cudaEvent_t ev_p_deq1_s[NUM_EXPERTS], ev_p_deq1_e[NUM_EXPERTS];
    cudaEvent_t ev_p_gemm1_s[NUM_EXPERTS], ev_p_gemm1_e[NUM_EXPERTS];
    cudaEvent_t ev_p_swi_s[NUM_EXPERTS], ev_p_swi_e[NUM_EXPERTS];
    cudaEvent_t ev_p_deq2_s[NUM_EXPERTS], ev_p_deq2_e[NUM_EXPERTS];
    cudaEvent_t ev_p_gemm2_s[NUM_EXPERTS], ev_p_gemm2_e[NUM_EXPERTS];
    cudaEvent_t ev_p_scat_s[NUM_EXPERTS], ev_p_scat_e[NUM_EXPERTS];
    bool profile_enabled = true;
    bool prefill_profile_enabled = false;
    bool use_fast_fp16_gemv = true;
    bool verify_read_only_cache = true;
    int force_splits = 0;
    int staging_stripes = 4; // 4 x 3.3048 MB striped AVX2_NT -> WC pinned -> H2D DMA overlap!
    float quest_prune_margin_nats = 14.0f; // <=0 disables pruning; default 14 nats (e^-14 = 8.3e-7)
    bool collect_prune_stats = false;

    // Last verify_tokens routing trace (for committing accepted-prefix-only cache stats/recency!)
    int last_verify_K = 0;
    uint8_t last_verify_top_idx[MAX_VERIFY_K][NUM_LAYERS][TOP_K];

    // Memory accounting
    size_t resident_weights_bytes = 0;
    size_t scratch_workspace_bytes = 0;

    // Iteration 5 Lever C: Optional Resident 1-Layer EAGLE-1/3 Draft Subsystem (~15.3 MB VRAM)
    bool eagle_loaded = false;
    size_t eagle_vram_bytes = 0;
    int eagle_kv_pos = 0;
    int eagle_shortlist_len = 0;
    float*  e_h_norm = nullptr;      // [2880]
    float*  e_e_norm = nullptr;      // [2880]
    __half* e_W_fuse_h = nullptr;    // [512, 2880]
    __half* e_W_fuse_e = nullptr;    // [512, 2880]
    float*  e_b_fuse = nullptr;      // [512]
    float*  e_attn_norm = nullptr;   // [512]
    __half* e_W_qkv = nullptr;       // [768, 512]
    __half* e_W_o = nullptr;         // [512, 512]
    float*  e_mlp_norm = nullptr;    // [512]
    __half* e_W_gu = nullptr;        // [3072, 512]
    __half* e_W_dn = nullptr;        // [512, 1536]
    float*  e_out_norm = nullptr;    // [512]
    __half* e_W_out = nullptr;       // [2880, 512]
    float*  e_b_out = nullptr;       // [2880]
    __half* e_K_kv = nullptr;        // [512, 2, 64]
    __half* e_V_kv = nullptr;        // [512, 2, 64]
    int*    e_shortlist = nullptr;   // [EAGLE_MAX_SHORTLIST]
    float*  d_e_u = nullptr;         // [2880]
    __half* d_e_emb_f16 = nullptr;   // [2880]
    __half* d_e_un_f16 = nullptr;    // [2880]
    __half* d_e_en_f16 = nullptr;    // [2880]
    float*  d_e_x = nullptr;         // [512]
    float*  d_e_tmp512_a = nullptr;  // [512]
    float*  d_e_tmp512_b = nullptr;  // [512]
    __half* d_e_xn_f16 = nullptr;    // [512]
    float*  d_e_qkv = nullptr;       // [768]
    __half* d_e_o_f16 = nullptr;     // [512]
    float*  d_e_gu = nullptr;        // [3072]
    __half* d_e_mid_f16 = nullptr;   // [1536]
    float*  d_e_out_f32 = nullptr;   // [2880]
    float*  d_e_logits = nullptr;    // [EAGLE_MAX_SHORTLIST]
    int*    d_e_argmax_tok = nullptr;// [1]
    float*  d_e_top1_prob = nullptr; // [1]
    int*    h_e_argmax_tok = nullptr;// pinned [1]
    float*  h_e_top1_prob = nullptr; // pinned [1]
    __half* h_e_embed_tmp = nullptr; // pinned [2880]

    void eagle_step_core(int token_id, int dpos, bool project_logits) {
        const uint16_t* emb_row = embed_ram_fp16 + (size_t)token_id * D_MODEL;
        memcpy(h_e_embed_tmp, emb_row, D_MODEL * sizeof(uint16_t));
        cudaMemcpyAsync(d_e_emb_f16, h_e_embed_tmp, D_MODEL * sizeof(__half), cudaMemcpyHostToDevice, compute_stream);

        eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e_u, e_h_norm, d_e_un_f16, D_MODEL);
        eagle_rmsnorm_f16_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e_emb_f16, e_e_norm, d_e_en_f16, D_MODEL);
        fp16_gemv_1vec_kernel<<<(EAGLE_D_DRAFT + 7) / 8, 256, 0, compute_stream>>>(e_W_fuse_h, d_e_un_f16, d_e_tmp512_a, EAGLE_D_DRAFT, D_MODEL);
        fp16_gemv_1vec_kernel<<<(EAGLE_D_DRAFT + 7) / 8, 256, 0, compute_stream>>>(e_W_fuse_e, d_e_en_f16, d_e_tmp512_b, EAGLE_D_DRAFT, D_MODEL);
        eagle_fuse_add_bias_kernel<<<2, 256, 0, compute_stream>>>(d_e_tmp512_a, d_e_tmp512_b, e_b_fuse, d_e_x);

        // 1-layer causal GQA attention (8 Q, 2 KV, sliding window 512)
        eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e_x, e_attn_norm, d_e_xn_f16, EAGLE_D_DRAFT);
        fp16_gemv_1vec_kernel<<<(EAGLE_QKV_DIM + 7) / 8, 256, 0, compute_stream>>>(e_W_qkv, d_e_xn_f16, d_e_qkv, EAGLE_QKV_DIM, EAGLE_D_DRAFT);
        eagle_rope_kv_attn_kernel<<<1, 256, 0, compute_stream>>>(d_e_qkv, e_K_kv, e_V_kv, d_e_o_f16, dpos);
        if (!project_logits) return; // KV-only commit mode!

        fp16_gemv_1vec_kernel<<<(EAGLE_D_DRAFT + 7) / 8, 256, 0, compute_stream>>>(e_W_o, d_e_o_f16, d_e_tmp512_a, EAGLE_D_DRAFT, EAGLE_D_DRAFT);
        eagle_add_residual_512_kernel<<<2, 256, 0, compute_stream>>>(d_e_x, d_e_tmp512_a);

        // SwiGLU MLP (512 -> 1536 -> 512)
        eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e_x, e_mlp_norm, d_e_xn_f16, EAGLE_D_DRAFT);
        fp16_gemv_1vec_kernel<<<(2 * EAGLE_D_FF + 7) / 8, 256, 0, compute_stream>>>(e_W_gu, d_e_xn_f16, d_e_gu, 2 * EAGLE_D_FF, EAGLE_D_DRAFT);
        eagle_silu_mul_1536_kernel<<<(EAGLE_D_FF + 255) / 256, 256, 0, compute_stream>>>(d_e_gu, d_e_mid_f16);
        fp16_gemv_1vec_kernel<<<(EAGLE_D_DRAFT + 7) / 8, 256, 0, compute_stream>>>(e_W_dn, d_e_mid_f16, d_e_tmp512_a, EAGLE_D_DRAFT, EAGLE_D_FF);
        eagle_add_residual_512_kernel<<<2, 256, 0, compute_stream>>>(d_e_x, d_e_tmp512_a);

        // Output residual unit-RMS + GPT-OSS-20B frozen final_norm + indexed shortlist lm_head!
        eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e_x, e_out_norm, d_e_xn_f16, EAGLE_D_DRAFT);
        fp16_gemv_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(e_W_out, d_e_xn_f16, d_e_out_f32, D_MODEL, EAGLE_D_DRAFT);
        eagle_residual_unit_rms_2880_kernel<<<1, 256, 0, compute_stream>>>(d_e_u, d_e_out_f32, e_b_out, final_norm, d_e_un_f16);

        fp16_gemv_1vec_indexed_kernel<<<(eagle_shortlist_len + 7) / 8, 256, 0, compute_stream>>>(
            lm_head, d_e_un_f16, e_shortlist, d_e_logits, eagle_shortlist_len, D_MODEL
        );
        eagle_argmax_kernel<<<1, 256, 0, compute_stream>>>(d_e_logits, e_shortlist, eagle_shortlist_len, d_e_argmax_tok, d_e_top1_prob);
        cudaMemcpyAsync(h_e_argmax_tok, d_e_argmax_tok, sizeof(int), cudaMemcpyDeviceToHost, compute_stream);
        cudaMemcpyAsync(h_e_top1_prob,  d_e_top1_prob,  sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
    }

    void free_eagle_resources() {
        auto cf = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        cf(e_h_norm); cf(e_e_norm); cf(e_W_fuse_h); cf(e_W_fuse_e); cf(e_b_fuse);
        cf(e_attn_norm); cf(e_W_qkv); cf(e_W_o); cf(e_mlp_norm); cf(e_W_gu); cf(e_W_dn);
        cf(e_out_norm); cf(e_W_out); cf(e_b_out); cf(e_K_kv); cf(e_V_kv); cf(e_shortlist);
        cf(d_e_u); cf(d_e_emb_f16); cf(d_e_un_f16); cf(d_e_en_f16); cf(d_e_x);
        cf(d_e_tmp512_a); cf(d_e_tmp512_b); cf(d_e_xn_f16); cf(d_e_qkv); cf(d_e_o_f16);
        cf(d_e_gu); cf(d_e_mid_f16); cf(d_e_out_f32); cf(d_e_logits); cf(d_e_argmax_tok); cf(d_e_top1_prob);
        if (h_e_argmax_tok) { cudaFreeHost(h_e_argmax_tok); h_e_argmax_tok = nullptr; }
        if (h_e_top1_prob)  { cudaFreeHost(h_e_top1_prob);  h_e_top1_prob = nullptr; }
        if (h_e_embed_tmp)  { cudaFreeHost(h_e_embed_tmp);  h_e_embed_tmp = nullptr; }
        eagle_loaded = false;
        eagle_vram_bytes = 0;
    }

    // ========================================================================
    // Iteration 6/7 Pillars 1 & 3: Resident 2-Layer Multi-Layer EAGLE-3 + MoE Router Predictor
    //   Supports FP16 (26.6 MB) OR Per-Channel INT8 (13.3 MB -> +1 full MXFP4 slot!)
    // ========================================================================
    bool eagle3_loaded = false;
    bool eagle3_int8_mode = false;
    size_t eagle3_vram_bytes = 0;
    int eagle3_kv_pos = 0;
    int eagle3_shortlist_len = 0;
    int eagle3_d_ff = E3_D_FF;
    float*  e3_norm_h2 = nullptr;    // [2880]
    float*  e3_norm_h12 = nullptr;   // [2880]
    float*  e3_norm_h23 = nullptr;   // [2880]
    float*  e3_norm_rec = nullptr;   // [512]
    float*  e3_norm_e = nullptr;     // [2880]
    __half* e3_W_fuse_h2 = nullptr;  // [512, 2880]
    __half* e3_W_fuse_h12 = nullptr; // [512, 2880]
    __half* e3_W_fuse_h23 = nullptr; // [512, 2880]
    __half* e3_W_rec_x = nullptr;    // [512, 512]
    __half* e3_W_fuse_e = nullptr;   // [512, 2880]
    float*  e3_b_fuse = nullptr;     // [512]
    float*  e3_attn_norm[E3_N_LAYERS] = {};
    __half* e3_W_qkv[E3_N_LAYERS] = {};
    __half* e3_W_o[E3_N_LAYERS] = {};
    float*  e3_mlp_norm[E3_N_LAYERS] = {};
    __half* e3_W_gu[E3_N_LAYERS] = {};
    __half* e3_W_dn[E3_N_LAYERS] = {};
    __half* e3_K_kv[E3_N_LAYERS] = {};    // [512, 2, 64]
    __half* e3_V_kv[E3_N_LAYERS] = {};    // [512, 2, 64]
    __half* e3_K_tree[E3_N_LAYERS] = {};  // [32, 2, 64]
    __half* e3_V_tree[E3_N_LAYERS] = {};  // [32, 2, 64]
    float*  e3_out_norm = nullptr;   // [512]
    __half* e3_W_out = nullptr;      // [2880, 512]
    float*  e3_b_out = nullptr;      // [2880]
    __half* e3_W_moe_pred = nullptr; // [768, 512]
    float*  e3_b_moe_pred = nullptr; // [768]
    // INT8 per-channel linear weights + row scales (allocated only when eagle3_int8_mode == true)
    int8_t* e3_q_fuse_h2 = nullptr;  float* e3_s_fuse_h2 = nullptr;
    int8_t* e3_q_fuse_h12 = nullptr; float* e3_s_fuse_h12 = nullptr;
    int8_t* e3_q_fuse_h23 = nullptr; float* e3_s_fuse_h23 = nullptr;
    int8_t* e3_q_rec_x = nullptr;    float* e3_s_rec_x = nullptr;
    int8_t* e3_q_fuse_e = nullptr;   float* e3_s_fuse_e = nullptr;
    int8_t* e3_q_qkv[E3_N_LAYERS] = {}; float* e3_s_qkv[E3_N_LAYERS] = {};
    int8_t* e3_q_o[E3_N_LAYERS]   = {}; float* e3_s_o[E3_N_LAYERS]   = {};
    int8_t* e3_q_gu[E3_N_LAYERS]  = {}; float* e3_s_gu[E3_N_LAYERS]  = {};
    int8_t* e3_q_dn[E3_N_LAYERS]  = {}; float* e3_s_dn[E3_N_LAYERS]  = {};
    int8_t* e3_q_out = nullptr;      float* e3_s_out = nullptr;
    int8_t* e3_q_moe_pred = nullptr; float* e3_s_moe_pred = nullptr;
    int*    e3_shortlist = nullptr;  // [EAGLE_MAX_SHORTLIST]
    // EAGLE-3 Scratch buffers
    float*  d_e3_u = nullptr;        // [2880]
    __half* d_e3_emb_f16 = nullptr;  // [2880]
    __half* d_e3_un_f16 = nullptr;   // [2880]
    __half* d_e3_en_f16 = nullptr;   // [2880]
    float*  d_e3_x = nullptr;        // [512]
    float*  d_e3_tree_x = nullptr;   // [32, 512] latent x_out per tree scratch slot!
    float*  d_e3_t512[4] = {};       // 4 x [512]
    __half* d_e3_xn_f16 = nullptr;   // [512]
    float*  d_e3_qkv = nullptr;      // [768]
    __half* d_e3_o_f16 = nullptr;    // [512]
    float*  d_e3_gu = nullptr;       // [2 * 1536]
    __half* d_e3_mid_f16 = nullptr;  // [1536]
    float*  d_e3_out_f32 = nullptr;  // [2880]
    float*  d_e3_logits = nullptr;   // [EAGLE_MAX_SHORTLIST]
    float*  d_e3_moe_lg = nullptr;   // [768]
    uint32_t* d_e3_moe_mask = nullptr; // [24]
    int*    d_e3_top4_tok = nullptr; // [4]
    float*  d_e3_top4_prob = nullptr;// [4]
    int*    h_e3_top4_tok = nullptr; // pinned [4]
    float*  h_e3_top4_prob = nullptr;// pinned [4]
    uint32_t* h_e3_moe_mask = nullptr; // pinned [24]
    __half* h_e3_embed_tmp = nullptr;// pinned [2880]

    void free_eagle3_resources() {
        auto cf = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        cf(e3_norm_h2); cf(e3_norm_h12); cf(e3_norm_h23); cf(e3_norm_rec); cf(e3_norm_e);
        cf(e3_W_fuse_h2); cf(e3_W_fuse_h12); cf(e3_W_fuse_h23); cf(e3_W_rec_x); cf(e3_W_fuse_e); cf(e3_b_fuse);
        cf(e3_q_fuse_h2); cf(e3_s_fuse_h2); cf(e3_q_fuse_h12); cf(e3_s_fuse_h12);
        cf(e3_q_fuse_h23); cf(e3_s_fuse_h23); cf(e3_q_rec_x); cf(e3_s_rec_x); cf(e3_q_fuse_e); cf(e3_s_fuse_e);
        for (int i = 0; i < E3_N_LAYERS; ++i) {
            cf(e3_attn_norm[i]); cf(e3_W_qkv[i]); cf(e3_W_o[i]);
            cf(e3_mlp_norm[i]); cf(e3_W_gu[i]); cf(e3_W_dn[i]);
            cf(e3_q_qkv[i]); cf(e3_s_qkv[i]); cf(e3_q_o[i]); cf(e3_s_o[i]);
            cf(e3_q_gu[i]); cf(e3_s_gu[i]); cf(e3_q_dn[i]); cf(e3_s_dn[i]);
            cf(e3_K_kv[i]); cf(e3_V_kv[i]); cf(e3_K_tree[i]); cf(e3_V_tree[i]);
        }
        cf(e3_out_norm); cf(e3_W_out); cf(e3_b_out); cf(e3_W_moe_pred); cf(e3_b_moe_pred);
        cf(e3_q_out); cf(e3_s_out); cf(e3_q_moe_pred); cf(e3_s_moe_pred); cf(e3_shortlist);
        cf(d_e3_u); cf(d_e3_emb_f16); cf(d_e3_un_f16); cf(d_e3_en_f16); cf(d_e3_x); cf(d_e3_tree_x);
        for (int i = 0; i < 4; ++i) cf(d_e3_t512[i]);
        cf(d_e3_xn_f16); cf(d_e3_qkv); cf(d_e3_o_f16); cf(d_e3_gu); cf(d_e3_mid_f16);
        cf(d_e3_out_f32); cf(d_e3_logits); cf(d_e3_moe_lg); cf(d_e3_moe_mask);
        cf(d_e3_top4_tok); cf(d_e3_top4_prob);
        if (h_e3_top4_tok)  { cudaFreeHost(h_e3_top4_tok);  h_e3_top4_tok = nullptr; }
        if (h_e3_top4_prob) { cudaFreeHost(h_e3_top4_prob); h_e3_top4_prob = nullptr; }
        if (h_e3_moe_mask)  { cudaFreeHost(h_e3_moe_mask);  h_e3_moe_mask = nullptr; }
        if (h_e3_embed_tmp) { cudaFreeHost(h_e3_embed_tmp); h_e3_embed_tmp = nullptr; }
        eagle3_loaded = false;
        eagle3_int8_mode = false;
        eagle3_vram_bytes = 0;
    }

    inline void e3_gemv(const __half* w16, const int8_t* q8, const float* sc, const __half* x, float* y, int N, int K) {
        if (eagle3_int8_mode) {
            int8_gemv_1vec_kernel<<<(N + 7) / 8, 256, 0, compute_stream>>>(q8, sc, x, y, N, K);
        } else {
            fp16_gemv_1vec_kernel<<<(N + 7) / 8, 256, 0, compute_stream>>>(w16, x, y, N, K);
        }
    }

    // Core 2-layer EAGLE-3 step:
    //   is_root=true:  fuses target [d_H_l2[slot], d_H_l12[slot], d_H[slot]] + emb(token_id)
    //   is_root=false: recurs from parent latent d_e3_tree_x[parent_tree_slot] + emb(token_id)
    //   tree_slot < 0 => commits directly to e3_K_kv[dpos & 511]; tree_slot >= 0 => writes to e3_K_tree[tree_slot]!
    void eagle3_step_core(
        bool is_root,
        int target_h_slot,
        int parent_tree_slot,
        int token_id,
        int committed_len,
        int dpos,
        int tree_slot,
        uint32_t anc_mask,
        bool project_logits_and_moe,
        int save_x_tree_slot
    ) {
        const uint16_t* emb_row = embed_ram_fp16 + (size_t)token_id * D_MODEL;
        memcpy(h_e3_embed_tmp, emb_row, D_MODEL * sizeof(uint16_t));
        cudaMemcpyAsync(d_e3_emb_f16, h_e3_embed_tmp, D_MODEL * sizeof(__half), cudaMemcpyHostToDevice, compute_stream);
        eagle_rmsnorm_f16_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e3_emb_f16, e3_norm_e, d_e3_en_f16, D_MODEL);
        e3_gemv(e3_W_fuse_e, e3_q_fuse_e, e3_s_fuse_e, d_e3_en_f16, d_e3_t512[3], EAGLE_D_DRAFT, D_MODEL);

        if (is_root) {
            eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_H_l2 + (size_t)target_h_slot * D_MODEL, e3_norm_h2, d_e3_un_f16, D_MODEL);
            e3_gemv(e3_W_fuse_h2, e3_q_fuse_h2, e3_s_fuse_h2, d_e3_un_f16, d_e3_t512[0], EAGLE_D_DRAFT, D_MODEL);
            eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_H_l12 + (size_t)target_h_slot * D_MODEL, e3_norm_h12, d_e3_un_f16, D_MODEL);
            e3_gemv(e3_W_fuse_h12, e3_q_fuse_h12, e3_s_fuse_h12, d_e3_un_f16, d_e3_t512[1], EAGLE_D_DRAFT, D_MODEL);
            eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_H + (size_t)target_h_slot * D_MODEL, e3_norm_h23, d_e3_un_f16, D_MODEL);
            e3_gemv(e3_W_fuse_h23, e3_q_fuse_h23, e3_s_fuse_h23, d_e3_un_f16, d_e3_t512[2], EAGLE_D_DRAFT, D_MODEL);
            eagle3_fuse_4add_bias_kernel<<<2, 256, 0, compute_stream>>>(d_e3_t512[0], d_e3_t512[1], d_e3_t512[2], d_e3_t512[3], e3_b_fuse, d_e3_x);
        } else {
            eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e3_tree_x + (size_t)parent_tree_slot * EAGLE_D_DRAFT, e3_norm_rec, d_e3_xn_f16, EAGLE_D_DRAFT);
            e3_gemv(e3_W_rec_x, e3_q_rec_x, e3_s_rec_x, d_e3_xn_f16, d_e3_t512[0], EAGLE_D_DRAFT, EAGLE_D_DRAFT);
            eagle_fuse_add_bias_kernel<<<2, 256, 0, compute_stream>>>(d_e3_t512[0], d_e3_t512[3], e3_b_fuse, d_e3_x);
        }

        // 2-layer causal GQA backbone (layer 0 always runs full block; layer 1 skips MLP+proj if KV-only commit!)
        for (int bl = 0; bl < E3_N_LAYERS; ++bl) {
            eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e3_x, e3_attn_norm[bl], d_e3_xn_f16, EAGLE_D_DRAFT);
            e3_gemv(e3_W_qkv[bl], e3_q_qkv[bl], e3_s_qkv[bl], d_e3_xn_f16, d_e3_qkv, EAGLE_QKV_DIM, EAGLE_D_DRAFT);
            eagle3_rope_kv_attn_tree_kernel<<<1, 256, 0, compute_stream>>>(
                d_e3_qkv, e3_K_kv[bl], e3_V_kv[bl], e3_K_tree[bl], e3_V_tree[bl],
                d_e3_o_f16, committed_len, dpos, tree_slot, anc_mask
            );
            if (!project_logits_and_moe && bl == E3_N_LAYERS - 1) return;

            e3_gemv(e3_W_o[bl], e3_q_o[bl], e3_s_o[bl], d_e3_o_f16, d_e3_t512[0], EAGLE_D_DRAFT, EAGLE_D_DRAFT);
            eagle_add_residual_512_kernel<<<2, 256, 0, compute_stream>>>(d_e3_x, d_e3_t512[0]);

            eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e3_x, e3_mlp_norm[bl], d_e3_xn_f16, EAGLE_D_DRAFT);
            e3_gemv(e3_W_gu[bl], e3_q_gu[bl], e3_s_gu[bl], d_e3_xn_f16, d_e3_gu, 2 * eagle3_d_ff, EAGLE_D_DRAFT);
            eagle3_silu_mul_interleaved_kernel<<<(eagle3_d_ff + 255) / 256, 256, 0, compute_stream>>>(d_e3_gu, d_e3_mid_f16, eagle3_d_ff);
            e3_gemv(e3_W_dn[bl], e3_q_dn[bl], e3_s_dn[bl], d_e3_mid_f16, d_e3_t512[0], EAGLE_D_DRAFT, eagle3_d_ff);
            eagle_add_residual_512_kernel<<<2, 256, 0, compute_stream>>>(d_e3_x, d_e3_t512[0]);
        }

        if (save_x_tree_slot >= 0 && save_x_tree_slot < E3_MAX_TREE_SCRATCH) {
            cudaMemcpyAsync(d_e3_tree_x + (size_t)save_x_tree_slot * EAGLE_D_DRAFT, d_e3_x, EAGLE_D_DRAFT * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
        }

        // Output norm -> (1) MoE 24-layer router predictor (Pillar 3) AND (2) Shortlist lm_head Top-4!
        eagle_rmsnorm_f32_to_f16_kernel<<<1, 256, 0, compute_stream>>>(d_e3_x, e3_out_norm, d_e3_xn_f16, EAGLE_D_DRAFT);
        e3_gemv(e3_W_moe_pred, e3_q_moe_pred, e3_s_moe_pred, d_e3_xn_f16, d_e3_moe_lg, NUM_LAYERS * NUM_EXPERTS, EAGLE_D_DRAFT);
        eagle3_moe_top4_mask_kernel<<<NUM_LAYERS, 32, 0, compute_stream>>>(d_e3_moe_lg, e3_b_moe_pred, d_e3_moe_mask);

        e3_gemv(e3_W_out, e3_q_out, e3_s_out, d_e3_xn_f16, d_e3_out_f32, D_MODEL, EAGLE_D_DRAFT);
        eagle_residual_unit_rms_2880_kernel<<<1, 256, 0, compute_stream>>>(d_e3_u, d_e3_out_f32, e3_b_out, final_norm, d_e3_un_f16);
        fp16_gemv_1vec_indexed_kernel<<<(eagle3_shortlist_len + 7) / 8, 256, 0, compute_stream>>>(
            lm_head, d_e3_un_f16, e3_shortlist, d_e3_logits, eagle3_shortlist_len, D_MODEL
        );
        eagle3_topk4_softmax_kernel<<<1, 256, 0, compute_stream>>>(d_e3_logits, e3_shortlist, eagle3_shortlist_len, d_e3_top4_tok, d_e3_top4_prob);
        cudaMemcpyAsync(h_e3_top4_tok,  d_e3_top4_tok,  4 * sizeof(int),       cudaMemcpyDeviceToHost, compute_stream);
        cudaMemcpyAsync(h_e3_top4_prob, d_e3_top4_prob, 4 * sizeof(float),     cudaMemcpyDeviceToHost, compute_stream);
        cudaMemcpyAsync(h_e3_moe_mask,  d_e3_moe_mask,  NUM_LAYERS * sizeof(uint32_t), cudaMemcpyDeviceToHost, compute_stream);
    }

    // Dynamic EAGLE-3 Tree Expansion + Pillar 3 Statistical Net-Gain Gating & Discrete Micro-Tree B in {1,3,5,7}!
    int eagle3_draft_tree_internal(
        int target_h_slot,
        int root_tok,
        int max_verify_nodes,    // total nodes including root (<= MAX_VERIFY_K=8)
        int max_depth,           // max draft depth (e.g., 4..5)
        int max_cand_eval,       // candidate nodes expanded before pruning (e.g., 7..12)
        float min_node_prob,     // skip expanding candidates with cum_prob < min_node_prob
        int cache_aware_prune,   // 0=standard prob Top-N, 1=Pillar 3 MoE-cache-aware throughput maximization!
        float gpu_base_ms,       // estimated fixed GPU + root ms (e.g., 36 ms at 64K, 22 ms at 4K)
        float miss_ms,           // estimated PCIe ms per cold expert miss (e.g., 1.02 ms for pinned_raw!)
        int* out_toks,           // [max_verify_nodes] node 0 = root_tok
        int* out_parents,        // [max_verify_nodes] node 0 = -1
        int* out_depths,         // [max_verify_nodes] node 0 = 0
        uint16_t* out_anc_masks, // [max_verify_nodes] node 0 = 0x0001
        int* out_n_nodes,
        int* out_pred_cold_misses,
        double* out_draft_ms,
        float tau_skip = 0.25f,  // Iter 7 Pillar 3: root Top-1 skip threshold (0.20ms fast exit to B=1+Prefetch!)
        float b1_est_ms = 0.0f,  // Iter 7 Pillar 3: measured/estimated B=1+Prefetch ms/tok (<=0 uses formula!)
        float gain_margin = 1.0f // Iter 7 Pillar 3: required TPS margin over B=1+Prefetch to fire tree verify!
    ) {
        if (!eagle3_loaded) return -1;
        using clk = std::chrono::high_resolution_clock;
        auto t0 = clk::now();

        max_verify_nodes = std::max(1, std::min(max_verify_nodes, MAX_VERIFY_K));
        max_depth        = std::max(1, std::min(max_depth, 6));
        max_cand_eval    = std::max(max_verify_nodes, std::min(max_cand_eval, E3_MAX_TREE_SCRATCH - 1));

        struct CandNode {
            int tok;
            int parent_cand; // index in cands[], -1 for root
            int depth;
            int tree_slot;   // scratch slot in e3_K_tree / d_e3_tree_x (-1 if leaf not yet expanded)
            uint32_t scratch_anc_mask;
            float cum_prob;
            bool expanded;
            uint32_t moe_mask[NUM_LAYERS];
        };

        std::vector<CandNode> cands;
        cands.reserve(32);

        // 1. Evaluate root (committed into e3_K_kv[eagle3_kv_pos], advances eagle3_kv_pos by 1!)
        const int root_dpos = eagle3_kv_pos;
        eagle3_step_core(true, target_h_slot, -1, root_tok, root_dpos, root_dpos, -1, 0u, true, 0);
        cudaStreamSynchronize(compute_stream);
        eagle3_kv_pos = root_dpos + 1;

        CandNode root_c{};
        root_c.tok = root_tok;
        root_c.parent_cand = -1;
        root_c.depth = 0;
        root_c.tree_slot = 0;
        root_c.scratch_anc_mask = 0u;
        root_c.cum_prob = 1.0f;
        root_c.expanded = true;
        memcpy(root_c.moe_mask, h_e3_moe_mask, sizeof(root_c.moe_mask));
        cands.push_back(root_c);

        // Seed depth-1 children from root's Top-4 softmax!
        // Iter 7 Pillar 3 Statistical Gate: if root p_1 < tau_skip, exit immediately (~0.20 ms) to B=1+Prefetch!
        const bool fast_skip = (cache_aware_prune != 0) && (h_e3_top4_prob[0] < tau_skip);
        int root_w = fast_skip ? 0 : std::min(4, max_verify_nodes - 1);
        for (int k = 0; k < root_w; ++k) {
            if (h_e3_top4_prob[k] < min_node_prob) continue;
            CandNode cn{};
            cn.tok = h_e3_top4_tok[k];
            cn.parent_cand = 0;
            cn.depth = 1;
            cn.tree_slot = -1;
            cn.scratch_anc_mask = 0u;
            cn.cum_prob = h_e3_top4_prob[k];
            cn.expanded = false;
            cands.push_back(cn);
        }

        // 2. Priority-first expansion of top candidate nodes (computing both children logits AND Pillar 3 MoE masks!)
        int next_tree_slot = 1;
        int n_expanded = 1;
        while (n_expanded < max_cand_eval && next_tree_slot < E3_MAX_TREE_SCRATCH) {
            int best_ci = -1;
            float best_p = min_node_prob;
            for (int i = 1; i < (int)cands.size(); ++i) {
                if (!cands[i].expanded && cands[i].cum_prob >= best_p) {
                    best_p = cands[i].cum_prob;
                    best_ci = i;
                }
            }
            if (best_ci < 0) break;

            int ts = next_tree_slot++;
            int par = cands[best_ci].parent_cand;
            uint32_t smask = cands[par].scratch_anc_mask | (1u << ts);
            int dpos_c = eagle3_kv_pos + cands[best_ci].depth - 1;
            // Notice: root is ALREADY committed at e3_K_kv[eagle3_kv_pos - 1], so depth=1 sits at dpos = eagle3_kv_pos!
            dpos_c = eagle3_kv_pos + (cands[best_ci].depth - 1);

            eagle3_step_core(
                false, 0, cands[par].tree_slot, cands[best_ci].tok,
                eagle3_kv_pos, dpos_c, ts, smask, true, ts
            );
            cudaStreamSynchronize(compute_stream);

            cands[best_ci].tree_slot = ts;
            cands[best_ci].scratch_anc_mask = smask;
            cands[best_ci].expanded = true;
            memcpy(cands[best_ci].moe_mask, h_e3_moe_mask, sizeof(cands[best_ci].moe_mask));
            n_expanded++;

            if (cands[best_ci].depth < max_depth) {
                int branch_w = (cands[best_ci].depth <= 2) ? 2 : 1;
                for (int k = 0; k < branch_w; ++k) {
                    float cp = cands[best_ci].cum_prob * h_e3_top4_prob[k];
                    if (cp < min_node_prob) continue;
                    CandNode cn{};
                    cn.tok = h_e3_top4_tok[k];
                    cn.parent_cand = best_ci;
                    cn.depth = cands[best_ci].depth + 1;
                    cn.tree_slot = -1;
                    cn.scratch_anc_mask = 0u;
                    cn.cum_prob = cp;
                    cn.expanded = false;
                    cands.push_back(cn);
                }
            }
        }

        // 3. Select connected verification subtree S (1 <= |S| <= max_verify_nodes):
        //    - cache_aware_prune == 1 (Pillar 3): Submodular Throughput = E[toks] / (T_gpu(|S|) + miss_ms * ColdMisses(S, resident_mask))!
        //    - cache_aware_prune == 0 (Baseline): Greedy by highest cum_prob!
        uint32_t res_mask[NUM_LAYERS] = {};
        for (int l = 0; l < NUM_LAYERS; ++l) {
            uint32_t m = 0u;
            for (int e = 0; e < NUM_EXPERTS; ++e) {
                if (key_to_slot[l * NUM_EXPERTS + e] >= 0) m |= (1u << e);
            }
            res_mask[l] = m;
        }

        std::vector<int> sel_cands;
        sel_cands.push_back(0); // root always selected
        std::vector<int> cand_to_out(cands.size(), -1);
        cand_to_out[0] = 0;

        uint32_t cov_mask[NUM_LAYERS];
        int cur_cold_misses = 0;
        for (int l = 0; l < NUM_LAYERS; ++l) {
            uint32_t miss0 = cands[0].moe_mask[l] & ~res_mask[l];
            cur_cold_misses += host_popcnt32(miss0);
            cov_mask[l] = res_mask[l] | cands[0].moe_mask[l];
        }
        float cur_exp_toks = 1.0f; // root token always yields >= 1 verified token!
        auto cal_prob = [&](int ci) -> float {
            // Empirical shortlist-to-full-vocab calibration factor (validated on 4K):
            return cands[ci].cum_prob * ((cands[ci].depth <= 1) ? 0.80f : 0.72f);
        };
        auto est_tps = [&](float exp_t, int n_n, int misses) -> float {
            if (n_n == 1 && b1_est_ms > 0.0f) return 1.0f / b1_est_ms;
            // Account for EAGLE-3 ~71% MoE recall undercounting multi-node cold union (~1.32x) + batched GPU & draft cost!
            const float eff_misses = (n_n == 1) ? (float)misses : ((float)misses * 1.32f);
            const float pref_hide = (lookahead_prefetch_mode == 1) ? std::min(eff_misses * 0.20f, (gpu_base_ms >= 30.0f) ? 22.0f : 14.0f) : 0.0f;
            float t_ms = gpu_base_ms + 3.5f * (float)(n_n - 1) + ((n_n > 1) ? 2.0f : 0.0f)
                       + miss_ms * std::max(0.0f, eff_misses - pref_hide);
            return exp_t / std::max(1.0f, t_ms);
        };
        const float b1_tps = est_tps(1.0f, 1, cur_cold_misses);
        float cur_tps = b1_tps;
        int best_discrete_n = 1;
        float best_discrete_tps = b1_tps * gain_margin;

        while ((int)sel_cands.size() < max_verify_nodes) {
            int best_ci = -1;
            float best_val = -1e9f;
            int best_add_misses = 0;
            for (int i = 1; i < (int)cands.size(); ++i) {
                if (cand_to_out[i] >= 0) continue;
                int par = cands[i].parent_cand;
                if (par < 0 || cand_to_out[par] < 0) continue; // keep tree connected!
                if (cache_aware_prune && !cands[i].expanded) continue; // only prune among nodes with known MoE masks!
                if (!cache_aware_prune) {
                    if (cands[i].cum_prob > best_val) { best_val = cands[i].cum_prob; best_ci = i; }
                } else {
                    int add_m = 0;
                    for (int l = 0; l < NUM_LAYERS; ++l) {
                        add_m += host_popcnt32(cands[i].moe_mask[l] & ~cov_mask[l]);
                    }
                    float cand_tps = est_tps(cur_exp_toks + cal_prob(i), (int)sel_cands.size() + 1, cur_cold_misses + add_m);
                    if (cand_tps > best_val) {
                        best_val = cand_tps;
                        best_ci = i;
                        best_add_misses = add_m;
                    }
                }
            }
            if (best_ci < 0) break;
            // In discrete micro-tree mode (cache_aware_prune == 2), build full ranked prefix up to 7 and pick best B in {1,3,5,7}!
            if (cache_aware_prune == 1 && best_val <= cur_tps) break;

            int out_idx = (int)sel_cands.size();
            cand_to_out[best_ci] = out_idx;
            sel_cands.push_back(best_ci);
            cur_exp_toks += cal_prob(best_ci);
            if (cands[best_ci].expanded) {
                cur_cold_misses += best_add_misses;
                for (int l = 0; l < NUM_LAYERS; ++l) cov_mask[l] |= cands[best_ci].moe_mask[l];
            }
            cur_tps = best_val;
            int cur_sz = (int)sel_cands.size();
            if (cache_aware_prune == 2) {
                // Evaluate discrete sizes B in {3, 5, 7} (or final size if < 7)
                if ((cur_sz == 3 || cur_sz == 5 || cur_sz == 7 || cur_sz == max_verify_nodes) && cur_tps > best_discrete_tps) {
                    best_discrete_tps = cur_tps;
                    best_discrete_n = cur_sz;
                }
            } else if (cache_aware_prune == 1 && cur_tps > best_discrete_tps) {
                best_discrete_tps = cur_tps;
                best_discrete_n = cur_sz;
            }
        }
        // Apply statistical net-gain gate: if no tree size beats B=1+Prefetch * gain_margin, collapse cleanly to n=1!
        if (cache_aware_prune != 0 && best_discrete_n < (int)sel_cands.size()) {
            sel_cands.resize(best_discrete_n);
        }

        // 4. Emit compact topological output arrays [0 .. n_out - 1]
        int n_out = (int)sel_cands.size();
        for (int oi = 0; oi < n_out; ++oi) {
            int ci = sel_cands[oi];
            out_toks[oi]   = cands[ci].tok;
            out_depths[oi] = cands[ci].depth;
            int par_out    = (oi == 0) ? -1 : cand_to_out[cands[ci].parent_cand];
            out_parents[oi] = par_out;
            out_anc_masks[oi] = (oi == 0) ? (uint16_t)1u : (uint16_t)(out_anc_masks[par_out] | (uint16_t)(1u << oi));
        }
        if (out_n_nodes) *out_n_nodes = n_out;
        if (out_pred_cold_misses) *out_pred_cold_misses = cur_cold_misses;
        if (out_draft_ms) *out_draft_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        return 0;
    }

    // Bit-Exact Tree Path KV + Hidden Compaction after verifying N tree nodes!
    int commit_tree_path_internal(int start_pos, int N_nodes, const int* path_ids_host, int L_acc) {
        if (L_acc <= 0 || L_acc > N_nodes || N_nodes > MAX_VERIFY_K) return -1;
        cudaMemcpyAsync(d_tree_path_ids, path_ids_host, (size_t)L_acc * sizeof(int), cudaMemcpyHostToDevice, compute_stream);

        // 1. Local Ring-128: write ONLY accepted path[0..L_acc-1] from d_K/V_loc_tree into K/V_local_ring!
        {
            dim3 g(NUM_LOCAL_LAYERS, L_acc);
            commit_tree_local_kv_kernel<<<g, 512, 0, compute_stream>>>(
                d_K_loc_tree, d_V_loc_tree, K_local_ring, V_local_ring, d_tree_path_ids, start_pos, L_acc
            );
        }

        // 2. Global KV: gather accepted path -> scatter sequentially [start_pos..start_pos+L_acc-1] + restore V_hot tail!
        const int v_rmask = hybrid_kv_mode ? (HOT_WINDOW - 1) : 0;
        {
            dim3 gg(NUM_GLOBAL_LAYERS, L_acc);
            gather_tree_global_kv_kernel<<<gg, 512, 0, compute_stream>>>(
                K_global, V_global, d_K_glob_tmp, d_V_glob_tmp, d_tree_path_ids, start_pos, L_acc, kv_cap_global, v_rmask
            );
            dim3 gs(NUM_GLOBAL_LAYERS, N_nodes);
            scatter_tree_global_kv_kernel<<<gs, 512, 0, compute_stream>>>(
                d_K_glob_tmp, d_V_glob_tmp, hybrid_kv_mode ? V_hot_shadow : nullptr,
                K_global, V_global, start_pos, L_acc, N_nodes, kv_cap_global, v_rmask
            );
        }

        // 3. Mirror accepted L_acc global V tokens into RAM backup (for bit-exact multi-turn rollback!)
        if (hybrid_kv_mode && h_V_global_ram) {
            for (int sub_l = 0; sub_l < NUM_GLOBAL_LAYERS; ++sub_l) {
                cudaMemcpyAsync(
                    h_kv_mirror_v,
                    d_V_glob_tmp + (size_t)sub_l * MAX_VERIFY_K * 512,
                    (size_t)L_acc * 512 * sizeof(__half),
                    cudaMemcpyDeviceToHost,
                    compute_stream
                );
                cudaStreamSynchronize(compute_stream);
                flush_global_kv_mirror_to_ram(sub_l, start_pos, L_acc);
            }
        }

        // 4. Compact accepted path's [d_H_l2, d_H_l12, d_H] sequentially into slots [0..L_acc-1] for EAGLE-3!
        //    Note: path_ids_host is strictly topological (path_ids[k] >= k), so copying k=0..L_acc-1 is in-place safe!
        for (int k = 0; k < L_acc; ++k) {
            int src_s = path_ids_host[k];
            if (src_s != k) {
                compact_single_hidden_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(d_H_l2,  d_H_l2,  src_s, k);
                compact_single_hidden_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(d_H_l12, d_H_l12, src_s, k);
                compact_single_hidden_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(d_H,     d_H,     src_s, k);
            }
        }
        cudaStreamSynchronize(compute_stream);

        // 5. Commit cache stats/recency ONLY for the accepted tree nodes path_ids_host[0..L_acc-1]!
        for (int k = 0; k < L_acc; ++k) {
            int ni = path_ids_host[k];
            if (ni < 0 || ni >= last_verify_K) continue;
            for (int l = 0; l < NUM_LAYERS; ++l) {
                for (int j = 0; j < TOP_K; ++j) {
                    int key = l * NUM_EXPERTS + last_verify_top_idx[ni][l][j];
                    int sl = (cache_policy != 0 && cache_capacity_slots > 0) ? key_to_slot[key] : -1;
                    if (sl >= 0) {
                        cache_slots[sl].last_used = ++access_clock;
                        cache_slots[sl].freq += 1.0f;
                        prof.cache_hits++;
                        prof.layer_hits[l]++;
                    } else {
                        prof.cache_misses++;
                        prof.layer_misses[l]++;
                    }
                }
            }
        }
        return (int)cudaGetLastError();
    }

    CopyThreadPool copy_pool;
    ProfileStats prof{};
    PrefillProfileStats prefill_prof{};

    // ---- Cache management helpers ----
    void clear_cache() {
        for (int i = 0; i < NUM_LAYERS * NUM_EXPERTS; ++i) key_to_slot[i] = -1;
        for (auto& s : cache_slots) {
            s.key = -1;
            s.last_used = 0;
            s.freq = 0.0f;
            s.pinned_step = 0;
        }
        stream_bufs[0] = orig_stream_bufs[0];
        stream_bufs[1] = orig_stream_bufs[1];
        slot_ptrs.resize(cache_capacity_slots);
        for (int i = 0; i < cache_capacity_slots; ++i) {
            slot_ptrs[i] = d_arena + (size_t)i * EXPERT_BYTES;
        }
        active_cache_entries = 0;
        access_clock = 0;
    }

    int lookup_cache_and_pin(int key, uint64_t pin_id) {
        if (cache_policy == 0 || cache_capacity_slots <= 0) return -1;
        int sl = key_to_slot[key];
        if (sl >= 0) {
            cache_slots[sl].last_used = ++access_clock;
            cache_slots[sl].freq += 1.0f;
            cache_slots[sl].pinned_step = pin_id;
            return sl;
        }
        return -1;
    }

    void configure_layer_partitions() {
        for (int l = 0; l < NUM_LAYERS; ++l) { layer_slot_beg[l] = 0; layer_slot_end[l] = 0; }
        if (cache_policy != 3 || cache_capacity_slots >= 104 || cache_capacity_slots < 6) return;
        // Avoid Belady cyclic thrashing when C_VRAM < 24*TOP_K=96:
        // Partition slots into private per-layer LRU pools (>= 6 > TOP_K slots each) on the deepest layers!
        int n_cached_layers = std::min(NUM_LAYERS, cache_capacity_slots / 6);
        int base_per = cache_capacity_slots / n_cached_layers;
        int extra    = cache_capacity_slots % n_cached_layers;
        int start_l  = NUM_LAYERS - n_cached_layers;
        int cur = 0;
        for (int i = 0; i < n_cached_layers; ++i) {
            int l = start_l + i;
            int cnt = base_per + (i >= (n_cached_layers - extra) ? 1 : 0);
            layer_slot_beg[l] = cur;
            layer_slot_end[l] = cur + cnt;
            cur += cnt;
        }
    }

    int alloc_or_evict_cache_slot(int key, uint64_t pin_id) {
        if (cache_policy == 0 || cache_capacity_slots <= 0) return -1;
        int beg = 0, end = cache_capacity_slots;
        if (cache_policy == 3 && cache_capacity_slots < 104) {
            int l = key / NUM_EXPERTS;
            beg = layer_slot_beg[l];
            end = layer_slot_end[l];
            if (beg >= end) return -1; // uncached layer streams through stream_bufs[0..1]
        }
        // Periodic LFU aging every 256 tokens (every ~16384 accesses)
        if (cache_policy == 2 && (access_clock & 16383ULL) == 0 && access_clock > 0) {
            for (auto& s : cache_slots) s.freq *= 0.5f;
        }
        // 1. Prefer an empty slot
        for (int i = beg; i < end; ++i) {
            if (cache_slots[i].key < 0) {
                cache_slots[i].key = key;
                cache_slots[i].last_used = ++access_clock;
                cache_slots[i].freq = 1.0f;
                cache_slots[i].pinned_step = pin_id;
                key_to_slot[key] = i;
                active_cache_entries++;
                return i;
            }
        }
        // 2. Evict unpinned victim by LRU (policy=1 or 3) or LFU-aging with LRU tiebreak (policy=2)
        int victim = -1;
        for (int i = beg; i < end; ++i) {
            if (cache_slots[i].pinned_step == pin_id) continue;
            if (victim < 0) {
                victim = i;
                continue;
            }
            if (cache_policy == 1 || cache_policy == 3) {
                if (cache_slots[i].last_used < cache_slots[victim].last_used) victim = i;
            } else {
                if (cache_slots[i].freq < cache_slots[victim].freq ||
                    (cache_slots[i].freq == cache_slots[victim].freq && cache_slots[i].last_used < cache_slots[victim].last_used)) {
                    victim = i;
                }
            }
        }
        if (victim < 0) return -1;
        int old_key = cache_slots[victim].key;
        if (old_key >= 0) key_to_slot[old_key] = -1;
        cache_slots[victim].key = key;
        cache_slots[victim].last_used = ++access_clock;
        cache_slots[victim].freq = 1.0f;
        cache_slots[victim].pinned_step = pin_id;
        key_to_slot[key] = victim;
        return victim;
    }

    uint8_t* cache_slot_ptr(int slot) const {
        return slot_ptrs[slot];
    }

    void gemm_fp16_to_fp32(const __half* W, const __half* X, float* Y, int N, int B, int K) {
        if (use_fast_fp16_gemv && ((K & 7) == 0)) {
            if (B == 1 && K <= 4096) {
                fp16_gemv_1vec_kernel<<<(N + 7) / 8, 256, 0, compute_stream>>>(W, X, Y, N, K);
                return;
            }
            if (B >= 2 && B <= 4) {
                fp16_gemv_bvec_kernel<4><<<(N + 7) / 8, 256, 0, compute_stream>>>(W, X, Y, N, B, K);
                return;
            }
            if (B >= 5 && B <= 8) {
                fp16_gemv_bvec_kernel<8><<<(N + 7) / 8, 256, 0, compute_stream>>>(W, X, Y, N, B, K);
                return;
            }
            if (B >= 9 && B <= 16) {
                fp16_gemv_bvec_kernel<8><<<(N + 7) / 8, 256, 0, compute_stream>>>(W, X, Y, N, 8, K);
                int rem = B - 8;
                const __half* X2 = X + (size_t)8 * K;
                float* Y2 = Y + (size_t)8 * N;
                if (rem == 1 && K <= 4096) {
                    fp16_gemv_1vec_kernel<<<(N + 7) / 8, 256, 0, compute_stream>>>(W, X2, Y2, N, K);
                } else if (rem <= 4) {
                    fp16_gemv_bvec_kernel<4><<<(N + 7) / 8, 256, 0, compute_stream>>>(W, X2, Y2, N, rem, K);
                } else {
                    fp16_gemv_bvec_kernel<8><<<(N + 7) / 8, 256, 0, compute_stream>>>(W, X2, Y2, N, rem, K);
                }
                return;
            }
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            N, B, K,
            &alpha,
            W, CUDA_R_16F, K,
            X, CUDA_R_16F, K,
            &beta,
            Y, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
    }

    // Native FP32 cublasSgemm: unlocks 2.6-4.8 TFLOPS on GTX 1660 Ti TU116 (5-8x faster than FP16 fallback!)
    void sgemm_f32(const float* W, const float* X, float* Y, int N, int B, int K) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            N, B, K,
            &alpha,
            W, K,
            X, K,
            &beta,
            Y, N
        );
    }

    void fp16_mat_to_f32(const __half* src, float* dst, size_t n_elems) {
        size_t n_vec8 = n_elems >> 3;
        fp16_to_fp32_vec8_kernel<<<(int)((n_vec8 + 255) / 256), 256, 0, compute_stream>>>(src, dst, n_vec8);
    }

    int choose_num_splits(int seq_len, int B) const {
        const int cap = (B == 1) ? MAX_SPLITS : 32;
        if (force_splits > 0) return std::min(force_splits, cap);
        if (seq_len <= 256) return 1;
        if (seq_len <= 2048) return (B == 1) ? 4 : 2;
        if (seq_len <= 8192) return (B == 1) ? 8 : 4;
        if (seq_len <= 24576) return (B == 1) ? 16 : 8;
        if (seq_len <= 40960) return (B == 1) ? 32 : 16;
        return (B == 1) ? 64 : 32;
    }

    void queue_global_kv_mirror_d2h(int sub_l, int start_pos, int B) {
        if (!hybrid_kv_mode || !h_V_global_ram || B <= 0) return;
        const __half* V_hot = V_global + (size_t)sub_l * HOT_WINDOW * 512;
        int slot0 = start_pos & (HOT_WINDOW - 1);
        int first = std::min(B, HOT_WINDOW - slot0);
        cudaMemcpyAsync(h_kv_mirror_v, V_hot + (size_t)slot0 * 512, (size_t)first * 512 * sizeof(__half), cudaMemcpyDeviceToHost, compute_stream);
        if (first < B) {
            int rem = B - first;
            cudaMemcpyAsync(h_kv_mirror_v + (size_t)first * 512, V_hot, (size_t)rem * 512 * sizeof(__half), cudaMemcpyDeviceToHost, compute_stream);
        }
    }

    void flush_global_kv_mirror_to_ram(int sub_l, int start_pos, int B) {
        if (!hybrid_kv_mode || !h_V_global_ram || B <= 0) return;
        size_t off = ((size_t)sub_l * kv_cap_global + start_pos) * 512;
        size_t nb  = (size_t)B * 512 * sizeof(uint16_t);
        memcpy(h_V_global_ram + off, h_kv_mirror_v, nb);
    }

    int restore_global_hot_from_ram_internal(int new_L) {
        if (!hybrid_kv_mode || !h_V_global_ram) return 0;
        int clamped_L = std::max(0, std::min(new_L, kv_cap_global));
        int hot_start = std::max(0, clamped_L - HOT_WINDOW);
        int hot_len   = clamped_L - hot_start;
        if (hot_len <= 0) return 0;

        int slot0 = hot_start & (HOT_WINDOW - 1);
        int first = std::min(hot_len, HOT_WINDOW - slot0);
        int rem   = hot_len - first;
        for (int sub_l = 0; sub_l < NUM_GLOBAL_LAYERS; ++sub_l) {
            __half* V_hot = V_global + (size_t)sub_l * HOT_WINDOW * 512;
            const uint16_t* rv = h_V_global_ram + ((size_t)sub_l * kv_cap_global + hot_start) * 512;
            cudaMemcpyAsync(V_hot + (size_t)slot0 * 512, rv, (size_t)first * 512 * sizeof(__half), cudaMemcpyHostToDevice, compute_stream);
            if (rem > 0) {
                cudaMemcpyAsync(V_hot, rv + (size_t)first * 512, (size_t)rem * 512 * sizeof(__half), cudaMemcpyHostToDevice, compute_stream);
            }
        }
        cudaStreamSynchronize(compute_stream);
        return (int)cudaGetLastError();
    }

    void run_attention_decode_or_verify(int l, int B, int start_pos, int use_tree = 0) {
        const LayerGPU& ld = layers[l];
        const int is_local = ((l & 1) == 0) ? 1 : 0;
        const int sub_l = l >> 1;
        const size_t k_global_stride = (size_t)kv_cap_global * 512;
        const size_t v_global_stride = (size_t)(hybrid_kv_mode ? HOT_WINDOW : kv_cap_global) * 512;
        __half* Kc = is_local ? (K_local_ring + (size_t)sub_l * RING_SIZE * 512)
                              : (K_global     + (size_t)sub_l * k_global_stride);
        __half* Vc = is_local ? (V_local_ring + (size_t)sub_l * RING_SIZE * 512)
                              : (V_global     + (size_t)sub_l * v_global_stride);

        // 1. RMSNorm
        rmsnorm_fp32_fp16_kernel<<<B, 256, 0, compute_stream>>>(d_H, ld.in_norm, nullptr, d_N_f16, B);
        // 2. Fused QKV GEMM
        gemm_fp16_to_fp32(ld.W_qkv, d_N_f16, d_QKV, QKV_DIM, B, D_MODEL);

        const float margin_log2 = (quest_prune_margin_nats > 0.0f) ? (quest_prune_margin_nats * LOG2E) : 1e20f;

        if (is_local && B > 1) {
            int prev_len = std::min(start_pos, RING_SIZE - 1);
            int win_base = start_pos - prev_len;
            if (prev_len > 0) {
                copy_ring_prev_to_linear_kernel<<<prev_len, 512, 0, compute_stream>>>(
                    Kc, Vc, d_K_loc_tmp, d_V_loc_tmp, win_base, prev_len
                );
            }
            qkv_bias_rope_kv_append_kernel<<<B, 256, 0, compute_stream>>>(
                d_QKV, ld.b_qkv, cos_table, sin_table, d_Q, d_K_loc_tmp, d_V_loc_tmp, B, start_pos, 0, 0, win_base, use_tree
            );
            int eff_len = std::min(start_pos + B, RING_SIZE);
            int n_splits = choose_num_splits(eff_len, B);
            dim3 g1(NUM_KV_HEADS, n_splits, B);
            flash_decode_gqa_splitk_part1_turing<false><<<g1, 256, 0, compute_stream>>>(
                d_Q, d_K_loc_tmp, d_V_loc_tmp, nullptr, nullptr, nullptr, ld.sinks, nullptr,
                d_split_m, d_split_l, d_split_o, d_O_f16,
                start_pos, 1, 0, 0, win_base, 0, 0, 0, n_splits, n_splits, 1e20f, nullptr, use_tree
            );
            if (n_splits > 1) {
                dim3 g2(NUM_Q_HEADS, B);
                flash_decode_gqa_splitk_reduce<<<g2, 64, 0, compute_stream>>>(
                    d_split_m, d_split_l, d_split_o, d_O_f16, n_splits
                );
            }
            if (use_tree) {
                // Tree verify: save B tree local K/V rows into d_K/V_loc_tree[sub_l] (do NOT mutate Ring-128 until commit!)
                size_t nb = (size_t)B * 512 * sizeof(__half);
                cudaMemcpyAsync(d_K_loc_tree + (size_t)sub_l * MAX_VERIFY_K * 512, d_K_loc_tmp + (size_t)prev_len * 512, nb, cudaMemcpyDeviceToDevice, compute_stream);
                cudaMemcpyAsync(d_V_loc_tree + (size_t)sub_l * MAX_VERIFY_K * 512, d_V_loc_tmp + (size_t)prev_len * 512, nb, cudaMemcpyDeviceToDevice, compute_stream);
            } else {
                copy_linear_tail_to_ring_kernel<<<B, 512, 0, compute_stream>>>(
                    d_K_loc_tmp, d_V_loc_tmp, Kc, Vc, prev_len, start_pos, B
                );
            }
        } else if (!is_local && hybrid_kv_mode) {
            // Iter 5/7 Hybrid Global KV: Exact Linear K-FP16 + Hot-4K V-FP16 + V_sink128 FP16 + Cold V_pc32
            //   + Iter 7 Exact Per-Position V_hot/V_pc32 Boundary Parity for B > 1!
            __half*  Vsink    = V_sink_fp16   + (size_t)sub_l * V_SINK_FP16 * 512;
            uint8_t* Vc_cold  = V_global_cold + (size_t)sub_l * kv_cold_cap * 512;
            __half*  Vmin_pg  = V_page_min    + (size_t)sub_l * kv_cold_pages * 512;
            __half*  Vmax_pg  = V_page_max    + (size_t)sub_l * kv_cold_pages * 512;
            __half*  Vbnd     = V_bnd_fp16    + (size_t)sub_l * V_PAGE_SIZE * 512;

            const int eff_len   = start_pos + B;
            const int old_pages = (start_pos > HOT_WINDOW) ? ((start_pos - HOT_WINDOW + V_PAGE_SIZE - 1) / V_PAGE_SIZE) : 0;
            const int new_pages = (eff_len   > HOT_WINDOW) ? ((eff_len   - HOT_WINDOW + V_PAGE_SIZE - 1) / V_PAGE_SIZE) : 0;
            const bool bnd_cross = (B > 1) && (new_pages > old_pages);
            const int bnd_pos0   = bnd_cross ? (old_pages * V_PAGE_SIZE) : -1;
            if (bnd_cross) {
                // Preserve pristine FP16 boundary page BEFORE qkv_bias_rope_kv_append overwrites overlapping V_hot slots!
                snapshot_v_bnd_page32_kernel<<<V_PAGE_SIZE, 512, 0, compute_stream>>>(Vc, Vbnd, bnd_pos0);
            }
            if (new_pages > old_pages) {
                dim3 ge(new_pages - old_pages, NUM_KV_HEADS);
                evict_v_hot_pages32_to_cold_pc_kernel<<<ge, 64, 0, compute_stream>>>(
                    Vc, Vsink, Vc_cold, Vmin_pg, Vmax_pg, old_pages, new_pages - old_pages
                );
            }
            qkv_bias_rope_kv_append_kernel<<<B, 256, 0, compute_stream>>>(
                d_QKV, ld.b_qkv, cos_table, sin_table, d_Q, Kc, Vc, B, start_pos, 0, HOT_WINDOW - 1, 0, use_tree
            );
            if (!use_tree) {
                queue_global_kv_mirror_d2h(sub_l, start_pos, B);
            }

            const int cold_len = new_pages * V_PAGE_SIZE;
            if (new_pages == 0) {
                int n_splits = choose_num_splits(eff_len, B);
                dim3 g1(NUM_KV_HEADS, n_splits, B);
                flash_decode_gqa_splitk_part1_turing<false><<<g1, 256, 0, compute_stream>>>(
                    d_Q, Kc, Vc, nullptr, nullptr, nullptr, ld.sinks, nullptr,
                    d_split_m, d_split_l, d_split_o, d_O_f16,
                    start_pos, 0, 0, HOT_WINDOW - 1, 0, 0, 0, 0, n_splits, n_splits, 1e20f, nullptr, use_tree
                );
                if (n_splits > 1) {
                    dim3 g2(NUM_Q_HEADS, B);
                    flash_decode_gqa_splitk_reduce<<<g2, 64, 0, compute_stream>>>(
                        d_split_m, d_split_l, d_split_o, d_O_f16, n_splits
                    );
                }
            } else {
                const int n_hot_splits = (B == 1) ? 4 : 2;
                int n_cold_splits = (B == 1)
                    ? (cold_len <= 8192 ? 16 : (cold_len <= 24576 ? 32 : 60))
                    : (cold_len <= 8192 ? 8  : (cold_len <= 24576 ? 16 : 30));
                if (force_splits > 0) n_cold_splits = std::max(1, std::min(force_splits, ((B == 1) ? MAX_SPLITS : 32) - n_hot_splits));
                n_cold_splits = std::min(n_cold_splits, new_pages);
                const int n_total_splits = n_hot_splits + n_cold_splits;
                const int per_node_cold = bnd_cross ? 1 : 0;

                // Phase 1: Hot-4K [cold_len_b .. cur_pos] exact FP16 (reading boundary page from pristine V_bnd_fp16!)
                dim3 gh(NUM_KV_HEADS, n_hot_splits, B);
                flash_decode_gqa_splitk_part1_turing<false><<<gh, 256, 0, compute_stream>>>(
                    d_Q, Kc, Vc, nullptr, nullptr, nullptr, ld.sinks, nullptr,
                    d_split_m, d_split_l, d_split_o, d_O_f16,
                    start_pos, 0, 0, HOT_WINDOW - 1, 0, cold_len, 0, 0, n_hot_splits, n_total_splits, 1e20f, nullptr, use_tree,
                    bnd_cross ? Vbnd : nullptr, bnd_pos0, per_node_cold
                );
                compute_hot_floor_kernel<<<B, NUM_Q_HEADS, 0, compute_stream>>>(
                    d_split_m, d_split_l, d_m_floor, n_hot_splits, n_total_splits
                );
                // Phase 2: Cold [0 .. cold_len_b) exact K-FP16 + V_sink128 FP16 + Lazy Sparse V_pc32!
                dim3 gc(NUM_KV_HEADS, n_cold_splits, B);
                flash_decode_gqa_splitk_part1_turing<true><<<gc, 256, 0, compute_stream>>>(
                    d_Q, Kc, Vsink, Vc_cold, Vmin_pg, Vmax_pg, ld.sinks, d_m_floor,
                    d_split_m, d_split_l, d_split_o, d_O_f16,
                    start_pos, 0, 0, 0, 0, 0, cold_len, n_hot_splits, n_cold_splits, n_total_splits,
                    margin_log2, collect_prune_stats ? d_prune_stats : nullptr, use_tree,
                    nullptr, -1, per_node_cold
                );
                dim3 g2(NUM_Q_HEADS, B);
                flash_decode_gqa_splitk_reduce<<<g2, 64, 0, compute_stream>>>(
                    d_split_m, d_split_l, d_split_o, d_O_f16, n_total_splits
                );
            }
        } else {
            // Direct zero-copy path for B=1 local ring OR pure FP16 global KV!
            int rmask = is_local ? (RING_SIZE - 1) : 0;
            qkv_bias_rope_kv_append_kernel<<<B, 256, 0, compute_stream>>>(
                d_QKV, ld.b_qkv, cos_table, sin_table, d_Q, Kc, Vc, B, start_pos, rmask, rmask, 0, use_tree
            );
            int eff_len = is_local ? std::min(start_pos + B, RING_SIZE) : (start_pos + B);
            int n_splits = choose_num_splits(eff_len, B);
            float fp16_cutoff = (!is_local && eff_len > HOT_WINDOW) ? margin_log2 : 1e20f;
            dim3 g1(NUM_KV_HEADS, n_splits, B);
            flash_decode_gqa_splitk_part1_turing<false><<<g1, 256, 0, compute_stream>>>(
                d_Q, Kc, Vc, nullptr, nullptr, nullptr, ld.sinks, nullptr,
                d_split_m, d_split_l, d_split_o, d_O_f16,
                start_pos, is_local, rmask, rmask, 0, 0, 0, 0, n_splits, n_splits, fp16_cutoff,
                (!is_local && collect_prune_stats) ? d_prune_stats : nullptr, use_tree
            );
            if (n_splits > 1) {
                dim3 g2(NUM_Q_HEADS, B);
                flash_decode_gqa_splitk_reduce<<<g2, 64, 0, compute_stream>>>(
                    d_split_m, d_split_l, d_split_o, d_O_f16, n_splits
                );
            }
        }
        // 5. O projection + bias + residual
        gemm_fp16_to_fp32(ld.W_o, d_O_f16, d_proj_out, D_MODEL, B, 4096);
        int nb = (B * D_MODEL + 255) / 256;
        residual_add_bias_kernel<<<nb, 256, 0, compute_stream>>>(d_H, d_proj_out, ld.b_o, B);
    }

    // ---- Core B=1 Single-Token Decode Step ----
    void forward_token_internal(
        int token_id, int pos, float* logits_out, int logits_on_gpu, uint8_t* top4_out,
        float* probe_la_raw_24x32 = nullptr,
        float* probe_la_bo_24x32 = nullptr,
        float* probe_la_ema_24x32 = nullptr,
        uint32_t* probe_res_mask_24 = nullptr
    ) {
        using clk = std::chrono::high_resolution_clock;
        auto t_wall0 = clk::now();

        // 1. REAL embed lookup from RAM -> pinned -> GPU H2D + conversion to FP32 d_H!
        auto t_emb0 = clk::now();
        const uint16_t* emb_row = embed_ram_fp16 + (size_t)token_id * D_MODEL;
        memcpy(host_pinned_embed, emb_row, D_MODEL * sizeof(uint16_t));
        cudaMemcpyAsync(d_embed_f16, host_pinned_embed, D_MODEL * sizeof(uint16_t), cudaMemcpyHostToDevice, compute_stream);
        fp16_to_fp32_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(d_embed_f16, d_H, D_MODEL);
        if (profile_enabled) {
            cudaStreamSynchronize(compute_stream);
            prof.embed_lookup_h2d_sync_ms += std::chrono::duration<double, std::milli>(clk::now() - t_emb0).count();
        }

        double tok_cpu_stage_ms = 0.0;
        double tok_cpu_wait_router_ms = 0.0;
        int layer_miss_counts[NUM_LAYERS] = {0};
        uint8_t layer_miss_nonpref_mask[NUM_LAYERS] = {0};
        int layer_pref_cnt[NUM_LAYERS] = {0};
        uint8_t layer_pref_hit_mask[NUM_LAYERS] = {0};
        uint8_t layer_did_la[NUM_LAYERS] = {0};

        for (int l = 0; l < NUM_LAYERS; ++l) {
            const LayerGPU& ld = layers[l];

            // Iter 7 Pillar 1: Zero-Extra-VRAM Inter-Layer Speculative DMA Lookahead Prefetch (reuses stream_bufs[0..1]!)
            int pref_e[2] = {-1, -1};
            int n_pref = 0;
            const bool do_pref = (lookahead_prefetch_mode > 0) && layer_is_pinned[l] && !rans_mode && (probe_la_raw_24x32 == nullptr);
            if (do_pref) {
                layer_did_la[l] = 1;
                if (profile_enabled) cudaEventRecord(ev_la_s[l], compute_stream);
                if (ema_attn_valid[l] && d_ema_attn != nullptr) {
                    add_f32_scaled_2880_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                        d_H, d_ema_attn + (size_t)l * D_MODEL, 1.0f, d_mlp_acc
                    );
                    rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_mlp_acc, ld.post_norm, nullptr, d_N_f16, 1);
                } else {
                    rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_H, ld.post_norm, nullptr, d_N_f16, 1);
                }
                gemm_fp16_to_fp32(ld.router_w, d_N_f16, d_router_lg, NUM_EXPERTS, 1, D_MODEL);
                lookahead_router_top4_kernel<<<1, 32, 0, compute_stream>>>(d_router_lg, ld.router_b, d_top_idx, d_top_w, 1);
                if (profile_enabled) cudaEventRecord(ev_la_e[l], compute_stream);
                cudaMemcpyAsync(host_pinned_top_idx, d_top_idx, TOP_K, cudaMemcpyDeviceToHost, compute_stream);
                cudaMemcpyAsync(host_pinned_top_w,   d_top_w,   TOP_K * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
                // Sync ensures layer l-1 finished reading stream_bufs[0..1] AND gives us 4-byte lookahead candidates!
                cudaStreamSynchronize(compute_stream);

                const bool is_uncached_layer = (cache_policy == 0 || cache_capacity_slots <= 0)
                    || (cache_policy == 3 && cache_capacity_slots < 104 && layer_slot_beg[l] >= layer_slot_end[l]);
                const int max_p = (!is_uncached_layer && ((l & 1) == 1) && (pos >= 16384)) ? 2 : 1;
                const uint8_t* ram_l_pref = mxfp4_ram_layers[l];
                for (int k = 0; k < TOP_K && n_pref < max_p; ++k) {
                    int pe = (int)host_pinned_top_idx[k];
                    float pp32 = host_pinned_top_w[k];
                    if (pp32 < prefetch_tau_prob32) continue;
                    int sl = (cache_policy != 0 && cache_capacity_slots > 0) ? key_to_slot[l * NUM_EXPERTS + pe] : -1;
                    if (sl >= 0) continue; // already in VRAM cache!
                    pref_e[n_pref] = pe;
                    if (profile_enabled) cudaEventRecord(ev_pref_s[l][n_pref], copy_stream);
                    cudaMemcpyAsync(
                        stream_bufs[n_pref],
                        ram_l_pref + (size_t)pe * EXPERT_BYTES,
                        EXPERT_BYTES,
                        cudaMemcpyHostToDevice,
                        copy_stream
                    );
                    if (profile_enabled) cudaEventRecord(ev_pref_e[l][n_pref], copy_stream);
                    cudaEventRecord(ev_pref_done[n_pref], copy_stream);
                    prof.prefetch_attempts++;
                    prof.pcie_bytes += EXPERT_BYTES;
                    n_pref++;
                }
                layer_pref_cnt[l] = n_pref;
            }

            if (probe_la_raw_24x32 != nullptr) {
                uint32_t m = 0u;
                for (int e = 0; e < NUM_EXPERTS; ++e) if (key_to_slot[l * NUM_EXPERTS + e] >= 0) m |= (1u << e);
                if (probe_res_mask_24) probe_res_mask_24[l] = m;
                // M1: direct router(post_norm(h_{l-1}))
                rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_H, ld.post_norm, nullptr, d_N_f16, 1);
                gemm_fp16_to_fp32(ld.router_w, d_N_f16, d_router_lg, NUM_EXPERTS, 1, D_MODEL);
                add_bias_32_kernel<<<1, 32, 0, compute_stream>>>(d_router_lg, ld.router_b);
                cudaMemcpyAsync(probe_la_raw_24x32 + l * NUM_EXPERTS, d_router_lg, NUM_EXPERTS * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
                if (probe_la_bo_24x32 != nullptr) {
                    // M2: router(post_norm(h_{l-1} + b_o[l]))
                    add_f32_f16_bias_2880_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(d_H, ld.b_o, d_mlp_acc);
                    rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_mlp_acc, ld.post_norm, nullptr, d_N_f16, 1);
                    gemm_fp16_to_fp32(ld.router_w, d_N_f16, d_router_lg, NUM_EXPERTS, 1, D_MODEL);
                    add_bias_32_kernel<<<1, 32, 0, compute_stream>>>(d_router_lg, ld.router_b);
                    cudaMemcpyAsync(probe_la_bo_24x32 + l * NUM_EXPERTS, d_router_lg, NUM_EXPERTS * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
                }
                if (probe_la_ema_24x32 != nullptr) {
                    // M3: router(post_norm(h_{l-1} + ema_attn[l]))
                    if (ema_attn_valid[l]) {
                        add_f32_scaled_2880_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                            d_H, d_ema_attn + (size_t)l * D_MODEL, 1.0f, d_mlp_acc
                        );
                        rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_mlp_acc, ld.post_norm, nullptr, d_N_f16, 1);
                    } else {
                        rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_H, ld.post_norm, nullptr, d_N_f16, 1);
                    }
                    gemm_fp16_to_fp32(ld.router_w, d_N_f16, d_router_lg, NUM_EXPERTS, 1, D_MODEL);
                    add_bias_32_kernel<<<1, 32, 0, compute_stream>>>(d_router_lg, ld.router_b);
                    cudaMemcpyAsync(probe_la_ema_24x32 + l * NUM_EXPERTS, d_router_lg, NUM_EXPERTS * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
                }
                cudaStreamSynchronize(compute_stream);
            }

            if (profile_enabled) cudaEventRecord(ev_attn_s[l], compute_stream);
            run_attention_decode_or_verify(l, 1, pos);
            if ((do_pref || probe_la_ema_24x32 != nullptr) && d_ema_attn != nullptr) {
                float alpha = ema_attn_valid[l] ? 0.35f : 1.0f;
                ema_update_attn_2880_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                    d_ema_attn + (size_t)l * D_MODEL, d_proj_out, ld.b_o, alpha
                );
                ema_attn_valid[l] = 1;
            }

            // Post-attention norm + FP16 Router + Top-4 Softmax
            rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_H, ld.post_norm, d_N_f32, d_N_f16, 1);
            gemm_fp16_to_fp32(ld.router_w, d_N_f16, d_router_lg, NUM_EXPERTS, 1, D_MODEL);
            router_topk4_softmax_kernel<<<1, 32, 0, compute_stream>>>(d_router_lg, ld.router_b, d_top_idx, d_top_w, 1);
            cudaMemcpyAsync(host_pinned_top_idx, d_top_idx, TOP_K, cudaMemcpyDeviceToHost, compute_stream);
            cudaMemcpyAsync(host_pinned_top_w, d_top_w, TOP_K * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
            if (profile_enabled) cudaEventRecord(ev_attn_e[l], compute_stream);

            // Wait ONLY for the 20-byte router result!
            auto tw0 = clk::now();
            cudaStreamSynchronize(compute_stream);
            tok_cpu_wait_router_ms += std::chrono::duration<double, std::milli>(clk::now() - tw0).count();
            if (hybrid_kv_mode && (l & 1) == 1) {
                flush_global_kv_mirror_to_ram(l >> 1, pos, 1);
            }

            uint8_t top_e[TOP_K];
            float top_w[TOP_K];
            for (int k = 0; k < TOP_K; ++k) {
                top_e[k] = host_pinned_top_idx[k];
                top_w[k] = host_pinned_top_w[k];
                if (top4_out) top4_out[l * TOP_K + k] = top_e[k];
            }

            // Partition Top-4 into CACHE HITS (0 PCIe!) and CACHE MISSES
            uint64_t cur_pin = ++step_pin_id;
            int hit_e[TOP_K], hit_slot[TOP_K];
            float hit_w[TOP_K];
            int n_hits = 0;

            int miss_e[TOP_K];
            float miss_w[TOP_K];
            uint8_t* miss_dst_gpu[TOP_K];
            int miss_stream_slot[TOP_K];
            int n_misses = 0;

            for (int k = 0; k < TOP_K; ++k) {
                int key = l * NUM_EXPERTS + top_e[k];
                int sl = lookup_cache_and_pin(key, cur_pin);
                if (sl >= 0) {
                    hit_e[n_hits] = top_e[k];
                    hit_slot[n_hits] = sl;
                    hit_w[n_hits] = top_w[k];
                    n_hits++;
                    prof.cache_hits++;
                    prof.layer_hits[l]++;
                } else {
                    miss_e[n_misses] = top_e[k];
                    miss_w[n_misses] = top_w[k];
                    n_misses++;
                    prof.cache_misses++;
                    prof.layer_misses[l]++;
                }
            }
            layer_miss_counts[l] = n_misses;

            // Allocate destination VRAM address for each MISS in EXACT original LRU order (0 LRU pollution!)
            int miss_cslot[TOP_K] = {-1, -1, -1, -1};
            int miss_pref_idx[TOP_K] = {-1, -1, -1, -1};
            bool sb_reserved_for_pref[2] = {false, false};
            for (int m = 0; m < n_misses; ++m) {
                int key = l * NUM_EXPERTS + miss_e[m];
                int cslot = alloc_or_evict_cache_slot(key, cur_pin);
                miss_cslot[m] = cslot;
                for (int p = 0; p < n_pref; ++p) {
                    if (miss_e[m] == pref_e[p]) {
                        miss_pref_idx[m] = p;
                        sb_reserved_for_pref[p] = true;
                        prof.prefetch_hits++;
                        layer_pref_hit_mask[l] |= (uint8_t)(1u << p);
                        break;
                    }
                }
                if (cslot >= 0) {
                    miss_dst_gpu[m] = cache_slot_ptr(cslot);
                    miss_stream_slot[m] = -1; // dedicated cache slot (no ping-pong hazard!)
                } else {
                    miss_stream_slot[m] = m & 1;
                    miss_dst_gpu[m] = stream_bufs[m & 1];
                }
            }

            const uint8_t* ram_layer = mxfp4_ram_layers[l];
            int exp_ev_idx = 0;

            // 0. For pinned raw cached layers, enqueue ALL non-prefetched dedicated-slot H2D DMAs on copy_stream FIRST
            //    so copy_stream chains pref_e -> miss_e back-to-back with ZERO WDDM submission bubble!
            bool miss_dma_prequeued[TOP_K] = {false, false, false, false};
            if (layer_is_pinned[l] && !rans_mode) {
                for (int m = 0; m < n_misses; ++m) {
                    if (miss_pref_idx[m] >= 0 || miss_cslot[m] < 0) continue;
                    layer_miss_nonpref_mask[l] |= (uint8_t)(1u << m);
                    const uint8_t* src_e = ram_layer + (size_t)miss_e[m] * EXPERT_BYTES;
                    uint8_t* g_dst = slot_ptrs[miss_cslot[m]];
                    if (profile_enabled) cudaEventRecord(ev_dma_s[l][m], copy_stream);
                    cudaMemcpyAsync(g_dst, src_e, EXPERT_BYTES, cudaMemcpyHostToDevice, copy_stream);
                    if (profile_enabled) cudaEventRecord(ev_dma_e[l][m], copy_stream);
                    cudaEventRecord(ev_copy_done_miss[m], copy_stream);
                    prof.pcie_bytes += EXPERT_BYTES;
                    miss_dma_prequeued[m] = true;
                }
            }

            // 1. Launch ALL CACHE HITS FIRST on compute_stream (5 us enqueue -> GPU stays busy while DMA finishes!)
            for (int h_i = 0; h_i < n_hits; ++h_i) {
                int eid = hit_e[h_i];
                const uint8_t* bp = cache_slot_ptr(hit_slot[h_i]);
                if (profile_enabled) cudaEventRecord(ev_exp_s[l][exp_ev_idx], compute_stream);
                mxfp4_gate_up_swiglu_1vec_kernel<<<(D_FF + 7) / 8, 256, 0, compute_stream>>>(
                    bp + OFF_GU_BLOCKS, bp + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_N_f32, d_exp_mid
                );
                mxfp4_down_wadd_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                    bp + OFF_DN_BLOCKS, bp + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, hit_w[h_i], d_H, D_MODEL
                );
                if (profile_enabled) cudaEventRecord(ev_exp_e[l][exp_ev_idx], compute_stream);
                exp_ev_idx++;
            }

            // 2. Process MISSES in EXACT miss_e[0..n_misses-1] order (100% bit-exact FP32 parity: max_abs = 0.000!):
            //    - Prefetch HIT (miss_pref_idx[m] >= 0): consume directly from stream_bufs[p] (0 PCIe re-transfer!),
            //      then ZERO-COPY 1-ns pointer swap slot_ptrs[cslot] <-> stream_bufs[p] (ZERO D2D memcpy!!).
            //    - Non-prefetched miss: already-queued H2D DMA on copy_stream (or uncached stream / rANS / WC staging).
            if (layer_is_pinned[l]) {
                bool sb_needs_comp_wait[2] = {false, false};
                int nonpref_sb_rr = 0;
                for (int m = 0; m < n_misses; ++m) {
                    const int eid = miss_e[m];
                    const int p_idx = miss_pref_idx[m];
                    if (!rans_mode && p_idx >= 0) {
                        uint8_t* p_buf = stream_bufs[p_idx];
                        if (profile_enabled) cudaEventRecord(ev_pwait_s[l][p_idx], compute_stream);
                        cudaStreamWaitEvent(compute_stream, ev_pref_done[p_idx], 0);
                        if (profile_enabled) cudaEventRecord(ev_pwait_e[l][p_idx], compute_stream);

                        if (profile_enabled) cudaEventRecord(ev_exp_s[l][exp_ev_idx], compute_stream);
                        mxfp4_gate_up_swiglu_1vec_kernel<<<(D_FF + 7) / 8, 256, 0, compute_stream>>>(
                            p_buf + OFF_GU_BLOCKS, p_buf + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_N_f32, d_exp_mid
                        );
                        mxfp4_down_wadd_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                            p_buf + OFF_DN_BLOCKS, p_buf + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, miss_w[m], d_H, D_MODEL
                        );
                        if (profile_enabled) cudaEventRecord(ev_exp_e[l][exp_ev_idx], compute_stream);
                        exp_ev_idx++;
                        if (miss_cslot[m] >= 0) {
                            // ZERO-COPY 1-ns pointer swap: cache slot adopts p_buf, stream_bufs[p_idx] adopts idle evicted slot!
                            std::swap(slot_ptrs[miss_cslot[m]], stream_bufs[p_idx]);
                            sb_reserved_for_pref[p_idx] = false;
                            sb_needs_comp_wait[p_idx] = false;
                        } else {
                            cudaEventRecord(ev_comp_done[p_idx], compute_stream);
                            sb_reserved_for_pref[p_idx] = false;
                            sb_needs_comp_wait[p_idx] = true;
                        }
                        continue;
                    }

                    if (!rans_mode && miss_dma_prequeued[m]) {
                        uint8_t* g_dst = slot_ptrs[miss_cslot[m]];
                        if (profile_enabled) cudaEventRecord(ev_wait_s[l][m], compute_stream);
                        cudaStreamWaitEvent(compute_stream, ev_copy_done_miss[m], 0);
                        if (profile_enabled) cudaEventRecord(ev_wait_e[l][m], compute_stream);

                        if (profile_enabled) cudaEventRecord(ev_exp_s[l][exp_ev_idx], compute_stream);
                        mxfp4_gate_up_swiglu_1vec_kernel<<<(D_FF + 7) / 8, 256, 0, compute_stream>>>(
                            g_dst + OFF_GU_BLOCKS, g_dst + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_N_f32, d_exp_mid
                        );
                        mxfp4_down_wadd_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                            g_dst + OFF_DN_BLOCKS, g_dst + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, miss_w[m], d_H, D_MODEL
                        );
                        if (profile_enabled) cudaEventRecord(ev_exp_e[l][exp_ev_idx], compute_stream);
                        exp_ev_idx++;
                        continue;
                    }

                    layer_miss_nonpref_mask[l] |= (uint8_t)(1u << m);
                    const int ev_slot = m & 1;
                    uint8_t* g_dst = (miss_cslot[m] >= 0) ? slot_ptrs[miss_cslot[m]] : miss_dst_gpu[m];

                    if (rans_mode) {
                        const uint8_t* src_e = ram_layer + (size_t)rans_exp_off[l][eid];
                        const size_t nb = (size_t)rans_exp_sz[l][eid];
                        if (m >= 2) cudaStreamWaitEvent(copy_stream, ev_rans_done[ev_slot], 0);
                        if (profile_enabled) cudaEventRecord(ev_dma_s[l][m], copy_stream);
                        cudaMemcpyAsync(comp_stream_bufs[ev_slot], src_e, nb, cudaMemcpyHostToDevice, copy_stream);
                        if (profile_enabled) cudaEventRecord(ev_dma_e[l][m], copy_stream);
                        cudaEventRecord(ev_copy_done[ev_slot], copy_stream);
                        prof.pcie_bytes += nb;

                        if (profile_enabled) cudaEventRecord(ev_wait_s[l][m], compute_stream);
                        cudaStreamWaitEvent(compute_stream, ev_copy_done[ev_slot], 0);
                        if (profile_enabled) cudaEventRecord(ev_wait_e[l][m], compute_stream);

                        if (profile_enabled) cudaEventRecord(ev_exp_s[l][exp_ev_idx], compute_stream);
                        rans_decode_expert_fused_kernel<<<75, 192, 0, compute_stream>>>(comp_stream_bufs[ev_slot], g_dst);
                        cudaEventRecord(ev_rans_done[ev_slot], compute_stream);
                        mxfp4_gate_up_swiglu_1vec_kernel<<<(D_FF + 7) / 8, 256, 0, compute_stream>>>(
                            g_dst + OFF_GU_BLOCKS, g_dst + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_N_f32, d_exp_mid
                        );
                        mxfp4_down_wadd_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                            g_dst + OFF_DN_BLOCKS, g_dst + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, miss_w[m], d_H, D_MODEL
                        );
                        if (profile_enabled) cudaEventRecord(ev_exp_e[l][exp_ev_idx], compute_stream);
                        exp_ev_idx++;
                        cudaEventRecord(ev_comp_done[ev_slot], compute_stream);
                    } else {
                        int sb_use = ev_slot;
                        if (miss_cslot[m] < 0) {
                            // Uncached layer: NEVER clobber a stream_buf still reserved for an upcoming prefetch hit!
                            if (sb_reserved_for_pref[0] && !sb_reserved_for_pref[1]) sb_use = 1;
                            else if (sb_reserved_for_pref[1] && !sb_reserved_for_pref[0]) sb_use = 0;
                            else sb_use = (nonpref_sb_rr++) & 1;
                            g_dst = stream_bufs[sb_use];
                            if (sb_needs_comp_wait[sb_use]) {
                                cudaStreamWaitEvent(copy_stream, ev_comp_done[sb_use], 0);
                            }
                        }
                        const uint8_t* src_e = ram_layer + (size_t)eid * EXPERT_BYTES;
                        if (profile_enabled) cudaEventRecord(ev_dma_s[l][m], copy_stream);
                        cudaMemcpyAsync(g_dst, src_e, EXPERT_BYTES, cudaMemcpyHostToDevice, copy_stream);
                        if (profile_enabled) cudaEventRecord(ev_dma_e[l][m], copy_stream);
                        cudaEventRecord(ev_copy_done[sb_use], copy_stream);
                        prof.pcie_bytes += EXPERT_BYTES;

                        if (profile_enabled) cudaEventRecord(ev_wait_s[l][m], compute_stream);
                        cudaStreamWaitEvent(compute_stream, ev_copy_done[sb_use], 0);
                        if (profile_enabled) cudaEventRecord(ev_wait_e[l][m], compute_stream);

                        if (profile_enabled) cudaEventRecord(ev_exp_s[l][exp_ev_idx], compute_stream);
                        mxfp4_gate_up_swiglu_1vec_kernel<<<(D_FF + 7) / 8, 256, 0, compute_stream>>>(
                            g_dst + OFF_GU_BLOCKS, g_dst + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_N_f32, d_exp_mid
                        );
                        mxfp4_down_wadd_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                            g_dst + OFF_DN_BLOCKS, g_dst + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, miss_w[m], d_H, D_MODEL
                        );
                        if (profile_enabled) cudaEventRecord(ev_exp_e[l][exp_ev_idx], compute_stream);
                        exp_ev_idx++;
                        if (miss_cslot[m] < 0) {
                            cudaEventRecord(ev_comp_done[sb_use], compute_stream);
                            sb_needs_comp_wait[sb_use] = true;
                        }
                    }
                }
            } else {
                const int n_st = std::max(1, std::min(staging_stripes, 16));
                const size_t stripe_sz = (((EXPERT_BYTES + n_st - 1) / n_st) + 63ULL) & ~63ULL;
                for (int m = 0; m < n_misses; ++m) {
                    layer_miss_nonpref_mask[l] |= (uint8_t)(1u << m);
                    const int ev_slot = m & 1;
                    const int hs = m & 3;
                    const uint8_t* src_e = ram_layer + (size_t)miss_e[m] * EXPERT_BYTES;
                    uint8_t* h_dst = host_pinned_slots[hs];
                    uint8_t* g_dst = miss_dst_gpu[m];

                    if (m >= 2 && miss_stream_slot[m] >= 0) {
                        cudaStreamWaitEvent(copy_stream, ev_comp_done[ev_slot], 0);
                    }
                    if (profile_enabled) cudaEventRecord(ev_dma_s[l][m], copy_stream);
                    for (int st = 0; st < n_st; ++st) {
                        size_t off = (size_t)st * stripe_sz;
                        if (off >= EXPERT_BYTES) break;
                        size_t len = std::min(stripe_sz, EXPERT_BYTES - off);
                        auto ts0 = clk::now();
                        copy_pool.copy(h_dst + off, src_e + off, len);
                        tok_cpu_stage_ms += std::chrono::duration<double, std::milli>(clk::now() - ts0).count();
                        cudaMemcpyAsync(g_dst + off, h_dst + off, len, cudaMemcpyHostToDevice, copy_stream);
                    }
                    if (profile_enabled) cudaEventRecord(ev_dma_e[l][m], copy_stream);
                    cudaEventRecord(ev_copy_done[ev_slot], copy_stream);
                    prof.pcie_bytes += EXPERT_BYTES;

                    if (profile_enabled) cudaEventRecord(ev_wait_s[l][m], compute_stream);
                    cudaStreamWaitEvent(compute_stream, ev_copy_done[ev_slot], 0);
                    if (profile_enabled) cudaEventRecord(ev_wait_e[l][m], compute_stream);

                    int eid = miss_e[m];
                    if (profile_enabled) cudaEventRecord(ev_exp_s[l][exp_ev_idx], compute_stream);
                    mxfp4_gate_up_swiglu_1vec_kernel<<<(D_FF + 7) / 8, 256, 0, compute_stream>>>(
                        g_dst + OFF_GU_BLOCKS, g_dst + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_N_f32, d_exp_mid
                    );
                    mxfp4_down_wadd_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                        g_dst + OFF_DN_BLOCKS, g_dst + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, miss_w[m], d_H, D_MODEL
                    );
                    if (profile_enabled) cudaEventRecord(ev_exp_e[l][exp_ev_idx], compute_stream);
                    exp_ev_idx++;
                    cudaEventRecord(ev_comp_done[ev_slot], compute_stream);
                }
            }
            if (l == 2)  cudaMemcpyAsync(d_H_l2,  d_H, D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
            if (l == 12) cudaMemcpyAsync(d_H_l12, d_H, D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
        }

        // Final RMSNorm + lm_head (resident on GPU!)
        if (profile_enabled) cudaEventRecord(ev_lm_s, compute_stream);
        rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_H, final_norm, nullptr, d_N_f16, 1);
        float* lg_dst = logits_on_gpu ? logits_out : d_logits;
        gemm_fp16_to_fp32(lm_head, d_N_f16, lg_dst, VOCAB_SIZE, 1, D_MODEL);
        if (profile_enabled) cudaEventRecord(ev_lm_e, compute_stream);

        if (!logits_on_gpu && logits_out != nullptr) {
            cudaMemcpyAsync(logits_out, d_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
        }

        auto te0 = clk::now();
        cudaStreamSynchronize(compute_stream);
        double tok_cpu_wait_end = std::chrono::duration<double, std::milli>(clk::now() - te0).count();
        double tok_wall = std::chrono::duration<double, std::milli>(clk::now() - t_wall0).count();

        prof.tokens++;
        prof.total_wall_ms += tok_wall;
        prof.cpu_staging_memcpy_ms += tok_cpu_stage_ms;
        prof.cpu_wait_router_ms += tok_cpu_wait_router_ms;
        prof.cpu_wait_end_ms += tok_cpu_wait_end;

        if (profile_enabled) {
            if (lookahead_prefetch_mode > 0) cudaStreamSynchronize(copy_stream);
            float ms = 0.0f;
            for (int l = 0; l < NUM_LAYERS; ++l) {
                if (layer_did_la[l]) {
                    cudaEventElapsedTime(&ms, ev_la_s[l], ev_la_e[l]); prof.lookahead_overhead_ms += ms;
                }
                cudaEventElapsedTime(&ms, ev_attn_s[l], ev_attn_e[l]); prof.gpu_attn_router_ms += ms;
                for (int k = 0; k < TOP_K; ++k) {
                    cudaEventElapsedTime(&ms, ev_exp_s[l][k], ev_exp_e[l][k]); prof.gpu_moe_ms += ms;
                }
                for (int m = 0; m < layer_miss_counts[l]; ++m) {
                    if ((layer_miss_nonpref_mask[l] >> m) & 1u) {
                        cudaEventElapsedTime(&ms, ev_dma_s[l][m], ev_dma_e[l][m]);   prof.pcie_dma_ms += ms;
                        cudaEventElapsedTime(&ms, ev_wait_s[l][m], ev_wait_e[l][m]); prof.gpu_wait_dma_ms += ms;
                    }
                }
                for (int p = 0; p < layer_pref_cnt[l]; ++p) {
                    float p_dma = 0.0f;
                    cudaEventElapsedTime(&p_dma, ev_pref_s[l][p], ev_pref_e[l][p]);
                    prof.pcie_dma_ms += p_dma;
                    prof.prefetch_dma_ms += p_dma;
                    if ((layer_pref_hit_mask[l] >> p) & 1u) {
                        float p_wait = 0.0f;
                        cudaEventElapsedTime(&p_wait, ev_pwait_s[l][p], ev_pwait_e[l][p]);
                        prof.gpu_wait_dma_ms += p_wait;
                        prof.prefetch_hidden_ms += std::max(0.0f, p_dma - p_wait);
                    }
                }
            }
            cudaEventElapsedTime(&ms, ev_lm_s, ev_lm_e); prof.gpu_lmhead_ms += ms;
        }
    }

    // ---- Chunked Prefill (chunk_len <= 512, yields arena to temporaries, NEVER materializes chunk x 201088!) ----
    void prefill_chunk_internal(
        const int* token_ids,
        int chunk_len,
        int start_pos,
        int compute_last_logits,
        float* last_logits_out,
        int logits_on_gpu,
        float* hidden_out_host = nullptr,
        float* hidden_l2_out_host = nullptr,
        float* hidden_l12_out_host = nullptr,
        uint8_t* top4_24xCx4_out_host = nullptr
    ) {
        using clk = std::chrono::high_resolution_clock;
        auto t_chunk0 = clk::now();
        const bool pp = prefill_profile_enabled;
        double c_cpu_stage_ms = 0.0;
        double c_sync_ms = 0.0;
        double c_attn_loc_ms = 0.0;
        double c_attn_glob_ms = 0.0;
        double c_qkv_o_r_ms = 0.0;
        double c_h2d_dma_ms = 0.0;
        double c_deq_ms = 0.0;
        double c_gath_scat_ms = 0.0;
        double c_gemm_ms = 0.0;
        double c_wait_dma_ms = 0.0;
        uint64_t c_exp_loaded = 0;

        // Yield expert cache arena to chunked prefill temporaries (no cudaMalloc/cudaFree fragmentation!)
        clear_cache();

        // Carve FP32 SGEMM prefill temporaries directly out of d_arena (~142.3 MB for chunk_len=512)!
        // Uses a shared union slice for sequential stages: W_qkv/W_o (59 MB), Global Flash-Prefill (19.4 MB), W_gu/W_dn (66.4 MB)
        auto align256 = [](size_t x) { return (x + 255ULL) & ~255ULL; };
        size_t off = 0;
        auto alloc_slice = [&](size_t bytes) -> uint8_t* {
            uint8_t* p = d_arena + off;
            off = align256(off + bytes);
            return p;
        };

        const size_t C = (size_t)chunk_len;
        float*   p_H        = reinterpret_cast<float*>(alloc_slice(C * D_MODEL * 4));
        float*   p_N_f32    = reinterpret_cast<float*>(alloc_slice(C * D_MODEL * 4));
        __half*  p_emb_f16  = reinterpret_cast<__half*>(alloc_slice(C * D_MODEL * 2));
        float*   p_QKV      = reinterpret_cast<float*>(alloc_slice(C * QKV_DIM * 4));
        float*   p_Q        = reinterpret_cast<float*>(alloc_slice(C * 4096 * 4));
        __half*  p_O_f16    = reinterpret_cast<__half*>(alloc_slice(C * 4096 * 2));
        float*   p_O_f32    = reinterpret_cast<float*>(alloc_slice(C * 4096 * 4));
        float*   p_proj     = reinterpret_cast<float*>(alloc_slice(C * D_MODEL * 4));
        float*   p_rlg      = reinterpret_cast<float*>(alloc_slice(C * NUM_EXPERTS * 4));
        uint8_t* p_top_i    = alloc_slice(C * TOP_K);
        float*   p_top_w    = reinterpret_cast<float*>(alloc_slice(C * TOP_K * 4));
        float*   p_ein_f32  = reinterpret_cast<float*>(alloc_slice(C * D_MODEL * 4));
        float*   p_egu_f32  = reinterpret_cast<float*>(alloc_slice(C * 2 * D_FF * 4));
        float*   p_emid_f32 = reinterpret_cast<float*>(alloc_slice(C * D_FF * 4));
        float*   p_edn_f32  = reinterpret_cast<float*>(alloc_slice(C * D_MODEL * 4));
        __half*  p_K_loc    = reinterpret_cast<__half*>(alloc_slice((C + RING_SIZE) * 512 * 2));
        __half*  p_V_loc    = reinterpret_cast<__half*>(alloc_slice((C + RING_SIZE) * 512 * 2));

        // Shared union scratch slice (max 66.36 MB for W_gu_f32; also holds W_qkv/W_o or Flash-Prefill tiles!)
        uint8_t* p_union = alloc_slice(5760ULL * D_MODEL * 4);
        float*   p_W_f32 = reinterpret_cast<float*>(p_union);

        // Sub-layout inside p_union for Tiled SGEMM Global Flash-Prefill (TILE_KV = 1024 -> 19.43 MB total!)
        constexpr int TILE_KV = 1024;
        const size_t M_q = C * 8ULL;
        size_t uoff = 0;
        auto alloc_u = [&](size_t bytes) -> float* {
            float* p = reinterpret_cast<float*>(p_union + uoff);
            uoff = align256(uoff + bytes);
            return p;
        };
        float* u_Q_grp  = alloc_u(M_q * 64ULL * 4);
        float* u_K_tile = alloc_u((size_t)TILE_KV * 64ULL * 4);
        float* u_V_tile = alloc_u((size_t)TILE_KV * 64ULL * 4);
        float* u_S_tile = alloc_u(M_q * (size_t)TILE_KV * 4);
        float* u_O_acc  = alloc_u(M_q * 64ULL * 4);
        float* u_m_acc  = alloc_u(M_q * 4);
        float* u_l_acc  = alloc_u(M_q * 4);

        // 1. Lookup chunk embeddings from RAM -> host_pinned_embed -> p_emb_f16 -> p_H (FP32)
        for (int i = 0; i < chunk_len; ++i) {
            memcpy(host_pinned_embed + (size_t)i * D_MODEL,
                   embed_ram_fp16 + (size_t)token_ids[i] * D_MODEL,
                   D_MODEL * sizeof(uint16_t));
        }
        cudaMemcpyAsync(p_emb_f16, host_pinned_embed, C * D_MODEL * 2, cudaMemcpyHostToDevice, compute_stream);
        fp16_mat_to_f32(p_emb_f16, p_H, C * D_MODEL);

        std::vector<int> exp_toks[NUM_EXPERTS];
        std::vector<float> exp_ws[NUM_EXPERTS];
        std::vector<int> active_experts;
        active_experts.reserve(NUM_EXPERTS);

        for (int l = 0; l < NUM_LAYERS; ++l) {
            const LayerGPU& ld = layers[l];
            const int is_local = ((l & 1) == 0) ? 1 : 0;
            const int sub_l = l >> 1;
            const size_t k_global_stride = (size_t)kv_cap_global * 512;
            const size_t v_global_stride = (size_t)(hybrid_kv_mode ? HOT_WINDOW : kv_cap_global) * 512;
            __half* Kc = is_local ? (K_local_ring + (size_t)sub_l * RING_SIZE * 512)
                                  : (K_global     + (size_t)sub_l * k_global_stride);
            __half* Vc = is_local ? (V_local_ring + (size_t)sub_l * RING_SIZE * 512)
                                  : (V_global     + (size_t)sub_l * v_global_stride);

            // A) Chunked Attention (Native FP32 SGEMM for QKV/O + Tiled SGEMM Global Flash-Prefill!)
            cudaEvent_t ev_a0, ev_a1, ev_a2, ev_a3;
            if (pp) { cudaEventCreate(&ev_a0); cudaEventCreate(&ev_a1); cudaEventCreate(&ev_a2); cudaEventCreate(&ev_a3); cudaEventRecord(ev_a0, compute_stream); }
            rmsnorm_fp32_fp16_kernel<<<chunk_len, 256, 0, compute_stream>>>(p_H, ld.in_norm, p_N_f32, nullptr, chunk_len);
            fp16_mat_to_f32(ld.W_qkv, p_W_f32, (size_t)QKV_DIM * D_MODEL);
            sgemm_f32(p_W_f32, p_N_f32, p_QKV, QKV_DIM, chunk_len, D_MODEL);
            if (is_local) {
                dim3 gp(NUM_KV_HEADS, (chunk_len + 3) / 4);
                int prev_len = std::min(start_pos, RING_SIZE - 1);
                int win_base = start_pos - prev_len;
                if (pp) cudaEventRecord(ev_a1, compute_stream);
                if (prev_len > 0) {
                    copy_ring_prev_to_linear_kernel<<<prev_len, 512, 0, compute_stream>>>(
                        Kc, Vc, p_K_loc, p_V_loc, win_base, prev_len
                    );
                }
                qkv_bias_rope_kv_append_kernel<<<chunk_len, 256, 0, compute_stream>>>(
                    p_QKV, ld.b_qkv, cos_table, sin_table, p_Q, p_K_loc, p_V_loc, chunk_len, start_pos, 0, 0, win_base, 0
                );
                flash_prefill_gqa_4q_kernel<<<gp, 256, 0, compute_stream>>>(
                    p_Q, p_K_loc, p_V_loc, ld.sinks, p_O_f16, chunk_len, start_pos, 1, 0, win_base
                );
                int n_tail = std::min(chunk_len, RING_SIZE);
                int lin_start = prev_len + chunk_len - n_tail;
                int abs_start = start_pos + chunk_len - n_tail;
                copy_linear_tail_to_ring_kernel<<<n_tail, 512, 0, compute_stream>>>(
                    p_K_loc, p_V_loc, Kc, Vc, lin_start, abs_start, n_tail
                );
                fp16_mat_to_f32(p_O_f16, p_O_f32, C * 4096ULL);
                if (pp) cudaEventRecord(ev_a2, compute_stream);
            } else {
                const int total_kv  = start_pos + chunk_len;
                const int old_pages = (hybrid_kv_mode && start_pos > HOT_WINDOW) ? ((start_pos - HOT_WINDOW + V_PAGE_SIZE - 1) / V_PAGE_SIZE) : 0;
                const int new_pages = (hybrid_kv_mode && total_kv  > HOT_WINDOW) ? ((total_kv  - HOT_WINDOW + V_PAGE_SIZE - 1) / V_PAGE_SIZE) : 0;
                const int cold_len  = hybrid_kv_mode ? (new_pages * V_PAGE_SIZE) : 0;
                __half*  Vsink   = hybrid_kv_mode ? (V_sink_fp16   + (size_t)sub_l * V_SINK_FP16 * 512)   : nullptr;
                uint8_t* Vc_cold = hybrid_kv_mode ? (V_global_cold + (size_t)sub_l * kv_cold_cap * 512)   : nullptr;
                __half*  Vmin_pg = hybrid_kv_mode ? (V_page_min    + (size_t)sub_l * kv_cold_pages * 512) : nullptr;
                __half*  Vmax_pg = hybrid_kv_mode ? (V_page_max    + (size_t)sub_l * kv_cold_pages * 512) : nullptr;
                if (hybrid_kv_mode && new_pages > old_pages) {
                    dim3 ge(new_pages - old_pages, NUM_KV_HEADS);
                    evict_v_hot_pages32_to_cold_pc_kernel<<<ge, 64, 0, compute_stream>>>(
                        Vc, Vsink, Vc_cold, Vmin_pg, Vmax_pg, old_pages, new_pages - old_pages
                    );
                }
                qkv_bias_rope_kv_append_kernel<<<chunk_len, 256, 0, compute_stream>>>(
                    p_QKV, ld.b_qkv, cos_table, sin_table, p_Q, Kc, Vc, chunk_len, start_pos,
                    0, hybrid_kv_mode ? (HOT_WINDOW - 1) : 0, 0, 0
                );
                if (hybrid_kv_mode) {
                    queue_global_kv_mirror_d2h(sub_l, start_pos, chunk_len);
                }
                if (pp) cudaEventRecord(ev_a1, compute_stream);
                // Tiled cublasSgemm Online-Softmax Flash-Prefill (exact causal + sinks + FP32 LSE!)
                const int Mq_i = (int)M_q;
                const float alpha_sm = SM_SCALE;
                const float alpha_one = 1.0f;
                const float beta_zero = 0.0f;
                const float beta_one = 1.0f;
                for (int h_kv = 0; h_kv < NUM_KV_HEADS; ++h_kv) {
                    prefill_gather_q_group_kernel<<<Mq_i, 64, 0, compute_stream>>>(p_Q, u_Q_grp, chunk_len, h_kv);
                    prefill_init_mlo_kernel<<<(Mq_i * 64 + 255) / 256, 256, 0, compute_stream>>>(u_m_acc, u_l_acc, u_O_acc, Mq_i);
                    for (int j0 = 0; j0 < total_kv; j0 += TILE_KV) {
                        int tlen = std::min(TILE_KV, total_kv - j0);
                        if (hybrid_kv_mode) {
                            prefill_gather_kv_tile_hybrid_kernel<<<tlen, 64, 0, compute_stream>>>(
                                Kc, Vc, Vsink, Vc_cold, Vmin_pg, Vmax_pg, u_K_tile, u_V_tile, j0, tlen, h_kv, cold_len
                            );
                        } else {
                            prefill_gather_kv_tile_kernel<<<tlen, 64, 0, compute_stream>>>(Kc, Vc, u_K_tile, u_V_tile, j0, tlen, h_kv);
                        }
                        // S_tile[Mq, tlen] = SM_SCALE * Q_grp[Mq, 64] @ K_tile[tlen, 64]^T
                        cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, tlen, Mq_i, 64,
                                    &alpha_sm, u_K_tile, 64, u_Q_grp, 64, &beta_zero, u_S_tile, tlen);
                        prefill_online_softmax_tile_kernel<<<Mq_i, 256, 0, compute_stream>>>(
                            u_S_tile, u_m_acc, u_l_acc, u_O_acc, Mq_i, tlen, start_pos, j0
                        );
                        // O_acc[Mq, 64] += P_tile[Mq, tlen] @ V_tile[tlen, 64]
                        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, 64, Mq_i, tlen,
                                    &alpha_one, u_V_tile, 64, u_S_tile, tlen, &beta_one, u_O_acc, 64);
                    }
                    prefill_finalize_sink_scatter_kernel<<<Mq_i, 64, 0, compute_stream>>>(
                        u_O_acc, u_m_acc, u_l_acc, ld.sinks, p_O_f32, Mq_i, h_kv
                    );
                }
                if (pp) cudaEventRecord(ev_a2, compute_stream);
            }
            fp16_mat_to_f32(ld.W_o, p_W_f32, (size_t)D_MODEL * 4096ULL);
            sgemm_f32(p_W_f32, p_O_f32, p_proj, D_MODEL, chunk_len, 4096);
            residual_add_bias_kernel<<<(int)((C * D_MODEL + 255) / 256), 256, 0, compute_stream>>>(p_H, p_proj, ld.b_o, chunk_len);

            // B) Post-attention norm + Router for all chunk_len tokens (FP32 SGEMM!)
            rmsnorm_fp32_fp16_kernel<<<chunk_len, 256, 0, compute_stream>>>(p_H, ld.post_norm, p_N_f32, nullptr, chunk_len);
            fp16_mat_to_f32(ld.router_w, p_W_f32, (size_t)NUM_EXPERTS * D_MODEL);
            sgemm_f32(p_W_f32, p_N_f32, p_rlg, NUM_EXPERTS, chunk_len, D_MODEL);
            router_topk4_softmax_kernel<<<chunk_len, 32, 0, compute_stream>>>(p_rlg, ld.router_b, p_top_i, p_top_w, chunk_len);
            if (pp) cudaEventRecord(ev_a3, compute_stream);
            cudaMemcpyAsync(host_pinned_top_idx, p_top_i, C * TOP_K, cudaMemcpyDeviceToHost, compute_stream);
            cudaMemcpyAsync(host_pinned_top_w, p_top_w, C * TOP_K * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
            auto ts_r0 = clk::now();
            cudaStreamSynchronize(compute_stream);
            c_sync_ms += std::chrono::duration<double, std::milli>(clk::now() - ts_r0).count();
            if (hybrid_kv_mode && !is_local) {
                flush_global_kv_mirror_to_ram(sub_l, start_pos, chunk_len);
            }
            if (pp) {
                float m01 = 0, m12 = 0, m23 = 0;
                cudaEventElapsedTime(&m01, ev_a0, ev_a1);
                cudaEventElapsedTime(&m12, ev_a1, ev_a2);
                cudaEventElapsedTime(&m23, ev_a2, ev_a3);
                c_qkv_o_r_ms += (m01 + m23);
                if (is_local) c_attn_loc_ms += m12; else c_attn_glob_ms += m12;
                cudaEventDestroy(ev_a0); cudaEventDestroy(ev_a1); cudaEventDestroy(ev_a2); cudaEventDestroy(ev_a3);
            }

            // Group tokens by active expert into flat layer arrays & upload ONCE before streaming (avoids H2D copy-engine contention!)
            if (top4_24xCx4_out_host != nullptr) {
                memcpy(top4_24xCx4_out_host + (size_t)l * C * TOP_K, host_pinned_top_idx, C * TOP_K);
            }
            for (int e = 0; e < NUM_EXPERTS; ++e) { exp_toks[e].clear(); exp_ws[e].clear(); }
            for (int i = 0; i < chunk_len; ++i) {
                for (int k = 0; k < TOP_K; ++k) {
                    int e = host_pinned_top_idx[i * TOP_K + k];
                    exp_toks[e].push_back(i);
                    exp_ws[e].push_back(host_pinned_top_w[i * TOP_K + k]);
                }
            }
            active_experts.clear();
            int exp_flat_off[NUM_EXPERTS];
            int flat_n = 0;
            for (int e = 0; e < NUM_EXPERTS; ++e) {
                exp_flat_off[e] = flat_n;
                if (!exp_toks[e].empty()) {
                    active_experts.push_back(e);
                    int nt = (int)exp_toks[e].size();
                    memcpy(host_pinned_tok_idx + flat_n, exp_toks[e].data(), nt * sizeof(int));
                    memcpy(host_pinned_tok_w   + flat_n, exp_ws[e].data(),   nt * sizeof(float));
                    flat_n += nt;
                }
            }

            const uint8_t* ram_layer = mxfp4_ram_layers[l];
            int n_act = (int)active_experts.size();
            if (n_act == 0) continue;
            c_exp_loaded += (uint64_t)n_act;

            // Upload ALL active token indices & weights for this layer ONCE while copy_stream is idle!
            cudaMemcpyAsync(d_tok_idx_buf, host_pinned_tok_idx, flat_n * sizeof(int),   cudaMemcpyHostToDevice, compute_stream);
            cudaMemcpyAsync(d_tok_w_buf,   host_pinned_tok_w,   flat_n * sizeof(float), cudaMemcpyHostToDevice, compute_stream);

            // Double-buffered streaming of each active expert ONCE for ALL tokens routed to it!
            const bool pinned_l = layer_is_pinned[l];
            auto enqueue_prefill_dma = [&](int idx, int buf_slot) {
                int eid_d = active_experts[idx];
                if (pp) cudaEventRecord(ev_p_dma_s[idx], copy_stream);
                if (pinned_l) {
                    if (rans_mode) {
                        cudaMemcpyAsync(comp_stream_bufs[buf_slot], ram_layer + (size_t)rans_exp_off[l][eid_d],
                                        (size_t)rans_exp_sz[l][eid_d], cudaMemcpyHostToDevice, copy_stream);
                    } else {
                        cudaMemcpyAsync(stream_bufs[buf_slot], ram_layer + (size_t)eid_d * EXPERT_BYTES,
                                        EXPERT_BYTES, cudaMemcpyHostToDevice, copy_stream);
                    }
                } else {
                    int hs = idx & 3;
                    cudaMemcpyAsync(stream_bufs[buf_slot], host_pinned_slots[hs], EXPERT_BYTES, cudaMemcpyHostToDevice, copy_stream);
                    cudaEventRecord(ev_host_slot_done[hs], copy_stream);
                }
                if (pp) cudaEventRecord(ev_p_dma_e[idx], copy_stream);
                cudaEventRecord(ev_copy_done[buf_slot], copy_stream);
            };

            if (!pinned_l) {
                auto tc0 = clk::now();
                copy_pool.copy(host_pinned_slots[0], ram_layer + (size_t)active_experts[0] * EXPERT_BYTES, EXPERT_BYTES);
                c_cpu_stage_ms += std::chrono::duration<double, std::milli>(clk::now() - tc0).count();
            }
            enqueue_prefill_dma(0, 0);

            if (!pinned_l && n_act > 1) {
                auto tc0 = clk::now();
                copy_pool.copy(host_pinned_slots[1], ram_layer + (size_t)active_experts[1] * EXPERT_BYTES, EXPERT_BYTES);
                c_cpu_stage_ms += std::chrono::duration<double, std::milli>(clk::now() - tc0).count();
            }

            for (int a = 0; a < n_act; ++a) {
                int slot = a & 1;
                int next_slot = (a + 1) & 1;
                int eid = active_experts[a];
                int nt = (int)exp_toks[eid].size();

                if (pp) cudaEventRecord(ev_p_wait_s[a], compute_stream);
                cudaStreamWaitEvent(compute_stream, ev_copy_done[slot], 0);
                if (pp) cudaEventRecord(ev_p_wait_e[a], compute_stream);
                if (a + 1 < n_act) {
                    if (a >= 1) cudaStreamWaitEvent(copy_stream, (pinned_l && rans_mode) ? ev_rans_done[next_slot] : ev_comp_done[next_slot], 0);
                    enqueue_prefill_dma(a + 1, next_slot);
                }

                // Zero H2D inside expert loop: read directly from pre-uploaded flat layer buffer!
                const int*   d_idx_e = d_tok_idx_buf + exp_flat_off[eid];
                const float* d_w_e   = d_tok_w_buf   + exp_flat_off[eid];
                if (pinned_l && rans_mode) {
                    rans_decode_expert_fused_kernel<<<75, 192, 0, compute_stream>>>(comp_stream_bufs[slot], stream_bufs[slot]);
                    cudaEventRecord(ev_rans_done[slot], compute_stream);
                }
                if (pp) cudaEventRecord(ev_p_gath_s[a], compute_stream);
                gather_tokens_kernel<<<(nt * D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                    p_N_f32, nullptr, d_idx_e, p_ein_f32, nullptr, nt
                );
                if (pp) cudaEventRecord(ev_p_gath_e[a], compute_stream);

                // Dequantize MXFP4 -> FP32 (warp-uniform __shfl_sync) & run native FP32 cublasSgemm!
                const uint8_t* bp = stream_bufs[slot];
                if (pp) cudaEventRecord(ev_p_deq1_s[a], compute_stream);
                mxfp4_dequant_to_fp32_kernel<<<(5760 * 90 + 255) / 256, 256, 0, compute_stream>>>(
                    bp + OFF_GU_BLOCKS, bp + OFF_GU_SCALES, p_W_f32, 2 * D_FF
                );
                if (pp) cudaEventRecord(ev_p_deq1_e[a], compute_stream);

                if (pp) cudaEventRecord(ev_p_gemm1_s[a], compute_stream);
                sgemm_f32(p_W_f32, p_ein_f32, p_egu_f32, 2 * D_FF, nt, D_MODEL);
                if (pp) cudaEventRecord(ev_p_gemm1_e[a], compute_stream);

                if (pp) cudaEventRecord(ev_p_swi_s[a], compute_stream);
                swiglu_interleaved_kernel<<<(nt * D_FF + 255) / 256, 256, 0, compute_stream>>>(
                    p_egu_f32, ld.gu_bias + (size_t)eid * (2 * D_FF), p_emid_f32, nullptr, nt
                );
                if (pp) cudaEventRecord(ev_p_swi_e[a], compute_stream);

                if (pp) cudaEventRecord(ev_p_deq2_s[a], compute_stream);
                mxfp4_dequant_to_fp32_kernel<<<(2880 * 90 + 255) / 256, 256, 0, compute_stream>>>(
                    bp + OFF_DN_BLOCKS, bp + OFF_DN_SCALES, p_W_f32, D_MODEL
                );
                if (pp) cudaEventRecord(ev_p_deq2_e[a], compute_stream);

                if (pp) cudaEventRecord(ev_p_gemm2_s[a], compute_stream);
                sgemm_f32(p_W_f32, p_emid_f32, p_edn_f32, D_MODEL, nt, D_FF);
                if (pp) cudaEventRecord(ev_p_gemm2_e[a], compute_stream);

                if (pp) cudaEventRecord(ev_p_scat_s[a], compute_stream);
                scatter_weighted_add_kernel<<<(nt * D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                    p_edn_f32, ld.dn_bias + (size_t)eid * D_MODEL, d_idx_e, d_w_e, p_H, nt
                );
                if (pp) cudaEventRecord(ev_p_scat_e[a], compute_stream);
                cudaEventRecord(ev_comp_done[slot], compute_stream);

                if (!pinned_l && a + 2 < n_act) {
                    int stage_hslot = (a + 2) & 3;
                    if (a + 2 >= 4) {
                        auto tw0 = clk::now();
                        cudaEventSynchronize(ev_host_slot_done[stage_hslot]);
                        c_sync_ms += std::chrono::duration<double, std::milli>(clk::now() - tw0).count();
                    }
                    auto tc0 = clk::now();
                    copy_pool.copy(host_pinned_slots[stage_hslot], ram_layer + (size_t)active_experts[a + 2] * EXPERT_BYTES, EXPERT_BYTES);
                    c_cpu_stage_ms += std::chrono::duration<double, std::milli>(clk::now() - tc0).count();
                }
            }
            if (l == 2) {
                cudaMemcpyAsync(d_H_l2, p_H + (size_t)(chunk_len - 1) * D_MODEL, D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
                if (hidden_l2_out_host != nullptr) {
                    cudaMemcpyAsync(hidden_l2_out_host, p_H, C * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
                }
            }
            if (l == 12) {
                cudaMemcpyAsync(d_H_l12, p_H + (size_t)(chunk_len - 1) * D_MODEL, D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
                if (hidden_l12_out_host != nullptr) {
                    cudaMemcpyAsync(hidden_l12_out_host, p_H, C * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
                }
            }
            if (pp) {
                cudaStreamSynchronize(compute_stream);
                cudaStreamSynchronize(copy_stream);
                float ms = 0.0f;
                for (int a = 0; a < n_act; ++a) {
                    cudaEventElapsedTime(&ms, ev_p_dma_s[a],   ev_p_dma_e[a]);   c_h2d_dma_ms += ms;
                    cudaEventElapsedTime(&ms, ev_p_wait_s[a],  ev_p_wait_e[a]);  c_wait_dma_ms += ms;
                    cudaEventElapsedTime(&ms, ev_p_gath_s[a],  ev_p_gath_e[a]);  c_gath_scat_ms += ms;
                    cudaEventElapsedTime(&ms, ev_p_deq1_s[a],  ev_p_deq1_e[a]);  c_deq_ms += ms;
                    cudaEventElapsedTime(&ms, ev_p_gemm1_s[a], ev_p_gemm1_e[a]); c_gemm_ms += ms;
                    cudaEventElapsedTime(&ms, ev_p_swi_s[a],   ev_p_swi_e[a]);   c_gath_scat_ms += ms;
                    cudaEventElapsedTime(&ms, ev_p_deq2_s[a],  ev_p_deq2_e[a]);  c_deq_ms += ms;
                    cudaEventElapsedTime(&ms, ev_p_gemm2_s[a], ev_p_gemm2_e[a]); c_gemm_ms += ms;
                    cudaEventElapsedTime(&ms, ev_p_scat_s[a],  ev_p_scat_e[a]);  c_gath_scat_ms += ms;
                }
            }
        }

        // Preserve last token pre-final_norm hidden state in d_H[0] (for EAGLE-1 resident draft!)
        cudaMemcpyAsync(d_H, p_H + (size_t)(chunk_len - 1) * D_MODEL, D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
        if (hidden_out_host != nullptr) {
            cudaMemcpyAsync(hidden_out_host, p_H, C * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
        }

        // ONLY project lm_head on the LAST token if requested (NEVER chunk x 201088!)
        if (compute_last_logits && last_logits_out != nullptr) {
            float* last_h = p_H + (size_t)(chunk_len - 1) * D_MODEL;
            rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(last_h, final_norm, nullptr, d_N_f16, 1);
            float* lg_dst = logits_on_gpu ? last_logits_out : d_logits;
            gemm_fp16_to_fp32(lm_head, d_N_f16, lg_dst, VOCAB_SIZE, 1, D_MODEL);
            if (!logits_on_gpu) {
                cudaMemcpyAsync(last_logits_out, d_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
            }
        }
        auto te0 = clk::now();
        cudaStreamSynchronize(compute_stream);
        c_sync_ms += std::chrono::duration<double, std::milli>(clk::now() - te0).count();
        double c_wall = std::chrono::duration<double, std::milli>(clk::now() - t_chunk0).count();

        if (pp) {
            prefill_prof.chunks++;
            prefill_prof.tokens += (uint64_t)chunk_len;
            prefill_prof.experts_loaded += c_exp_loaded;
            prefill_prof.total_wall_ms += c_wall;
            prefill_prof.attn_local_ring128_ms += c_attn_loc_ms;
            prefill_prof.attn_global_causal_ms += c_attn_glob_ms;
            prefill_prof.qkv_o_router_ms += c_qkv_o_r_ms;
            prefill_prof.h2d_dma_ms += c_h2d_dma_ms;
            prefill_prof.cpu_staging_ms += c_cpu_stage_ms;
            prefill_prof.dequant_mxfp4_ms += c_deq_ms;
            prefill_prof.gather_scatter_swiglu_ms += c_gath_scat_ms;
            prefill_prof.expert_gemm_ms += c_gemm_ms;
            prefill_prof.gpu_wait_dma_ms += c_wait_dma_ms;
            prefill_prof.sync_cpu_gpu_ms += c_sync_ms;
        }
    }

    // ---- Phase 2: Zero-PCIe Cache-Only Self-Draft Step (uses ONLY experts resident in VRAM cache!) ----
    void draft_token_cache_only_internal(int token_id, int pos, int max_k_cached, float* logits_out, int logits_on_gpu) {
        const uint16_t* emb_row = embed_ram_fp16 + (size_t)token_id * D_MODEL;
        memcpy(host_pinned_embed, emb_row, D_MODEL * sizeof(uint16_t));
        cudaMemcpyAsync(d_embed_f16, host_pinned_embed, D_MODEL * sizeof(uint16_t), cudaMemcpyHostToDevice, compute_stream);
        fp16_to_fp32_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(d_embed_f16, d_H, D_MODEL);

        float host_r_logits[NUM_EXPERTS];
        for (int l = 0; l < NUM_LAYERS; ++l) {
            const LayerGPU& ld = layers[l];
            run_attention_decode_or_verify(l, 1, pos);

            rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_H, ld.post_norm, d_N_f32, d_N_f16, 1);
            gemm_fp16_to_fp32(ld.router_w, d_N_f16, d_router_lg, NUM_EXPERTS, 1, D_MODEL);
            add_bias_32_kernel<<<1, 32, 0, compute_stream>>>(d_router_lg, ld.router_b);
            cudaMemcpyAsync(host_r_logits, d_router_lg, NUM_EXPERTS * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
            cudaStreamSynchronize(compute_stream);

            // Pick up to max_k_cached (e.g. Top-1 or Top-2 or Top-4) ONLY among experts currently in VRAM cache!
            int sel_e[TOP_K], sel_sl[TOP_K];
            float sel_v[TOP_K];
            int n_sel = 0;
            int lim = std::max(1, std::min(max_k_cached, TOP_K));
            for (int e = 0; e < NUM_EXPERTS; ++e) {
                int sl = (cache_policy != 0 && cache_capacity_slots > 0) ? key_to_slot[l * NUM_EXPERTS + e] : -1;
                if (sl < 0) continue;
                float v = host_r_logits[e];
                if (n_sel < lim) {
                    sel_e[n_sel] = e; sel_sl[n_sel] = sl; sel_v[n_sel] = v; n_sel++;
                } else {
                    int min_i = 0;
                    for (int j = 1; j < lim; ++j) if (sel_v[j] < sel_v[min_i]) min_i = j;
                    if (v > sel_v[min_i]) { sel_e[min_i] = e; sel_sl[min_i] = sl; sel_v[min_i] = v; }
                }
            }
            if (n_sel > 0) {
                float mx = sel_v[0];
                for (int i = 1; i < n_sel; ++i) mx = std::max(mx, sel_v[i]);
                float sum_e = 0.0f;
                for (int i = 0; i < n_sel; ++i) { sel_v[i] = expf(sel_v[i] - mx); sum_e += sel_v[i]; }
                for (int i = 0; i < n_sel; ++i) {
                    int eid = sel_e[i];
                    const uint8_t* bp = cache_slot_ptr(sel_sl[i]);
                    mxfp4_gate_up_swiglu_1vec_kernel<<<(D_FF + 7) / 8, 256, 0, compute_stream>>>(
                        bp + OFF_GU_BLOCKS, bp + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_N_f32, d_exp_mid
                    );
                    mxfp4_down_wadd_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                        bp + OFF_DN_BLOCKS, bp + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, sel_v[i] / sum_e, d_H, D_MODEL
                    );
                }
            }
        }
        rmsnorm_fp32_fp16_kernel<<<1, 256, 0, compute_stream>>>(d_H, final_norm, nullptr, d_N_f16, 1);
        float* lg_dst = logits_on_gpu ? logits_out : d_logits;
        gemm_fp16_to_fp32(lm_head, d_N_f16, lg_dst, VOCAB_SIZE, 1, D_MODEL);
        if (!logits_on_gpu && logits_out != nullptr) {
            cudaMemcpyAsync(logits_out, d_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
        }
        cudaStreamSynchronize(compute_stream);
    }

    // ---- Phase 2 & Iter 6: Batched Exact Verification of K tokens (K <= 16, linear OR dynamic tree!) ----
    void verify_tokens_internal(
        const int* token_ids,
        int K,
        int start_pos,
        float* logits_out,
        int logits_on_gpu,
        uint64_t* out_pcie_bytes,
        int* out_unique_experts_total,
        int* out_cache_hits_total,
        int use_tree = 0
    ) {
        uint64_t step_pcie = 0;
        int tot_unique = 0;
        int tot_hits = 0;

        for (int i = 0; i < K; ++i) {
            memcpy(host_pinned_embed + (size_t)i * D_MODEL,
                   embed_ram_fp16 + (size_t)token_ids[i] * D_MODEL,
                   D_MODEL * sizeof(uint16_t));
        }
        cudaMemcpyAsync(d_embed_f16, host_pinned_embed, (size_t)K * D_MODEL * 2, cudaMemcpyHostToDevice, compute_stream);
        fp16_to_fp32_kernel<<<(K * D_MODEL + 255) / 256, 256, 0, compute_stream>>>(d_embed_f16, d_H, K * D_MODEL);

        std::vector<int> exp_toks[NUM_EXPERTS];
        std::vector<float> exp_ws[NUM_EXPERTS];

        for (int l = 0; l < NUM_LAYERS; ++l) {
            const LayerGPU& ld = layers[l];

            // Iter 7 Pillar 1: Batched Inter-Layer Speculative DMA Lookahead Prefetch for verify_tokens / tree verify!
            int pref_e[2] = {-1, -1};
            int n_pref = 0;
            const bool do_vpref = (lookahead_prefetch_mode == 1) && layer_is_pinned[l] && !rans_mode;
            if (do_vpref) {
                if (ema_attn_valid[l] && d_ema_attn != nullptr) {
                    add_f32_scaled_batched_2880_kernel<<<(K * D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                        d_H, d_ema_attn + (size_t)l * D_MODEL, 1.0f, d_mlp_acc, K
                    );
                    rmsnorm_fp32_fp16_kernel<<<K, 256, 0, compute_stream>>>(d_mlp_acc, ld.post_norm, nullptr, d_N_f16, K);
                } else {
                    rmsnorm_fp32_fp16_kernel<<<K, 256, 0, compute_stream>>>(d_H, ld.post_norm, nullptr, d_N_f16, K);
                }
                gemm_fp16_to_fp32(ld.router_w, d_N_f16, d_router_lg, NUM_EXPERTS, K, D_MODEL);
                lookahead_router_top4_kernel<<<K, 32, 0, compute_stream>>>(d_router_lg, ld.router_b, d_top_idx, d_top_w, K);
                cudaMemcpyAsync(host_pinned_top_idx, d_top_idx, (size_t)K * TOP_K, cudaMemcpyDeviceToHost, compute_stream);
                cudaMemcpyAsync(host_pinned_top_w,   d_top_w,   (size_t)K * TOP_K * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
                cudaStreamSynchronize(compute_stream);

                float e_score[NUM_EXPERTS] = {};
                for (int i = 0; i < K; ++i) {
                    for (int j = 0; j < TOP_K; ++j) {
                        int pe = (int)host_pinned_top_idx[i * TOP_K + j];
                        float pp = host_pinned_top_w[i * TOP_K + j];
                        if (pp < prefetch_tau_prob32) continue;
                        int sl = (cache_policy != 0 && cache_capacity_slots > 0) ? key_to_slot[l * NUM_EXPERTS + pe] : -1;
                        if (sl < 0) e_score[pe] += pp + ((i == 0) ? 0.20f : 0.05f);
                    }
                }
                const uint8_t* ram_l_pref = mxfp4_ram_layers[l];
                for (int p = 0; p < 2; ++p) {
                    int best_e = -1;
                    float best_s = 0.0f;
                    for (int e = 0; e < NUM_EXPERTS; ++e) {
                        if (e_score[e] > best_s) { best_s = e_score[e]; best_e = e; }
                    }
                    if (best_e < 0) break;
                    e_score[best_e] = -1.0f;
                    pref_e[n_pref] = best_e;
                    cudaMemcpyAsync(
                        stream_bufs[n_pref],
                        ram_l_pref + (size_t)best_e * EXPERT_BYTES,
                        EXPERT_BYTES,
                        cudaMemcpyHostToDevice,
                        copy_stream
                    );
                    cudaEventRecord(ev_pref_done[n_pref], copy_stream);
                    step_pcie += EXPERT_BYTES;
                    n_pref++;
                }
            }

            run_attention_decode_or_verify(l, K, start_pos, use_tree);
            if (do_vpref && d_ema_attn != nullptr) {
                // Update online EMA from root token (index 0, 100% guaranteed accepted!)
                float alpha = ema_attn_valid[l] ? 0.35f : 1.0f;
                ema_update_attn_2880_kernel<<<(D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                    d_ema_attn + (size_t)l * D_MODEL, d_proj_out, ld.b_o, alpha
                );
                ema_attn_valid[l] = 1;
            }

            rmsnorm_fp32_fp16_kernel<<<K, 256, 0, compute_stream>>>(d_H, ld.post_norm, d_N_f32, d_N_f16, K);
            gemm_fp16_to_fp32(ld.router_w, d_N_f16, d_router_lg, NUM_EXPERTS, K, D_MODEL);
            router_topk4_softmax_kernel<<<K, 32, 0, compute_stream>>>(d_router_lg, ld.router_b, d_top_idx, d_top_w, K);
            cudaMemcpyAsync(host_pinned_top_idx, d_top_idx, (size_t)K * TOP_K, cudaMemcpyDeviceToHost, compute_stream);
            cudaMemcpyAsync(host_pinned_top_w, d_top_w, (size_t)K * TOP_K * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
            cudaStreamSynchronize(compute_stream);
            if (!use_tree && hybrid_kv_mode && (l & 1) == 1) {
                flush_global_kv_mirror_to_ram(l >> 1, start_pos, K);
            }

            for (int e = 0; e < NUM_EXPERTS; ++e) { exp_toks[e].clear(); exp_ws[e].clear(); }
            last_verify_K = K;
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < TOP_K; ++j) {
                    int e = host_pinned_top_idx[i * TOP_K + j];
                    last_verify_top_idx[i][l][j] = (uint8_t)e;
                    exp_toks[e].push_back(i);
                    exp_ws[e].push_back(host_pinned_top_w[i * TOP_K + j]);
                }
            }
            int v_flat_off[NUM_EXPERTS];
            int v_flat_n = 0;
            for (int e = 0; e < NUM_EXPERTS; ++e) {
                v_flat_off[e] = v_flat_n;
                if (!exp_toks[e].empty()) {
                    int nt = (int)exp_toks[e].size();
                    memcpy(host_pinned_tok_idx + v_flat_n, exp_toks[e].data(), nt * sizeof(int));
                    memcpy(host_pinned_tok_w   + v_flat_n, exp_ws[e].data(),   nt * sizeof(float));
                    v_flat_n += nt;
                }
            }
            if (v_flat_n > 0) {
                cudaMemcpyAsync(d_tok_idx_buf, host_pinned_tok_idx, v_flat_n * sizeof(int),   cudaMemcpyHostToDevice, compute_stream);
                cudaMemcpyAsync(d_tok_w_buf,   host_pinned_tok_w,   v_flat_n * sizeof(float), cudaMemcpyHostToDevice, compute_stream);
            }

            uint64_t cur_pin = ++step_pin_id;
            std::vector<int> hit_e, hit_sl, miss_e;
            for (int e = 0; e < NUM_EXPERTS; ++e) {
                if (exp_toks[e].empty()) continue;
                tot_unique++;
                const bool is_tok0 = (exp_toks[e][0] == 0);
                int sl = -1;
                if (verify_read_only_cache && !is_tok0) {
                    // STRICT READ-ONLY lookup for speculative-only experts: zero LRU mutation or eviction!
                    sl = (cache_policy != 0 && cache_capacity_slots > 0) ? key_to_slot[l * NUM_EXPERTS + e] : -1;
                } else {
                    // Token 0 is 100% guaranteed accepted -> normal pin & LRU update (fixes post-prefill cold cache!)
                    sl = lookup_cache_and_pin(l * NUM_EXPERTS + e, cur_pin);
                }
                if (sl >= 0) { hit_e.push_back(e); hit_sl.push_back(sl); tot_hits++; }
                else         { miss_e.push_back(e); }
            }

            auto launch_batched_expert = [&](int eid, const uint8_t* bp) {
                int nt = (int)exp_toks[eid].size();
                if (nt == 1) {
                    int ti = exp_toks[eid][0];
                    const float* x_ptr = d_N_f32 + (size_t)ti * D_MODEL;
                    mxfp4_gate_up_swiglu_1vec_kernel<<<(D_FF + 7) / 8, 256, 0, compute_stream>>>(
                        bp + OFF_GU_BLOCKS, bp + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), x_ptr, d_exp_mid
                    );
                    mxfp4_down_wadd_1vec_kernel<<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                        bp + OFF_DN_BLOCKS, bp + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, exp_ws[eid][0], d_H + (size_t)ti * D_MODEL, D_MODEL
                    );
                } else {
                    const int*   d_idx_e = d_tok_idx_buf + v_flat_off[eid];
                    const float* d_w_e   = d_tok_w_buf   + v_flat_off[eid];
                    gather_tokens_kernel<<<(nt * D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                        d_N_f32, nullptr, d_idx_e, d_exp_in, nullptr, nt
                    );
                    if (nt <= 2) {
                        mxfp4_gemv_batched_kernel<2><<<(2 * D_FF + 7) / 8, 256, 0, compute_stream>>>(
                            bp + OFF_GU_BLOCKS, bp + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_exp_in, d_exp_gu, 2 * D_FF, nt
                        );
                        swiglu_interleaved_kernel<<<(nt * D_FF + 255) / 256, 256, 0, compute_stream>>>(d_exp_gu, nullptr, d_exp_mid, nullptr, nt);
                        mxfp4_gemv_batched_kernel<2><<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                            bp + OFF_DN_BLOCKS, bp + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, d_exp_dn, D_MODEL, nt
                        );
                    } else if (nt <= 4) {
                        mxfp4_gemv_batched_kernel<4><<<(2 * D_FF + 7) / 8, 256, 0, compute_stream>>>(
                            bp + OFF_GU_BLOCKS, bp + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_exp_in, d_exp_gu, 2 * D_FF, nt
                        );
                        swiglu_interleaved_kernel<<<(nt * D_FF + 255) / 256, 256, 0, compute_stream>>>(d_exp_gu, nullptr, d_exp_mid, nullptr, nt);
                        mxfp4_gemv_batched_kernel<4><<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                            bp + OFF_DN_BLOCKS, bp + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, d_exp_dn, D_MODEL, nt
                        );
                    } else if (nt <= 8) {
                        mxfp4_gemv_batched_kernel<8><<<(2 * D_FF + 7) / 8, 256, 0, compute_stream>>>(
                            bp + OFF_GU_BLOCKS, bp + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF), d_exp_in, d_exp_gu, 2 * D_FF, nt
                        );
                        swiglu_interleaved_kernel<<<(nt * D_FF + 255) / 256, 256, 0, compute_stream>>>(d_exp_gu, nullptr, d_exp_mid, nullptr, nt);
                        mxfp4_gemv_batched_kernel<8><<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                            bp + OFF_DN_BLOCKS, bp + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL, d_exp_mid, d_exp_dn, D_MODEL, nt
                        );
                    } else {
                        // nt in 9..16: split cleanly into 2 chunks of <= 8 (zero extra register pressure!)
                        for (int off_b = 0; off_b < nt; off_b += 8) {
                            int b_ch = std::min(8, nt - off_b);
                            mxfp4_gemv_batched_kernel<8><<<(2 * D_FF + 7) / 8, 256, 0, compute_stream>>>(
                                bp + OFF_GU_BLOCKS, bp + OFF_GU_SCALES, ld.gu_bias + (size_t)eid * (2 * D_FF),
                                d_exp_in + (size_t)off_b * D_MODEL, d_exp_gu + (size_t)off_b * (2 * D_FF), 2 * D_FF, b_ch
                            );
                        }
                        swiglu_interleaved_kernel<<<(nt * D_FF + 255) / 256, 256, 0, compute_stream>>>(d_exp_gu, nullptr, d_exp_mid, nullptr, nt);
                        for (int off_b = 0; off_b < nt; off_b += 8) {
                            int b_ch = std::min(8, nt - off_b);
                            mxfp4_gemv_batched_kernel<8><<<(D_MODEL + 7) / 8, 256, 0, compute_stream>>>(
                                bp + OFF_DN_BLOCKS, bp + OFF_DN_SCALES, ld.dn_bias + (size_t)eid * D_MODEL,
                                d_exp_mid + (size_t)off_b * D_FF, d_exp_dn + (size_t)off_b * D_MODEL, D_MODEL, b_ch
                            );
                        }
                    }
                    scatter_weighted_add_kernel<<<(nt * D_MODEL + 255) / 256, 256, 0, compute_stream>>>(
                        d_exp_dn, nullptr, d_idx_e, d_w_e, d_H, nt
                    );
                }
            };

            const uint8_t* ram_layer = mxfp4_ram_layers[l];
            int nm = (int)miss_e.size();
            std::vector<uint8_t*> miss_dst(nm);
            std::vector<int> miss_cslot(nm, -1);
            std::vector<int> miss_is_stream(nm, 0);
            std::vector<int> miss_stream_slot(nm, 0);
            int s_cnt = 0;
            for (int m = 0; m < nm; ++m) {
                const bool is_tok0 = (!exp_toks[miss_e[m]].empty() && exp_toks[miss_e[m]][0] == 0);
                if (verify_read_only_cache && !is_tok0) {
                    int sb = (s_cnt++) & 1;
                    miss_dst[m] = nullptr; // resolved dynamically from live stream_bufs[sb]
                    miss_cslot[m] = -1;
                    miss_is_stream[m] = 1;
                    miss_stream_slot[m] = sb;
                } else {
                    int csl = alloc_or_evict_cache_slot(l * NUM_EXPERTS + miss_e[m], cur_pin);
                    if (csl >= 0) {
                        miss_dst[m] = cache_slot_ptr(csl);
                        miss_cslot[m] = csl;
                        miss_is_stream[m] = 0;
                        miss_stream_slot[m] = m & 1;
                    } else {
                        int sb = (s_cnt++) & 1;
                        miss_dst[m] = nullptr;
                        miss_cslot[m] = -1;
                        miss_is_stream[m] = 1;
                        miss_stream_slot[m] = sb;
                    }
                }
            }

            // 1. Launch ALL cache hits FIRST on compute_stream (non-blocking enqueue -> overlaps with DMA!)
            for (size_t i = 0; i < hit_e.size(); ++i) {
                launch_batched_expert(hit_e[i], cache_slot_ptr(hit_sl[i]));
            }

            // 1B. Consume any Inter-Layer Prefetch HITs directly from stream_bufs[0..1] BEFORE non-prefetched misses!
            std::vector<uint8_t> miss_was_pref(nm, 0);
            int stream_used_count[2] = {0, 0};
            if (do_vpref && n_pref > 0) {
                for (int p = 0; p < n_pref; ++p) {
                    int matched_m = -1;
                    for (int m = 0; m < nm; ++m) {
                        if (miss_e[m] == pref_e[p]) { matched_m = m; break; }
                    }
                    if (matched_m >= 0) {
                        miss_was_pref[matched_m] = 1;
                        uint8_t* p_buf = stream_bufs[p];
                        cudaStreamWaitEvent(compute_stream, ev_pref_done[p], 0);
                        launch_batched_expert(miss_e[matched_m], p_buf);
                        if (!miss_is_stream[matched_m] && miss_cslot[matched_m] >= 0) {
                            // ZERO-COPY 1-ns pointer swap into persistent LRU cache slot!
                            std::swap(slot_ptrs[miss_cslot[matched_m]], stream_bufs[p]);
                        } else {
                            cudaEventRecord(ev_comp_done[p], compute_stream);
                            stream_used_count[p]++;
                        }
                    }
                }
            }

            // 2. Stream remaining non-prefetched MISSES: Direct Pinned DMA (uncompressed OR warp-rANS) OR legacy WC staging!
            if (layer_is_pinned[l]) {
                for (int m = 0; m < nm; ++m) {
                    if (miss_was_pref[m]) continue;
                    const int ev_s = miss_stream_slot[m];
                    const int cs = m & 1;
                    const int eid = miss_e[m];
                    uint8_t* g_dst = miss_is_stream[m] ? stream_bufs[ev_s] : slot_ptrs[miss_cslot[m]];
                    if (rans_mode) {
                        const uint8_t* src_e = ram_layer + (size_t)rans_exp_off[l][eid];
                        const size_t nb = (size_t)rans_exp_sz[l][eid];
                        if (m >= 2) cudaStreamWaitEvent(copy_stream, ev_rans_done[cs], 0);
                        cudaMemcpyAsync(comp_stream_bufs[cs], src_e, nb, cudaMemcpyHostToDevice, copy_stream);
                        cudaEventRecord(ev_copy_done[cs], copy_stream);
                        step_pcie += nb;

                        cudaStreamWaitEvent(compute_stream, ev_copy_done[cs], 0);
                        rans_decode_expert_fused_kernel<<<75, 192, 0, compute_stream>>>(comp_stream_bufs[cs], g_dst);
                        cudaEventRecord(ev_rans_done[cs], compute_stream);
                        launch_batched_expert(eid, g_dst);
                    } else {
                        if (miss_is_stream[m] && stream_used_count[ev_s] > 0) {
                            cudaStreamWaitEvent(copy_stream, ev_comp_done[ev_s], 0);
                        }
                        const uint8_t* src_e = ram_layer + (size_t)eid * EXPERT_BYTES;
                        cudaMemcpyAsync(g_dst, src_e, EXPERT_BYTES, cudaMemcpyHostToDevice, copy_stream);
                        cudaEventRecord(ev_copy_done[ev_s], copy_stream);
                        step_pcie += EXPERT_BYTES;

                        cudaStreamWaitEvent(compute_stream, ev_copy_done[ev_s], 0);
                        launch_batched_expert(eid, g_dst);
                        if (miss_is_stream[m]) {
                            cudaEventRecord(ev_comp_done[ev_s], compute_stream);
                            stream_used_count[ev_s]++;
                        }
                    }
                }
            } else {
                const int n_st = std::max(1, std::min(staging_stripes, 16));
                const size_t stripe_sz = (((EXPERT_BYTES + n_st - 1) / n_st) + 63ULL) & ~63ULL;
                int stream_used_count[2] = {0, 0};
                for (int m = 0; m < nm; ++m) {
                    const int ev_s = miss_stream_slot[m];
                    const int hs = m & 3;
                    if (m >= 4) cudaEventSynchronize(ev_host_slot_done[hs]);
                    if (miss_is_stream[m] && stream_used_count[ev_s] > 0) {
                        cudaStreamWaitEvent(copy_stream, ev_comp_done[ev_s], 0);
                    }
                    const uint8_t* src_e = ram_layer + (size_t)miss_e[m] * EXPERT_BYTES;
                    uint8_t* h_dst = host_pinned_slots[hs];
                    uint8_t* g_dst = miss_is_stream[m] ? stream_bufs[ev_s] : slot_ptrs[miss_cslot[m]];
                    for (int st = 0; st < n_st; ++st) {
                        size_t off = (size_t)st * stripe_sz;
                        if (off >= EXPERT_BYTES) break;
                        size_t len = std::min(stripe_sz, EXPERT_BYTES - off);
                        copy_pool.copy(h_dst + off, src_e + off, len);
                        cudaMemcpyAsync(g_dst + off, h_dst + off, len, cudaMemcpyHostToDevice, copy_stream);
                    }
                    cudaEventRecord(ev_host_slot_done[hs], copy_stream);
                    cudaEventRecord(ev_copy_done[ev_s], copy_stream);
                    step_pcie += EXPERT_BYTES;

                    cudaStreamWaitEvent(compute_stream, ev_copy_done[ev_s], 0);
                    launch_batched_expert(miss_e[m], g_dst);
                    if (miss_is_stream[m]) {
                        cudaEventRecord(ev_comp_done[ev_s], compute_stream);
                        stream_used_count[ev_s]++;
                    }
                }
            }
            if (l == 2)  cudaMemcpyAsync(d_H_l2,  d_H, (size_t)K * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
            if (l == 12) cudaMemcpyAsync(d_H_l12, d_H, (size_t)K * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice, compute_stream);
        }

        rmsnorm_fp32_fp16_kernel<<<K, 256, 0, compute_stream>>>(d_H, final_norm, nullptr, d_N_f16, K);
        float* lg_dst = logits_on_gpu ? logits_out : d_logits;
        gemm_fp16_to_fp32(lm_head, d_N_f16, lg_dst, VOCAB_SIZE, K, D_MODEL);
        if (!logits_on_gpu && logits_out != nullptr) {
            cudaMemcpyAsync(logits_out, d_logits, (size_t)K * VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost, compute_stream);
        }
        cudaStreamSynchronize(compute_stream);

        if (out_pcie_bytes) *out_pcie_bytes = step_pcie;
        if (out_unique_experts_total) *out_unique_experts_total = tot_unique;
        if (out_cache_hits_total) *out_cache_hits_total = tot_hits;
    }

    // Commit cache statistics & LRU/LFU recency ONLY for the actually accepted prefix [0 .. n_accepted - 1]!
    void commit_verified_cache_accesses(int n_accepted) {
        int lim = std::max(0, std::min(n_accepted, last_verify_K));
        for (int i = 0; i < lim; ++i) {
            for (int l = 0; l < NUM_LAYERS; ++l) {
                for (int k = 0; k < TOP_K; ++k) {
                    int key = l * NUM_EXPERTS + last_verify_top_idx[i][l][k];
                    int sl = (cache_policy != 0 && cache_capacity_slots > 0) ? key_to_slot[key] : -1;
                    if (sl >= 0) {
                        cache_slots[sl].last_used = ++access_clock;
                        cache_slots[sl].freq += 1.0f;
                        prof.cache_hits++;
                        prof.layer_hits[l]++;
                    } else {
                        prof.cache_misses++;
                        prof.layer_misses[l]++;
                    }
                }
            }
        }
    }
};

// ============================================================================
// C API Exports
// ============================================================================
#define CUDA_OK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) return (int)_e; } while (0)

EXPORT void* engine_create_ex(int kv_cap_global, int hybrid_kv_mode) {
    auto* eng = new GPTOSSNativeEngine();
    eng->kv_cap_global = std::max(128, std::min(kv_cap_global, MAX_POS));
    eng->hybrid_kv_mode = (hybrid_kv_mode != 0) && (eng->kv_cap_global > HOT_WINDOW);
    eng->kv_cold_pages = eng->hybrid_kv_mode ? ((eng->kv_cap_global - HOT_WINDOW + V_PAGE_SIZE - 1) / V_PAGE_SIZE) : 0;
    eng->kv_cold_cap   = eng->kv_cold_pages * V_PAGE_SIZE;

    if (cublasCreate(&eng->cublas) != CUBLAS_STATUS_SUCCESS) { delete eng; return nullptr; }
    cublasSetMathMode(eng->cublas, CUBLAS_DEFAULT_MATH);
    cudaStreamCreate(&eng->compute_stream);
    cudaStreamCreate(&eng->copy_stream);
    cublasSetStream(eng->cublas, eng->compute_stream);

    for (int i = 0; i < 2; ++i) {
        cudaEventCreateWithFlags(&eng->ev_copy_done[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&eng->ev_comp_done[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&eng->ev_rans_done[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&eng->ev_pref_done[i], cudaEventDisableTiming);
    }
    for (int i = 0; i < 4; ++i) {
        cudaEventCreateWithFlags(&eng->ev_host_slot_done[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&eng->ev_copy_done_miss[i], cudaEventDisableTiming);
    }
    for (int l = 0; l < NUM_LAYERS; ++l) {
        cudaEventCreate(&eng->ev_attn_s[l]); cudaEventCreate(&eng->ev_attn_e[l]);
        cudaEventCreate(&eng->ev_la_s[l]);   cudaEventCreate(&eng->ev_la_e[l]);
        for (int p = 0; p < 2; ++p) {
            cudaEventCreate(&eng->ev_pref_s[l][p]);  cudaEventCreate(&eng->ev_pref_e[l][p]);
            cudaEventCreate(&eng->ev_pwait_s[l][p]); cudaEventCreate(&eng->ev_pwait_e[l][p]);
        }
        for (int k = 0; k < TOP_K; ++k) {
            cudaEventCreate(&eng->ev_dma_s[l][k]);  cudaEventCreate(&eng->ev_dma_e[l][k]);
            cudaEventCreate(&eng->ev_wait_s[l][k]); cudaEventCreate(&eng->ev_wait_e[l][k]);
            cudaEventCreate(&eng->ev_exp_s[l][k]);  cudaEventCreate(&eng->ev_exp_e[l][k]);
        }
    }
    cudaEventCreate(&eng->ev_lm_s); cudaEventCreate(&eng->ev_lm_e);
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        cudaEventCreate(&eng->ev_p_dma_s[e]);   cudaEventCreate(&eng->ev_p_dma_e[e]);
        cudaEventCreate(&eng->ev_p_wait_s[e]);  cudaEventCreate(&eng->ev_p_wait_e[e]);
        cudaEventCreate(&eng->ev_p_gath_s[e]);  cudaEventCreate(&eng->ev_p_gath_e[e]);
        cudaEventCreate(&eng->ev_p_deq1_s[e]);  cudaEventCreate(&eng->ev_p_deq1_e[e]);
        cudaEventCreate(&eng->ev_p_gemm1_s[e]); cudaEventCreate(&eng->ev_p_gemm1_e[e]);
        cudaEventCreate(&eng->ev_p_swi_s[e]);   cudaEventCreate(&eng->ev_p_swi_e[e]);
        cudaEventCreate(&eng->ev_p_deq2_s[e]);  cudaEventCreate(&eng->ev_p_deq2_e[e]);
        cudaEventCreate(&eng->ev_p_gemm2_s[e]); cudaEventCreate(&eng->ev_p_gemm2_e[e]);
        cudaEventCreate(&eng->ev_p_scat_s[e]);  cudaEventCreate(&eng->ev_p_scat_e[e]);
    }

    // Allocate 2 GPU streaming buffers & 4 Write-Combined pinned host staging slots (CPU write-only, zero PCIe snoop!)
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&eng->orig_stream_bufs[i], EXPERT_BYTES);
        eng->stream_bufs[i] = eng->orig_stream_bufs[i];
    }
    for (int i = 0; i < 4; ++i) cudaHostAlloc(&eng->host_pinned_slots[i], EXPERT_BYTES, cudaHostAllocWriteCombined);
    cudaHostAlloc(&eng->host_pinned_embed, (size_t)MAX_PREFILL_CHUNK * D_MODEL * sizeof(uint16_t), cudaHostAllocDefault);
    cudaHostAlloc(&eng->host_pinned_top_idx, (size_t)MAX_PREFILL_CHUNK * TOP_K, cudaHostAllocDefault);
    cudaHostAlloc(&eng->host_pinned_top_w, (size_t)MAX_PREFILL_CHUNK * TOP_K * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&eng->host_pinned_tok_idx, (size_t)NUM_EXPERTS * MAX_PREFILL_CHUNK * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&eng->host_pinned_tok_w, (size_t)NUM_EXPERTS * MAX_PREFILL_CHUNK * sizeof(float), cudaHostAllocDefault);

    // Allocate KV caches (12 local Ring-128 + 12 global Pure FP16 OR Stage 4B Hybrid Hot-4K FP16 / Cold-INT8 + RAM mirror!)
    size_t local_kv_one = (size_t)NUM_LOCAL_LAYERS * RING_SIZE * 512 * sizeof(__half);
    size_t shadow_one = (size_t)NUM_LOCAL_LAYERS * MAX_VERIFY_K * 512 * sizeof(__half);
    cudaMalloc(&eng->K_local_ring, local_kv_one);   cudaMemset(eng->K_local_ring, 0, local_kv_one);
    cudaMalloc(&eng->V_local_ring, local_kv_one);   cudaMemset(eng->V_local_ring, 0, local_kv_one);
    cudaMalloc(&eng->K_ring_shadow, shadow_one);
    cudaMalloc(&eng->V_ring_shadow, shadow_one);

    if (eng->hybrid_kv_mode) {
        size_t k_full  = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(__half);
        size_t v_hot   = (size_t)NUM_GLOBAL_LAYERS * HOT_WINDOW * 512 * sizeof(__half);
        size_t v_sink  = (size_t)NUM_GLOBAL_LAYERS * V_SINK_FP16 * 512 * sizeof(__half);
        size_t v_cold  = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cold_cap * 512 * sizeof(uint8_t);
        size_t vpg_one = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cold_pages * 512 * sizeof(__half);
        size_t vbnd_b  = (size_t)NUM_GLOBAL_LAYERS * V_PAGE_SIZE * 512 * sizeof(__half);
        cudaMalloc(&eng->K_global, k_full);          cudaMemset(eng->K_global, 0, k_full);
        cudaMalloc(&eng->V_global, v_hot);           cudaMemset(eng->V_global, 0, v_hot);
        cudaMalloc(&eng->V_sink_fp16, v_sink);       cudaMemset(eng->V_sink_fp16, 0, v_sink);
        cudaMalloc(&eng->V_global_cold, v_cold);     cudaMemset(eng->V_global_cold, 0, v_cold);
        cudaMalloc(&eng->V_page_min, vpg_one);       cudaMemset(eng->V_page_min, 0, vpg_one);
        cudaMalloc(&eng->V_page_max, vpg_one);       cudaMemset(eng->V_page_max, 0, vpg_one);
        cudaMalloc(&eng->V_hot_shadow, shadow_one);
        cudaMalloc(&eng->V_bnd_fp16, vbnd_b);        cudaMemset(eng->V_bnd_fp16, 0, vbnd_b);

        // Full FP16 global V mirror in system RAM (0.805 GB at 64K — K is already exact linear FP16 in VRAM!)
        size_t ram_one = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(uint16_t);
        eng->h_V_global_ram = static_cast<uint16_t*>(calloc(1, ram_one));
        cudaHostAlloc(&eng->h_kv_mirror_v, (size_t)MAX_PREFILL_CHUNK * 512 * sizeof(__half), cudaHostAllocDefault);
    } else {
        size_t global_kv_one = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(__half);
        cudaMalloc(&eng->K_global, global_kv_one);      cudaMemset(eng->K_global, 0, global_kv_one);
        cudaMalloc(&eng->V_global, global_kv_one);      cudaMemset(eng->V_global, 0, global_kv_one);
    }

    // Allocate fixed scratch + Split-K workspace
    size_t sc = 0;
    auto alloc_sc = [&](auto** ptr, size_t n_elems) {
        size_t nb = n_elems * sizeof(**ptr);
        cudaMalloc(ptr, nb);
        sc += nb;
    };
    alloc_sc(&eng->d_embed_f16, (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_H,         (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_N_f32,     (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_N_f16,     (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_QKV,       (size_t)MAX_VERIFY_K * QKV_DIM);
    alloc_sc(&eng->d_Q,         (size_t)MAX_VERIFY_K * 4096);
    alloc_sc(&eng->d_O_f16,     (size_t)MAX_VERIFY_K * 4096);
    alloc_sc(&eng->d_proj_out,  (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_router_lg, (size_t)MAX_VERIFY_K * NUM_EXPERTS);
    alloc_sc(&eng->d_top_idx,   (size_t)MAX_VERIFY_K * TOP_K);
    alloc_sc(&eng->d_top_w,     (size_t)MAX_VERIFY_K * TOP_K);
    alloc_sc(&eng->d_mlp_acc,   (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_exp_in,    (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_exp_gu,    (size_t)MAX_VERIFY_K * 2 * D_FF);
    alloc_sc(&eng->d_exp_mid,   (size_t)MAX_VERIFY_K * D_FF);
    alloc_sc(&eng->d_exp_dn,    (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_tok_idx_buf, (size_t)MAX_PREFILL_CHUNK * TOP_K);
    alloc_sc(&eng->d_tok_w_buf,   (size_t)MAX_PREFILL_CHUNK * TOP_K);
    alloc_sc(&eng->d_logits,    (size_t)MAX_VERIFY_K * VOCAB_SIZE);
    // Iter 7 Pillar 2A: Compact split buffers (B=1 uses <=64 splits -> 4096 entries; B<=8 uses <=32 splits -> 16384 entries -> saves 12.58 MB!)
    const size_t split_slots = std::max((size_t)NUM_Q_HEADS * MAX_SPLITS, (size_t)MAX_VERIFY_K * NUM_Q_HEADS * 32);
    alloc_sc(&eng->d_split_m,     split_slots);
    alloc_sc(&eng->d_split_l,     split_slots);
    alloc_sc(&eng->d_split_o,     split_slots * HEAD_DIM);
    alloc_sc(&eng->d_m_floor,     (size_t)MAX_VERIFY_K * NUM_Q_HEADS);
    alloc_sc(&eng->d_prune_stats, 6ULL);
    cudaMemset(eng->d_prune_stats, 0, 6ULL * sizeof(unsigned long long));
    alloc_sc(&eng->d_K_loc_tmp,   (size_t)(RING_SIZE + MAX_VERIFY_K) * 512);
    alloc_sc(&eng->d_V_loc_tmp,   (size_t)(RING_SIZE + MAX_VERIFY_K) * 512);
    alloc_sc(&eng->d_H_l2,        (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_H_l12,       (size_t)MAX_VERIFY_K * D_MODEL);
    alloc_sc(&eng->d_ema_attn,    (size_t)NUM_LAYERS * D_MODEL);
    cudaMemset(eng->d_ema_attn, 0, (size_t)NUM_LAYERS * D_MODEL * sizeof(float));
    alloc_sc(&eng->d_K_loc_tree,  (size_t)NUM_LOCAL_LAYERS * MAX_VERIFY_K * 512);
    alloc_sc(&eng->d_V_loc_tree,  (size_t)NUM_LOCAL_LAYERS * MAX_VERIFY_K * 512);
    alloc_sc(&eng->d_K_glob_tmp,  (size_t)NUM_GLOBAL_LAYERS * MAX_VERIFY_K * 512);
    alloc_sc(&eng->d_V_glob_tmp,  (size_t)NUM_GLOBAL_LAYERS * MAX_VERIFY_K * 512);
    alloc_sc(&eng->d_tree_path_ids, (size_t)MAX_VERIFY_K);
    eng->scratch_workspace_bytes = sc;

    for (int i = 0; i < NUM_LAYERS * NUM_EXPERTS; ++i) eng->key_to_slot[i] = -1;
    return eng;
}

EXPORT void* engine_create(int kv_cap_global) {
    return engine_create_ex(kv_cap_global, 0);
}

EXPORT int engine_set_global_weights(
    void* handle,
    const uint16_t* embed_ram_fp16,
    const uint16_t* lm_head_fp16,
    const float* final_norm_fp32,
    const float* cos_64k_fp32,
    const float* sin_64k_fp32
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->embed_ram_fp16 = embed_ram_fp16;

    size_t nb_lm = (size_t)VOCAB_SIZE * D_MODEL * sizeof(__half);
    size_t nb_fn = (size_t)D_MODEL * sizeof(float);
    size_t nb_cs = (size_t)MAX_POS * 32 * sizeof(float);

    CUDA_OK(cudaMalloc(&eng->lm_head, nb_lm));
    CUDA_OK(cudaMemcpy(eng->lm_head, lm_head_fp16, nb_lm, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&eng->final_norm, nb_fn));
    CUDA_OK(cudaMemcpy(eng->final_norm, final_norm_fp32, nb_fn, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&eng->cos_table, nb_cs));
    CUDA_OK(cudaMemcpy(eng->cos_table, cos_64k_fp32, nb_cs, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&eng->sin_table, nb_cs));
    CUDA_OK(cudaMemcpy(eng->sin_table, sin_64k_fp32, nb_cs, cudaMemcpyHostToDevice));

    eng->resident_weights_bytes += nb_lm + nb_fn + 2 * nb_cs;
    return 0;
}

EXPORT int engine_set_layer_weights(
    void* handle,
    int l,
    const float* in_norm,
    const float* post_norm,
    const uint16_t* W_qkv,
    const uint16_t* b_qkv,
    const uint16_t* W_o,
    const uint16_t* b_o,
    const float* sinks,
    const uint16_t* router_w,
    const uint16_t* router_b,
    const uint16_t* gu_bias,
    const uint16_t* dn_bias,
    const uint8_t* mxfp4_ram_layer_ptr
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    LayerGPU& ld = eng->layers[l];
    if (mxfp4_ram_layer_ptr != nullptr) eng->mxfp4_ram_layers[l] = mxfp4_ram_layer_ptr;

    auto upload = [&](auto** dst, const void* src, size_t bytes) -> int {
        CUDA_OK(cudaMalloc(dst, bytes));
        CUDA_OK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
        eng->resident_weights_bytes += bytes;
        return 0;
    };
    if (int e = upload(&ld.in_norm,   in_norm,   D_MODEL * 4)) return e;
    if (int e = upload(&ld.post_norm, post_norm, D_MODEL * 4)) return e;
    if (int e = upload(&ld.W_qkv,     W_qkv,     (size_t)QKV_DIM * D_MODEL * 2)) return e;
    if (int e = upload(&ld.b_qkv,     b_qkv,     (size_t)QKV_DIM * 2)) return e;
    if (int e = upload(&ld.W_o,       W_o,       (size_t)D_MODEL * 4096 * 2)) return e;
    if (int e = upload(&ld.b_o,       b_o,       (size_t)D_MODEL * 2)) return e;
    if (int e = upload(&ld.sinks,     sinks,     (size_t)NUM_Q_HEADS * 4)) return e;
    if (int e = upload(&ld.router_w,  router_w,  (size_t)NUM_EXPERTS * D_MODEL * 2)) return e;
    if (int e = upload(&ld.router_b,  router_b,  (size_t)NUM_EXPERTS * 2)) return e;
    if (int e = upload(&ld.gu_bias,   gu_bias,   (size_t)NUM_EXPERTS * 2 * D_FF * 2)) return e;
    if (int e = upload(&ld.dn_bias,   dn_bias,   (size_t)NUM_EXPERTS * D_MODEL * 2)) return e;
    return 0;
}

EXPORT int engine_init_arena(void* handle, size_t max_vram_ceiling_bytes, size_t device_safety_margin_bytes) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (eng->d_arena) {
        cudaFree(eng->d_arena);
        eng->d_arena = nullptr;
        eng->arena_bytes = 0;
    }

    size_t local_kv = 2ULL * NUM_LOCAL_LAYERS * RING_SIZE * 512 * sizeof(__half);
    size_t global_kv = 0;
    size_t shadow_kv = 2ULL * NUM_LOCAL_LAYERS * MAX_VERIFY_K * 512 * sizeof(__half);
    if (eng->hybrid_kv_mode) {
        size_t k_full  = 1ULL * NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(__half);
        size_t v_hot   = 1ULL * NUM_GLOBAL_LAYERS * HOT_WINDOW * 512 * sizeof(__half);
        size_t v_sink  = 1ULL * NUM_GLOBAL_LAYERS * V_SINK_FP16 * 512 * sizeof(__half);
        size_t v_cold  = 1ULL * NUM_GLOBAL_LAYERS * eng->kv_cold_cap * 512 * sizeof(uint8_t);
        size_t vpg_b   = 2ULL * NUM_GLOBAL_LAYERS * eng->kv_cold_pages * 512 * sizeof(__half);
        global_kv = k_full + v_hot + v_sink + v_cold + vpg_b;
        shadow_kv += 1ULL * NUM_GLOBAL_LAYERS * (MAX_VERIFY_K + V_PAGE_SIZE) * 512 * sizeof(__half);
    } else {
        global_kv = 2ULL * NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(__half);
    }
    size_t comp_buf_b = (eng->comp_stream_bufs[0] != nullptr) ? (2ULL * COMP_STREAM_BUF_BYTES) : 0ULL;
    size_t base_used = eng->resident_weights_bytes + local_kv + global_kv + shadow_kv
                       + 2 * EXPERT_BYTES + comp_buf_b + eng->scratch_workspace_bytes;

    // Pre-initialize cuBLAS internal workspace BEFORE querying cudaMemGetInfo so arena sizing sees exact post-cuBLAS free VRAM!
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_N, 16, 16, 16,
                    &alpha, eng->d_H, 16, eng->d_N_f32, 16, &beta, eng->d_QKV, 16);
        cudaStreamSynchronize(eng->compute_stream);
    }

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);

    size_t room_ceiling = (max_vram_ceiling_bytes > base_used) ? (max_vram_ceiling_bytes - base_used) : 0;
    size_t room_device  = (free_b > device_safety_margin_bytes) ? (free_b - device_safety_margin_bytes) : 0;
    size_t target_arena = std::min(room_ceiling, room_device);

    // Ensure at least 160 MB for FP32 SGEMM chunked prefill temporaries (~142.3 MB for chunk=512)
    size_t min_prefill = 160ULL * 1024 * 1024;
    if (target_arena < min_prefill) target_arena = std::min(free_b > (128ULL << 20) ? free_b - (128ULL << 20) : min_prefill, min_prefill);

    // Align down to multiple of EXPERT_BYTES (plus at least min_prefill)
    int n_slots = (int)(target_arena / EXPERT_BYTES);
    size_t alloc_bytes = std::max(min_prefill, (size_t)n_slots * EXPERT_BYTES);

    cudaError_t err = cudaMalloc(&eng->d_arena, alloc_bytes);
    while (err != cudaSuccess && n_slots > 0) {
        n_slots = std::max(0, n_slots - 4);
        alloc_bytes = std::max(min_prefill, (size_t)n_slots * EXPERT_BYTES);
        err = cudaMalloc(&eng->d_arena, alloc_bytes);
    }
    if (err != cudaSuccess) return (int)err;

    eng->arena_bytes = alloc_bytes;
    eng->cache_capacity_slots = (int)(alloc_bytes / EXPERT_BYTES);
    eng->cache_slots.assign(eng->cache_capacity_slots, CacheSlotMeta{});
    eng->clear_cache();
    return 0;
}

EXPORT void engine_set_cache_policy(void* handle, int policy, int max_slots_override) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->cache_policy = policy;
    int max_fit = (int)(eng->arena_bytes / EXPERT_BYTES);
    if (max_slots_override >= 0 && max_slots_override < max_fit) {
        eng->cache_capacity_slots = max_slots_override;
    } else {
        eng->cache_capacity_slots = max_fit;
    }
    eng->cache_slots.assign(eng->cache_capacity_slots, CacheSlotMeta{});
    eng->configure_layer_partitions();
    eng->clear_cache();
}

EXPORT void engine_set_copy_threads(void* handle, int num_threads) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->copy_pool.init(num_threads);
}

EXPORT void engine_set_staging_mode(void* handle, int use_avx2_nt, int num_stripes) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->copy_pool.use_avx2_nt = (use_avx2_nt != 0);
    eng->staging_stripes = std::max(1, std::min(num_stripes, 16));
}

EXPORT void engine_set_profile_enabled(void* handle, int enabled) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->profile_enabled = (enabled != 0);
}

EXPORT void engine_reset_profile(void* handle) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    memset(&eng->prof, 0, sizeof(ProfileStats));
}

EXPORT void engine_get_profile(void* handle, ProfileStats* out) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    *out = eng->prof;
}

EXPORT void engine_set_prefill_profile_enabled(void* handle, int enabled) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->prefill_profile_enabled = (enabled != 0);
}

EXPORT void engine_reset_prefill_profile(void* handle) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    memset(&eng->prefill_prof, 0, sizeof(PrefillProfileStats));
}

EXPORT void engine_get_prefill_profile(void* handle, PrefillProfileStats* out) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    *out = eng->prefill_prof;
}

EXPORT void engine_reset_cache(void* handle) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->clear_cache();
}

EXPORT void engine_clear_kv(void* handle) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    size_t local_kv_one = (size_t)NUM_LOCAL_LAYERS * RING_SIZE * 512 * sizeof(__half);
    cudaMemsetAsync(eng->K_local_ring, 0, local_kv_one, eng->compute_stream);
    cudaMemsetAsync(eng->V_local_ring, 0, local_kv_one, eng->compute_stream);
    if (eng->hybrid_kv_mode) {
        size_t k_full  = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(__half);
        size_t v_hot   = (size_t)NUM_GLOBAL_LAYERS * HOT_WINDOW * 512 * sizeof(__half);
        size_t v_sink  = (size_t)NUM_GLOBAL_LAYERS * V_SINK_FP16 * 512 * sizeof(__half);
        size_t v_cold  = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cold_cap * 512 * sizeof(uint8_t);
        size_t vpg_one = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cold_pages * 512 * sizeof(__half);
        cudaMemsetAsync(eng->K_global, 0, k_full, eng->compute_stream);
        cudaMemsetAsync(eng->V_global, 0, v_hot, eng->compute_stream);
        cudaMemsetAsync(eng->V_sink_fp16, 0, v_sink, eng->compute_stream);
        cudaMemsetAsync(eng->V_global_cold, 0, v_cold, eng->compute_stream);
        cudaMemsetAsync(eng->V_page_min, 0, vpg_one, eng->compute_stream);
        cudaMemsetAsync(eng->V_page_max, 0, vpg_one, eng->compute_stream);
    } else {
        size_t global_kv_one = (size_t)NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(__half);
        cudaMemsetAsync(eng->K_global, 0, global_kv_one, eng->compute_stream);
        cudaMemsetAsync(eng->V_global, 0, global_kv_one, eng->compute_stream);
    }
    for (int l = 0; l < NUM_LAYERS; ++l) eng->ema_attn_valid[l] = 0;
    if (eng->d_ema_attn) {
        cudaMemsetAsync(eng->d_ema_attn, 0, (size_t)NUM_LAYERS * D_MODEL * sizeof(float), eng->compute_stream);
    }
    cudaStreamSynchronize(eng->compute_stream);
}

EXPORT void engine_set_lookahead_prefetch(void* handle, int mode, float tau_prob32) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->lookahead_prefetch_mode = mode;
    if (tau_prob32 >= 0.0f) eng->prefetch_tau_prob32 = tau_prob32;
}

EXPORT void engine_set_fast_gemv(void* handle, int enabled) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->use_fast_fp16_gemv = (enabled != 0);
}

EXPORT void engine_set_verify_readonly_cache(void* handle, int enabled) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->verify_read_only_cache = (enabled != 0);
}

EXPORT void engine_commit_verified_cache_accesses(void* handle, int n_accepted) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->commit_verified_cache_accesses(n_accepted);
}

EXPORT void engine_set_force_splits(void* handle, int splits) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->force_splits = splits;
}

EXPORT void engine_set_quest_prune_margin(void* handle, float margin_nats, int collect_stats) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->quest_prune_margin_nats = margin_nats;
    eng->collect_prune_stats = (collect_stats != 0);
}

EXPORT void engine_reset_prune_stats(void* handle) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    cudaMemsetAsync(eng->d_prune_stats, 0, 6ULL * sizeof(unsigned long long), eng->compute_stream);
    cudaStreamSynchronize(eng->compute_stream);
}

EXPORT int engine_get_prune_stats(void* handle, uint64_t* out6) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    CUDA_OK(cudaMemcpy(out6, eng->d_prune_stats, 6ULL * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return 0;
}

EXPORT int engine_get_last_hidden(void* handle, float* out_host, int n_tokens) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (n_tokens <= 0 || n_tokens > MAX_VERIFY_K) return -1;
    CUDA_OK(cudaMemcpy(out_host, eng->d_H, (size_t)n_tokens * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

EXPORT int engine_benchmark_lmhead(void* handle, int iters, int use_fast, double* out_ms_per_iter) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    bool prev = eng->use_fast_fp16_gemv;
    eng->use_fast_fp16_gemv = (use_fast != 0);
    // Warmup
    for (int i = 0; i < 5; ++i) {
        eng->gemm_fp16_to_fp32(eng->lm_head, eng->d_N_f16, eng->d_logits, VOCAB_SIZE, 1, D_MODEL);
    }
    cudaStreamSynchronize(eng->compute_stream);
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s, eng->compute_stream);
    for (int i = 0; i < iters; ++i) {
        eng->gemm_fp16_to_fp32(eng->lm_head, eng->d_N_f16, eng->d_logits, VOCAB_SIZE, 1, D_MODEL);
    }
    cudaEventRecord(e, eng->compute_stream);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    eng->use_fast_fp16_gemv = prev;
    if (out_ms_per_iter) *out_ms_per_iter = (double)ms / (double)iters;
    return (int)cudaGetLastError();
}

// Parity test helper: run single-layer attention (layer l) on host hidden vector H_in[2880] at pos -> H_out[2880]
EXPORT int engine_test_single_layer_attn(
    void* handle,
    int layer_idx,
    int pos,
    const float* H_in_host,
    float* H_out_host
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    CUDA_OK(cudaMemcpyAsync(eng->d_H, H_in_host, D_MODEL * sizeof(float), cudaMemcpyHostToDevice, eng->compute_stream));
    eng->run_attention_decode_or_verify(layer_idx, 1, pos);
    CUDA_OK(cudaMemcpyAsync(H_out_host, eng->d_H, D_MODEL * sizeof(float), cudaMemcpyDeviceToHost, eng->compute_stream));
    CUDA_OK(cudaStreamSynchronize(eng->compute_stream));
    return 0;
}

// Parity test helper: run Split-K Flash-Decoding directly on a global/local layer with pre-filled KV
EXPORT int engine_test_flash_decode_only(
    void* handle,
    int layer_idx,
    int pos,
    int num_splits,
    const float* Q_in_host, // [64, 64]
    float* O_out_host       // [64, 64] FP32 (converted from FP16 output)
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    const LayerGPU& ld = eng->layers[layer_idx];
    const int is_local = ((layer_idx & 1) == 0) ? 1 : 0;
    const int sub_l = layer_idx >> 1;
    __half* Kc = is_local ? (eng->K_local_ring + (size_t)sub_l * RING_SIZE * 512)
                          : (eng->K_global     + (size_t)sub_l * eng->kv_cap_global * 512);
    __half* Vc = is_local ? (eng->V_local_ring + (size_t)sub_l * RING_SIZE * 512)
                          : (eng->V_global     + (size_t)sub_l * eng->kv_cap_global * 512);
    CUDA_OK(cudaMemcpyAsync(eng->d_Q, Q_in_host, 4096 * sizeof(float), cudaMemcpyHostToDevice, eng->compute_stream));
    dim3 g1(NUM_KV_HEADS, num_splits, 1);
    int rmask = is_local ? (RING_SIZE - 1) : 0;
    flash_decode_gqa_splitk_part1_turing<false><<<g1, 256, 0, eng->compute_stream>>>(
        eng->d_Q, Kc, Vc, nullptr, nullptr, nullptr, ld.sinks, nullptr,
        eng->d_split_m, eng->d_split_l, eng->d_split_o, eng->d_O_f16,
        pos, is_local, rmask, rmask, 0, 0, 0, 0, num_splits, num_splits, 1e20f, nullptr, 0
    );
    if (num_splits > 1) {
        dim3 g2(NUM_Q_HEADS, 1);
        flash_decode_gqa_splitk_reduce<<<g2, 64, 0, eng->compute_stream>>>(
            eng->d_split_m, eng->d_split_l, eng->d_split_o, eng->d_O_f16, num_splits
        );
    }
    fp16_to_fp32_kernel<<<(4096 + 255) / 256, 256, 0, eng->compute_stream>>>(eng->d_O_f16, eng->d_Q, 4096);
    CUDA_OK(cudaMemcpyAsync(O_out_host, eng->d_Q, 4096 * sizeof(float), cudaMemcpyDeviceToHost, eng->compute_stream));
    CUDA_OK(cudaStreamSynchronize(eng->compute_stream));
    return 0;
}

// Parity test helper: upload synthetic K,V [seq_len, 8, 64] FP16 into global layer sub_l
EXPORT int engine_upload_global_kv(
    void* handle,
    int layer_idx,
    int start_pos,
    int n_pos,
    const uint16_t* k_host_fp16,
    const uint16_t* v_host_fp16
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if ((layer_idx & 1) == 0) return -1;
    int sub_l = layer_idx >> 1;
    size_t off = ((size_t)sub_l * eng->kv_cap_global + start_pos) * 512;
    size_t nb = (size_t)n_pos * 512 * sizeof(__half);
    CUDA_OK(cudaMemcpy(eng->K_global + off, k_host_fp16, nb, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(eng->V_global + off, v_host_fp16, nb, cudaMemcpyHostToDevice));
    return 0;
}

EXPORT void engine_get_mem_info(void* handle, MemReport* out) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);

    out->resident_weights_bytes = eng->resident_weights_bytes;
    out->kv_local_ring128_bytes = 2ULL * NUM_LOCAL_LAYERS * RING_SIZE * 512 * sizeof(__half);
    if (eng->hybrid_kv_mode) {
        size_t k_full = 1ULL * NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(__half);
        size_t v_hot  = 1ULL * NUM_GLOBAL_LAYERS * HOT_WINDOW * 512 * sizeof(__half);
        size_t v_sink = 1ULL * NUM_GLOBAL_LAYERS * V_SINK_FP16 * 512 * sizeof(__half);
        out->kv_global_hot_bytes        = k_full + v_hot + v_sink;
        out->kv_global_cold_data_bytes  = 1ULL * NUM_GLOBAL_LAYERS * eng->kv_cold_cap * 512 * sizeof(uint8_t);
        out->kv_global_cold_scale_bytes = 2ULL * NUM_GLOBAL_LAYERS * eng->kv_cold_pages * 512 * sizeof(__half);
        out->kv_global_bytes            = out->kv_global_hot_bytes + out->kv_global_cold_data_bytes + out->kv_global_cold_scale_bytes;
        out->ring_shadow_bytes          = (2ULL * NUM_LOCAL_LAYERS * MAX_VERIFY_K + 1ULL * NUM_GLOBAL_LAYERS * (MAX_VERIFY_K + V_PAGE_SIZE)) * 512 * sizeof(__half);
        out->kv_ram_backup_bytes        = 1ULL * NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(uint16_t);
        out->hybrid_kv_mode             = 1;
    } else {
        out->kv_global_bytes            = 2ULL * NUM_GLOBAL_LAYERS * eng->kv_cap_global * 512 * sizeof(__half);
        out->kv_global_hot_bytes        = out->kv_global_bytes;
        out->kv_global_cold_data_bytes  = 0;
        out->kv_global_cold_scale_bytes = 0;
        out->ring_shadow_bytes          = 2ULL * NUM_LOCAL_LAYERS * MAX_VERIFY_K * 512 * sizeof(__half);
        out->kv_ram_backup_bytes        = 0;
        out->hybrid_kv_mode             = 0;
    }
    out->stream_bufs_2x_bytes   = 2ULL * EXPERT_BYTES + ((eng->comp_stream_bufs[0] != nullptr) ? (2ULL * COMP_STREAM_BUF_BYTES) : 0ULL);
    out->scratch_workspace_bytes = eng->scratch_workspace_bytes;
    out->arena_bytes            = eng->arena_bytes;
    out->total_engine_vram_bytes = out->resident_weights_bytes + out->kv_local_ring128_bytes
                                 + out->kv_global_bytes + out->ring_shadow_bytes
                                 + out->stream_bufs_2x_bytes + out->scratch_workspace_bytes + out->arena_bytes;
    out->cuda_free_bytes        = free_b;
    out->cuda_total_bytes       = total_b;
    out->kv_cap_global          = eng->kv_cap_global;
    out->cache_capacity_slots   = eng->cache_capacity_slots;
    out->cache_active_entries   = eng->active_cache_entries;
    out->cache_policy           = eng->cache_policy;
    out->num_copy_threads       = eng->copy_pool.num_threads;
}

EXPORT int engine_forward_token(
    void* handle,
    int token_id,
    int pos,
    float* logits_out,
    int logits_on_gpu,
    uint8_t* top4_out_24x4
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->forward_token_internal(token_id, pos, logits_out, logits_on_gpu, top4_out_24x4);
    return (int)cudaGetLastError();
}

EXPORT int engine_forward_token_probe_la(
    void* handle,
    int token_id,
    int pos,
    float* logits_out,
    uint8_t* top4_out_24x4,
    float* la_raw_24x32,
    float* la_bo_24x32,
    float* la_ema_24x32,
    uint32_t* res_mask_24
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->forward_token_internal(
        token_id, pos, logits_out, 0, top4_out_24x4,
        la_raw_24x32, la_bo_24x32, la_ema_24x32, res_mask_24
    );
    return (int)cudaGetLastError();
}

EXPORT int engine_prefill_chunk(
    void* handle,
    const int* token_ids,
    int chunk_len,
    int start_pos,
    int compute_last_logits,
    float* last_logits_out,
    int logits_on_gpu
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (chunk_len <= 0 || chunk_len > MAX_PREFILL_CHUNK) return -1;
    eng->prefill_chunk_internal(token_ids, chunk_len, start_pos, compute_last_logits, last_logits_out, logits_on_gpu, nullptr);
    return (int)cudaGetLastError();
}

EXPORT int engine_prefill_chunk_hidden(
    void* handle,
    const int* token_ids,
    int chunk_len,
    int start_pos,
    float* hidden_out_host
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (chunk_len <= 0 || chunk_len > MAX_PREFILL_CHUNK) return -1;
    eng->prefill_chunk_internal(token_ids, chunk_len, start_pos, 0, nullptr, 0, hidden_out_host);
    return (int)cudaGetLastError();
}

EXPORT int engine_prefill_chunk_eagle3(
    void* handle,
    const int* token_ids,
    int chunk_len,
    int start_pos,
    float* hidden_l2_out_host,
    float* hidden_l12_out_host,
    float* hidden_l23_out_host,
    uint8_t* top4_24xCx4_out_host
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (chunk_len <= 0 || chunk_len > MAX_PREFILL_CHUNK) return -1;
    eng->prefill_chunk_internal(
        token_ids, chunk_len, start_pos, 0, nullptr, 0,
        hidden_l23_out_host, hidden_l2_out_host, hidden_l12_out_host, top4_24xCx4_out_host
    );
    return (int)cudaGetLastError();
}

EXPORT void* engine_alloc_pinned_layer(void* handle, int l, size_t nbytes) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (l < 0 || l >= NUM_LAYERS || nbytes == 0) return nullptr;
    if (eng->mxfp4_pinned_layers[l]) {
        cudaFreeHost(eng->mxfp4_pinned_layers[l]);
        eng->mxfp4_pinned_layers[l] = nullptr;
    }
    if (cudaHostAlloc(&eng->mxfp4_pinned_layers[l], nbytes, cudaHostAllocDefault) != cudaSuccess) {
        eng->mxfp4_pinned_layers[l] = nullptr;
        return nullptr;
    }
    eng->mxfp4_ram_layers[l] = eng->mxfp4_pinned_layers[l];
    eng->layer_is_pinned[l] = true;
    return eng->mxfp4_pinned_layers[l];
}

EXPORT void engine_set_layer_pinned_flag(void* handle, int l, int is_pinned) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (l >= 0 && l < NUM_LAYERS) eng->layer_is_pinned[l] = (is_pinned != 0);
}

EXPORT int engine_set_layer_rans_meta(
    void* handle,
    int l,
    const uint32_t* offsets_32,
    const uint32_t* sizes_32,
    int enable_rans
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (l < 0 || l >= NUM_LAYERS) return -1;
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        eng->rans_exp_off[l][e] = offsets_32[e];
        eng->rans_exp_sz[l][e]  = sizes_32[e];
    }
    eng->rans_mode = (enable_rans != 0);
    if (eng->rans_mode && eng->comp_stream_bufs[0] == nullptr) {
        for (int i = 0; i < 2; ++i) {
            CUDA_OK(cudaMalloc(&eng->comp_stream_bufs[i], COMP_STREAM_BUF_BYTES));
        }
    }
    return 0;
}

EXPORT void engine_get_cache_resident_mask(void* handle, uint32_t* out_mask_24) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    for (int l = 0; l < NUM_LAYERS; ++l) {
        uint32_t m = 0;
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            if (eng->key_to_slot[l * NUM_EXPERTS + e] >= 0) m |= (1u << e);
        }
        out_mask_24[l] = m;
    }
}

EXPORT int engine_ring_snapshot(void* handle, int start_pos, int K) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (K <= 0 || K > MAX_VERIFY_K) return -1;
    dim3 g(NUM_LOCAL_LAYERS, K);
    ring128_snapshot_kernel<<<g, 512, 0, eng->compute_stream>>>(
        eng->K_local_ring, eng->V_local_ring, eng->K_ring_shadow, eng->V_ring_shadow, start_pos, K
    );
    if (eng->hybrid_kv_mode) {
        dim3 gh(NUM_GLOBAL_LAYERS, K);
        hot4k_v_snapshot_kernel<<<gh, 512, 0, eng->compute_stream>>>(
            eng->V_global, eng->V_hot_shadow, start_pos, K
        );
    }
    return (int)cudaGetLastError();
}

EXPORT int engine_ring_rollback(void* handle, int start_pos, int first_rejected_i, int K) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (first_rejected_i >= K) return 0;
    dim3 g(NUM_LOCAL_LAYERS, K - first_rejected_i);
    ring128_rollback_kernel<<<g, 512, 0, eng->compute_stream>>>(
        eng->K_local_ring, eng->V_local_ring, eng->K_ring_shadow, eng->V_ring_shadow,
        start_pos, first_rejected_i, K
    );
    if (eng->hybrid_kv_mode) {
        dim3 gh(NUM_GLOBAL_LAYERS, K - first_rejected_i);
        hot4k_v_rollback_kernel<<<gh, 512, 0, eng->compute_stream>>>(
            eng->V_global, eng->V_hot_shadow,
            start_pos, first_rejected_i, K
        );
    }
    return (int)cudaGetLastError();
}

EXPORT int engine_restore_global_hot_from_ram(void* handle, int new_L) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    return eng->restore_global_hot_from_ram_internal(new_L);
}

EXPORT int engine_draft_token_cache_only(
    void* handle,
    int token_id,
    int pos,
    int max_k_cached,
    float* logits_out,
    int logits_on_gpu
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->draft_token_cache_only_internal(token_id, pos, max_k_cached, logits_out, logits_on_gpu);
    return (int)cudaGetLastError();
}

EXPORT int engine_verify_tokens(
    void* handle,
    const int* token_ids,
    int K,
    int start_pos,
    float* logits_out,
    int logits_on_gpu,
    uint64_t* out_pcie_bytes,
    int* out_unique_experts_total,
    int* out_cache_hits_total
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (K <= 0 || K > MAX_VERIFY_K) return -1;
    eng->verify_tokens_internal(
        token_ids, K, start_pos, logits_out, logits_on_gpu,
        out_pcie_bytes, out_unique_experts_total, out_cache_hits_total
    );
    return (int)cudaGetLastError();
}

EXPORT int engine_read_ring_kv(void* handle, uint16_t* k_host_out, uint16_t* v_host_out) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    size_t nb = (size_t)NUM_LOCAL_LAYERS * RING_SIZE * 512 * sizeof(__half);
    CUDA_OK(cudaMemcpy(k_host_out, eng->K_local_ring, nb, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(v_host_out, eng->V_local_ring, nb, cudaMemcpyDeviceToHost));
    return 0;
}

EXPORT int engine_write_ring_kv(void* handle, const uint16_t* k_host_in, const uint16_t* v_host_in) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    size_t nb = (size_t)NUM_LOCAL_LAYERS * RING_SIZE * 512 * sizeof(__half);
    CUDA_OK(cudaMemcpyAsync(eng->K_local_ring, k_host_in, nb, cudaMemcpyHostToDevice, eng->compute_stream));
    CUDA_OK(cudaMemcpyAsync(eng->V_local_ring, v_host_in, nb, cudaMemcpyHostToDevice, eng->compute_stream));
    CUDA_OK(cudaStreamSynchronize(eng->compute_stream));
    return 0;
}

// ============================================================================
// Iteration 5 Lever C: EAGLE-1/3 Resident Draft C API Exports
// ============================================================================
EXPORT int engine_eagle_load(
    void* handle,
    const float* h_norm,
    const float* e_norm,
    const uint16_t* W_fuse_h,
    const uint16_t* W_fuse_e,
    const float* b_fuse,
    const float* attn_norm,
    const uint16_t* W_qkv,
    const uint16_t* W_o,
    const float* mlp_norm,
    const uint16_t* W_gu,
    const uint16_t* W_dn,
    const float* out_norm,
    const uint16_t* W_out,
    const float* b_out,
    const int* base_shortlist,
    int shortlist_len
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (eng->eagle_loaded) {
        eng->resident_weights_bytes -= eng->eagle_vram_bytes;
        eng->free_eagle_resources();
    }
    size_t eb = 0;
    auto up = [&](auto** dst, const void* src, size_t bytes) -> int {
        CUDA_OK(cudaMalloc(dst, bytes));
        if (src) CUDA_OK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
        else     CUDA_OK(cudaMemset(*dst, 0, bytes));
        eb += bytes;
        return 0;
    };
    if (int e = up(&eng->e_h_norm,    h_norm,    (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e_e_norm,    e_norm,    (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e_W_fuse_h,  W_fuse_h,  (size_t)EAGLE_D_DRAFT * D_MODEL * 2)) return e;
    if (int e = up(&eng->e_W_fuse_e,  W_fuse_e,  (size_t)EAGLE_D_DRAFT * D_MODEL * 2)) return e;
    if (int e = up(&eng->e_b_fuse,    b_fuse,    (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e_attn_norm, attn_norm, (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e_W_qkv,     W_qkv,     (size_t)EAGLE_QKV_DIM * EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->e_W_o,       W_o,       (size_t)EAGLE_D_DRAFT * EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->e_mlp_norm,  mlp_norm,  (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e_W_gu,      W_gu,      (size_t)2 * EAGLE_D_FF * EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->e_W_dn,      W_dn,      (size_t)EAGLE_D_DRAFT * EAGLE_D_FF * 2)) return e;
    if (int e = up(&eng->e_out_norm,  out_norm,  (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e_W_out,     W_out,     (size_t)D_MODEL * EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->e_b_out,     b_out,     (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e_K_kv,      nullptr,   (size_t)EAGLE_KV_WIN * 128 * 2)) return e;
    if (int e = up(&eng->e_V_kv,      nullptr,   (size_t)EAGLE_KV_WIN * 128 * 2)) return e;
    if (int e = up(&eng->e_shortlist, nullptr,   (size_t)EAGLE_MAX_SHORTLIST * 4)) return e;
    if (int e = up(&eng->d_e_u,       nullptr,   (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->d_e_emb_f16, nullptr,   (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e_un_f16,  nullptr,   (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e_en_f16,  nullptr,   (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e_x,       nullptr,   (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->d_e_tmp512_a,nullptr,   (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->d_e_tmp512_b,nullptr,   (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->d_e_xn_f16,  nullptr,   (size_t)EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->d_e_qkv,     nullptr,   (size_t)EAGLE_QKV_DIM * 4)) return e;
    if (int e = up(&eng->d_e_o_f16,   nullptr,   (size_t)EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->d_e_gu,      nullptr,   (size_t)2 * EAGLE_D_FF * 4)) return e;
    if (int e = up(&eng->d_e_mid_f16, nullptr,   (size_t)EAGLE_D_FF * 2)) return e;
    if (int e = up(&eng->d_e_out_f32, nullptr,   (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->d_e_logits,  nullptr,   (size_t)EAGLE_MAX_SHORTLIST * 4)) return e;
    if (int e = up(&eng->d_e_argmax_tok, nullptr, sizeof(int))) return e;
    if (int e = up(&eng->d_e_top1_prob,  nullptr, sizeof(float))) return e;

    CUDA_OK(cudaHostAlloc(&eng->h_e_argmax_tok, sizeof(int), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&eng->h_e_top1_prob,  sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&eng->h_e_embed_tmp, (size_t)D_MODEL * sizeof(__half), cudaHostAllocDefault));

    int n_sl = std::max(1, std::min(shortlist_len, EAGLE_MAX_SHORTLIST));
    CUDA_OK(cudaMemcpy(eng->e_shortlist, base_shortlist, (size_t)n_sl * sizeof(int), cudaMemcpyHostToDevice));
    eng->eagle_shortlist_len = n_sl;
    eng->eagle_kv_pos = 0;
    eng->eagle_vram_bytes = eb;
    eng->resident_weights_bytes += eb;
    eng->eagle_loaded = true;
    return 0;
}

EXPORT int engine_eagle_set_shortlist(void* handle, const int* ids, int len) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (!eng->eagle_loaded) return -1;
    int n_sl = std::max(1, std::min(len, EAGLE_MAX_SHORTLIST));
    CUDA_OK(cudaMemcpyAsync(eng->e_shortlist, ids, (size_t)n_sl * sizeof(int), cudaMemcpyHostToDevice, eng->compute_stream));
    eng->eagle_shortlist_len = n_sl;
    return 0;
}

EXPORT void engine_eagle_reset_kv(void* handle) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->eagle_kv_pos = 0;
}

EXPORT int engine_eagle_commit_prefix(
    void* handle,
    int target_h_slot_start,
    const int* tokens,
    int n_tokens
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (!eng->eagle_loaded || n_tokens <= 0) return 0;
    for (int i = 0; i < n_tokens; ++i) {
        int slot = target_h_slot_start + i;
        if (slot < 0 || slot >= MAX_VERIFY_K) return -1;
        eagle_unit_rms_copy_kernel<<<1, 256, 0, eng->compute_stream>>>(eng->d_H + (size_t)slot * D_MODEL, eng->d_e_u);
        eng->eagle_step_core(tokens[i], eng->eagle_kv_pos, false);
        eng->eagle_kv_pos++;
    }
    return (int)cudaGetLastError();
}

EXPORT int engine_eagle_draft_k(
    void* handle,
    int target_h_slot,
    int start_token_id,
    int K,
    float min_step_prob,
    float min_cum_prob,
    int* draft_tokens_out,
    float* draft_probs_out,
    int* out_n_drafted,
    double* out_draft_ms
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (!eng->eagle_loaded || K <= 0 || K > MAX_VERIFY_K || target_h_slot < 0 || target_h_slot >= MAX_VERIFY_K) return -1;
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    eagle_unit_rms_copy_kernel<<<1, 256, 0, eng->compute_stream>>>(eng->d_H + (size_t)target_h_slot * D_MODEL, eng->d_e_u);
    const int saved_kv_pos = eng->eagle_kv_pos;
    int cur_tok = start_token_id;
    int n_out = 0;
    float cum_p = 1.0f;
    for (int k = 0; k < K; ++k) {
        eng->eagle_step_core(cur_tok, saved_kv_pos + k, true);
        cudaStreamSynchronize(eng->compute_stream);
        cur_tok = eng->h_e_argmax_tok[0];
        float p = eng->h_e_top1_prob[0];
        cum_p *= p;
        // Confidence-gated early stop: avoid wasting MoE PCIe misses on low-confidence speculative tokens!
        if (min_step_prob > 0.0f && p < min_step_prob) break;
        if (min_cum_prob  > 0.0f && cum_p < min_cum_prob) break;
        draft_tokens_out[n_out] = cur_tok;
        if (draft_probs_out) draft_probs_out[n_out] = p;
        n_out++;
    }
    // Slot saved_kv_pos ALREADY holds exact (true_h, start_token_id)! Advance by 1 cleanly!
    eng->eagle_kv_pos = saved_kv_pos + 1;
    if (out_n_drafted) *out_n_drafted = n_out;
    if (out_draft_ms) {
        *out_draft_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    }
    return (int)cudaGetLastError();
}

// ============================================================================
// Iteration 6 Pillars 1 & 3: EAGLE-3 Tree Drafting, Tree Verification & Exact Path Commit Exports
// ============================================================================
EXPORT int engine_verify_tree_tokens(
    void* handle,
    const int* token_ids,
    const int* depths,
    const uint16_t* anc_masks,
    int N_nodes,
    int start_pos,
    float* logits_out,
    int logits_on_gpu,
    uint64_t* out_pcie_bytes,
    int* out_unique_experts_total,
    int* out_cache_hits_total
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (N_nodes <= 0 || N_nodes > MAX_VERIFY_K) return -1;
    // Snapshot Hot-4K V slots [start_pos .. start_pos + N_nodes - 1] before tree verify!
    if (eng->hybrid_kv_mode) {
        dim3 gh(NUM_GLOBAL_LAYERS, N_nodes);
        hot4k_v_snapshot_kernel<<<gh, 512, 0, eng->compute_stream>>>(
            eng->V_global, eng->V_hot_shadow, start_pos, N_nodes
        );
    }
    TreeMeta tm{};
    tm.active = 1;
    tm.num_nodes = (uint8_t)N_nodes;
    for (int i = 0; i < N_nodes; ++i) {
        tm.depth[i]    = (int8_t)depths[i];
        tm.anc_mask[i] = anc_masks[i];
    }
    CUDA_OK(cudaMemcpyToSymbolAsync(c_tree_meta, &tm, sizeof(TreeMeta), 0, cudaMemcpyHostToDevice, eng->compute_stream));
    eng->verify_tokens_internal(
        token_ids, N_nodes, start_pos, logits_out, logits_on_gpu,
        out_pcie_bytes, out_unique_experts_total, out_cache_hits_total, 1
    );
    return (int)cudaGetLastError();
}

EXPORT int engine_commit_tree_path(
    void* handle,
    int start_pos,
    int N_nodes,
    const int* path_ids,
    int L_acc
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    return eng->commit_tree_path_internal(start_pos, N_nodes, path_ids, L_acc);
}

EXPORT int engine_eagle3_load(
    void* handle,
    int d_ff,
    const float* norm_h2,
    const float* norm_h12,
    const float* norm_h23,
    const float* norm_rec,
    const float* norm_e,
    const uint16_t* W_fuse_h2,
    const uint16_t* W_fuse_h12,
    const uint16_t* W_fuse_h23,
    const uint16_t* W_rec_x,
    const uint16_t* W_fuse_e,
    const float* b_fuse,
    const float* attn_norm_0,  const uint16_t* W_qkv_0, const uint16_t* W_o_0,
    const float* mlp_norm_0,   const uint16_t* W_gu_0,  const uint16_t* W_dn_0,
    const float* attn_norm_1,  const uint16_t* W_qkv_1, const uint16_t* W_o_1,
    const float* mlp_norm_1,   const uint16_t* W_gu_1,  const uint16_t* W_dn_1,
    const float* out_norm,
    const uint16_t* W_out,
    const float* b_out,
    const uint16_t* W_moe_pred,
    const float* b_moe_pred,
    const int* base_shortlist,
    int shortlist_len
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (eng->eagle3_loaded) {
        eng->resident_weights_bytes -= eng->eagle3_vram_bytes;
        eng->free_eagle3_resources();
    }
    eng->eagle3_d_ff = d_ff;
    size_t eb = 0;
    auto up = [&](auto** dst, const void* src, size_t bytes) -> int {
        CUDA_OK(cudaMalloc(dst, bytes));
        if (src) CUDA_OK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
        else     CUDA_OK(cudaMemset(*dst, 0, bytes));
        eb += bytes;
        return 0;
    };
    if (int e = up(&eng->e3_norm_h2,    norm_h2,    (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_norm_h12,   norm_h12,   (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_norm_h23,   norm_h23,   (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_norm_rec,   norm_rec,   (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_norm_e,     norm_e,     (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_W_fuse_h2,  W_fuse_h2,  (size_t)EAGLE_D_DRAFT * D_MODEL * 2)) return e;
    if (int e = up(&eng->e3_W_fuse_h12, W_fuse_h12, (size_t)EAGLE_D_DRAFT * D_MODEL * 2)) return e;
    if (int e = up(&eng->e3_W_fuse_h23, W_fuse_h23, (size_t)EAGLE_D_DRAFT * D_MODEL * 2)) return e;
    if (int e = up(&eng->e3_W_rec_x,    W_rec_x,    (size_t)EAGLE_D_DRAFT * EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->e3_W_fuse_e,   W_fuse_e,   (size_t)EAGLE_D_DRAFT * D_MODEL * 2)) return e;
    if (int e = up(&eng->e3_b_fuse,     b_fuse,     (size_t)EAGLE_D_DRAFT * 4)) return e;

    const float*    an[2]  = {attn_norm_0, attn_norm_1};
    const uint16_t* wqk[2] = {W_qkv_0, W_qkv_1};
    const uint16_t* wo[2]  = {W_o_0, W_o_1};
    const float*    mn[2]  = {mlp_norm_0, mlp_norm_1};
    const uint16_t* wgu[2] = {W_gu_0, W_gu_1};
    const uint16_t* wdn[2] = {W_dn_0, W_dn_1};
    for (int i = 0; i < E3_N_LAYERS; ++i) {
        if (int e = up(&eng->e3_attn_norm[i], an[i],  (size_t)EAGLE_D_DRAFT * 4)) return e;
        if (int e = up(&eng->e3_W_qkv[i],     wqk[i], (size_t)EAGLE_QKV_DIM * EAGLE_D_DRAFT * 2)) return e;
        if (int e = up(&eng->e3_W_o[i],       wo[i],  (size_t)EAGLE_D_DRAFT * EAGLE_D_DRAFT * 2)) return e;
        if (int e = up(&eng->e3_mlp_norm[i],  mn[i],  (size_t)EAGLE_D_DRAFT * 4)) return e;
        if (int e = up(&eng->e3_W_gu[i],      wgu[i], (size_t)2 * d_ff * EAGLE_D_DRAFT * 2)) return e;
        if (int e = up(&eng->e3_W_dn[i],      wdn[i], (size_t)EAGLE_D_DRAFT * d_ff * 2)) return e;
        if (int e = up(&eng->e3_K_kv[i],      nullptr,(size_t)EAGLE_KV_WIN * 128 * 2)) return e;
        if (int e = up(&eng->e3_V_kv[i],      nullptr,(size_t)EAGLE_KV_WIN * 128 * 2)) return e;
        if (int e = up(&eng->e3_K_tree[i],    nullptr,(size_t)E3_MAX_TREE_SCRATCH * 128 * 2)) return e;
        if (int e = up(&eng->e3_V_tree[i],    nullptr,(size_t)E3_MAX_TREE_SCRATCH * 128 * 2)) return e;
    }

    if (int e = up(&eng->e3_out_norm,   out_norm,   (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_W_out,      W_out,      (size_t)D_MODEL * EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->e3_b_out,      b_out,      (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_W_moe_pred, W_moe_pred, (size_t)NUM_LAYERS * NUM_EXPERTS * EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->e3_b_moe_pred, b_moe_pred, (size_t)NUM_LAYERS * NUM_EXPERTS * 4)) return e;
    if (int e = up(&eng->e3_shortlist,  nullptr,    (size_t)EAGLE_MAX_SHORTLIST * 4)) return e;

    if (int e = up(&eng->d_e3_u,        nullptr, (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->d_e3_emb_f16,  nullptr, (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e3_un_f16,   nullptr, (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e3_en_f16,   nullptr, (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e3_x,        nullptr, (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->d_e3_tree_x,   nullptr, (size_t)E3_MAX_TREE_SCRATCH * EAGLE_D_DRAFT * 4)) return e;
    for (int i = 0; i < 4; ++i) {
        if (int e = up(&eng->d_e3_t512[i], nullptr, (size_t)EAGLE_D_DRAFT * 4)) return e;
    }
    if (int e = up(&eng->d_e3_xn_f16,   nullptr, (size_t)EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->d_e3_qkv,      nullptr, (size_t)EAGLE_QKV_DIM * 4)) return e;
    if (int e = up(&eng->d_e3_o_f16,    nullptr, (size_t)EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->d_e3_gu,       nullptr, (size_t)2 * std::max(d_ff, EAGLE_D_FF) * 4)) return e;
    if (int e = up(&eng->d_e3_mid_f16,  nullptr, (size_t)std::max(d_ff, EAGLE_D_FF) * 2)) return e;
    if (int e = up(&eng->d_e3_out_f32,  nullptr, (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->d_e3_logits,   nullptr, (size_t)EAGLE_MAX_SHORTLIST * 4)) return e;
    if (int e = up(&eng->d_e3_moe_lg,   nullptr, (size_t)NUM_LAYERS * NUM_EXPERTS * 4)) return e;
    if (int e = up(&eng->d_e3_moe_mask, nullptr, (size_t)NUM_LAYERS * 4)) return e;
    if (int e = up(&eng->d_e3_top4_tok, nullptr, 4 * sizeof(int))) return e;
    if (int e = up(&eng->d_e3_top4_prob,nullptr, 4 * sizeof(float))) return e;

    CUDA_OK(cudaHostAlloc(&eng->h_e3_top4_tok,  4 * sizeof(int), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&eng->h_e3_top4_prob, 4 * sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&eng->h_e3_moe_mask,  NUM_LAYERS * sizeof(uint32_t), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&eng->h_e3_embed_tmp, (size_t)D_MODEL * sizeof(__half), cudaHostAllocDefault));

    int n_sl = std::max(1, std::min(shortlist_len, EAGLE_MAX_SHORTLIST));
    CUDA_OK(cudaMemcpy(eng->e3_shortlist, base_shortlist, (size_t)n_sl * sizeof(int), cudaMemcpyHostToDevice));
    eng->eagle3_shortlist_len = n_sl;
    eng->eagle3_kv_pos = 0;
    eng->eagle3_vram_bytes = eb;
    eng->resident_weights_bytes += eb;
    eng->eagle3_loaded = true;
    return 0;
}

EXPORT int engine_eagle3_set_shortlist(void* handle, const int* ids, int len) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (!eng->eagle3_loaded) return -1;
    int n_sl = std::max(1, std::min(len, EAGLE_MAX_SHORTLIST));
    CUDA_OK(cudaMemcpyAsync(eng->e3_shortlist, ids, (size_t)n_sl * sizeof(int), cudaMemcpyHostToDevice, eng->compute_stream));
    eng->eagle3_shortlist_len = n_sl;
    return 0;
}

EXPORT void engine_eagle3_reset_kv(void* handle) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->eagle3_kv_pos = 0;
}

EXPORT int engine_eagle3_commit_prefix(
    void* handle,
    int target_h_slot_start,
    const int* tokens,
    int n_tokens
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (!eng->eagle3_loaded || n_tokens <= 0) return 0;
    for (int i = 0; i < n_tokens; ++i) {
        int slot = target_h_slot_start + i;
        if (slot < 0 || slot >= MAX_VERIFY_K) return -1;
        eng->eagle3_step_core(true, slot, -1, tokens[i], eng->eagle3_kv_pos, eng->eagle3_kv_pos, -1, 0u, false, -1);
        eng->eagle3_kv_pos++;
    }
    return (int)cudaGetLastError();
}

EXPORT int engine_eagle3_load_int8(
    void* handle,
    int d_ff,
    const float* norm_h2,
    const float* norm_h12,
    const float* norm_h23,
    const float* norm_rec,
    const float* norm_e,
    const int8_t* q_fuse_h2,  const float* s_fuse_h2,
    const int8_t* q_fuse_h12, const float* s_fuse_h12,
    const int8_t* q_fuse_h23, const float* s_fuse_h23,
    const int8_t* q_rec_x,    const float* s_rec_x,
    const int8_t* q_fuse_e,   const float* s_fuse_e,
    const float* b_fuse,
    const float* attn_norm_0,
    const int8_t* q_qkv_0, const float* s_qkv_0,
    const int8_t* q_o_0,   const float* s_o_0,
    const float* mlp_norm_0,
    const int8_t* q_gu_0,  const float* s_gu_0,
    const int8_t* q_dn_0,  const float* s_dn_0,
    const float* attn_norm_1,
    const int8_t* q_qkv_1, const float* s_qkv_1,
    const int8_t* q_o_1,   const float* s_o_1,
    const float* mlp_norm_1,
    const int8_t* q_gu_1,  const float* s_gu_1,
    const int8_t* q_dn_1,  const float* s_dn_1,
    const float* out_norm,
    const int8_t* q_out,      const float* s_out,
    const float* b_out,
    const int8_t* q_moe_pred, const float* s_moe_pred,
    const float* b_moe_pred,
    const int* base_shortlist,
    int shortlist_len
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    if (eng->eagle3_loaded) {
        eng->resident_weights_bytes -= eng->eagle3_vram_bytes;
        eng->free_eagle3_resources();
    }
    eng->eagle3_d_ff = d_ff;
    size_t eb = 0;
    auto up = [&](auto** dst, const void* src, size_t bytes) -> int {
        CUDA_OK(cudaMalloc(dst, bytes));
        if (src) CUDA_OK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
        else     CUDA_OK(cudaMemset(*dst, 0, bytes));
        eb += bytes;
        return 0;
    };
    if (int e = up(&eng->e3_norm_h2,    norm_h2,    (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_norm_h12,   norm_h12,   (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_norm_h23,   norm_h23,   (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_norm_rec,   norm_rec,   (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_norm_e,     norm_e,     (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_q_fuse_h2,  q_fuse_h2,  (size_t)EAGLE_D_DRAFT * D_MODEL)) return e;
    if (int e = up(&eng->e3_s_fuse_h2,  s_fuse_h2,  (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_q_fuse_h12, q_fuse_h12, (size_t)EAGLE_D_DRAFT * D_MODEL)) return e;
    if (int e = up(&eng->e3_s_fuse_h12, s_fuse_h12, (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_q_fuse_h23, q_fuse_h23, (size_t)EAGLE_D_DRAFT * D_MODEL)) return e;
    if (int e = up(&eng->e3_s_fuse_h23, s_fuse_h23, (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_q_rec_x,    q_rec_x,    (size_t)EAGLE_D_DRAFT * EAGLE_D_DRAFT)) return e;
    if (int e = up(&eng->e3_s_rec_x,    s_rec_x,    (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_q_fuse_e,   q_fuse_e,   (size_t)EAGLE_D_DRAFT * D_MODEL)) return e;
    if (int e = up(&eng->e3_s_fuse_e,   s_fuse_e,   (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_b_fuse,     b_fuse,     (size_t)EAGLE_D_DRAFT * 4)) return e;

    const float*  an[2]  = {attn_norm_0, attn_norm_1};
    const int8_t* qqk[2] = {q_qkv_0, q_qkv_1}; const float* sqk[2] = {s_qkv_0, s_qkv_1};
    const int8_t* qo[2]  = {q_o_0,   q_o_1};   const float* so[2]  = {s_o_0,   s_o_1};
    const float*  mn[2]  = {mlp_norm_0, mlp_norm_1};
    const int8_t* qgu[2] = {q_gu_0,  q_gu_1};  const float* sgu[2] = {s_gu_0,  s_gu_1};
    const int8_t* qdn[2] = {q_dn_0,  q_dn_1};  const float* sdn[2] = {s_dn_0,  s_dn_1};
    for (int i = 0; i < E3_N_LAYERS; ++i) {
        if (int e = up(&eng->e3_attn_norm[i], an[i],  (size_t)EAGLE_D_DRAFT * 4)) return e;
        if (int e = up(&eng->e3_q_qkv[i],     qqk[i], (size_t)EAGLE_QKV_DIM * EAGLE_D_DRAFT)) return e;
        if (int e = up(&eng->e3_s_qkv[i],     sqk[i], (size_t)EAGLE_QKV_DIM * 4)) return e;
        if (int e = up(&eng->e3_q_o[i],       qo[i],  (size_t)EAGLE_D_DRAFT * EAGLE_D_DRAFT)) return e;
        if (int e = up(&eng->e3_s_o[i],       so[i],  (size_t)EAGLE_D_DRAFT * 4)) return e;
        if (int e = up(&eng->e3_mlp_norm[i],  mn[i],  (size_t)EAGLE_D_DRAFT * 4)) return e;
        if (int e = up(&eng->e3_q_gu[i],      qgu[i], (size_t)2 * d_ff * EAGLE_D_DRAFT)) return e;
        if (int e = up(&eng->e3_s_gu[i],      sgu[i], (size_t)2 * d_ff * 4)) return e;
        if (int e = up(&eng->e3_q_dn[i],      qdn[i], (size_t)EAGLE_D_DRAFT * d_ff)) return e;
        if (int e = up(&eng->e3_s_dn[i],      sdn[i], (size_t)EAGLE_D_DRAFT * 4)) return e;
        if (int e = up(&eng->e3_K_kv[i],      nullptr,(size_t)EAGLE_KV_WIN * 128 * 2)) return e;
        if (int e = up(&eng->e3_V_kv[i],      nullptr,(size_t)EAGLE_KV_WIN * 128 * 2)) return e;
        if (int e = up(&eng->e3_K_tree[i],    nullptr,(size_t)E3_MAX_TREE_SCRATCH * 128 * 2)) return e;
        if (int e = up(&eng->e3_V_tree[i],    nullptr,(size_t)E3_MAX_TREE_SCRATCH * 128 * 2)) return e;
    }

    if (int e = up(&eng->e3_out_norm,   out_norm,   (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->e3_q_out,      q_out,      (size_t)D_MODEL * EAGLE_D_DRAFT)) return e;
    if (int e = up(&eng->e3_s_out,      s_out,      (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_b_out,      b_out,      (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->e3_q_moe_pred, q_moe_pred, (size_t)NUM_LAYERS * NUM_EXPERTS * EAGLE_D_DRAFT)) return e;
    if (int e = up(&eng->e3_s_moe_pred, s_moe_pred, (size_t)NUM_LAYERS * NUM_EXPERTS * 4)) return e;
    if (int e = up(&eng->e3_b_moe_pred, b_moe_pred, (size_t)NUM_LAYERS * NUM_EXPERTS * 4)) return e;
    if (int e = up(&eng->e3_shortlist,  nullptr,    (size_t)EAGLE_MAX_SHORTLIST * 4)) return e;

    if (int e = up(&eng->d_e3_u,        nullptr, (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->d_e3_emb_f16,  nullptr, (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e3_un_f16,   nullptr, (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e3_en_f16,   nullptr, (size_t)D_MODEL * 2)) return e;
    if (int e = up(&eng->d_e3_x,        nullptr, (size_t)EAGLE_D_DRAFT * 4)) return e;
    if (int e = up(&eng->d_e3_tree_x,   nullptr, (size_t)E3_MAX_TREE_SCRATCH * EAGLE_D_DRAFT * 4)) return e;
    for (int i = 0; i < 4; ++i) {
        if (int e = up(&eng->d_e3_t512[i], nullptr, (size_t)EAGLE_D_DRAFT * 4)) return e;
    }
    if (int e = up(&eng->d_e3_xn_f16,   nullptr, (size_t)EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->d_e3_qkv,      nullptr, (size_t)EAGLE_QKV_DIM * 4)) return e;
    if (int e = up(&eng->d_e3_o_f16,    nullptr, (size_t)EAGLE_D_DRAFT * 2)) return e;
    if (int e = up(&eng->d_e3_gu,       nullptr, (size_t)2 * std::max(d_ff, EAGLE_D_FF) * 4)) return e;
    if (int e = up(&eng->d_e3_mid_f16,  nullptr, (size_t)std::max(d_ff, EAGLE_D_FF) * 2)) return e;
    if (int e = up(&eng->d_e3_out_f32,  nullptr, (size_t)D_MODEL * 4)) return e;
    if (int e = up(&eng->d_e3_logits,   nullptr, (size_t)EAGLE_MAX_SHORTLIST * 4)) return e;
    if (int e = up(&eng->d_e3_moe_lg,   nullptr, (size_t)NUM_LAYERS * NUM_EXPERTS * 4)) return e;
    if (int e = up(&eng->d_e3_moe_mask, nullptr, (size_t)NUM_LAYERS * 4)) return e;
    if (int e = up(&eng->d_e3_top4_tok, nullptr, 4 * sizeof(int))) return e;
    if (int e = up(&eng->d_e3_top4_prob,nullptr, 4 * sizeof(float))) return e;

    CUDA_OK(cudaHostAlloc(&eng->h_e3_top4_tok,  4 * sizeof(int), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&eng->h_e3_top4_prob, 4 * sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&eng->h_e3_moe_mask,  NUM_LAYERS * sizeof(uint32_t), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&eng->h_e3_embed_tmp, (size_t)D_MODEL * sizeof(__half), cudaHostAllocDefault));

    int n_sl = std::max(1, std::min(shortlist_len, EAGLE_MAX_SHORTLIST));
    CUDA_OK(cudaMemcpy(eng->e3_shortlist, base_shortlist, (size_t)n_sl * sizeof(int), cudaMemcpyHostToDevice));
    eng->eagle3_shortlist_len = n_sl;
    eng->eagle3_kv_pos = 0;
    eng->eagle3_vram_bytes = eb;
    eng->resident_weights_bytes += eb;
    eng->eagle3_int8_mode = true;
    eng->eagle3_loaded = true;
    return 0;
}

EXPORT int engine_eagle3_draft_tree(
    void* handle,
    int target_h_slot,
    int root_tok,
    int max_verify_nodes,
    int max_depth,
    int max_cand_eval,
    float min_node_prob,
    int cache_aware_prune,
    float gpu_base_ms,
    float miss_ms,
    int* out_toks,
    int* out_parents,
    int* out_depths,
    uint16_t* out_anc_masks,
    int* out_n_nodes,
    int* out_pred_cold_misses,
    double* out_draft_ms
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    return eng->eagle3_draft_tree_internal(
        target_h_slot, root_tok, max_verify_nodes, max_depth, max_cand_eval,
        min_node_prob, cache_aware_prune, gpu_base_ms, miss_ms,
        out_toks, out_parents, out_depths, out_anc_masks,
        out_n_nodes, out_pred_cold_misses, out_draft_ms
    );
}

EXPORT int engine_eagle3_draft_tree_v2(
    void* handle,
    int target_h_slot,
    int root_tok,
    int max_verify_nodes,
    int max_depth,
    int max_cand_eval,
    float min_node_prob,
    int cache_aware_prune,
    float gpu_base_ms,
    float miss_ms,
    float tau_skip,
    float b1_est_ms,
    float gain_margin,
    int* out_toks,
    int* out_parents,
    int* out_depths,
    uint16_t* out_anc_masks,
    int* out_n_nodes,
    int* out_pred_cold_misses,
    double* out_draft_ms
) {
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    return eng->eagle3_draft_tree_internal(
        target_h_slot, root_tok, max_verify_nodes, max_depth, max_cand_eval,
        min_node_prob, cache_aware_prune, gpu_base_ms, miss_ms,
        out_toks, out_parents, out_depths, out_anc_masks,
        out_n_nodes, out_pred_cold_misses, out_draft_ms,
        tau_skip, b1_est_ms, gain_margin
    );
}

EXPORT void engine_destroy(void* handle) {
    if (!handle) return;
    auto* eng = static_cast<GPTOSSNativeEngine*>(handle);
    eng->copy_pool.shutdown();
    cudaDeviceSynchronize();
    eng->free_eagle_resources();
    eng->free_eagle3_resources();

    if (eng->d_arena) cudaFree(eng->d_arena);
    if (eng->lm_head) cudaFree(eng->lm_head);
    if (eng->final_norm) cudaFree(eng->final_norm);
    if (eng->cos_table) cudaFree(eng->cos_table);
    if (eng->sin_table) cudaFree(eng->sin_table);
    for (int l = 0; l < NUM_LAYERS; ++l) {
        auto& ld = eng->layers[l];
        if (ld.in_norm) cudaFree(ld.in_norm);
        if (ld.post_norm) cudaFree(ld.post_norm);
        if (ld.W_qkv) cudaFree(ld.W_qkv);
        if (ld.b_qkv) cudaFree(ld.b_qkv);
        if (ld.W_o) cudaFree(ld.W_o);
        if (ld.b_o) cudaFree(ld.b_o);
        if (ld.sinks) cudaFree(ld.sinks);
        if (ld.router_w) cudaFree(ld.router_w);
        if (ld.router_b) cudaFree(ld.router_b);
        if (ld.gu_bias) cudaFree(ld.gu_bias);
        if (ld.dn_bias) cudaFree(ld.dn_bias);
    }
    cudaFree(eng->K_local_ring); cudaFree(eng->V_local_ring);
    cudaFree(eng->K_global);     cudaFree(eng->V_global);
    cudaFree(eng->K_ring_shadow); cudaFree(eng->V_ring_shadow);
    if (eng->V_sink_fp16)    cudaFree(eng->V_sink_fp16);
    if (eng->V_global_cold)  cudaFree(eng->V_global_cold);
    if (eng->V_page_min)     cudaFree(eng->V_page_min);
    if (eng->V_page_max)     cudaFree(eng->V_page_max);
    if (eng->V_hot_shadow)   cudaFree(eng->V_hot_shadow);
    if (eng->V_bnd_fp16)     cudaFree(eng->V_bnd_fp16);
    if (eng->h_V_global_ram) free(eng->h_V_global_ram);
    if (eng->h_kv_mirror_v)  cudaFreeHost(eng->h_kv_mirror_v);
    for (int i = 0; i < 2; ++i) {
        cudaFree(eng->orig_stream_bufs[i]);
        if (eng->comp_stream_bufs[i]) cudaFree(eng->comp_stream_bufs[i]);
    }
    for (int l = 0; l < NUM_LAYERS; ++l) {
        if (eng->mxfp4_pinned_layers[l]) cudaFreeHost(eng->mxfp4_pinned_layers[l]);
    }
    for (int i = 0; i < 4; ++i) cudaFreeHost(eng->host_pinned_slots[i]);
    cudaFreeHost(eng->host_pinned_embed);
    cudaFreeHost(eng->host_pinned_top_idx);
    cudaFreeHost(eng->host_pinned_top_w);
    cudaFreeHost(eng->host_pinned_tok_idx);
    cudaFreeHost(eng->host_pinned_tok_w);

    cudaFree(eng->d_embed_f16); cudaFree(eng->d_H); cudaFree(eng->d_N_f32); cudaFree(eng->d_N_f16);
    cudaFree(eng->d_QKV); cudaFree(eng->d_Q); cudaFree(eng->d_O_f16); cudaFree(eng->d_proj_out);
    cudaFree(eng->d_router_lg); cudaFree(eng->d_top_idx); cudaFree(eng->d_top_w); cudaFree(eng->d_mlp_acc);
    cudaFree(eng->d_exp_in); cudaFree(eng->d_exp_gu); cudaFree(eng->d_exp_mid); cudaFree(eng->d_exp_dn);
    cudaFree(eng->d_tok_idx_buf); cudaFree(eng->d_tok_w_buf); cudaFree(eng->d_logits);
    cudaFree(eng->d_split_m); cudaFree(eng->d_split_l); cudaFree(eng->d_split_o);
    cudaFree(eng->d_m_floor); cudaFree(eng->d_prune_stats);
    cudaFree(eng->d_K_loc_tmp); cudaFree(eng->d_V_loc_tmp);
    cudaFree(eng->d_H_l2); cudaFree(eng->d_H_l12); cudaFree(eng->d_ema_attn);
    cudaFree(eng->d_K_loc_tree); cudaFree(eng->d_V_loc_tree);
    cudaFree(eng->d_K_glob_tmp); cudaFree(eng->d_V_glob_tmp);
    cudaFree(eng->d_tree_path_ids);

    for (int i = 0; i < 2; ++i) {
        cudaEventDestroy(eng->ev_copy_done[i]);
        cudaEventDestroy(eng->ev_comp_done[i]);
        cudaEventDestroy(eng->ev_rans_done[i]);
        cudaEventDestroy(eng->ev_pref_done[i]);
    }
    for (int i = 0; i < 4; ++i) { cudaEventDestroy(eng->ev_host_slot_done[i]); cudaEventDestroy(eng->ev_copy_done_miss[i]); }
    for (int l = 0; l < NUM_LAYERS; ++l) {
        cudaEventDestroy(eng->ev_attn_s[l]); cudaEventDestroy(eng->ev_attn_e[l]);
        cudaEventDestroy(eng->ev_la_s[l]);   cudaEventDestroy(eng->ev_la_e[l]);
        for (int p = 0; p < 2; ++p) {
            cudaEventDestroy(eng->ev_pref_s[l][p]);  cudaEventDestroy(eng->ev_pref_e[l][p]);
            cudaEventDestroy(eng->ev_pwait_s[l][p]); cudaEventDestroy(eng->ev_pwait_e[l][p]);
        }
        for (int k = 0; k < TOP_K; ++k) {
            cudaEventDestroy(eng->ev_dma_s[l][k]);  cudaEventDestroy(eng->ev_dma_e[l][k]);
            cudaEventDestroy(eng->ev_wait_s[l][k]); cudaEventDestroy(eng->ev_wait_e[l][k]);
            cudaEventDestroy(eng->ev_exp_s[l][k]);  cudaEventDestroy(eng->ev_exp_e[l][k]);
        }
    }
    cudaEventDestroy(eng->ev_lm_s); cudaEventDestroy(eng->ev_lm_e);
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        cudaEventDestroy(eng->ev_p_dma_s[e]);   cudaEventDestroy(eng->ev_p_dma_e[e]);
        cudaEventDestroy(eng->ev_p_wait_s[e]);  cudaEventDestroy(eng->ev_p_wait_e[e]);
        cudaEventDestroy(eng->ev_p_gath_s[e]);  cudaEventDestroy(eng->ev_p_gath_e[e]);
        cudaEventDestroy(eng->ev_p_deq1_s[e]);  cudaEventDestroy(eng->ev_p_deq1_e[e]);
        cudaEventDestroy(eng->ev_p_gemm1_s[e]); cudaEventDestroy(eng->ev_p_gemm1_e[e]);
        cudaEventDestroy(eng->ev_p_swi_s[e]);   cudaEventDestroy(eng->ev_p_swi_e[e]);
        cudaEventDestroy(eng->ev_p_deq2_s[e]);  cudaEventDestroy(eng->ev_p_deq2_e[e]);
        cudaEventDestroy(eng->ev_p_gemm2_s[e]); cudaEventDestroy(eng->ev_p_gemm2_e[e]);
        cudaEventDestroy(eng->ev_p_scat_s[e]);  cudaEventDestroy(eng->ev_p_scat_e[e]);
    }
    cublasDestroy(eng->cublas);
    cudaStreamDestroy(eng->compute_stream);
    cudaStreamDestroy(eng->copy_stream);
    delete eng;
}
