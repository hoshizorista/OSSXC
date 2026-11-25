# OSSXC Verified Benchmarks (GTX 1660 Ti 6 GB, `sm_75`, Windows WDDM + 32 GB RAM)

All measurements were collected in isolated processes with **`openai/gpt-oss-20b` (`20.91B` params, 4-bit native MXFP4 100% untouched)**, compiled with CUDA 12.4 (`66` kernels, `0 bytes` stack frame, `0` register spills).

## End-to-End Throughput & Exactness (4K / 32K / 64K)

| Context | Workload | Mode | Speed (`tok/s`) | Latency (`ms/tok`) | `C_VRAM` | `nvidia-smi` (`GiB`) | RAM RSS (`GB`) | Exact Greedy Match |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| **4K** | C4 Natural Text | `B=1` + Prefetch ON | **`20.603`** | `48.54` | `197` | `5.53` | `14.03` | **True (`max_abs=0.0`)** |
| **4K** | Novel Python (`new_code`) | `B=1` + Prefetch ON | **`15.353`** | `65.13` | `193` | `5.58` | `12.35` | **True (`180/180`)** |
| **32K** | Novel Python (`new_code`) | `B=1` + Prefetch ON | **`12.808`** | `78.07` | `151` | `5.67` | `12.24` | **True (`180/180`)** |
| **32K** | Repo Edit (`repo_verbatim`) | Hybrid (`N-gram + Prefetch`) | **`18.053`** | `55.39` | `151` | `5.62` | `12.24` | **True (`180/180`)** |
| **64K** | C4 Natural Text | `B=1` + Prefetch ON | **`13.635`** | `73.34` | `107` | `5.63` | `14.04` | **True (`max_abs=0.0`)** |
| **64K** | Warmed Code (`Test 4`) | `B=1` + Prefetch ON | **`13.765`** | `72.65` | `106` | `5.48` | `12.24` | **True (`32/32`)** |
| **64K** | Novel Python (`new_code`) | `B=1` + Prefetch ON | **`12.715`** | `78.65` | `106` | `5.40` | `12.26` | **True (`180/180`)** |
| **64K** | Repo Edit (`repo_verbatim`) | Hybrid (`N-gram + Prefetch`) | **`16.424`** | `60.89` | `106` | `5.53` | `12.24` | **True (`180/180`)** |

## Inter-Layer Speculative DMA Lookahead Telemetry (`M3_ema`, 0 MB Extra VRAM)

| Benchmark (`B=1`) | `T_total` OFF → ON (`ms`) | `gpu_wait_dma` OFF → ON (`ms`) | Prefetch Hit Rate | DMA Hidden (`ms/tok`) | Lookahead GPU (`ms/tok`) |
|---|---:|---:|---:|---:|---:|
| **4K `C4`** | `52.80 → 48.54` (`+8.8%`) | `21.87 → 16.31` | `88.7%` (`14.0 / 15.8`) | **`12.18 ms`** | `0.64 ms` |
| **4K `new_code`** | `71.82 → 65.13` (`+10.3%`) | `35.42 → 25.84` | **`96.4%`** (`19.2 / 19.9`) | **`17.32 ms`** | `0.61 ms` |
| **32K `new_code`** | `90.18 → 78.07` (`+15.5%`) | `50.31 → 32.60` | **`97.2%`** (`29.7 / 30.6`) | **`25.09 ms`** | `0.55 ms` |
| **64K `C4`** | `87.28 → 73.34` (`+19.0%`) | `45.03 → 25.93` | `94.2%` (`29.1 / 30.9`) | **`27.03 ms`** | `0.52 ms` |
| **64K `new_code`** | `93.78 → 78.65` (`+19.2%`) | `51.59 → 30.72` | **`96.6%`** (`30.6 / 31.7`) | **`27.88 ms`** | `0.53 ms` |
| **64K `repo_verbatim`** | `97.92 → 82.48` (`+18.7%`) | `52.91 → 31.40` | `92.8%` (`30.6 / 32.9`) | **`28.89 ms`** | `0.56 ms` |

## Multi-Turn KV Continuity & Page-32 Boundary Parity
- **4K Cold vs Warm TTFT:** `15,349.7 ms` (cold) → **`882.0 ms`** (Case A direct extension, **`17.4×` faster**) / **`678.6 ms`** (Case B Ring-128 restore in `0.46 ms`, `KL = 1.88e-7`).
- **Page-32 `V_hot / V_pc32` Boundary Parity (`pos = 4256 = 4096 + 32·5`):** `33/33` exact greedy matches across linear `verify_tokens`, branching `verify_tree_tokens`, partial rollback, and boundary-crossing commits.
