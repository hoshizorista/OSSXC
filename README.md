# OSSXC — Native MXFP4 CUDA Engine for `openai/gpt-oss-20b` on 6 GB GPUs (64K Context)

**OSSXC** is a standalone, zero-bloat C++/CUDA inference engine (`66` custom CUDA kernels, **0 register spills** on `sm_75`) built from scratch to run **`openai/gpt-oss-20b` (`20.91B` MoE parameters, 4-bit native MXFP4 100% untouched)** up to **65,536 tokens (64K context)** on a **6 GB VRAM GPU** (such as an NVIDIA GTX 1660 Ti) + 32 GB system RAM — complete with an **OpenCode-ready OpenAI-compatible HTTP/SSE server**, official **Harmony tool calling & reasoning channels**, and **exact multi-turn KV prefix continuity**.

---

## Highlights (Measured on GTX 1660 Ti 6 GB, PCIe Gen3 x16, Windows WDDM)

| Context | Workload | Speed (`tok/s`) | VRAM (`nvidia-smi`) | RAM RSS | Exact Parity |
|---:|---|---:|---:|---:|:---:|
| **4K** | Natural Text (`C4 B=1`, no speculation) | **`20.60 tok/s`** (`48.5 ms/tok`) | `5.53 GiB` | `14.0 GB` | `max_abs = 0.000` |
| **4K** | Brand-New Python Code (`new_code B=1`) | **`15.35 tok/s`** (`65.1 ms/tok`) | `5.58 GiB` | `12.4 GB` | `100%` exact |
| **32K** | Brand-New Python Code (`new_code B=1`) | **`12.81 tok/s`** (`78.1 ms/tok`) | `5.67 GiB` | `12.2 GB` | `100%` exact |
| **32K** | Code Editing / Refactoring (`repo_verbatim`) | **`18.05 tok/s`** (`55.4 ms/tok`) | `5.62 GiB` | `12.2 GB` | `100%` exact |
| **64K** | Natural Text (`C4 B=1`, no speculation) | **`13.64 tok/s`** (`73.3 ms/tok`) | `5.63 GiB` | `14.0 GB` | `max_abs = 0.000` |
| **64K** | Brand-New Python Code (`new_code B=1`) | **`12.72–13.77 tok/s`** | `5.40–5.48 GiB` | `12.3 GB` | `100%` exact |
| **64K** | Code Editing / Refactoring (`repo_verbatim`) | **`16.42 tok/s`** (`60.9 ms/tok`) | `5.53 GiB` | `12.2 GB` | `100%` exact |

- **Zero additional weight quantization:** Official OpenAI 4-bit MXFP4 nibbles and E8M0 scales are streamed untouched.
- **Zero PyTorch CUDA runtime overhead:** PyTorch is used strictly on CPU to unpack weights; 100% of GPU execution runs in pure C++/CUDA via `ctypes`.
- **17.4× faster warm multi-turn TTFT:** Exact 4-case KV continuity + Ring-128 snapshots (`3 MiB` each) reuse 100% of prior context across agentic tool-call turns (`~0.7–2.5 s` warm TTFT even at 32K–64K).

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full telemetry breakdown and parity audit.

---

## Architecture Overview

Running a `20.91B` MoE model at 64K context inside 6 GB VRAM normally hits three hard walls: PCIe Gen3 offloading stalls, cyclic LRU cache thrashing (`< 96` slots for `24 layers × Top-4`), and 64K attention SFU/bandwidth bottlenecks. OSSXC resolves all three in [`engine/ossxc_engine.cu`](engine/ossxc_engine.cu):

1. **Direct Per-Layer Pinned DMA (`~12.3 GB/s`, `0.0 ms` CPU staging):** All 24 MXFP4 expert layers (`10.15 GB`) live in `cudaHostAlloc` pinned RAM and stream directly over PCIe Gen3 without CPU staging buffers.
2. **Inter-Layer Speculative DMA Lookahead Prefetch (`M3_ema`) + 1-ns Zero-Copy Pointer Swap:** Before layer `l` executes Attention, an online zero-training predictor (`router(post_norm(h_{l-1} + ema_attn[l]))`, `276 KB` VRAM, `0.55 ms/tok` GPU cost) predicts uncached cold experts with **`92.8%–97.2%` precision** and prefetches them on `copy_stream` into existing `stream_bufs[0..1]` (**0 MB extra VRAM**), hiding **`12–29 ms/tok`** of PCIe DMA underneath Attention+GPU. On a hit, `std::swap(slot_ptrs[cslot], stream_bufs[p])` promotes the expert into the LRU cache in **1 nanosecond** with zero D2D copies and **`max_abs = 0.000`**.
3. **Breaking the Belady Threshold at 64K (`C_VRAM >= 104` slots > `96`):** Compact scratch workspace (`13.2 MB`) + `embed_tokens` in RAM (`1.16 GB` saved) + Hybrid Global KV unlock **`104–107` resident MXFP4 expert slots at 64K** inside 6 GB VRAM, avoiding cyclic LRU thrashing.
4. **Turing-Optimized 64K Flash-Decoding & Exact Boundary Hybrid KV:** Uses PTX `ex2.approx.ftz.f32`, lossless `14`-nat Lazy-V tile skipping, exact linear `K-FP16` + `V_hot4K` + `V_sink128` + page-32 per-channel `V_pc32` with full FP16 RAM backup, plus a `393 KB` `V_bnd_fp16` shadow buffer guaranteeing bit-exact per-position parity across 32-token eviction boundaries during batched verification.
5. **OpenCode-Hardened Harmony Server:** Native support for `openai/gpt-oss-20b` official Harmony `analysis` / `commentary` / `final` channels, multi-tool-call history normalization, Zod/JSON-Schema sanitization, streaming SSE `tool_calls`, `prompt_tokens_details.cached_tokens`, and live client-disconnect cancellation.

---

## Requirements

- **GPU:** NVIDIA Turing (`sm_75`, e.g., GTX 1660 Ti / RTX 2060+) or newer (`sm_80`, `sm_86`, `sm_89`, `sm_90`) with **>= 6 GB VRAM**
- **RAM:** **>= 24–32 GB** system RAM (10.15 GB pinned MXFP4 experts + 1.16 GB embeddings + 0.8 GB 64K V mirror)
- **Disk:** ~28 GB free during initial setup (~14 GB HF checkpoint + ~13.8 GB packed OSSXC weights; you may delete the HF `.safetensors` shards after packing while keeping tokenizer/template JSON/Jinja files!)
- **Toolchain:**
  - **Windows:** Visual Studio 2022/2019 Build Tools (C++ Desktop Development) + NVIDIA CUDA Toolkit 12.x + Python 3.10+
  - **Linux:** `g++` + NVIDIA CUDA Toolkit 12.x + Python 3.10+

---

## Quick Start (Recreating from Scratch)

### 1. Clone & Install Python Dependencies
```bash
git clone https://github.com/hoshizorista/OSSXC.git
cd OSSXC
pip install -r requirements.txt
```
*(CPU-only PyTorch is completely fine — OSSXC never initializes PyTorch CUDA!)*

### 2. Compile the Native CUDA Engine

**Windows (PowerShell):**
```powershell
# Default sm_75 (GTX 1660 Ti / RTX 20xx). For RTX 30xx use -Arch sm_86; RTX 40xx use -Arch sm_89.
powershell -ExecutionPolicy Bypass -File engine\build_windows.ps1 -Arch sm_75
```

**Linux:**
```bash
# Default sm_75. Pass sm_86 / sm_89 / etc. as argument for newer GPUs.
bash engine/build_linux.sh sm_75
```
Verify `engine/ptxas_engine_report.txt` shows `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`.

### 3. Download & Pack `openai/gpt-oss-20b` Native MXFP4 Weights
```bash
# Downloads openai/gpt-oss-20b from Hugging Face into ./hf_model and packs ./compressed
python scripts/prepare_weights.py --download
```
If you already have `openai/gpt-oss-20b` downloaded locally:
```bash
python scripts/prepare_weights.py --hf-dir /path/to/gpt-oss-20b --out-dir ./compressed
```

### 4. Start the OSSXC Local Server (64K Context by Default)
```bash
python -m ossxc.server --ctx 65536 --host 127.0.0.1 --port 8000
```
- For **4K** pure FP16 (max `C_VRAM = 197` slots, fastest startup): `python -m ossxc.server --ctx 4096`
- For **8 GB / 12 GB+ GPUs** (unlock even more cached expert slots!): `python -m ossxc.server --ctx 65536 --ceiling-gb 7.2`
- Server-side default sampling flags: `--temp 0.8 --top-p 0.8 --top-k 40` (add `--override-sampling` to enforce even when clients pass their own values)
- Environment overrides: `OSSXC_MODEL_DIR`, `OSSXC_WEIGHTS_DIR`, `OSSXC_ENGINE_LIB`

### 5. Interactive CLI Chat (Optional)
```bash
# Connect to the running server
python -m ossxc.cli --url http://127.0.0.1:8000 --temp 0.6 --show-thinking

# Or run directly without starting the HTTP server
python -m ossxc.cli --ctx 65536 --temp 0.6
```
In-chat slash commands: `/temp <val>`, `/top_p <val>`, `/top_k <n>`, `/max <n>`, `/reasoning <low|medium|high>`, `/undo` (branches back via Ring-128 snapshot!), `/reset`, `/stats`, `/quit`.

---

## Using OSSXC with OpenCode

OSSXC's server is hardened for **[OpenCode](https://opencode.ai)** (`@ai-sdk/openai-compatible`), including multi-tool-call Harmony normalization, Zod schema sanitization, reasoning channels, and KV prefix cache reporting.

1. Start the OSSXC server:
   ```bash
   python -m ossxc.server --ctx 65536 --port 8000
   ```
2. Copy [`opencode.json`](opencode.json) into your project directory (or `~/.config/opencode/opencode.json`):
   ```json
   {
     "$schema": "https://opencode.ai/config.json",
     "provider": {
       "ossxc": {
         "npm": "@ai-sdk/openai-compatible",
         "name": "OSSXC Local (GPT-OSS-20B MXFP4)",
         "options": {
           "baseURL": "http://127.0.0.1:8000/v1",
           "apiKey": "ossxc-local"
         },
         "models": {
           "openai/gpt-oss-20b": {
             "name": "GPT-OSS-20B (OSSXC Native MXFP4 • 64K)",
             "reasoning": true,
             "tool_call": true,
             "limit": {
               "context": 65536,
               "output": 8192
             }
           }
         }
       }
     },
     "model": "ossxc/openai/gpt-oss-20b"
   }
   ```
3. Run `opencode` in your workspace — tool calls (`read`, `grep`, `glob`, `edit`, `bash`), Harmony reasoning, and warm multi-turn KV continuity will work automatically!

> **Tip for OpenCode latency:** Leave `temperature` at `0.0` for maximum code-editing speed (activates lossless O(1) N-gram speculation alongside lookahead prefetch) or `0.4–0.7` for creative coding (lookahead prefetch stays 100% active at `12.7–20.6 tok/s`).

---

## Repository Layout

```text
OSSXC/
├── LICENSE                     # MIT License
├── README.md                   # This guide
├── requirements.txt            # Minimal Python dependencies
├── opencode.json               # Ready-to-use OpenCode provider configuration
├── docs/
│   └── BENCHMARKS.md           # Verified 4K / 32K / 64K benchmarks & parity audit
├── engine/
│   ├── ossxc_engine.cu         # Portable C++/CUDA inference engine (66 kernels, 0 spills)
│   ├── mxfp4_rans_format.h     # Shared format header
│   ├── build_windows.ps1       # Auto-detecting Windows MSVC + NVCC build script
│   └── build_linux.sh          # Linux g++ + NVCC build script
├── scripts/
│   └── prepare_weights.py      # HF downloader + untouched native MXFP4 weight packer
└── ossxc/
    ├── config.py               # Portable path & architecture configuration
    ├── rope.py                 # Exact 64K YaRN RoPE cos/sin generator
    ├── engine_wrapper.py       # Cross-platform ctypes engine wrapper (.dll / .so)
    ├── harmony.py              # OpenCode-hardened Harmony renderer & stream parser
    ├── runtime.py              # Persistent runtime + 4-case KV continuity + Ring-128 snapshots
    ├── server.py               # OpenAI / OpenCode-compatible HTTP & SSE server
    └── cli.py                  # Interactive terminal chat client
```

---

## Credits & License

Created by **Juan** & **Aria**.  
**Tools:** Antigravity, CC.

Released under the **MIT License** — see [`LICENSE`](LICENSE) for details.  
Target model weights (`openai/gpt-oss-20b`) are subject to OpenAI's respective model license on Hugging Face.
