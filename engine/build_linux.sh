#!/usr/bin/env bash
# OSSXC Native C++/CUDA Engine Build Script (Linux g++ + NVCC)
# Usage:
#   bash engine/build_linux.sh           # defaults to sm_75 (Turing / GTX 1660 Ti)
#   bash engine/build_linux.sh sm_86     # Ampere (RTX 30xx)
#   bash engine/build_linux.sh sm_89     # Ada (RTX 40xx)
set -euo pipefail

ARCH="${1:-sm_75}"
COMPUTE_ARCH="${ARCH/sm_/compute_}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CU_SRC="${SCRIPT_DIR}/ossxc_engine.cu"
OUT_SO="${SCRIPT_DIR}/libossxc_engine.so"
PTXAS_LOG="${SCRIPT_DIR}/ptxas_engine_report.txt"

NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
if [[ ! -x "${NVCC}" ]]; then
    echo "ERROR: nvcc not found. Install NVIDIA CUDA Toolkit 12.x or set NVCC=/path/to/nvcc." >&2
    exit 1
fi

echo "Using nvcc:  ${NVCC}"
echo "Target arch: ${ARCH} (${COMPUTE_ARCH})"

"${NVCC}" -std=c++17 -O3 \
    -gencode "arch=${COMPUTE_ARCH},code=${ARCH}" \
    --ptxas-options=-v \
    --shared \
    -Xcompiler "-fPIC -O3 -mavx2 -pthread" \
    -lcublas -lcudart \
    -o "${OUT_SO}" "${CU_SRC}" 2>&1 | tee "${PTXAS_LOG}"

echo "Successfully built: ${OUT_SO}"
echo "Saved ptxas resource report: ${PTXAS_LOG}"
