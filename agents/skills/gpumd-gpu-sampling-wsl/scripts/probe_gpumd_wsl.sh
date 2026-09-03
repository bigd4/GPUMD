#!/usr/bin/env bash
set -euo pipefail

source_root="${1:-/mnt/d/data/software/GPUMD}"
if ! command -v nvcc >/dev/null 2>&1 && [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi
printf 'source_root=%s\n' "$source_root"
printf 'kernel=%s\n' "$(uname -r)"

for command in nvidia-smi nvcc cmake ninja; do
  if command -v "$command" >/dev/null 2>&1; then
    printf '%s=%s\n' "$command" "$(command -v "$command")"
  else
    printf '%s=MISSING\n' "$command"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
fi
if command -v cmake >/dev/null 2>&1; then
  cmake --version | head -n 1
fi

if [[ ! -f "$source_root/src/CMakeLists.txt" ]]; then
  printf 'SOURCE_CHECK=FAIL: missing src/CMakeLists.txt\n'
  exit 2
fi
printf 'SOURCE_CHECK=OK\n'

if [[ -f "$source_root/libtorch/lib/libtorch.so" && -f "$source_root/libtorch/share/cmake/Torch/TorchConfig.cmake" ]]; then
  printf 'CHECKOUT_LIBTORCH=LINUX\n'
elif [[ -f "$source_root/libtorch/lib/torch_cpu.dll" ]]; then
  printf 'CHECKOUT_LIBTORCH=WINDOWS_ONLY\n'
else
  printf 'CHECKOUT_LIBTORCH=MISSING_OR_UNKNOWN\n'
fi

for root in /home/tensor/workspace/libtorch* /opt/libtorch*; do
  if [[ -f "$root/lib/libtorch.so" && -f "$root/share/cmake/Torch/TorchConfig.cmake" ]]; then
    printf 'LINUX_LIBTORCH=%s\n' "$root"
  fi
done

if [[ -n "${GPU_SAMPLING_EXECUTABLE:-}" && -x "${GPU_SAMPLING_EXECUTABLE}" ]]; then
  printf 'GPU_SAMPLING_EXECUTABLE=%s\n' "$GPU_SAMPLING_EXECUTABLE"
fi
