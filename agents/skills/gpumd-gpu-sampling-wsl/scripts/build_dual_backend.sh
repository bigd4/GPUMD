#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: build_dual_backend.sh SOURCE_ROOT BUILD_ROOT LIBTORCH_ROOT CUDA_ROOT [ARCH]" >&2
  exit 2
fi
source_root=$1
build_root=$2
libtorch_root=$3
cuda_root=$4
arch=${5:-}
script_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}
"$script_root/preflight_aoti.sh" "$python_bin" "$libtorch_root" "$cuda_root"
nvidia_library_path=$("$python_bin" -c 'import pathlib,site; roots=[pathlib.Path(p)/"nvidia" for p in site.getsitepackages()]; print(":".join(str(p) for root in roots if root.is_dir() for p in root.glob("*/lib") if p.is_dir()))')
build_library_path=/usr/lib/wsl/lib
if [[ -n "$nvidia_library_path" ]]; then
  build_library_path="$build_library_path:$nvidia_library_path"
fi
if [[ -n "${LIBRARY_PATH:-}" ]]; then
  build_library_path="$build_library_path:$LIBRARY_PATH"
fi
build_runtime_path=$nvidia_library_path
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  build_runtime_path="${build_runtime_path:+$build_runtime_path:}$LD_LIBRARY_PATH"
fi

args=(
  -S "$source_root/src"
  -B "$build_root"
  -DCMAKE_BUILD_TYPE=Release
  -DLIBTORCH_ROOT="$libtorch_root"
  -DENABLE_GAS_AOTI=ON
  -DBUILD_GAS_AOTI_SMOKE=ON
  -DUSE_GAS=ON
)
if [[ -n "$arch" ]]; then
  args+=("-DCMAKE_CUDA_ARCHITECTURES=$arch")
fi
env PATH="$cuda_root/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  LIBRARY_PATH="$build_library_path" LD_LIBRARY_PATH="$build_runtime_path" \
  cmake "${args[@]}"
env LIBRARY_PATH="$build_library_path" LD_LIBRARY_PATH="$build_runtime_path" \
  cmake --build "$build_root" --target gpu-sampling gas-aoti-smoke --parallel
