#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: preflight_aoti.sh PYTHON LIBTORCH_ROOT CUDA_ROOT" >&2
  exit 2
fi
python_bin=$1
libtorch_root=$2
cuda_root=$3

[[ -x "$python_bin" ]] || { echo "Python is not executable: $python_bin" >&2; exit 1; }
[[ -f "$libtorch_root/lib/libtorch.so" ]] || { echo "Missing libtorch.so" >&2; exit 1; }
[[ -f "$libtorch_root/share/cmake/Torch/TorchConfig.cmake" ]] || { echo "Missing TorchConfig.cmake" >&2; exit 1; }
[[ -f "$libtorch_root/include/torch/csrc/inductor/aoti_package/model_package_loader.h" ]] || { echo "Missing AOTI package loader header" >&2; exit 1; }
[[ -x "$cuda_root/bin/nvcc" ]] || { echo "Missing nvcc under $cuda_root" >&2; exit 1; }
[[ -d /usr/lib/wsl/lib ]] || { echo "Missing WSL driver library directory" >&2; exit 1; }

python_info=$(
  "$python_bin" -c 'import json,torch; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"cxx11_abi":bool(torch.compiled_with_cxx11_abi()),"available":torch.cuda.is_available(),"capability":list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None}))'
)
nvcc_version=$("$cuda_root/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)
python_cuda=$("$python_bin" -c 'import torch; print(torch.version.cuda or "")')
python_base=$("$python_bin" -c 'import torch; print(torch.__version__.split("+")[0])')
python_abi=$("$python_bin" -c 'import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)')
version_header="$libtorch_root/include/torch/headeronly/version.h"
[[ -f "$version_header" ]] || { echo "Missing LibTorch version header" >&2; exit 1; }
libtorch_major=$(awk '$2 == "TORCH_VERSION_MAJOR" {print $3}' "$version_header")
libtorch_minor=$(awk '$2 == "TORCH_VERSION_MINOR" {print $3}' "$version_header")
libtorch_patch=$(awk '$2 == "TORCH_VERSION_PATCH" {print $3}' "$version_header")
libtorch_base="$libtorch_major.$libtorch_minor.$libtorch_patch"
libtorch_abi=$(sed -n 's/.*_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' "$libtorch_root/share/cmake/Torch/TorchConfig.cmake" | head -n 1)
libtorch_abi=${libtorch_abi:-1}
[[ "$python_base" == "2.13.0" ]] || { echo "AOTI requires PyTorch 2.13.0, found $python_base" >&2; exit 1; }
[[ "$libtorch_base" == "$python_base" ]] || { echo "PyTorch $python_base != LibTorch $libtorch_base" >&2; exit 1; }
[[ "$libtorch_abi" == "$python_abi" ]] || { echo "PyTorch C++ ABI $python_abi != LibTorch ABI $libtorch_abi" >&2; exit 1; }
[[ "$python_cuda" == "$nvcc_version" ]] || { echo "PyTorch CUDA $python_cuda != nvcc $nvcc_version" >&2; exit 1; }
probe_dir=$(mktemp -d)
trap 'rm -rf -- "$probe_dir"' EXIT
printf '#include <cuda.h>\nint main(){CUdevice device; return cuInit(0) || cuDeviceGet(&device, 0);}\n' |
  env LIBRARY_PATH=/usr/lib/wsl/lib "$cuda_root/bin/nvcc" -x cu - -lcuda -o "$probe_dir/cuda-driver-probe"
"$probe_dir/cuda-driver-probe"
printf '%s\n' "$python_info"
printf 'nvcc_cuda=%s\nlibtorch=%s\nlibtorch_abi=%s\nlibtorch_root=%s\n' \
  "$nvcc_version" "$libtorch_base" "$libtorch_abi" "$libtorch_root"
