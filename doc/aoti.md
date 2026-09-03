# GPU-Sampling TorchScript and AOTInductor deployment

GPU-Sampling selects the pyGAS model runner from the file suffix while keeping
the positional command syntax unchanged:

```text
GASMD GASCVModel.pt|GASCVModel.pt2 GAScfg.yaml [GASGaussian.txt]
PathSampling GASCVModel.pt|GASCVModel.pt2 GAScfg.yaml
FFSampling GASCVModel.pt|GASCVModel.pt2 GAScfg.yaml
OPES GASCVModel.pt|GASCVModel.pt2 GAScfg.yaml [OPES_STATE.pt]
```

## Compatibility policy

- `.pt` remains the compatibility artifact. Its existing
  `forward(Dict[str, Tensor])` ABI and loading behavior are unchanged. Use it
  for CUDA 11.8, Windows, CPU workflows, or any environment without the exact
  AOTInductor toolchain.
- `.pt2` is an additional Linux CUDA deployment binary. It never silently
  falls back to `.pt`. The sibling `GASCVModel.manifest.json` is mandatory.
- An AOTI package must match the runtime's PyTorch/LibTorch base version, CUDA
  major/minor, C++ ABI, compute capability, dtype, atom/CV sizes, dynamic
  neighbor/history ranges, and OPES capacity. Build one `.pt2` package per
  target toolchain/GPU architecture.
- The supported prebuilt primary AOTI stack is PyTorch/LibTorch 2.13.0 with
  CUDA 12.9. CUDA 11.8 continues to use the unchanged `.pt` route unless a
  separately maintained matching PyTorch/LibTorch 2.13.0 CUDA 11.8 stack is
  supplied.

## Configure and build

From WSL, point `LIBTORCH_ROOT` to the LibTorch files installed by the matching
Python wheel. The CUDA toolkit used by CMake must match that wheel:

```bash
cmake -S src -B build-py213 \
  -DLIBTORCH_ROOT=/path/to/torch \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DENABLE_GAS_AOTI=ON \
  -DBUILD_GAS_AOTI_SMOKE=ON
cmake --build build-py213 -j
```

For a TorchScript-only build, use the LibTorch/CUDA pair already used by that
environment and set `-DENABLE_GAS_AOTI=OFF`. A compatible LibTorch version
without the 2.13 package-loader API automatically reports TorchScript-only
mode rather than breaking `.pt` support.

Under WSL, AOTI packaging may require the process-local setting
`LIBRARY_PATH=/usr/lib/wsl/lib` so the generated linker can find `libcuda`.
The build helper also adds the selected Python wheel's `nvidia/*/lib`
directories to the build process for dependencies such as NVSHMEM. Do not copy
or replace driver libraries in system CUDA directories.

## Validation

Set `PYTHON` to the matching wheel environment when using the build helper.
Run `agents/skills/gpumd-gpu-sampling-wsl/scripts/preflight_aoti.sh`, then use
`gas-aoti-smoke` on every generated route. A release candidate must compare
eager, `.pt`, Python `.pt2`, and C++ `.pt2` outputs; exercise neighbor counts
at 0, 1, normal, capacity, and overflow; and test exact OPES checkpoint
continuation in both backend directions. Record commands, commits, artifact
hashes, tolerances, maximum errors, timing, and memory in `reproduction.json`
and its Markdown report.
