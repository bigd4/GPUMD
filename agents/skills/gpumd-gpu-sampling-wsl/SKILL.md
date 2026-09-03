---
name: gpumd-gpu-sampling-wsl
description: Build, verify, and compare GPUMD GPU-Sampling under WSL with portable TorchScript `.pt` and toolchain-bound AOTInductor `.pt2` models. Use for CUDA/LibTorch compatibility, AOTI loader builds, route smoke tests, exact restarts, or GPU performance validation.
---

# GPU-Sampling dual-backend deployment on WSL

Keep builds and disposable runs in the Linux filesystem; use `/mnt/d/...` for source and inputs. Preserve the user's dirty checkout. Read the pyGAS artifact manifest before selecting a toolchain.

## Select a compatibility lane

- TorchScript lane: preserve the existing `.pt` dictionary ABI and behavior. This is the compatibility path for CUDA 11.8, Windows, older LibTorch, and any host that cannot exactly load a `.pt2`.
- AOTI lane: requires LibTorch 2.13.0 with the package-loader header and a CUDA toolkit whose major/minor matches `manifest.toolchain.cuda_version`. PyTorch 2.13 has an official cu129 build, so CUDA 12.9 is a primary lane. A cu132 lane is also valid when generated and loaded consistently.
- Never infer `.pt2` portability from a newer NVIDIA driver. Package and loader must also agree on PyTorch base version, C++ ABI, compute capability, dtype, dimensions and declared ranges. A mismatch is fatal; do not fall back after the user selected `.pt2`.

Run `scripts/preflight_aoti.sh PYTHON LIBTORCH_ROOT CUDA_ROOT` before an AOTI build. For TorchScript-only builds, `ENABLE_GAS_AOTI=ON` safely degrades when the selected LibTorch is not 2.13/AOTI-capable; `.pt2` then reports a clear runtime error while `.pt` remains available.

## Build

Use `scripts/build_dual_backend.sh SOURCE_ROOT BUILD_ROOT LIBTORCH_ROOT CUDA_ROOT [ARCH]`. `ARCH` is the target compute capability without a dot (for example `89`), not a repository-wide constant. The script enables the standalone `gas-aoti-smoke` executable as well as `gpu-sampling`.

The AOTI packager under WSL needs process-local `LIBRARY_PATH=/usr/lib/wsl/lib` so `-lcuda` resolves. The build helper also adds wheel-owned `nvidia/*/lib` directories process-locally (needed by recent LibTorch NVSHMEM dependencies). Do not copy or symlink driver libraries into system toolkit directories.

## Validate

For each applicable route, compare eager, `.pt`, Python `.pt2`, and `gas-aoti-smoke` results from the same structure/history/state. Test edge counts `0,1,2,normal,max,overflow`, history `0,1,max`, non-orthogonal cells and disconnected gradients. Floating output normally uses `rtol=atol=1e-10`; status, shapes, state tensors, deposition and restart steps are exact.

For OPES, run uninterrupted and split checkpoint jobs in both cross-backend directions. `OPES_STATE.pt` is backend-neutral, but the `.pt2` kernel capacity is static. Neighbor overflow, state incompatibility and capacity rollback are hard failures.

Use `scripts/smoke_aoti_routes.sh` for the standalone loader and `scripts/compare_backend_runs.py` for fixed-trajectory or run-directory reports. Run the existing TorchScript MetaD smoke independently so compatibility is not inferred from AOTI success.

## Performance gate

Warm up, run at least five measured repetitions, and record inference-only plus end-to-end median/p95, peak GPU memory and artifact size. A `.pt2` median steady-state regression over 10% requires profiling before merge candidacy. Run at least 10,000 loader calls when checking memory growth or temporary-file leakage. Store commands, commits, environment, hashes, tolerances, errors and timing in `reproduction.json` plus a Markdown report.

## Failure handling

- Missing AOTI header or non-2.13 LibTorch: build/use the TorchScript lane; do not weaken `.pt2` checks.
- CUDA mismatch: generate a separate `.pt2` with the matching PyTorch CUDA build and LibTorch. Do not reuse a cu129 package in cu118/cu132 loaders.
- `-lcuda` packaging failure in WSL: set only the process `LIBRARY_PATH` described above.
- Smoke mismatch or leak: retain logs and reproduction data, mark the backend failed, and do not use it for production.
