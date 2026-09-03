# GPU-Sampling agent guide

Use the portable project skill at `agents/skills/gpumd-gpu-sampling-wsl` for WSL, CUDA, LibTorch, TorchScript, and AOTInductor work.

Preserve the existing TorchScript `.pt` dictionary ABI and runtime behavior. Treat `.pt2` as an additional Linux/CUDA backend selected only by file suffix. Never weaken or bypass its manifest, PyTorch/LibTorch, CUDA major/minor, C++ ABI, compute-capability, dtype, shape, neighbor/history, or OPES-capacity checks, and never silently fall back after an AOTI error.

Keep build trees and disposable calculations outside the repository. Do not modify `FFS_src/FFS_WORKFLOW_MANUAL.md` or `FFS_src/remote-runner/` unless the user explicitly places them in scope.

Before committing deployment changes, build both an AOTI-capable LibTorch 2.13 stack and a TorchScript-only compatibility stack when available. Run the existing GPU-Sampling tests plus the independent `gas-aoti-smoke` target, route-level `.pt`/`.pt2` comparisons, OPES cross-backend restart tests, and performance/memory gates. Record exact commands, commits, environment, artifact hashes, errors and timings in a reproduction report.
