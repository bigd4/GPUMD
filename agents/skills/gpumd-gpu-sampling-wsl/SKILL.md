---
name: gpumd-gpu-sampling-wsl
description: Build, verify, and compare GPUMD GPU-Sampling under WSL with portable TorchScript `.pt` and toolchain-bound AOTInductor `.pt2` models. Use for CUDA/LibTorch compatibility, AOTI loader builds, route smoke tests, exact restarts, or GPU performance validation.
---

# GPU-Sampling dual-backend deployment on WSL

This project skill in `agents/skills/` is the portable, Git-published entry point. Machine-local `.codex/skills/`, `.agents/`, and user-level copies are development overlays and are not publication inputs. When a same-named user-level skill is also available, follow this project version.

Capabilities are source-version-specific: inspect the requested branch's parser, loader, CMake options, and artifact contract before claiming a route exists. Record branch, commit, relevant paths, and dirty-source provenance. A capability implemented on dev can be built and tested locally within the authorized task; do not require publication first. Distinguish implementation, historical validation, and current measured evidence.

Keep build trees and disposable runs outside the source repository, in the Linux filesystem; use `/mnt/d/...` for source and inputs. Preserve the user's dirty checkout. Compile the source actually under test, including relevant authorized uncommitted changes in an isolated source snapshot when needed; do not pair an old committed binary with new source/test edits and claim validation.

## Select the task and compatibility lane

Proceed with already authorized routine preparation, local builds, and relevant tests without repeated confirmation. Reuse supplied scientific and environment choices, record their provenance, and ask only for unresolved choices that affect the requested result. Preserve explicit draft-only, remote-only, approval-before-execution, and resource limits.

- Source-only or draft work: inspect the selected source and prepare commands/artifacts; do not require an AOTI manifest before one exists or claim runtime validation.
- Ordinary single-backend use: preserve the requested `.pt` or `.pt2`, validate that artifact and route in the target environment, and report that tested scope. An optional unrequested second backend is not a prerequisite for normal use.
- Development/deployment work: preserve every relevant GPU, ABI, route, restart, performance, and memory gate below. Missing required evidence limits deployment readiness rather than excusing the gate.

- TorchScript lane: preserve the existing `.pt` dictionary ABI and behavior. It remains the compatibility path for supported older LibTorch/CUDA stacks and hosts that cannot load `.pt2`; verify the actual target environment.
- AOTI lane: the current loader/helper contract requires LibTorch/PyTorch 2.13.0 and the package-loader header. Before deploying an existing `.pt2`, read its manifest; select a toolkit whose major/minor matches `manifest.toolchain.cuda_version`. cu129 and cu132 are separate package/loader lanes, usable only when the actual package and loader agree. Verify any version change against the requested source and manifest rather than assuming a newer stack is compatible.
- Never infer `.pt2` portability from a newer NVIDIA driver. Package and loader must agree on PyTorch base version, C++ ABI, CUDA major/minor, compute capability, dtype, dimensions, declared ranges, and neighbor/history/OPES capacity. A mismatch is a hard failure for that combination. Do not fall back after the user selected `.pt2`.

## Build

For the AOTI lane, use `scripts/preflight_aoti.sh PYTHON LIBTORCH_ROOT CUDA_ROOT` and `scripts/build_dual_backend.sh SOURCE_ROOT BUILD_ROOT LIBTORCH_ROOT CUDA_ROOT [ARCH]`. The build helper also invokes preflight and enables `gas-aoti-smoke`; it is AOTI-only and cannot serve an older TorchScript compatibility stack. `ARCH` is the verified target compute capability without a dot, not a repository-wide constant.

For a `.pt`-only compatibility build, configure CMake directly with AOTI and its smoke target disabled, then build only `gpu-sampling`. Set these task variables to verified paths and use an external build directory:

```bash
cmake -S "$SAMPLING_SOURCE_ROOT/src" -B "$SAMPLING_BUILD_ROOT" \
  -DLIBTORCH_ROOT="$SAMPLING_LIBTORCH_ROOT" \
  -DCMAKE_CUDA_COMPILER="$SAMPLING_CUDA_ROOT/bin/nvcc" \
  -DCMAKE_BUILD_TYPE=Release -DUSE_GAS=ON \
  -DENABLE_GAS_AOTI=OFF -DBUILD_GAS_AOTI_SMOKE=OFF
cmake --build "$SAMPLING_BUILD_ROOT" --target gpu-sampling --parallel
```

CMake's `ENABLE_GAS_AOTI=ON` compatibility degradation does not bypass the AOTI-only helper's preflight or make `BUILD_GAS_AOTI_SMOKE=ON` valid without AOTI. Absence of AOTI is not a reason to upgrade or block a requested `.pt`-only environment.

The AOTI packager under WSL needs process-local `LIBRARY_PATH=/usr/lib/wsl/lib` so `-lcuda` resolves. The AOTI build helper also adds wheel-owned `nvidia/*/lib` directories process-locally for dependencies such as NVSHMEM. For either build lane, inspect missing dependencies and reuse compatible verified process-local paths; do not copy or symlink driver libraries into system toolkit directories. System package changes retain their explicit user-authorization requirement.

## Validate

For ordinary single-backend acceptance, load the saved selected artifact, verify its actual target structure, dictionary/tensor/neighbor contract, and run the applicable route/runtime checks. Stateful use also needs the relevant same-backend continuation/restart check. State exactly the backend, artifact, route, and environment accepted; do not infer untested dual-backend or production-physics equivalence.

For development/deployment changes affecting these contracts, retain the full relevant matrix: compare eager, `.pt`, Python `.pt2`, and `gas-aoti-smoke` from the same structure/history/state for each affected route. Test edge counts `0,1,2,normal,max,overflow`, history `0,1,max`, non-orthogonal cells and disconnected gradients. Floating output normally uses `rtol=atol=1e-10`; status, shapes, state tensors, deposition and restart steps are exact. Expected overflow must produce the specified error, not unchecked execution.

For OPES/state/restart changes, run uninterrupted and split checkpoint jobs in both cross-backend directions. `OPES_STATE.pt` is backend-neutral, but `.pt2` kernel capacity is static. Neighbor overflow, state incompatibility and capacity rollback are hard failures of the attempted runtime combination.

Use `scripts/smoke_aoti_routes.sh` for the standalone loader and `scripts/compare_backend_runs.py` for fixed-trajectory or run-directory reports. Run the existing relevant GPU-Sampling tests and TorchScript MetaD smoke independently so compatibility is not inferred from AOTI success. These are necessary local development checks when their behavior is affected, including dev-only functionality.

## Performance and memory gates

For changes affecting inference/runtime performance, warm up, run at least five measured repetitions, and record inference-only plus end-to-end median/p95, peak GPU memory and artifact size. A `.pt2` median steady-state regression over 10% requires profiling before merge candidacy. Run at least 10,000 loader calls when checking memory growth or temporary-file leakage. Preserve these gates for the affected development/deployment change; ordinary draft or single-artifact loading does not by itself request a benchmark campaign.

Store commands, source provenance, environment, hashes, tolerances, errors, and timings in `reproduction.json` plus a Markdown report. Report each applicable check as `passed`, `failed`, `blocked`, or `not_run`; historical reports do not supply new passing results. Missing required environments/tests prevent claims dependent on them, while unaffected preparation and checks continue.

## Failure handling

- Missing AOTI header, incompatible LibTorch, or CUDA mismatch: for a requested `.pt`, continue with its compatible TorchScript lane. For a requested `.pt2`, retain that target, preserve logs, and mark only the affected AOTI build/load/check blocked while prerequisites are missing. Continue independent preparation and diagnosis; do not weaken checks, substitute `.pt`, or report the requested `.pt2` validation complete.
- Repair or regenerate a `.pt2` within the already authorized target-toolchain scope when feasible. Do not reuse cu129 packages in cu118/cu132 loaders. Ask only if a required backend/toolchain change is outside existing authorization; do not invent a new approval step for routine compatible repairs already in scope.
- `-lcuda` packaging failure in WSL: set only the process-local `LIBRARY_PATH` described above.
- Smoke mismatch or leak: retain reproduction data, mark the affected backend failed, and do not use the failed artifact/combination for production. Continue any independent deliverable that remains feasible.
