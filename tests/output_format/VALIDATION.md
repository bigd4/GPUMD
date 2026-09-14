# Validation record — 2026-09-14

Target: `/fs09/home/js_jiaqh/bin/GPUMD-NEB`. Format library base: `31ff53565e77efa8604246f87dfde14ba1abc90d`, with the documented local RawFrame extension. The pre-existing modification to `src/neb/neb.cu` was preserved and its SHA-256 checked unchanged.

## Build and CPU results

- CUDA 12.8.1 Make build passed with BXYZ enabled and disabled, including switching the same `gpumd` target off and back on. The final `src/gpumd` has BXYZ enabled. The disabled executable did not link OpenSSL or Zstd.
- CMake adapter builds and CTest passed with `USE_BXYZ=ON` and `OFF`. This validates the shared adapter CMake configuration; the main Torch-dependent CMake project and MDI/HIP executables were not fully built.
- Original native C++ API checks passed.
- All 14 upstream Python/C++ interoperability tests passed.
- All 15 generic Python-reader CPU tests passed, covering arrays, labels, same-name metadata/properties, empty frames, lossless values, legacy v1–v5 structures, corruption and truncation.
- `git diff --check` passed.

## GPU results

Final GPU job: `14148012`, free queue `75434090ib!`, host `m008`, CUDA 12.0.0, RTX 4090 target `sm_89`. All 40 integration cases passed: 21 successful runs and 19 expected error exits. The executable was built on the compute node with GCC and the existing shared Anaconda OpenSSL/Zstd development libraries; those two libraries were linked statically because the node lacks system development headers. The tested executable is `build/output_format/gpu_source/gpumd`.

Covered: all nine selected dump commands; pure velocity/force tables; group ordering and empty selections; custom and split filenames; configuration persistence and reset; restart overwrite and reread; binary model input; default mass/charge and Z-only input; missing/singular cell and invalid mass/velocity/group data; four PIMD bead files; two-potential observer numbering; real QNEP charge, force and 9-component BEC output. Corrupt append inputs exit normally with status 1 and an error message. Total and per-atom virial coexist and match the text reference.

The 216-atom, 4-step multi-output comparison used lossless binary precision:

| Format | Output bytes | Process wall time (s) |
| --- | ---: | ---: |
| Existing text formats | 627193 | 0.872 |
| bxyz | 428059 | 0.252 |
| zbxyz | 319816 | 0.234 |

These are single small runs including startup and statistical companion files, not an MD speed benchmark. Field/metadata comparisons account for rounding already present in text output. They verify serialization and execution, not long-time scientific convergence.

Logs and generated data remain under `build/output_format/`: `gpu_tests_final/report.json`, `lsf_14148012.out`, `gpu_build.log`, `pytest_cpu.log`, `interop.log`, `cmake_on.log`, `cmake_off.log`, and the build-mode switching logs.
