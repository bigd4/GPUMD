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

## NEB extension — 2026-09-14

Final job `14148124` completed successfully on `m007`, free queue `75434090ib!`, using CUDA 12.0.0 and `make -j8`. All 23 NEB cases passed (16 successful runs and 7 expected status-1 errors), followed by all 40 existing MD integration cases. A `USE_BXYZ=0` Make build also ran text NEB successfully and rejected binary output with status 1. The job then restored the enabled executable at `build/output_format/gpu_source/gpumd`. CUDA 12.8.1 compilation and complete linking also passed locally. CPU adapter CMake/CTest passed with the feature enabled and disabled; the 15 Python reader tests passed.

The NEB checks cover fixed-cell mapping and variable-cell paths; BXYZ and ZBXYZ endpoints; binary intermediate-state input; full-path rereading; all five structure outputs including relaxed endpoints; `auto` tolerance; exact lossless round trips; append and snapshot replacement; truncated or bad-checksum later frames; empty and single-frame paths; a later array-only frame; inconsistent atom counts; and corrupted append targets. Binary output uses physical atom coordinates rather than variable-cell optimizer coordinates. Text/binary comparisons account for existing text rounding. Three NEB iterations use steps 0, 1 and 2; the existing dump logic skips step 0, giving two five-image path dumps and one five-image final snapshot. These are serialization and execution checks, not convergence or throughput benchmarks.

The first two test attempts exposed fixture assumptions: the example endpoints have unequal cells, so fixed-cell tests explicitly request `match_cell_to_initial`; the existing dump logic skips step 0, so three iterations produce two snapshots. Both expectations were corrected without changing NEB timing or fixed-cell rules. The final full run passed.

Logs and reports: `build/neb_output_format/gpu_results_final/report.json`, `build/neb_output_format/md_regression/report.json`, `build/neb_output_format/lsf_14148124.out`, `build/neb_output_format/gpu_build.log`, `build/neb_output_format/gpu_disabled_build.log`, `build/neb_output_format/gpu_restore_build.log`, `build/neb_output_format/link.log` and `build/neb_output_format/pytest.log`. The original local energy-marker change in `src/neb/neb.cu` and workspace configuration were preserved.
