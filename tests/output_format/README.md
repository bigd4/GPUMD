# Output-format validation

CPU checks use synthetic frames and do not need CUDA:

```sh
python3 -m pytest -q tests/output_format/test_reader.py --basetemp=build/output_format/pytest_cpu
cmake -S tests/output_format -B build/output_format/cmake_on -DUSE_BXYZ=ON
cmake --build build/output_format/cmake_on -j8
ctest --test-dir build/output_format/cmake_on --output-on-failure
cmake -S tests/output_format -B build/output_format/cmake_off -DUSE_BXYZ=OFF
cmake --build build/output_format/cmake_off -j8
ctest --test-dir build/output_format/cmake_off --output-on-failure
```

Select a C++17-capable compiler with `-DCMAKE_CXX_COMPILER=...` where needed. NumPy and pytest are required; compressed Python reads need zstandard. The CMake test project uses the same `src/io/CMakeLists.txt` as GPUMD, without requiring the main project's Torch configuration.

Run the integration script on a GPU compute node with a fresh output directory. It generates a 216-atom carbon fixture and uses the checked-in QNEP training fixture for BEC validation:

```sh
python3 tests/output_format/gpu_smoke.py /absolute/path/to/gpumd /absolute/path/to/fresh_results
```

The script checks all nine affected commands, field values and units, group selections, empty selections, filename rules, append and overwrite behavior, configuration persistence, two-potential observer output, four PIMD beads, QNEP BEC, model/restart input, missing/default model fields, Z-only models, and invalid data. Failure cases must exit with GPUMD's normal error status, not a signal. Per-process wall times and output sizes are recorded in `report.json`; these small tests include process startup and are not throughput benchmarks.

The original upstream native API and 14 Python/C++ interoperability tests can also be run against the local codec. The latter require the upstream Python package from the pinned source revision; set `BXYZ_CPP_CLI`, `BXYZ_CPP_EXAMPLE`, and `PYTHONPATH` to the relevant executables and upstream package. They are separate from the new generic Python reader tests.
