# Peak-energy convergence validation

The optional input is `neb_set energy_convergence <window_steps> <tolerance_eV>`. A nonzero window contains that many consecutive regular NEB energy evaluations. Zero disables the criterion; positive windows must be at least two. The statistic is the full-window maximum minus minimum of the path's peak energy, including endpoints. Variable-cell paths use the existing enthalpy objective. Either the energy criterion or the existing force criterion may terminate the run.

CPU check (from the repository root, using a shared build directory):

```sh
mkdir -p build/neb_energy
g++ -std=c++14 -Isrc tests/gpumd/gpucfneb/test_energy_window.cpp -o build/neb_energy/test_window
build/neb_energy/test_window
```

This checks oscillation, sliding-window expiry, inclusive tolerance, explicit reset, gaps, non-finite samples and disabled behavior.

GPU check (compute node, with a fresh output directory):

```sh
python3 tests/gpumd/gpucfneb/energy_convergence_smoke.py /absolute/path/to/gpumd /absolute/path/to/fresh_results
```

On 2026-09-16, job `14160177` passed all 15 cases on `m005`, queue `734090ib`, CUDA 12.0.0. The free GPU queues were pending, so this short test used the available paid 4090 queue. Default, disabled and incomplete-window cases ran all 8 requested regular steps. A wide threshold with a 3-sample window stopped after 3 regular evaluations; the existing force criterion stopped after 1 when given a loose force threshold. The strict 4-sample zero-tolerance case did not trigger. The endpoint-peak case correctly reached zero range after the fixed initial endpoint became the highest image. The INA case performed 8 regular plus 8 local steps without falsely converging across path changes. Seven malformed options exited with status 1. Final trajectories and energy tables were checked for successful runs.

The initial test expected a strict 3-sample zero-tolerance window never to stop, but the actual path maximum reached a fixed endpoint. That is stable under the documented whole-path definition, so it became an explicit endpoint-peak test; the non-triggering strict test uses a 4-sample window. No runtime semantics were changed to satisfy the test.

Results and logs are under `build/neb_energy/results_final/report.json`, `build/neb_energy/lsf_14160177.out`, `build/neb_energy/gpu_build.log`, `build/neb_energy/compile.log` and `build/neb_energy/link.log`. This validates stopping behavior, not saddle-point or force convergence. Energy-stop logs explicitly retain the current `fmax`.
