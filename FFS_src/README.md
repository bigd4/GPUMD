# FFS_src

`FFS_src` contains the serial Forward Flux Sampling driver for `gpu-sampling`.
The complete workflow manual is in `FFS_WORKFLOW_MANUAL.md`.

Quick commands:

    /home/tensor/anaconda3/envs/py38/bin/python FFS_src/ffs_driver.py FFS_src/ffs-test/ffs_smoke.yaml --dry-run
    /home/tensor/anaconda3/envs/py38/bin/python FFS_src/ffs_driver.py FFS_src/ffs-test/ffs_smoke.yaml
    /home/tensor/anaconda3/envs/py38/bin/python FFS_src/ffs_driver.py FFS_src/ffs-test/ffs_running_resume_smoke.yaml
    /home/tensor/anaconda3/envs/py38/bin/python FFS_src/ffs_driver.py FFS_src/ffs-test/ffs_recovery_smoke.yaml || true
    /home/tensor/anaconda3/envs/py38/bin/python FFS_src/ffs_driver.py FFS_src/ffs-test/ffs_flux_resume_smoke.yaml

Main files:

- `ffs_driver.py`: single-GPU serial FFS task orchestrator.
- `FFS_WORKFLOW_MANUAL.md`: generic FFS workflow manual.
- `ffs_config.example.yaml`: generic configuration template.
- `ffs-test/ffs_smoke.yaml`: tiny repository-local smoke test.
- `ffs-test/ffs_running_resume_smoke.yaml`: resume companion for the smoke test; useful after manually or programmatically rewinding `driver_state.json` to a running phase.
- `ffs-test/ffs_recovery_smoke.yaml`: expected-failure test that verifies previous-interface recovery events and `max_rounds_per_interface` stop behavior.
- `ffs-test/ffs_flux_resume_smoke.yaml`: tiny smoke config used to verify flux-stage resume reconciliation.
- `ffs-test/ffs_flux_resume_resume_smoke.yaml`: resume companion for the flux reconciliation smoke.
- `ffs-test/ffs_recovery_resume_smoke.yaml`: resume companion for the recovery smoke test; run it after `ffs_recovery_smoke.yaml` has written `driver_state.json`.
- `ffs-test/ffs_1000K_explore_v2.yaml`: GaN B4-B1 fixed-interface example for workflow exploration only.
- `ffs-test/ffs_1000K_adaptive.yaml`: GaN B4-B1 adaptive-refinement example for workflow exploration only.
- `ffs-test/ffs_1500K_fine_probe.yaml`: GaN B4-B1 high-temperature fine-interface probe; useful for testing the 4.4 coordination-number bottleneck, not a production setup.

Driver behavior:

- The external TorchScript model remains CV-only and may output `cv_now`, `cv`, or legacy `commitor`.
- Each task gets a generated `GAScfg.ffs.yaml` and `run.in`.
- Existing `FFSampling`, `PathSampling`, `MetaD`, `GASMD`, and `run` lines in the template are regenerated per segment.
- `ffs.flux_max_steps` controls A-to-lambda_0 runs; `ffs.interface_max_steps` controls lambda_i-to-lambda_{i+1} trials. Missing values fall back to the first `run N` in the template.
- `md.temperature_K` can override velocity and common ensemble temperatures in generated `run.in`.
- `md.resample_velocities` defaults to true. When an interface state contains velocities, the driver strips the velocity column for the launched trial and writes a fresh `velocity T seed N` line when `ffs.random_seed` is set. When `md.resample_velocities` is false under deterministic dynamics, use `replicas_per_crossing: 1` and collect more independent crossing states instead.
- `run.resume` plus `run.state_file` continues from `work_dir/driver_state.json`; use `overwrite_work_dir: false` when resuming. On resume, the driver reconciles `event_cursor`, `events.jsonl`, and saved `states/interface_*` files so a crash after result files were written does not lose valid completed trials.
- `adaptive.enabled` can split zero-success intervals at the midpoint when their width is larger than `adaptive.min_interval_width`.
- `recovery.mode` controls whether a failed layer stops, refills candidates from the previous interface, or refills from initial-state samples.
- `recovery.max_rounds_per_interface` is the hard per-interface refill limit. Once a failed state reaches this limit, resume will not launch more same-pool trials unless recovery settings are changed to permit another controlled refill.

Outputs under `work_dir`:

- `states/interface_*/state_*.xyz`: saved crossing or successful endpoint states.
- `success_segments/interface_*/segment_*`: complete directories for successful trajectory segments.
- `events.jsonl`: per-launch status, CV, steps, velocity seed, retained/deleted marker, and adaptive/recovery/resume events.
- `driver_state.json`: resumable state ledger with phase, current interface, state pools, dynamic interfaces, accumulated stats, recovery rounds, launch counters, and `event_cursor`.
- `interfaces.csv`: conditional probabilities per interface.
- `summary.json`: flux, probability product, rate estimate, barrier estimate, run limits, adaptive settings, and recovery settings.
