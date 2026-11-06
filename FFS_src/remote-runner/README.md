# LSF remote FFS runner

`ffs_remote_runner.py` turns direct FFS into stage-wise LSF arrays: one batch
per flux/interface stage, a single collector, then the next batch.  It never
lets workers edit the campaign state file.

Use `config.example.yaml` as the full schema reference.  The normal LSF loop
is:

```bash
python3 ffs_remote_runner.py campaign.yaml init
python3 ffs_remote_runner.py campaign.yaml cycle --submit
# Wait until bjobs reports the batch is finished.
python3 ffs_remote_runner.py campaign.yaml cycle --submit
# Repeat cycle --submit until status reports phase=complete.
python3 ffs_remote_runner.py campaign.yaml status
```

`cycle --submit` first collects finished task directories, advances only when
the entire current batch is terminal, generates the next stage, and submits it
with `bsub`.  Do not invoke it concurrently for the same campaign.

For a no-LSF local end-to-end test:

```bash
python3 ffs_remote_runner.py campaign.yaml init
python3 ffs_remote_runner.py campaign.yaml simulate --seed 20260808
```

The simulator validates the rendered shell after converting only local LSF
metadata to comments, then writes deterministic short-run CV outcomes.  It
does not run scientific MD.

For an integration test on a workstation that has the real executable, first
generate a normal batch, then run the rendered job bodies in that same Linux
environment.  This does not contact LSF and must never be used for production
campaigns:

```bash
python3 ffs_remote_runner.py campaign.yaml init
python3 ffs_remote_runner.py campaign.yaml plan
python3 ffs_remote_runner.py campaign.yaml run-local --random-crossings --seed 20260808
python3 ffs_remote_runner.py campaign.yaml collect
python3 ffs_remote_runner.py campaign.yaml advance
```

`run-local` actually invokes the configured GPU-Sampling or LAMMPS executable
and leaves its original `ffs_runtime.out`, CV log, trajectory, and exit-code
marker in every task directory.  With `--random-crossings`, it then writes a
separate `.ffs_local_decision.json` based on the task ID and seed.  The
collector uses the final real CV row and trajectory frame, but takes only the
success/failure label from that test record.  It is restricted to
`gpu_sampling_monitor` and `lammps_plumed`: native `gpu_sampling_ffs` must
always use its real stopping decision and restart structure.

## Backends

`gpu_sampling_ffs` rewrites the supplied `run.in` to use
`FFSampling model.pt GAScfg.ffs.yaml`; the executable stops at the first target
or failure boundary and writes `restart.xyz` for a successful crossing.

`gpu_sampling_monitor` rewrites `run.in` to use the dev-branch CV-only
`Monitor` command, writes the configured `MonCVlog.txt` path, and adds
`dump_exyz <stride> 1`.  It selects the first CV crossing and its matching
extended-XYZ snapshot.  The monitor logs the initial frame but `dump_exyz` does
not, so crossing row `k>0` maps to dump frame `k-1`.

`lammps_plumed` reads a PLUMED `COLVAR` file and a matching LAMMPS custom dump.
Its input template should use these markers for exact control:

```text
{{FFS_START_COMMAND}}
pair_coeff * * {{FFS_NEP_FILE}}
fix ffs all plumed plumedfile {{FFS_PLUMED_FILE}}
{{FFS_TRAJECTORY_COMMAND}}
run {{FFS_STEPS}}
```

The PLUMED template should contain `{{FFS_CV_LOG}}` and
`{{FFS_MONITOR_STRIDE}}` in its `PRINT` action.  The selected dump snapshot is
written with timestep zero and is loaded by the generated `read_dump` command
after the base data file.  Keep dump and COLVAR strides equal; otherwise set a
validated `lammps.allow_index_matching: true` only when their row/frame order
is known to match.

## Sampling guarantees

When `randomize_velocities: false`, every saved state has `consumed_by` written
at planning time and produces exactly one downstream trial.  Both successful
and failed trials consume that source; only success writes a new state in the
next interface pool.  When velocity randomization is enabled,
`replicas_per_crossing` trajectories are generated with a seed derived from
the immutable task ID, so resubmission cannot reuse a previous seed.

The generated job renderer changes only active `#BSUB -q`, `#BSUB -J`, and the
GPU/CPU resource directive.  Lines starting `! #BSUB` are preserved unchanged
as disabled LSF directives.  A GPU job uses `#BSUB -gpu "num=1:aff=yes"` by
default; CPU mode replaces it with `#BSUB -n 24` unless configured otherwise.
