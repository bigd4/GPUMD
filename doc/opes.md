# GPU-Sampling OPES runtime

This development branch provides an independent `OPES` route for pyGAS
TorchScript models and manifest-backed AOTInductor packages. It does not
change the existing `GASMD` or `MetaD` TorchScript dictionary ABI.

## Run syntax

```text
OPES GASCVModel.pt|GASCVModel.pt2 GAScfg.yaml [OPES_STATE.pt]
```

The optional fourth argument is a complete exact-state checkpoint, not a
kernel text file.  It overrides `opes_state_path`: the runtime loads it and
writes later exact checkpoints back to the same path.  Copy it before the run
if the source checkpoint must remain immutable.  A model or parameter
mismatch is a hard error.

Required common `GAScfg.yaml` fields are `n_atoms`, `cv_size`, `neighbor_rc`,
`max_neighbors`, `cv_log_interval`, and `cv_log_path`.  The OPES runtime also
reads:

```yaml
opes_max_kernels: 4096
opes_state_interval: 50000
opes_state_path: OPES_STATE.pt
opes_kernels_path: OPES_KERNELS
opes_stop_on_status: false
opes_stop_status: 0
```

`opes_stop_status: 0` means stop on any nonzero model status.  A nonzero value
selects that exact terminal status.  Before a status stop, the runtime writes
the OPES checkpoint and the ordinary atomic restart.

For periodic exact restart, set `opes_state_interval` equal to the
`dump_restart` interval.  The OPES checkpoint written during force evaluation
then has the next-step index matching the full-step `restart.xyz` written by
GPU-Sampling at the same interval.  After loading such a pair, GPU-Sampling's
pre-integration force refresh does not advance or log OPES state; the first
integrated step consumes the checkpoint's recorded next-step index.  Atomic
restart positions, masses, cell values, and velocities are written with 17
significant digits so binary64 values survive a text round trip.

## Model contract

The legacy `.pt` model exports `initialize_state(capacity)` and accepts a
`Dict[str, Tensor]` containing positions, cell, directed neighbour pairs,
`opes_step`, compressed-kernel tensors, normalization/adaptive statistics, and
compatibility metadata.  It returns updated state, `bias`, `forces`, flat
nine-component `virial`, `cv_now`, `status`, `committor`, and OPES diagnostics.
The `.pt2` adapter exposes the same values through a fixed ordered-tensor ABI;
its eight state tensors keep the exact `.pt` checkpoint schema. This permits
`.pt` to write `OPES_STATE.pt` for a `.pt2` continuation and vice versa. Use
the pyGAS `pygas-check-gpumd-contract` skill before handing over a model.

The C++ runtime checks atom count, neighbour overflow, kernel capacity,
state ABI version, and checkpoint/model metadata.  All failures are explicit;
the OPES cell list emits unique directed pairs for nonperiodic, mixed-PBC, and
small periodic cells.  Kernel-capacity exhaustion
rolls back the attempted update and preserves the
last paired OPES/atomic checkpoint instead of overwriting it from a half-step
MD state.  Increase `opes_max_kernels` and resume from that matching
`OPES_STATE.pt` plus `restart.xyz`; smaller checkpoint tensors are padded.

## Files

- `OPES_COLVAR` is a headered trajectory with CVs, instantaneous bias,
  `extra_bias`, `rct`, `zed`, `neff`, compressed-kernel count, work,
  flooding status, and deposition diagnostics.
- `OPES_KERNELS` records actual pre-compression depositions for audit.  It is
  not sufficient for an exact restart.
- `OPES_STATE.pt` is the exact binary state and next-step index.

For reweighting, use the instantaneous total bias.  `rct` and `zed` are
diagnostics and must not be used as time-dependent offsets.

## Supported scope

The model implements PLUMED-style single-walker `OPES_METAD` and
`OPES_METAD_EXPLORE`, compressed Gaussian kernels, OPES-internal adaptive
sigma state (not the removed pyGAS `AdaptiveSigma` helper), mixed
periodic CVs, exact restart, and flooding hooks.  MPI multiple walkers, the
experimental PLUMED kernel neighbour list, and `OPES_EXPANDED` are not part of
this runtime version.
