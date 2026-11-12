# GPU-CFNEB Manual
The input of GPU-CFNEB is embedded in the GPUMD standard input file **run.in**. There are two prefix keywords: **neb_set** and **neb_run**. **neb_set** is followed by NEB settings. **neb_run** is followed by optimizer settings. There are two kinds of input parameters. Each parameter is separated by spaces. The first kind is a flag, which toggles a Boolean option. The second kind is key-value input. The key is the parameter name and the following value or values are assigned to this key. Each key has a specific number of values. If the same or conflicting parameters are set, the new value overrides the previous one.

Long commands may continue onto the next physical line using a standalone `\` or `&` at the end of the line, before any `#` comment. Blank/comment-only lines inside a continuation are skipped. The last line has no continuation marker; an unfinished continuation at EOF is an error. This works for both `neb_set` and `neb_run` and does not change XYZ or potential-file parsing.

There can be several **neb_set** lines, but there should be only one **neb_run** line. In each line, there can be several flags and key-value inputs. If a key takes a variable number of values, it should be the last key in the line because it takes all values after it.


## NEB Settings
### Input Files
Input structures may use extended XYZ, BXYZ or ZBXYZ (binary formats require `USE_BXYZ=1`). Endpoint and intermediate-state files use the first frame; `traj_name` reads and validates every frame. All images must have the same atom count and element order. Array-only binary frames cannot initialize NEB.
- is_name: key 1, initial state input file name. default value: is.xyz
- fs_name: key 1, final state input file name. default value: fs.xyz
- mid_name: key 1, intermediate state input file name. Specifying this key enables the intermediate state.
- mid_name_list: key variable, several intermediate state input file names. Specifying this key enables all listed intermediate states.
- traj_name: key 1, whole initial trajectory. If this parameter is set, images are read from the trajectory instead of from is_name, fs_name, and mid_name.
- interpolate: key 1, number of images inserted between input states. If several initial states are set, the inserted images are distributed evenly.
- need_relax: flag, relax the initial and final states before the NEB run.

### System Settings
- no_vc: flag, do not use variable cell feature. That is to say, variable cell feature is open by default.
- p: key 1, hydrostatic pressure, unit GPa. default value: 0
- p3: key 3, 3-components diagonal pressure, unit GPa.
- p6: key 6, 6-components full pressure using Voigt notation, unit GPa.
- remove_translation: flag, remove translation from the input images. default value: true
- remove_rotation: flag, remove rotation from the input images. default value: true
- find_mic: flag, align each input image to the previous image using the minimum image convention before relaxation, translation removal, and interpolation. default value: false
- cell_factor: key 1, weight factor used by the variable-cell filter to scale cell degrees of freedom. If unset, it is estimated from the reference cell volume and number of atoms.
- dist_range: key 2, the min and max distance when checking the image intervals. If the distance is smaller than min_dist, the image will be removed. If the distance is larger than max_dist, a new image will be inserted between the two images. default value: 0.01 0.5
  - When energy_based_spacing is enabled, the interval-specific dist_range is divided by the current energy-based spacing multiplier.
- dist_ncount: key 1, `auto` (default) or a positive integer, controlling how many largest atomic displacements enter the INA distance. `auto` resolves once at path initialization to `min(natoms, max(1, nactive))`, using the existing endpoint active-atom estimate in both fixed-cell and variable-cell runs. Endpoint coordinates are compared in the initial reference cell, with MIC when `find_mic` is enabled; atoms displaced by more than `active_atom_threshold` (default 0.1 Angstrom) are active. This estimate is made before optional endpoint relaxation and translation removal, and is not updated during optimization or INA. An explicit positive integer keeps manual control and is capped at the atom count; zero and negative values are rejected. For example, `neb_set dist_ncount auto` is equivalent to omitting this option, while `neb_set dist_ncount 10` restores the previous default count. Logs report the mode, resolved count and, for auto, the active-atom count and threshold. Existing optional coordination-based adjustments still apply to the resolved base count. Detailed distance formulation can be found in the supplementary material of my paper.

### NEB Method Settings
- k: key 1, elastic coefficient. default value: 0.1
- energy_based_spacing: flag, adjust spring constants and INA distance thresholds according to the relative energy height of each spring.
  - energy_based_k is still accepted as a backward-compatible alias.
- energy_spacing_damping: key 1, damping factor used when energy_based_spacing updates its effective spacing multipliers. default value: 0.1
  - energy_k_damping is still accepted as a backward-compatible alias.
- energy_spacing_coeff: key 2, strength and exponent coefficients used by energy_based_spacing: `1 - strength * (1 - relative_energy^exponent)`, where `relative_energy` is the average endpoint energy of the spring normalized by the current path energy range. The highest-energy springs have multiplier 1, while lower-energy springs are reduced. default value: `0.8 0.9`
  - `strength` should be in `[0, 1)`, and `exponent` should be positive. Larger exponent values make the spacing enhancement more localized near the highest-energy images.
  - energy_k_coeff is still accepted as a backward-compatible alias.
- energy_spacing_dist_power: key 1, exponent used when scaling INA dist_range by the current energy-based spacing multiplier: `dist_range / multiplier^power`. default value: 0.5
- tangent: key 1, tangent method. default value: improved
	- improved: improved tangent
	- normal: normal tangent
	- modified: doubly nudged version of improved tangent
- climb: flag, climbing image feature
- find_min: flag, fully using real force to relax the minimum images
- dyneb: flag, enable dynamic relaxation. Intermediate images whose NEB force is below their local convergence threshold are frozen and reuse their previous potential-energy/force result. They are reactivated if changes in neighboring images move them above the threshold. `dynamic_relaxation` is accepted as an alias. default value: false
- scale_fmax: key 1, nonnegative maximum relative increase of the DYNEB local convergence threshold. The threshold factor is `1 + scale_fmax * (1 - strictness_weight)`, so it is bounded by `1 + scale_fmax`. The strictness weight is the maximum of the normalized-energy weight and Gaussian proximity to any local energy maximum. This option requires `dyneb`. default value: 0
- dyneb_energy_exponent: key 1, positive exponent applied to the normalized image energy when constructing the DYNEB strictness weight. default value: 1
- dyneb_peak_width: key 1, Gaussian width in normalized cumulative path coordinate around every local energy maximum. Images close to a maximum retain a convergence threshold near the original `force_tolerance`. Range: `(0, 1]`. default value: 0.1
- etol: key 1, the energy tolerance that judges if the image is maximum or minimum.

### Image Number Adjustment
- no_ina: flag, do not use the image number adjustment feature.
- If image number adjustment is enabled and only initial/final states are provided, one initial intermediate image is inserted automatically.
- ina_memory_guard: key 1, `1` (default) checks CUDA free memory before each INA insertion, including the automatic first intermediate image; `0` disables this preflight check. Rejected insertions do not change the path, reset FIRE, or start INA local relaxation; image removal remains enabled. If the first intermediate image cannot be added to a two-endpoint path, the run reports an error.
- ina_memory_reserve_mb: key 1, finite non-negative memory reserve in MiB (1024^2 bytes), default 512. For example: `neb_set ina_memory_guard 1 ina_memory_reserve_mb 1024`.
- The conservative budget includes the new image's atom/cell data and the full future NEB/FIRE/DYNEB workspace (12 generalized-coordinate vectors plus per-image scratch), with a 25% margin. Live workspaces are not credited as reclaimable. Every candidate uses the current image count, including earlier insertions in the same pass, to reserve the deferred rebuild cost. Shared force-calculator allocations are already reflected in CUDA free memory and are not multiplied by image count.
- Memory-limited insertions are reported at most once per spacing pass and counted in a final summary; the requested image spacing may remain unsatisfied even when forces converge. This is a conservative preflight estimate, not an OOM recovery mechanism: initial path allocation, scratch allocations before the check, future calculator workspace growth, fragmentation and other GPU processes can still exhaust memory. Native allocation failures retain their existing fatal behavior.
- ina_interval: key 1, the minimal neb steps interval between two image number adjustment operations. default value: 20
- ina_local_relax_steps: key 1, number of additional local NEB steps performed after an INA insertion or removal. During these steps, only changed images and their selected neighbors are optimized; all other images remain fixed. These local steps do not consume the `max_steps` specified by `neb_run` and are reported separately as `INA local step`; the regular `step` counter remains unchanged during local relaxation. `0` disables this stage. default value: 0
- ina_local_relax_neighbors: key 1, number of neighboring images on each side included around every inserted image or changed adjacency during INA local relaxation. This option is used when `ina_local_relax_steps > 0`. default value: 1
- ina_k: flag, adjust spring constants when inserting or removing images.
  - When energy_based_spacing is enabled, ina_k renormalizes the average effective spring constant (`klist * energy_spacing_factor`) to k.
- ina_k_efficient: key 1, factor used to increase or decrease spring constants during image number adjustment. default value: 1.8
- ina_insert_midpoint_weight: key 1, weight in [0, 1] used to place an inserted image between the max_dist position and the midpoint. 0 keeps the max_dist position, 1 uses the midpoint. default value: 0
- ina_force_tol: key variable, staged force residual thresholds for image number adjustment. Values are pairs of stage and residual, where stage is the number of NEB steps since the previous image number adjustment and residual is a positive real. Use a very large residual for an unconditional final stage. If other options follow on the same line, end this option with **ina_force_tol_end**. default value: ina_interval 1 3*ina_interval 3 10*ina_interval 1e100
- trim_images: flag, remove repeated local-minimum sections before the usual image insertion/removal checks.
- trim_similar_tol: key 1, position tolerance used by trim_images to judge whether two local minima are the same. default value: 0.001
- trim_etol: key 1, energy tolerance used only by trim_images when judging local extrema. If unset, trim_images uses etol.
- ina_check_coord: key 1, coordination-number threshold used to check image insertion. default value: 0
- inacc_num: key 1, number of atoms, or fraction of atoms if smaller than 1, used in coordination checks. default value: 0
- inacc_rc: key 1, cutoff radius for coordination checks. default value: 1.7

### Output Settings
- peek_interval: key 1, the interval of peeking neb trajectory. The output trajectory file is named as **peek_traj.xyz**. The output energy profile file is named as **neb_energies.out**. In each peeking operation, the new data overwrites the old data. default value: ceil(max_steps/50)
- dump_interval: key 1, the interval of dumping neb trajectory to **dump_traj.xyz**. default value: ceil(max_steps/10)
- print_interval: key 1, the interval of printing energy and force residual information. default value: 1
- count_force_calc: flag, print the number of force calculations.

## Optimizer Settings
The first parameter is the type of optimizer. Now *fire* is the only option.
The second is force tolerance in unit eV/Å.
The third is max steps.
The following parameters are optional. For the detail meaning of each parameter, you should refer to the original reference (E. Bitzek, P. Koskinen, F. Gähler, M. Moseler, and P. Gumbsch, Structural Relaxation Made Simple, Phys. Rev. Lett. 97, 170201 (2006). https://link.aps.org/doi/10.1103/PhysRevLett.97.170201).
- max_move: key 1, the max movement each step. default value: 0.2
- dt_max: key 1, unit fs. default value: 1
- dt_min: key 1, unit fs. default value: 0.02
- dt_0: key 1, unit fs. default value: 1
- f_inc: key 1. default value: 1.1
- alpha_start: key 1. default value: 0.25
- f_alpha: key 1. default value: 0.99
- alpha_min: key 1, lower bound for the FIRE mixing parameter after multiplication by `f_alpha`. It applies to both global and imagewise FIRE. `0` preserves the original unrestricted decay. Range: `[0, alpha_start]`. default value: 0
- N_min: key 1. default value: 5
- imagewise: flag, use independent FIRE time step, mixing parameter, positive-power counter, velocity norm, and force norm for every movable NEB image. NEB forces remain coupled through the tangent and springs, while optimizer state and FIRE resets are image-local. Images with zero force are frozen until they become active again. This option is intended for NEB calculations using the BaseAtoms block interface. default value: false
- min_alignment_cosine: key 1, minimum allowed cosine between the velocity and force vectors of each image. The image enters the positive-power FIRE branch only when `v dot F - min_alignment_cosine * |v| * |F| > 0`; otherwise only that image is reset. `0` exactly recovers the original power-sign criterion. This option requires `imagewise`. Range: `[0, 1)`. default value: 0

Imagewise FIRE rollback follows the last applied, displacement-limited move. On each restart, `dt_0` is clamped to `[dt_min, dt_max]` and used as the initial timestep. NEB trajectory snapshots retain 17 significant digits so that reloading a converged path does not lose coordinate precision.

### Cyclic Transverse FIRE

`neb_set cyclic_transverse on` enables alternating tangential-spring and physical transverse FIRE optimization. `off` (the default) retains ordinary NEB; `1` and `0` are accepted aliases. The physical transverse force is `F_PES - (F_PES dot tangent) tangent`, with the usual translation removal and generalized cell-force metric. The parallel phase uses only the tangential spring force. Neither phase includes the DNEB perpendicular spring correction, and the current modified tangent is recomputed at every force evaluation.

```text
neb_set tangent modified
neb_set cyclic_transverse on
neb_set energy_spacing_damping 1  # Required when energy_based_spacing is enabled.
neb_run fire 0.001 100000 dt_0 0.1 dt_max 1 dt_min 0.005 max_move 0.1 imagewise alpha_min 0.01
```

The default schedule alternates 5 parallel evaluations and 50 transverse evaluations while INA is changing the path. After 220 evaluations in completed chunks without image-identity changes, transverse chunks continue until the target is met. Every chunk restarts FIRE from zero velocity and the specified `dt_0`; rotation-free generalized coordinates are rebuilt from the current physical configuration. An INA change can restart FIRE inside a chunk, but does not reset the global regular-step budget. Chunk boundaries do not append an unevaluated displacement or image insertion. The reference cell metric is retained within the native run; unlike separate executable restarts, no text round-trip is involved.

Every transverse chunk ends with a fresh PES force evaluation at fixed geometry using freshly computed current tangents. This verification does not advance the regular-step count, INA, or diagnostic histories. A qualifying cycle requires both live and independently verified transverse maxima to be at most the `neb_run` tolerance, a settled topology, and no identity change in that chunk. Then the next parallel/transverse cycle begins. Ten qualifying cycle ends on an unchanged image identity set are required; any INA insertion/deletion invalidates previous evidence, even if the net image count stays unchanged. A low spring-only force never means convergence. Full NEB force is recorded as a diagnostic and is not an acceptance requirement.

Optional `neb_set` counts, each a positive integer no larger than 1000000:

| Parameter | Default | Meaning |
|---|---:|---|
| `cyclic_parallel_steps` | 5 | Maximum parallel evaluations per chunk |
| `cyclic_perpendicular_steps` | 50 | Maximum transverse evaluations per chunk |
| `cyclic_settling_steps` | 220 | Stable-topology evaluations before acceptance |
| `cyclic_required_cycles` | 10 | Qualifying cycle ends required |

The mode requires imagewise FIRE, `tangent modified`, isotropic pressure, and a rotation-free or fixed cell. When energy-based spacing is active, use `energy_spacing_damping 1` to keep fixed-geometry verification independent of damping history. Unsupported combinations are rejected: climb, find_min, DYNEB, INA local relaxation and ina_k. Ordinary INA and `no_ina` are supported. FIRE timestep/mixing and path initialization remain explicit user settings; the switch does not interpolate a fixed number of images or select intermediate input frames.

`neb_cyclic_transverse.out` records each chunk's phase, regular evaluations, image generation, topology changes, stable count, passed cycles, live/verified residuals and force calls. `neb_cyclic_transverse_summary.out` explicitly reports `accepted 1` or `accepted 0`; exhausting `max_steps` is not convergence. `final_traj.xyz` contains evaluated final geometry. Counts include the initial two endpoint evaluations and fresh internal-image verification calls, but exclude optional endpoint relaxation. Computing the full-force diagnostic reuses the same PES forces. Reloading a saved trajectory starts a new schedule and acceptance history. Passing this transverse criterion does not establish convergence of full NEB force, image-density convergence, or a globally minimal path.

## Binary structure output
The global `output_format` keyword applies at `neb_run`, regardless of its position relative to `neb_set`. For example, `output_format zbxyz precision lossless compression_level 3` selects compressed output. The default `xyz` preserves the existing text files. Use one `neb_run` per input file as above.

`interpolate.xyz`, `dump_traj.xyz`, `peek_traj.xyz`, `final_traj.xyz` and `relaxed_is_fs.xyz` change their suffix to `.bxyz` or `.zbxyz`. Dump trajectories append full paths at `dump_interval`; the other files replace the previous snapshot (two endpoints for `relaxed_is_fs`). The binary frames contain species, Cartesian positions, physical lattice/PBC, energy and enthalpy, with `Step`, zero-based `image` and `nimages` metadata. Energies remain in eV and positions in angstrom. Energy and FIRE diagnostic tables remain text. Binary precision uses the global `auto` or `lossless` setting.

To reuse a saved path, set `neb_set traj_name final_traj.zbxyz interpolate 0`. This loads structures, not optimizer state. Do not use an accumulated `dump_traj` as a single path unless you first extract the intended snapshot. All input frames and existing appended output are checksum-validated. See `doc/gpumd/input_parameters/output_format.rst` for full options and Python reading examples.

## GPU 显存峰值记录

每次 `neb_run` 默认独立记录显存采样峰值，无需新增输入选项。在运行入口、图像初始化后、优化器分配及完成后、每次 NEB 力计算入口和弹簧工作区分配后调用 `cudaMemGetInfo`。INA 调整图像后重新分配优化器时同样采样，局部松弛也纳入本次统计。发现新峰值时打印并刷新 `NEB GPU memory high-water mark`；正常结束时打印 `NEB GPU memory summary`，包括峰值 MiB/GiB、设备总容量和采样次数。

这是当前 CUDA 设备的总已用显存采样峰值（total−free），包括CUDA上下文和同卡其他进程，不是本进程分配量。它不是精确的全时段峰值，可能漏掉采样间隙的瞬时分配，以及分配失败前未采样的需求；进程异常终止时没有结束汇总，可查看此前已刷新的峰值行。此记录不改变 INA 显存保护策略，也不增加显式设备同步。

## Optional peak-energy convergence
`neb_set energy_convergence <window_steps> <tolerance_eV>` enables an alternative stopping condition. For example, `neb_set energy_convergence 100 1e-5` stops when the maximum minus minimum peak energy over the last 100 consecutive regular NEB evaluations is at most 1e-5 eV. The default is disabled; `energy_convergence 0 0` disables it explicitly. A nonzero window must be at least 2, and the tolerance must be finite and nonnegative.

The peak is the maximum image energy over the whole path, including endpoints, matching `emax` in the existing log. Variable-cell NEB uses the existing enthalpy objective. The highest image may change within the window. This is a full-window range test, not an adjacent-step difference or just a comparison of the first and last values. A window of 100 counts 100 evaluations (earliest samples are steps 0 through 99). The threshold is total energy in eV, not per-atom energy.

Either the existing force condition or this energy condition may stop optimization. The energy stop prints the step interval, measured range, threshold and current force; it does not imply force convergence. Image insertion/removal or an INA minimizer restart clears the history, and local INA relaxation does not count toward the window. A non-finite energy/force also clears the window. Final trajectory and energy output follow the normal completion path.

## ODE12r with a tangential force window

Example for an already prepared path:

```text
neb_set tangent modified
neb_set energy_based_spacing energy_spacing_damping 1
neb_run ode12r 0.005 3000 tangent_window on window_off 0.015 window_on 0.03 window_hold 100 rtol 0.01 h0 0.02 max_move 0.1 secant_guard 0.1
```

ODE12r is opt-in and uses accepted/rejected force-based trial steps. Defaults are `rtol 0.01`, `h0 0.02`, `hmin 1e-10`, `max_move 0.1`, `secant_guard 0.1`, `residual_extrapolation on`, and `tangent_window off`. Numeric options require positive finite values; `secant_guard` is at most 1 and `hmin` must be below `h0`. Disable the secant extrapolator using `residual_extrapolation off`. The evaluation budget includes initial/final forces, rejected trials and evaluations following INA changes; it is not a count of accepted FIRE steps.

With the window enabled, each image's raw tangential spring residual above `window_on` turns its tangential term on; below `window_off` turns it off; equality or the interval between thresholds retains the previous state. Defaults are `window_off 0.015` and `window_on 0.03`, with `0 < off < on`. This operates on maximum force components in the existing cell-scaled residual metric. Physical perpendicular forces and the DNEB correction remain active on every internal image. Spacing and segment angles do not trigger the window. INA continues to control spacing, including energy-based spacing, and runs only after accepted steps. Image IDs/latches survive insertions and deletions; new images initialize independently and all topology changes reset the convergence streak and ODE history.

In window mode, `force_tolerance` is the physical transverse target. All images must satisfy that target, have raw tangential residual at most `window_on`, and have their tangential latches off for `window_hold` consecutive accepted updates (default 100, configurable positive integer). Initial/final static checks and rejected trials do not increment the streak. The final accepted path is freshly verified before reporting `window_converged`. This is distinct from convergence of the raw full NEB force. With the window disabled, the full driving force determines `full_force_converged`. `step_too_small` and `evaluation_budget` are not convergence.

The current ODE implementation requires modified tangents, fixed cells or rotation-free variable cells with isotropic pressure, and `energy_spacing_damping 1` when enabled. It rejects `climb`, `find_min`, DYNEB, endpoint or INA-local relaxation, and `ina_k`. Combining ODE12r with `cyclic_transverse` or `energy_convergence` is rejected; its own force/window acceptance remains authoritative. `find_mic` alignment remains available. The original projected tangential spring and DNEB correction are retained; no length-difference spring or velocity damping is introduced.

`window_restart FILE` restores a paired final path's versioned `window_state.txt`, retaining image IDs, latches and switch counters while restarting the convergence streak. It requires window mode and matching thresholds, atom and image counts. Use the matching final trajectory, not an unrelated path with the same size. Logs are `neb_ode12r.out`, `neb_ode12r_summary.out`, `neb_window.out`, and `neb_window_topology.out`; event codes are 0 rejected trial, 1 accepted trial, 2 initial/final static check, and 3 INA rebuild.
