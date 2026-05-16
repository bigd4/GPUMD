# GPU-CFNEB Manual
The input of GPU-CFNEB is embedded in the GPUMD standard input file **run.in**. There are two prefix keywords: **neb_set** and **neb_run**. **neb_set** is followed by NEB settings. **neb_run** is followed by optimizer settings. There are two kinds of input parameters. Each parameter is separated by spaces. The first kind is a flag, which toggles a Boolean option. The second kind is key-value input. The key is the parameter name and the following value or values are assigned to this key. Each key has a specific number of values. If the same or conflicting parameters are set, the new value overrides the previous one.

There can be several **neb_set** lines, but there should be only one **neb_run** line. In each line, there can be several flags and key-value inputs. If a key takes a variable number of values, it should be the last key in the line because it takes all values after it.


## NEB Settings
### Input Files
All input structure files should be in extended XYZ format.
- is_name: key 1, initial state input file name. default value: is.xyz
- fs_name: key 1, final state input file name. default value: fs.xyz
- has_mid: flag, if the intermediate states are needed.
- mid_name: key 1, intermediate state input file name. default value: mid.xyz
- mid_name_list: key variable, several intermediate state input file names.
- suffix: key 1, set is_name, fs_name and mid_name to is\_{suffix}.xyz, fs\_{suffix}.xyz and mid\_{suffix}.xyz
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
- dist_range: key 2, the min and max distance when checking the image intervals. If the distance is smaller than min_dist, the image will be removed. If the distance is larger than max_dist, a new image will be inserted between the two images. default value: 0.01 0.1
- dist_ncount: key 1, count how many largest displacements when computing the distance. Detailed formulation can be found in the supplementary material of my paper.

### NEB Method Settings
- k: key 1, elastic coefficient. default value: 0.1
- energy_based_k: flag, adjust spring constants according to the energy extrema.
- energy_k_damping: key 1, damping factor used when energy_based_k updates its effective spring multipliers. default value: 0.1
- tangent: key 1, tangent method. default value: improved
	- improved: improved tangent
	- normal: normal tangent
	- modified: doubly nudged version of improved tangent
- climb: flag, climbing image feature
- find_min: flag, fully using real force to relax the minimum images
- etol: key 1, the energy tolerance that judges if the image is maximum or minimum.

### Image Number Adjustment
- no_ina: flag, do not use the image number adjustment feature.

- ina_interval: key 1, the minimal neb steps interval between two image number adjustment operations. default value: 20
- ina_k: flag, adjust spring constants when inserting or removing images.
- ina_k_efficient: key 1, factor used to increase or decrease spring constants during image number adjustment. default value: 1.8
- ina_cell_factor: key 1, weight factor of cell distance in variable-cell distance checks. default value: natoms^(1/6)
- ina_force_tol: key variable, staged force residual thresholds for image number adjustment. Values are pairs of stage and residual, where stage is the number of NEB steps since the previous image number adjustment and residual is a positive real. Use a very large residual for an unconditional final stage. If other options follow on the same line, end this option with **ina_force_tol_end**. default value: ina_interval 1 3*ina_interval 3 10*ina_interval 1e100
- trim_images: flag, remove repeated local-minimum sections before the usual image insertion/removal checks.
- trim_similar_tol: key 1, position tolerance used by trim_images to judge whether two local minima are the same. default value: 0.001
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
- N_min: key 1. default value: 20
