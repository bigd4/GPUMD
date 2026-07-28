# GPU-CFNEB examples

This directory contains two GPU-CFNEB examples arranged by complexity.

- `diamond_vacancy`: a small 63-atom introductory example. It uses only the initial and final states, a fixed number of interpolated images, and standard FIRE minimization. DYNEB and image-number adjustment are intentionally disabled.
- `large_carbon_transition`: a 672-atom advanced example. It demonstrates an explicit intermediate state, endpoint relaxation, minimum-image alignment, image-number adjustment, and energy-based image spacing.

Run either example from its own directory with a GPU-enabled `gpumd` executable. Generated trajectories, logs, and energy files are not included.
