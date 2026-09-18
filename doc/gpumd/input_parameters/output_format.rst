.. _kw_output_format:

output_format
=============

Select the output encoding for MD structures, per-atom arrays and NEB structures. This setting does not enable any dump command or change its interval or atom selection.

Syntax
------

.. code:: text

   output_format xyz
   output_format bxyz precision lossless
   output_format zbxyz precision auto compression_level 3

The default is ``xyz``: every command keeps its existing text format, including ``velocity.out`` and ``force.out``. Binary formats require a build with ``USE_BXYZ=1``. ``bxyz`` writes uncompressed v4 frames; ``zbxyz`` writes independently compressed v5 frames.

Optional key/value pairs may appear in either order. ``precision auto`` (the default for binary output) selects float32 or float64 per property. Cartesian ``pos`` and ``unwrapped_position`` use absolute tolerance 1e-6 angstrom and relative tolerance zero; ``forces`` uses absolute tolerance 1e-5 eV/angstrom and ``vel`` uses absolute tolerance 1e-6 angstrom/fs, both with relative tolerance zero. Other floating arrays retain relative tolerance 1e-7 and absolute tolerance zero. These absolute bounds apply to each component, not the vector norm. If any value exceeds the tolerance after float32 conversion, the whole property stays float64. Lattice and other frame metadata retain their existing text precision. ``precision lossless`` uses zero tolerance and preserves the original floating values; exactly representable values may still use float32. Integer and string values are lossless. ``compression_level`` is allowed only for ``zbxyz`` and accepts 1 through 22, default 3. Text mode accepts no binary options. Invalid, repeated or incomplete options are errors.

The last setting before each ``run`` applies to all affected dumps in that run, regardless of their declaration order. It persists across runs. Every new ``output_format`` command starts from the selected format's defaults. Existing per-command text precision options still control text output; binary precision is controlled exclusively by ``output_format``.

Affected outputs
----------------

.. list-table:: Default filenames
   :header-rows: 1

   * - Command
     - Text output
     - Compressed binary output
   * - ``dump_exyz``
     - ``dump.xyz`` / ``dump.<step>.xyz``
     - ``dump.zbxyz`` / ``dump.<step>.zbxyz``
   * - ``dump_position``
     - ``movie.xyz``
     - ``movie.zbxyz``
   * - ``dump_velocity``
     - ``velocity.out``
     - ``velocity.zbxyz``
   * - ``dump_force``
     - ``force.out``
     - ``force.zbxyz``
   * - ``dump_restart``
     - ``restart.xyz``
     - ``restart.zbxyz``
   * - ``dump_beads``
     - ``beads_dump_<bead>.xyz``
     - ``beads_dump_<bead>.zbxyz``
   * - ``dump_observer``
     - ``observer[<index>].xyz``
     - ``observer[<index>].zbxyz``
   * - ``dump_cg``
     - ``train.xyz``
     - ``train.zbxyz``

For ``bxyz``, replace ``.zbxyz`` with ``.bxyz``. ``dump_xyz`` retains its explicit basename: a terminal ``.xyz``, ``.bxyz`` or ``.zbxyz`` is replaced with the selected extension, otherwise the extension is appended. For a trailing ``*``, the step number is substituted first: ``snapshot.*`` produces ``snapshot.100.zbxyz`` at step 100. Text naming is unchanged.

All existing arrays, units, selection order and file numbering are preserved. Velocity-only and force-only outputs contain only ``vel:R:3`` and ``forces:R:3``, respectively; they do not synthesize coordinates or species. Their headers record time in fs, the run-local step and the selected grouping method/group ID when applicable. Empty selections produce valid zero-row binary frames. Coarse-grained species labels remain strings, including labels that are not chemical elements. Frame metadata and per-atom properties are separate namespaces, so total and per-atom ``virial`` can coexist.

``dump_netcdf`` always writes NetCDF. Thermodynamic tables, total dipole/polarizability, RDF and shock statistics are unchanged, including the statistical files accompanying observer and coarse-grained dumps.

NEB structures
--------------

The last ``output_format`` before ``neb_run`` also controls NEB structure output, independently of the ordering of ``neb_set`` lines. Keep the existing single-``neb_run`` input convention. The default remains text XYZ.

.. list-table:: NEB filenames and write behavior
   :header-rows: 1

   * - Text filename
     - Compressed filename
     - Behavior
   * - ``interpolate.xyz``
     - ``interpolate.zbxyz``
     - Replace with the interpolated path
   * - ``dump_traj.xyz``
     - ``dump_traj.zbxyz``
     - Append the full path at each ``dump_interval``
   * - ``peek_traj.xyz``
     - ``peek_traj.zbxyz``
     - Replace with the latest path at each ``peek_interval``
   * - ``final_traj.xyz``
     - ``final_traj.zbxyz``
     - Replace with the final path
   * - ``relaxed_is_fs.xyz``
     - ``relaxed_is_fs.zbxyz``
     - Replace with the two endpoints after ``need_relax``

Use ``.bxyz`` for uncompressed output. Each image is a structure frame containing ``species`` and Cartesian ``pos`` in angstrom, its physical lattice and PBC, ``energy`` (potential energy, eV) and ``enthalpy`` (the existing NEB image objective, eV). Variable-cell filter coordinates are not serialized as atoms. Binary metadata also records ``Step``, zero-based ``image`` and ``nimages``. Existing text precision and diagnostic/energy tables are unchanged; binary precision follows the global setting. No velocities, forces or integrator state are added to NEB output.

``is_name``, ``fs_name``, ``mid_name`` and ``mid_name_list`` accept the first frame of binary structure files. ``traj_name`` reads all frames sequentially, validating every checksum; use a single path such as ``final_traj.zbxyz`` or ``peek_traj.zbxyz``. ``dump_traj`` contains successive paths and is not automatically split into snapshots on input. All images must have the same atom count and element order. Array-only frames, missing/singular cells, damaged payloads and paths with fewer than two frames are rejected. Input detection is independent of the selected output format.

.. code:: text

   potential nep.txt
   neb_set traj_name final_traj.zbxyz interpolate 0
   output_format zbxyz precision lossless compression_level 3
   neb_run fire 0.01 1000

This resumes optimization from saved structures; it does not restore FIRE or INA internal state. For pure NEB input, ``output_format`` does not require an additional ``model.xyz``.

Reading and restarting
----------------------

.. code:: console

   gpumd -m restart.zbxyz

The model loader detects BXYZ headers and reads the first frame. A model must contain valid species (or atomic numbers), positions and a nonsingular cell; optional mass, charge, velocity and grouping data retain the text loader's semantics. Array-only files cannot initialize a model. The default model filename remains ``model.xyz``.

``dump_restart`` retains only the newest frame and follows the selected precision; choose ``lossless`` to avoid storage rounding. It saves the same fields as the existing text restart, not a complete thermostat/integrator checkpoint. Unit conversion can still incur floating-point rounding.

Continuous dumps append and flush each frame. Existing data are validated when opening for append; damaged files are rejected. Each binary frame is independent, with SHA-256 integrity checking and no inter-frame compression dependencies. Writing is synchronous and requires memory for the current frame and encoding buffers.

The bundled Python reader provides generic property tables without constructing ASE Atoms:

.. code:: python

   # Add this checkout's tools directory to PYTHONPATH.
   from binary_xyz.raw import iter_raw
   for frame in iter_raw("velocity.zbxyz"):
       velocities = frame.arrays["vel"]
       time_fs = float(frame.info["Time"])

It requires NumPy and, for compressed files, zstandard. Older structure-only BXYZ/OVITO readers reject array-only frames; this change does not extend OVITO. The frame specification and local C++ API extensions are documented in ``src/third_party/binary_xyz``.

Build
-----

.. code:: console

   module load cuda/12.8.1-el9
   make -C src -j8 USE_BXYZ=1 gpumd

The default build leaves BXYZ disabled. Zstd and OpenSSL development libraries and a C++17 host compiler are needed when enabled; CUDA sources retain their existing language standard. CMake accepts ``-DUSE_BXYZ=ON``. The bundled codec needs no download during a build. ``CXX``, ``BXYZ_CXXFLAGS`` and ``BXYZ_LIBS`` can select host dependencies in Make builds. On the 4090 cluster use CUDA 12.0.0 and a C++17-capable compiler; its system compiler may be too old.
