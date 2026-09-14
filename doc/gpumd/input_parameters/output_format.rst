.. _kw_output_format:

output_format
=============

Select the output encoding for MD structures and per-atom arrays. This setting does not enable any dump command or change its interval or atom selection.

Syntax
------

.. code:: text

   output_format xyz
   output_format bxyz precision lossless
   output_format zbxyz precision auto compression_level 3

The default is ``xyz``: every command keeps its existing text format, including ``velocity.out`` and ``force.out``. Binary formats require a build with ``USE_BXYZ=1``. ``bxyz`` writes uncompressed v4 frames; ``zbxyz`` writes independently compressed v5 frames.

Optional key/value pairs may appear in either order. ``precision auto`` (the default for binary output) selects float32 or float64 per property using relative tolerance 1e-7 and absolute tolerance zero. ``precision lossless`` uses zero tolerance and preserves the original floating values; exactly representable values may still use float32. Integer and string values are lossless. ``compression_level`` is allowed only for ``zbxyz`` and accepts 1 through 22, default 3. Text mode accepts no binary options. Invalid, repeated or incomplete options are errors.

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

``dump_netcdf`` always writes NetCDF. Thermodynamic tables, total dipole/polarizability, RDF and shock statistics are unchanged, including the statistical files accompanying observer and coarse-grained dumps. NEB-specific output is not controlled by this keyword.

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
