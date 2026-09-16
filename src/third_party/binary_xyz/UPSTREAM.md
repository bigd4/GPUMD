# Source provenance

Source: git@gitlab.com:qhjia/binary_xyz.git, commit 31ff53565e77efa8604246f87dfde14ba1abc90d.

This snapshot contains the native C++ library and its tests. The upstream snapshot supplies no standalone LICENSE file; no license is inferred or reassigned here. Existing source notices are preserved. This is the user-owned format library integrated at the user's request.

Local extension: RawFrame, Reader::next_raw/read_raw and Writer::write(RawFrame) support generic property tables, arbitrary species labels and separate frame/array namespaces. The v4/v5 wire encoding is unchanged; older structure-only consumers reject incomplete structures. The original Frame interface still requires valid elements and positions. See FORMAT.md for details.

Local precision extension: `Options::property_tolerances` overrides `(rtol, atol)` by property name for both Frame and RawFrame writers, without changing the byte layout. GPUMD auto output uses `(0, 1e-6)` for `pos` and `unwrapped_position` (angstrom); `forces` uses `(0, 1e-5)` in eV/angstrom and `vel` uses `(0, 1e-6)` in angstrom/fs; other arrays keep the library defaults. Lossless output does not install overrides.
