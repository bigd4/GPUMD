# Source provenance

Source: git@gitlab.com:qhjia/binary_xyz.git, commit 31ff53565e77efa8604246f87dfde14ba1abc90d.

This snapshot contains the native C++ library and its tests. The upstream snapshot supplies no standalone LICENSE file; no license is inferred or reassigned here. Existing source notices are preserved. This is the user-owned format library integrated at the user's request.

Local extension: RawFrame, Reader::next_raw/read_raw and Writer::write(RawFrame) support generic property tables, arbitrary species labels and separate frame/array namespaces. The v4/v5 wire encoding is unchanged; older structure-only consumers reject incomplete structures. The original Frame interface still requires valid elements and positions. See FORMAT.md for details.
