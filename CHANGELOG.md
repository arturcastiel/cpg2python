# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Initial repo skeleton (PR-1).
- Toy 2×2×2 cartesian CPG fixture + reference topology pipeline
  (`cpg2unstructured._toy`) with integer-invariant smoke tests (PR-2).
- GRDECL reader (`read_grdecl`, `Grdecl`) supporting DIMENS / SPECGRID /
  COORD / ZCORN / ACTNUM with `--` comments and `N*VALUE` run-length
  expansion. Unknown keywords are collected on `Grdecl.unrecognized`
  rather than raised. Domain exceptions in `cpg2unstructured._errors`
  (PR-3).
- GRDECL reader follows `INCLUDE 'file.inc' /` directives recursively,
  resolving filenames relative to the file containing the directive.
  Cycle detection raises `GrdeclIncludeError`. Property keywords
  (PORO, PERMX/Y/Z, NTG, MULTX/Y/Z, PRESSURE, SWAT, SGAS, SOIL,
  SATNUM, PVTNUM, EQLNUM, FIPNUM) populate `Grdecl.props` as 1-D arrays
  of length `NX·NY·NZ`. Integer properties use `int32` (sufficient for
  region IDs > 127). Duplicate property keywords warn via
  `GrdeclDuplicatePropertyWarning` with last-write-wins semantics
  (PR-4).
- Corner-cube assembly + node deduplication via
  `cpg2unstructured.cpg.from_grdecl(g) -> CornerCube`. Returns
  `nodes: (Nnodes, 3) float64`, `P: (2*NX, 2*NY, 2*NZ) int64`
  (the back-map from corner-cube slot to unique node, with `-1` for
  slots owned by inactive cells), and `actnum_3d: (NX, NY, NZ) bool`.
  Linear pillar interpolation; quantised int64 dedup via `np.unique`
  (no kd-tree, no hash table) with `_QUANT = 1e9` matching `_toy._QUANT`.
  `from_grdecl` and `CornerCube` re-exported at the top level (PR-5).
- `cpg2unstructured.unstructured.find_i_faces` and `find_j_faces` cover
  inner + boundary face topology along the i and j directions for
  matching-Z grids. Output is a `Faces` record with sorted node-quads,
  neighbour pairs (`-1` in column 1 for boundary faces), and an int8
  `direction` field (0 = i, 1 = j, 2 = k for PR-7). Faulted decks raise
  `NonMatchingZError`. Inactive cells (ACTNUM=0) cleanly degrade
  shared faces to boundary faces of the active side without leaking
  the `-1` sentinel into face quads. Top-level re-exports for `Faces`,
  `find_i_faces`, `find_j_faces`, and `find_ij_faces` (PR-6).
- `find_k_faces` for the k direction (symmetric to i/j) and
  `build_topology(cube)` which assembles i+j+k faces, inverts to a
  `cell_faces` table of shape `(Ncells, 6)` with slot ordering
  `[i-, i+, j-, j+, k-, k+]`, drops inactive + zero-volume cells, and
  rebuilds contiguous cell ids. Returns the topology-only `_TopoMesh`
  (geometry lands in PR-8) carrying `nodes`, `faces`, `cell_faces`,
  `cell_corners`, and Eclipse-natural `original_ijk`. Zero-volume
  detection is per-face: a cell is dropped iff at least one of its
  6 faces has < 3 unique node ids. Top-level re-exports for
  `find_k_faces` and `build_topology` (PR-7).
- `cpg2unstructured.geometry` module: `face_geometry(nodes, faces,
  cell_corners) -> FaceGeometry` and `cell_geometry(nodes, faces,
  cell_faces, fg) -> CellGeometry`. Face centroid = arithmetic mean of
  the 4 corner nodes; face area = sum of two triangle areas via
  cross-product; face normal = unit vector consistent with
  `faces.neighbours[:, 0] → faces.neighbours[:, 1]` (outward from cell 0
  for boundary faces). Cell volume + centroid via the divergence
  theorem (`V = (1/3) Σ c·n·A`, exact to 1e-12 on unit cubes).
  Degenerate faces (norm_len < 1e-15) return zero normal + zero area
  without raising. Top-level re-exports for `FaceGeometry`,
  `CellGeometry`, `face_geometry`, `cell_geometry` (PR-8).
