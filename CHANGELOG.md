# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added (phase 2-faults — working toward v0.2.0)
- `cpg2unstructured.unstructured._faulted_pillar_mask_{i,j,k}` —
  per-cell-pair bool predicates that return ndarrays of shape
  `(NX-1, NY, NZ)` / `(NX, NY-1, NZ)` / `(NX, NY, NZ-1)`. True means
  the active face between the cell pair has non-matching-Z corner
  values. Replaces the v0.1.0 `_verify_matching_z_*` raise-on-fault
  helpers (renamed; D-2F-1). The new predicate is the dispatch
  primitive that future PRs (PR-6/PR-7/PR-8) will use to widen
  `find_*_faces` to handle faulted decks inline. Currently
  `find_*_faces` and `build_topology` still raise `NonMatchingZError`
  on faulted decks — behaviour-preserving — but the new
  `simple_faulted_2x2x2.grdecl` fixture exercises the new mask
  helpers and confirms the rename refactor is correct. (PR-1 of
  phase 2-faults.)
- `tests/fixtures/simple_faulted_2x2x2.grdecl` — 8-cell deck with a
  HALF-CELL Z fault between i=0 and i=1 columns. All-active vertical
  pillars 0..3 (extra headroom for the offset bot). The i=0 column
  cells use Z values 0/1/2, the i=1 column uses 0.5/1.5/2.5 — pure
  per-column offset, no j or k faulting.
- `tests/test_faulted_mask.py` — 11 tests covering matching-Z
  fixtures (all 3 mask helpers return all-False on simple_2x2x2,
  simple_3x3x2, with_include), the faulted fixture (mask flips True
  along the i-fault stack only), behaviour-preserving raise points
  in `find_i_faces`/`find_j_faces`/`find_k_faces`/`build_topology`,
  and a Tier A regression test on matching-Z `find_i_faces`.

## [0.1.0] - 2026-05-22

First "you can actually use this" alpha release. NumPy-only by default;
optional scipy.sparse fast path under `[fast]`. Matching-Z grids only.
Not on PyPI — local editable install per the README.

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
  the 4 corner nodes; face area + normal via the diagonal-cross-product
  formula (`(1/2) |d1 × d2|`, where d1, d2 are the 2 longest pairwise
  inter-vertex distances — order-independent on PR-6's sort-by-id quads);
  normal oriented per `faces.neighbours[:, 0] → faces.neighbours[:, 1]`
  (outward from cell 0 for boundary faces). Cell volume + centroid via
  the divergence theorem (`V = (1/3) Σ c·n·A`, exact to 1e-12 on unit
  cubes). Degenerate faces (norm_len < 1e-15) return zero normal + zero
  area without raising. Top-level re-exports for `FaceGeometry`,
  `CellGeometry`, `face_geometry`, `cell_geometry` (PR-8).
- `cpg2unstructured.index_map.IndexMap` — bi-directional active-cell id
  ↔ Eclipse `(i, j, k)` map. Constructed from `(cart_dims,
  original_ijk)` (e.g. `mesh.original_ijk` from `build_topology`).
  Forward lookup is an array index. Reverse lookup builds a dense
  `(NX, NY, NZ)` int64 array lazily on first access and caches it.
  Inactive cells return `None` from `active_id(i, j, k)`; out-of-range
  raises `IndexError`. Helpers: `cell_ijk`, `linear_to_cell`,
  `cells_in_layer`, `n_active`, `cart_dims`. `__slots__` keeps the
  per-instance footprint small. Top-level re-export for `IndexMap` (PR-9).
- `cpg2unstructured.graph` module — pure-numpy cell-cell adjacency
  and Dijkstra shortest-path. `cell_adjacency(mesh, weights=...)`
  returns a `list[list[(neighbour_id, weight)]]` with three weight
  modes: `"unit"`, `"distance"` (requires `cell_centroids` from
  `cell_geometry`), `"none"`. Adjacency is symmetric; only inner
  faces produce edges. `shortest_path(adj, src, dst)` runs Dijkstra
  via heapq + numpy and returns `(total_weight, path)` where path
  includes both endpoints. Unknown weight modes raise `ValueError`
  (defence against typos). Unreachable destinations raise
  `ValueError("no path …")`. Scipy-free; PR-11 will add a
  scipy-CSR fast path under the `[fast]` extra. Top-level re-exports
  for `cell_adjacency` and `shortest_path` (PR-10).
- Optional scipy.sparse fast path under the `[fast]` extra.
  `cell_adjacency_csgraph(mesh, weights=...)` returns a symmetric
  `scipy.sparse.csr_array` of shape `(n_active, n_active)`; nnz equals
  the slow-path directed-edge count. `shortest_path_fast(csr, src,
  dst=None)` wraps `scipy.sparse.csgraph.dijkstra` — drop-in for
  `shortest_path` when `dst` is an int (returns `(dist, path)`),
  full-source mode when `dst is None` (returns `(distances,
  predecessors)` arrays). Distances agree with the pure-numpy path to
  `atol=1e-12` (`rtol=0`); tie-breaking on equal-cost routes may
  differ between heapq-Python and scipy's C impl, so the agreement
  test asserts cost-only on the distance-weight case. Without scipy,
  importing the package still works; calling the fast functions
  raises `ImportError` whose message names the `[fast]` extra. New
  `tests/test_graph_fast.py` is skipped via `pytest.importorskip` when
  scipy is absent. `tests/conftest.py` added — `mesh_2x2x2` moved
  here from `tests/test_graph.py`; new `mesh_5x5x3_with_centroids`
  session fixture builds a synthetic 75-cell deck for agreement
  tests (downscaled from the spec's 10×10×5 to keep the synthetic
  GRDECL string + parse cheap). Top-level re-exports for
  `cell_adjacency_csgraph` and `shortest_path_fast` (PR-11).
- `UnstructuredGrid` — public face of the package; promotes the
  internal `_TopoMesh` dataclass into a documented API with a
  read-only properties dict, a `grid["PORO"]` accessor, `n_cells`
  property, lazy-cached `index_map` property, and `__contains__`.
  All previous fields (`nodes`, `faces`, `cell_faces`, `cell_corners`,
  `original_ijk`, `cart_dims`) carry over unchanged. `_TopoMesh`
  survives as a backward-compat type alias so existing
  `cpg2unstructured.graph` imports keep working unchanged. Frozen
  dataclass; the lazy `_index_map` cache uses `object.__setattr__`
  (D-PR12-1, same pattern `functools.cached_property` uses on frozen
  dataclasses). `build_topology` now returns `UnstructuredGrid`
  (typed); the runtime shape is identical to PR-7's `_TopoMesh`.
- `attach_props(grid, grdecl) -> UnstructuredGrid` filters every
  property in `grdecl.props` (Eclipse-natural, length `NX·NY·NZ`)
  down to active-cell order via `grid.index_map.linear_to_cell`.
  Returns a NEW grid (via `dataclasses.replace`) — the input grid is
  not mutated. Raises `ValueError` on `cart_dims` mismatch with a
  helpful diff. Property dtypes (float64 / int32) are preserved from
  PR-4's whitelist classification. Missing-key access via
  `grid["DOES_NOT_EXIST"]` raises `KeyError` listing the available
  property keys. New `tests/test_props.py` (13 tests) covers round-trip,
  active-cell filtering, dtype preservation, purity, frozen
  enforcement, lazy IndexMap caching, and a PR-10 graph
  interop regression (the rename must not break callers). New
  `fixtures` pytest fixture in `tests/conftest.py` exposes the
  fixture-path constant for tests that need it. Top-level re-exports
  for `UnstructuredGrid` and `attach_props` (PR-12).
- Full README rewrite: status banner, install paths (`-e .`,
  `-e .[fast]`, `-e .[dev]`), 30-second quickstart, API map with
  per-function PR links, `UnstructuredGrid`-at-a-glance reference,
  v0.1.0 limitations. `examples/quickstart.py` is a runnable demo
  that parses `examples/quickstart.grdecl` (a self-contained 5×5×3
  deck with 70 active cells, mild PORO heterogeneity, and 5 inactive
  cells to exercise filtering) end-to-end through parse → mesh →
  properties → graph → shortest path; if scipy is installed it also
  exercises the fast path. Version stamped at **0.1.0** in
  `pyproject.toml` and `cpg2unstructured.__version__` (PR-13).

### Known limitations
- Matching-Z (no fault) grids only — non-matching-Z raises
  `NonMatchingZError`.
- ASCII GRDECL only; no `.EGRID` binary.
- Single-process, in-memory.
- Not on PyPI yet — local editable install only (D-06).
