# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added (phase 2-faults — working toward v0.2.0)
- **PINCH detection — pinched-out cells are dropped.** `from_grdecl`
  now folds zero-thickness (collapsed) cells out of the active set via
  `cpg._pinched_cell_mask(g, tol)`: a cell is pinched when its vertical
  thickness is ≤ `pinch_tol` on all four pillars. New `pinch_tol`
  parameter on `from_grdecl` (default `0.0` → only exactly-collapsed
  cells; ordinary decks unchanged). New fixture `pinched_1x1x3.grdecl`
  (middle layer collapsed) + `tests/test_pinch.py`. PR-12 will bridge
  the cells above/below a pinch with an NNC. (PR-11 of phase 2-faults.)
- **j-fault regression: `simple_faulted_3x3x2_jfault.grdecl` + 2 tests.**
  The j-direction analogue of PR-9's i-fault fixture — a 3×3×2 deck
  with a +0.6 Z offset on the j=2 row, faulted across all 3 i-columns.
  Exercises `find_j_faces` dispatch on a multi-column grid (27 j-faces:
  6 boundary + 6 matching + 9 fault + 6 boundary) and `build_topology`
  end-to-end (18 cells). Symmetric pin to the PR-9 i-fault check.
  (PR-10 of phase 2-faults.)
- **Regression fixture + 3 tests for the 3×3×2 multi-column i-fault.**
  Adds `tests/fixtures/simple_faulted_3x3x2.grdecl` — a 3×3×2 deck
  with a +0.6 Z offset on the i=2 column, faulted across all 3
  j-rows. The offset makes the upper i=1 cell straddle both i=2
  cells → 3 connections per stack → 9 faulted i-faces. This exercises the
  `find_i_faces` dispatch on a multi-j-row grid (the 2×2×2 fixtures
  only have 2 j-rows). 3 new tests in
  `tests/test_build_topology_faulted.py`:
    · `test_build_topology_on_simple_faulted_3x3x2` — end-to-end:
      `build_topology` returns 18 cells.
    · `test_simple_faulted_3x3x2_i_face_dispatch_count` — verifies
      the dispatch produces 27 i-faces (6 boundary + 6 matching
      inner + 9 fault inner + 6 boundary).
    · `test_simple_faulted_3x3x2_neighbour_convention` — confirms
      `_move_minus_one_to_col1` is preserved on the larger fixture.
  Pins the v0.2.0 capability so a future change can't quietly break
  multi-row fault dispatch. (PR-9 of phase 2-faults.)
- **`build_topology` end-to-end on faulted decks** (headline of
  phase 2-faults). The top-level i/j fault guard inside
  `build_topology` was removed; `find_i_faces` (PR-6) and
  `find_j_faces` (PR-7) now handle the dispatch internally, and
  the topology builder simply consumes their unified Faces
  output. On `simple_faulted_2x2x2.grdecl` and
  `simple_faulted_2x2x2_j.grdecl`, `build_topology(cube)` now
  produces a valid `UnstructuredGrid` with `n_cells == 8` and
  the expected `cell_faces` shape — what would previously raise
  `NonMatchingZError`. **k-direction faults still raise**:
  the 2-pillar sweep-line algorithm of PR-2/PR-4 doesn't apply
  to the 4-pillar geometry of k-faces. The remaining
  `build_topology` raise is now narrowed to `_faulted_pillar_mask_k`
  only, with a clear "k-direction fault dispatch not yet
  implemented" message. k-fault dispatch is on the PR-9 polish
  TODO. Five v0.1.0+PR-1 Tier A tests were updated from
  "expect raise" to "expect success" (with `grid.n_cells == 8`
  assertions); the renamed test names are
  `test_build_topology_succeeds_on_*`. New test file
  `tests/test_build_topology_faulted.py` adds 3 end-to-end
  scenarios: build_topology on simple_faulted_2x2x2 (i-fault),
  simple_faulted_2x2x2_j (j-fault), and a synthetic k-fault
  fixture that still raises with the new message. (PR-8 of
  phase 2-faults.)
- **`find_j_faces` widened to dispatch matching-Z vs faulted inline**
  (D-2F-1, symmetric to PR-6). Adds the new `_find_faulted_j_faces`
  helper that mirrors `_find_faulted_i_faces` along the j-axis:
  walks each faulted (i, j) j-pillar-pair stack, builds (za, zb) z
  arrays + pa/pb corner-id arrays from the +j/−j sides of A/B cells
  (j-slots 2j+1 and 2j+2), calls `find_connections` then
  `compute_face_geometry`, and emits a Faces record with
  `direction = 1`. When the j-mask is empty, the matching-Z fast path
  is byte-identical to v0.1.0 (Tier A invariance). PR-5 restrictions
  carry over: all-active stacks + 4-corner polygons only. New test
  fixture `tests/fixtures/simple_faulted_2x2x2_j.grdecl` is a
  j-axis mirror of `simple_faulted_2x2x2.grdecl` (j=0 row flat,
  j=1 row offset +0.5 — produces 2 i-stacks × 3 connections =
  6 j-fault faces). 9 new tests in
  `tests/test_find_j_faces_dispatch.py` cover Tier A invariance on
  3 matching-Z fixtures, the dispatch on the new j-fault fixture
  (14 total j-faces, 6 inner / 8 boundary, direction=1, valid
  node + neighbour IDs, `_move_minus_one_to_col1` convention),
  and the still-raising `build_topology` guard.  `build_topology`
  and `find_k_faces` still raise on faulted decks — PR-8 widens
  `find_k_faces` and drops the `build_topology` guard. (PR-7 of
  phase 2-faults.)
- **`find_i_faces` widened to dispatch matching-Z vs faulted inline**
  (D-2F-1). When no faulted stacks are present, the matching-Z fast
  path is byte-identical to v0.1.0 (Tier A invariance). When faulted
  stacks are present, the matching-Z path runs with faulted (i, j)
  stacks dropped and concatenates with the output of
  ``_find_faulted_i_faces`` (PR-5). On the simple_faulted_2x2x2
  fixture, `find_i_faces` now returns 14 i-faces (8 boundary + 6
  fault inner) where v0.1.0 raised `NonMatchingZError`.
  `build_topology` and `find_j_faces` / `find_k_faces` still raise
  on faulted decks; those widenings land in PR-7 (j), PR-8 (k) +
  PR-8 drops the `build_topology` raise. The v0.1.0 Tier A test
  `test_fault_raises_clean_error` was renamed to
  `test_fault_dispatches_in_find_i_faces` and updated to assert
  success: `find_i_faces(fault_simple)` now returns 12 i-faces
  (2 fault via dispatch + 10 matching) instead of raising. 8 new
  tests in `tests/test_find_i_faces_dispatch.py` cover the Tier A
  fast path on 3 matching-Z fixtures and the dispatch on the faulted
  fixture (face counts, valid node + neighbour IDs,
  _move_minus_one_to_col1 convention, inner/boundary split 6/8,
  still-raising `build_topology` guard). (PR-6 of phase 2-faults
  — note: PR-6 shipped as the (#19) main merge plus a follow-up
  PR that added the missing CHANGELOG entry and the
  `test_unstructured.py` rename, both lost to a stash/pop race
  with the wrong PR-5 re-run.)
- `cpg2unstructured.unstructured._find_faulted_i_faces` — first
  end-to-end integration of phase 2-faults. Walks every faulted
  i-pillar-pair stack, builds (za, zb) arrays from cube.P + cube.nodes,
  invokes find_connections (PR-2) for the sweep-line pairing, then
  compute_face_geometry (PR-4) for the 4-corner polygons, and returns
  a standard Faces record (with neighbours, direction=0, nodes sorted
  per the existing Faces convention). Stack-local (no MRST dZ trick
  — per-stack loop is plenty fast for typical decks). **PR-5 scope
  restrictions** that will be lifted in PR-9: (a) all-active stacks
  only — inactive cell mid-stack raises NonMatchingZError; (b)
  4-corner polygons only — slanted-cell fault faces with envelope
  crossings or diamond cases raise (PR-9 will extend Faces to
  variable-length polygons). The common case of "rigid blocks with
  vertical Z offset" — including the simple_faulted_2x2x2 fixture
  — is fully supported. NOT yet integrated into find_i_faces or
  build_topology; both still raise on faulted decks (the widening
  is PR-6). 8 unit tests in tests/test_find_faulted_i_faces.py
  cover: 3 matching-Z fixtures return empty, simple_faulted_2x2x2
  produces 6 fault faces (2 j-stacks × 3 connections per stack =
  6 faces, matching the §6b worked example), neighbour-id validity,
  sorted-corners convention, no new nodes appended (4-corner only),
  inactive-cell-in-stack raise, and build_topology still raising
  pre-widening. (PR-5 of phase 2-faults.)
- `cpg2unstructured._fault_geometry.compute_face_geometry` — the
  8-position J-table polygon assembler for fault-face polygons.
  For each connection (a row of the connection table from PR-2),
  constructs a clockwise polygon whose normal points from cell-A
  to cell-B. Output polygons have between 3 and 8 corners depending
  on which envelope crossings + diamond cases trigger. Input: ``pa``,
  ``pb`` of shape ``(n, 4)`` int64 (node IDs for A-face and B-face
  corners, column order ``[bot_p1, bot_p2, top_p1, top_p2]``), plus
  global points table ``(n_points, 3)`` float64. Output: updated
  points table (intersection nodes appended), ``num_nodes`` per face,
  flat ``corners`` array splittable via ``cumsum(num_nodes)``.
  Implements all 4 diamond cases from MRST (Cases 1/2 for A12 × B34,
  Cases 3/4 for A34 × B12), pillar pinch dedup (bot == top on the
  same pillar collapses), and intersection dedup via ``np.unique``.
  Direct port of MRST's ``computeFaceGeometry`` (``processGRDECL.m``
  lines 853-1160). 12 unit tests in
  ``tests/test_compute_face_geometry.py`` cover empty input, matching
  faces (single + multi-row), simple half-offset (no crossings),
  top-edge crossing (5 corners with p6), bot-edge crossing (5
  corners with p2), both-edge crossings (6 corners), diamond Case 1
  (vertex@7 with two pillar nullifications), pillar-pinch (3-corner
  triangle), no-intersection contract (points unchanged), new-node
  ID contiguity, and clockwise normal invariant on matching faces.
  (PR-4 of phase 2-faults — the algorithmic hot spot.)
- `cpg2unstructured._fault_geometry.line_intersection` — XYZ
  intersection of two line-segment families, parametrised by z.
  Two lines, each defined by its start + end XYZ on the same pillar
  pair (so both go pillar-1 → pillar-2; same (x, y) trajectory in
  the vertical-pillar limit; z values differ). Solves
  ``t = (z_B^{p1} - z_A^{p1}) / (Δz_A - Δz_B)`` and reconstructs
  XYZ along line A at ``t``. Vectorised over leading axes. Raises
  `ValueError` if any pair has parallel z-trajectories
  (``|Δz_A - Δz_B| < 1e-12`` — R-3 guard from the phase 2-faults
  study). Direct port of MRST's ``intersection`` helper
  (``processGRDECL.m`` lines 1164-1190). 9 unit tests cover simple
  crossings, intersection at line endpoints, beyond-endpoint
  extrapolation, exact-z output, parallel-z raise, horizontal A
  meets sloped B, vectorised (n, 3) shape, float32 input → float64
  output, and a slanted-(x,y)-pillar case. (PR-3 of phase 2-faults.)
- `cpg2unstructured._fault_connections` module — pure 1D segment-overlap
  algorithm used to pair up faulted-stack cells. Three functions:
  `overlap(xa1, xa2, xb1, xb2) -> bool` (strict interval overlap),
  `do_intersect(za_top, za_bot, zb_top, zb_bot) -> bool` (A-face vs
  B-face overlap including pinch guards and "le fix speciale" for the
  diamond case where two faces cross without single-pillar overlap),
  and `find_connections(za, zb) -> (n, 2) int64` (the sweep-line
  driver — walks A-cells outer, B-cells inner with per-pillar
  checkpoint tracking for amortised O(n_a + n_b) cost). No geometry
  in this PR — pure integer index pairs out. Tests on the §6b worked
  example confirm 3 connections for the 2×2 half-offset case;
  17 unit tests in `tests/test_fault_connections.py` cover overlap +
  do_intersect predicates (including diamond + pinch edge cases),
  matching stacks, fully-disjoint stacks, asymmetric stack sizes,
  pinched A-cells in the middle of a stack, and the int64 type
  contract on empty outputs. (PR-2 of phase 2-faults.)

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
