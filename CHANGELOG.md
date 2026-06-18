# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/). This project adheres to
Semantic Versioning.

## [Unreleased]

### Documentation
- README gained a **Grid input** section showing text-GRDECL vs binary
  EGRID side by side (PR-102).
- New `docs/grid-input.md` covering MAPAXES semantics, dtype trade-off,
  streaming behaviour, error contract, and dep notes.
- "EGRID is out of scope" line removed from the README's *Scope* and
  *Limitations* sections.

### Fixed
- `tests/test_reduce_area.py` no longer hard-codes a developer-machine
  path to `fault-projections-benchmarks`. It now looks up the sibling
  via `$FAULT_PROJECTIONS_BENCHMARKS`, then a sibling-of-repo path, then
  the legacy `/mnt/...` path, and uses `pytest.importorskip` if none
  resolves. CI environments without the sibling repo cleanly skip the
  module instead of erroring at collection.

### Added
- **Binary EGRID reader** (`egrid_reader.py`): `read_egrid(path, *,
  apply_mapaxes=True, dtype=np.float32) → CornerPointGrid`. Streams via
  `resfo.lazy_read` and short-circuits on the wanted keyword set, so
  LGR / NNC blocks are not materialised. MAPAXES rotation is applied
  by default; opt out with `apply_mapaxes=False` and a `RuntimeWarning`
  flags the choice.
- **Format-agnostic pivot type** `CornerPointGrid` (`corner_point.py`).
  Both `read_grdecl` and `read_egrid` return this type; downstream
  consumers (`from_grdecl`, `CornerCube`, mesh3d, ...) are unchanged.
  New fields: `mapaxes`, `mapaxes_applied`, `source` (`"grdecl"` /
  `"egrid"`).
- `resfo>=5,<6` promoted to a **hard dependency** — pure-Python on top of
  numpy you already require. No `[egrid]` extra; `read_egrid` is a
  first-class capability.
- `examples/read_egrid_demo.py` — minimal EGRID-read demo.
- `tests/data/_make_fixtures.py` (one-off generator) + committed fixtures
  `cart_2x2x2.EGRID`, `cart_2x2x2_rotated.EGRID`, `cart_2x2x2_no_actnum.EGRID`.

### Changed
- `Grdecl` is now a `TypeAlias` of `CornerPointGrid` (back-compatible —
  every existing import keeps working). The dataclass definition moved
  from `grdecl_reader.py` to `corner_point.py`.
- `read_grdecl` returns a `CornerPointGrid` (with `source="grdecl"`).
- EGRID reader keeps the on-disk dtype (float32) by default; the
  text-deck reader still returns float64.

## [Unreleased — prior]

### Changed
- Licensed under the **GNU GPL v3**. Portions of the grid-processing
  algorithms derive from MRST's GPLv3 `processGRDECL.m`; see `NOTICE`.
- Continuous integration runs on the Node 24 Actions runtime.

### Added
- Formal end-to-end tutorial, [`docs/TUTORIAL.md`](docs/TUTORIAL.md).
- **2.5D plan-view topology** (`planview.py`): `projected_topology(grid, plane)`
  collapses the grid onto an `xy`/`xz`/`yz` plane into nodes, edges, and
  face→cell-index records.
- **Full 3D mesh graph** (`mesh3d.py`): `build_mesh_graph_3d` (node-edge graph
  over all hex edges + a 3D KD-tree, with node-to-node shortest paths) and
  `mesh_edges`.
- **Boundary operator ∂₂ + geometry** (`mesh3d.py`): `face_edges` (each face's 4
  edges), `face_areas`, `face_centroids`, `cell_centroids`.
- **Planar minimal surface** (`minimal_surface.py`): `planar_minimal_surface`
  (least-area, hole-free cell-faces for a planar boundary), `fit_plane`,
  `project_to_plane`, `point_in_polygon`.
- **Surface boundary (∂₂ rim)** (`minimal_surface.py`): `boundary_edges` and
  `surface_boundary_loops` — the topological boundary of a face set, contained
  in the surface by construction.

## [0.2.0] — 2026-05-24

Handles **any** CPG grid, not just the matching-Z case.

### Added
- **Faulted (non-matching-Z) grids.** `build_topology` resolves i- and
  j-direction pillar offsets into real faces; a cell that straddles two
  cells across a fault produces the correct extra connections.
- **Pinch-outs.** `from_grdecl(grdecl, pinch_tol=0.0)` drops
  zero-thickness (collapsed) cells from the active set.
- **PINCH non-neighbour connections** — `UnstructuredGrid.nnc` bridges
  the cells above and below a collapsed layer (they are physically
  adjacent once it vanishes). A genuine inactive gap is *not* bridged.
- **Disconnected grids** — `UnstructuredGrid.component` labels every cell
  by connected component, and `UnstructuredGrid.split()` returns one
  independent grid per component.

### Known limitations
- k-direction faults raise `NonMatchingZError` (i/j fault dispatch only).

## [0.1.0] — 2026-05-22

First usable alpha — matching-Z grids.

### Added
- **GRDECL reader** (`read_grdecl` → `Grdecl`): DIMENS / SPECGRID / COORD /
  ZCORN / ACTNUM, `--` comments, `N*VALUE` run-length expansion, recursive
  `INCLUDE`, and cell properties (PORO, PERMX/Y/Z, NTG, SATNUM, …).
- **Corner-cube assembly + node deduplication** (`from_grdecl` →
  `CornerCube`).
- **Topology builder** (`build_topology` → `UnstructuredGrid`): i/j/k face
  finding, the `cell_faces` table, and removal of inactive / zero-volume
  cells.
- **Cell properties** (`attach_props`), **cell & face geometry**
  (`cell_geometry`, `face_geometry` — centroids, areas, normals, volumes),
  and the active-id ↔ `(i, j, k)` **`IndexMap`**.
- **Adjacency graph + shortest path** (`cell_adjacency`, `shortest_path`),
  with an optional `scipy.sparse` fast path (`cell_adjacency_csgraph`,
  `shortest_path_fast`) under the `[fast]` extra.

### Known limitations
- Matching-Z grids only; ASCII GRDECL only (no binary `.EGRID`); not on PyPI.
