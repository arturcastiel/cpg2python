# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/). This project adheres to
Semantic Versioning.

## [Unreleased]

### Changed
- Licensed under the **GNU GPL v3**. Portions of the grid-processing
  algorithms derive from MRST's GPLv3 `processGRDECL.m`; see `NOTICE`.
- Continuous integration runs on the Node 24 Actions runtime.

### Added
- Formal end-to-end tutorial, [`docs/TUTORIAL.md`](docs/TUTORIAL.md).

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
