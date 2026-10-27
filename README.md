# cpg2unstructured

Turn an Eclipse corner-point geometry (CPG) `.grdecl` deck into a clean
**unstructured mesh** plus a **cell-to-cell adjacency graph** — in a few
lines of NumPy. Built for grid *topology*: it handles **faulted
(non-matching-Z) grids**, **pinch-outs**, and **disconnected reservoirs**,
not just the simple matching-Z case.

- **Pure NumPy** core (optional `scipy.sparse` backend for fast graph ops)
- **Faulted grids** — non-matching-Z pillar offsets resolved into real faces
- **PINCH non-neighbour connections** — collapsed layers bridged via `grid.nnc`
- **Disconnected grids** — connected-component labels + `grid.split()`
- **Properties on cells**, **cell/face geometry**, and **shortest-path** search

> **Scope.** A focused topology library — *any* CPG grid → an unstructured
> representation. It is intentionally **not** a reservoir simulator (no
> transmissibility, no property algebra).
>
> Reads both **ASCII GRDECL** decks and **binary EGRID** (`.EGRID` /
> `.FEGRID`) files into the same `CornerPointGrid` pivot — every
> downstream stage is format-agnostic.

Python ≥ 3.10 · GNU GPL v3 · by Dr. Artur Castiel Reis de Souza

---

## Install

```bash
git clone <repo> && cd cpg2python
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # numpy-only
pip install -e ".[fast]"    # + scipy.sparse for fast Dijkstra
pip install -e ".[dev]"     # + pytest, pytest-cov, build
```

## Grid input

`cpg2unstructured` reads two corner-point grid formats into the same
`CornerPointGrid` pivot dataclass; the rest of the pipeline doesn't care
which reader produced it.

| Format | Reader | Notes |
|--------|--------|-------|
| Text GRDECL (`.grdecl`)     | `read_grdecl(path)`                                              | `INCLUDE` recursion · PORO / PERMX / SATNUM pass-through |
| Binary EGRID (`.EGRID` / `.FEGRID`) | `read_egrid(path, *, apply_mapaxes=True, dtype=np.float32)` | streams via [`resfo`](https://github.com/equinor/resfo); MAPAXES applied by default; LGR/NNC blocks skipped |

```python
from cpg2unstructured import read_grdecl, read_egrid, from_grdecl

g = read_grdecl("model.grdecl")        # text deck
g = read_egrid("MODEL.EGRID")          # binary EGRID, COORD/ZCORN as f32
g = read_egrid("MODEL.EGRID", apply_mapaxes=False)   # raw LOCAL coords (warns)
g = read_egrid("MODEL.EGRID", dtype=np.float64)      # widen on read

cube = from_grdecl(g)                  # same downstream call for both readers
```

The legacy name `Grdecl` is a `TypeAlias` of `CornerPointGrid` — every
existing `from cpg2unstructured import Grdecl` keeps working unchanged.
See [`examples/read_egrid_demo.py`](examples/read_egrid_demo.py) for an
end-to-end demo, and the `tests/test_egrid_*.py` modules for the full
behavioural contract (MAPAXES, dtype, streaming short-circuit, hypothesis
property test).

## Quickstart

```python
from cpg2unstructured import (
    read_grdecl, from_grdecl, build_topology, attach_props,
    cell_adjacency, shortest_path,
)

g    = read_grdecl("examples/quickstart.grdecl")
grid = attach_props(build_topology(from_grdecl(g)), g)

print(grid.cart_dims, grid.n_cells)   # (5, 5, 3)  70 active cells
print(grid["PORO"].mean())            # ~0.20

adj         = cell_adjacency(grid, weights="unit")
dist, path  = shortest_path(adj, 0, grid.n_cells - 1)
print(f"shortest path: {len(path)} cells, distance {dist}")
```

Run the bundled demo: `python examples/quickstart.py`. A full,
worked walkthrough lives in [`docs/TUTORIAL.md`](docs/TUTORIAL.md).

## Capabilities

**Faulted (non-matching-Z) grids.** Offset pillars are resolved inline — a
cell straddling two cells across a fault produces the correct extra faces:

```python
grid = build_topology(from_grdecl(read_grdecl("deck_with_faults.grdecl")))
```

**Pinch-outs → NNCs.** Zero-thickness (collapsed) cells are dropped, and the
cells above and below are reconnected with a non-neighbour connection:

```python
cube = from_grdecl(g, pinch_tol=0.0)   # drop exactly-collapsed cells
grid = build_topology(cube)
grid.nnc                               # (M, 2) bridged cell-id pairs
```

**Disconnected grids.** Independent "islands" are labelled, and can be split
into standalone grids on demand:

```python
grid.component          # (Ncells,) component label per cell
parts = grid.split()    # list[UnstructuredGrid], one per component
```

**Editing ZCORN (fault slides).** Moving corners along their pillars changes
ZCORN only; `unwelded_faces` shows which lateral faces opened and
`zcorn_validity` rejects inverted or overlapping cells:

```python
dz = np.zeros((2 * nx, 2 * ny))       # one depth shift per column corner
dz[2 * i0:, :] = 7.0                  # columns i >= i0 slide 7 m down
slid = apply_zcorn_displacement(g, dz)
open_i, open_j = unwelded_faces(slid) # (NX-1, NY), (NX, NY-1) bool
assert zcorn_validity(slid).ok
```

## Examples

More runnable, self-checking examples live in the
[cpg-fault-fieldbook](https://github.com/arturcastiel/cpg-fault-fieldbook),
which uses this library together with
[fault-projections](https://github.com/arturcastiel/fault-projections):
29 lessons, from reading a deck to sliding faults in Z by editing ZCORN
and re-processing the grid (lessons 19–29, with figures). Re-processing a
slid deck relies on `build_topology` pairing cells across a fault by depth
overlap (0.3.0).

## API

| Function / class | Purpose |
|---|---|
| `read_grdecl(path) → Grdecl` | parse a `.grdecl` deck |
| `from_grdecl(grdecl, pinch_tol=0.0) → CornerCube` | corner-cube assembly + node dedup; drops pinched cells |
| `build_topology(cube) → UnstructuredGrid` | full mesh: i/j/k faces, cell-face table, NNCs, components |
| `attach_props(grid, grdecl) → UnstructuredGrid` | attach cell properties (PORO, PERM, …) |
| `face_geometry(...)` / `cell_geometry(...)` | centroids, areas, normals, volumes |
| `zcorn_cube(grid)` / `with_zcorn(grid, cube)` | ZCORN as `(2NX, 2NY, 2NZ)` corner slots and back |
| `apply_zcorn_displacement(grid, dz)` | shift corners along their pillars (per corner, horizon or slot) |
| `unwelded_faces(grid)` / `zcorn_validity(grid)` | opened lateral faces; inverted / overlapping cells |
| `IndexMap(cart_dims, ijk)` | bidirectional active-id ↔ `(i, j, k)` |
| `cell_adjacency(grid, weights=...)` | adjacency list (NumPy) |
| `cell_adjacency_csgraph(grid, ...)` | adjacency CSR (`scipy.sparse`) |
| `shortest_path(adj, src, dst)` | Dijkstra (NumPy + heapq) |
| `shortest_path_fast(csr, src, dst=None)` | Dijkstra (`scipy.sparse.csgraph`) |

### `UnstructuredGrid` at a glance

```python
grid.nodes          # (Nnodes, 3) float64 — unique XYZ
grid.faces          # Faces: nodes (N,M), loops (N,M), neighbours (N,2),
                    #   direction (N,); loops = the nodes in cyclic order,
                    #   normal out of neighbours[:,0]. M = 4 unless a fault
                    #   polygon or a k-face has 5+ corners
grid.n_corner_nodes # nodes[:n] are cell corners; later ones are fault
                    #   edge-intersection nodes (None = all corners)
grid.cell_faces     # (Ncells, 6+E) int64 — [i-, i+, j-, j+, k-, k+], then extra
                    #   fault faces; -1 = unused (E = 0 without faults)
grid.cell_corners   # (Ncells, 8) int64 — node ids per cell
grid.original_ijk   # (Ncells, 3) int64 — Eclipse-natural (i, j, k)
grid.nnc            # (M, 2) int64 — non-neighbour connections
grid.component      # (Ncells,) int64 — connected-component label
grid.cart_dims      # (NX, NY, NZ)
grid.props          # dict[str, np.ndarray] — from attach_props
grid.n_cells        # int
grid.index_map      # lazy-cached IndexMap
grid.split()        # list[UnstructuredGrid] — one per component
```

## Limitations

- **k-direction faults** raise `NonMatchingZError` (i/j fault dispatch only).
- Where the two pillars of a fault face are skew (twisted), a cell edge
  and the edge it crosses across the fault miss each other; their shared
  intersection node is placed midway between them. Cells still close
  exactly, but their volumes deviate slightly from the trilinear ones
  (Norne: 99th percentile 0.09 %, worst 4.4 %, on thin cells beside
  large throws).
- `mesh3d.face_edges` (∂₂ chain complex) does not support faulted grids.
- No transmissibility, well models, or property algebra.
- EGRID coverage: root grid only — LGR sub-grids and INIT/UNRST property
  pass-through are explicit non-goals.
- Single-process, in-memory.

## License & attribution

Released under the **GNU GPL v3** — see [`LICENSE`](LICENSE).

Portions of the grid-processing algorithms are derived from
[MRST](https://www.sintef.no/projectweb/mrst/) (MATLAB Reservoir Simulation
Toolbox), © SINTEF Digital, also GPL v3. See [`NOTICE`](NOTICE) for details.

## Author

Dr. Artur Castiel Reis de Souza — [arturcastiel.github.io](https://arturcastiel.github.io)
