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
> transmissibility, no property algebra, no `.EGRID`).

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

## API

| Function / class | Purpose |
|---|---|
| `read_grdecl(path) → Grdecl` | parse a `.grdecl` deck |
| `from_grdecl(grdecl, pinch_tol=0.0) → CornerCube` | corner-cube assembly + node dedup; drops pinched cells |
| `build_topology(cube) → UnstructuredGrid` | full mesh: i/j/k faces, cell-face table, NNCs, components |
| `attach_props(grid, grdecl) → UnstructuredGrid` | attach cell properties (PORO, PERM, …) |
| `face_geometry(...)` / `cell_geometry(...)` | centroids, areas, normals, volumes |
| `IndexMap(cart_dims, ijk)` | bidirectional active-id ↔ `(i, j, k)` |
| `cell_adjacency(grid, weights=...)` | adjacency list (NumPy) |
| `cell_adjacency_csgraph(grid, ...)` | adjacency CSR (`scipy.sparse`) |
| `shortest_path(adj, src, dst)` | Dijkstra (NumPy + heapq) |
| `shortest_path_fast(csr, src, dst=None)` | Dijkstra (`scipy.sparse.csgraph`) |

### `UnstructuredGrid` at a glance

```python
grid.nodes          # (Nnodes, 3) float64 — unique XYZ
grid.faces          # Faces: nodes (N,4), neighbours (N,2), direction (N,)
grid.cell_faces     # (Ncells, 6) int64 — face ids: [i-, i+, j-, j+, k-, k+]
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
- No transmissibility, well models, or property algebra.
- ASCII GRDECL only — no binary `.EGRID`; no coordinate transforms / MAPAXES.
- Single-process, in-memory.

## License & attribution

Released under the **GNU GPL v3** — see [`LICENSE`](LICENSE).

Portions of the grid-processing algorithms are derived from
[MRST](https://www.sintef.no/projectweb/mrst/) (MATLAB Reservoir Simulation
Toolbox), © SINTEF Digital, also GPL v3. See [`NOTICE`](NOTICE) for details.

## Author

Dr. Artur Castiel Reis de Souza — [arturcastiel.github.io](https://arturcastiel.github.io)
