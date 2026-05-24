# cpg2unstructured

Lightweight Python library: read Eclipse CPG (corner-point geometry)
`.grdecl` files → unstructured mesh + cell-cell adjacency graph for
shortest-path algorithms. NumPy-only by default; optional `scipy.sparse`
backend for fast Dijkstra.

**Status:** v0.2.0 — alpha. Handles faulted (non-matching-Z) grids,
PINCH non-neighbour connections (`UnstructuredGrid.nnc`), and
disconnected-grid splitting (`UnstructuredGrid.split()`).
**Distribution:** local editable install only — not on PyPI.
**Main author:** Dr. Artur Castiel Reis de Souza · arturcastiel.github.io

**Tutorial:** a formal end-to-end walkthrough (incl. faults, PINCH NNCs,
and disconnected-grid splitting) lives in [`docs/TUTORIAL.md`](docs/TUTORIAL.md).

## Install

```bash
git clone <repo>
cd cpg2python
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # numpy-only
pip install -e .[fast]      # + scipy.sparse for fast Dijkstra
pip install -e .[dev]       # + pytest, pytest-cov, build
```

## 30-second quickstart

```python
from cpg2unstructured import (
    read_grdecl, from_grdecl, build_topology, attach_props,
    cell_adjacency, shortest_path,
)

g = read_grdecl("examples/quickstart.grdecl")
grid = attach_props(build_topology(from_grdecl(g)), g)

print(grid.cart_dims, grid.n_cells)   # (5, 5, 3)  70 active cells
print(grid["PORO"].mean())            # ~0.20

adj = cell_adjacency(grid, weights="unit")
dist, path = shortest_path(adj, 0, grid.n_cells - 1)
print(f"shortest path: {len(path)} cells, distance {dist}")
```

Run the full demo end-to-end:

```bash
python examples/quickstart.py
```

## API map

| Function / class                                | Purpose                                       | PR    |
|-------------------------------------------------|-----------------------------------------------|-------|
| `read_grdecl(path) → Grdecl`                    | parse `.grdecl` deck                          | PR-3, PR-4 |
| `from_grdecl(grdecl) → CornerCube`              | corner-cube + node dedup                      | PR-5  |
| `build_topology(cube) → UnstructuredGrid`       | full mesh (i + j + k faces, cell-face table)  | PR-7  |
| `attach_props(grid, grdecl) → UnstructuredGrid` | properties on active cells                    | PR-12 |
| `face_geometry(...)`                            | face centroids, areas, normals                | PR-8  |
| `cell_geometry(...)`                            | cell centroids, volumes (divergence theorem)  | PR-8  |
| `IndexMap(cart_dims, ijk)`                      | bi-directional active-id ↔ (i, j, k)          | PR-9  |
| `cell_adjacency(grid, weights=...)`             | adjacency list (numpy, list-of-list)          | PR-10 |
| `cell_adjacency_csgraph(grid, ...)`             | adjacency CSR (`scipy.sparse.csr_array`)      | PR-11 |
| `shortest_path(adj, src, dst)`                  | Dijkstra (numpy + heapq)                      | PR-10 |
| `shortest_path_fast(csr, src, dst=None)`        | Dijkstra (`scipy.sparse.csgraph.dijkstra`)    | PR-11 |

### `UnstructuredGrid` at a glance

```python
grid.nodes              # (Nnodes, 3) float64
grid.faces              # Faces: nodes, neighbours, direction
grid.cell_faces         # (Ncells, 6) int64 — slots [i-, i+, j-, j+, k-, k+]
grid.cell_corners       # (Ncells, 8) int64 — node ids per cell
grid.original_ijk       # (Ncells, 3) int64 — Eclipse-natural (i, j, k)
grid.cart_dims          # tuple (NX, NY, NZ)
grid.props              # dict[str, np.ndarray] — populated by attach_props
grid.n_cells            # int — same as grid.cell_faces.shape[0]
grid.index_map          # lazy-cached IndexMap
grid["PORO"]            # accessor with helpful "available keys" KeyError
"PORO" in grid          # bool
```

## Limitations (v0.1.0)

- Matching-Z grids only. Faulted (non-matching-Z) decks raise
  `NonMatchingZError`.
- No coordinate transforms or projections.
- No transmissibility or well models.
- ASCII GRDECL only (Eclipse spec). No binary `.EGRID`.
- Single-process, in-memory; no streaming for huge decks.

## Author

Dr. Artur Castiel Reis de Souza — [arturcastiel.github.io](https://arturcastiel.github.io)

## License

MIT — see [LICENSE](LICENSE).
