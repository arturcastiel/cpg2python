# cpg2unstructured — Tutorial

**Main author:** Dr. Artur Castiel Reis de Souza
**Applies to:** cpg2unstructured **v0.2.0**

A formal, worked walkthrough of the library: how to turn an Eclipse
corner-point geometry (CPG) `.grdecl` deck into an unstructured mesh +
cell-cell adjacency graph — including the v0.2.0 capabilities for
**faulted** (non-matching-Z) grids, **PINCH** non-neighbour connections,
and **disconnected-grid** splitting.

> Scope. The library's single goal is *topology*: transform **any** CPG
> grid (including faulted) into an unstructured-grid representation. It is
> deliberately **not** a reservoir-simulation library — no transmissibility,
> no property algebra, no `.EGRID`, no PyPI release.

---

## 1. Install

```bash
git clone <repo> && cd cpg2python
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # numpy only
pip install -e ".[fast]"    # + scipy.sparse for fast Dijkstra
pip install -e ".[dev]"     # + pytest, pytest-cov, build
```

Requires Python ≥ 3.10. NumPy is the only hard dependency.

---

## 2. The pipeline

Three calls take you from a deck to a mesh:

```python
from cpg2unstructured import read_grdecl, from_grdecl, build_topology

g    = read_grdecl("examples/quickstart.grdecl")  # Grdecl   — parsed deck
cube = from_grdecl(g)                             # CornerCube — corner cube + unique nodes
grid = build_topology(cube)                       # UnstructuredGrid — the mesh

print(grid.cart_dims, grid.n_cells)               # e.g. (5, 5, 3)  70
```

| Stage | Call | Produces |
|-------|------|----------|
| Parse | `read_grdecl(path)` | `Grdecl` (SPECGRID/COORD/ZCORN/ACTNUM …) |
| Assemble | `from_grdecl(g, pinch_tol=0.0)` | `CornerCube` (unique `nodes`, index map `P`) |
| Topology | `build_topology(cube)` | `UnstructuredGrid` |

---

## 3. Anatomy of an `UnstructuredGrid`

| Attribute | Shape | Meaning |
|-----------|-------|---------|
| `nodes` | `(Nnodes, 3)` | unique XYZ coordinates |
| `faces.nodes` | `(Nfaces, 4)` | node ids per face (sorted) |
| `faces.neighbours` | `(Nfaces, 2)` | adjacent cell ids; boundary → `-1` in col 1 |
| `faces.direction` | `(Nfaces,)` | `0`=i, `1`=j, `2`=k |
| `cell_faces` | `(Ncells, 6)` | the 6 face ids of each cell |
| `cell_corners` | `(Ncells, 8)` | the 8 node ids of each cell |
| `original_ijk` | `(Ncells, 3)` | Eclipse `(i, j, k)` each cell came from |
| `nnc` | `(M, 2)` | non-neighbour connections (see §5) |
| `component` | `(Ncells,)` | connected-component label (see §6) |
| `props` | dict | cell properties (see §7) |
| `n_cells` | property | active cell count |
| `index_map` | property | bidirectional active-id ↔ `(i, j, k)` |

---

## 4. Faulted (non-matching-Z) grids — v0.2.0

A *fault* is a pillar pair whose ZCORN values do not match across a face
(the layers are offset). As of v0.2.0, `build_topology` dispatches matching
vs faulted i- and j-faces **inline** — no extra calls, faulted decks just
flow through:

```python
g    = read_grdecl("tests/fixtures/simple_faulted_3x3x2.grdecl")
grid = build_topology(from_grdecl(g))
# faulted i-faces are built; a cell on one side of the fault may
# connect to TWO cells on the other (offset → straddling overlap).
```

Where a cell straddles two cells across the fault, you get the expected
extra faces — e.g. a +0.6 offset on unit cells yields **3** connections
per faulted pillar stack, not 2.

> **Limitation.** k-direction faults still raise `NonMatchingZError`
> (the 2-pillar sweep does not apply to the 4-pillar k-face geometry).

---

## 5. Pinched-out cells and NNCs — v0.2.0

A *pinched-out* cell has zero vertical thickness (its top and bottom
collapse). `from_grdecl` drops such cells from the active set, and
`build_topology` reconnects the survivors above and below with a
**non-neighbour connection** (they are physically adjacent once the
collapsed layer vanishes):

```python
g    = read_grdecl("tests/fixtures/pinched_1x1x3.grdecl")  # middle layer collapsed
grid = build_topology(from_grdecl(g))

grid.n_cells        # 2   (pinched middle cell dropped)
grid.nnc            # array([[0, 1]])  → the two survivors are NNC-bridged
```

- `from_grdecl(g, pinch_tol=t)` — a cell is pinched when its thickness is
  ≤ `t` on all four pillars. Default `0.0` catches only exactly-collapsed
  cells, so ordinary decks are untouched.
- A genuine `ACTNUM=0` gap is **not** an NNC — only collapsed (pinched)
  cells are bridged.

---

## 6. Disconnected grids — `component` & `split()` — v0.2.0

A deck can yield several independent grid "islands". Every cell carries a
`component` label, and `split()` materialises the islands as independent
grids:

```python
g    = read_grdecl("tests/fixtures/islands_5x1x1.grdecl")  # two blocks, middle off
grid = build_topology(from_grdecl(g))

grid.component                 # e.g. array([0, 0, 1, 1])  → 2 components
parts = grid.split()           # list[UnstructuredGrid], one per component
[p.n_cells for p in parts]     # [2, 2]
```

`split()` returns `[self]` unchanged for a single-component grid. Node ids
stay global (the `nodes` array is shared); cells, faces, `nnc` and `props`
are re-indexed per sub-grid.

---

## 7. Cell properties

```python
from cpg2unstructured import attach_props

grid = attach_props(build_topology(from_grdecl(g)), g)
grid["PORO"].mean()            # property arrays are keyed by Eclipse keyword
```

`attach_props` returns a **new** grid with the `props` dict populated on
active cells. Use `grid["PORO"]` (not `grid.props["PORO"]`) for a helpful
"available keys" error on a miss.

---

## 8. Geometry & graph

```python
from cpg2unstructured import (
    face_geometry, cell_geometry,        # centroids, areas, normals, volumes
    cell_adjacency, shortest_path,       # numpy graph
)

adj         = cell_adjacency(grid, weights="unit")
dist, path  = shortest_path(adj, 0, grid.n_cells - 1)
```

With the `[fast]` extra, `cell_adjacency_csgraph` + `shortest_path_fast`
use `scipy.sparse` for large grids.

---

## 9. Visual debugging

Rendering lives in a separate side folder (personal tooling, never
packaged): `/mnt/c/projects/viz-cpg2unstructured`. Headless PNG export:

```bash
bash /mnt/c/projects/viz-cpg2unstructured/export-pngs.sh tests/fixtures/pinched_1x1x3.grdecl
```

The side-by-side render draws raw CPG vs the unstructured grid, the
**NNC bridges** (dashed magenta), **per-component colours**, and any
**dropped cells** (red dashed).

---

## 10. Out of scope

- Properties algebra / transmissibility, `.EGRID`, MAPAXES, PyPI publish.
- k-direction fault dispatch (i/j only).

---

*cpg2unstructured is authored by Dr. Artur Castiel Reis de Souza —
arturcastiel.github.io.*
