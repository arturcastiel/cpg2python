# Grid input — text GRDECL vs binary EGRID

`cpg2unstructured` reads two corner-point grid formats. Both land in the
same `CornerPointGrid` pivot dataclass, so the rest of the pipeline
(`from_grdecl`, `CornerCube`, `build_mesh_graph_3d`, projection
workflows, ...) doesn't know — or need to know — which reader produced
the grid.

```
.grdecl ─┐
         ├─► CornerPointGrid ──► from_grdecl ──► CornerCube ──► ...
.EGRID  ─┘
```

## API summary

| Function | Returns | Extras |
|----------|---------|--------|
| `read_grdecl(path)`                            | `CornerPointGrid` (source=`"grdecl"`) | `INCLUDE` recursion; PORO / PERMX / SATNUM in `.props` |
| `read_egrid(path, *, apply_mapaxes=True, dtype=np.float32)` | `CornerPointGrid` (source=`"egrid"`) | streaming reader; MAPAXES rotation by default |
| `from_grdecl(grid, pinch_tol=0.0)`             | `CornerCube`                          | accepts either source unchanged |

The legacy name `Grdecl` is a `TypeAlias` of `CornerPointGrid` —
every existing `from cpg2unstructured import Grdecl` keeps working.

## When to use which

- **Text GRDECL** — hand-authored decks, small fixtures, anything you
  want to read in a diff. `INCLUDE` lets you split a model across
  files; properties (PORO / PERMX / SATNUM / ...) pass through into
  `grid.props`.
- **Binary EGRID** — outputs of OPM Flow, Eclipse, Intersect; real
  field grids that ship in UTM (so MAPAXES matters); anything large
  enough that text parsing time is noticeable. The reader streams
  via `resfo.lazy_read` and stops at `ENDGRID`, so LGR / NNC blocks
  appended after the root grid are never materialised.

## MAPAXES — applied by default

Real fields ship EGRID with a `MAPAXES` record because the local grid
frame is rotated relative to the project map (typically UTM). With
`apply_mapaxes=True` (the default) `read_egrid` rotates the pillar
(x, y) coordinates into the map frame, so a fault polyline from a
shapefile in the same UTM zone lines up directly with grid corners.

```python
g = read_egrid("MODEL.EGRID")             # rotated → MAP coords (UTM)
g.mapaxes                                  # array([y_end_x, ..., x_end_y])
g.mapaxes_applied                          # True
```

Opt out only if you specifically want raw LOCAL coords. The reader
emits a `RuntimeWarning` when MAPAXES is present and skipped, so
silent-bad-output is impossible:

```python
g = read_egrid("MODEL.EGRID", apply_mapaxes=False)
# RuntimeWarning: ... MAPAXES present but apply_mapaxes=False ...
g.mapaxes_applied                          # False
```

The 6-number MAPAXES record stays available regardless of whether
the rotation was applied, so downstream code can implement its own
transform if needed.

## dtype — native float32 by default

EGRID stores `COORD` / `ZCORN` as float32 on disk. `read_egrid` keeps
that dtype to avoid the 2× memory cost of an implicit promotion to
float64 on real (multi-million-cell) grids. Widen explicitly when
downstream needs it:

```python
g = read_egrid("MODEL.EGRID")                       # f32
g = read_egrid("MODEL.EGRID", dtype=np.float64)     # f64
```

The text-deck `read_grdecl` still returns float64 (text decks are
typically small and the dtype change there would be a real
behavioural break).

## Streaming — what the reader skips

`read_egrid` materialises numpy arrays for exactly these keywords:

    GRIDHEAD   COORD   ZCORN   ACTNUM   MAPAXES

and stops at the first `ENDGRID` keyword. Everything that follows
`ENDGRID` in a typical EGRID — LGR sub-grids (`LGR`, `LGRSGONE`,
`COORDS`, `ZCORNS`, `ACTNUMSG`, ...), non-neighbour connections
(`NNC1`, `NNC2`, `NNCG`), additional metadata — is **not** read.

Concretely: on a Volve-scale grid (~9M cells) where the root section
is ~600 MB and LGR sections add another ~300 MB, `read_egrid` allocates
only the root-section arrays.

LGR sub-grids are out of scope for this reader — they need their own
pivot type because their `cart_dims` differs from the host. If you
need them, file an issue.

## Errors

| Error class | When |
|-------------|------|
| `GrdeclSyntaxError`      | `GRIDHEAD` / `COORD` / `ZCORN` missing |
| `GrdeclConsistencyError` | grid type ≠ corner-point (`GRIDHEAD[0] != 1`); array sizes inconsistent with declared dimensions; MAPAXES has a degenerate axis |

Both inherit from `Cpg2UError`.

## Dependencies

`resfo` (LGPL-3.0, Equinor) is a hard dependency. It's pure Python on
top of `numpy`, so the install footprint is one `pip` line and no
compiled wheels.
