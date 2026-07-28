# Mesh morphing — moving pillars without breaking the grid

`morph_pillars` moves selected pillars toward prescribed targets while keeping every
cell valid. It knows nothing about faults: it takes *targets* and returns a grid.
The fault-specific part (which pillars a fault claims, and where they should go)
lives in `fault-projections`.

Nothing about the corner-point structure changes. Same pillars, same
`(NX+1, NY+1)` lattice, same topology, `ZCORN` untouched. Only `COORD`'s `(x, y)`
values move.

---

## The idea in one line

Minimise *how far the chosen pillars are from where you want them*, plus
*how much the grid has to deform to get there* — and refuse any answer that breaks
a cell.

```
minimise    fidelity(d)  +  w · smoothness(d)
subject to  no folded, welded or degenerate cells
```

`d` is the displacement of every pillar. The two terms pull against each other and
`w` sets the balance; the constraint is enforced by re-solving at a larger `w`
until the grid is clean.

---

## Why it is posed on pillars

A corner-point grid's pillars form a regular `(NX+1) × (NY+1)` lattice regardless
of how the cells above them are shaped. That makes the pillar lattice the natural
domain for lateral motion:

- neighbours are just `(i±1, j)` and `(i, j±1)`, so no adjacency structure is built;
- the unknown count is `2 · (NX+1)(NY+1)` and is **independent of NZ** — Norne has
  113,344 cells but only 5,311 pillars, so the system is 10,622 unknowns;
- moving a pillar moves its whole column coherently, which is exactly what keeps
  the deck a valid CPG.

The engine reads `COORD` directly and never goes through `projected_topology`.
That is deliberate: plan view collapses nodes by coordinate equality and silently
stops collapsing once pillars tilt — 5 m of tilt on a 100 m cell is enough, and real
decks are overwhelmingly tilted (Norne: 7 vertical pillars out of 5,311).

---

## The controls

### `smoothness` — what "deform as little as possible" means

| value | penalises | use when |
|-------|-----------|----------|
| `"elastic"` *(default)* | linear-elastic strain energy | you care about cell shape |
| `"laplacian"` | the gradient of the displacement | you want the simplest, cheapest model |

This is the single most important setting. The Laplacian treats `x` and `y` as two
independent scalar fields, so it resists *stretching* but is nearly blind to
**shear** — and shear is exactly what a diagonal fault does to an axis-aligned grid.
Linear elasticity couples the components through the `(λ+μ)∇(∇·u)` term, so
distorting a cell's angles costs energy.

Measured on a 12×12 grid with a kinked diagonal fault:

| operator | cells below SJ 0.3 | skew | non-orthogonality p95 |
|----------|-------------------:|-----:|----------------------:|
| Laplacian | 16 | 0.28 | 72.8° |
| elastic | 5 | 0.25 | 65.7° |
| elastic + 2 stiffness iterations | **2** | **0.23** | **54.6°** |

### `poisson` — how much cells resist changing area

`0.0` lets cells change area freely; `0.3` (default) is a good general value;
approaching `0.5` increasingly preserves area, which pushes the deformation into
shear elsewhere. Only meaningful with `smoothness="elastic"`.

### `w0` — fidelity against deformation

Dimensionless, so it transfers between decks: displacements are solved in units of
the mean pillar spacing and each term is divided by its count. Lower `w` follows
the target more closely; higher `w` deforms the grid less.

In practice you rarely need to tune it. If the starting value folds cells, the
solver **bisects `w` upward** until the result is clean and returns the largest
fidelity that stays valid — so the answer is usually insensitive to `w0` over
orders of magnitude. `MorphResult.w` reports what was actually used, and
`MorphResult.history` records every attempt.

### `boundary` and `boundary_scope` — what happens at the edge

| `boundary` | meaning |
|------------|---------|
| `"slide"` *(default)* | rim pillars may move ALONG the boundary but not off it — the outline is preserved as a polygon while pillars redistribute |
| `"pinned"` | rim pillars fixed entirely |
| `"free"` | no boundary constraint; the domain outline may move |

| `boundary_scope` | meaning |
|------------------|---------|
| `"active"` *(default)* | the rim of the ACTIVE domain — the reservoir outline |
| `"lattice"` | the bounding-box edge |

The scope matters more than it looks. On Norne the active domain is 39.6% of the
bounding box, and the two rims share **zero** pillars — the reservoir never touches
its own bounding box.

Choose by where the fault ends:

| fault | setting | typical result |
|-------|---------|----------------|
| terminates inside the domain | `pinned` | ~99% of the distance removed |
| terminates on the domain edge | `slide` | edge pillars slide along the outline to meet it |
| edge, and the outline may move | `free` | best fidelity, but the footprint shifts |

A fault that runs off the edge simply cannot be met by pinned pillars — that is
geometry, not a solver limitation.

### `support_rings` — how far the change is allowed to spread

`None` (default) leaves every pillar free, which gives the best mesh quality
because distortion can spread out. An integer confines the morph to that many
lattice rings around the targeted pillars; everything beyond is pinned and the deck
is byte-identical there.

Counter-intuitively, a *small* radius often scores slightly better on fidelity, not
worse: it denies the solver the room to shear distant cells.

### `stiffness_iterations` and `stiffness_power` — evening out the worst cells

After solving, cells are stiffened in proportion to how distorted they became
(`E ∝ 1/scaled_jacobian ** power`) and the solve is repeated. Already-bad cells turn
rigid, so the next pass pushes deformation into cells that still have shape to give.

Two iterations is usually enough — the measurement above converged after 2 and
changed nothing at 4.

> **Trap.** `stiffness_power` above ~1 tends to stiffen so hard that the re-solve
> folds a cell; the loop then detects this and keeps the previous result, so the
> setting silently does nothing. If raising `power` changes no metric, that is why.
> Prefer more iterations at `power=1` over fewer at `power=2`.

### `normal_only` — point targets or line targets

`True` (default) constrains only the component of the offset along the target's
normal, leaving the pillar free to slide *along* the fault. `False` pins it to an
exact point.

Pinning both components over-constrains the problem: the requirement is that the
pillar lie ON the line, which is one equation, not two. The extra equation removes
the freedom needed to space pillars along the fault, and that shows up as shear.

### `min_fraction` — how close pillars may get

Default `0.10`: no pillar may come within 10% of the original spacing of another.
Checked both for lattice neighbours **and**, via a KD-tree, for any other pillar —
a band of cells can be sheared flat so that pillars two rows apart nearly touch
while every adjacent pair still looks fine.

The hard floor beneath this is `1e-9`, where node deduplication merges two pillars
into one and silently changes the topology.

---

## What "valid" means

`check_validity` refuses a result if any of these hold:

- **folded cell** — the signed area of a column's plan-view quad changes sign, *or*
  any single corner's Jacobian flips. The second test matters: a shoelace area can
  stay positive while one corner turns inside out.
- **welded pillars** — two pillars within `1e-9`, where node dedup merges them.
- **separation violation** — pillars closer than `min_fraction` of the original.
- **depth crossing** — two pillars that stay apart in plan view but intersect
  somewhere between their top and bottom, which only becomes possible once top and
  bottom move independently.

Checks run at **both** the top and bottom planes. A single-plane test is sufficient
only for perfectly vertical pillars, which real decks are not.

---

## Measuring the result

`cell_quality` reports the standard quadrilateral measures:

| metric | good | what it tells you |
|--------|------|-------------------|
| `scaled_jacobian` | → 1 | cell shape; `< 0.3` is usually rejected, `< 0` is inverted |
| `aspect_ratio` | → 1 | stretching |
| `skew` | → 0 | angle distortion — separates shear from stretch |
| `non_orthogonality` | → 0° | **the reservoir one**: a two-point flux scheme assumes the centre-to-centre line is parallel to the face normal, so every degree here is transmissibility error |

If you only watch one number for simulation quality, watch non-orthogonality.

---

## A worked call

```python
from cpg2unstructured import morph_pillars, read_grdecl, write_grid_include
from cpg2unstructured.quality import cell_quality
from cpg2unstructured.pillars import pillar_lattice

grid = read_grdecl("field.grdecl")

result = morph_pillars(
    grid, targets,
    smoothness="elastic",       # penalise shear, not just stretch
    poisson=0.3,
    boundary="slide",           # keep the outline, let rim pillars redistribute
    boundary_scope="active",    # the reservoir rim, not the bounding box
    stiffness_iterations=2,     # even out the worst cells
    support_rings=None,         # let distortion spread for best quality
)

print(result.summary())
print(cell_quality(pillar_lattice(result.grid)).line())

write_grid_include(result.grid, "adapted_grid.grdecl")
```

The output is an INCLUDE file, never a rewritten deck — `read_grdecl` keeps
unparsed keywords by name only, so a read-modify-write would silently drop
`EQUALS`, `COPY`, `MULTFLT`, regions and schedule data. Add one line to the master
deck:

```
INCLUDE
  'adapted_grid.grdecl' /
```

---

## Choosing settings

Start here and only change what the metrics tell you to:

```python
smoothness="elastic", poisson=0.3, boundary="slide",
boundary_scope="active", stiffness_iterations=2
```

| symptom | try |
|---------|-----|
| fault not close enough | lower `w0`; `boundary="free"` if the outline may move |
| cells too distorted | `stiffness_iterations=2..4`; raise `poisson` |
| deck diff too wide | `support_rings=2..5` |
| fault ends on the domain edge and won't meet it | `boundary="slide"` |
| `w` came back large | the constraint is binding — the grid cannot do better without folding |
| raising `stiffness_power` changes nothing | it is folding and being rejected; add iterations instead |
