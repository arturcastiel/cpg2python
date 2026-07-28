# The barrier polish

`morph_pillars(..., polish="barrier")` adds a second stage that fixes the cells the
quadratic solve damages. It is **off by default** — it changes results, so it is
opt-in until measured on the decks that matter.

```python
result = morph_pillars(grid, targets, w0=0.05, boundary="slide", polish="barrier")
print(result.polish.summary())
# polish 4.71 -> 0.0957 (98.0% of the distortion removed) · 61 iteration(s) · converged
```

---

## Why the first stage needs a second one

`morph_pillars` minimises

```
    || F d - r ||²  +  w · S(d)
```

with `S` quadratic — the Dirichlet energy or linear-elastic strain energy. Two
things follow, and both are measurable:

**A quadratic energy is finite for an inverted cell.** Nothing in the objective
resists a fold. The solver has no idea it is destroying a cell; it finds out from a
checker afterwards.

**The repair is global, the damage is local.** When a cell does fold, `w` is bisected
upward — stiffening the whole model to fix one place, and paying fidelity everywhere
for it.

There is a third problem, which is what actually motivated this. On the benchmark's
`zigzag` case the elastic operator returns a grid that `check_validity` calls
**valid** — no folds, no welds, no separation violations — containing a cell at
scaled Jacobian **5.2e-04** and 14 cells below 0.3. A cell can be flattened to
near-zero area without any corner crossing. Every check tests crossing. Degenerate is
not inverted, so nothing saw it.

---

## What the polish does instead

It minimises a **barrier** distortion energy, starting from the quadratic solution:

```
    E(P) = fidelity · mean‖P_claimed - target‖² / h²  +  mean μ(T)
```

`μ` is TMOP metric 002, `0.5 |T|² / det(T) − 1`, with `T = J W⁻¹`. It diverges as
`det(T) → 0⁺`, so a backtracking line search **cannot** step into an inverted or
degenerate configuration. Validity stops being a post-hoc check and becomes a
property of the objective.

### The target matrix is the real idea

`W` is not the identity. It is **the cell's own original Jacobian**, so the energy
says *"keep the shape you had"* rather than *"become square"*.

For a corner-point grid this is the only defensible target. A graded or sheared deck
is not a bad deck, and an objective pulling it toward squares would damage it in the
name of quality. `shape_targets` builds `W` from the original lattice; a cell that
has not changed scores exactly zero, whatever shape it is.

```python
from cpg2unstructured import cell_quads, shape_targets, distortion_energy, DistortionTerms

q = cell_quads(nx1, ny1)
winv = shape_targets(original_xy.reshape(-1, 2), q)      # keep-your-shape targets
E, grad = distortion_energy(points, q, winv, DistortionTerms(shape=1.0))
E, grad = distortion_energy(points, q, None, DistortionTerms(shape=1.0))  # be-square
```

---

## Measured

12×12 grid, `slide` boundary, `w0=0.05`, `polish_fidelity=4`:

| case | polish | distance to fault | SJ min | cells < 0.3 | ortho p95 | ortho max |
|---|---|---:|---:|---:|---:|---:|
| kink_sharp | none | **4.89 m** | 0.327 | 0 | 26.0° | 49.3° |
| | barrier | 6.64 m | **0.492** | 0 | **25.1°** | **42.0°** |
| zigzag | none | **5.58 m** | 0.001 | **14** | 28.4° | 65.1° |
| | barrier | 7.22 m | **0.485** | **0** | **24.3°** | **44.1°** |
| arc_tight | none | **2.62 m** | 0.001 | **8** | 26.8° | 54.6° |
| | barrier | 6.08 m | **0.389** | **0** | **18.9°** | **39.9°** |

Every degenerate cell removed, on every case. The cost is 1.7–3.5 m of fidelity on a
100 m grid. `polish_fidelity` is the dial: **larger is closer** — the opposite sense
to `w0`, because here it is the fault term being weighted.

---

## Controls

| parameter | default | what it does |
|---|---|---|
| `polish` | `"none"` | `"barrier"` turns the stage on |
| `polish_fidelity` | `4.0` | how hard the pillars are held on the fault. **Larger is closer** |
| `polish_size` | `0.0` | add TMOP metric 056, `0.5(det T + 1/det T) − 1`, a barrier against area collapse. Worth ~0.04 of SJ min and ~1.5° of ortho p95 for ~1 m of fidelity |
| `polish_shape_size` | `0.0` | add metric 007, `\|T − T⁻ᵀ\|²`, the symmetric Dirichlet energy |
| `polish_iterations` | `200` | L-BFGS iteration cap |

The polish honours the **same** boundary mode and `support_rings` as the solve — it
reuses the engine's Dirichlet elimination, so a pillar the solve was not allowed to
move is not moved here either. A rigid (stage-1) morph stays rigid: the polish is
applied to the top plane and copied to the bottom, so pillars never acquire a tilt
the fault did not ask for.

---

## It cannot make things worse

Three guards, each of which returns the input untouched with `applied=False` and a
reason in `report.message`:

1. the starting lattice is already inverted, so the barrier is infinite everywhere
   and there is nothing to descend;
2. L-BFGS finds no descent, or stops above the starting energy;
3. the polished grid fails the **same** validity checks the solve had to pass.

A stage that can silently degrade a deck is not a polish.

---

## The degenerate-cell check

Independent of the polish, `check_validity` now takes `min_scaled_jacobian`:

```python
report = check_validity(candidate, original, min_scaled_jacobian=0.3)
report.n_degenerate     # cells below the threshold
report.degenerate       # (NX, NY) bool
```

**Off by default**, because switching it on changes what "valid" means for decks that
pass today. `0.3` is the usual reject level; `0.05` is a conservative floor that only
catches cells that are genuinely collapsing.

---

## Cost

Dense L-BFGS over the free DOFs, warm-started, typically 40–90 iterations on a 12×12
grid. The Hessian is as sparse as the lattice, so an inexact Newton–CG with a
block-Jacobi preconditioner is the obvious next step — worth doing when the polish is
proven on a field-scale deck and its cost there is measured, not before.

---

## The orthogonality term

`polish_ortho` adds an explicit orthogonality penalty — the one thing no other
operator here targets directly, and the metric a two-point flux scheme actually
pays for.

```
    mu_ortho = (u . v)^2 / (|u|^2 |v|^2),    u, v = the columns of T
```

Zero exactly when the two columns of `T` are perpendicular — that is, when the cell
corner **kept its original angle**. It is measured on `T`, not on `J`, and that
distinction is the whole of it: a sheared or graded deck has cells that are not
square to begin with, and an absolute penalty would pull them square, "improving"
the mesh by destroying the geometry it was built with. Measured on the sheared
benchmark grid: identical quality either way, but the relative form holds the fault
**2.2 m closer** because it is not spending the budget fighting the shear.

12×12, kinked trace, `slide`, `w0=0.05`:

| `polish_ortho` | distance | SJ min | ortho p95 | ortho max |
|---:|---:|---:|---:|---:|
| 0 (default) | **6.64 m** | 0.492 | 25.1° | 42.0° |
| 0.5 | 7.25 m | 0.535 | 22.7° | 40.2° |
| 2 | 8.87 m | 0.665 | 18.9° | 32.9° |
| 8 | 12.28 m | **0.862** | **12.4°** | **15.9°** |

A clean monotone dial: at 8 it **halves the non-orthogonality maximum** and nearly
doubles the worst scaled Jacobian, for roughly double the residual. At 0.5 it is
nearly free.

It is **not** a barrier — it says nothing about inversion — so it is only ever used
alongside `shape`, which is.

### Where it came from, and what did not transfer

Ryskin and Leal's covariant Laplace system, and Zhang, Jia & Wang's control of its
distortion function (*J. Comput. Phys.* 218:549–571, 2006), make orthogonality the
**objective** rather than a by-product, expressed as a directional weight ratio
`f = h_η/h_ξ`. Their discretisation is a 5-point stencil with one weight per edge —
structurally identical to `laplacian_matrix`, which already carries per-edge
weights. So the operator was transplanted directly and measured.

It did not help. At **equal fidelity** the RL-weighted operator is slightly worse
than the shipped ones on three of four benchmark cases, and once the barrier polish
runs the difference vanishes entirely — the polish dominates whatever smoothness
produced its starting point. Their specific contribution, the `r` blend between
local and line-averaged scale factors, made almost no difference here (`r`=0, 1 and
auto agree to three figures) because RL is acting as one penalty among several
rather than as *the* generator with prescribed boundaries, which is the role the
paper's analysis assumes.

The **idea** transferred; the equation did not.
