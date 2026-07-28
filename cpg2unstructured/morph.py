"""Least-squares lateral morphing of the pillar lattice.

Drives selected pillars toward prescribed targets while keeping the mesh valid.
Fault-agnostic by design: it knows nothing about faults, only about pillars,
targets and validity, so the same engine serves well paths or boundary
conformance.

The problem
-----------
Minimise, over pillar displacements ``d``::

    (1/N_f) Σ_(p,s) ω_ps ‖d_p − t_ps‖²   +   w · (1/N_p) Σ_j ‖L(d)_j‖²
    └────────── fidelity ──────────┘         └──── smoothness ────┘

**On the smoothness term.** What we want to preserve is the grid's *shape*, not
to flatten it — so the penalty is on the CHANGE in differential coordinates,
``L(x) − L(x⁰)``. Because ``L`` is linear and ``x = x⁰ + d``, that is identically
``L(d)``. Penalising ``L(x)`` instead would drive the grid toward a uniform
harmonic layout and destroy deliberate grading wherever it acted; penalising
``L(d)`` leaves ``L(x⁰)`` untouched by construction.

**On the large-w limit.** With boundary DOFs eliminated (``pinned`` or ``slide``),
``L(d) = 0`` under zero boundary data forces ``d = 0``, so ``w → ∞`` reproduces the
input grid exactly — which is what gives bisection a provably valid bracket. Under
``boundary="free"`` the limit is instead the best *affine* fit to the targets:
still fold-free for any non-degenerate linear part, but not the identity. That
distinction is why ``free`` is not the default.

Scaling
-------
Displacements are solved in units of the local pillar spacing (``d / h_local``),
and each sum is divided by its term count. Both matter: fidelity residuals are
invariant under grid refinement while ``‖L(d)‖ ≈ h²‖∇²d‖`` shrinks as ``h²``, so an
unnormalised ``w`` silently smooths less on finer grids; and dividing by term
counts stops a long fault outweighing a short one purely by having more targets.
``w`` is then a dimensionless ratio of mean residuals that transfers between decks.

x and y decouple in the fidelity and smoothness terms alike, but ``slide`` couples
them through its constraint direction, so the full ``2N`` system is assembled once
rather than two scalar systems — at these sizes (Norne: 10,622 unknowns, and the
count is independent of NZ) that costs nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from cpg2unstructured.corner_point import CornerPointGrid
from cpg2unstructured.distortion import DistortionTerms
from cpg2unstructured.pillars import (
    PillarLattice,
    apply_displacement,
    boundary_mask,
    local_spacing,
    pillar_lattice,
)
from cpg2unstructured.polish import PolishReport
from cpg2unstructured.validity import ValidityReport, check_validity

POLISH_MODES = ("none", "barrier")

__all__ = ["PillarTargets", "MorphResult", "morph_pillars", "laplacian_matrix",
           "elastic_stiffness", "stiffness_from_quality", "stiffness_from_band",
           "support_mask"]

BOUNDARY_MODES = ("slide", "pinned", "free")


@dataclass(frozen=True)
class PillarTargets:
    """Where selected pillars should go.

    One row per ``(pillar, source)`` incidence — deliberately NOT one row per
    pillar. A pillar claimed by two sources (a polyline kink, two facets of a
    curved surface) carries two residuals, and the solve places it at the point
    minimising the sum. For two non-parallel lines that point is their
    intersection, so a kink vertex falls out of the arithmetic instead of needing
    an ownership rule.

    Attributes
    ----------
    ij
        ``(M, 2)`` int — lattice indices of the targeted pillar.
    xy_top
        ``(M, 2)`` float — target position for the pillar's TOP endpoint.
    xy_bot
        ``(M, 2)`` float or ``None``. ``None`` means rigid translation: the bottom
        endpoint takes the same displacement as the top (stage 1). Supplying it
        tilts the pillar (stage 2).
    weight
        ``(M,)`` float — per-row fidelity weight. Used by rim tapering; defaults
        to 1.
    """

    ij: np.ndarray
    xy_top: np.ndarray
    xy_bot: np.ndarray | None = None
    weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        m = self.ij.shape[0]
        if self.ij.ndim != 2 or self.ij.shape[1] != 2:
            raise ValueError(f"ij must be (M, 2); got {self.ij.shape}")
        if self.xy_top.shape != (m, 2):
            raise ValueError(f"xy_top must be {(m, 2)}; got {self.xy_top.shape}")
        if self.xy_bot is not None and self.xy_bot.shape != (m, 2):
            raise ValueError(f"xy_bot must be {(m, 2)}; got {self.xy_bot.shape}")
        if self.weight is not None and self.weight.shape != (m,):
            raise ValueError(f"weight must be {(m,)}; got {self.weight.shape}")

    @property
    def n(self) -> int:
        return self.ij.shape[0]

    @property
    def is_rigid(self) -> bool:
        """True when top and bottom share one displacement (stage 1)."""
        return self.xy_bot is None

    def weights(self) -> np.ndarray:
        return np.ones(self.n) if self.weight is None else np.asarray(self.weight, float)


@dataclass
class MorphResult:
    """Outcome of a morph, including everything needed to judge it."""

    grid: CornerPointGrid
    d_top: np.ndarray                     # (NX+1, NY+1, 2) applied displacement
    d_bot: np.ndarray
    w: float                              # smoothness weight actually used
    validity: ValidityReport
    residuals: np.ndarray                 # (M,) distance from each target after the solve
    iterations: int = 0
    bisection_steps: int = 0
    history: list = field(default_factory=list)
    polish: PolishReport | None = None

    @property
    def ok(self) -> bool:
        return self.validity.ok

    @property
    def n_moved(self) -> int:
        return int((np.linalg.norm(self.d_top, axis=-1) > 0).sum())

    def summary(self) -> str:
        r = self.residuals
        if r.size:
            stats = (
                f"residual max {r.max():.4g} · mean {r.mean():.4g} · "
                f"median {np.median(r):.4g}"
            )
        else:
            stats = "no targets"
        tail = f" · {self.polish.summary()}" if self.polish is not None else ""
        return (
            f"w={self.w:.4g} · {self.bisection_steps} bisection step(s) · "
            f"{self.n_moved} pillar(s) moved · {stats} · {self.validity.summary()}"
            + tail
        )


def laplacian_matrix(nx1: int, ny1: int, xy: np.ndarray | None = None) -> sp.csr_array:
    """5-point Laplacian over an ``(nx1, ny1)`` lattice, as ``L = D - A``.

    Row ``j`` gives ``Σ_k c_jk (d_j − d_k)``. Edge and corner nodes simply have
    fewer neighbours — no ghost padding — which keeps ``L(constant) = 0``
    everywhere, rim included.

    ``xy`` (``(nx1, ny1, 2)`` pillar positions) switches on INVERSE-DISTANCE
    weights ``c_jk = h̄ / |x_j − x_k|``. This matters on any grid that is not
    uniform: with unit weights the operator penalises the raw difference between
    neighbouring displacements regardless of how far apart those pillars are, so a
    graded lattice is smoothed inconsistently. Measured on Norne (spacing varying
    6.4×), the unweighted form drove pillars up to 577 m — on ~100 m cells — while
    reducing the distance to the fault by only 13.5%.

    The ``h̄`` factor keeps ``c_jk`` dimensionless and makes the weighted operator
    reduce exactly to the unweighted one on a uniform lattice.
    """
    n = nx1 * ny1
    idx = np.arange(n).reshape(nx1, ny1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    flat = None if xy is None else xy.reshape(-1, 2)
    for a, b in (
        (idx[:-1, :], idx[1:, :]),
        (idx[:, :-1], idx[:, 1:]),
    ):
        ra, rb = a.ravel(), b.ravel()
        if flat is None:
            w = np.ones(ra.size)
        else:
            w = np.linalg.norm(flat[rb] - flat[ra], axis=1)
        rows += [ra, rb]
        cols += [rb, ra]
        vals += [w, w]
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    if flat is not None:
        v = np.mean(v) / np.maximum(v, 1e-12)      # inverse distance, scaled by h̄
    adj = sp.csr_array((v, (r, c)), shape=(n, n))
    deg = sp.diags_array(np.asarray(adj.sum(axis=1)).ravel())
    return (deg - adj).tocsr()


def elastic_stiffness(
    xy: np.ndarray,
    *,
    poisson: float = 0.3,
    stiffness: np.ndarray | None = None,
) -> sp.csr_array:
    """Plane-stress linear-elasticity stiffness over the pillar lattice (Q1 elements).

    The Laplacian smoothness term penalises the GRADIENT of the displacement, which
    resists stretching but is nearly blind to SHEAR — the x and y components never
    talk to each other. That is why a diagonal fault through an axis-aligned grid
    produced skewed cells and 70-85 degree non-orthogonality no matter how the
    weight was tuned.

    Linear elasticity couples the components through the ``(λ+μ)∇(∇·u)`` term, so
    shearing a cell costs energy directly. The grid deforms like a sheet of rubber
    rather than like two independent scalar fields, which is the behaviour wanted
    when a fault drags one part of the mesh past another.

    Nothing about the CPG structure changes: same pillars, same topology, same
    ``(NX+1, NY+1)`` lattice — only the quadratic form being minimised.

    Parameters
    ----------
    xy
        ``(NX+1, NY+1, 2)`` pillar positions; element geometry comes from these.
    poisson
        Poisson's ratio. Higher values resist area change more strongly; 0.3 is a
        reasonable default for morphing, 0.0 leaves cells free to change area.
    stiffness
        Optional ``(NX, NY)`` per-cell Young's modulus. Raising it on cells that are
        already distorted pushes deformation into cells that can absorb it — the
        iterated-stiffness idea — without touching the topology.
    """
    nx1, ny1, _ = xy.shape
    ncx, ncy = nx1 - 1, ny1 - 1
    n = nx1 * ny1
    idx = np.arange(n).reshape(nx1, ny1)

    # element corner node ids, counter-clockwise
    e = np.stack([
        idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(),
        idx[1:, 1:].ravel(), idx[:-1, 1:].ravel(),
    ], axis=1)                                            # (Ne, 4)
    xe = np.stack([
        xy[:-1, :-1], xy[1:, :-1], xy[1:, 1:], xy[:-1, 1:],
    ], axis=2).reshape(-1, 4, 2)                          # (Ne, 4, 2)

    E = (np.ones(ncx * ncy) if stiffness is None
         else np.asarray(stiffness, float).ravel())
    nu = float(poisson)
    # plane stress constitutive matrix, per unit E
    D0 = np.array([[1.0, nu, 0.0],
                   [nu, 1.0, 0.0],
                   [0.0, 0.0, (1.0 - nu) / 2.0]]) / (1.0 - nu * nu)

    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    Ke = np.zeros((xe.shape[0], 8, 8))
    for xi, eta in gauss:
        dN = np.array([
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
        ]) / 4.0                                          # (2, 4)
        J = np.einsum("ij,njk->nik", dN, xe)              # (Ne, 2, 2)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        detJ = np.where(np.abs(detJ) < 1e-300, 1e-300, detJ)
        Jinv = np.empty_like(J)
        Jinv[:, 0, 0] = J[:, 1, 1] / detJ
        Jinv[:, 1, 1] = J[:, 0, 0] / detJ
        Jinv[:, 0, 1] = -J[:, 0, 1] / detJ
        Jinv[:, 1, 0] = -J[:, 1, 0] / detJ
        dNxy = np.einsum("nij,jk->nik", Jinv, dN)         # (Ne, 2, 4)

        B = np.zeros((xe.shape[0], 3, 8))
        B[:, 0, 0::2] = dNxy[:, 0, :]
        B[:, 1, 1::2] = dNxy[:, 1, :]
        B[:, 2, 0::2] = dNxy[:, 1, :]
        B[:, 2, 1::2] = dNxy[:, 0, :]
        Ke += (E * np.abs(detJ))[:, None, None] * np.einsum(
            "nki,kl,nlj->nij", B, D0, B)

    # Normalise so `w` means the same thing for both operators. Element stiffness
    # carries a factor |detJ| — cell AREA, ~1e4 for 100 m cells — while the
    # Laplacian's entries are O(1). Without this the same w over-smooths the
    # elastic solve by orders of magnitude and fidelity collapses.
    scale = float(np.mean(np.abs(Ke.diagonal(axis1=1, axis2=2))))
    if scale > 0:
        Ke /= scale

    dofs = np.empty((e.shape[0], 8), dtype=np.int64)
    dofs[:, 0::2] = e * 2
    dofs[:, 1::2] = e * 2 + 1
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    return sp.csr_array((Ke.ravel(), (rows, cols)), shape=(2 * n, 2 * n))


def stiffness_from_band(
    shape: tuple[int, int],
    ij: np.ndarray,
    *,
    rings: int = 1,
    softness: float = 0.02,
) -> np.ndarray:
    """``(NX, NY)`` stiffness that is SOFT at the fault and rigid away from it.

    The sacrificial-band idea: rather than spreading distortion thinly over a wide
    region — which leaves many cells mediocre — concentrate it deliberately in a
    narrow band of cells along the fault, and let everything outside stay close to
    its original shape.

    Those band cells are expected to end up small and skewed. That is the trade:
    they are sacrificed so their neighbours do not have to be. Quality is then
    judged OUTSIDE the band, where it matters for the flow solution, and the band
    itself is held only to the hard validity constraints — no folds, no welds.

    ``softness`` is the band's modulus relative to the surrounding grid; 0.02 makes
    the band 50x more compliant. ``rings`` sets its half-width in cells.

    Contrast with :func:`stiffness_from_quality`, which stiffens cells that have
    ALREADY gone bad. This one decides in advance which cells are allowed to.
    """
    nx1, ny1 = shape
    node_band = support_mask(shape, ij, rings)
    # a cell is in the band if any of its four corner pillars is
    cell = (
        node_band[:-1, :-1] | node_band[1:, :-1]
        | node_band[1:, 1:] | node_band[:-1, 1:]
    )
    out = np.ones((nx1 - 1, ny1 - 1), dtype=np.float64)
    out[cell] = float(softness)
    return out


def stiffness_from_quality(
    lattice: PillarLattice,
    *,
    power: float = 1.0,
    floor: float = 0.05,
    plane: str = "top",
) -> np.ndarray:
    """``(NX, NY)`` per-cell stiffness that rises as cell quality falls.

    ``E_cell = 1 / max(scaled_jacobian, floor) ** power``

    Feeding this back into :func:`elastic_stiffness` makes already-distorted cells
    RIGID, so the next solve pushes deformation into cells that still have shape to
    give. Iterating this is the standard way to even out a morph's worst tail
    without changing the grid's topology: nothing moves in the CPG structure, the
    material just becomes non-uniform.

    ``power`` controls aggressiveness — 0 disables it, 1 is gentle, 2+ strongly
    protects bad cells. ``floor`` stops a near-degenerate cell producing an
    unbounded modulus.
    """
    from cpg2unstructured.quality import scaled_jacobian

    xy = lattice.top if plane == "top" else lattice.bot
    sj = np.clip(scaled_jacobian(xy), floor, None)
    return (1.0 / sj) ** float(power)


def support_mask(shape: tuple[int, int], ij: np.ndarray, rings: int) -> np.ndarray:
    """``(NX+1, NY+1)`` bool — pillars within ``rings`` lattice steps of ``ij``.

    A dilation of the targeted set over the 4-neighbourhood. Used to confine a morph
    to the fault's neighbourhood: everything outside is pinned, so the deck diff and
    any distortion stay local and the rest of the grid is byte-identical.

    ``rings=0`` frees only the targeted pillars themselves — the sharpest possible
    change, and the one most likely to fold, since their immediate neighbours cannot
    give way at all.
    """
    m = np.zeros(shape, dtype=bool)
    m[ij[:, 0], ij[:, 1]] = True
    for _ in range(max(int(rings), 0)):
        g = m.copy()
        g[:-1, :] |= m[1:, :]
        g[1:, :] |= m[:-1, :]
        g[:, :-1] |= m[:, 1:]
        g[:, 1:] |= m[:, :-1]
        m = g
    return m


def _rim_tangents(mask: np.ndarray) -> np.ndarray:
    """Unit tangent along the boundary at each rim pillar; zero where ill-defined.

    A rim pillar's tangent is the direction through its rim neighbours. With two
    collinear neighbours it is their difference; with one it points at that
    neighbour. A corner — two neighbours that are not collinear — gets a zero
    vector, which the caller treats as fully pinned, since sliding along either
    edge would leave the outline.
    """
    nx1, ny1 = mask.shape
    tang = np.zeros((nx1, ny1, 2))
    ii, jj = np.nonzero(mask)
    for i, j in zip(ii, jj):
        dirs = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx1 and 0 <= b < ny1 and mask[a, b]:
                dirs.append(np.array([di, dj], dtype=float))
        if len(dirs) == 1:
            tang[i, j] = dirs[0]
        elif len(dirs) >= 2:
            # collinear pair -> a clean tangent; anything else is a corner
            d0, d1 = dirs[0], dirs[1]
            if abs(float(d0 @ d1)) > 0.5:          # parallel or antiparallel
                tang[i, j] = d0 - d1 if float(d0 @ d1) < 0 else d0
        nrm = np.linalg.norm(tang[i, j])
        if nrm > 0:
            tang[i, j] /= nrm
    return tang


def _reduction(
    n_pillars: int,
    shape: tuple[int, int],
    bmask: np.ndarray,
    mode: str,
    free_mask: np.ndarray | None = None,
) -> sp.csr_array:
    """Map free DOFs to the full ``2N`` displacement vector.

    Dirichlet elimination, expressed as a rectangular ``R`` so the reduced system
    ``RᵀAR z = Rᵀb`` stays symmetric positive definite in every mode:

    - ``free``   — identity; nothing eliminated.
    - ``pinned`` — rim columns dropped; both DOFs fixed at zero.
    - ``slide``  — each rim pillar keeps ONE DOF, along the local rim tangent, so
      it may redistribute along the outline but never leave it. Corners keep none.
    """
    nx1, ny1 = shape
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    free = 0
    tang = _rim_tangents(bmask) if mode == "slide" else None

    for p in range(n_pillars):
        i, j = divmod(p, ny1)
        if free_mask is not None and not free_mask[i, j]:
            continue                                # outside the support: pinned
        on_rim = bool(bmask[i, j])
        if not on_rim or mode == "free":
            for k in (0, 1):
                rows.append(2 * p + k)
                cols.append(free)
                vals.append(1.0)
                free += 1
        elif mode == "pinned":
            continue
        else:                                       # slide
            t = tang[i, j]                          # type: ignore[index]
            if np.linalg.norm(t) == 0:              # corner -> fully pinned
                continue
            rows.extend([2 * p, 2 * p + 1])
            cols.extend([free, free])
            vals.extend([float(t[0]), float(t[1])])
            free += 1

    return sp.csr_array((vals, (rows, cols)), shape=(2 * n_pillars, max(free, 1)))


def _assemble(
    lattice: PillarLattice,
    targets: PillarTargets,
    which: str,
    h: np.ndarray,
    w: float,
    bmask: np.ndarray,
    mode: str,
    free_mask: np.ndarray | None = None,
    normal_only: bool = True,
    smoothness: str = "laplacian",
    poisson: float = 0.3,
    stiffness: np.ndarray | None = None,
):
    """Build the reduced normal equations for one endpoint plane."""
    nx1, ny1 = lattice.shape
    n = nx1 * ny1
    xy = lattice.top if which == "top" else lattice.bot
    tgt = targets.xy_top if (which == "top" or targets.is_rigid) else targets.xy_bot
    assert tgt is not None

    flat = targets.ij[:, 0] * ny1 + targets.ij[:, 1]
    om = targets.weights()
    m = max(len(flat), 1)

    # Fidelity rows pick a pillar DOF; the RHS is the offset to the target, scaled
    # by ONE GLOBAL length rather than each pillar's own spacing. A per-pillar scale
    # makes the unknowns mean different physical distances at different pillars,
    # and the Laplacian then couples quantities that are not comparable — which on
    # a graded grid is the difference between a 13.5% and a usable result.
    hbar = float(np.mean(h))
    here = xy[targets.ij[:, 0], targets.ij[:, 1]]
    rhs_xy = (tgt - here) / hbar

    if normal_only:
        # ONE equation per incidence: only the component of the offset ALONG the
        # target's own normal is constrained, leaving the tangential position free.
        #
        # Pinning both components fixes the pillar to an exact POINT, but the
        # requirement is only that it lie ON the line. The extra equation removes
        # the freedom the solver needs to redistribute pillars along the fault, and
        # that freedom is what keeps cells well shaped: over-constraining shows up
        # directly as shear, skew and non-orthogonality.
        nrm = rhs_xy.copy()
        ln = np.linalg.norm(nrm, axis=1)
        safe = ln > 1e-14
        nrm[safe] /= ln[safe, None]

        # A pillar ALREADY ON the fault has target == current, so this normal is
        # undefined. Zeroing it — which is what a plain `nrm[~safe] = 0` does —
        # leaves an all-zero fidelity row: the pillar is constrained by NOTHING and
        # the smoothness term is free to drag it off the fault it was already on.
        # Measured on a 45-degree trace through a 12x12 lattice, where the line
        # passes exactly through 11 of the 25 claimed pillars: those 11 started at
        # distance 0 and finished 51-52 m away, taking the case mean to 29.97 m.
        #
        # Such a pillar is at its target, so PIN it there — both components. That
        # costs no tangential freedom worth having: the freedom exists to let a
        # pillar slide to where it fits best, and this one is already there.
        pinned = ~safe
        r = np.arange(len(flat))
        data = [np.sqrt(om) * nrm[:, 0], np.sqrt(om) * nrm[:, 1]]
        cols = [flat * 2, flat * 2 + 1]
        idx = [r, r]
        b_fid = np.sqrt(om) * np.sum(rhs_xy * nrm, axis=1)
        n_rows = len(flat)
        if pinned.any():
            k = np.nonzero(pinned)[0]
            extra = np.arange(len(k)) + n_rows
            # two rows each: x then y, both pinning the pillar to its target
            data += [np.sqrt(om[k]), np.sqrt(om[k])]
            cols += [flat[k] * 2, flat[k] * 2 + 1]
            idx += [extra, extra + len(k)]
            b_fid = np.concatenate([b_fid, rhs_xy[k, 0], rhs_xy[k, 1]])
            n_rows += 2 * len(k)
        F = sp.csr_array(
            (np.concatenate(data), (np.concatenate(idx), np.concatenate(cols))),
            shape=(n_rows, 2 * n),
        )
    else:
        rows = np.arange(len(flat)) * 2
        F = sp.csr_array(
            (
                np.concatenate([np.sqrt(om), np.sqrt(om)]),
                (
                    np.concatenate([rows, rows + 1]),
                    np.concatenate([flat * 2, flat * 2 + 1]),
                ),
            ),
            shape=(2 * len(flat), 2 * n),
        )
        b_fid = np.empty(2 * len(flat))
        b_fid[0::2] = np.sqrt(om) * rhs_xy[:, 0]
        b_fid[1::2] = np.sqrt(om) * rhs_xy[:, 1]

    if smoothness == "elastic":
        # Couples the x and y components, so shearing a cell costs energy — the
        # Laplacian cannot see shear at all.
        L2 = elastic_stiffness(xy, poisson=poisson, stiffness=stiffness)
    else:
        L = laplacian_matrix(nx1, ny1, xy)
        L2 = sp.block_diag([L, L], format="csr")
        # interleave (x0,y0,x1,y1,...) rather than block [all x | all y]
        perm = np.empty(2 * n, dtype=np.int64)
        perm[0::2] = np.arange(n)
        perm[1::2] = np.arange(n) + n
        P = sp.csr_array((np.ones(2 * n), (np.arange(2 * n), perm)),
                         shape=(2 * n, 2 * n))
        L2 = P @ L2 @ P.T

    # Smoothness as the DIRICHLET ENERGY d^T L d — the sum over lattice edges of
    # c_jk‖d_j − d_k‖² — not ‖L d‖².
    #
    # ‖L d‖² makes the normal matrix F^T F + (w/n) L^T L, and L^T L is a BIHARMONIC
    # operator whose null space contains every harmonic field. With only 2·N_f
    # constrained rows out of 2·N_p (Norne: 460 of 10,622) the system is then
    # near-singular at small w, and spsolve returns noise: measured displacements of
    # 577 m on ~100 m cells while the distance to the fault fell only 13.5%.
    #
    # L itself is positive semi-definite with a null space of just the constants, so
    # F^T F + (w/n) L is well conditioned for any w > 0 and still expresses exactly
    # what is wanted — that the DISPLACEMENT field vary smoothly, leaving the grid's
    # own grading (L(x⁰)) untouched.
    A = (F.T @ F) / m + (w / n) * L2
    b = (F.T @ b_fid) / m

    R = _reduction(n, (nx1, ny1), bmask, mode, free_mask)
    Ar = (R.T @ A @ R).tocsc()
    br = R.T @ b
    return Ar, br, R, hbar, tgt


def _solve_once(
    lattice: PillarLattice,
    targets: PillarTargets,
    h: np.ndarray,
    w: float,
    bmask: np.ndarray,
    mode: str,
    free_mask: np.ndarray | None = None,
    normal_only: bool = True,
    smoothness: str = "laplacian",
    poisson: float = 0.3,
    stiffness: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for (d_top, d_bot) in model units at one value of ``w``."""
    nx1, ny1 = lattice.shape
    out = []
    planes = ("top",) if targets.is_rigid else ("top", "bot")
    for which in planes:
        Ar, br, R, hbar, _ = _assemble(lattice, targets, which, h, w, bmask, mode, free_mask, normal_only,
                  smoothness, poisson, stiffness)
        # tiny Tikhonov term keeps the system non-singular when a mode leaves a
        # free component wholly unconstrained (e.g. free boundary, no targets)
        Ar = Ar + sp.eye_array(Ar.shape[0], format="csc") * 1e-12
        z = spla.spsolve(Ar, br)
        d_flat = R @ np.atleast_1d(z)
        d = np.empty((nx1, ny1, 2))
        d[..., 0] = d_flat[0::2].reshape(nx1, ny1) * hbar
        d[..., 1] = d_flat[1::2].reshape(nx1, ny1) * hbar
        out.append(d)
    return (out[0], out[0]) if targets.is_rigid else (out[0], out[1])


def _residuals(
    lattice: PillarLattice,
    targets: PillarTargets,
    original: PillarLattice | None = None,
    normal_only: bool = False,
) -> np.ndarray:
    """How far each target row still is from what the solve was asked to achieve.

    With point targets that is the distance to the target POINT. Under
    ``normal_only`` the objective constrains only the component along the target's
    own normal — the tangential position is deliberately free — so the point
    distance would overstate the error. The residual is then the perpendicular
    distance to the target's line, which is what "minimise the distance to the
    fault" actually means.
    """
    if targets.n == 0:
        return np.empty(0)
    now = lattice.top[targets.ij[:, 0], targets.ij[:, 1]]
    delta = now - targets.xy_top
    if not normal_only or original is None:
        return np.linalg.norm(delta, axis=1)
    ref = original.top[targets.ij[:, 0], targets.ij[:, 1]]
    n = targets.xy_top - ref                       # the direction assembly used
    ln = np.linalg.norm(n, axis=1)
    safe = ln > 1e-14
    n = np.where(safe[:, None], n / np.where(safe, ln, 1.0)[:, None], 0.0)
    return np.abs(np.sum(delta * n, axis=1))


def morph_pillars(
    grid: CornerPointGrid,
    targets: PillarTargets,
    *,
    w0: float = 0.3,
    boundary: str = "slide",
    boundary_scope: str = "active",
    support_rings: int | None = None,
    normal_only: bool = True,
    smoothness: str = "elastic",
    poisson: float = 0.3,
    stiffness: np.ndarray | None = None,
    stiffness_iterations: int = 0,
    stiffness_power: float = 1.0,
    band_rings: int | None = None,
    band_softness: float = 0.02,
    write_tol: float = 1e-9,
    min_fraction: float = 0.10,
    max_bisect: int = 20,
    w_max: float = 1e6,
    polish: str = "none",
    polish_fidelity: float = 4.0,
    polish_size: float = 0.0,
    polish_shape_size: float = 0.0,
    polish_ortho: float = 0.0,
    polish_iterations: int = 200,
) -> MorphResult:
    """Move ``targets``' pillars toward their targets, keeping every cell valid.

    Solves at ``w0`` first. If that folds a cell, bisects ``w`` upward between the
    last folding value and a known-valid one, returning the largest fidelity that
    stays valid — see the module docstring for why the bracket is sound.

    Parameters
    ----------
    w0
        Starting smoothness weight. Dimensionless (see module docstring).
    boundary, boundary_scope
        ``slide`` | ``pinned`` | ``free`` over the ``active`` rim or the ``lattice``
        edge. Defaults keep the domain outline while letting rim pillars
        redistribute along it.
    write_tol
        Displacements below this (in model units) are zeroed before the grid is
        written, so the deck diff stays local instead of touching every pillar.
    min_fraction
        Minimum surviving pillar separation, as a fraction of the original.
    smoothness
        ``"elastic"`` (default) minimises linear-elastic strain energy, which
        penalises SHEAR directly because it couples the x and y components.
        ``"laplacian"`` penalises the displacement gradient only; it is cheaper to
        reason about but nearly blind to shear, which shows up as skewed cells.
    poisson
        Poisson's ratio for the elastic operator. ``0.0`` lets cells change area
        freely, ``0.3`` is a good default, and values toward ``0.5`` increasingly
        preserve area at the cost of more shear elsewhere.
    stiffness
        Optional ``(NX, NY)`` per-cell modulus. Non-uniform stiffness steers WHERE
        the deformation goes without changing the grid.
    stiffness_iterations
        Re-solve this many times, each time stiffening cells in proportion to how
        distorted they became (see :func:`stiffness_from_quality`). Evens out the
        worst-cell tail. ``0`` (default) disables it.
    stiffness_power
        How aggressively bad cells are stiffened between those iterations.
    band_rings
        Enable the SACRIFICIAL BAND: cells within this many rings of the fault are
        made compliant (``band_softness``) so they absorb the deformation, while the
        rest of the grid stays near its original shape. Those band cells are expected
        to be small and skewed — that is the point — so quality is judged outside the
        band and the band is held only to the hard validity constraints.
    band_softness
        The band's modulus relative to the surrounding grid. Lower is softer;
        0.02 makes the band 50x more compliant.
    polish
        ``"none"`` (default) or ``"barrier"``. The barrier polish takes the
        quadratic solution as a starting point and minimises a TMOP distortion
        energy measured against each cell's ORIGINAL shape — see
        :mod:`cpg2unstructured.polish`. It cannot invert a cell, because the energy
        diverges as one degenerates, and it removes the degenerate cells the
        quadratic solve leaves behind: 14 cells below scaled Jacobian 0.3 became 0
        on the benchmark's `zigzag` case. It costs fidelity, so it is opt-in.
    polish_fidelity
        How hard the polish holds the pillars on the fault. LARGER is closer — the
        opposite sense to ``w``, because here it is the fault term being weighted.
    polish_size, polish_shape_size
        Optional extra barrier terms: ``size`` resists area collapse (TMOP metric
        056), ``shape_size`` is the symmetric Dirichlet energy (metric 007). Both
        default to off; ``shape`` alone carried nearly all of the measured gain.
    polish_ortho
        Weight on an explicit ORTHOGONALITY term — the one thing no other operator
        here targets directly, and the metric a two-point flux scheme actually pays
        for. It is measured against each cell's ORIGINAL angle, so a sheared deck is
        not pulled square. Measured on the 12x12 kinked case: at 0.5 it costs 0.6 m
        of fidelity for 2.4 deg of p95; at 8 it halves the non-orthogonality MAXIMUM
        (42.0 -> 15.9 deg) and takes the worst scaled Jacobian from 0.49 to 0.86,
        for roughly double the residual. A clean monotone dial.
    polish_iterations
        L-BFGS iteration cap for the polish.
    support_rings
        Confine the morph to pillars within this many lattice rings of the targeted
        set; everything beyond is pinned. ``None`` (default) leaves the whole grid
        free, which gives the best mesh quality because distortion can spread. A
        finite value keeps the change local — the rest of the deck is untouched —
        at the cost of concentrating strain at the edge of the band.
    """
    if boundary not in BOUNDARY_MODES:
        raise ValueError(f"boundary must be one of {BOUNDARY_MODES}; got {boundary!r}")
    if polish not in POLISH_MODES:
        raise ValueError(f"polish must be one of {POLISH_MODES}; got {polish!r}")

    original = pillar_lattice(grid)
    h = local_spacing(original)
    bmask = boundary_mask(grid, scope=boundary_scope)
    free_mask = (
        support_mask(original.shape, targets.ij, support_rings)
        if support_rings is not None and targets.n
        else None
    )
    if band_rings is not None and targets.n:
        # Sacrificial band: soften the cells at the fault so they absorb the
        # deformation, leaving the surrounding grid close to its original shape.
        stiffness = stiffness_from_band(
            original.shape, targets.ij, rings=band_rings, softness=band_softness
        )

    def attempt(w: float) -> tuple[MorphResult, PillarLattice]:
        d_top, d_bot = _solve_once(original, targets, h, w, bmask, boundary, free_mask, normal_only,
                                   smoothness, poisson, stiffness)
        small = np.linalg.norm(d_top, axis=-1) < write_tol
        d_top = np.where(small[..., None], 0.0, d_top)
        d_bot = np.where(small[..., None], 0.0, d_bot)
        moved = apply_displacement(grid, d_top, d_bot)
        lat = pillar_lattice(moved)
        rep = check_validity(lat, original, min_fraction=min_fraction)
        return (
            MorphResult(
                grid=moved, d_top=d_top, d_bot=d_bot, w=w,
                validity=rep, residuals=_residuals(lat, targets, original, normal_only),
            ),
            lat,
        )

    def refine(res: MorphResult) -> MorphResult:
        """Re-solve with cells stiffened in proportion to how bad they became."""
        nonlocal stiffness
        out = res
        for _ in range(max(int(stiffness_iterations), 0)):
            stiffness = stiffness_from_quality(
                pillar_lattice(out.grid), power=stiffness_power
            )
            cand, _ = attempt(out.w)
            if not cand.ok:
                break
            out = cand
        return out

    def finish(res: MorphResult) -> MorphResult:
        """Apply the barrier polish, if asked, and never return a worse grid."""
        if polish != "barrier" or not targets.n:
            return res
        return _apply_polish(
            grid, original, res, targets, bmask, boundary, free_mask,
            fidelity=polish_fidelity, size=polish_size,
            shape_size=polish_shape_size, ortho=polish_ortho,
            max_iter=polish_iterations,
            min_fraction=min_fraction, normal_only=normal_only,
        )

    best, _ = attempt(w0)
    best.history.append((w0, best.validity.ok, best.validity.n_folded))
    if best.ok:
        return finish(refine(best))

    # w0 folded — climb to a valid upper bracket, then bisect toward w0.
    lo, hi = w0, w0
    valid: MorphResult | None = None
    for _ in range(max_bisect):
        hi = min(hi * 10.0, w_max)
        cand, _ = attempt(hi)
        best.history.append((hi, cand.validity.ok, cand.validity.n_folded))
        if cand.ok:
            valid = cand
            break
        if hi >= w_max:
            break
    if valid is None:
        best.bisection_steps = len(best.history) - 1
        return best                                  # no valid w found; caller decides

    steps = 0
    for _ in range(max_bisect):
        mid = float(np.sqrt(lo * hi))                # geometric — w spans decades
        cand, _ = attempt(mid)
        steps += 1
        best.history.append((mid, cand.validity.ok, cand.validity.n_folded))
        if cand.ok:
            valid, hi = cand, mid
        else:
            lo = mid
        if hi / lo < 1.01:
            break

    valid.bisection_steps = steps
    valid.history = best.history
    return finish(refine(valid))


def _apply_polish(
    grid: CornerPointGrid,
    original: PillarLattice,
    res: MorphResult,
    targets: PillarTargets,
    bmask: np.ndarray,
    boundary: str,
    free_mask: np.ndarray | None,
    *,
    fidelity: float,
    size: float,
    shape_size: float,
    ortho: float,
    max_iter: int,
    min_fraction: float,
    normal_only: bool,
) -> MorphResult:
    """Run the barrier polish on a finished morph and re-validate the outcome.

    The polish is a strict improvement or nothing: if it fails to descend, or if the
    polished grid does not pass the SAME validity checks the solve had to pass, the
    original result is returned with the report attached saying so. A stage that can
    silently make a deck worse is not a polish.
    """
    from cpg2unstructured.polish import polish_lattice

    nx1, ny1 = original.shape
    R = _reduction(nx1 * ny1, (nx1, ny1), bmask, boundary, free_mask)
    terms = DistortionTerms(shape=1.0, size=size, shape_size=shape_size, ortho=ortho)
    started = pillar_lattice(res.grid)

    rigid = targets.xy_bot is None

    tops, report = polish_lattice(
        original.top, started.top, R, targets.ij, targets.xy_top,
        fidelity=fidelity, terms=terms, max_iter=max_iter,
    )
    if not report.applied:
        res.polish = report
        return res

    d_top = tops - original.top
    if rigid:
        # Stage 1 moves the whole pillar rigidly; the polish must not break that or
        # the pillars would acquire a tilt the fault never asked for.
        d_bot = d_top.copy()
    else:
        assert targets.xy_bot is not None          # `rigid` is exactly this test
        bots, rep_b = polish_lattice(
            original.bot, started.bot, R, targets.ij, targets.xy_bot,
            fidelity=fidelity, terms=terms, max_iter=max_iter,
        )
        d_bot = (bots - original.bot) if rep_b.applied else res.d_bot

    moved = apply_displacement(grid, d_top, d_bot)
    lat = pillar_lattice(moved)
    rep = check_validity(lat, original, min_fraction=min_fraction)
    if not rep.ok:
        res.polish = PolishReport(
            report.energy_before, report.energy_after, report.iterations, False,
            f"polished grid failed validity ({rep.summary()}); the input was kept",
            applied=False,
        )
        return res

    return MorphResult(
        grid=moved, d_top=d_top, d_bot=d_bot, w=res.w, validity=rep,
        residuals=_residuals(lat, targets, original, normal_only),
        iterations=res.iterations, bisection_steps=res.bisection_steps,
        history=res.history, polish=report,
    )
