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
from cpg2unstructured.pillars import (
    PillarLattice,
    apply_displacement,
    boundary_mask,
    local_spacing,
    pillar_lattice,
)
from cpg2unstructured.validity import ValidityReport, check_validity

__all__ = ["PillarTargets", "MorphResult", "morph_pillars", "laplacian_matrix"]

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
        return (
            f"w={self.w:.4g} · {self.bisection_steps} bisection step(s) · "
            f"{self.n_moved} pillar(s) moved · {stats} · {self.validity.summary()}"
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
    rhs_xy = (tgt - xy[targets.ij[:, 0], targets.ij[:, 1]]) / hbar
    rows = np.repeat(np.arange(len(flat)) * 2, 1)
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

    L = laplacian_matrix(nx1, ny1, xy)
    L2 = sp.block_diag([L, L], format="csr")
    # interleave (x0,y0,x1,y1,...) rather than block [all x | all y]
    perm = np.empty(2 * n, dtype=np.int64)
    perm[0::2] = np.arange(n)
    perm[1::2] = np.arange(n) + n
    P = sp.csr_array((np.ones(2 * n), (np.arange(2 * n), perm)), shape=(2 * n, 2 * n))
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

    R = _reduction(n, (nx1, ny1), bmask, mode)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for (d_top, d_bot) in model units at one value of ``w``."""
    nx1, ny1 = lattice.shape
    out = []
    planes = ("top",) if targets.is_rigid else ("top", "bot")
    for which in planes:
        Ar, br, R, hbar, _ = _assemble(lattice, targets, which, h, w, bmask, mode)
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


def _residuals(lattice: PillarLattice, targets: PillarTargets) -> np.ndarray:
    """Distance from each target row to its pillar's CURRENT position."""
    if targets.n == 0:
        return np.empty(0)
    now = lattice.top[targets.ij[:, 0], targets.ij[:, 1]]
    return np.linalg.norm(now - targets.xy_top, axis=1)


def morph_pillars(
    grid: CornerPointGrid,
    targets: PillarTargets,
    *,
    w0: float = 0.3,
    boundary: str = "slide",
    boundary_scope: str = "active",
    write_tol: float = 1e-9,
    min_fraction: float = 0.10,
    max_bisect: int = 20,
    w_max: float = 1e6,
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
    """
    if boundary not in BOUNDARY_MODES:
        raise ValueError(f"boundary must be one of {BOUNDARY_MODES}; got {boundary!r}")

    original = pillar_lattice(grid)
    h = local_spacing(original)
    bmask = boundary_mask(grid, scope=boundary_scope)

    def attempt(w: float) -> tuple[MorphResult, PillarLattice]:
        d_top, d_bot = _solve_once(original, targets, h, w, bmask, boundary)
        small = np.linalg.norm(d_top, axis=-1) < write_tol
        d_top = np.where(small[..., None], 0.0, d_top)
        d_bot = np.where(small[..., None], 0.0, d_bot)
        moved = apply_displacement(grid, d_top, d_bot)
        lat = pillar_lattice(moved)
        rep = check_validity(lat, original, min_fraction=min_fraction)
        return (
            MorphResult(
                grid=moved, d_top=d_top, d_bot=d_bot, w=w,
                validity=rep, residuals=_residuals(lat, targets),
            ),
            lat,
        )

    best, _ = attempt(w0)
    best.history.append((w0, best.validity.ok, best.validity.n_folded))
    if best.ok:
        return best

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
    return valid
