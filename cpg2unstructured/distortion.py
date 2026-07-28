"""Barrier distortion metrics for the pillar lattice — TMOP, in 2D.

The morphing engine's smoothness term is quadratic. A quadratic energy is FINITE
for an inverted cell, so nothing in the objective resists a fold: the solver
destroys a cell without knowing, and validity is recovered afterwards by raising
``w`` globally — trading fidelity everywhere to fix a defect in one place.

A *barrier* metric diverges as a cell degenerates. An optimiser that keeps the
energy finite therefore cannot produce an inverted cell, and validity stops being a
post-hoc check and becomes a property of the objective.

Quality is measured on each cell corner's Jacobian ``J`` against a **target** ``W``,
through ``T = J W⁻¹``. The choice of ``W`` is the substance:

    ``W = I``                the cell should become a unit square
    ``W = J_original``       the cell should KEEP THE SHAPE IT HAD      <- what we want

For a corner-point grid the second is the only defensible target. A graded or
sheared deck is not a bad deck, and an objective pulling it toward squares would
damage it in the name of quality. ``shape_targets`` builds ``W`` from the original
lattice, so the energy measures how much a cell was CHANGED, not how far it is from
some ideal.

The metric names and formulas follow MFEM's TMOP implementation so they can be
cross-checked against it:

===========  =====================================  ==============================
metric       formula                                controls
===========  =====================================  ==============================
``shape``    ``0.5 |T|² / det(T) - 1``              shape (barrier, polyconvex)
``size``     ``0.5 (det T + 1/det T) - 1``          area collapse (barrier)
``shape_size`` ``|T - T⁻ᵀ|²``                       both (barrier, symmetric Dirichlet)
===========  =====================================  ==============================

Every energy here returns its analytic gradient with it; ``tests/test_distortion.py``
checks each one against finite differences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CORNER_STENCIL",
    "cell_quads",
    "corner_jacobians",
    "shape_targets",
    "distortion_energy",
    "DistortionTerms",
]

# The four corners of a quad (p00, p10, p11, p01), each giving the two edges that
# leave it in counter-clockwise order. On a positively oriented cell every one of
# these has a positive determinant, so a sign change anywhere means that corner
# inverted — which is exactly what the barrier must see.
CORNER_STENCIL = (
    ((1, 0), (3, 0)),
    ((2, 1), (0, 1)),
    ((3, 2), (1, 2)),
    ((0, 3), (2, 3)),
)


def cell_quads(nx1: int, ny1: int) -> np.ndarray:
    """``(Ncell, 4)`` flat pillar indices of every column, counter-clockwise.

    Flat index of pillar ``(i, j)`` is ``i * ny1 + j`` — the same ordering
    ``PillarLattice`` uses when reshaped.
    """
    i, j = np.meshgrid(np.arange(nx1 - 1), np.arange(ny1 - 1), indexing="ij")

    def flat(a, b):
        return (a * ny1 + b).ravel()

    return np.stack(
        [flat(i, j), flat(i + 1, j), flat(i + 1, j + 1), flat(i, j + 1)], axis=1
    )


def corner_jacobians(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """``(Ncell, 4, 2, 2)`` corner Jacobians; columns are the two outgoing edges."""
    pts = points[quads]
    cols = [
        np.stack([pts[:, a] - pts[:, c], pts[:, b] - pts[:, d]], axis=-1)
        for (a, c), (b, d) in CORNER_STENCIL
    ]
    return np.stack(cols, axis=1)


def shape_targets(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """``(Ncell, 4, 2, 2)`` inverse target matrices from an ORIGINAL lattice.

    Raises if any original cell is already degenerate: ``W`` must be invertible, and
    a deck that arrives with a zero-area column cannot be measured against itself.
    Use ``metric='shape'`` with identity targets, or repair the deck first.
    """
    W = corner_jacobians(points, quads)
    det = W[..., 0, 0] * W[..., 1, 1] - W[..., 0, 1] * W[..., 1, 0]
    bad = int((np.abs(det) < 1e-30).sum())
    if bad:
        raise ValueError(
            f"{bad} corner(s) of the ORIGINAL lattice are degenerate, so they cannot "
            "be used as shape targets; pass targets=None for identity targets"
        )
    return np.linalg.inv(W)


def _adjugate_t(T: np.ndarray) -> np.ndarray:
    """``d det(T) / dT`` — the transposed adjugate, for 2x2 blocks."""
    out = np.empty_like(T)
    out[..., 0, 0] = T[..., 1, 1]
    out[..., 0, 1] = -T[..., 1, 0]
    out[..., 1, 0] = -T[..., 0, 1]
    out[..., 1, 1] = T[..., 0, 0]
    return out


@dataclass
class DistortionTerms:
    """Which metrics contribute, and with what weight.

    ``shape`` is the one that matters; the others are options measured to be worth
    less. All are barriers, so any positive weight makes the total a barrier.
    """

    shape: float = 1.0
    size: float = 0.0
    shape_size: float = 0.0

    def any_active(self) -> bool:
        return bool(self.shape or self.size or self.shape_size)


def distortion_energy(
    points: np.ndarray,
    quads: np.ndarray,
    winv: np.ndarray | None,
    terms: DistortionTerms,
) -> tuple[float, np.ndarray]:
    """Mean distortion over every cell corner, and its gradient w.r.t. ``points``.

    Returns ``(inf, zeros)`` the moment any ``det(T) <= 0``. That is the barrier
    doing its job: an optimiser doing a backtracking line search will reject the
    step rather than walk into an inverted configuration, so no cell can invert
    without the search first seeing an infinite energy.
    """
    J = corner_jacobians(points, quads)
    T = J if winv is None else J @ winv
    det = T[..., 0, 0] * T[..., 1, 1] - T[..., 0, 1] * T[..., 1, 0]
    if not np.all(det > 1e-14):
        return float("inf"), np.zeros_like(points)

    n = det.size
    fro = (T ** 2).sum(axis=(-2, -1))
    adj = _adjugate_t(T)

    energy = 0.0
    dE_dT = np.zeros_like(T)

    if terms.shape:
        energy += terms.shape * float((0.5 * fro / det - 1.0).sum()) / n
        dE_dT += terms.shape * (
            T / det[..., None, None] - (0.5 * fro / det ** 2)[..., None, None] * adj
        ) / n

    if terms.size:
        energy += terms.size * float((0.5 * (det + 1.0 / det) - 1.0).sum()) / n
        dE_dT += terms.size * (0.5 * (1.0 - 1.0 / det ** 2))[..., None, None] * adj / n

    if terms.shape_size:
        # |T - T^-T|^2 = |T|^2 - 4 + |T^-1|^2, since |T^-T|^2 = |T|^2/det^2 in 2D
        # and T:T^-T = 2 (the 2x2 identity trace). Written directly to keep the
        # gradient simple and to avoid forming T^-1.
        inv_fro = fro / det ** 2
        energy += terms.shape_size * float((fro - 4.0 + inv_fro).sum()) / n
        dE_dT += terms.shape_size * (
            2.0 * T * (1.0 + 1.0 / det ** 2)[..., None, None]
            - (2.0 * fro / det ** 3)[..., None, None] * adj
        ) / n

    dE_dJ = dE_dT if winv is None else dE_dT @ np.swapaxes(winv, -1, -2)

    grad = np.zeros_like(points)
    for k, ((a, c), (b, d)) in enumerate(CORNER_STENCIL):
        g0 = dE_dJ[:, k, :, 0]
        g1 = dE_dJ[:, k, :, 1]
        np.add.at(grad, quads[:, a], g0)
        np.add.at(grad, quads[:, c], -g0)
        np.add.at(grad, quads[:, b], g1)
        np.add.at(grad, quads[:, d], -g1)
    return float(energy), grad
