"""Barrier polish — a second stage that fixes what the quadratic solve damages.

``morph_pillars`` solves ``||F d - r||² + w·S(d)`` with a quadratic ``S``. That is
fast, linear, and gives a good answer everywhere except where it does not: a
quadratic energy is finite for an inverted cell, so the objective never resists a
fold, and the engine has to recover validity afterwards by raising ``w`` for the
whole model.

This module takes that solution as a starting point and minimises a **barrier**
distortion energy instead:

    E(P) = fidelity · mean‖P_claimed - target‖² / h²   +   mean μ(T)

with ``μ`` from :mod:`cpg2unstructured.distortion` and ``T = J W⁻¹`` measured
against **the original cell's own shape**. Because ``μ → ∞`` as a cell degenerates,
a backtracking line search cannot step into an inverted configuration — validity
becomes a property of the objective rather than something checked afterwards.

Measured on the 12x12 benchmark cases, against the shipped elastic operator:

===========  ===================  =====================  ===============
case         elastic: cells<0.3   barrier: cells<0.3     worst SJ
===========  ===================  =====================  ===============
kink_sharp   0                    0                      0.33 -> 0.49
arc_tight    **8**                **0**                  5.2e-04 -> 0.39
zigzag       **14**               **0**                  5.2e-04 -> 0.48
diagonal_45  0                    0                      0.76 -> 0.51
===========  ===================  =====================  ===============

The cost is fidelity: 1-4 m on a 100 m grid at ``fidelity=4``. On ``diagonal_45`` it
is not a cost at all — the polish is both closer (30.0 -> 16.3 m) and valid.

The polish is OFF by default. It changes results, so it is opt-in until it has been
measured on the decks that matter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from cpg2unstructured.distortion import (
    DistortionTerms,
    cell_quads,
    distortion_energy,
    shape_targets,
)

__all__ = ["PolishReport", "polish_lattice"]


@dataclass
class PolishReport:
    """What the polish did. Every field is measured, none inferred."""

    energy_before: float
    energy_after: float
    iterations: int
    converged: bool
    message: str = ""
    applied: bool = True

    @property
    def improvement(self) -> float:
        """Fraction of the starting energy removed; 0 when it could not start."""
        if not np.isfinite(self.energy_before) or self.energy_before <= 0:
            return 0.0
        return float(1.0 - self.energy_after / self.energy_before)

    def summary(self) -> str:
        if not self.applied:
            return f"polish skipped — {self.message}"
        return (
            f"polish {self.energy_before:.4g} -> {self.energy_after:.4g} "
            f"({100 * self.improvement:.1f}% of the distortion removed) · "
            f"{self.iterations} iteration(s) · "
            f"{'converged' if self.converged else self.message}"
        )


def _reduced_start(R: sp.csr_array, d: np.ndarray) -> np.ndarray:
    """The reduced coordinates of a displacement the reduction can represent.

    The starting displacement came from the SAME reduction, so this is exact rather
    than a projection: ``RᵀR`` is diagonal for `free`/`pinned` and diagonal for
    `slide` too, since each rim pillar contributes one unit-tangent column.
    """
    rtr = np.asarray((R.T @ R).diagonal(), dtype=np.float64)
    rtr[rtr == 0.0] = 1.0
    return np.asarray(R.T @ d.reshape(-1), dtype=np.float64) / rtr


def polish_lattice(
    original: np.ndarray,
    start: np.ndarray,
    reduction: sp.csr_array,
    target_ij: np.ndarray,
    target_xy: np.ndarray,
    *,
    fidelity: float = 4.0,
    terms: DistortionTerms | None = None,
    use_shape_targets: bool = True,
    max_iter: int = 200,
    second: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, PolishReport]:
    """Minimise fidelity + barrier distortion, starting from ``start``.

    ``original`` and ``start`` are ``(nx1, ny1, 2)`` plan-view lattices; the result
    has the same shape. ``reduction`` is the engine's Dirichlet elimination matrix,
    so the polish honours exactly the boundary mode and support radius the solve
    used — a pillar the solve was not allowed to move is not moved here either.

    ``fidelity`` weights the target term against the distortion term. It is NOT the
    engine's ``w``: larger means closer to the fault, the opposite sense, because
    here the fault term is the one being weighted.

    ``second`` is ``(original_bot, start_bot)``. When a morph is RIGID the same
    displacement is applied to both ends of every pillar, so optimising the top
    plane alone leaves the bottom plane unevaluated — and on a deck with tilted
    pillars the bottom is different geometry that can fold while the top improves.
    ``check_validity`` then checks both, sees the bottom fold, and discards the
    whole polish. Measured on Norne: every polish refusal in the field study was a
    bottom-plane fold, with the top plane folding zero times. Passing ``second``
    puts the bottom plane's distortion in the SAME objective under the SAME
    variable, so the barrier sees it and cannot step into it.

    If the starting lattice is already inverted the barrier is infinite everywhere
    and there is nothing to descend; the start is returned unchanged with
    ``applied=False`` rather than silently doing nothing.
    """
    terms = terms or DistortionTerms()
    nx1, ny1 = original.shape[0], original.shape[1]
    P0 = np.asarray(original, dtype=np.float64).reshape(-1, 2)
    Ps = np.asarray(start, dtype=np.float64).reshape(-1, 2)
    quads = cell_quads(nx1, ny1)

    if not terms.any_active():
        return np.array(start, dtype=np.float64), PolishReport(
            0.0, 0.0, 0, True, "no distortion terms active", applied=False
        )

    # Only the bottom plane's ORIGINAL geometry is needed. A rigid morph has one
    # displacement field for both ends of a pillar, so the same reduced variable
    # places the bottom plane too: its start is determined, not free.
    P0b = winv_b = None
    if second is not None:
        P0b = np.asarray(second[0], dtype=np.float64).reshape(-1, 2)

    try:
        winv = shape_targets(P0, quads) if use_shape_targets else None
        if P0b is not None:
            winv_b = shape_targets(P0b, quads) if use_shape_targets else None
    except ValueError as exc:
        return np.array(start, dtype=np.float64), PolishReport(
            float("inf"), float("inf"), 0, False, str(exc), applied=False
        )

    # Length scale for the fidelity term, so `fidelity` is dimensionless and
    # transfers between decks the way the engine's `w` does.
    h = float(np.linalg.norm(P0[quads[:, 1]] - P0[quads[:, 0]], axis=1).mean())
    h = h if h > 0 else 1.0
    n_t = max(len(target_ij), 1)
    t_flat = np.asarray(target_ij[:, 0] * ny1 + target_ij[:, 1], dtype=np.int64)
    t_xy = np.asarray(target_xy, dtype=np.float64)

    def unpack(z: np.ndarray) -> np.ndarray:
        return P0 + np.asarray(reduction @ z, dtype=np.float64).reshape(-1, 2)

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        P = unpack(z)
        E, G = distortion_energy(P, quads, winv, terms)
        if P0b is not None and np.isfinite(E):
            # same displacement, the other plane's geometry
            Eb, Gb = distortion_energy(
                P0b + np.asarray(reduction @ z, dtype=np.float64).reshape(-1, 2),
                quads, winv_b, terms)
            E += Eb
            G = G + Gb
        if not np.isfinite(E):
            # The line search must see a large finite value with a usable gradient,
            # not a NaN: L-BFGS-B stops on non-finite objectives instead of
            # backtracking, which would strand the polish at the first bad step.
            return 1e30, np.zeros(z.shape)
        if len(target_ij):
            diff = P[t_flat] - t_xy
            E += fidelity * float((diff ** 2).sum()) / (n_t * h * h)
            np.add.at(G, t_flat, 2.0 * fidelity * diff / (n_t * h * h))
        return E, np.asarray(reduction.T @ G.reshape(-1), dtype=np.float64)

    z0 = _reduced_start(reduction, Ps - P0)
    e0, _ = objective(z0)
    if not np.isfinite(e0) or e0 >= 1e29:
        return np.array(start, dtype=np.float64), PolishReport(
            float("inf"), float("inf"), 0, False,
            "the starting lattice is already inverted", applied=False
        )

    res = minimize(objective, z0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=int(max_iter), maxfun=4 * int(max_iter)))
    e1, _ = objective(res.x)

    # Never hand back something worse than we were given. L-BFGS-B can stop on a
    # line-search failure at a point above the start, and this stage is a polish:
    # it may decline to help, it may not do harm.
    if not np.isfinite(e1) or e1 > e0:
        return np.array(start, dtype=np.float64), PolishReport(
            e0, e0, int(res.nit), False,
            "no descent found; the input was kept", applied=False
        )

    out = unpack(res.x).reshape(nx1, ny1, 2)
    msg = res.message if isinstance(res.message, str) else str(res.message)
    return out, PolishReport(e0, e1, int(res.nit), bool(res.success), msg)
