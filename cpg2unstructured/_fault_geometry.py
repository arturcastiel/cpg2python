"""Fault-face geometry helpers (PR-3+ of phase 2-faults).

Pure geometric primitives — line intersection by z-parametrisation
(this PR), face-polygon construction (PR-4's ``compute_face_geometry``).

PR-3 ships :func:`line_intersection` only. PR-4 will extend this
module with the full 8-position J-table polygon assembler.

NOT public API — ``cpg2unstructured.unstructured`` consumes these
internally when integration lands in PR-5+ of phase 2-faults.
"""
from __future__ import annotations

import numpy as np

__all__ = ["line_intersection"]

# Threshold for "lines have parallel z-trajectories" — denominator
# below this is treated as numerically singular. Real CPG pillars
# deviate <10% from vertical so Δz_A − Δz_B is typically of order
# the cell thickness (≥1e-6 in any non-degenerate deck).
_DENOM_EPS = 1e-12


def line_intersection(la_start: np.ndarray,
                      la_end: np.ndarray,
                      lb_start: np.ndarray,
                      lb_end: np.ndarray) -> np.ndarray:
    """XYZ intersection of two line-segment families, z-parametrised.

    Each line is defined by its start + end XYZ on the same pillar
    pair (so both lines go from pillar-1-XYZ to pillar-2-XYZ; their
    (x, y) trajectory is shared in the vertical-pillar limit; only
    the z values at the endpoints differ).

    Parameterising by ``t`` along line A (pillar 1 → pillar 2):

    .. math::
        z_A(t) = z_A^{p1} + t \\, (z_A^{p2} - z_A^{p1}) \\\\
        z_B(t) = z_B^{p1} + t \\, (z_B^{p2} - z_B^{p1}) \\\\
        z_A(t^*) = z_B(t^*) \\;\\Rightarrow\\;
            t^* = \\frac{z_B^{p1} - z_A^{p1}}{\\Delta z_A - \\Delta z_B}

    The intersection XYZ is then computed by interpolating along
    line A at ``t = t*``. This matches MRST's ``intersection``
    helper (``processGRDECL.m`` lines 1164-1190).

    Parameters
    ----------
    la_start, la_end : array-like, shape ``(..., 3)``
        Start + end XYZ of A-line(s).
    lb_start, lb_end : array-like, shape ``(..., 3)``
        Start + end XYZ of B-line(s).

    Returns
    -------
    pts : ndarray, shape ``(..., 3)``
        Intersection XYZ per line pair. The z column is the exact
        intersection z (not an interpolation — avoids round-off
        drift on the parametric reconstruction).

    Raises
    ------
    ValueError
        If any line pair has ``|Δz_A - Δz_B| < _DENOM_EPS`` —
        the two z-trajectories are parallel and no unique
        intersection exists by this parametrisation. (R-3 from
        the phase 2-faults study.)
    """
    la_start = np.asarray(la_start, dtype=np.float64)
    la_end   = np.asarray(la_end,   dtype=np.float64)
    lb_start = np.asarray(lb_start, dtype=np.float64)
    lb_end   = np.asarray(lb_end,   dtype=np.float64)

    za_p1 = la_start[..., 2]
    za_p2 = la_end[..., 2]
    zb_p1 = lb_start[..., 2]
    zb_p2 = lb_end[..., 2]

    delta_za = za_p2 - za_p1
    delta_zb = zb_p2 - zb_p1
    denom = delta_za - delta_zb

    # R-3 guard — parallel z-trajectories are numerically singular.
    # Use abs().min() over any leading axes so we surface the WORST
    # case rather than only the last.
    if np.abs(denom).min() < _DENOM_EPS:
        raise ValueError(
            "line_intersection: a line pair has parallel z-trajectories "
            "(|Δz_A − Δz_B| below numerical threshold); no unique "
            "intersection by z-parametrisation."
        )

    t = (zb_p1 - za_p1) / denom               # scalar or (n,)
    delta_xyz = la_end - la_start
    out = la_start + t[..., None] * delta_xyz
    # Exact z (avoid drift on the parametric form)
    out[..., 2] = za_p1 + t * delta_za
    return out
