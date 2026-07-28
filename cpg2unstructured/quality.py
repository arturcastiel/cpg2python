"""Cell-quality metrics for the plan-view grid.

"The cells look bad" needs a number before it can be improved. These are the
standard quadrilateral measures, computed over each column's plan-view quad —
which is what lateral pillar motion actually changes.

Which metric matters depends on what you care about:

``scaled_jacobian``
    The workhorse validity-and-distortion measure. At each corner, the normalised
    cross product of the two incident edges; the cell's score is the minimum over
    its four corners. ``1`` is a perfect rectangle, ``0`` a degenerate corner, and
    anything ``< 0`` an inverted cell. Simulator meshing guides typically want
    ``> 0.3``; ``< 0.1`` is usually rejected outright.

``aspect_ratio``
    Longest edge over shortest. Cheap, and the first thing to blow up when a band
    of cells is sheared flat. ``1`` is square.

``skew``
    Worst corner-angle deviation from 90°, normalised to ``[0, 1]`` where ``0`` is
    perfectly orthogonal. Distinguishes a *sheared* cell (bad angles, fine edge
    lengths) from a merely *stretched* one — the two look similar in an aspect
    ratio but behave very differently.

``non_orthogonality``
    The reservoir-specific one. The angle between the vector joining two adjacent
    cell centres and the shared face normal. A two-point flux approximation assumes
    these are aligned; every degree of deviation is transmissibility error, so this
    is the metric that predicts simulation damage rather than mesh ugliness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpg2unstructured.pillars import PillarLattice

__all__ = [
    "QualityReport",
    "scaled_jacobian",
    "aspect_ratio",
    "skew",
    "non_orthogonality",
    "cell_quality",
]


def _corners(xy: np.ndarray):
    """The four corners of every column quad, in cyclic order."""
    return xy[:-1, :-1], xy[1:, :-1], xy[1:, 1:], xy[:-1, 1:]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def scaled_jacobian(xy: np.ndarray) -> np.ndarray:
    """``(NX, NY)`` — min over the four corners of the normalised corner Jacobian.

    ``1`` = rectangle, ``0`` = degenerate corner, ``< 0`` = inverted.
    """
    p00, p10, p11, p01 = _corners(xy)
    quad = [p00, p10, p11, p01]
    out = np.full(p00.shape[:2], np.inf)
    for k in range(4):
        prev = quad[(k - 1) % 4]
        cur = quad[k]
        nxt = quad[(k + 1) % 4]
        e1 = nxt - cur
        e2 = prev - cur
        n1 = np.linalg.norm(e1, axis=-1)
        n2 = np.linalg.norm(e2, axis=-1)
        denom = np.maximum(n1 * n2, 1e-300)
        out = np.minimum(out, _cross(e1, e2) / denom)
    # orient so a consistently-wound grid scores positive
    return out * np.sign(np.median(out[np.isfinite(out)])) if out.size else out


def aspect_ratio(xy: np.ndarray) -> np.ndarray:
    """``(NX, NY)`` — longest edge over shortest edge. ``1`` is square."""
    p00, p10, p11, p01 = _corners(xy)
    e = np.stack([
        np.linalg.norm(p10 - p00, axis=-1),
        np.linalg.norm(p11 - p10, axis=-1),
        np.linalg.norm(p01 - p11, axis=-1),
        np.linalg.norm(p00 - p01, axis=-1),
    ], axis=-1)
    return e.max(axis=-1) / np.maximum(e.min(axis=-1), 1e-300)


def skew(xy: np.ndarray) -> np.ndarray:
    """``(NX, NY)`` — worst corner-angle deviation from 90°, scaled to ``[0, 1]``.

    ``0`` is perfectly orthogonal, ``1`` a fully collapsed corner. Separates shear
    from stretch: a sheared cell can have perfectly reasonable edge lengths.
    """
    p00, p10, p11, p01 = _corners(xy)
    quad = [p00, p10, p11, p01]
    worst = np.zeros(p00.shape[:2])
    for k in range(4):
        prev = quad[(k - 1) % 4]
        cur = quad[k]
        nxt = quad[(k + 1) % 4]
        e1 = nxt - cur
        e2 = prev - cur
        c = np.sum(e1 * e2, axis=-1) / np.maximum(
            np.linalg.norm(e1, axis=-1) * np.linalg.norm(e2, axis=-1), 1e-300
        )
        ang = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        worst = np.maximum(worst, np.abs(ang - 90.0) / 90.0)
    return worst


def non_orthogonality(xy: np.ndarray) -> np.ndarray:
    """Angle in DEGREES between the cell-centre line and the shared face normal.

    Returned as a flat array over all interior column faces (i- and j-directed).
    This is the measure a two-point flux approximation cares about: TPFA assumes
    the centre-to-centre line is parallel to the face normal, and the error grows
    with this angle. Above ~40-60 degrees most reservoir simulators are visibly
    inaccurate.
    """
    p00, p10, p11, p01 = _corners(xy)
    centre = (p00 + p10 + p11 + p01) / 4.0

    out = []
    # i-directed faces: shared edge between column (i,j) and (i+1,j)
    if centre.shape[0] > 1:
        edge = xy[1:-1, 1:] - xy[1:-1, :-1]                 # the shared edge
        nrm = np.stack([edge[..., 1], -edge[..., 0]], axis=-1)
        d = centre[1:, :] - centre[:-1, :]
        out.append(_angle(d, nrm))
    # j-directed faces
    if centre.shape[1] > 1:
        edge = xy[1:, 1:-1] - xy[:-1, 1:-1]
        nrm = np.stack([edge[..., 1], -edge[..., 0]], axis=-1)
        d = centre[:, 1:] - centre[:, :-1]
        out.append(_angle(d, nrm))
    return np.concatenate([o.ravel() for o in out]) if out else np.empty(0)


def _angle(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    dn = np.linalg.norm(d, axis=-1) * np.linalg.norm(n, axis=-1)
    c = np.abs(np.sum(d * n, axis=-1)) / np.maximum(dn, 1e-300)
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


@dataclass
class QualityReport:
    """Summary of cell quality over one grid."""

    sj_min: float
    sj_mean: float
    sj_below_0_3: int
    aspect_max: float
    aspect_p95: float
    skew_max: float
    skew_mean: float
    ortho_max: float
    ortho_p95: float
    ortho_mean: float
    n_cells: int

    def line(self) -> str:
        return (
            f"SJ min {self.sj_min:5.2f} mean {self.sj_mean:4.2f} "
            f"(<0.3: {self.sj_below_0_3:3d}) · "
            f"AR p95 {self.aspect_p95:5.1f} max {self.aspect_max:6.1f} · "
            f"skew mean {self.skew_mean:4.2f} · "
            f"nonorth mean {self.ortho_mean:4.1f}° p95 {self.ortho_p95:4.1f}° "
            f"max {self.ortho_max:4.1f}°"
        )


def cell_quality(lattice: PillarLattice, *, plane: str = "top") -> QualityReport:
    """Compute every metric over one endpoint plane and summarise."""
    xy = lattice.top if plane == "top" else lattice.bot
    sj = scaled_jacobian(xy)
    ar = aspect_ratio(xy)
    sk = skew(xy)
    no = non_orthogonality(xy)
    return QualityReport(
        sj_min=float(sj.min()),
        sj_mean=float(sj.mean()),
        sj_below_0_3=int((sj < 0.3).sum()),
        aspect_max=float(ar.max()),
        aspect_p95=float(np.percentile(ar, 95)),
        skew_max=float(sk.max()),
        skew_mean=float(sk.mean()),
        ortho_max=float(no.max()) if no.size else 0.0,
        ortho_p95=float(np.percentile(no, 95)) if no.size else 0.0,
        ortho_mean=float(no.mean()) if no.size else 0.0,
        n_cells=int(sj.size),
    )
