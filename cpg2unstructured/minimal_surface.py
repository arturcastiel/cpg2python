"""Planar minimal surface: the cell-faces approximating a planar fault disk.

For a CLOSED PLANAR fault polygon, the minimum-area, hole-free surface of
cell-faces is obtained directly (ADR-4): keep the faces whose two adjacent cells
lie on OPPOSITE sides of the fault plane (the "straddling" faces — the staircase
closest to the plane) and whose footprint falls inside the polygon. This is the
exact discrete minimal surface for a planar boundary; general (non-planar)
boundaries would need a min-cut solver, which is out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpg2unstructured.mesh3d import cell_centroids, face_areas, face_centroids
from cpg2unstructured.unstructured import UnstructuredGrid

__all__ = [
    "MinimalSurface",
    "fit_plane",
    "project_to_plane",
    "point_in_polygon",
    "planar_minimal_surface",
]


@dataclass(frozen=True)
class MinimalSurface:
    """Selected cell-faces approximating a planar fault.

    face_ids   (M,) int64 — ids into grid.faces
    area       float       — total face area
    cell_ijk   (M, 2, 3) int64 — the two adjacent cells' (i, j, k)
    direction  (M,) int8   — 0 = i-face (X), 1 = j-face (Y), 2 = k-face (Z)
    """

    face_ids: np.ndarray
    area: float
    cell_ijk: np.ndarray
    direction: np.ndarray


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit plane of (N,3) points → (p0, normal, u, v).

    p0 = centroid, normal = smallest-singular-value direction; (u, v) is an
    orthonormal in-plane basis (the two largest singular directions).
    """
    pts = np.asarray(points, dtype=np.float64)
    p0 = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - p0, full_matrices=False)
    u, v, normal = vt[0], vt[1], vt[2]
    return p0, normal, u, v


def project_to_plane(points: np.ndarray, p0: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project 3D points into the (u, v) plane frame → (N, 2)."""
    d = np.asarray(points, dtype=np.float64) - p0
    return np.column_stack([d @ u, d @ v])


def point_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized crossing-number test: which of pts (M,2) are inside poly (N,2)."""
    pts = np.atleast_2d(pts)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)
    n = poly.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        denom = yj - yi
        denom = np.where(denom == 0.0, np.finfo(float).eps, denom)
        x_at = (xj - xi) * (y - yi) / denom + xi
        inside ^= crosses & (x < x_at)
        j = i
    return inside


def planar_minimal_surface(
    grid: UnstructuredGrid,
    p0: np.ndarray,
    normal: np.ndarray,
    polygon_uv: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> MinimalSurface:
    """Select the straddling cell-faces inside the polygon footprint.

    p0, normal define the fault plane; (u, v) is its in-plane frame; polygon_uv
    is the fault polygon projected into that frame.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)

    sd = (cell_centroids(grid) - p0) @ normal           # signed distance per cell
    nb = grid.faces.neighbours
    lo, hi = nb[:, 0], nb[:, 1]
    interior = (lo >= 0) & (hi >= 0)

    prod = np.zeros(nb.shape[0], dtype=np.float64)
    prod[interior] = sd[lo[interior]] * sd[hi[interior]]
    straddle = interior & (prod < 0.0)                  # opposite sides, neither on-plane

    f2d = project_to_plane(face_centroids(grid), p0, u, v)
    inside = point_in_polygon(f2d, np.asarray(polygon_uv, dtype=np.float64))

    sel = np.nonzero(straddle & inside)[0]
    areas = face_areas(grid)[sel]
    cell_ijk = grid.original_ijk[nb[sel]]               # (M, 2, 3)
    direction = grid.faces.direction[sel]
    return MinimalSurface(
        face_ids=sel.astype(np.int64, copy=False),
        area=float(areas.sum()),
        cell_ijk=cell_ijk.astype(np.int64, copy=False),
        direction=direction.astype(np.int8, copy=False),
    )
