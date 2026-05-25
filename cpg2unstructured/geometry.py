"""Face + cell geometry for the unstructured mesh.

 scope: centroids, areas, volumes via the divergence theorem.
Consumes the topology produced by `unstructured.build_topology`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpg2unstructured.unstructured import Faces

__all__ = ["FaceGeometry", "CellGeometry", "face_geometry", "cell_geometry"]


@dataclass(frozen=True)
class FaceGeometry:
    """Per-face geometric quantities.

    centroids — (Nfaces, 3) float64
    areas     — (Nfaces,)   float64
    normals   — (Nfaces, 3) float64; unit length where area > 0, zero
                vector for degenerate faces. Oriented per
                `faces.neighbours[:, 0] → faces.neighbours[:, 1]` for
                inner faces and outward-from-cell-0 for boundary faces.
    """
    centroids: np.ndarray
    areas: np.ndarray
    normals: np.ndarray


@dataclass(frozen=True)
class CellGeometry:
    """Per-cell geometric quantities.

    centroids — (Ncells, 3) float64
    volumes   — (Ncells,)   float64
    """
    centroids: np.ndarray
    volumes: np.ndarray


def face_geometry(
    nodes: np.ndarray,
    faces: Faces,
    cell_corners: np.ndarray,
) -> FaceGeometry:
    """Compute (centroids, areas, normals) for every face.

    Normals are oriented so that for each face `f` with cell
    `faces.neighbours[f, 0]` on the "low" side, the normal points
    AWAY from that cell (i.e. from cell0 toward cell1 for inner
    faces, outward from cell0 for boundary faces).

    `cell_corners` (mesh.cell_corners from `_TopoMesh`) provides a
    cell-centroid estimate (mean of the 8 corner node positions) used
    ONLY for the orientation flip. See .
    """
    v = nodes[faces.nodes]                              # (Nfaces, 4, 3)
    centroids = v.mean(axis=1)
    N = v.shape[0]

    # Faces.nodes is sorted by node ID, NOT in
    # spatial cyclic order. Triangle-fan from vertex 0 only works for
    # cyclic-ordered quads. For arbitrary vertex order on a planar
    # convex quad, area = (1/2) |d1 × d2| where d1, d2 are the two
    # diagonals. Diagonals are the 2 longest of the 6 pairwise
    # distances.
    PAIRS = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    diff = v[:, PAIRS[:, 0]] - v[:, PAIRS[:, 1]]        # (N, 6, 3)
    dists_sq = (diff ** 2).sum(axis=-1)                 # (N, 6)
    top2 = np.argsort(dists_sq, axis=1)[:, -2:]         # (N, 2)
    diag1 = PAIRS[top2[:, 0]]                           # (N, 2)
    diag2 = PAIRS[top2[:, 1]]
    row = np.arange(N)
    d1 = v[row, diag1[:, 1]] - v[row, diag1[:, 0]]      # (N, 3)
    d2 = v[row, diag2[:, 1]] - v[row, diag2[:, 0]]      # (N, 3)
    cross_d = np.cross(d1, d2)                          # (N, 3)
    norm_len = np.linalg.norm(cross_d, axis=1)
    areas = 0.5 * norm_len
    safe_len = np.where(norm_len < 1e-15, 1.0, norm_len)   #  guard
    normals = cross_d / safe_len[:, None]
    # Degenerate faces (norm_len < 1e-15) get a zero normal vector.
    normals = np.where(norm_len[:, None] < 1e-15, 0.0, normals)

    # Orient normals per  using cell-corners mean as centroid estimate.
    cell_centroid_est = nodes[cell_corners].mean(axis=1)   # (Ncells, 3)
    cell0_ids = faces.neighbours[:, 0]
    # Every face has at least one active cell on col 0 ( invariant:
    # boundary faces have -1 in col 1, never col 0 — _move_minus_one_to_col1).
    cell0_centroid = cell_centroid_est[cell0_ids]          # (Nfaces, 3)
    away_dir = centroids - cell0_centroid
    dots = (away_dir * normals).sum(axis=1)
    flip = dots < 0
    if flip.any():
        normals = normals.copy()
        normals[flip] = -normals[flip]

    return FaceGeometry(centroids=centroids, areas=areas, normals=normals)


def cell_geometry(
    nodes: np.ndarray,
    faces: Faces,
    cell_faces: np.ndarray,
    fg: FaceGeometry,
) -> CellGeometry:
    """Compute (centroids, volumes) per cell via the divergence theorem.

    Volume:    V_c = (1/3) Σ_f sign · (centroid_f · normal_f) · area_f
    Centroid:  c   = (1/(4V)) Σ_f sign · centroid_f · (centroid_f · normal_f) · area_f
    sign = +1 iff `faces.neighbours[f, 0] == c`, else -1.
    """
    n_cells = cell_faces.shape[0]
    volumes = np.zeros(n_cells, dtype=np.float64)
    centroids = np.zeros((n_cells, 3), dtype=np.float64)

    fc = fg.centroids
    fn = fg.normals
    fa = fg.areas
    nb = faces.neighbours

    for c in range(n_cells):
        face_ids = cell_faces[c]
        v_sum = 0.0
        c_sum = np.zeros(3, dtype=np.float64)
        for f in face_ids:
            f = int(f)
            sign = 1.0 if nb[f, 0] == c else -1.0
            cn = float(fc[f] @ fn[f])
            contrib = sign * cn * fa[f]
            v_sum += contrib
            c_sum += contrib * fc[f]
        v = v_sum / 3.0
        volumes[c] = v
        if v != 0.0:
            centroids[c] = c_sum / (4.0 * v)

    return CellGeometry(centroids=centroids, volumes=volumes)
