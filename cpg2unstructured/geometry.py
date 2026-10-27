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

    ``faces.nodes`` may have ``M > 4`` columns (fault polygons with
    edge-intersection corners, see :class:`Faces`). A face with 5 or
    more distinct corners is put in cyclic order around its best-fit
    plane and fan-triangulated: exact area and area-weighted centroid
    for a planar convex polygon. Faces with at most 4 distinct corners
    get the same treatment as in a 4-column ``faces.nodes``.
    """
    fn = faces.nodes
    if fn.shape[1] == 4:
        centroids, areas, cross_d = _quad_geometry(nodes, fn)
    else:
        centroids, areas, cross_d = _padded_face_geometry(nodes, fn)

    norm_len = 2.0 * areas
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


def _quad_geometry(
    nodes: np.ndarray, fn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(centroids, areas, area vectors x2)`` of ``(n, 4)`` sorted faces."""
    v = nodes[fn]                                       # (Nfaces, 4, 3)
    centroids = v.mean(axis=1)
    N = v.shape[0]

    # A triangular fault face is stored with one node repeated; its
    # centroid is the mean of its 3 distinct corners. Sorted ids make
    # repeats adjacent.
    distinct = np.ones(fn.shape, dtype=bool)
    distinct[:, 1:] = fn[:, 1:] != fn[:, :-1]
    tri = ~distinct.all(axis=1)
    if tri.any():
        w = distinct[tri][..., None]
        centroids[tri] = (v[tri] * w).sum(axis=1) / w.sum(axis=1)

    # Faces.nodes is sorted by node ID, NOT in spatial cyclic order, so
    # the diagonals are unknown. For a planar quad, area = (1/2)|d1 × d2|
    # over its two diagonals. Each of the 3 ways to split the 4 vertices
    # into two pairs is a candidate; the true diagonals give the largest
    # |d1 × d2| (the other two pairings trace self-crossing bow-ties).
    # This also gives the right area for a triangle with a repeated node.
    # (Picking the 2 longest pairwise distances, as before, fails on
    # trapezoids: a long parallel side can beat the short diagonal.)
    DIAGS = ((0, 2, 1, 3), (0, 3, 1, 2), (0, 1, 2, 3))
    crosses = np.stack(
        [np.cross(v[:, b] - v[:, a], v[:, d] - v[:, c]) for a, b, c, d in DIAGS],
        axis=1,
    )                                                   # (N, 3, 3)
    lens = np.linalg.norm(crosses, axis=-1)             # (N, 3)
    best = lens.argmax(axis=1)
    row = np.arange(N)
    cross_d = crosses[row, best]                        # (N, 3)
    areas = 0.5 * lens[row, best]
    return centroids, areas, cross_d


def _padded_face_geometry(
    nodes: np.ndarray, fn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`_quad_geometry` for ``(n, M > 4)`` sorted, padded faces."""
    n, m = fn.shape
    distinct = np.ones(fn.shape, dtype=bool)
    distinct[:, 1:] = fn[:, 1:] != fn[:, :-1]
    count = distinct.sum(axis=1)
    # Distinct ids first, in sorted order, padded with the largest id.
    rank = np.cumsum(distinct, axis=1) - 1
    compact = np.repeat(fn[:, -1:], m, axis=1)
    compact[np.repeat(np.arange(n), m), rank.ravel()] = fn.ravel()

    centroids = np.zeros((n, 3), dtype=np.float64)
    areas = np.zeros(n, dtype=np.float64)
    cross_d = np.zeros((n, 3), dtype=np.float64)
    small = count <= 4
    if small.any():
        centroids[small], areas[small], cross_d[small] = _quad_geometry(
            nodes, compact[small, :4],
        )
    for k in np.unique(count[~small]):
        sel = count == k
        centroids[sel], areas[sel], cross_d[sel] = _polygon_geometry(
            nodes[compact[sel, :k]],
        )
    return centroids, areas, cross_d


def _polygon_geometry(
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid, area and area vector x2 of ``(n, k, 3)`` polygon corners
    given in any order: sorted by angle around the vertex mean in the
    best-fit plane, then fan-triangulated."""
    mean = v.mean(axis=1, keepdims=True)
    d = v - mean
    _, _, vt = np.linalg.svd(d)                         # rows: plane axes
    ang = np.arctan2(
        (d * vt[:, None, 1]).sum(axis=-1), (d * vt[:, None, 0]).sum(axis=-1),
    )
    order = np.argsort(ang, axis=1)
    p = np.take_along_axis(v, order[..., None], axis=1)
    tri = np.cross(p[:, 1:-1] - p[:, :1], p[:, 2:] - p[:, :1])   # (n, k-2, 3)
    cross_d = tri.sum(axis=1)
    norm_len = np.linalg.norm(cross_d, axis=1)
    unit = cross_d / np.where(norm_len < 1e-15, 1.0, norm_len)[:, None]
    w = (tri * unit[:, None]).sum(axis=-1)              # 2x triangle areas
    tri_c = (p[:, :1] + p[:, 1:-1] + p[:, 2:]) / 3.0
    w_sum = w.sum(axis=1)
    flat = w_sum <= 0.0                                 # degenerate polygon
    weighted = (tri_c * w[..., None]).sum(axis=1)
    centroids = np.where(
        flat[:, None], mean[:, 0], weighted / np.where(flat, 1.0, w_sum)[:, None],
    )
    return centroids, 0.5 * norm_len, cross_d


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
    Negative entries in `cell_faces` (the padding of cells with more
    than one face on a fault side) are skipped.
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
            if f < 0:
                continue
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
