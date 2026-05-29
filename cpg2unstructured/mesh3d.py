"""Full-3D mesh graph + nearest-node (for fault-surface projection).

Unlike the 2.5D plan-view collapse in :mod:`planview`, this builds the real 3D
edge set of the unstructured mesh: every hexahedral cell contributes its 12
edges (derived from the known ``_LOCAL_CORNERS`` ordering of ``cell_corners`` —
no geometry or cyclic reconstruction needed), deduplicated. From that we expose
a weighted node-edge graph (edge length) and a 3D KD-tree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from cpg2unstructured.geometry import cell_geometry, face_geometry
from cpg2unstructured.graph import shortest_path_fast
from cpg2unstructured.unstructured import UnstructuredGrid

__all__ = [
    "LOCAL_EDGES",
    "mesh_edges",
    "Mesh3DGraph",
    "build_mesh_graph_3d",
    "face_edges",
    "oriented_face_edges",
    "face_areas",
    "face_centroids",
    "cell_centroids",
]

# Hexahedron edges as local-corner index pairs, per `_LOCAL_CORNERS`:
#   (0,0,0)(1,0,0)(0,1,0)(1,1,0)(0,0,1)(1,0,1)(0,1,1)(1,1,1)
# An edge connects corners differing in exactly one of i/j/k.
LOCAL_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),   # i-edges
    (0, 2), (1, 3), (4, 6), (5, 7),   # j-edges
    (0, 4), (1, 5), (2, 6), (3, 7),   # k-edges
)


def mesh_edges(grid: UnstructuredGrid) -> np.ndarray:
    """Unique mesh edges as (E, 2) int64 node-id pairs (sorted per row).

    Degenerate edges (collapsed/pinched corners where both endpoints are the
    same node) are dropped.
    """
    cc = grid.cell_corners
    cols_a = [a for a, _ in LOCAL_EDGES]
    cols_b = [b for _, b in LOCAL_EDGES]
    a = cc[:, cols_a].reshape(-1)
    b = cc[:, cols_b].reshape(-1)
    pairs = np.sort(np.stack([a, b], axis=1), axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0).astype(np.int64, copy=False)


@dataclass(frozen=True)
class Mesh3DGraph:
    """Weighted 3D node-edge graph + KD-tree over the mesh nodes."""

    nodes: np.ndarray          # (Nn, 3) — grid.nodes
    edges: np.ndarray          # (E, 2) int64
    edge_len: np.ndarray       # (E,) float64
    csr: sp.csr_array          # (Nn, Nn) symmetric, weight = edge length
    kdtree: cKDTree            # over nodes (3D)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    def nearest_node(self, points: np.ndarray) -> np.ndarray:
        """Nearest mesh-node id for each 3D query point. Returns 1-D int64."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        _, idx = self.kdtree.query(pts)
        return np.atleast_1d(idx).astype(np.int64, copy=False)

    def shortest_path(self, src: int, dst: int) -> tuple[float, list[int]]:
        """Shortest node-to-node path; reuses the scipy Dijkstra."""
        return shortest_path_fast(self.csr, int(src), int(dst))


_FACE_COMBOS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# Cyclic-order recovery for oriented ∂₂. ``faces.nodes`` is sorted by node id,
# NOT in spatial cyclic order (see geometry.py). Among a quad's 6 node-pairs the
# 4 sides are mesh edges and the 2 diagonals are not. Corner 0's diagonal partner
# is the local position whose pair with 0 is NOT an edge; the boundary cycle is
# then [0, side, opp, side]. Indexed by that diagonal partner (1, 2, or 3):
_CYCLE_BY_OPP = np.array(
    [
        [0, 1, 2, 3],   # opp=0 — placeholder, never used
        [0, 2, 1, 3],   # opp=1 — diagonal 0–1
        [0, 1, 2, 3],   # opp=2 — diagonal 0–2
        [0, 1, 3, 2],   # opp=3 — diagonal 0–3
    ]
)


def _lookup_edge_ids(
    edges: np.ndarray, keys: np.ndarray, nn: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map encoded node-pair keys (lo*nn + hi) to ids into ``edges``.

    Returns ``(ids, found)`` with the same shape as ``keys``. Positions whose key
    is not a mesh edge get the sentinel id ``edges.shape[0]`` and ``found`` False.
    """
    edge_key = edges[:, 0].astype(np.int64) * nn + edges[:, 1]
    order = np.argsort(edge_key)
    sorted_key = edge_key[order]
    pos = np.clip(np.searchsorted(sorted_key, keys), 0, sorted_key.size - 1)
    found = sorted_key[pos] == keys
    ids = np.where(found, order[pos], edges.shape[0])
    return ids, found


def face_edges(grid: UnstructuredGrid, edges: np.ndarray | None = None) -> np.ndarray:
    """Each face's 4 boundary edges as ids into the unique-edge list (∂₂).

    A quad face has 6 node-pairs: its 4 sides (which ARE mesh edges) and its 2
    diagonals (which are not). We keep the 4 that appear in ``edges``. Result is
    (Nf, 4) int64, edge ids ascending per row.
    """
    if edges is None:
        edges = mesh_edges(grid)
    fn = grid.faces.nodes
    nn = grid.nodes.shape[0]

    a = np.stack([fn[:, i] for i, _ in _FACE_COMBOS], axis=1).astype(np.int64)
    b = np.stack([fn[:, j] for _, j in _FACE_COMBOS], axis=1).astype(np.int64)
    cand_key = np.minimum(a, b) * nn + np.maximum(a, b)   # (Nf, 6)

    eid, found = _lookup_edge_ids(edges, cand_key, nn)
    if not (found.sum(axis=1) == 4).all():
        bad = int((found.sum(axis=1) != 4).sum())
        raise ValueError(f"{bad} face(s) did not resolve to exactly 4 edges")

    eid.sort(axis=1)
    return eid[:, :4].astype(np.int64, copy=False)


def oriented_face_edges(
    grid: UnstructuredGrid, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Signed ∂₂: each face's 4 boundary edges in CYCLIC order, with orientation.

    Returns ``(edge_ids, signs)``, both (Nf, 4) and column-aligned. For face
    ``f``, walking ``edge_ids[f]`` traverses the face boundary as a closed loop;
    ``signs[f, k]`` is +1 when that edge is followed in ascending node order
    (lo→hi) and −1 when followed hi→lo. Hence ``sign·(lo→hi)`` is the actual
    directed boundary edge, and the four signed edges sum to a 1-cycle
    (∂₁∂₂ = 0) for every face.

    Unlike :func:`face_edges` (columns sorted by id, unsigned), the columns here
    follow the boundary cycle so the signs are meaningful. The per-face cycle
    direction is intrinsic (derived from the node-id ordering); making it
    globally coherent across a surface is orientation propagation, done by the
    caller — see the assembly step that unions per-patch fills.
    """
    if edges is None:
        edges = mesh_edges(grid)
    fn = grid.faces.nodes
    nn = grid.nodes.shape[0]

    # Which of the 6 node-pairs are real edges → locate corner 0's diagonal.
    a = np.stack([fn[:, i] for i, _ in _FACE_COMBOS], axis=1).astype(np.int64)
    b = np.stack([fn[:, j] for _, j in _FACE_COMBOS], axis=1).astype(np.int64)
    cand_key = np.minimum(a, b) * nn + np.maximum(a, b)
    _, found = _lookup_edge_ids(edges, cand_key, nn)
    if not (found.sum(axis=1) == 4).all():
        bad = int((found.sum(axis=1) != 4).sum())
        raise ValueError(f"{bad} face(s) did not resolve to exactly 4 edges")

    # Combos 0,1,2 are (0,1),(0,2),(0,3) → partners 1,2,3. Corner 0 lies on
    # exactly one diagonal, so exactly one of these is not an edge; argmin picks
    # that single False per row.
    opp_pos = np.array([1, 2, 3])[np.argmin(found[:, [0, 1, 2]], axis=1)]
    cyc = _CYCLE_BY_OPP[opp_pos]                          # (Nf, 4) local indices
    ordered = np.take_along_axis(fn, cyc, axis=1).astype(np.int64)  # cyclic node ids

    src = ordered
    dst = np.roll(ordered, -1, axis=1)
    key = np.minimum(src, dst) * nn + np.maximum(src, dst)
    eid, found2 = _lookup_edge_ids(edges, key, nn)
    if not found2.all():                                  # cyclic sides are all edges
        raise ValueError("oriented_face_edges: a face side was not a mesh edge")
    signs = np.where(src < dst, 1, -1).astype(np.int8)
    return eid.astype(np.int64, copy=False), signs


def _face_geometry(grid: UnstructuredGrid):
    return face_geometry(grid.nodes, grid.faces, grid.cell_corners)


def face_areas(grid: UnstructuredGrid) -> np.ndarray:
    """(Nf,) float64 — area of each cell-face."""
    return _face_geometry(grid).areas


def face_centroids(grid: UnstructuredGrid) -> np.ndarray:
    """(Nf, 3) float64 — centroid of each cell-face."""
    return _face_geometry(grid).centroids


def cell_centroids(grid: UnstructuredGrid) -> np.ndarray:
    """(Ncells, 3) float64 — centroid of each cell."""
    fg = _face_geometry(grid)
    return cell_geometry(grid.nodes, grid.faces, grid.cell_faces, fg).centroids


def build_mesh_graph_3d(grid: UnstructuredGrid) -> Mesh3DGraph:
    """Build the 3D node-edge graph + KD-tree from a grid."""
    edges = mesh_edges(grid)
    nodes = grid.nodes
    elen = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    n = nodes.shape[0]
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([elen, elen])
    csr = sp.csr_array((data, (rows, cols)), shape=(n, n))
    return Mesh3DGraph(
        nodes=nodes, edges=edges,
        edge_len=elen.astype(np.float64, copy=False),
        csr=csr, kdtree=cKDTree(nodes),
    )
