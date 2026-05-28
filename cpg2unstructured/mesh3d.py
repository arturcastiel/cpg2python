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

from cpg2unstructured.graph import shortest_path_fast
from cpg2unstructured.unstructured import UnstructuredGrid

__all__ = ["LOCAL_EDGES", "mesh_edges", "Mesh3DGraph", "build_mesh_graph_3d"]

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
