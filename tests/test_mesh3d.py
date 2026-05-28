"""Tests for the full-3D mesh graph + nearest-node."""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import Mesh3DGraph, build_mesh_graph_3d, mesh_edges


def _node_at(grid, x, y, z) -> int:
    hit = np.nonzero(
        (grid.nodes[:, 0] == x) & (grid.nodes[:, 1] == y) & (grid.nodes[:, 2] == z)
    )[0]
    assert hit.size == 1
    return int(hit[0])


def test_node_and_edge_counts(mesh_3x3x2):
    g = build_mesh_graph_3d(mesh_3x3x2)
    assert isinstance(g, Mesh3DGraph)
    # 4x4x3 node lattice.
    assert g.n_nodes == 48
    assert mesh_3x3x2.nodes.shape[0] == 48
    # i:3*4*3=36, j:4*3*3=36, k:4*4*2=32 -> 104 unique edges.
    assert g.edges.shape == (104, 2)
    assert np.allclose(g.edge_len, 1.0)


def test_edges_sorted_unique_no_degenerate(mesh_3x3x2):
    e = mesh_edges(mesh_3x3x2)
    assert (e[:, 0] < e[:, 1]).all()
    assert np.unique(e, axis=0).shape[0] == e.shape[0]


def test_nearest_node_3d(mesh_3x3x2):
    g = build_mesh_graph_3d(mesh_3x3x2)
    idx = g.nearest_node([2.1, 0.9, 0.1])
    assert idx.shape == (1,)
    assert idx[0] == _node_at(mesh_3x3x2, 2, 1, 0)
    batch = g.nearest_node(np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 2.0]]))
    assert batch.tolist() == [
        _node_at(mesh_3x3x2, 0, 0, 0),
        _node_at(mesh_3x3x2, 3, 3, 2),
    ]


def test_shortest_path_manhattan(mesh_3x3x2):
    g = build_mesh_graph_3d(mesh_3x3x2)
    a = _node_at(mesh_3x3x2, 0, 0, 0)
    b = _node_at(mesh_3x3x2, 3, 3, 2)
    dist, path = g.shortest_path(a, b)
    assert dist == pytest.approx(8.0)  # 3 + 3 + 2 unit edges
    assert path[0] == a and path[-1] == b
    for u, v in zip(path[:-1], path[1:]):
        assert np.linalg.norm(mesh_3x3x2.nodes[u] - mesh_3x3x2.nodes[v]) == pytest.approx(1.0)
