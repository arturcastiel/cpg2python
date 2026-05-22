import numpy as np
import pytest

from cpg2unstructured import (
    cell_adjacency,
    cell_geometry,
    face_geometry,
    shortest_path,
)


def test_edge_count_2x2x2(mesh_2x2x2):
    adj = cell_adjacency(mesh_2x2x2, weights="unit")
    # 8 cells; 12 inner faces × 2 directed entries each = 24.
    assert len(adj) == 8
    n_edges_directed = sum(len(a) for a in adj)
    assert n_edges_directed == 24


def test_corner_to_corner_path_unit(mesh_2x2x2):
    adj = cell_adjacency(mesh_2x2x2, weights="unit")
    # Eclipse-natural F-order on all-active 2x2x2:
    # cell (0,0,0)=id 0, cell (1,1,1)=id 7.
    dist, path = shortest_path(adj, 0, 7)
    assert dist == 3.0
    assert len(path) == 4
    assert path[0] == 0
    assert path[-1] == 7


def test_distance_weight_unit_cube(mesh_2x2x2):
    # Real PR-8 centroids — each unit cube centred at (0.5+i, 0.5+j, 0.5+k).
    fg = face_geometry(mesh_2x2x2.nodes, mesh_2x2x2.faces,
                       mesh_2x2x2.cell_corners)
    cg = cell_geometry(mesh_2x2x2.nodes, mesh_2x2x2.faces,
                       mesh_2x2x2.cell_faces, fg)
    adj = cell_adjacency(mesh_2x2x2, weights="distance",
                         cell_centroids=cg.centroids)
    dist, _ = shortest_path(adj, 0, 7)
    # 3 unit hops along edges of the cube lattice → 3.0.
    np.testing.assert_allclose(dist, 3.0, atol=1e-12)


def test_distance_weight_requires_centroids(mesh_2x2x2):
    with pytest.raises(ValueError, match="cell_centroids"):
        cell_adjacency(mesh_2x2x2, weights="distance")


def test_unknown_weights_mode_raises(mesh_2x2x2):
    # D-PR10-4 defence: typo like "Unit" should NOT silently work.
    with pytest.raises(ValueError, match="unknown weights mode"):
        cell_adjacency(mesh_2x2x2, weights="Unit")   # type: ignore[arg-type]


def test_weights_none_equals_unit(mesh_2x2x2):
    a_unit = cell_adjacency(mesh_2x2x2, weights="unit")
    a_none = cell_adjacency(mesh_2x2x2, weights="none")
    # Same adjacency, same weights (1.0 everywhere).
    assert len(a_unit) == len(a_none)
    for ru, rn in zip(a_unit, a_none):
        assert sorted(ru) == sorted(rn)


def test_n_inner_faces_match_undirected_edges(mesh_2x2x2):
    adj = cell_adjacency(mesh_2x2x2)
    nb = mesh_2x2x2.faces.neighbours
    n_inner = int(((nb[:, 0] != -1) & (nb[:, 1] != -1)).sum())
    n_edges_undirected = sum(len(a) for a in adj) // 2
    assert n_edges_undirected == n_inner


def test_adjacency_is_symmetric(mesh_2x2x2):
    adj = cell_adjacency(mesh_2x2x2)
    edges = set()
    for u, neigh in enumerate(adj):
        for v, _ in neigh:
            edges.add((u, v))
    for (u, v) in edges:
        assert (v, u) in edges


def test_unreachable_raises(mesh_2x2x2):
    adj = cell_adjacency(mesh_2x2x2)
    # Artificially disconnect cell 7 from the rest.
    adj[7] = []
    for a in range(7):
        adj[a] = [(b, w) for b, w in adj[a] if b != 7]
    with pytest.raises(ValueError, match="no path"):
        shortest_path(adj, 0, 7)


def test_path_endpoints_included(mesh_2x2x2):
    adj = cell_adjacency(mesh_2x2x2, weights="unit")
    _, path = shortest_path(adj, 0, 3)
    assert path[0] == 0
    assert path[-1] == 3
