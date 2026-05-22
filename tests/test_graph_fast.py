"""Scipy-backed graph fast path (PR-11).

Skipped entirely if scipy is not installed (use the ``[fast]`` extra).
Verifies numerical agreement with the pure-numpy reference from PR-10.
"""
from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")  # noqa: F841 — gates the entire module

from cpg2unstructured import (
    cell_adjacency,
    cell_adjacency_csgraph,
    shortest_path,
    shortest_path_fast,
)
from cpg2unstructured import graph as _graph


def test_csr_shape_and_symmetric(mesh_2x2x2):
    csr = cell_adjacency_csgraph(mesh_2x2x2, weights="unit")
    n = mesh_2x2x2.cell_faces.shape[0]
    assert csr.shape == (n, n)
    diff = csr - csr.T
    diff.eliminate_zeros()
    assert diff.nnz == 0


def test_csr_directed_count_matches_slow(mesh_2x2x2):
    """nnz on the CSR equals the slow-path's directed-edge count."""
    adj = cell_adjacency(mesh_2x2x2, weights="unit")
    n_slow_directed = sum(len(a) for a in adj)
    csr = cell_adjacency_csgraph(mesh_2x2x2, weights="unit")
    assert csr.nnz == n_slow_directed


def test_unit_weight_corner_to_corner_2x2x2(mesh_2x2x2):
    """For the 2x2x2 corner-to-corner walk, unit-weight slow and fast
    paths must agree on cost and length. The 2x2x2 grid has SIX
    equal-cost 3-hop routes from cell 0 to cell 7; tie-breaking
    between heapq-Python and scipy's C impl is not guaranteed
    identical (Q-PR11-2 confirmed empirically — DEV-02). So we
    assert cost + length + endpoint validity, NOT path-list equality.
    """
    n = mesh_2x2x2.cell_faces.shape[0]
    adj = cell_adjacency(mesh_2x2x2, weights="unit")
    csr = cell_adjacency_csgraph(mesh_2x2x2, weights="unit")
    d_slow, p_slow = shortest_path(adj, 0, n - 1)
    d_fast, p_fast = shortest_path_fast(csr, 0, n - 1)
    assert d_slow == d_fast == 3.0
    assert len(p_slow) == len(p_fast) == 4
    assert p_slow[0] == p_fast[0] == 0
    assert p_slow[-1] == p_fast[-1] == n - 1
    # Both paths must be valid in the slow adjacency (every consecutive
    # pair is a neighbour with weight 1.0).
    slow_neighbours = [{nb for nb, _ in row} for row in adj]
    for path in (p_slow, p_fast):
        for u, v in zip(path[:-1], path[1:]):
            assert v in slow_neighbours[u], (
                f"invalid hop {u}→{v} in path {path}"
            )


def test_full_source_mode_2x2x2(mesh_2x2x2):
    csr = cell_adjacency_csgraph(mesh_2x2x2, weights="unit")
    distances, predecessors = shortest_path_fast(csr, 0, dst=None)
    assert distances.shape == (8,)
    assert predecessors.shape == (8,)
    assert distances[0] == 0.0
    assert distances[7] == 3.0


def test_src_equals_dst_returns_self(mesh_2x2x2):
    csr = cell_adjacency_csgraph(mesh_2x2x2, weights="unit")
    d, p = shortest_path_fast(csr, 0, 0)
    assert d == 0.0
    assert p == [0]


def test_distance_weights_agree_5x5x3(mesh_5x5x3_with_centroids):
    """Numerical agreement on a non-trivial grid.

    Tie-breaking between heapq-Python and scipy's C impl is not
    guaranteed identical (Q-PR11-2), so we assert on cost only.
    """
    mesh, centroids = mesh_5x5x3_with_centroids
    n = mesh.cell_faces.shape[0]
    adj = cell_adjacency(mesh, weights="distance", cell_centroids=centroids)
    csr = cell_adjacency_csgraph(mesh, weights="distance",
                                 cell_centroids=centroids)
    d_slow, _ = shortest_path(adj, 0, n - 1)
    d_fast, _ = shortest_path_fast(csr, 0, n - 1)
    np.testing.assert_allclose(d_fast, d_slow, rtol=0, atol=1e-12)


def test_unit_weights_agree_5x5x3(mesh_5x5x3_with_centroids):
    mesh, _ = mesh_5x5x3_with_centroids
    n = mesh.cell_faces.shape[0]
    adj = cell_adjacency(mesh, weights="unit")
    csr = cell_adjacency_csgraph(mesh, weights="unit")
    d_slow, _ = shortest_path(adj, 0, n - 1)
    d_fast, _ = shortest_path_fast(csr, 0, n - 1)
    assert d_slow == d_fast


def test_distance_requires_centroids(mesh_2x2x2):
    with pytest.raises(ValueError, match="cell_centroids"):
        cell_adjacency_csgraph(mesh_2x2x2, weights="distance")


def test_unknown_weights_mode_raises(mesh_2x2x2):
    with pytest.raises(ValueError, match="unknown weights mode"):
        cell_adjacency_csgraph(mesh_2x2x2, weights="Unit")  # type: ignore[arg-type]


def test_unreachable_raises_in_fast_path(mesh_2x2x2):
    csr = cell_adjacency_csgraph(mesh_2x2x2, weights="unit")
    csr_lil = csr.tolil()
    csr_lil[7, :] = 0
    csr_lil[:, 7] = 0
    csr2 = csr_lil.tocsr()
    csr2.eliminate_zeros()
    with pytest.raises(ValueError, match="no path"):
        shortest_path_fast(csr2, 0, 7)


def test_helpful_error_without_scipy(monkeypatch, mesh_2x2x2):
    """Even with scipy installed, simulate the import-error path."""
    monkeypatch.setattr(_graph, "_HAS_SCIPY", False)
    with pytest.raises(ImportError, match=r"\[fast\]"):
        cell_adjacency_csgraph(mesh_2x2x2, weights="unit")
    with pytest.raises(ImportError, match=r"\[fast\]"):
        shortest_path_fast(None, 0)
