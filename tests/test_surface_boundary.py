"""Tests for the ∂₂ surface-boundary (rim) helpers."""
from __future__ import annotations

import numpy as np

from cpg2unstructured import (
    boundary_edges,
    face_edges,
    planar_minimal_surface,
    surface_boundary_loops,
)

BIG = np.array([[-9, -9], [9, -9], [9, 9], [-9, 9]], dtype=float)


def _surface_face_ids(mesh_3x3x2):
    surf = planar_minimal_surface(
        mesh_3x3x2, np.array([1.0, 1, 1]), np.array([1.0, 0, 0]),
        BIG, np.array([0.0, 1, 0]), np.array([0.0, 0, 1]))
    return surf.face_ids


def test_rim_edges_are_all_surface_edges(mesh_3x3x2):
    fids = _surface_face_ids(mesh_3x3x2)
    rim = boundary_edges(mesh_3x3x2, fids)
    surf_edges = set(face_edges(mesh_3x3x2)[fids].ravel().tolist())
    # every rim edge belongs to a selected face — the key guarantee
    assert rim.size > 0
    assert set(rim.tolist()).issubset(surf_edges)


def test_rim_edges_have_odd_incidence(mesh_3x3x2):
    fids = _surface_face_ids(mesh_3x3x2)
    counts = np.bincount(face_edges(mesh_3x3x2)[fids].ravel())
    rim = set(boundary_edges(mesh_3x3x2, fids).tolist())
    for e, c in enumerate(counts):
        if c > 0:
            assert (e in rim) == (c % 2 == 1)


def test_boundary_loops_closed(mesh_3x3x2):
    fids = _surface_face_ids(mesh_3x3x2)
    loops = surface_boundary_loops(mesh_3x3x2, fids)
    assert len(loops) >= 1
    for lp in loops:
        assert lp[0] == lp[-1]            # closed
    # one rectangular disk -> a single rim loop
    assert len(loops) == 1


def test_empty_face_set(mesh_3x3x2):
    assert boundary_edges(mesh_3x3x2, np.empty(0, np.int64)).size == 0
    assert surface_boundary_loops(mesh_3x3x2, np.empty(0, np.int64)) == []
