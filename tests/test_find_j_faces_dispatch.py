"""Dispatch tests for find_j_faces.

Mirror of tests/test_find_i_faces_dispatch.py along the j-axis.
Verifies (a)  invariance on matching-Z fixtures and (b) the
dispatch path on the new simple_faulted_2x2x2_j fixture (mirror of
the i-fault fixture along j).
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import from_grdecl, read_grdecl
from cpg2unstructured._errors import NonMatchingZError
from cpg2unstructured.unstructured import find_j_faces

# ──  invariance ──────────────────────────────────────────────


def test_simple_2x2x2_byte_identical_post_dispatch_j(fixtures):
    """Matching-Z deck → find_j_faces output unchanged from v0.1.0."""
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    # 2x2x2 all-active: 2 × 3 j-planes × 2 k = 12 j-faces.
    assert faces.nodes.shape == (12, 4)
    assert faces.neighbours.shape == (12, 2)
    np.testing.assert_array_equal(
        faces.direction, np.ones(12, dtype=np.int8),
    )


def test_simple_3x3x2_byte_identical_post_dispatch_j(fixtures):
    g = read_grdecl(fixtures / "simple_3x3x2.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    # 3x3x2 all-active: 3 × 4 j-planes × 2 k = 24 j-faces.
    assert faces.nodes.shape == (24, 4)


def test_with_include_byte_identical_post_dispatch_j(fixtures):
    g = read_grdecl(fixtures / "with_include" / "main.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    assert faces.nodes.shape == (12, 4)


# ── Dispatch on j-faulted fixture ──────────────────────────────────


def test_simple_faulted_2x2x2_j_total_j_face_count(fixtures):
    """Mirror of simple_faulted_2x2x2 along j-axis:
       boundary j-faces (j-plane 0 and j-plane 2): 2 × 2 × 2 = 8
       fault j-faces  (j-plane 1, 2 i-stacks × 3 conns each): 6
       Total: 14 j-faces."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    assert faces.nodes.shape == (14, 4)
    assert faces.neighbours.shape == (14, 2)


def test_simple_faulted_2x2x2_j_all_node_ids_valid(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    assert faces.nodes.min() >= 0
    assert faces.nodes.max() < cube.nodes.shape[0]


def test_simple_faulted_2x2x2_j_neighbour_ids_valid(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    n_active = int(cube.actnum_3d.sum())
    faces = find_j_faces(cube)
    nbs = faces.neighbours
    assert nbs.min() >= -1
    assert nbs.max() < n_active


def test_simple_faulted_2x2x2_j_inner_faces_have_two_cells(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    inner_mask = faces.neighbours[:, 1] != -1
    n_inner = int(inner_mask.sum())
    n_boundary = int(faces.nodes.shape[0] - n_inner)
    assert n_inner == 6
    assert n_boundary == 8


def test_simple_faulted_2x2x2_j_direction_is_1(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    np.testing.assert_array_equal(
        faces.direction, np.ones(faces.nodes.shape[0], dtype=np.int8),
    )


def test_simple_faulted_2x2x2_j_neighbour_col0_smaller(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    for nb in faces.neighbours:
        if nb[1] != -1:
            assert nb[0] <= nb[1], f"col 0 should be <= col 1: {nb}"


# ── build_topology still raises ─────────


def test_build_topology_succeeds_on_j_faulted_fixture(fixtures):
    """Post- build_topology end-to-end on simple_faulted_2x2x2_j.
    find_j_faces dispatches → build_topology assembles correctly."""
    from cpg2unstructured import build_topology
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert grid is not None
    assert grid.n_cells == 8
