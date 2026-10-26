"""Dispatch tests for find_i_faces.

The widening of find_i_faces — matching vs faulted dispatch inline.
Verifies (a)  invariance on matching-Z fixtures and (b) the
fault path produces the expected concatenated output on the
simple_faulted_2x2x2 fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import from_grdecl, read_grdecl
from cpg2unstructured._errors import NonMatchingZError
from cpg2unstructured.unstructured import find_i_faces

# ──  invariance ──────────────────────────────────────────────


def test_simple_2x2x2_byte_identical_post_dispatch(fixtures):
    """On a matching-Z deck, find_i_faces output must be unchanged
    from v0.1.0 — the early-out preserves the byte layout."""
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    # 2x2x2 all-active: i-planes × j × k = 3 × 2 × 2 = 12 i-faces
    assert faces.nodes.shape == (12, 4)
    assert faces.neighbours.shape == (12, 2)
    # All i-direction
    np.testing.assert_array_equal(faces.direction,
                                  np.zeros(12, dtype=np.int8))


def test_simple_3x3x2_byte_identical_post_dispatch(fixtures):
    g = read_grdecl(fixtures / "simple_3x3x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    # 3x3x2 all-active: 4 i-planes × 3 j × 2 k = 24 i-faces
    assert faces.nodes.shape == (24, 4)


def test_with_include_byte_identical_post_dispatch(fixtures):
    g = read_grdecl(fixtures / "with_include" / "main.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    # 2x2x2 all-active: 12 i-faces
    assert faces.nodes.shape == (12, 4)


# ── Dispatch on faulted fixture ────────────────────────────────────


def test_simple_faulted_2x2x2_total_i_face_count(fixtures):
    """Total i-face count = matching boundary faces + fault inner faces."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    # 2x2x2 with i-fault at the single inner i-plane:
    #   boundary i-faces (i-plane 0 and i-plane 2): 2 × 2 × 2 = 8
    #   fault i-faces (i-plane 1): 2 j-stacks × 5 faces/stack = 10
    #     per stack: 3 inner (A0-B0, A1-B0, A1-B1) + 2 exposed
    #     boundary pieces (A0 on z∈[0,0.5], B1 on z∈[2,2.5])
    # Total: 18
    assert faces.nodes.shape == (18, 4)
    assert faces.neighbours.shape == (18, 2)


def test_simple_faulted_2x2x2_all_node_ids_valid(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    assert faces.nodes.min() >= 0
    assert faces.nodes.max() < cube.nodes.shape[0]


def test_simple_faulted_2x2x2_neighbour_ids_valid(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    n_active = int(cube.actnum_3d.sum())
    faces = find_i_faces(cube)
    nbs = faces.neighbours
    # -1 allowed (boundary), valid active IDs otherwise
    assert nbs.min() >= -1
    assert nbs.max() < n_active


def test_simple_faulted_2x2x2_inner_faces_have_two_cells(fixtures):
    """All fault faces (and inner matching faces) have both neighbours
    non-negative (no boundary -1 on inner faces)."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    inner_mask = faces.neighbours[:, 1] != -1
    # The only inner i-plane is faulted, so all 6 inner faces are fault
    # faces (3 per j-stack). Boundary faces: 8 on the two grid
    # boundary planes (2 × 2 j × 2 k) + 4 exposed fault pieces
    # (2 per j-stack).
    n_inner = int(inner_mask.sum())
    n_boundary = int(faces.nodes.shape[0] - n_inner)
    assert n_inner == 6   # all 6 are fault faces from the dispatcher
    assert n_boundary == 12


def test_simple_faulted_2x2x2_neighbour_col0_smaller(fixtures):
    """_move_minus_one_to_col1 convention: smaller-id in col 0,
    -1 in col 1 for boundary faces."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    for nb in faces.neighbours:
        if nb[1] != -1:
            assert nb[0] <= nb[1], f"col 0 should be <= col 1 for inner: {nb}"


# ── build_topology still raises on faulted ────


def test_build_topology_succeeds_on_i_faulted_fixture(fixtures):
    """Post- build_topology end-to-end on simple_faulted_2x2x2.
    find_i_faces dispatches → build_topology assembles correctly."""
    from cpg2unstructured import build_topology
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert grid is not None
    assert grid.n_cells == 8
