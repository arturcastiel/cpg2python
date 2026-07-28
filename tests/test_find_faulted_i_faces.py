"""Integration tests for ``_find_faulted_i_faces``.

First PR that wires together  (find_connections) + 
(compute_face_geometry) + an earlier faulted mask helper into a real
fault-face producer.

Restrictions tested (per  spec):
  - all-active stacks only
  - 4-corner fault polygons only
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import from_grdecl, read_grdecl
from cpg2unstructured._errors import NonMatchingZError
from cpg2unstructured.unstructured import (
    _find_faulted_i_faces,
)

# ── matching-Z fixtures: should return empty Faces ──────────────────


def test_simple_2x2x2_no_fault_returns_empty(fixtures):
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces, new_nodes = _find_faulted_i_faces(cube)
    assert faces.nodes.shape == (0, 4)
    assert faces.neighbours.shape == (0, 2)
    assert faces.direction.shape == (0,)
    np.testing.assert_array_equal(new_nodes, cube.nodes)  # unchanged


def test_simple_3x3x2_no_fault_returns_empty(fixtures):
    g = read_grdecl(fixtures / "simple_3x3x2.grdecl")
    cube = from_grdecl(g)
    faces, _ = _find_faulted_i_faces(cube)
    assert faces.nodes.shape == (0, 4)


def test_with_include_no_fault_returns_empty(fixtures):
    g = read_grdecl(fixtures / "with_include" / "main.grdecl")
    cube = from_grdecl(g)
    faces, _ = _find_faulted_i_faces(cube)
    assert faces.nodes.shape == (0, 4)


# ── simple_faulted_2x2x2: 3 fault faces expected ────────────────────


def test_simple_faulted_2x2x2_produces_3_fault_faces(fixtures):
    """The §6b worked example: 2 A-cells × 2 B-cells (half-cell
    offset between i=0 and i=1) produces 3 connections per j-stack.

    The fixture has 2 j-stacks (j=0 and j=1), so total = 6 fault faces."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces, new_nodes = _find_faulted_i_faces(cube)

    # 2 j-stacks × 3 connections per stack = 6 fault faces total
    assert faces.nodes.shape == (6, 4), (
        f"expected 6 fault faces (3 per j-stack), got {faces.nodes.shape}"
    )
    assert faces.neighbours.shape == (6, 2)
    assert faces.direction.shape == (6,)
    # All i-direction
    np.testing.assert_array_equal(faces.direction, np.zeros(6, dtype=np.int8))

    # No intersection nodes appended (4-corner polygons use only pillar
    # corners —  restriction enforced).
    assert new_nodes.shape == cube.nodes.shape


def test_simple_faulted_2x2x2_neighbour_pairs_are_valid_active_ids(fixtures):
    """All neighbour IDs should be valid active-cell IDs (0..n_active-1)."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    n_active = int(cube.actnum_3d.sum())
    faces, _ = _find_faulted_i_faces(cube)
    nbs = faces.neighbours
    # All IDs in valid range (or -1 for boundary, but this fixture is all-active)
    assert nbs.min() >= -1
    assert nbs.max() < n_active


def test_simple_faulted_2x2x2_nodes_sorted_per_face(fixtures):
    """Faces.nodes convention: each row is sorted ascending."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces, _ = _find_faulted_i_faces(cube)
    for row in faces.nodes:
        assert (row[:-1] <= row[1:]).all(), f"row not sorted: {row}"


def test_simple_faulted_2x2x2_uses_only_existing_node_ids(fixtures):
    """All face corner IDs reference existing nodes in cube.nodes."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces, new_nodes = _find_faulted_i_faces(cube)
    assert faces.nodes.min() >= 0
    assert faces.nodes.max() < new_nodes.shape[0]


def test_simple_faulted_2x2x2_face_normal_a_to_b_via_neighbours(fixtures):
    """Sanity: neighbour col 0 = smaller active-cell id (the
    `_move_minus_one_to_col1` convention shared with matching-Z)."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces, _ = _find_faulted_i_faces(cube)
    for nb in faces.neighbours:
        if nb[1] != -1:
            assert nb[0] <= nb[1], f"col 0 should be <= col 1 for inner: {nb}"


# ──  restriction enforcement: inactive cells in fault stack ────


def test_inactive_cell_in_fault_stack_raises(tmp_path, fixtures):
    """Take the simple_faulted_2x2x2 deck and deactivate cell (0,0,0)
    — should raise NonMatchingZError with the v2-only hint."""
    src = (fixtures / "simple_faulted_2x2x2.grdecl").read_text()
    # Replace "8*1 /" with explicit ACTNUM and deactivate cell (0,0,0).
    # Cell F-order index for (0,0,0) = 0. So first ACTNUM value flips to 0.
    src = src.replace(
        "ACTNUM\n  8*1 /",
        "ACTNUM\n  0 1 1 1 1 1 1 1 /",
    )
    p = tmp_path / "faulted_with_inactive.grdecl"
    p.write_text(src)
    g = read_grdecl(p)
    cube = from_grdecl(g)
    with pytest.raises(NonMatchingZError,
                       match="inactive cells in faulted i-stack"):
        _find_faulted_i_faces(cube)


# ── build_topology still raises ─


def test_build_topology_succeeds_on_faulted_fixture(fixtures):
    """Post- build_topology succeeds end-to-end on the i-fault
    fixture (the build_topology guard was relaxed to only catch
    k-direction faults)."""
    from cpg2unstructured import build_topology
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert grid is not None
    assert grid.n_cells == 8
