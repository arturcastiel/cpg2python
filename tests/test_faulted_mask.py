"""Faulted-pillar mask helpers (PR-1 of phase 2-faults).

Verifies that the new ``_faulted_pillar_mask_{i,j,k}`` predicates
return the right shapes + content on both matching-Z and faulted
fixtures, AND that ``find_*_faces`` still raise ``NonMatchingZError``
on faulted decks (behaviour-preserving — the widening to dispatch
inline lands in PR-6/PR-7/PR-8).
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import build_topology, from_grdecl, read_grdecl
from cpg2unstructured._errors import NonMatchingZError
from cpg2unstructured.unstructured import (
    _faulted_pillar_mask_i,
    _faulted_pillar_mask_j,
    _faulted_pillar_mask_k,
    find_i_faces,
    find_j_faces,
    find_k_faces,
)


# ── matching-Z decks: every mask should be empty ────────────────────


def test_simple_2x2x2_no_fault_all_directions(fixtures):
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    cube = from_grdecl(g)
    m_i = _faulted_pillar_mask_i(cube.P, cube.actnum_3d)
    m_j = _faulted_pillar_mask_j(cube.P, cube.actnum_3d)
    m_k = _faulted_pillar_mask_k(cube.P, cube.actnum_3d)
    assert m_i.shape == (1, 2, 2) and not m_i.any()
    assert m_j.shape == (2, 1, 2) and not m_j.any()
    assert m_k.shape == (2, 2, 1) and not m_k.any()


def test_simple_3x3x2_no_fault_all_directions(fixtures):
    g = read_grdecl(fixtures / "simple_3x3x2.grdecl")
    cube = from_grdecl(g)
    assert not _faulted_pillar_mask_i(cube.P, cube.actnum_3d).any()
    assert not _faulted_pillar_mask_j(cube.P, cube.actnum_3d).any()
    assert not _faulted_pillar_mask_k(cube.P, cube.actnum_3d).any()


def test_with_include_no_fault(fixtures):
    g = read_grdecl(fixtures / "with_include" / "main.grdecl")
    cube = from_grdecl(g)
    assert not _faulted_pillar_mask_i(cube.P, cube.actnum_3d).any()
    assert not _faulted_pillar_mask_j(cube.P, cube.actnum_3d).any()
    assert not _faulted_pillar_mask_k(cube.P, cube.actnum_3d).any()


# ── faulted fixture: mask pinpoints the fault stack ─────────────────


def test_simple_faulted_2x2x2_i_fault(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    m = _faulted_pillar_mask_i(cube.P, cube.actnum_3d)
    # Fault between i=0 and i=1 → row i=0 of (NX-1, NY, NZ) mask
    # should be ALL True (every active i-face in that stack is faulted).
    assert m.shape == (1, 2, 2)
    assert m.all(), "every active i-face should be flagged faulted"


def test_simple_faulted_2x2x2_no_j_fault(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    m = _faulted_pillar_mask_j(cube.P, cube.actnum_3d)
    # j-faces compare cells at the SAME i. Offset is per-i-column,
    # so j-neighbours within a column align → no j-fault.
    assert m.shape == (2, 1, 2)
    assert not m.any()


def test_simple_faulted_2x2x2_no_k_fault(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    m = _faulted_pillar_mask_k(cube.P, cube.actnum_3d)
    # k-faces compare cells at the SAME i, j. Offset doesn't break
    # k-direction matching within a column.
    assert m.shape == (2, 2, 1)
    assert not m.any()


# ── behaviour-preserving: find_*_faces still raise on faulted deck ─


def test_find_i_faces_dispatches_on_faulted_fixture(fixtures):
    """After PR-6 (phase 2-faults), find_i_faces dispatches matching
    vs faulted inline and SUCCEEDS on simple_faulted_2x2x2 (was raising
    before PR-6). The output Faces record contains matching-Z faces
    from non-faulted stacks + fault-faces from faulted stacks."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    # simple_faulted_2x2x2 is 2x2x2 with both j-stacks faulted (single
    # i-pair, both j-rows). All inner i-faces in faulted stacks become
    # fault-faces; boundary i-faces stay as matching-Z.
    # · matching: 2 boundary i-planes × 2 j × 2 k = 8 boundary faces
    # · fault:    2 j-stacks × 3 connections = 6 fault faces
    # Total: 14 i-faces.
    assert faces.nodes.shape == (14, 4)
    assert faces.neighbours.shape == (14, 2)
    np.testing.assert_array_equal(faces.direction,
                                  np.zeros(14, dtype=np.int8))


def test_find_j_faces_no_raise_on_faulted_fixture(fixtures):
    """j-direction is unfaulted on this fixture; find_j_faces should
    complete normally even though the deck has an i-fault."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    assert faces is not None
    assert faces.nodes.shape[1] == 4


def test_find_k_faces_no_raise_on_faulted_fixture(fixtures):
    """k-direction is unfaulted on this fixture (offset is per-column)."""
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_k_faces(cube)
    assert faces is not None


def test_build_topology_still_raises_on_faulted_fixture(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    # Message contains both "not yet supported" (new context) and
    # "v2 only" (preserved for the Tier A regression test).
    with pytest.raises(NonMatchingZError, match="not yet supported"):
        build_topology(cube)


# ── Tier A regression: matching-Z deck still works ───────────────────


def test_find_i_faces_no_raise_on_matching_fixture(fixtures):
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    assert faces is not None
    assert faces.nodes.shape[1] == 4
