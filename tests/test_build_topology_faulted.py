"""End-to-end tests for ``build_topology`` on faulted decks
.

The headline outcome of  ``build_topology`` no longer raises on
i- or j-direction faulted decks. The i/j fault dispatch from
/ + the removal of the top-level guard in ``build_topology``
mean a faulted deck flows end-to-end through the topology builder.

k-direction faults still raise — k-fault dispatch is deferred to a
follow-up because the 2-pillar sweep-line algorithm doesn't apply
directly to the 4-pillar geometry of k-faces.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import (
    UnstructuredGrid,
    build_topology,
    from_grdecl,
    read_grdecl,
)
from cpg2unstructured._errors import NonMatchingZError


# ── headline: build_topology on i-faulted deck ─────────────────────


def test_build_topology_on_simple_faulted_2x2x2_i(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert isinstance(grid, UnstructuredGrid)
    assert grid.n_cells == 8     # all-active 2x2x2
    # Sanity on outputs:
    assert grid.cell_faces.shape == (8, 6)
    assert grid.faces.nodes.shape[1] == 4
    assert grid.faces.neighbours.shape[1] == 2
    # Direction labels: 0 = i, 1 = j, 2 = k
    assert set(np.unique(grid.faces.direction).tolist()) <= {0, 1, 2}


def test_build_topology_on_simple_faulted_2x2x2_j(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert grid.n_cells == 8
    assert grid.cell_faces.shape == (8, 6)


# ──  regression: 3x3x2 multi-column i-fault ────────────────────


def test_build_topology_on_simple_faulted_3x3x2(fixtures):
    """3x3x2 multi-column i-fault: +0.6 Z offset on the i=2 column.
    Exercises the find_i_faces dispatch on a multi-j-row grid
    (the 2x2x2 fixtures only have 2 j-rows; this has 3).

    The fault plane spans 3 j-rows; the +0.6 offset makes the upper
    i=1 cell straddle both i=2 cells, so each stack yields 3 conns
    → 9 faulted i-faces."""
    g = read_grdecl(fixtures / "simple_faulted_3x3x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert isinstance(grid, UnstructuredGrid)
    assert grid.n_cells == 18         # all-active 3x3x2
    assert grid.cell_faces.shape == (18, 6)
    assert grid.faces.nodes.shape[1] == 4
    assert grid.faces.neighbours.shape[1] == 2


def test_simple_faulted_3x3x2_i_face_dispatch_count(fixtures):
    """find_i_faces called directly on the 3x3x2 fault deck produces
    the expected mix of matching + fault faces.

        i-plane 0  (boundary)        : 3 j × 2 k = 6
        i-plane 1  (matching, i=0..1): 3 j × 2 k = 6
        i-plane 2  (FAULTED, i=1..2) : 3 j-stacks × 3 conns = 9
        i-plane 3  (boundary)        : 3 j × 2 k = 6

        Total i-faces                : 27

        (the +0.6 offset makes the upper i=1 cell overlap BOTH i=2
         cells, so each faulted stack yields 3 connections, not 2.)
    """
    from cpg2unstructured.unstructured import find_i_faces
    g = read_grdecl(fixtures / "simple_faulted_3x3x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    assert faces.nodes.shape == (27, 4)
    # All direction=0 (i-faces)
    np.testing.assert_array_equal(
        faces.direction, np.zeros(27, dtype=np.int8),
    )
    # All neighbours in valid range
    nbs = faces.neighbours
    assert nbs.min() >= -1
    assert nbs.max() < 18


def test_simple_faulted_3x3x2_neighbour_convention(fixtures):
    """`_move_minus_one_to_col1` invariant: inner faces have smaller
    id in col 0; boundary faces have -1 in col 1."""
    g = read_grdecl(fixtures / "simple_faulted_3x3x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    for nb in grid.faces.neighbours:
        if nb[1] == -1:
            assert nb[0] != -1, f"boundary face should have valid col 0: {nb}"
        else:
            assert nb[0] <= nb[1], f"inner: col 0 should be ≤ col 1: {nb}"


# ── still raises on k-faults ───────────────────────────────────────


def test_build_topology_raises_on_k_fault(tmp_path, fixtures):
    """Synthetic k-fault: simple_2x2x2 deck with one ZCORN value
    perturbed at the k=0 ↔ k=1 boundary (z-slot 1 on cell (0,0,0)
    set to 1.5 instead of 1)."""
    src = (fixtures / "simple_2x2x2.grdecl").read_text()
    # Original ZCORN block:
    #   ZCORN
    #     16*0   16*1
    #     16*1   16*2 /
    # The k=0-bot row is the 16*1 in the first ZCORN line. Replace
    # that with one explicit value of 1.5 + 15*1 to break matching-Z
    # between k=0 and k=1 at slot (i_slot=0, j_slot=0, k_slot=1).
    src = src.replace(
        "16*0   16*1\n  16*1   16*2 /",
        "16*0   1.5 15*1\n  16*1   16*2 /",
    )
    p = tmp_path / "k_fault.grdecl"
    p.write_text(src)
    g = read_grdecl(p)
    cube = from_grdecl(g)
    with pytest.raises(NonMatchingZError, match="k-direction"):
        build_topology(cube)


# ──  regression: 3x3x2 multi-column j-fault (mirror of ) ───


def test_build_topology_on_simple_faulted_3x3x2_jfault(fixtures):
    """3x3x2 multi-column j-fault: +0.6 Z offset on the j=2 row.
    The j-direction analogue of the  i-fault fixture — exercises
    find_j_faces dispatch on a multi-column grid (3 i-columns)."""
    g = read_grdecl(fixtures / "simple_faulted_3x3x2_jfault.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert isinstance(grid, UnstructuredGrid)
    assert grid.n_cells == 18         # all-active 3x3x2
    assert grid.cell_faces.shape == (18, 6)
    assert grid.faces.nodes.shape[1] == 4
    assert grid.faces.neighbours.shape[1] == 2


def test_simple_faulted_3x3x2_jfault_j_face_dispatch_count(fixtures):
    """find_j_faces on the 3x3x2 j-fault deck — symmetric to 

        j-plane 0  (boundary)        : 3 i × 2 k = 6
        j-plane 1  (matching, j=0..1): 3 i × 2 k = 6
        j-plane 2  (FAULTED, j=1..2) : 3 i-stacks × 3 conns = 9
        j-plane 3  (boundary)        : 3 i × 2 k = 6

        Total j-faces                : 27

        (the +0.6 offset makes the upper j=1 cell overlap BOTH j=2
         cells, so each faulted stack yields 3 connections, not 2.)
    """
    from cpg2unstructured.unstructured import find_j_faces
    g = read_grdecl(fixtures / "simple_faulted_3x3x2_jfault.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    assert faces.nodes.shape == (27, 4)
    # All direction=1 (j-faces)
    np.testing.assert_array_equal(
        faces.direction, np.ones(27, dtype=np.int8),
    )
    # All neighbours in valid range
    nbs = faces.neighbours
    assert nbs.min() >= -1
    assert nbs.max() < 18
