"""PINCH detection + pinched-cell drop (PR-11 of phase 2-faults).

A pinched-out cell has zero vertical thickness (top z-slot == bottom
z-slot on all pillars). `from_grdecl` folds such cells out of the active
set, so they drop from the topology exactly like ACTNUM=0 cells. PR-12
will bridge the cells above/below a pinch with an NNC.
"""
from __future__ import annotations

import numpy as np

from cpg2unstructured import build_topology, from_grdecl, read_grdecl
from cpg2unstructured.cpg import _pinched_cell_mask


def test_pinched_cell_mask_detects_middle(fixtures):
    g = read_grdecl(fixtures / "pinched_1x1x3.grdecl")
    mask = _pinched_cell_mask(g)            # (1, 1, 3) bool
    assert mask.shape == (1, 1, 3)
    # Only the middle cell (k=1) is collapsed.
    assert mask[0, 0, 1]
    assert not mask[0, 0, 0]
    assert not mask[0, 0, 2]
    assert mask.sum() == 1


def test_from_grdecl_drops_pinched_cell(fixtures):
    g = read_grdecl(fixtures / "pinched_1x1x3.grdecl")
    cube = from_grdecl(g)
    # ACTNUM was all-1, but the pinched k=1 cell is now inactive.
    assert cube.actnum_3d[0, 0, 0]
    assert not cube.actnum_3d[0, 0, 1]
    assert cube.actnum_3d[0, 0, 2]
    assert cube.actnum_3d.sum() == 2


def test_build_topology_skips_pinched_cell(fixtures):
    g = read_grdecl(fixtures / "pinched_1x1x3.grdecl")
    grid = build_topology(from_grdecl(g))
    # 3 cells in the deck, 1 pinched → 2 active.
    assert grid.n_cells == 2
    ijk = grid.original_ijk
    assert ijk.shape == (2, 3)
    # The two survivors are k=0 and k=2 (k=1 dropped).
    np.testing.assert_array_equal(
        ijk[np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))],
        np.array([[0, 0, 0], [0, 0, 2]]),
    )


def test_unpinched_deck_is_unchanged(fixtures):
    """Invariance: an ordinary unit-thickness deck has no pinched cells
    and its active count is untouched."""
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    assert not _pinched_cell_mask(g).any()
    grid = build_topology(from_grdecl(g))
    assert grid.n_cells == 8


def test_pinch_tol_threshold(fixtures):
    """A sub-tolerance-thin (not exactly zero) cell is caught only when
    pinch_tol is raised above its thickness."""
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    # Unit cells: tol=0 catches nothing; tol≥1 would catch everything.
    assert not _pinched_cell_mask(g, tol=0.0).any()
    assert _pinched_cell_mask(g, tol=1.0).all()
