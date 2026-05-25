"""Disconnected-grid splitting via UnstructuredGrid.split().

`split()` materialises each connected component as an independent grid.
A single-component grid returns `[self]`. Node ids stay global; cells,
faces, nnc and props are re-indexed per sub-grid.
"""
from __future__ import annotations

import numpy as np

from cpg2unstructured import UnstructuredGrid, build_topology, from_grdecl, read_grdecl


def _grid(fixtures, name):
    return build_topology(from_grdecl(read_grdecl(fixtures / name)))


def test_split_single_component_returns_self(fixtures):
    grid = _grid(fixtures, "simple_2x2x2.grdecl")
    parts = grid.split()
    assert len(parts) == 1
    assert parts[0] is grid


def test_islands_build_two_components(fixtures):
    grid = _grid(fixtures, "islands_5x1x1.grdecl")
    assert grid.n_cells == 4                       # i=2 switched off
    assert int(grid.component.max()) + 1 == 2


def test_split_two_islands(fixtures):
    grid = _grid(fixtures, "islands_5x1x1.grdecl")
    parts = grid.split()
    assert len(parts) == 2
    assert all(isinstance(p, UnstructuredGrid) for p in parts)
    assert [p.n_cells for p in parts] == [2, 2]
    # Each sub-grid is itself a single component.
    for p in parts:
        assert np.array_equal(p.component, np.zeros(p.n_cells, dtype=np.int64))
    # Island A = original i {0,1}; island B = i {3,4} (label order).
    a_i = set(parts[0].original_ijk[:, 0].tolist())
    b_i = set(parts[1].original_ijk[:, 0].tolist())
    assert a_i == {0, 1}
    assert b_i == {3, 4}


def test_split_subgrids_are_structurally_valid(fixtures):
    grid = _grid(fixtures, "islands_5x1x1.grdecl")
    for p in grid.split():
        # every cell has all 6 face slots filled
        assert (p.cell_faces != -1).all()
        assert p.cell_faces.shape == (p.n_cells, 6)
        # face ids referenced are in range
        assert p.cell_faces.max() < p.faces.nodes.shape[0]
        # neighbour ids in [-1, n_cells)
        assert p.faces.neighbours.min() >= -1
        assert p.faces.neighbours.max() < p.n_cells
        # each island has an internal i-face (the 2 cells are adjacent)
        assert (p.faces.neighbours[:, 1] != -1).any()
