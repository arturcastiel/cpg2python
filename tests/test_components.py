"""Connected-component labelling (PR-13 of phase 2-faults).

`UnstructuredGrid.component[c]` is the 0-based component id of cell `c`,
computed over the inner-face + NNC adjacency. Labels are assigned in
order of first cell id, so a fully-connected grid is all zeros.
"""
from __future__ import annotations

import numpy as np

from cpg2unstructured import build_topology, from_grdecl, read_grdecl


def test_single_component_is_all_zero(fixtures):
    grid = build_topology(from_grdecl(read_grdecl(fixtures / "simple_2x2x2.grdecl")))
    assert grid.component.shape == (8,)
    assert np.array_equal(grid.component, np.zeros(8, dtype=np.int64))


def test_pinch_stays_single_component(fixtures):
    """An NNC keeps the pinch-bridged survivors in ONE component."""
    grid = build_topology(from_grdecl(read_grdecl(fixtures / "pinched_1x1x3.grdecl")))
    assert grid.component.shape == (2,)
    assert np.array_equal(grid.component, np.zeros(2, dtype=np.int64))
    assert grid.component.max() == 0          # single component


def test_hole_yields_two_components(fixtures):
    """A genuine ACTNUM=0 gap (no NNC, no shared face) disconnects the
    column into two components."""
    grid = build_topology(from_grdecl(read_grdecl(fixtures / "hole_1x1x3.grdecl")))
    assert grid.component.shape == (2,)
    assert int(grid.component.max()) + 1 == 2     # two components
    # cell 0 is always in component 0
    assert grid.component[0] == 0
    assert set(grid.component.tolist()) == {0, 1}
