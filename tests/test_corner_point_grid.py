"""Tests for the format-agnostic ``CornerPointGrid`` and the ``Grdecl`` alias."""
from __future__ import annotations

import numpy as np

from cpg2unstructured import CornerPointGrid, Grdecl


def test_grdecl_is_alias_of_corner_point_grid():
    assert Grdecl is CornerPointGrid


def test_default_fields():
    g = CornerPointGrid(
        cart_dims=(2, 3, 4),
        coord=np.zeros((12, 6), dtype=np.float32),
        zcorn=np.zeros(8 * 2 * 3 * 4, dtype=np.float32),
        actnum=np.ones(2 * 3 * 4, dtype=np.int8),
    )
    assert g.mapaxes is None
    assert g.mapaxes_applied is False
    assert g.props == {}
    assert g.unrecognized == []
    assert g.source == "grdecl"
    assert g.n_cells == 24
    assert g.n_pillars == 12


def test_frozen():
    g = CornerPointGrid(
        cart_dims=(1, 1, 1),
        coord=np.zeros((4, 6), dtype=np.float32),
        zcorn=np.zeros(8, dtype=np.float32),
        actnum=np.ones(1, dtype=np.int8),
    )
    import dataclasses

    with np.testing.assert_raises(dataclasses.FrozenInstanceError):
        g.source = "egrid"  # type: ignore[misc]
