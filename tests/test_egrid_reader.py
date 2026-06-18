"""Tests for ``cpg2unstructured.read_egrid``.

Covers invariants T1, T2, T4, T5, T7 from
``my-axon/dev-projects/fault-projections/phases/4-egrid-input/02-prs.md``.
T3 (streaming short-circuit) lives in ``test_egrid_streaming.py``.
T6 (hypothesis property test) lives in ``test_egrid_property.py``.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured import (
    CornerPointGrid,
    Grdecl,
    read_egrid,
)
from cpg2unstructured._errors import (
    GrdeclConsistencyError,
    GrdeclSyntaxError,
)

DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------- T7 ---

def test_returns_corner_point_grid_with_egrid_source():
    g = read_egrid(DATA / "cart_2x2x2.EGRID")
    assert isinstance(g, CornerPointGrid)
    assert isinstance(g, Grdecl)  # alias
    assert g.source == "egrid"
    assert g.cart_dims == (2, 2, 2)


# ---------------------------------------------------------------------- T1 ---

def test_coord_shape_and_dtype_default_float32():
    g = read_egrid(DATA / "cart_2x2x2.EGRID")
    nx, ny, nz = g.cart_dims
    assert g.coord.shape == ((nx + 1) * (ny + 1), 6)
    assert g.coord.dtype == np.float32
    assert g.zcorn.shape == (8 * nx * ny * nz,)
    assert g.zcorn.dtype == np.float32


def test_coord_zcorn_widen_with_dtype_kwarg():
    g = read_egrid(DATA / "cart_2x2x2.EGRID", dtype=np.float64)
    assert g.coord.dtype == np.float64
    assert g.zcorn.dtype == np.float64


def test_unit_cartesian_pillar_geometry():
    g = read_egrid(DATA / "cart_2x2x2.EGRID")
    # 3x3 pillars on the unit grid; first pillar at LOCAL origin.
    np.testing.assert_array_equal(g.coord[0], np.array([0, 0, 0, 0, 0, 2], dtype=np.float32))
    # Last pillar at (2, 2).
    np.testing.assert_array_equal(g.coord[-1], np.array([2, 2, 0, 2, 2, 2], dtype=np.float32))


# ---------------------------------------------------------------------- T2 ---

def test_mapaxes_applied_by_default():
    g = read_egrid(DATA / "cart_2x2x2_rotated.EGRID")
    assert g.mapaxes is not None
    assert g.mapaxes.shape == (6,)
    assert g.mapaxes_applied is True
    # LOCAL pillar 0 = (0, 0) → MAP origin (10, 20) per the fixture's MAPAXES.
    np.testing.assert_allclose(g.coord[0, :2], (10.0, 20.0), atol=1e-5)
    np.testing.assert_allclose(g.coord[0, 3:5], (10.0, 20.0), atol=1e-5)
    # LOCAL pillar at (1, 0) → MAP origin + 1 * x_hat where x_hat = (cos45, sin45).
    c = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)
    np.testing.assert_allclose(g.coord[1, :2], (10.0 + c, 20.0 + s), atol=1e-5)


def test_mapaxes_opt_out_warns_and_keeps_raw_coords():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = read_egrid(DATA / "cart_2x2x2_rotated.EGRID", apply_mapaxes=False)
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("MAPAXES" in str(w.message) for w in runtime_warnings), (
        "Expected a RuntimeWarning naming MAPAXES on opt-out"
    )
    assert g.mapaxes is not None
    assert g.mapaxes_applied is False
    # Raw LOCAL pillar 0 = (0, 0).
    np.testing.assert_array_equal(g.coord[0, :2], (0.0, 0.0))


def test_no_mapaxes_no_warning_no_application():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = read_egrid(DATA / "cart_2x2x2.EGRID")
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert g.mapaxes is None
    assert g.mapaxes_applied is False


# ---------------------------------------------------------------------- T5 ---

def test_actnum_optional_defaults_all_active():
    g = read_egrid(DATA / "cart_2x2x2_no_actnum.EGRID")
    assert g.actnum.dtype == np.int8
    assert int(g.actnum.sum()) == g.actnum.size  # all active


def test_actnum_present_is_int8():
    g = read_egrid(DATA / "cart_2x2x2.EGRID")
    assert g.actnum.dtype == np.int8
    assert g.actnum.size == 8
    assert int(g.actnum.sum()) == 8


# ---------------------------------------------------------------------- T7 ---

def test_missing_gridhead_raises(tmp_path):
    import resfo

    p = tmp_path / "no_gridhead.EGRID"
    resfo.write(p, [("COORD   ", np.zeros(6 * 4, dtype=np.float32))])
    with pytest.raises(GrdeclSyntaxError, match="GRIDHEAD"):
        read_egrid(p)


def test_non_corner_point_gridhead_raises(tmp_path):
    import resfo

    head = np.zeros(100, dtype=np.int32)
    head[0] = 2  # not corner-point
    head[1:4] = [2, 2, 2]
    p = tmp_path / "wrong_type.EGRID"
    resfo.write(
        p,
        [
            ("GRIDHEAD", head),
            ("COORD   ", np.zeros(9 * 6, dtype=np.float32)),
            ("ZCORN   ", np.zeros(64, dtype=np.float32)),
            ("ENDGRID ", np.zeros(0, dtype=np.int32)),
        ],
    )
    with pytest.raises(GrdeclConsistencyError, match="grid_type=2"):
        read_egrid(p)


def test_inconsistent_coord_size_raises(tmp_path):
    import resfo

    head = np.zeros(100, dtype=np.int32)
    head[0] = 1
    head[1:4] = [2, 2, 2]  # expects 9 pillars
    p = tmp_path / "bad_coord.EGRID"
    resfo.write(
        p,
        [
            ("GRIDHEAD", head),
            ("COORD   ", np.zeros(7 * 6, dtype=np.float32)),  # wrong
            ("ZCORN   ", np.zeros(64, dtype=np.float32)),
            ("ENDGRID ", np.zeros(0, dtype=np.int32)),
        ],
    )
    with pytest.raises(GrdeclConsistencyError, match="COORD has 7 pillars"):
        read_egrid(p)
