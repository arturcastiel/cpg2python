"""Pillar lattice read directly from COORD, and the boundary masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured.grdecl_reader import read_grdecl
from cpg2unstructured.pillars import (
    apply_displacement,
    boundary_mask,
    local_spacing,
    pillar_lattice,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def simple():
    return read_grdecl(FIXTURES / "simple_3x3x2.grdecl")


def test_lattice_shape_matches_cart_dims(simple):
    lat = pillar_lattice(simple)
    nx, ny, _ = simple.cart_dims
    assert lat.shape == (nx + 1, ny + 1)
    assert lat.top.shape == (nx + 1, ny + 1, 2)
    assert lat.bot.shape == (nx + 1, ny + 1, 2)
    assert lat.n_pillars == (nx + 1) * (ny + 1)


def test_lattice_is_independent_of_nz(simple):
    """Pillar count must not depend on layer count — that is the whole point."""
    lat = pillar_lattice(simple)
    nx, ny, _ = simple.cart_dims
    assert lat.n_pillars == (nx + 1) * (ny + 1)


def test_indexing_is_i_then_j(simple):
    """coord is stored I-fastest; the lattice must expose [i, j]."""
    lat = pillar_lattice(simple)
    xs = lat.top[:, 0, 0]
    assert np.all(np.diff(xs) > 0), "x must increase along the i axis at fixed j"
    ys = lat.top[0, :, 1]
    assert np.all(np.diff(ys) > 0), "y must increase along the j axis at fixed i"


def test_cartesian_fixture_pillars_are_vertical(simple):
    assert pillar_lattice(simple).is_vertical().all()


def test_is_vertical_uses_an_absolute_tolerance(simple):
    """np.isclose's default rtol=1e-5 is fatal at UTM magnitudes.

    Shift the lattice into field coordinates and lean every pillar by 1 m. Under a
    RELATIVE tolerance that lean disappears into ~4.5 m of slack and the pillars are
    reported vertical. On Norne that turned 7 genuinely vertical pillars into 469.
    """
    from cpg2unstructured.corner_point import CornerPointGrid

    coord = np.asarray(simple.coord, float).reshape(-1, 6).copy()
    coord[:, [0, 3]] += 453_114.0          # UTM easting, Norne-sized
    coord[:, [1, 4]] += 7_319_921.0        # UTM northing
    coord[:, 3] += 1.0                     # lean each bottom by 1 m
    shifted = CornerPointGrid(
        cart_dims=simple.cart_dims, coord=coord,
        zcorn=simple.zcorn, actnum=simple.actnum,
    )
    assert not pillar_lattice(shifted).is_vertical().any()


def test_local_spacing_is_cell_sized(simple):
    lat = pillar_lattice(simple)
    h = local_spacing(lat)
    assert h.shape == lat.shape
    assert np.all(h > 0)
    # A regular fixture: every spacing equals the cell size.
    assert np.allclose(h, h.flat[0])


class TestBoundaryMask:
    def test_lattice_scope_is_the_bounding_box(self, simple):
        m = boundary_mask(simple, scope="lattice")
        nx, ny, _ = simple.cart_dims
        assert m[0].all() and m[-1].all() and m[:, 0].all() and m[:, -1].all()
        assert not m[1:-1, 1:-1].any()

    def test_all_active_grid_rim_matches_bounding_box(self, simple):
        """With every column active, the active rim IS the lattice edge."""
        assert np.array_equal(
            boundary_mask(simple, scope="active"),
            boundary_mask(simple, scope="lattice"),
        )

    def test_inactive_interior_column_creates_interior_rim(self, simple):
        """An inactive interior column must produce rim pillars INSIDE the box.

        Built by deactivating the centre column of a 3x3 grid rather than using a
        fixture: the small fixtures are 1-cell wide, so every pillar sits on the
        lattice edge and the two notions coincide vacuously.
        """
        from cpg2unstructured.corner_point import CornerPointGrid

        nx, ny, nz = simple.cart_dims
        actnum = np.asarray(simple.actnum).reshape(nz, ny, nx).transpose(2, 1, 0).copy()
        actnum[1, 1, :] = 0                      # centre column, all layers
        holed = CornerPointGrid(
            cart_dims=simple.cart_dims,
            coord=simple.coord,
            zcorn=simple.zcorn,
            actnum=actnum.transpose(2, 1, 0).ravel().astype(np.int8),
        )
        active = boundary_mask(holed, scope="active")
        lattice = boundary_mask(holed, scope="lattice")

        assert not np.array_equal(active, lattice), "an interior hole must change the rim"
        # the four pillars around the hole are interior yet on the active rim
        for i, j in ((1, 1), (2, 1), (1, 2), (2, 2)):
            assert active[i, j], f"pillar {(i, j)} borders the hole and must be rim"
            assert not lattice[i, j], f"pillar {(i, j)} is interior to the bounding box"

    def test_unknown_scope_raises(self, simple):
        with pytest.raises(ValueError, match="unknown scope"):
            boundary_mask(simple, scope="nonsense")


class TestApplyDisplacement:
    def test_rigid_move_shifts_both_endpoints(self, simple):
        lat = pillar_lattice(simple)
        d = np.zeros(lat.top.shape)
        d[..., 0] = 3.0
        moved = pillar_lattice(apply_displacement(simple, d))
        assert np.allclose(moved.top[..., 0], lat.top[..., 0] + 3.0)
        assert np.allclose(moved.bot[..., 0], lat.bot[..., 0] + 3.0)

    def test_rigid_move_preserves_slant(self, simple):
        lat = pillar_lattice(simple)
        d = np.zeros(lat.top.shape)
        d[..., 1] = -2.0
        moved = pillar_lattice(apply_displacement(simple, d))
        np.testing.assert_allclose(moved.top - moved.bot, lat.top - lat.bot, atol=1e-12)

    def test_independent_endpoints_tilt_the_pillar(self, simple):
        lat = pillar_lattice(simple)
        d_top = np.zeros(lat.top.shape)
        d_bot = np.zeros(lat.bot.shape)
        d_bot[..., 0] = 5.0
        moved = pillar_lattice(apply_displacement(simple, d_top, d_bot))
        assert not moved.is_vertical().any()
        np.testing.assert_allclose(moved.top, lat.top, atol=1e-12)

    def test_zcorn_is_never_touched(self, simple):
        lat = pillar_lattice(simple)
        d = np.full(lat.top.shape, 1.5)
        out = apply_displacement(simple, d)
        np.testing.assert_array_equal(
            np.asarray(out.zcorn, float), np.asarray(simple.zcorn, float)
        )

    def test_wrong_shape_raises(self, simple):
        with pytest.raises(ValueError, match="d_top must be"):
            apply_displacement(simple, np.zeros((2, 2, 2)))

    def test_round_trips_through_the_writer(self, simple, tmp_path):
        """Displacement must survive a write/read cycle unchanged."""
        from cpg2unstructured.grdecl_writer import write_grid_include

        lat = pillar_lattice(simple)
        d = np.zeros(lat.top.shape)
        d[..., 0] = 1.25
        moved = apply_displacement(simple, d)
        out = write_grid_include(moved, tmp_path / "moved.grdecl")
        back = pillar_lattice(read_grdecl(out))
        np.testing.assert_allclose(back.top, pillar_lattice(moved).top, atol=1e-12)
