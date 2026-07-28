"""Fold, weld and separation detection over a deformed pillar lattice."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured.grdecl_reader import read_grdecl
from cpg2unstructured.pillars import apply_displacement, pillar_lattice
from cpg2unstructured.validity import (
    WELD_TOLERANCE,
    check_validity,
    column_signed_areas,
    separation_violations,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def grid():
    return read_grdecl(FIXTURES / "simple_3x3x2.grdecl")


@pytest.fixture
def lat(grid):
    return pillar_lattice(grid)


class TestSignedArea:
    def test_regular_grid_has_uniform_nonzero_sign(self, lat):
        a = column_signed_areas(lat.top)
        nx, ny, _ = lat.cart_dims
        assert a.shape == (nx, ny)
        assert np.all(np.abs(a) > 0)
        assert len(set(np.sign(a).ravel().tolist())) == 1, "orientation must be uniform"

    def test_area_magnitude_is_the_cell_area(self, lat):
        a = np.abs(column_signed_areas(lat.top))
        dx = np.linalg.norm(lat.top[1, 0] - lat.top[0, 0])
        dy = np.linalg.norm(lat.top[0, 1] - lat.top[0, 0])
        assert np.allclose(a, dx * dy)

    def test_sign_flips_when_a_quad_is_folded(self, lat):
        xy = lat.top.copy()
        # drag one corner across the diagonal — the classic fold
        xy[1, 1] = xy[0, 0] - (xy[1, 1] - xy[0, 0])
        assert np.any(np.sign(column_signed_areas(xy)) != np.sign(column_signed_areas(lat.top)))


class TestCheckValidity:
    def test_untouched_grid_is_valid(self, lat):
        r = check_validity(lat, lat)
        assert r.ok and r.n_folded == 0 and r.n_welded == 0
        assert "valid" in r.summary()

    def test_small_rigid_shift_is_valid(self, grid, lat):
        d = np.zeros(lat.top.shape)
        d[..., 0] = 0.05
        moved = pillar_lattice(apply_displacement(grid, d))
        assert check_validity(moved, lat).ok

    def test_uniform_translation_of_everything_is_valid(self, grid, lat):
        """Moving the whole grid changes no relative geometry."""
        d = np.full(lat.top.shape, 7.0)
        moved = pillar_lattice(apply_displacement(grid, d))
        assert check_validity(moved, lat).ok

    def test_dragging_one_pillar_across_its_neighbour_folds(self, grid, lat):
        d = np.zeros(lat.top.shape)
        span = float(np.linalg.norm(lat.top[1, 0] - lat.top[0, 0]))
        d[1, 1, 0] = -2.5 * span          # far past its neighbour
        d[1, 1, 1] = -2.5 * span
        moved = pillar_lattice(apply_displacement(grid, d))
        r = check_validity(moved, lat)
        assert not r.ok
        assert r.n_folded > 0
        assert "INVALID" in r.summary()

    def test_both_planes_are_checked_independently(self, grid, lat):
        """A fold introduced only at the BOTTOM plane must still be caught.

        This is the case a single-plane test misses, and slanted pillars are the
        norm on real decks — so the bottom-only fold is not a contrived scenario.
        """
        span = float(np.linalg.norm(lat.top[1, 0] - lat.top[0, 0]))
        d_top = np.zeros(lat.top.shape)
        d_bot = np.zeros(lat.bot.shape)
        d_bot[1, 1] = -2.5 * span
        moved = pillar_lattice(apply_displacement(grid, d_top, d_bot))
        r = check_validity(moved, lat)
        assert not r.ok
        assert r.folded_bot.any(), "bottom-plane fold must be detected"
        assert not r.folded_top.any(), "top plane was untouched"


class TestSeparation:
    def test_weld_is_detected_at_the_dedup_tolerance(self, grid, lat):
        d = np.zeros(lat.top.shape)
        # move pillar (1,0) onto pillar (0,0) exactly
        d[1, 0] = lat.top[0, 0] - lat.top[1, 0]
        moved = pillar_lattice(apply_displacement(grid, d))
        welded, _ = separation_violations(moved, lat)
        assert welded[1, 0] and welded[0, 0]

    def test_too_close_flags_below_the_fraction(self, grid, lat):
        span = float(np.linalg.norm(lat.top[1, 0] - lat.top[0, 0]))
        d = np.zeros(lat.top.shape)
        d[1, 0, 0] = -0.95 * span          # 5% of the original spacing remains
        moved = pillar_lattice(apply_displacement(grid, d))
        _, too_close = separation_violations(moved, lat, min_fraction=0.10)
        assert too_close[1, 0]

    def test_fraction_is_configurable(self, grid, lat):
        span = float(np.linalg.norm(lat.top[1, 0] - lat.top[0, 0]))
        d = np.zeros(lat.top.shape)
        d[1, 0, 0] = -0.80 * span          # 20% remains
        moved = pillar_lattice(apply_displacement(grid, d))
        _, strict = separation_violations(moved, lat, min_fraction=0.30)
        _, lenient = separation_violations(moved, lat, min_fraction=0.10)
        assert strict[1, 0] and not lenient[1, 0]

    def test_weld_tolerance_matches_the_dedup_quantisation(self):
        """The floor must track cpg._QUANT — 1e-9 is not an arbitrary number."""
        from cpg2unstructured.cpg import _QUANT

        assert WELD_TOLERANCE == pytest.approx(1.0 / _QUANT)


class TestDepthCrossings:
    """Stage-2 failure mode: pillars that stay apart in plan view but meet in depth."""

    def test_untilted_grid_has_no_crossings(self, lat):
        from cpg2unstructured.validity import depth_crossings

        assert not depth_crossings(lat).any()

    def test_pillars_swapping_sides_with_depth_are_caught(self, grid, lat):
        """Top and bottom offsets that reverse the separation must be detected.

        Neither plane's signed area sees this: at the top the pair is apart, at the
        bottom it is apart again — but the separation passes through zero between.
        """
        from cpg2unstructured.pillars import apply_displacement
        from cpg2unstructured.validity import depth_crossings

        span = float(np.linalg.norm(lat.top[1, 0] - lat.top[0, 0]))
        d_top = np.zeros(lat.top.shape)
        d_bot = np.zeros(lat.bot.shape)
        # tilt pillar (0,0): its top stops just short of neighbour (1,0) while its
        # bottom passes it, so the separation REVERSES with depth and the two
        # pillars intersect somewhere between the planes
        d_top[0, 0, 0] = +0.9 * span
        d_bot[0, 0, 0] = +1.1 * span
        moved = pillar_lattice(apply_displacement(grid, d_top, d_bot))
        assert depth_crossings(moved, tol=1e-6)[0, 0]

    def test_crossing_is_reported_by_check_validity(self, grid, lat):
        from cpg2unstructured.pillars import apply_displacement

        span = float(np.linalg.norm(lat.top[1, 0] - lat.top[0, 0]))
        d_top = np.zeros(lat.top.shape)
        d_bot = np.zeros(lat.bot.shape)
        d_top[0, 0, 0] = +0.9 * span
        d_bot[0, 0, 0] = +1.1 * span
        moved = pillar_lattice(apply_displacement(grid, d_top, d_bot))
        assert not check_validity(moved, lat).ok

    def test_plan_view_alone_would_miss_it(self, grid, lat):
        """Pin the claim in the docstring: neither plane's area flips."""
        from cpg2unstructured.pillars import apply_displacement
        from cpg2unstructured.validity import column_signed_areas, depth_crossings

        span = float(np.linalg.norm(lat.top[1, 0] - lat.top[0, 0]))
        d_top = np.zeros(lat.top.shape)
        d_bot = np.zeros(lat.bot.shape)
        d_top[0, 0, 0] = +0.9 * span
        d_bot[0, 0, 0] = +1.1 * span
        moved = pillar_lattice(apply_displacement(grid, d_top, d_bot))

        top_ok = np.all(
            np.sign(column_signed_areas(moved.top)) == np.sign(column_signed_areas(lat.top))
        )
        bot_ok = np.all(
            np.sign(column_signed_areas(moved.bot)) == np.sign(column_signed_areas(lat.bot))
        )
        # Verified empirically, and it is NOT obvious: a reversed edge does not have
        # to flip its column's signed area — the quad's other corners compensate, so
        # the shoelace sum keeps its sign while the quad self-intersects.
        assert top_ok and bot_ok, "both planes look fine in isolation"
        assert depth_crossings(moved, tol=1e-6).any(), "yet the pillars cross in depth"
