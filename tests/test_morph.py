"""Least-squares pillar morphing: the operator, the boundary modes, and validity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured.grdecl_reader import read_grdecl
from cpg2unstructured.morph import (
    PillarTargets,
    laplacian_matrix,
    morph_pillars,
)
from cpg2unstructured.pillars import pillar_lattice

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def grid():
    return read_grdecl(FIXTURES / "simple_3x3x2.grdecl")


@pytest.fixture
def lat(grid):
    return pillar_lattice(grid)


def one_target(lat, ij=(1, 1), delta=(0.3, 0.0)):
    return PillarTargets(
        ij=np.array([ij]),
        xy_top=(lat.top[ij] + np.asarray(delta, float))[None, :],
    )


class TestLaplacian:
    def test_constants_are_in_the_null_space(self):
        L = laplacian_matrix(4, 5)
        assert np.allclose(L @ np.ones(20), 0.0)

    def test_is_symmetric(self):
        L = laplacian_matrix(4, 5).toarray()
        assert np.allclose(L, L.T)

    def test_is_positive_semidefinite(self):
        L = laplacian_matrix(3, 3).toarray()
        assert np.all(np.linalg.eigvalsh(L) > -1e-10)

    def test_degree_counts_existing_neighbours_only(self):
        """Corners have 2 neighbours, edges 3, interior 4 — no ghost padding."""
        L = laplacian_matrix(3, 3).toarray()
        deg = np.diag(L).reshape(3, 3)
        assert deg[0, 0] == 2 and deg[1, 0] == 3 and deg[1, 1] == 4


class TestBoundaryModes:
    def test_all_modes_produce_valid_grids(self, grid, lat):
        t = one_target(lat)
        for mode in ("slide", "pinned", "free"):
            assert morph_pillars(grid, t, boundary=mode).ok, mode

    def test_pinned_leaves_the_rim_exactly_fixed(self, grid, lat):
        from cpg2unstructured.pillars import boundary_mask

        r = morph_pillars(grid, one_target(lat), boundary="pinned")
        rim = boundary_mask(grid, scope="lattice")
        assert np.allclose(r.d_top[rim], 0.0)

    def test_pinned_moves_fewer_pillars_than_free(self, grid, lat):
        t = one_target(lat)
        assert (
            morph_pillars(grid, t, boundary="pinned").n_moved
            < morph_pillars(grid, t, boundary="free").n_moved
        )

    def test_free_boundary_reaches_the_target_almost_exactly(self, grid, lat):
        """A constant displacement is harmonic, so `free` can translate at no cost."""
        r = morph_pillars(grid, one_target(lat), boundary="free")
        assert r.residuals.max() < 1e-8

    def test_unknown_mode_raises(self, grid, lat):
        with pytest.raises(ValueError, match="boundary must be one of"):
            morph_pillars(grid, one_target(lat), boundary="sideways")


class TestLargeWLimit:
    @pytest.mark.parametrize("mode", ["pinned", "slide"])
    def test_huge_w_reproduces_the_input_grid(self, grid, lat, mode):
        """The property that makes the bisection bracket valid.

        With rim DOFs eliminated, L(d)=0 under zero boundary data forces d=0.
        """
        r = morph_pillars(grid, one_target(lat), w0=1e8, boundary=mode)
        assert np.abs(r.d_top).max() < 1e-6
        assert r.ok

    def test_small_w_follows_the_target_more_closely(self, grid, lat):
        t = one_target(lat)
        loose = morph_pillars(grid, t, w0=0.01, boundary="slide")
        tight = morph_pillars(grid, t, w0=100.0, boundary="slide")
        assert loose.residuals.max() < tight.residuals.max()


class TestKinkGeometry:
    def test_two_targets_on_one_pillar_land_between_them(self, grid, lat):
        """No ownership rule: a pillar with two residuals minimises their sum."""
        ij = (1, 1)
        p = lat.top[ij]
        t = PillarTargets(
            ij=np.array([ij, ij]),
            xy_top=np.array([p + [0.4, 0.0], p + [0.0, 0.4]]),
        )
        r = morph_pillars(grid, t, boundary="free")
        d = r.d_top[ij]
        assert d[0] > 0 and d[1] > 0, "must move toward both targets"
        assert abs(d[0] - d[1]) < 1e-6, "symmetric targets must give a symmetric move"

    def test_weights_bias_the_outcome(self, grid, lat):
        ij = (1, 1)
        p = lat.top[ij]
        t = PillarTargets(
            ij=np.array([ij, ij]),
            xy_top=np.array([p + [0.4, 0.0], p + [0.0, 0.4]]),
            weight=np.array([10.0, 1.0]),
        )
        d = morph_pillars(grid, t, boundary="free").d_top[ij]
        assert d[0] > d[1], "the heavier target must dominate"


class TestValidityAndBisection:
    def test_extreme_target_is_repaired_not_returned_folded(self, grid, lat):
        """A target far past a neighbour folds at w0; the result must still be valid."""
        span = float(np.linalg.norm(lat.top[1, 0] - lat.top[0, 0]))
        t = PillarTargets(
            ij=np.array([[1, 1]]),
            xy_top=(lat.top[1, 1] + np.array([-4.0 * span, -4.0 * span]))[None, :],
        )
        r = morph_pillars(grid, t, w0=1e-4, boundary="pinned")
        assert r.ok, f"solver returned an invalid grid: {r.validity.summary()}"
        assert r.w > 1e-4, "w must have been raised to repair the fold"
        assert any(not ok for _, ok, _ in r.history), "history should record the folds"

    def test_history_is_recorded(self, grid, lat):
        r = morph_pillars(grid, one_target(lat))
        assert r.history and r.history[0][0] == pytest.approx(0.3)

    def test_zcorn_is_never_touched(self, grid, lat):
        r = morph_pillars(grid, one_target(lat))
        np.testing.assert_array_equal(
            np.asarray(r.grid.zcorn, float), np.asarray(grid.zcorn, float)
        )

    def test_write_tol_zeroes_tiny_displacements(self, grid, lat):
        """Keeps the deck diff local instead of touching every pillar."""
        r = morph_pillars(grid, one_target(lat), boundary="slide", write_tol=1e-3)
        nonzero = np.linalg.norm(r.d_top, axis=-1) > 0
        assert np.all(np.linalg.norm(r.d_top[nonzero], axis=-1) >= 1e-3)


class TestStageTwo:
    def test_rigid_targets_move_both_endpoints_equally(self, grid, lat):
        r = morph_pillars(grid, one_target(lat))
        np.testing.assert_allclose(r.d_top, r.d_bot, atol=1e-12)

    def test_independent_targets_tilt_the_pillars(self, grid, lat):
        ij = (1, 1)
        t = PillarTargets(
            ij=np.array([ij]),
            xy_top=(lat.top[ij] + [0.3, 0.0])[None, :],
            xy_bot=(lat.bot[ij] + [-0.3, 0.0])[None, :],
        )
        r = morph_pillars(grid, t, boundary="free")
        assert not np.allclose(r.d_top, r.d_bot), "top and bottom must differ"
        assert r.d_top[ij][0] > 0 and r.d_bot[ij][0] < 0

    def test_stage_two_result_is_valid(self, grid, lat):
        ij = (1, 1)
        t = PillarTargets(
            ij=np.array([ij]),
            xy_top=(lat.top[ij] + [0.2, 0.0])[None, :],
            xy_bot=(lat.bot[ij] + [-0.2, 0.0])[None, :],
        )
        assert morph_pillars(grid, t).ok


class TestTargetValidation:
    def test_bad_ij_shape(self):
        with pytest.raises(ValueError, match="ij must be"):
            PillarTargets(ij=np.zeros((3, 3)), xy_top=np.zeros((3, 2)))

    def test_mismatched_xy_top(self):
        with pytest.raises(ValueError, match="xy_top must be"):
            PillarTargets(ij=np.zeros((3, 2), int), xy_top=np.zeros((2, 2)))

    def test_mismatched_weight(self):
        with pytest.raises(ValueError, match="weight must be"):
            PillarTargets(
                ij=np.zeros((3, 2), int), xy_top=np.zeros((3, 2)), weight=np.ones(2)
            )

    def test_is_rigid_reflects_xy_bot(self):
        t = PillarTargets(ij=np.zeros((1, 2), int), xy_top=np.zeros((1, 2)))
        assert t.is_rigid
        t2 = PillarTargets(
            ij=np.zeros((1, 2), int), xy_top=np.zeros((1, 2)), xy_bot=np.zeros((1, 2))
        )
        assert not t2.is_rigid
