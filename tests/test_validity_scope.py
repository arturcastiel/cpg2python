"""The validity gate: what it counts, what it can see, and what it says when it gives up."""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import (
    PillarTargets,
    ValidityExhausted,
    ValidityExhaustedWarning,
    active_columns,
    check_validity,
    morph_pillars,
    pillar_lattice,
    read_grdecl,
    write_grid_include,
)
from cpg2unstructured._toy import build_toy_cpg
from cpg2unstructured.corner_point import CornerPointGrid


def _grid(tmp_path, nx=10, ny=10, nz=2, dead=None):
    """A toy grid, optionally with some columns switched off."""
    toy = build_toy_cpg(nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0, dz=20.0)
    act = np.ones((nx, ny, nz), dtype=np.int8)
    for i, j in (dead or []):
        act[i, j, :] = 0
    g = CornerPointGrid(
        cart_dims=(nx, ny, nz),
        coord=toy.coord.transpose(1, 0, 2).reshape(-1, 6),
        zcorn=toy.zcorn.transpose(2, 1, 0).ravel(),
        actnum=act.transpose(2, 1, 0).ravel(),
    )
    return read_grdecl(write_grid_include(g, tmp_path / "vs.grdecl"))


class TestActiveColumns:
    def test_all_live_by_default(self, tmp_path):
        assert active_columns(_grid(tmp_path)).all()

    def test_dead_columns_are_reported(self, tmp_path):
        g = _grid(tmp_path, dead=[(3, 3), (3, 4)])
        a = active_columns(g)
        assert not a[3, 3] and not a[3, 4]
        assert a.sum() == a.size - 2


class TestCheckValidityMask:
    """A fold in a column with no cells is arithmetic on absent geometry."""

    @staticmethod
    def _fold(grid, i, j):
        """Drag one pillar across its neighbour so column (i, j) inverts."""
        lat = pillar_lattice(grid)
        top = lat.top.copy()
        bot = lat.bot.copy()
        top[i, j] = lat.top[i + 1, j + 1]
        bot[i, j] = lat.bot[i + 1, j + 1]
        import dataclasses
        return dataclasses.replace(lat, top=top, bot=bot)

    def test_a_fold_in_a_dead_column_is_ignored_when_masked(self, tmp_path):
        g = _grid(tmp_path, dead=[(3, 3)])
        lat0 = pillar_lattice(g)
        bad = self._fold(g, 3, 3)
        loose = check_validity(bad, lat0)
        assert loose.n_folded > 0, "the fixture must actually fold something"
        tight = check_validity(bad, lat0, active=active_columns(g))
        assert tight.n_folded < loose.n_folded

    def test_a_fold_in_a_LIVE_column_is_never_ignored(self, tmp_path):
        g = _grid(tmp_path, dead=[(8, 8)])
        lat0 = pillar_lattice(g)
        bad = self._fold(g, 3, 3)
        masked = check_validity(bad, lat0, active=active_columns(g))
        assert masked.n_folded > 0
        assert not masked.ok

    def test_the_mask_never_invents_a_violation(self, tmp_path):
        """Masking can only ever remove findings, never add them."""
        g = _grid(tmp_path, dead=[(2, 2), (7, 4)])
        lat0 = pillar_lattice(g)
        bad = self._fold(g, 5, 5)
        loose = check_validity(bad, lat0)
        tight = check_validity(bad, lat0, active=active_columns(g))
        assert tight.n_folded <= loose.n_folded
        assert tight.n_welded <= loose.n_welded
        assert tight.n_too_close <= loose.n_too_close


class TestValidityScope:
    @staticmethod
    def _targets(grid, dx=45.0):
        lat = pillar_lattice(grid)
        ij = np.array([(5, j) for j in range(1, lat.shape[1] - 1)], dtype=np.int64)
        xy = lat.top[ij[:, 0], ij[:, 1]].copy()
        xy[:, 0] += dx
        return PillarTargets(ij=ij, xy_top=xy)

    def test_active_is_the_default(self, tmp_path):
        g = _grid(tmp_path)
        a = morph_pillars(g, self._targets(g), w0=0.05)
        b = morph_pillars(g, self._targets(g), w0=0.05, validity_scope="active")
        np.testing.assert_array_equal(a.d_top, b.d_top)

    def test_a_fully_active_deck_is_unaffected_by_the_scope(self, tmp_path):
        """The synthetics have no dead cells, so the fix changes nothing there."""
        g = _grid(tmp_path)
        a = morph_pillars(g, self._targets(g), w0=0.05, validity_scope="lattice")
        b = morph_pillars(g, self._targets(g), w0=0.05, validity_scope="active")
        np.testing.assert_allclose(a.d_top, b.d_top, atol=1e-12)
        assert a.w == b.w

    def test_an_unknown_scope_is_refused(self, tmp_path):
        g = _grid(tmp_path)
        with pytest.raises(ValueError, match="validity_scope"):
            morph_pillars(g, self._targets(g), validity_scope="everywhere")


class TestExhaustedBracket:
    """Returning an invalid grid with no signal is how a caller ships a broken deck."""

    @staticmethod
    def _impossible(grid):
        """Targets that cannot be met at any smoothness weight: two adjacent
        pillars asked to swap places."""
        lat = pillar_lattice(grid)
        ij = np.array([(4, 4), (5, 4)], dtype=np.int64)
        xy = np.stack([lat.top[5, 4], lat.top[4, 4]])
        return PillarTargets(ij=ij, xy_top=xy)

    def test_it_warns_by_default(self, tmp_path):
        g = _grid(tmp_path)
        with pytest.warns(ValidityExhaustedWarning, match="no valid smoothness"):
            r = morph_pillars(g, self._impossible(g), w0=1e-6, w_max=1e-4,
                              max_bisect=2)
        assert not r.ok
        assert r.bracket_exhausted

    def test_raise_mode_raises(self, tmp_path):
        g = _grid(tmp_path)
        with pytest.raises(ValidityExhausted):
            morph_pillars(g, self._impossible(g), w0=1e-6, w_max=1e-4,
                          max_bisect=2, on_invalid="raise")

    def test_ignore_mode_is_silent(self, tmp_path):
        g = _grid(tmp_path)
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            r = morph_pillars(g, self._impossible(g), w0=1e-6, w_max=1e-4,
                              max_bisect=2, on_invalid="ignore")
        assert r.bracket_exhausted

    def test_a_solvable_case_never_sets_the_flag(self, tmp_path):
        g = _grid(tmp_path)
        lat = pillar_lattice(g)
        ij = np.array([(5, 5)], dtype=np.int64)
        xy = lat.top[ij[:, 0], ij[:, 1]] + np.array([10.0, 0.0])
        r = morph_pillars(g, PillarTargets(ij=ij, xy_top=xy), w0=0.05)
        assert r.ok and not r.bracket_exhausted

    def test_an_unknown_mode_is_refused(self, tmp_path):
        g = _grid(tmp_path)
        with pytest.raises(ValueError, match="on_invalid"):
            morph_pillars(g, self._impossible(g), on_invalid="explode")
