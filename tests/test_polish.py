"""The barrier polish: does it help, does it ever harm, does it respect the rules."""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import (
    PillarTargets,
    check_validity,
    morph_pillars,
    pillar_lattice,
    read_grdecl,
    scaled_jacobian,
    write_grid_include,
)
from cpg2unstructured._toy import build_toy_cpg
from cpg2unstructured.corner_point import CornerPointGrid

# A trace that reverses direction repeatedly: the quadratic solve leaves degenerate
# cells here, which is what the polish exists to remove.
ZIGZAG = np.array([[100.0, 100.0], [400.0, 500.0], [200.0, 800.0],
                   [700.0, 900.0], [500.0, 400.0], [1100.0, 300.0]])


def _grid(tmp_path, nx=12, ny=12, nz=1, dz=20.0):
    toy = build_toy_cpg(nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0, dz=dz)
    g = CornerPointGrid(
        cart_dims=(nx, ny, nz),
        coord=toy.coord.transpose(1, 0, 2).reshape(-1, 6),
        zcorn=toy.zcorn.transpose(2, 1, 0).ravel(),
        actnum=toy.actnum.transpose(2, 1, 0).ravel().astype(np.int8),
    )
    return read_grdecl(write_grid_include(g, tmp_path / "polish.grdecl"))


def _targets(grid, line):
    """Claim the pillars near a polyline, without depending on fault-projections."""
    lat = pillar_lattice(grid)
    xy = lat.top.reshape(-1, 2)
    ij, tgt = [], []
    nx1, ny1 = lat.shape
    for k in range(xy.shape[0]):
        best, bd = None, np.inf
        for a, b in zip(line[:-1], line[1:]):
            ab = b - a
            t = np.clip(((xy[k] - a) @ ab) / (ab @ ab), 0.0, 1.0)
            foot = a + t * ab
            d = float(np.linalg.norm(xy[k] - foot))
            if d < bd:
                best, bd = foot, d
        if bd <= 55.0:
            ij.append((k // ny1, k % ny1))
            tgt.append(best)
    return PillarTargets(ij=np.asarray(ij, dtype=np.int64),
                         xy_top=np.asarray(tgt, dtype=np.float64))


@pytest.fixture
def grid(tmp_path):
    return _grid(tmp_path)


@pytest.fixture
def targets(grid):
    t = _targets(grid, ZIGZAG)
    assert t.n > 10
    return t


class TestPolishImproves:
    def test_it_removes_the_degenerate_cells(self, grid, targets):
        """The whole reason it exists."""
        base = morph_pillars(grid, targets, w0=0.05, boundary="slide")
        good = morph_pillars(grid, targets, w0=0.05, boundary="slide",
                             polish="barrier")
        sj_b = scaled_jacobian(pillar_lattice(base.grid).top)
        sj_g = scaled_jacobian(pillar_lattice(good.grid).top)
        assert (sj_b < 0.3).sum() > 0, "this fixture is meant to be a hard case"
        assert (sj_g < 0.3).sum() == 0
        assert sj_g.min() > sj_b.min()

    def test_it_reports_what_it_did(self, grid, targets):
        r = morph_pillars(grid, targets, w0=0.05, boundary="slide", polish="barrier")
        assert r.polish is not None
        assert r.polish.applied
        assert r.polish.energy_after < r.polish.energy_before
        assert 0.0 < r.polish.improvement <= 1.0
        assert "polish" in r.summary()

    def test_the_result_is_still_valid(self, grid, targets):
        r = morph_pillars(grid, targets, w0=0.05, boundary="slide", polish="barrier")
        assert r.ok, r.validity.summary()

    def test_it_costs_fidelity(self, grid, targets):
        """Honest: the polish trades distance to the fault for cell shape."""
        base = morph_pillars(grid, targets, w0=0.05, boundary="slide")
        good = morph_pillars(grid, targets, w0=0.05, boundary="slide",
                             polish="barrier")
        assert good.residuals.mean() > base.residuals.mean()

    def test_more_fidelity_weight_stays_closer(self, grid, targets):
        low = morph_pillars(grid, targets, w0=0.05, boundary="slide",
                            polish="barrier", polish_fidelity=1.0)
        high = morph_pillars(grid, targets, w0=0.05, boundary="slide",
                             polish="barrier", polish_fidelity=40.0)
        assert high.residuals.mean() < low.residuals.mean()


class TestPolishIsSafe:
    def test_off_by_default(self, grid, targets):
        r = morph_pillars(grid, targets, w0=0.05, boundary="slide")
        assert r.polish is None

    def test_default_result_is_byte_identical(self, grid, targets):
        """Adding the option must not perturb anything that does not ask for it."""
        a = morph_pillars(grid, targets, w0=0.05, boundary="slide")
        b = morph_pillars(grid, targets, w0=0.05, boundary="slide", polish="none")
        np.testing.assert_array_equal(a.d_top, b.d_top)
        np.testing.assert_array_equal(a.d_bot, b.d_bot)

    def test_an_unknown_mode_is_refused(self, grid, targets):
        with pytest.raises(ValueError, match="polish must be one of"):
            morph_pillars(grid, targets, polish="magic")

    def test_no_targets_means_nothing_to_polish(self, grid):
        empty = PillarTargets(ij=np.empty((0, 2), dtype=np.int64),
                              xy_top=np.empty((0, 2)))
        r = morph_pillars(grid, empty, polish="barrier")
        assert r.polish is None
        assert r.ok


class TestPolishRespectsTheRules:
    def test_pinned_boundary_stays_pinned(self, grid, targets):
        r = morph_pillars(grid, targets, w0=0.05, boundary="pinned",
                          polish="barrier")
        lat0, lat1 = pillar_lattice(grid), pillar_lattice(r.grid)
        for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
            np.testing.assert_allclose(lat1.top[sl], lat0.top[sl], atol=1e-9)

    def test_slide_boundary_keeps_the_outline(self, grid, targets):
        r = morph_pillars(grid, targets, w0=0.05, boundary="slide",
                          polish="barrier")
        lat0, lat1 = pillar_lattice(grid), pillar_lattice(r.grid)
        # left and right edges may redistribute in y but never move in x
        np.testing.assert_allclose(lat1.top[0, :, 0], lat0.top[0, :, 0], atol=1e-9)
        np.testing.assert_allclose(lat1.top[-1, :, 0], lat0.top[-1, :, 0], atol=1e-9)
        np.testing.assert_allclose(lat1.top[:, 0, 1], lat0.top[:, 0, 1], atol=1e-9)
        np.testing.assert_allclose(lat1.top[:, -1, 1], lat0.top[:, -1, 1], atol=1e-9)

    def test_support_rings_still_confines_the_change(self, grid, targets):
        r = morph_pillars(grid, targets, w0=0.05, boundary="slide",
                          support_rings=1, polish="barrier")
        lat0, lat1 = pillar_lattice(grid), pillar_lattice(r.grid)
        moved = np.linalg.norm(lat1.top - lat0.top, axis=-1) > 1e-9
        assert moved.sum() < 0.5 * moved.size

    def test_a_rigid_morph_stays_rigid(self, grid, targets):
        """Stage 1 moves whole pillars; the polish must not introduce a tilt."""
        r = morph_pillars(grid, targets, w0=0.05, boundary="slide",
                          polish="barrier")
        np.testing.assert_allclose(r.d_top, r.d_bot, atol=1e-12)


class TestDegenerateCheck:
    def test_off_by_default(self, grid, targets):
        r = morph_pillars(grid, targets, w0=0.05, boundary="slide")
        assert r.validity.n_degenerate == 0
        assert r.validity.degenerate is None
        assert r.ok

    def test_it_catches_what_the_fold_test_cannot(self, grid, targets):
        """A cell flattened to near-zero area has no crossing corner."""
        base = morph_pillars(grid, targets, w0=0.05, boundary="slide")
        lat0, lat1 = pillar_lattice(grid), pillar_lattice(base.grid)
        assert base.validity.ok and base.validity.n_folded == 0
        strict = check_validity(lat1, lat0, min_scaled_jacobian=0.3)
        assert strict.n_degenerate > 0
        assert not strict.ok
        assert "degenerate" in strict.summary()

    def test_the_polished_grid_passes_the_strict_check(self, grid, targets):
        r = morph_pillars(grid, targets, w0=0.05, boundary="slide",
                          polish="barrier")
        strict = check_validity(pillar_lattice(r.grid), pillar_lattice(grid),
                                min_scaled_jacobian=0.3)
        assert strict.ok, strict.summary()


class TestPillarAlreadyOnTheFault:
    """A pillar whose target IS its current position must be held there.

    `normal_only` builds one fidelity row per pillar from the direction
    `target - current`. When a pillar already sits on the fault that direction is
    undefined, and zeroing it leaves an ALL-ZERO row: the pillar is constrained by
    nothing and the smoothness term drags it off the fault it was already on.
    """

    @staticmethod
    def _diag_targets(grid):
        """A 45-degree trace through a square lattice hits pillars exactly."""
        lat = pillar_lattice(grid)
        nx1, ny1 = lat.shape
        ij = np.array([(k, k) for k in range(min(nx1, ny1))], dtype=np.int64)
        xy = lat.top[ij[:, 0], ij[:, 1]]
        return PillarTargets(ij=ij, xy_top=xy.copy())      # target == current

    def test_pillars_on_the_fault_are_not_dragged_off(self, grid):
        t = self._diag_targets(grid)
        r = morph_pillars(grid, t, w0=0.05, boundary="slide")
        lat0, lat1 = pillar_lattice(grid), pillar_lattice(r.grid)
        moved = np.linalg.norm(
            lat1.top[t.ij[:, 0], t.ij[:, 1]] - lat0.top[t.ij[:, 0], t.ij[:, 1]], axis=1
        )
        assert moved.max() < 1e-6, f"a pillar already on the fault moved {moved.max()} m"

    def test_the_grid_is_untouched_when_nothing_needs_to_move(self, grid):
        """Every target already satisfied ⇒ the identity is the answer."""
        t = self._diag_targets(grid)
        r = morph_pillars(grid, t, w0=0.05, boundary="slide")
        assert np.abs(r.d_top).max() < 1e-6
        assert r.residuals.max() < 1e-6

    def test_a_mix_of_satisfied_and_unsatisfied_targets(self, grid):
        """The degenerate rows must not disturb the pillars that do need moving."""
        lat = pillar_lattice(grid)
        ij = np.array([(3, 3), (4, 4), (5, 5), (6, 6)], dtype=np.int64)
        xy = lat.top[ij[:, 0], ij[:, 1]].copy()
        xy[1] += np.array([25.0, 0.0])                     # one real request
        t = PillarTargets(ij=ij, xy_top=xy)
        r = morph_pillars(grid, t, w0=0.05, boundary="slide")
        assert r.ok
        lat1 = pillar_lattice(r.grid)
        got = lat1.top[ij[:, 0], ij[:, 1]]
        # the three satisfied pillars stay put, the fourth moves toward its target
        assert np.linalg.norm(got[0] - xy[0]) < 2.0
        assert np.linalg.norm(got[2] - xy[2]) < 2.0
        assert np.linalg.norm(got[1] - lat.top[4, 4]) > 5.0
