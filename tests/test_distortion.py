"""Barrier distortion metrics: gradients, invariances, and the barrier itself."""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured.distortion import (
    CORNER_STENCIL,
    DistortionTerms,
    cell_quads,
    corner_jacobians,
    distortion_energy,
    shape_targets,
)


def _lattice(nx1=6, ny1=6, h=100.0):
    i, j = np.meshgrid(np.arange(nx1), np.arange(ny1), indexing="ij")
    return np.stack([i * h, j * h], axis=-1).reshape(-1, 2).astype(float), nx1, ny1


@pytest.fixture
def setup():
    P0, nx1, ny1 = _lattice()
    q = cell_quads(nx1, ny1)
    return P0, q, shape_targets(P0, q)


class TestQuads:
    def test_one_quad_per_column(self):
        assert cell_quads(6, 5).shape == ((6 - 1) * (5 - 1), 4)

    def test_corners_are_counter_clockwise(self, setup):
        """Every corner Jacobian of a positively oriented cell has det > 0."""
        P0, q, _ = setup
        J = corner_jacobians(P0, q)
        det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        assert (det > 0).all()

    def test_stencil_covers_every_corner_once(self):
        assert sorted(a for (a, _), _ in CORNER_STENCIL) == [0, 1, 2, 3]


class TestTargets:
    def test_original_lattice_scores_zero(self, setup):
        """A cell that has not changed shape has no distortion."""
        P0, q, winv = setup
        E, G = distortion_energy(P0, q, winv, DistortionTerms(1.0, 1.0, 1.0))
        assert E == pytest.approx(0.0, abs=1e-12)
        assert np.abs(G).max() == pytest.approx(0.0, abs=1e-9)

    def test_a_graded_grid_is_not_penalised_for_being_graded(self):
        """The point of shape targets: only CHANGE is measured, not shape itself."""
        i, j = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
        P0 = np.stack([np.exp(i * 0.4) * 50.0, j * 100.0], axis=-1).reshape(-1, 2)
        q = cell_quads(6, 6)
        E, _ = distortion_energy(P0, q, shape_targets(P0, q), DistortionTerms())
        assert E == pytest.approx(0.0, abs=1e-12)
        # against identity targets the same grid is heavily "distorted"
        E_id, _ = distortion_energy(P0, q, None, DistortionTerms())
        assert E_id > 0.1

    def test_degenerate_original_is_refused(self):
        P0, nx1, ny1 = _lattice(4, 4)
        q = cell_quads(nx1, ny1)
        P0[q[0, 1]] = P0[q[0, 0]]                    # collapse one cell's edge
        with pytest.raises(ValueError, match="degenerate"):
            shape_targets(P0, q)


class TestBarrier:
    def test_energy_is_infinite_for_an_inverted_cell(self, setup):
        P0, q, winv = setup
        P = P0.copy()
        P[q[0, 2]] = P0[q[0, 0]] - 50.0              # drag a corner across
        E, G = distortion_energy(P, q, winv, DistortionTerms())
        assert not np.isfinite(E)
        assert np.abs(G).max() == 0.0

    def test_energy_grows_without_bound_as_a_cell_flattens(self, setup):
        """Each decade closer to degenerate costs a decade of energy.

        The energy is a MEAN over all 100 corners, so one collapsing corner shows up
        divided by 100 — the growth RATE is the property that matters, and it is
        1/(1-t) exactly as the 1/det barrier predicts.
        """
        P0, q, winv = setup
        seen = []
        for t in (0.9, 0.99, 0.999, 0.9999):
            P = P0.copy()
            P[q[0, 3]] = P0[q[0, 0]] + (1 - t) * (P0[q[0, 3]] - P0[q[0, 0]])
            seen.append(distortion_energy(P, q, winv, DistortionTerms())[0])
        assert all(b > a for a, b in zip(seen, seen[1:]))
        for a, b in zip(seen, seen[1:]):
            assert b / a == pytest.approx(10.0, rel=0.15)

    def test_shape_metric_is_scale_invariant(self, setup):
        """0.5|T|^2/det(T) does not see uniform scaling — it is a SHAPE metric."""
        P0, q, winv = setup
        rng = np.random.default_rng(0)
        P = P0 + rng.normal(0, 5.0, P0.shape)
        a, _ = distortion_energy(P, q, winv, DistortionTerms(1, 0, 0))
        b, _ = distortion_energy(P * 3.0, q, winv, DistortionTerms(1, 0, 0))
        assert b == pytest.approx(a, rel=1e-10)

    def test_size_metric_is_not_scale_invariant(self, setup):
        """...which is exactly why a separate size term exists."""
        P0, q, winv = setup
        a, _ = distortion_energy(P0 * 1.5, q, winv, DistortionTerms(0, 1, 0))
        assert a > 0.05


class TestGradients:
    @pytest.mark.parametrize(
        "terms",
        [DistortionTerms(1, 0, 0), DistortionTerms(0, 1, 0), DistortionTerms(0, 0, 1),
         DistortionTerms(1, 0.35, 0.2)],
        ids=["shape", "size", "shape_size", "mixed"],
    )
    @pytest.mark.parametrize("targets", [True, False], ids=["shape-targets", "identity"])
    def test_matches_finite_differences(self, terms, targets):
        # Unit spacing on purpose. With identity targets on a 100 m grid the size
        # metric is O(det J) = O(1e4), and the finite difference loses the precision
        # needed to test the gradient rather than the differencing. The code path is
        # identical either way.
        P0, nx1, ny1 = _lattice(6, 6, h=1.0)
        q = cell_quads(nx1, ny1)
        w = shape_targets(P0, q) if targets else None
        rng = np.random.default_rng(7)
        P = P0 + rng.normal(0, 0.04, P0.shape)
        _, G = distortion_energy(P, q, w, terms)

        h = 1e-6
        checked = 0
        for k in range(0, P.size, 3):
            a, b = divmod(k, 2)
            Pp, Pm = P.copy(), P.copy()
            Pp[a, b] += h
            Pm[a, b] -= h
            num = (distortion_energy(Pp, q, w, terms)[0]
                   - distortion_energy(Pm, q, w, terms)[0]) / (2 * h)
            if abs(num) > 1e-9:
                assert G[a, b] == pytest.approx(num, rel=2e-5, abs=1e-9)
                checked += 1
        assert checked > 15, "the perturbation was too small to test anything"


class TestTerms:
    def test_no_active_terms_is_reported(self):
        assert not DistortionTerms(0, 0, 0).any_active()
        assert DistortionTerms(0, 0, 1).any_active()
