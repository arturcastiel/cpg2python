"""Cell-quality metrics — pinned against shapes whose answers are known."""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured.quality import (
    aspect_ratio,
    cell_quality,
    non_orthogonality,
    scaled_jacobian,
    skew,
)


def lattice_xy(nx, ny, dx=1.0, dy=1.0):
    x = np.arange(nx + 1) * dx
    y = np.arange(ny + 1) * dy
    return np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).astype(float)


class TestScaledJacobian:
    def test_square_cells_score_one(self):
        assert np.allclose(scaled_jacobian(lattice_xy(3, 3)), 1.0)

    def test_rectangles_still_score_one(self):
        """SJ measures angle, not stretch — a 10:1 rectangle is still perfect."""
        assert np.allclose(scaled_jacobian(lattice_xy(3, 3, dx=10.0, dy=1.0)), 1.0)

    def test_sheared_cells_score_below_one(self):
        xy = lattice_xy(3, 3)
        xy[..., 0] += 0.5 * xy[..., 1]          # uniform shear
        sj = scaled_jacobian(xy)
        assert np.all(sj < 1.0) and np.all(sj > 0.0)

    def test_inverted_corner_goes_negative(self):
        xy = lattice_xy(2, 2)
        xy[1, 1] = xy[0, 0] - (xy[1, 1] - xy[0, 0])   # drag a corner across
        assert scaled_jacobian(xy).min() < 0.0


class TestAspectRatio:
    def test_square_is_one(self):
        assert np.allclose(aspect_ratio(lattice_xy(3, 3)), 1.0)

    def test_ten_to_one_rectangle(self):
        assert np.allclose(aspect_ratio(lattice_xy(3, 3, dx=10.0, dy=1.0)), 10.0)


class TestSkew:
    def test_square_has_no_skew(self):
        assert np.allclose(skew(lattice_xy(3, 3)), 0.0)

    def test_rectangle_has_no_skew(self):
        """Skew is about ANGLES: stretching must not register."""
        assert np.allclose(skew(lattice_xy(3, 3, dx=8.0, dy=1.0)), 0.0)

    def test_shear_registers(self):
        xy = lattice_xy(3, 3)
        xy[..., 0] += 1.0 * xy[..., 1]          # 45 degrees of shear
        s = skew(xy)
        assert np.all(s > 0.4), "45 degrees off square should be about half of 90"


class TestNonOrthogonality:
    def test_cartesian_grid_is_orthogonal(self):
        no = non_orthogonality(lattice_xy(4, 4))
        assert no.size > 0
        assert np.allclose(no, 0.0, atol=1e-9)

    def test_stretched_cartesian_is_still_orthogonal(self):
        no = non_orthogonality(lattice_xy(4, 4, dx=7.0, dy=1.0))
        assert np.allclose(no, 0.0, atol=1e-9)

    def test_shear_creates_non_orthogonality(self):
        xy = lattice_xy(4, 4)
        xy[..., 0] += 0.6 * xy[..., 1]
        assert non_orthogonality(xy).max() > 20.0


class TestReport:
    def test_perfect_grid_reports_perfect(self):
        from cpg2unstructured.pillars import PillarLattice

        xy = lattice_xy(4, 4, dx=100.0, dy=100.0)
        lat = PillarLattice(top=xy, bot=xy.copy(),
                            z_top=np.zeros(xy.shape[:2]),
                            z_bot=np.ones(xy.shape[:2]), cart_dims=(4, 4, 1))
        q = cell_quality(lat)
        assert q.sj_min == pytest.approx(1.0)
        assert q.sj_below_0_3 == 0
        assert q.aspect_max == pytest.approx(1.0)
        assert q.skew_max == pytest.approx(0.0, abs=1e-9)
        assert q.ortho_mean == pytest.approx(0.0, abs=1e-9)
        assert q.n_cells == 16
        assert "SJ min" in q.line()

    def test_report_flags_a_degraded_grid(self):
        from cpg2unstructured.pillars import PillarLattice

        xy = lattice_xy(4, 4, dx=100.0, dy=100.0)
        xy[..., 0] += 0.9 * xy[..., 1]
        lat = PillarLattice(top=xy, bot=xy.copy(),
                            z_top=np.zeros(xy.shape[:2]),
                            z_bot=np.ones(xy.shape[:2]), cart_dims=(4, 4, 1))
        q = cell_quality(lat)
        assert q.sj_min < 1.0
        assert q.skew_mean > 0.0
        assert q.ortho_mean > 10.0
