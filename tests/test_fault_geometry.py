"""Tests for the line-intersection helper.

Coord-based unit tests with hand-crafted line endpoints. No GRDECL,
no nodes table — pure math.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured._fault_geometry import line_intersection


# ── Single-pair sanity ──────────────────────────────────────────────


def test_x_constant_simple_crossing():
    """Two lines on vertical pillars (x=0 → x=1, y=0 → y=0)
    with z-trajectories that cross between the pillars.

    A: z goes 0 → 2  (slope +2 in z per unit x)
    B: z goes 2 → 0  (slope -2)
    They cross at x = 0.5, z = 1.
    """
    la_start = np.array([0.0, 0.0, 0.0])
    la_end   = np.array([1.0, 0.0, 2.0])
    lb_start = np.array([0.0, 0.0, 2.0])
    lb_end   = np.array([1.0, 0.0, 0.0])
    pt = line_intersection(la_start, la_end, lb_start, lb_end)
    np.testing.assert_allclose(pt, np.array([0.5, 0.0, 1.0]), atol=1e-12)


def test_intersection_at_la_start():
    """Edge case: lines cross AT line A's start point."""
    la_start = np.array([0.0, 0.0, 0.0])
    la_end   = np.array([1.0, 0.0, 1.0])
    # B-line: z=0 at p1 (same as la_start), z=2 at p2
    lb_start = np.array([0.0, 0.0, 0.0])
    lb_end   = np.array([1.0, 0.0, 2.0])
    pt = line_intersection(la_start, la_end, lb_start, lb_end)
    # t = (0 - 0) / (1 - 2) = 0 → intersection at la_start
    np.testing.assert_allclose(pt, la_start, atol=1e-12)


def test_intersection_beyond_la_end_returns_extrapolation():
    """t > 1 is geometrically valid; the function does NOT clamp.

    A: z 0 → 1.  B: z 0.5 → 1.4.  Difference of slopes: 1 - 0.9 = 0.1.
    t = (0.5 - 0) / 0.1 = 5 → intersection at x = 5, z = 5.
    """
    la_start = np.array([0.0, 0.0, 0.0])
    la_end   = np.array([1.0, 0.0, 1.0])
    lb_start = np.array([0.0, 0.0, 0.5])
    lb_end   = np.array([1.0, 0.0, 1.4])
    pt = line_intersection(la_start, la_end, lb_start, lb_end)
    np.testing.assert_allclose(pt, np.array([5.0, 0.0, 5.0]), atol=1e-12)


def test_z_exact_not_interpolated():
    """The returned z is the EXACT intersection z (avoid drift)."""
    la_start = np.array([0.0, 0.0, 0.0])
    la_end   = np.array([3.0, 0.0, 6.0])
    lb_start = np.array([0.0, 0.0, 4.0])
    lb_end   = np.array([3.0, 0.0, 4.0])
    pt = line_intersection(la_start, la_end, lb_start, lb_end)
    # B is horizontal at z=4. A reaches z=4 at t = (4-0)/6 = 2/3.
    # → x = 2.0, z = 4.0 exactly.
    np.testing.assert_allclose(pt, np.array([2.0, 0.0, 4.0]), atol=1e-15)


# ── Parallel z-trajectories raise ───────────────────────────────────


def test_parallel_z_trajectories_raise():
    """Δz_A == Δz_B → singular denominator → raise."""
    la_start = np.array([0.0, 0.0, 0.0])
    la_end   = np.array([1.0, 0.0, 1.0])
    lb_start = np.array([0.0, 0.0, 0.5])
    lb_end   = np.array([1.0, 0.0, 1.5])   # same slope as A
    with pytest.raises(ValueError, match="parallel z-trajectories"):
        line_intersection(la_start, la_end, lb_start, lb_end)


def test_parallel_z_horizontal_both_raise():
    """Both lines horizontal in z (Δz = 0 for both) → singular."""
    la_start = np.array([0.0, 0.0, 1.0])
    la_end   = np.array([1.0, 0.0, 1.0])
    lb_start = np.array([0.0, 0.0, 2.0])
    lb_end   = np.array([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="parallel z-trajectories"):
        line_intersection(la_start, la_end, lb_start, lb_end)


def test_horizontal_a_meets_sloped_b_ok():
    """A horizontal in z + B sloped → NOT parallel; intersection ok."""
    la_start = np.array([0.0, 0.0, 1.0])
    la_end   = np.array([1.0, 0.0, 1.0])  # A horizontal at z=1
    lb_start = np.array([0.0, 0.0, 0.0])
    lb_end   = np.array([1.0, 0.0, 2.0])  # B z 0 → 2
    pt = line_intersection(la_start, la_end, lb_start, lb_end)
    # Solving: t = (0 - 1) / (0 - 2) = 0.5 → x=0.5, z=1.0
    np.testing.assert_allclose(pt, np.array([0.5, 0.0, 1.0]), atol=1e-12)


# ── Vectorised ──────────────────────────────────────────────────────


def test_vectorised_three_pairs_at_once():
    """Call with shape (n, 3) inputs — should return shape (n, 3)."""
    la_start = np.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])
    la_end   = np.array([[1.0, 0.0, 2.0],
                         [1.0, 0.0, 1.0],
                         [3.0, 0.0, 6.0]])
    lb_start = np.array([[0.0, 0.0, 2.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 4.0]])
    lb_end   = np.array([[1.0, 0.0, 0.0],
                         [1.0, 0.0, 2.0],
                         [3.0, 0.0, 4.0]])
    pts = line_intersection(la_start, la_end, lb_start, lb_end)
    assert pts.shape == (3, 3)
    expected = np.array([[0.5, 0.0, 1.0],
                         [0.0, 0.0, 0.0],
                         [2.0, 0.0, 4.0]])
    np.testing.assert_allclose(pts, expected, atol=1e-12)


def test_vectorised_returns_float64():
    """Type contract: output is float64 even if inputs are float32."""
    la_start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    la_end   = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    lb_start = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    lb_end   = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pt = line_intersection(la_start, la_end, lb_start, lb_end)
    assert pt.dtype == np.float64


# ── Tilted-pillar use case (X varies with z) ────────────────────────


def test_sloped_xy_pillars_still_intersects_correctly():
    """When pillars are slanted (x changes between pillar 1 and 2),
    the interpolation along A still gives the correct intersection
    point — line A's (x, y) trajectory IS the answer.

    A goes from (0, 0, 0) to (1, 2, 4)  — slanted in x, y AND z.
    B goes from (0, 0, 4) to (1, 2, 0)  — same (x, y) trajectory, mirror z.
    Cross at midpoint: (0.5, 1.0, 2.0).
    """
    la_start = np.array([0.0, 0.0, 0.0])
    la_end   = np.array([1.0, 2.0, 4.0])
    lb_start = np.array([0.0, 0.0, 4.0])
    lb_end   = np.array([1.0, 2.0, 0.0])
    pt = line_intersection(la_start, la_end, lb_start, lb_end)
    np.testing.assert_allclose(pt, np.array([0.5, 1.0, 2.0]), atol=1e-12)
