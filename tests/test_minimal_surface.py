"""Tests for the planar minimal-surface selector."""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import (
    MinimalSurface,
    fit_plane,
    planar_minimal_surface,
    point_in_polygon,
)

BIG_SQUARE = np.array([[-9, -9], [9, -9], [9, 9], [-9, 9]], dtype=float)


def test_fit_plane_recovers_normal():
    # Points on the plane z = 2.
    rng = np.random.default_rng(0)
    xy = rng.uniform(-3, 3, size=(20, 2))
    pts = np.column_stack([xy, np.full(20, 2.0)])
    p0, normal, u, v = fit_plane(pts)
    assert abs(abs(normal @ np.array([0, 0, 1.0])) - 1.0) < 1e-9
    assert abs(normal @ u) < 1e-9 and abs(normal @ v) < 1e-9


def test_point_in_polygon_square():
    poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    pts = np.array([[1, 1], [0.1, 0.1], [3, 1], [-1, -1], [1, 3]])
    inside = point_in_polygon(pts, poly)
    assert inside.tolist() == [True, True, False, False, False]


def test_constant_x_plane_selects_i_faces(mesh_3x3x2):
    # Plane x = 1.0 separates column i=0 (cx=0.5) from i=1 (cx=1.5).
    p0 = np.array([1.0, 1.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    u, v = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    surf = planar_minimal_surface(mesh_3x3x2, p0, normal, BIG_SQUARE, u, v)
    assert isinstance(surf, MinimalSurface)
    # 3 (j) * 2 (k) i-faces between i=0 and i=1.
    assert surf.face_ids.shape == (6,)
    assert surf.area == pytest.approx(6.0)
    assert (surf.direction == 0).all()            # all i-faces (X)
    # adjacent cells are i=0 and i=1.
    assert set(surf.cell_ijk[:, :, 0].ravel().tolist()) == {0, 1}


def test_footprint_restricts_selection(mesh_3x3x2):
    p0 = np.array([1.0, 1.0, 1.0])
    normal = np.array([1.0, 0.0, 0.0])
    u, v = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    # Tiny footprint around one face centroid (j=0,k=0 -> plane coords (-0.5,-0.5)).
    small = np.array([[-0.9, -0.9], [-0.1, -0.9], [-0.1, -0.1], [-0.9, -0.1]])
    surf = planar_minimal_surface(mesh_3x3x2, p0, normal, small, u, v)
    assert surf.face_ids.shape == (1,)
    assert surf.area == pytest.approx(1.0)


def test_empty_when_plane_outside(mesh_3x3x2):
    # Plane far outside the grid -> no straddling faces.
    p0 = np.array([100.0, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    u, v = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    surf = planar_minimal_surface(mesh_3x3x2, p0, normal, BIG_SQUARE, u, v)
    assert surf.face_ids.shape == (0,)
    assert surf.area == 0.0
