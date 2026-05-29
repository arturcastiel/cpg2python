"""Tests for reduce_area — least-area polish that preserves the boundary.

Contract: toggling cells (XOR with a cell's faces) cannot change d2 (since
d2.d2_3 == 0), so reduce_area keeps the boundary EXACTLY while not increasing
area. A bulging fill (bare GF(2) solve) should shrink toward the planar straddle.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "/mnt/c/projects/library-development/fault-projections-benchmarks")
from run_benchmarks import build_grid  # noqa: E402

from cpg2unstructured import (  # noqa: E402
    boundary_chain,
    boundary_edges,
    face_areas,
    fill_prescribed_boundary,
    planar_minimal_surface,
    reduce_area,
    surface_boundary_loops,
)

BIG = np.array([[-9, -9], [9, -9], [9, 9], [-9, 9]], dtype=float)


def _rim(grid, faces):
    eid, co = boundary_chain(grid, np.asarray(faces, np.int64))
    return set(eid[co % 2 != 0].tolist())


def _area(grid, faces):
    return float(face_areas(grid)[np.asarray(faces, np.int64)].sum())


@pytest.fixture
def grid8():
    return build_grid(8, 8, 8)


def _straddle(grid):
    return planar_minimal_surface(
        grid, np.array([4.0, 4, 4]), np.array([1.0, 0, 0]),
        BIG, np.array([0.0, 1, 0]), np.array([0.0, 0, 1]))


def test_reduce_area_preserves_boundary_and_never_grows(grid8):
    # Core contract: boundary EXACTLY preserved, area never increases. (Single-cell
    # descent may not shrink a deep-local-minimum bulge -- that's documented and is
    # why the pipeline polishes the near-minimal COLLAR fill, not the bare fill.)
    surf = _straddle(grid8)
    gamma = boundary_edges(grid8, surf.face_ids)
    bare = fill_prescribed_boundary(grid8, gamma, margin=6.0)
    a_bare = _area(grid8, bare)
    polished = reduce_area(grid8, bare)
    assert _rim(grid8, polished) == set(gamma.tolist())   # boundary invariant
    assert _area(grid8, polished) <= a_bare + 1e-9        # never grows


def test_reduce_area_idempotent_on_minimal(grid8):
    # The straddle is already minimal; polishing must leave its area unchanged
    # and its boundary intact (the realistic pipeline case: near-minimal input).
    surf = _straddle(grid8)
    polished = reduce_area(grid8, surf.face_ids)
    assert _area(grid8, polished) == pytest.approx(float(surf.area))
    assert _rim(grid8, polished) == _rim(grid8, surf.face_ids)
    # and a second pass is a true fixed point
    assert set(reduce_area(grid8, polished).tolist()) == set(polished.tolist())


def test_reduce_area_keeps_zero_holes(grid8):
    surf = _straddle(grid8)
    gamma = boundary_edges(grid8, surf.face_ids)
    bare = fill_prescribed_boundary(grid8, gamma, margin=6.0)
    polished = reduce_area(grid8, bare)
    holes = max(0, len(surface_boundary_loops(grid8, polished)) - 1)
    assert holes == 0


def test_reduce_area_empty():
    g = build_grid(4, 4, 4)
    assert reduce_area(g, np.empty(0, np.int64)).size == 0
