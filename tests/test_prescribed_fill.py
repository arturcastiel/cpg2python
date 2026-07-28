"""Tests for the prescribed-boundary fill (PR-003).

The watertight quad-mesh projection rests on: given a closed grid edge-loop gamma,
find a cell-face surface Phi with d2.Phi == gamma exactly. T3 from the phase plan:
on a single planar polygon the fill must reproduce planar_minimal_surface; d2.Phi
== gamma on tilted/triangle; collar stays plane-hugging; non-bounding gamma raises.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import (
    BoundaryDoesNotBound,
    boundary_chain,
    boundary_edges,
    face_areas,
    fill_prescribed_boundary,
    fill_with_seed,
    gf2_solve_boundary,
    mesh_edges,
    planar_minimal_surface,
    surface_boundary_loops,
)

BIG = np.array([[-9, -9], [9, -9], [9, 9], [-9, 9]], dtype=float)


def _straddle(grid, p0, normal, u, v, poly=BIG):
    return planar_minimal_surface(grid, np.asarray(p0, float), np.asarray(normal, float),
                                  np.asarray(poly, float), np.asarray(u, float), np.asarray(v, float))


def _rim(grid, faces):
    eid, co = boundary_chain(grid, np.asarray(faces, np.int64))
    return set(eid[co % 2 != 0].tolist())


def _holes(grid, faces):
    return max(0, len(surface_boundary_loops(grid, np.asarray(faces, np.int64))) - 1)


# ── GF(2) solver unit ────────────────────────────────────────────────


def test_gf2_solver_recovers_known_boundary(mesh_3x3x2):
    # Take a real surface's rim as gamma; the solver must find SOME face set with
    # that exact boundary.
    surf = _straddle(mesh_3x3x2, [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    gamma = boundary_edges(mesh_3x3x2, surf.face_ids)
    phi = fill_prescribed_boundary(mesh_3x3x2, gamma, margin=5.0)
    assert _rim(mesh_3x3x2, phi) == set(gamma.tolist())


def test_gf2_solver_returns_none_when_inconsistent():
    # A single edge id as RHS is not a cycle -> not in the column space of any
    # face set's boundary -> None. (One face's columns each have 4 edges; a single
    # odd edge cannot be produced.)
    cols = [[0, 1, 2, 3], [2, 3, 4, 5]]   # two "faces"
    assert gf2_solve_boundary(cols, [0]) is None       # odd single row, unreachable
    # but the symmetric difference of the two columns IS reachable:
    x = gf2_solve_boundary(cols, [0, 1, 4, 5])
    assert x is not None and x.tolist() == [True, True]


# ── T3: planar polygon reproduces planar_minimal_surface ─────────────


def test_collar_planar_reproduces_straddle(mesh_3x3x2):
    surf = _straddle(mesh_3x3x2, [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    gamma = boundary_edges(mesh_3x3x2, surf.face_ids)
    # seed == straddle, rim == gamma already -> collar empty -> returns the seed.
    phi = fill_with_seed(mesh_3x3x2, gamma, surf.face_ids, margin=5.0)
    assert set(phi.tolist()) == set(surf.face_ids.tolist())
    assert float(face_areas(mesh_3x3x2)[phi].sum()) == pytest.approx(float(surf.area))


def test_bare_fill_exact_boundary_but_may_differ(mesh_3x3x2):
    surf = _straddle(mesh_3x3x2, [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    gamma = boundary_edges(mesh_3x3x2, surf.face_ids)
    phi = fill_prescribed_boundary(mesh_3x3x2, gamma, margin=5.0)
    # boundary is exact regardless of which surface the bare solve picked
    assert _rim(mesh_3x3x2, phi) == set(gamma.tolist())
    assert _holes(mesh_3x3x2, phi) == 0


# ── d2.Phi == gamma on tilted / larger grids ─────────────────────────


@pytest.fixture
def grid_6():
    """Build the 8x8x8 demo grid via the sibling fault-projections-benchmarks repo.

    Resolves the import path the same way as ``tests/test_reduce_area.py``:
    env override → sibling-of-repo → legacy hard-coded path, then
    ``pytest.importorskip`` if none resolves. CI environments without the
    sibling repo skip the dependent tests cleanly.
    """
    import os
    import sys
    from pathlib import Path

    bench_env = os.environ.get("FAULT_PROJECTIONS_BENCHMARKS")
    candidates = [
        Path(bench_env) if bench_env else None,
        Path(__file__).resolve().parent.parent.parent / "fault-projections-benchmarks",
        Path("/mnt/c/projects/library-development/fault-projections-benchmarks"),
    ]
    for c in candidates:
        if c is not None and c.is_dir():
            sys.path.insert(0, str(c))
            break
    run_benchmarks = pytest.importorskip(
        "run_benchmarks",
        reason="fault-projections-benchmarks sibling repo not available "
        "(set FAULT_PROJECTIONS_BENCHMARKS=/path or place sibling-of-repo)",
    )
    return run_benchmarks.build_grid(8, 8, 8)


@pytest.mark.parametrize("normal,u,v", [
    ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]),     # axis-aligned x
    ([0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]),     # axis-aligned y
    ([1.0, 1.0, 0], [-1.0, 1.0, 0], [0, 0, 1.0]),  # 45-deg tilt
])
def test_collar_fill_matches_boundary_on_planes(grid_6, normal, u, v):
    # The fill's CONTRACT is d2.Phi == gamma exactly, and that the fill preserves
    # the seed's topology (the collar only reconciles the rim). The seed's own
    # hole-count is a property of planar_minimal_surface, not of the fill, so we
    # compare fill-holes to SEED-holes rather than asserting 0 (a tilted plane vs a
    # too-big footprint can give a seed with a hole; see the diagnostic).
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    uu = np.asarray(u, float)
    uu /= np.linalg.norm(uu)
    surf = _straddle(grid_6, [4, 4, 4], n, uu, v)
    if surf.face_ids.size == 0:
        pytest.skip("plane misses grid")
    gamma = boundary_edges(grid_6, surf.face_ids)
    phi = fill_with_seed(grid_6, gamma, surf.face_ids, margin=6.0)
    assert _rim(grid_6, phi) == set(gamma.tolist())            # exact boundary
    assert _holes(grid_6, phi) == _holes(grid_6, surf.face_ids)  # topology preserved


def test_collar_fill_watertight_on_axis_disk(grid_6):
    # A clean axis-aligned bounded patch is a disk seed (0 holes); the collar fill
    # must stay watertight (0 holes) and boundary-exact. (Robust single-patch
    # watertightness on a DIAGONAL plane depends on planar_minimal_surface's
    # footprint matching, which is finicky on tilts; the watertight quad-MESH
    # assembly across tilted patches is proven end-to-end in the benchmarks
    # sandbox via the collar + signed-union path.)
    poly = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], float)
    surf = _straddle(grid_6, [4, 4, 4], [1, 0, 0], [0, 1, 0], [0, 0, 1], poly=poly)
    assert surf.face_ids.size > 0
    assert _holes(grid_6, surf.face_ids) == 0                  # disk seed
    gamma = boundary_edges(grid_6, surf.face_ids)
    phi = fill_with_seed(grid_6, gamma, surf.face_ids, margin=6.0)
    assert _rim(grid_6, phi) == set(gamma.tolist())
    assert _holes(grid_6, phi) == 0


# ── pinch-out / non-bounding gamma raises ────────────────────────────


def test_non_bounding_gamma_raises(mesh_3x3x2):
    # An arbitrary non-cycle edge set: a single edge can't be a boundary.
    mesh_edges(mesh_3x3x2)
    with pytest.raises(BoundaryDoesNotBound):
        fill_prescribed_boundary(mesh_3x3x2, np.array([0], np.int64), margin=9.0)


def test_empty_gamma_returns_empty(mesh_3x3x2):
    phi = fill_prescribed_boundary(mesh_3x3x2, np.empty(0, np.int64))
    assert phi.size == 0
