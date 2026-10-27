"""ZCORN-only edits: corner addressing, shifts along pillars, welds and validity."""
from __future__ import annotations

import numpy as np
import pytest

import cpg2unstructured
from cpg2unstructured import (
    CornerPointGrid,
    apply_zcorn_displacement,
    build_topology,
    from_grdecl,
    unwelded_faces,
    with_zcorn,
    zcorn_cube,
    zcorn_validity,
)


def _layer_cake(nx: int = 3, ny: int = 2, nz: int = 2, dx: float = 100.0,
                dz: float = 10.0) -> CornerPointGrid:
    x = dx * np.arange(nx + 1)
    y = dx * np.arange(ny + 1)
    px, py = np.meshgrid(x, y, indexing="ij")
    coord = np.stack([px, py, np.zeros_like(px), px, py, np.full_like(px, nz * dz)], -1)
    cube = np.empty((2 * nx, 2 * ny, 2 * nz))
    for s in range(2 * nz):
        cube[:, :, s] = dz * ((s + 1) // 2)
    zcorn = np.ascontiguousarray(cube.transpose(2, 1, 0)).ravel()
    return CornerPointGrid((nx, ny, nz), coord.transpose(1, 0, 2).reshape(-1, 6), zcorn,
                           np.ones(nx * ny * nz, np.int8))


def test_cube_layout_and_round_trip():
    g = _layer_cake()
    cube = zcorn_cube(g)
    assert cube.shape == (6, 4, 4)
    # slot (2i+u, 2j+v, 2k+w): top (w=0) / base (w=1) of layer k
    assert cube[3, 2, 0] == 0.0 and cube[3, 2, 1] == 10.0 and cube[3, 2, 3] == 20.0
    assert np.array_equal(with_zcorn(g, cube).zcorn, g.zcorn)
    with pytest.raises(ValueError):
        with_zcorn(g, cube[:, :, :2])


def test_column_corner_shift_opens_only_the_cut_faces():
    g = _layer_cake()
    dz = np.zeros((6, 4))
    dz[2:, :] = 7.0                         # columns i >= 1 slide down rigidly
    slid = apply_zcorn_displacement(g, dz)
    open_i, open_j = unwelded_faces(slid)
    assert open_i.tolist() == [[True, True], [False, False]]
    assert not open_j.any()
    thick = np.diff(zcorn_cube(slid), axis=2)[:, :, 0::2]
    assert np.allclose(thick, 10.0)         # every cell keeps its thickness
    assert zcorn_validity(slid).ok
    assert np.array_equal(slid.coord, g.coord)
    # the slid deck is a faulted deck the topology builder accepts
    assert build_topology(from_grdecl(slid)).n_cells == g.n_cells


def test_scalar_and_per_slot_shifts():
    g = _layer_cake()
    assert np.allclose(zcorn_cube(apply_zcorn_displacement(g, 5.0)), zcorn_cube(g) + 5.0)
    per_slot = np.random.default_rng(0).normal(size=(6, 4, 4))
    assert np.allclose(zcorn_cube(apply_zcorn_displacement(g, per_slot)),
                       zcorn_cube(g) + per_slot)
    with pytest.raises(ValueError):
        apply_zcorn_displacement(g, np.zeros((3, 2)))


def test_per_horizon_shift_keeps_layers_touching_and_flags_inversion():
    g = _layer_cake()
    dz = np.zeros((6, 4, 3))
    dz[:, :, 1] = 4.0                       # horizon 1: base of layer 0 = top of layer 1
    thick = np.diff(zcorn_cube(apply_zcorn_displacement(g, dz)), axis=2)
    assert np.allclose(thick[:, :, 0], 14.0) and np.allclose(thick[:, :, 2], 6.0)
    assert np.allclose(thick[:, :, 1], 0.0)  # still touching

    dz[0, 0, 1] = 15.0                      # horizon 1 below horizon 2 at one corner
    bad = zcorn_validity(apply_zcorn_displacement(g, dz))
    assert not bad.ok and bad.n_inverted == 1 and bool(bad.inverted[0, 0, 1])
    assert "inverted" in bad.summary()


def test_overlapping_layers_are_flagged_but_gaps_are_not():
    g = _layer_cake()
    gap = zcorn_cube(g)
    gap[:, :, 2] += 1.0                     # top of layer 1 lowered: a gap, allowed
    assert zcorn_validity(with_zcorn(g, gap)).ok
    lap = zcorn_cube(g)
    lap[1, 1, 1] += 3.0                     # base of layer 0 below top of layer 1
    rep = zcorn_validity(with_zcorn(g, lap))
    assert not rep.ok and rep.n_overlapping == 1 and bool(rep.overlapping[0, 0, 0])


def test_every_public_name_is_bound():
    missing = [n for n in cpg2unstructured.__all__ if not hasattr(cpg2unstructured, n)]
    assert missing == []
