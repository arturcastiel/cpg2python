"""Property test (T6): random small CornerPointGrids round-trip through resfo.

For each generated grid:
    write via resfo.write → read via cpg2unstructured.read_egrid
    → assert cart_dims, coord, zcorn, actnum all match the source.

Skipped if ``hypothesis`` is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")

import resfo
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cpg2unstructured import read_egrid


def _make_arrays(nx: int, ny: int, nz: int, seed: int):
    rng = np.random.default_rng(seed)
    n_pillars = (nx + 1) * (ny + 1)
    coord = rng.uniform(-5.0, 5.0, size=(n_pillars, 6)).astype(np.float32)
    # Ensure top z < bottom z so the grid is non-degenerate.
    coord[:, 2] = 0.0
    coord[:, 5] = float(nz)
    zcorn = rng.uniform(0.0, float(nz), size=8 * nx * ny * nz).astype(np.float32)
    actnum = rng.integers(0, 2, size=nx * ny * nz, dtype=np.int32)
    gridhead = np.zeros(100, dtype=np.int32)
    gridhead[:5] = [1, nx, ny, nz, 1]
    return coord, zcorn, actnum, gridhead


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    nx=st.integers(min_value=1, max_value=5),
    ny=st.integers(min_value=1, max_value=5),
    nz=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_random_grid_round_trip(tmp_path_factory, nx, ny, nz, seed):
    coord, zcorn, actnum, gridhead = _make_arrays(nx, ny, nz, seed)
    p = tmp_path_factory.mktemp("egrid") / "g.EGRID"
    resfo.write(
        p,
        [
            ("FILEHEAD", np.zeros(100, dtype=np.int32)),
            ("GRIDHEAD", gridhead),
            ("COORD   ", coord.ravel()),
            ("ZCORN   ", zcorn),
            ("ACTNUM  ", actnum),
            ("ENDGRID ", np.zeros(0, dtype=np.int32)),
        ],
    )
    g = read_egrid(p)
    assert g.cart_dims == (nx, ny, nz)
    # No MAPAXES → coord stays in LOCAL frame; bytes preserved (with
    # f32 cast — already f32 here, so np.array_equal holds).
    np.testing.assert_array_equal(g.coord, coord)
    np.testing.assert_array_equal(g.zcorn, zcorn)
    np.testing.assert_array_equal(g.actnum, actnum.astype(np.int8))
