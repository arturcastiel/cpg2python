import numpy as np
import pytest
from pathlib import Path

from cpg2unstructured import CornerCube, from_grdecl, read_grdecl

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cube_2x2x2():
    g = read_grdecl(FIXTURES / "simple_2x2x2.grdecl")
    return from_grdecl(g)


def test_node_count_2x2x2(cube_2x2x2):
    assert cube_2x2x2.nodes.shape == (27, 3)


def test_P_shape_2x2x2(cube_2x2x2):
    assert cube_2x2x2.P.shape == (4, 4, 4)


def test_P_indexes_in_range(cube_2x2x2):
    assert cube_2x2x2.P.min() == 0
    assert cube_2x2x2.P.max() == 26


def test_handedness_corner_origin(cube_2x2x2):
    # P[0, 0, 0] must point at the geometric (0, 0, 0) corner.
    n = cube_2x2x2.nodes[cube_2x2x2.P[0, 0, 0]]
    np.testing.assert_allclose(n, (0.0, 0.0, 0.0))


def test_handedness_corner_far(cube_2x2x2):
    n = cube_2x2x2.nodes[cube_2x2x2.P[-1, -1, -1]]
    np.testing.assert_allclose(n, (2.0, 2.0, 2.0))


def test_coord_layout_offdiag_pillars_correct(cube_2x2x2):
    # Defence test for D-PR5-3. Under the old buggy reshape these would
    # return swapped pillars (1,0,0) ↔ (0,1,0). Catches a layout regression
    # without needing a non-square fixture.
    n_10 = cube_2x2x2.nodes[cube_2x2x2.P[1, 0, 0]]
    n_01 = cube_2x2x2.nodes[cube_2x2x2.P[0, 1, 0]]
    np.testing.assert_allclose(n_10, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(n_01, (0.0, 1.0, 0.0))


def test_actnum_zero_excludes_corners(tmp_path):
    # Disable cell (1,1,1); its strictly-private corner P[3,3,3] should
    # become -1; nodes count = 26.
    src = (FIXTURES / "simple_2x2x2.grdecl").read_text().replace(
        "ACTNUM\n  8*1 /",
        "ACTNUM\n  1 1 1 1  1 1 1 0 /",
    )
    p = tmp_path / "deck.grdecl"
    p.write_text(src)
    cube = from_grdecl(read_grdecl(p))
    assert cube.nodes.shape == (26, 3)
    assert cube.P[3, 3, 3] == -1


def test_3x3x2_node_count():
    g = read_grdecl(FIXTURES / "simple_3x3x2.grdecl")
    cube = from_grdecl(g)
    # (3+1)*(3+1)*(2+1) = 48
    assert cube.nodes.shape == (48, 3)


def test_cornercube_dataclass_fields(cube_2x2x2):
    assert isinstance(cube_2x2x2, CornerCube)
    assert cube_2x2x2.cart_dims == (2, 2, 2)
    assert cube_2x2x2.actnum_3d.shape == (2, 2, 2)
    assert cube_2x2x2.actnum_3d.dtype == bool
    assert cube_2x2x2.actnum_3d.all()
    assert cube_2x2x2.P.dtype == np.int64
    assert cube_2x2x2.nodes.dtype == np.float64
