from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured import (
    IndexMap,
    build_topology,
    from_grdecl,
    read_grdecl,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _eclipse_natural_ijk(actnum_3d: np.ndarray) -> np.ndarray:
    """Return (N, 3) (i, j, k) for active cells in Eclipse F-order."""
    nx, ny, nz = actnum_3d.shape
    flat = actnum_3d.ravel(order="F")
    flat_idx = np.where(flat)[0]
    i = flat_idx % nx
    j = (flat_idx // nx) % ny
    k = flat_idx // (nx * ny)
    return np.stack([i, j, k], axis=1).astype(np.int64)


def test_round_trip_dense():
    nx, ny, nz = 4, 5, 3
    actnum = np.ones((nx, ny, nz), dtype=bool)
    ijk = _eclipse_natural_ijk(actnum)
    im = IndexMap((nx, ny, nz), ijk)
    assert im.n_active == nx * ny * nz
    for a in range(im.n_active):
        i, j, k = im.cell_ijk(a)
        assert im.active_id(i, j, k) == a


def test_round_trip_sparse_actnum():
    rng = np.random.default_rng(42)
    nx, ny, nz = 6, 6, 4
    mask = rng.random((nx, ny, nz)) > 0.3
    ijk = _eclipse_natural_ijk(mask)
    im = IndexMap((nx, ny, nz), ijk)
    assert im.n_active == int(mask.sum())
    for a in range(im.n_active):
        i, j, k = im.cell_ijk(a)
        assert im.active_id(i, j, k) == a
    # Inactive cell returns None.
    inactive_pos = tuple(int(x) for x in np.argwhere(~mask)[0])
    assert im.active_id(*inactive_pos) is None


def test_out_of_range_raises():
    im = IndexMap((2, 2, 2), np.array([[0, 0, 0]], dtype=np.int64))
    with pytest.raises(IndexError):
        im.active_id(5, 0, 0)


def test_cells_in_layer():
    nx, ny, nz = 3, 3, 3
    actnum = np.ones((nx, ny, nz), dtype=bool)
    ijk = _eclipse_natural_ijk(actnum)
    im = IndexMap((nx, ny, nz), ijk)
    layer0 = im.cells_in_layer(0)
    assert len(layer0) == nx * ny
    # All should have k == 0.
    for a in layer0:
        assert im.cell_ijk(int(a))[2] == 0


def test_linear_to_cell_eclipse_natural():
    nx, ny, nz = 3, 3, 3
    actnum = np.ones((nx, ny, nz), dtype=bool)
    ijk = _eclipse_natural_ijk(actnum)
    im = IndexMap((nx, ny, nz), ijk)
    # Cell at (1, 0, 0) → linear = 1.
    a = im.active_id(1, 0, 0)
    assert im.linear_to_cell[a] == 1
    # Cell at (0, 1, 0) → linear = NX.
    a2 = im.active_id(0, 1, 0)
    assert im.linear_to_cell[a2] == nx
    # Cell at (0, 0, 1) → linear = NX * NY.
    a3 = im.active_id(0, 0, 1)
    assert im.linear_to_cell[a3] == nx * ny


def test_reverse_built_lazily():
    # Defence test for acceptance #6.
    nx, ny, nz = 2, 2, 2
    ijk = _eclipse_natural_ijk(np.ones((nx, ny, nz), dtype=bool))
    im = IndexMap((nx, ny, nz), ijk)
    # Reverse not yet built.
    assert im._reverse is None
    # First call triggers the build.
    im.active_id(0, 0, 0)
    assert im._reverse is not None
    # Subsequent call reuses the same array (no rebuild).
    first_rev = im._reverse
    im.active_id(1, 1, 1)
    assert im._reverse is first_rev


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        IndexMap((2, 2, 2), np.array([0, 0, 0]))  # 1-D, not (N, 3)
    with pytest.raises(ValueError):
        IndexMap((2, 2, 2), np.zeros((3, 4)))     # wrong columns


def test_repr_shows_shape_and_count():
    im = IndexMap((3, 4, 5), np.array([[0, 0, 0]], dtype=np.int64))
    s = repr(im)
    assert "cart_dims=(3, 4, 5)" in s
    assert "n_active=1" in s


def test_integrates_with_build_topology():
    # End-to-end: build_topology → IndexMap round-trip on the 2x2x2 fixture.
    cube = from_grdecl(read_grdecl(FIXTURES / "simple_2x2x2.grdecl"))
    mesh = build_topology(cube)
    im = IndexMap(mesh.cart_dims, mesh.original_ijk)
    assert im.n_active == mesh.cell_faces.shape[0]
    # Cell ids returned by IndexMap must match _TopoMesh's cell ordering.
    for a in range(im.n_active):
        i, j, k = im.cell_ijk(a)
        assert tuple(int(x) for x in mesh.original_ijk[a]) == (i, j, k)
        assert im.active_id(i, j, k) == a


def test_linear_to_cell_dtype():
    im = IndexMap((2, 2, 2),
                  _eclipse_natural_ijk(np.ones((2, 2, 2), dtype=bool)))
    assert im.linear_to_cell.dtype == np.int64
