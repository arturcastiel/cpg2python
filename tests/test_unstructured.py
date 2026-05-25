import numpy as np
import pytest
from pathlib import Path

from cpg2unstructured import (
    Faces,
    build_topology,
    find_i_faces,
    find_ij_faces,
    find_j_faces,
    find_k_faces,
    from_grdecl,
    read_grdecl,
)
from cpg2unstructured._errors import NonMatchingZError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cube_2x2x2():
    return from_grdecl(read_grdecl(FIXTURES / "simple_2x2x2.grdecl"))


def test_i_faces_count_2x2x2(cube_2x2x2):
    fi = find_i_faces(cube_2x2x2)
    # (NX+1)*NY*NZ = 3*2*2 = 12 i-faces; 4 inner + 8 boundary.
    assert fi.nodes.shape == (12, 4)
    inner = (fi.neighbours[:, 1] != -1).sum()
    bnd   = (fi.neighbours[:, 1] == -1).sum()
    assert inner == 4
    assert bnd == 8


def test_j_faces_count_2x2x2(cube_2x2x2):
    fj = find_j_faces(cube_2x2x2)
    assert fj.nodes.shape == (12, 4)
    inner = (fj.neighbours[:, 1] != -1).sum()
    assert inner == 4


def test_face_quads_sorted(cube_2x2x2):
    fi = find_i_faces(cube_2x2x2)
    assert np.all(np.diff(fi.nodes, axis=1) >= 0)


def test_neighbour_pairs_sorted(cube_2x2x2):
    fi = find_i_faces(cube_2x2x2)
    inner = fi.neighbours[fi.neighbours[:, 1] != -1]
    assert (inner[:, 0] < inner[:, 1]).all()


def test_boundary_neg1_in_col1(cube_2x2x2):
    fi = find_i_faces(cube_2x2x2)
    bnd = fi.neighbours[fi.neighbours[:, 1] == -1]
    # -1 must be in column 1, never column 0.
    assert (bnd[:, 1] == -1).all()
    assert (bnd[:, 0] != -1).all()


def test_direction_int8(cube_2x2x2):
    fi = find_i_faces(cube_2x2x2)
    fj = find_j_faces(cube_2x2x2)
    assert fi.direction.dtype == np.int8
    assert fj.direction.dtype == np.int8
    assert (fi.direction == 0).all()
    assert (fj.direction == 1).all()


def test_find_ij_faces_shape(cube_2x2x2):
    fij = find_ij_faces(cube_2x2x2)
    assert fij.nodes.shape == (24, 4)
    assert fij.neighbours.shape == (24, 2)
    assert fij.direction.shape == (24,)


def test_fault_dispatches_in_find_i_faces():
    """Post- find_i_faces no longer raises on
    fault_simple.grdecl — it dispatches matching vs faulted inline
. The single-corner Z mismatch flags one i-pillar stack
    as faulted; the dispatcher drops the matching inner faces in that
    stack and emits replacement fault faces from _find_faulted_i_faces.
    Total i-face count is preserved (single-corner mismatch produces
    the same number of inner faces, just via the fault path)."""
    cube = from_grdecl(read_grdecl(FIXTURES / "fault_simple.grdecl"))
    faces = find_i_faces(cube)
    # 2x2x2 all-active: 3 i-planes × 2 j × 2 k = 12 i-faces total.
    # The fault path produces 2 fault faces (single faulted stack, 2
    # connections) in place of 2 matching inner faces. Net: 12.
    assert faces.nodes.shape == (12, 4)
    assert faces.neighbours.shape == (12, 2)


def test_actnum_excludes_inactive_face(tmp_path):
    # Criterion #7 defence: with cell (1,1,1) inactive, the matching-Z
    # check must NOT fire; face quads on the active side of
    # the degraded boundary must NOT contain -1.
    src = (FIXTURES / "simple_2x2x2.grdecl").read_text().replace(
        "ACTNUM\n  8*1 /",
        "ACTNUM\n  1 1 1 1  1 1 1 0 /",
    )
    p = tmp_path / "deck.grdecl"
    p.write_text(src)
    cube = from_grdecl(read_grdecl(p))
    fi = find_i_faces(cube)
    # 12 total in the all-active case; with cell (1,1,1) inactive, the
    # ip=2 j=1 k=1 face has BOTH sides inactive → filtered. Total = 11.
    assert fi.nodes.shape == (11, 4)
    # No face quad contains -1.
    assert (fi.nodes >= 0).all()
    # Inner-face count drops: ip=1 (j=1, k=1) degrades to boundary
    # because cell (1,1,1) is inactive. 4 inner → 3 inner.
    inner = (fi.neighbours[:, 1] != -1).sum()
    assert inner == 3


def test_faces_dataclass_fields(cube_2x2x2):
    fi = find_i_faces(cube_2x2x2)
    assert isinstance(fi, Faces)
    assert fi.nodes.dtype == np.int64
    assert fi.neighbours.dtype == np.int64
    assert fi.direction.dtype == np.int8


# ────────────────────────────────────────────────────────────────────
#   — k-faces + cell-face inversion + active-cell rebuild
# ────────────────────────────────────────────────────────────────────


def test_k_faces_count_2x2x2(cube_2x2x2):
    fk = find_k_faces(cube_2x2x2)
    # NX*NY*(NZ+1) = 2*2*3 = 12 k-faces; 4 inner + 8 boundary.
    assert fk.nodes.shape == (12, 4)
    assert (fk.direction == 2).all()
    inner = (fk.neighbours[:, 1] != -1).sum()
    bnd   = (fk.neighbours[:, 1] == -1).sum()
    assert inner == 4
    assert bnd == 8


def test_total_faces_2x2x2(cube_2x2x2):
    mesh = build_topology(cube_2x2x2)
    # 12 + 12 + 12 = 36 faces total; 4 inner per axis = 12 inner.
    assert mesh.faces.nodes.shape == (36, 4)
    inner = (mesh.faces.neighbours[:, 1] != -1).sum()
    assert inner == 12


def test_cell_faces_shape_2x2x2(cube_2x2x2):
    mesh = build_topology(cube_2x2x2)
    assert mesh.cell_faces.shape == (8, 6)
    assert (mesh.cell_faces >= 0).all()


def test_cell_corners_shape_2x2x2(cube_2x2x2):
    mesh = build_topology(cube_2x2x2)
    assert mesh.cell_corners.shape == (8, 8)
    assert mesh.cell_corners.min() >= 0
    assert mesh.cell_corners.max() < mesh.nodes.shape[0]


def test_each_cell_three_inner_neighbours_2x2x2(cube_2x2x2):
    mesh = build_topology(cube_2x2x2)
    inner = mesh.faces.neighbours[mesh.faces.neighbours[:, 1] != -1]
    counts = np.bincount(inner.ravel(), minlength=8)
    # 2x2x2: every cell touches exactly 3 inner faces (no strictly-interior).
    assert (counts == 3).all()


def test_original_ijk_eclipse_natural_2x2x2(cube_2x2x2):
    mesh = build_topology(cube_2x2x2)
    expected = np.array([
        [0, 0, 0], [1, 0, 0],
        [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1],
        [0, 1, 1], [1, 1, 1],
    ], dtype=np.int64)
    np.testing.assert_array_equal(mesh.original_ijk, expected)


def test_pinchout_drops_zero_volume_cell():
    cube = from_grdecl(read_grdecl(FIXTURES / "pinchout_2x2x2.grdecl"))
    mesh = build_topology(cube)
    # Corner-pinchout on a 2x2x2 structured matching-Z grid drops more
    # than the strictly-collapsed cell:
    #   - cell (1,1,1)              fully collapsed (4 faces with 2 unique)
    #   - cells (0,1,1), (1,0,1)    each shares 2 pillars with (1,1,1),
    #                               so its i+ / j+ face collapses to 2 unique
    #   - cell (0,0,1)              shares ONE pillar (1,1) only → single-pair
    #                               wedge with worst-face 3 unique → KEPT
    # → 8 - 3 = 5 active+real cells.
    # Spec criterion #5 ("one collapsed cell → 7 active") was geometrically
    # naive; the corrected per-face uniqueness rule makes this
    # concrete.
    assert mesh.cell_faces.shape[0] == 5
    # The strictly-collapsed cell (1, 1, 1) must NOT appear in survivors.
    assert not (mesh.original_ijk == np.array([1, 1, 1])).all(axis=1).any()
    # Cell (0, 0, 1) — the single-pair wedge — IS in survivors.
    assert (mesh.original_ijk == np.array([0, 0, 1])).all(axis=1).any()


def test_inactive_cell_excluded_from_topology(tmp_path):
    src = (FIXTURES / "simple_2x2x2.grdecl").read_text().replace(
        "ACTNUM\n  8*1 /",
        "ACTNUM\n  1 1 1 1  1 1 1 0 /",
    )
    p = tmp_path / "deck.grdecl"
    p.write_text(src)
    cube = from_grdecl(read_grdecl(p))
    mesh = build_topology(cube)
    # 8 cells, cell 7 (i=1, j=1, k=1) inactive → 7 active.
    assert mesh.cell_faces.shape[0] == 7
    # Faces: 33 surviving (12*3 raw - 3 fully-dropped boundaries on cell 7's
    # outside i+/j+/k+).
    assert mesh.faces.nodes.shape == (33, 4)
    assert (mesh.cell_faces >= 0).all()


def test_3x3x3_strictly_interior_cell_six_neighbours():
    cube = from_grdecl(read_grdecl(FIXTURES / "simple_3x3x3.grdecl"))
    mesh = build_topology(cube)
    # 27 active cells. Cell (1, 1, 1) in F-order: 1 + 1*3 + 1*9 = 13.
    interior_idx = 13
    assert tuple(int(x) for x in mesh.original_ijk[interior_idx]) == (1, 1, 1)
    cell_face_ids = mesh.cell_faces[interior_idx]
    assert (cell_face_ids >= 0).all()
    nbs = mesh.faces.neighbours[cell_face_ids]
    assert (nbs != -1).all(), f"interior cell faces have boundary: {nbs}"


def test_topology_face_quads_sorted(cube_2x2x2):
    mesh = build_topology(cube_2x2x2)
    assert np.all(np.diff(mesh.faces.nodes, axis=1) >= 0)


def test_topology_no_negative_node_ids(cube_2x2x2):
    mesh = build_topology(cube_2x2x2)
    assert (mesh.faces.nodes >= 0).all()
    assert (mesh.cell_corners >= 0).all()


def test_fault_succeeds_end_to_end_in_build_topology():
    """Post- build_topology no longer raises on
    i-direction faults. The fault_simple.grdecl deck has a single
    i-fault that find_i_faces dispatches through _find_faulted_i_faces;
    build_topology now produces a valid UnstructuredGrid end-to-end.
    """
    cube = from_grdecl(read_grdecl(FIXTURES / "fault_simple.grdecl"))
    grid = build_topology(cube)
    assert grid is not None
    # 2x2x2 all-active: 8 cells, all retained (single-corner mismatch
    # doesn't produce zero-volume cells).
    assert grid.n_cells == 8
