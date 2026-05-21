import numpy as np
import pytest
from pathlib import Path

from cpg2unstructured import (
    Faces,
    find_i_faces,
    find_ij_faces,
    find_j_faces,
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


def test_fault_raises_clean_error():
    cube = from_grdecl(read_grdecl(FIXTURES / "fault_simple.grdecl"))
    with pytest.raises(NonMatchingZError, match="v2 only"):
        find_i_faces(cube)


def test_actnum_excludes_inactive_face(tmp_path):
    # Criterion #7 defence: with cell (1,1,1) inactive, the matching-Z
    # check must NOT fire (D-PR6-3); face quads on the active side of
    # the degraded boundary must NOT contain -1 (D-PR6-4).
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
