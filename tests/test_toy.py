import numpy as np
import pytest

from cpg2unstructured._toy import build_toy_cpg, build_toy_mesh


@pytest.fixture(scope="module")
def mesh():
    return build_toy_mesh(build_toy_cpg(nx=2, ny=2, nz=2))


def test_node_count(mesh):
    # 3x3x3 unique grid points
    assert mesh.nodes.shape == (27, 3)


def test_cell_count(mesh):
    assert mesh.cell_nodes.shape == (8, 8)


def test_face_count_total(mesh):
    # 3 axes × 3 face-planes × 4 faces-per-plane = 36
    assert mesh.faces.shape == (36, 4)


def test_face_count_inner_vs_boundary(mesh):
    boundary = (mesh.face_cells[:, 1] == -1).sum()
    inner    = (mesh.face_cells[:, 1] != -1).sum()
    # Corrected vs wave-spec: 12 inner / 24 boundary in a 2x2x2 grid.
    assert inner == 12
    assert boundary == 24


def test_each_cell_has_three_neighbours(mesh):
    # 2x2x2 has no strictly-interior cell; each of 8 cells touches 3 inner faces.
    counts = np.zeros(8, dtype=int)
    for a, b in mesh.neighbours:
        counts[a] += 1
        counts[b] += 1
    assert (counts == 3).all()


def test_cell_faces_shape(mesh):
    assert mesh.cell_faces.shape == (8, 6)


def test_no_duplicate_faces(mesh):
    assert len(np.unique(mesh.faces, axis=0)) == mesh.faces.shape[0]


def test_neighbour_pairs_unique_and_sorted(mesh):
    pairs = mesh.neighbours
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert len(np.unique(pairs, axis=0)) == pairs.shape[0]
    # 12 inner faces ↔ 12 neighbour pairs
    assert pairs.shape[0] == 12
