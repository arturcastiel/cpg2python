from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured import (
    CellGeometry,
    FaceGeometry,
    build_topology,
    cell_geometry,
    face_geometry,
    from_grdecl,
    read_grdecl,
)

FIXTURES = Path(__file__).parent / "fixtures"

_UNIT_CUBE_DECK = """\
SPECGRID
  1 1 1 1 F /

COORD
  0 0 0  0 0 1
  1 0 0  1 0 1
  0 1 0  0 1 1
  1 1 0  1 1 1 /

ZCORN
  4*0  4*1 /

ACTNUM
  1 /
"""


@pytest.fixture
def unit_cube_mesh(tmp_path):
    p = tmp_path / "unit_cube.grdecl"
    p.write_text(_UNIT_CUBE_DECK)
    cube = from_grdecl(read_grdecl(p))
    return build_topology(cube)


@pytest.fixture
def cube_mesh_2x2x2():
    cube = from_grdecl(read_grdecl(FIXTURES / "simple_2x2x2.grdecl"))
    return build_topology(cube)


# ────────────────────────────────────────────────────────────────────
#   — geometry (centroids, areas, volumes)
# ────────────────────────────────────────────────────────────────────


def test_unit_cube_face_areas(unit_cube_mesh):
    fg = face_geometry(unit_cube_mesh.nodes, unit_cube_mesh.faces,
                       unit_cube_mesh.cell_corners)
    # 6 faces, each unit area on a 1x1x1 cube.
    assert fg.areas.shape == (6,)
    np.testing.assert_allclose(fg.areas, 1.0, atol=1e-12)


def test_unit_cube_face_normals_unit_length(unit_cube_mesh):
    fg = face_geometry(unit_cube_mesh.nodes, unit_cube_mesh.faces,
                       unit_cube_mesh.cell_corners)
    norms = np.linalg.norm(fg.normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_unit_cube_volume(unit_cube_mesh):
    fg = face_geometry(unit_cube_mesh.nodes, unit_cube_mesh.faces,
                       unit_cube_mesh.cell_corners)
    cg = cell_geometry(unit_cube_mesh.nodes, unit_cube_mesh.faces,
                       unit_cube_mesh.cell_faces, fg)
    np.testing.assert_allclose(cg.volumes, 1.0, atol=1e-12)


def test_unit_cube_centroid(unit_cube_mesh):
    fg = face_geometry(unit_cube_mesh.nodes, unit_cube_mesh.faces,
                       unit_cube_mesh.cell_corners)
    cg = cell_geometry(unit_cube_mesh.nodes, unit_cube_mesh.faces,
                       unit_cube_mesh.cell_faces, fg)
    np.testing.assert_allclose(cg.centroids, [[0.5, 0.5, 0.5]], atol=1e-12)


def test_2x2x2_total_volume(cube_mesh_2x2x2):
    fg = face_geometry(cube_mesh_2x2x2.nodes, cube_mesh_2x2x2.faces,
                       cube_mesh_2x2x2.cell_corners)
    cg = cell_geometry(cube_mesh_2x2x2.nodes, cube_mesh_2x2x2.faces,
                       cube_mesh_2x2x2.cell_faces, fg)
    # 8 unit cubes side by side → total volume 8.
    np.testing.assert_allclose(cg.volumes.sum(), 8.0, atol=1e-12)
    # Each individual cell should be exactly 1.
    np.testing.assert_allclose(cg.volumes, 1.0, atol=1e-12)


def test_2x2x2_all_normals_unit_length(cube_mesh_2x2x2):
    fg = face_geometry(cube_mesh_2x2x2.nodes, cube_mesh_2x2x2.faces,
                       cube_mesh_2x2x2.cell_corners)
    norms = np.linalg.norm(fg.normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_2x2x2_inner_normals_point_low_to_high(cube_mesh_2x2x2):
    # For each inner face, normal should point from cell0 (low neighbour)
    # toward cell1 (high neighbour).
    fg = face_geometry(cube_mesh_2x2x2.nodes, cube_mesh_2x2x2.faces,
                       cube_mesh_2x2x2.cell_corners)
    nb = cube_mesh_2x2x2.faces.neighbours
    inner_mask = nb[:, 1] != -1
    # Use cell-corner-mean as centroid estimate for both cells of the face.
    cc = cube_mesh_2x2x2.nodes[cube_mesh_2x2x2.cell_corners].mean(axis=1)
    c0 = cc[nb[inner_mask, 0]]
    c1 = cc[nb[inner_mask, 1]]
    direction_low_to_high = c1 - c0
    dots = (direction_low_to_high * fg.normals[inner_mask]).sum(axis=1)
    assert (dots > 0).all(), "inner normals should point from cell0 to cell1"


def test_2x2x2_boundary_normals_point_outward(cube_mesh_2x2x2):
    fg = face_geometry(cube_mesh_2x2x2.nodes, cube_mesh_2x2x2.faces,
                       cube_mesh_2x2x2.cell_corners)
    nb = cube_mesh_2x2x2.faces.neighbours
    bnd_mask = nb[:, 1] == -1
    cc = cube_mesh_2x2x2.nodes[cube_mesh_2x2x2.cell_corners].mean(axis=1)
    c0 = cc[nb[bnd_mask, 0]]
    outward = fg.centroids[bnd_mask] - c0
    dots = (outward * fg.normals[bnd_mask]).sum(axis=1)
    assert (dots > 0).all(), "boundary normals should point outward from cell0"


def test_3x3x2_unit_volumes():
    cube = from_grdecl(read_grdecl(FIXTURES / "simple_3x3x2.grdecl"))
    mesh = build_topology(cube)
    fg = face_geometry(mesh.nodes, mesh.faces, mesh.cell_corners)
    cg = cell_geometry(mesh.nodes, mesh.faces, mesh.cell_faces, fg)
    # 18 unit-volume cells.
    assert cg.volumes.shape == (18,)
    np.testing.assert_allclose(cg.volumes, 1.0, atol=1e-12)
    np.testing.assert_allclose(cg.volumes.sum(), 18.0, atol=1e-12)


def test_geometry_dataclass_fields(unit_cube_mesh):
    fg = face_geometry(unit_cube_mesh.nodes, unit_cube_mesh.faces,
                       unit_cube_mesh.cell_corners)
    cg = cell_geometry(unit_cube_mesh.nodes, unit_cube_mesh.faces,
                       unit_cube_mesh.cell_faces, fg)
    assert isinstance(fg, FaceGeometry)
    assert isinstance(cg, CellGeometry)
    assert fg.centroids.dtype == np.float64
    assert fg.areas.dtype == np.float64
    assert fg.normals.dtype == np.float64
    assert cg.centroids.dtype == np.float64
    assert cg.volumes.dtype == np.float64


def _single_face_geometry(points, quad):
    """face_geometry on one face; a 5th point is the cell-centre stand-in."""
    from cpg2unstructured import Faces
    nodes = np.vstack([points, [[-1.0, -1.0, -1.0]]])
    faces = Faces(
        nodes=np.array([sorted(quad)], dtype=np.int64),
        neighbours=np.array([[0, -1]], dtype=np.int64),
        direction=np.array([1], dtype=np.int8),
    )
    cell_corners = np.full((1, 8), nodes.shape[0] - 1, dtype=np.int64)
    return face_geometry(nodes, faces, cell_corners)


def test_trapezoid_face_area():
    """Parallel sides 1 and 3, height 1 → area 2. A long parallel side
    (3) beats the short diagonal (√2), so the diagonals are not the two
    longest node pairs — as on j-faces next to tilted pillars."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [3, 0, 1]], dtype=float)
    fg = _single_face_geometry(pts, [0, 1, 2, 3])
    assert fg.areas[0] == pytest.approx(2.0)
    np.testing.assert_allclose(np.abs(fg.normals[0]), [0, 1, 0])


def test_triangle_face_with_repeated_node():
    """A triangular fault face is stored as a quad with one node
    repeated: legs 1 and 1 → area 0.5, centroid at the vertex mean."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    fg = _single_face_geometry(pts, [0, 1, 2, 2])
    assert fg.areas[0] == pytest.approx(0.5)
    np.testing.assert_allclose(fg.centroids[0], [1 / 3, 0, 1 / 3])
