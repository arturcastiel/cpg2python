"""3-corner fault polygons stored as degenerate quads, and the grown node table.

Step B of the Norne ingest work. No shipped fixture produces a 3-corner fault
polygon (they all have flat, rigidly-offset blocks), so the packing rule and its
downstream consequences are pinned here directly rather than through a deck.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured._errors import NonMatchingZError
from cpg2unstructured.cpg import from_grdecl, pillar_of_node
from cpg2unstructured.geometry import face_geometry
from cpg2unstructured.grdecl_reader import read_grdecl
from cpg2unstructured.planview import projected_topology
from cpg2unstructured.unstructured import (
    Faces,
    _polygons_to_quads,
    build_topology,
    find_ij_faces_and_nodes,
)

# ── packing ────────────────────────────────────────────────────────────────


def test_triangle_becomes_degenerate_quad():
    """A 3-corner polygon (a, b, c) is stored as (a, b, c, c) then sorted.

    It is the polygon's LAST corner that repeats, not its largest id, so the
    duplicate pair lands wherever that id sorts — here c=5 gives (2, 5, 5, 7),
    with the pair in slots (1, 2) rather than (2, 3).
    """
    quads = _polygons_to_quads(
        np.array([3], dtype=np.int64), np.array([7, 2, 5], dtype=np.int64), "t",
    )
    assert quads.tolist() == [[2, 5, 5, 7]]
    assert len(set(quads[0].tolist())) == 3


def test_quad_is_unchanged_by_packing():
    quads = _polygons_to_quads(
        np.array([4], dtype=np.int64), np.array([9, 1, 4, 3], dtype=np.int64), "t",
    )
    assert quads.tolist() == [[1, 3, 4, 9]]


def test_mixed_batch_packs_row_by_row():
    quads = _polygons_to_quads(
        np.array([3, 4, 3], dtype=np.int64),
        np.array([1, 2, 3, 10, 11, 12, 13, 20, 21, 22], dtype=np.int64),
        "t",
    )
    assert quads.tolist() == [[1, 2, 3, 3], [10, 11, 12, 13], [20, 21, 22, 22]]


@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_five_or_more_corners_still_raises(n):
    """Step C, not Step B: 5+ corners cannot fit a (Nfaces, 4) array."""
    with pytest.raises(NonMatchingZError, match="3- and 4-corner"):
        _polygons_to_quads(
            np.array([n], dtype=np.int64), np.arange(n, dtype=np.int64), "the stack",
        )


# ── geometry: the degenerate quad must not read as a zero-area face ────────


def test_degenerate_quad_gets_the_true_triangle_area():
    """(a, b, c, c) must measure as triangle abc, not as a collapsed quad.

    The quad path takes the two longest of the 6 pairwise distances as
    diagonals. On a degenerate quad the repeated vertex duplicates a distance,
    so when the repeat lies on the longest edge both "diagonals" are the SAME
    vector and the cross product vanishes. Norne's i=4-5, j=2 stack hit that on
    22 of 22 triangles.
    """
    # Right triangle with legs 3 and 4 in the z=0 plane → area 6.
    nodes = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float64,
    )
    faces = Faces(
        nodes=np.array([[0, 1, 2, 2]], dtype=np.int64),
        neighbours=np.array([[0, -1]], dtype=np.int64),
        direction=np.zeros(1, dtype=np.int8),
    )
    fg = face_geometry(nodes, faces, np.zeros((1, 8), dtype=np.int64))
    assert fg.areas[0] == pytest.approx(6.0)
    # Centroid is the triangle's, not the c-weighted quad mean (1.0, 2.0, 0).
    assert fg.centroids[0] == pytest.approx([1.0, 4.0 / 3.0, 0.0])
    assert np.linalg.norm(fg.normals[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("repeat_at", [0, 1, 2])
def test_triangle_area_is_right_wherever_the_repeat_sorts(repeat_at):
    """The duplicate pair can land in slots (0,1), (1,2) or (2,3) depending on
    how the repeated node id sorts against the other two."""
    ids = [[5, 5, 7, 9], [5, 7, 7, 9], [5, 7, 9, 9]][repeat_at]
    xyz = {5: [0.0, 0.0, 0.0], 7: [3.0, 0.0, 0.0], 9: [0.0, 4.0, 0.0]}
    nodes = np.zeros((10, 3), dtype=np.float64)
    for k, p in xyz.items():
        nodes[k] = p
    faces = Faces(
        nodes=np.array([ids], dtype=np.int64),
        neighbours=np.array([[0, -1]], dtype=np.int64),
        direction=np.zeros(1, dtype=np.int8),
    )
    fg = face_geometry(nodes, faces, np.zeros((1, 8), dtype=np.int64))
    assert fg.areas[0] == pytest.approx(6.0)


def test_real_quads_are_untouched_by_the_degenerate_branch(mesh_2x2x2):
    """Unit square face still measures 1.0 — the branch must not fire."""
    fg = face_geometry(
        mesh_2x2x2.nodes, mesh_2x2x2.faces, mesh_2x2x2.cell_corners,
    )
    assert (fg.areas > 0).all()
    assert fg.areas == pytest.approx(1.0)


# ── grown node table ───────────────────────────────────────────────────────


def test_unfaulted_deck_keeps_cube_node_table(fixtures):
    """No fault, no appended nodes — the grid must carry cube.nodes as-is."""
    cube = from_grdecl(read_grdecl(fixtures / "simple_2x2x2.grdecl"))
    grid = build_topology(cube)
    assert grid.nodes.shape == cube.nodes.shape
    assert np.array_equal(grid.nodes, cube.nodes)


def test_face_node_ids_index_the_grid_table(fixtures):
    """Every face id must be addressable in the table the grid ships."""
    for name in ("simple_faulted_2x2x2.grdecl", "simple_faulted_3x3x2.grdecl"):
        grid = build_topology(from_grdecl(read_grdecl(fixtures / name)))
        assert grid.faces.nodes.max() < grid.nodes.shape[0]
        assert grid.faces.nodes.min() >= 0


def test_ij_finder_threads_the_table_so_ids_stay_disjoint(fixtures):
    """find_i/j_faces each append starting at the end of the table they were
    given; find_ij_faces_and_nodes must chain them, not restart both."""
    cube = from_grdecl(read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl"))
    faces, points = find_ij_faces_and_nodes(cube)
    assert points.shape[0] >= cube.nodes.shape[0]
    assert faces.nodes.max() < points.shape[0]


# ── pillar_of_node tolerates the pillar-less appended nodes ────────────────


def test_pillar_of_node_pads_to_a_grown_table(fixtures):
    cube = from_grdecl(read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl"))
    n = cube.nodes.shape[0]
    pon = pillar_of_node(cube, n + 5)
    assert pon.shape[0] == n + 5
    assert (pon[n:] == -1).all()                      # appended → no pillar
    assert np.array_equal(pon[:n], pillar_of_node(cube))


def test_pillar_of_node_never_shrinks(fixtures):
    cube = from_grdecl(read_grdecl(fixtures / "simple_2x2x2.grdecl"))
    n = cube.nodes.shape[0]
    assert pillar_of_node(cube, 3).shape[0] == n


def test_projected_topology_does_not_weld_the_pillar_less_nodes(fixtures):
    """-1 means 'no pillar owns me', not 'we are all one pillar'. A plain
    np.unique groups every -1 together and welds unrelated nodes into a single
    projected node."""
    cube = from_grdecl(read_grdecl(fixtures / "simple_2x2x2.grdecl"))
    grid = build_topology(cube)
    n = grid.nodes.shape[0]
    baseline = projected_topology(grid, "xy", pillar_of_node=pillar_of_node(cube, n))

    pon = pillar_of_node(cube, n)
    free = np.array([0, 1, 2])                        # pretend 3 appended nodes
    pon[free] = -1
    got = projected_topology(grid, "xy", pillar_of_node=pon)
    # Each freed node keeps its own identity, so they land on 3 distinct
    # projected nodes — not one.
    assert len(set(got.node_of_grid_node[free].tolist())) == 3
    assert got.nodes2d.shape[0] >= baseline.nodes2d.shape[0]
