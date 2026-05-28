"""Tests for the projected (2.5D) plan-view topology."""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import ProjectedTopology, projected_topology


def test_xy_node_and_edge_counts(mesh_3x3x2):
    t = projected_topology(mesh_3x3x2, "xy")
    assert isinstance(t, ProjectedTopology)
    assert t.plane == "xy"
    # 4x4 pillar corners collapse out the 2 z-layers.
    assert t.nodes2d.shape == (16, 2)
    # (nx+1)*ny i-faces + (ny+1)*nx j-faces = 12 + 12.
    assert t.edges.shape == (24, 2)
    # Unit spacing in the fixture.
    assert np.allclose(t.edge_len, 1.0)


def test_node_map_covers_all_3d_nodes(mesh_3x3x2):
    t = projected_topology(mesh_3x3x2, "xy")
    assert t.node_of_grid_node.shape == (mesh_3x3x2.nodes.shape[0],)
    # Every projected node id 0..15 is used; nothing out of range.
    assert set(np.unique(t.node_of_grid_node).tolist()) == set(range(16))
    # Each projected node coordinate matches the 3D node it came from (xy).
    xy = mesh_3x3x2.nodes[:, (0, 1)]
    assert np.allclose(xy, t.nodes2d[t.node_of_grid_node])


def test_xy_edge_dir_split(mesh_3x3x2):
    t = projected_topology(mesh_3x3x2, "xy")
    # 12 X-faces (normal to i) and 12 Y-faces (normal to j).
    assert int((t.edge_dir == 0).sum()) == 12
    assert int((t.edge_dir == 1).sum()) == 12
    assert set(np.unique(t.edge_dir).tolist()) <= {0, 1}


def test_xy_interior_adjacency(mesh_3x3x2):
    t = projected_topology(mesh_3x3x2, "xy")
    interior = (t.edge_uv >= 0).all(axis=(1, 2))
    assert interior.any()
    for e in np.nonzero(interior)[0]:
        uv = t.edge_uv[e]
        d = int(t.edge_dir[e])
        diff = np.abs(uv[0] - uv[1])
        # Adjacent cells differ by exactly 1 along the face-normal axis only.
        assert diff[d] == 1
        assert diff[1 - d] == 0
    # Domain boundary produces edges with a (-1,-1) side.
    assert (t.edge_uv < 0).any()


def test_edges_sorted_and_unique(mesh_3x3x2):
    t = projected_topology(mesh_3x3x2, "xy")
    assert (t.edges[:, 0] < t.edges[:, 1]).all()
    assert np.unique(t.edges, axis=0).shape[0] == t.edges.shape[0]


@pytest.mark.parametrize(
    "plane, n_nodes, n_edges",
    [
        ("xy", 16, 24),
        ("xz", 12, 17),
        ("yz", 12, 17),
    ],
)
def test_all_planes(mesh_3x3x2, plane, n_nodes, n_edges):
    t = projected_topology(mesh_3x3x2, plane)
    assert t.plane == plane
    assert t.nodes2d.shape == (n_nodes, 2)
    assert t.edges.shape == (n_edges, 2)
    assert np.isfinite(t.edge_len).all()
    assert (t.edge_len > 0).all()
    assert t.edge_uv.shape == (n_edges, 2, 2)


def test_unknown_plane_raises(mesh_3x3x2):
    with pytest.raises(ValueError, match="unknown plane"):
        projected_topology(mesh_3x3x2, "ab")


def test_smoke_2x2x2(mesh_2x2x2):
    t = projected_topology(mesh_2x2x2, "xy")
    assert t.nodes2d.shape == (9, 2)
    assert t.edges.shape == (12, 2)
    assert np.allclose(t.edge_len, 1.0)
