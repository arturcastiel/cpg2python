"""Tests for face->edge incidence (∂₂) + face/cell areas and centroids."""
from __future__ import annotations

import numpy as np

from cpg2unstructured import (
    cell_centroids,
    face_areas,
    face_centroids,
    face_edges,
    mesh_edges,
)


def test_face_edges_shape_and_range(mesh_3x3x2):
    edges = mesh_edges(mesh_3x3x2)
    fe = face_edges(mesh_3x3x2, edges)
    nf = mesh_3x3x2.faces.nodes.shape[0]
    assert fe.shape == (nf, 4)
    assert fe.min() >= 0 and fe.max() < edges.shape[0]


def test_face_edges_endpoints_subset_of_face_nodes(mesh_3x3x2):
    edges = mesh_edges(mesh_3x3x2)
    fe = face_edges(mesh_3x3x2, edges)
    fn = mesh_3x3x2.faces.nodes
    for f in range(fn.shape[0]):
        face_node_set = set(fn[f].tolist())
        for e in fe[f]:
            a, b = edges[e]
            assert a in face_node_set and b in face_node_set
        # 4 distinct edges per face
        assert len(set(fe[f].tolist())) == 4


def test_boundary_operator_consistency(mesh_3x3x2):
    # Every edge used by a face is used by >=1 face; interior edges by >=2.
    fe = face_edges(mesh_3x3x2)
    counts = np.bincount(fe.ravel())
    assert counts.max() >= 2  # shared edges exist


def test_face_areas_unit(mesh_3x3x2):
    a = face_areas(mesh_3x3x2)
    assert a.shape == (mesh_3x3x2.faces.nodes.shape[0],)
    assert np.allclose(a, 1.0)  # unit cartesian faces
    assert (a > 0).all()


def test_centroid_shapes(mesh_3x3x2):
    fc = face_centroids(mesh_3x3x2)
    cc = cell_centroids(mesh_3x3x2)
    assert fc.shape == (mesh_3x3x2.faces.nodes.shape[0], 3)
    assert cc.shape == (mesh_3x3x2.n_cells, 3)
