"""csgraph index width — scipy's csgraph is compiled against int32 (ITYPE).

``csr_array((data, (rows, cols)))`` preserves the integer width of ``rows`` /
``cols``. Ours are int64 (node and cell ids), and scipy >= 1.14 rejects that
outright with::

    ValueError: Buffer dtype mismatch, expected 'const int' but got 'long'

Every CSR handed to ``scipy.sparse.csgraph`` must therefore carry int32 index
arrays. These tests pin that at the two construction sites.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured._toy import build_toy_cpg
from cpg2unstructured.corner_point import CornerPointGrid
from cpg2unstructured.cpg import from_grdecl
from cpg2unstructured.mesh3d import build_mesh_graph_3d
from cpg2unstructured.unstructured import build_topology

sp = pytest.importorskip("scipy.sparse")
csgraph = pytest.importorskip("scipy.sparse.csgraph")

from cpg2unstructured.graph import (  # noqa: E402  (after importorskip)
    cell_adjacency_csgraph,
    csgraph_indices,
)


def _toy_grid(nx=3, ny=3, nz=2):
    toy = build_toy_cpg(nx=nx, ny=ny, nz=nz)
    g = CornerPointGrid(
        cart_dims=(toy.nx, toy.ny, toy.nz),
        coord=toy.coord.transpose(1, 0, 2).reshape(-1, 6),
        zcorn=toy.zcorn.transpose(2, 1, 0).ravel(),
        actnum=toy.actnum.transpose(2, 1, 0).ravel().astype(np.int8),
        source="grdecl",
    )
    return build_topology(from_grdecl(g))


def test_csgraph_indices_casts_both_arrays():
    csr = sp.csr_array(
        (np.ones(2), (np.array([0, 1], dtype=np.int64), np.array([1, 2], dtype=np.int64))),
        shape=(3, 3),
    )
    assert csr.indices.dtype == np.int64
    out = csgraph_indices(csr)
    assert out.indices.dtype == np.int32
    assert out.indptr.dtype == np.int32


def test_csgraph_indices_does_not_mutate_input():
    csr = sp.csr_array(
        (np.ones(2), (np.array([0, 1], dtype=np.int64), np.array([1, 2], dtype=np.int64))),
        shape=(3, 3),
    )
    csgraph_indices(csr)
    assert csr.indices.dtype == np.int64, "input must be left untouched"


def test_cell_adjacency_csgraph_is_dijkstra_ready():
    grid = _toy_grid()
    csr = cell_adjacency_csgraph(grid)
    assert csr.indices.dtype == np.int32
    assert csr.indptr.dtype == np.int32
    csgraph.dijkstra(csr, directed=False, indices=0)  # must not raise


def test_mesh_graph_3d_is_dijkstra_ready():
    grid = _toy_grid()
    mg = build_mesh_graph_3d(grid)
    assert mg.csr.indices.dtype == np.int32
    assert mg.csr.indptr.dtype == np.int32
    csgraph.dijkstra(mg.csr, directed=False, indices=0)  # must not raise
