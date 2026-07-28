"""Plan view on tilted pillars — coordinate collapse versus pillar identity."""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured._toy import build_toy_cpg
from cpg2unstructured.corner_point import CornerPointGrid
from cpg2unstructured.cpg import from_grdecl, pillar_of_node
from cpg2unstructured.planview import projected_topology
from cpg2unstructured.unstructured import build_topology

NX = NY = 6
NZ = 4
EXPECTED = (NX + 1) * (NY + 1)


def tilted(slant: float):
    toy = build_toy_cpg(nx=NX, ny=NY, nz=NZ, dx=100.0, dy=100.0, dz=10.0)
    coord = toy.coord.copy()
    coord[..., 3] += slant                       # shift every pillar's bottom x
    g = CornerPointGrid(
        cart_dims=(NX, NY, NZ),
        coord=coord.transpose(1, 0, 2).reshape(-1, 6),
        zcorn=toy.zcorn.transpose(2, 1, 0).ravel(),
        actnum=toy.actnum.transpose(2, 1, 0).ravel().astype(np.int8),
    )
    cube = from_grdecl(g)
    return cube, build_topology(cube)


@pytest.mark.parametrize("slant", [0.0, 5.0, 50.0])
def test_pillar_identity_collapses_correctly_at_any_tilt(slant):
    cube, grid = tilted(slant)
    topo = projected_topology(grid, "xy", pillar_of_node=pillar_of_node(cube))
    assert topo.nodes2d.shape[0] == EXPECTED


def test_coordinate_collapse_fails_once_pillars_tilt():
    """Pins the defect this replaces: it does not raise, it returns nonsense.

    5 m of tilt on a 100 m cell is enough for every node in every layer to become
    its own projected node — the ratio is exactly NZ+1, i.e. nothing collapsed.
    """
    _, flat = tilted(0.0)
    _, leaning = tilted(5.0)
    assert projected_topology(flat, "xy").nodes2d.shape[0] == EXPECTED
    bad = projected_topology(leaning, "xy").nodes2d.shape[0]
    assert bad == EXPECTED * (NZ + 1), "the failure mode is a clean NZ+1 blow-up"


def test_edges_are_unaffected_by_tilt_under_identity_collapse():
    cube_a, grid_a = tilted(0.0)
    cube_b, grid_b = tilted(50.0)
    ta = projected_topology(grid_a, "xy", pillar_of_node=pillar_of_node(cube_a))
    tb = projected_topology(grid_b, "xy", pillar_of_node=pillar_of_node(cube_b))
    assert ta.edges.shape == tb.edges.shape


def test_pillar_of_node_covers_every_node():
    cube, _ = tilted(20.0)
    pon = pillar_of_node(cube)
    assert pon.shape[0] == cube.nodes.shape[0]
    assert (pon >= 0).all(), "every node of an all-active grid belongs to a pillar"
    assert len(np.unique(pon)) == EXPECTED


def test_wrong_length_pillar_map_is_rejected():
    cube, grid = tilted(10.0)
    with pytest.raises(ValueError, match="one entry per node"):
        projected_topology(grid, "xy", pillar_of_node=np.zeros(3, dtype=np.int64))
