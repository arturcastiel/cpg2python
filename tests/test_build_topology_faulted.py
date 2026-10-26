"""End-to-end tests for ``build_topology`` on faulted decks
.

The headline outcome of  ``build_topology`` no longer raises on
i- or j-direction faulted decks. The i/j fault dispatch from
/ + the removal of the top-level guard in ``build_topology``
mean a faulted deck flows end-to-end through the topology builder.

k-direction faults still raise — k-fault dispatch is deferred to a
follow-up because the 2-pillar sweep-line algorithm doesn't apply
directly to the 4-pillar geometry of k-faces.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import (
    UnstructuredGrid,
    build_topology,
    from_grdecl,
    read_grdecl,
)
from cpg2unstructured._errors import NonMatchingZError

# ── headline: build_topology on i-faulted deck ─────────────────────


def test_build_topology_on_simple_faulted_2x2x2_i(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert isinstance(grid, UnstructuredGrid)
    assert grid.n_cells == 8     # all-active 2x2x2
    # Sanity on outputs. The 4 cells on the fault each touch 2 faces on
    # their fault side (see test_fault_juxtaposition_*), so cell_faces
    # gains one padding column.
    assert grid.cell_faces.shape == (8, 7)
    assert grid.faces.nodes.shape[1] == 4
    assert grid.faces.neighbours.shape[1] == 2
    # Direction labels: 0 = i, 1 = j, 2 = k
    assert set(np.unique(grid.faces.direction).tolist()) <= {0, 1, 2}


def test_build_topology_on_simple_faulted_2x2x2_j(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert grid.n_cells == 8
    assert grid.cell_faces.shape == (8, 7)


# ──  regression: 3x3x2 multi-column i-fault ────────────────────


def test_build_topology_on_simple_faulted_3x3x2(fixtures):
    """3x3x2 multi-column i-fault: +0.6 Z offset on the i=2 column.
    Exercises the find_i_faces dispatch on a multi-j-row grid
    (the 2x2x2 fixtures only have 2 j-rows; this has 3).

    The fault plane spans 3 j-rows; the +0.6 offset makes the lower
    i=1 cell straddle both i=2 cells, so each stack yields 3 conns
    → 9 inner fault faces, plus 2 exposed boundary pieces per stack."""
    g = read_grdecl(fixtures / "simple_faulted_3x3x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert isinstance(grid, UnstructuredGrid)
    assert grid.n_cells == 18         # all-active 3x3x2
    assert grid.cell_faces.shape == (18, 7)
    assert grid.faces.nodes.shape[1] == 4
    assert grid.faces.neighbours.shape[1] == 2


def test_simple_faulted_3x3x2_i_face_dispatch_count(fixtures):
    """find_i_faces called directly on the 3x3x2 fault deck produces
    the expected mix of matching + fault faces.

        i-plane 0  (boundary)        : 3 j × 2 k = 6
        i-plane 1  (matching, i=0..1): 3 j × 2 k = 6
        i-plane 2  (FAULTED, i=1..2) : 3 j-stacks × 5 faces = 15
        i-plane 3  (boundary)        : 3 j × 2 k = 6

        Total i-faces                : 33

        Per faulted stack (A = i=1 column z∈[0,1],[1,2]; B = i=2 column
        z∈[0.6,1.6],[1.6,2.6]): 3 inner faces (A0-B0 [0.6,1],
        A1-B0 [1,1.6], A1-B1 [1.6,2]) + 2 exposed boundary pieces
        (A0 on [0,0.6], B1 on [2,2.6]).
    """
    from cpg2unstructured.unstructured import find_i_faces
    g = read_grdecl(fixtures / "simple_faulted_3x3x2.grdecl")
    cube = from_grdecl(g)
    faces = find_i_faces(cube)
    assert faces.nodes.shape == (33, 4)
    # All direction=0 (i-faces)
    np.testing.assert_array_equal(
        faces.direction, np.zeros(33, dtype=np.int8),
    )
    # All neighbours in valid range
    nbs = faces.neighbours
    assert nbs.min() >= -1
    assert nbs.max() < 18


def test_simple_faulted_3x3x2_neighbour_convention(fixtures):
    """`_move_minus_one_to_col1` invariant: inner faces have smaller
    id in col 0; boundary faces have -1 in col 1."""
    g = read_grdecl(fixtures / "simple_faulted_3x3x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    for nb in grid.faces.neighbours:
        if nb[1] == -1:
            assert nb[0] != -1, f"boundary face should have valid col 0: {nb}"
        else:
            assert nb[0] <= nb[1], f"inner: col 0 should be ≤ col 1: {nb}"


# ── still raises on k-faults ───────────────────────────────────────


def test_build_topology_raises_on_k_fault(tmp_path, fixtures):
    """Synthetic k-fault: simple_2x2x2 deck with one ZCORN value
    perturbed at the k=0 ↔ k=1 boundary (z-slot 1 on cell (0,0,0)
    set to 1.5 instead of 1)."""
    src = (fixtures / "simple_2x2x2.grdecl").read_text()
    # Original ZCORN block:
    #   ZCORN
    #     16*0   16*1
    #     16*1   16*2 /
    # The k=0-bot row is the 16*1 in the first ZCORN line. Replace
    # that with one explicit value of 1.5 + 15*1 to break matching-Z
    # between k=0 and k=1 at slot (i_slot=0, j_slot=0, k_slot=1).
    src = src.replace(
        "16*0   16*1\n  16*1   16*2 /",
        "16*0   1.5 15*1\n  16*1   16*2 /",
    )
    p = tmp_path / "k_fault.grdecl"
    p.write_text(src)
    g = read_grdecl(p)
    cube = from_grdecl(g)
    with pytest.raises(NonMatchingZError, match="k-direction"):
        build_topology(cube)


# ──  regression: 3x3x2 multi-column j-fault (mirror of ) ───


def test_build_topology_on_simple_faulted_3x3x2_jfault(fixtures):
    """3x3x2 multi-column j-fault: +0.6 Z offset on the j=2 row.
    The j-direction analogue of the  i-fault fixture — exercises
    find_j_faces dispatch on a multi-column grid (3 i-columns)."""
    g = read_grdecl(fixtures / "simple_faulted_3x3x2_jfault.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert isinstance(grid, UnstructuredGrid)
    assert grid.n_cells == 18         # all-active 3x3x2
    assert grid.cell_faces.shape == (18, 7)
    assert grid.faces.nodes.shape[1] == 4
    assert grid.faces.neighbours.shape[1] == 2


def test_simple_faulted_3x3x2_jfault_j_face_dispatch_count(fixtures):
    """find_j_faces on the 3x3x2 j-fault deck — symmetric to 

        j-plane 0  (boundary)        : 3 i × 2 k = 6
        j-plane 1  (matching, j=0..1): 3 i × 2 k = 6
        j-plane 2  (FAULTED, j=1..2) : 3 i-stacks × 5 faces = 15
        j-plane 3  (boundary)        : 3 i × 2 k = 6

        Total j-faces                : 33

        (3 inner faces + 2 exposed boundary pieces per faulted stack,
         as in the i-fault fixture.)
    """
    from cpg2unstructured.unstructured import find_j_faces
    g = read_grdecl(fixtures / "simple_faulted_3x3x2_jfault.grdecl")
    cube = from_grdecl(g)
    faces = find_j_faces(cube)
    assert faces.nodes.shape == (33, 4)
    # All direction=1 (j-faces)
    np.testing.assert_array_equal(
        faces.direction, np.ones(33, dtype=np.int8),
    )
    # All neighbours in valid range
    nbs = faces.neighbours
    assert nbs.min() >= -1
    assert nbs.max() < 18


# ── D1: fault-aware topology — juxtaposition, closure, volumes ─────


def _closure_and_geometry(grid):
    """Per-cell sum of outward n·A over all of the cell's faces (must be
    ~0 for a closed surface), plus face and cell geometry."""
    from cpg2unstructured.geometry import cell_geometry, face_geometry
    fg = face_geometry(grid.nodes, grid.faces, grid.cell_corners)
    cg = cell_geometry(grid.nodes, grid.faces, grid.cell_faces, fg)
    nb = grid.faces.neighbours
    closure = np.zeros((grid.n_cells, 3))
    for c, row in enumerate(grid.cell_faces):
        for f in row[row >= 0]:
            sign = 1.0 if nb[f, 0] == c else -1.0
            closure[c] += sign * fg.normals[f] * fg.areas[f]
    return closure, fg, cg


@pytest.mark.parametrize("name", [
    "simple_faulted_2x2x2", "simple_faulted_2x2x2_j",
    "simple_faulted_3x3x2", "simple_faulted_3x3x2_jfault",
])
def test_faulted_cells_closed_with_unit_volumes(fixtures, name):
    """Every cell of the rigid-offset fixtures is a unit cube: its faces
    (several on the fault side) must close and enclose volume 1."""
    grid = build_topology(from_grdecl(read_grdecl(fixtures / f"{name}.grdecl")))
    closure, _, cg = _closure_and_geometry(grid)
    np.testing.assert_allclose(closure, 0.0, atol=1e-12)
    np.testing.assert_allclose(cg.volumes, 1.0, rtol=1e-12)


def test_fault_juxtaposition_simple_faulted_2x2x2(fixtures):
    """Hand-derived per j-stack. A = i=0 column z∈[0,1],[1,2];
    B = i=1 column z∈[0.5,1.5],[1.5,2.5]. Inner faces A0-B0 [0.5,1],
    A1-B0 [1,1.5], A1-B1 [1.5,2] (area 0.5 each); exposed pieces A0 on
    [0,0.5] and B1 on [2,2.5] (boundary, area 0.5 each). The old
    same-k pairing (A0-B0, A1-B1 with full unit faces) is gone."""
    grid = build_topology(from_grdecl(read_grdecl(
        fixtures / "simple_faulted_2x2x2.grdecl")))
    _, fg, _ = _closure_and_geometry(grid)
    ijk = [tuple(map(int, r)) for r in grid.original_ijk]
    nb = grid.faces.neighbours
    fault = (grid.faces.direction == 0) & np.isclose(fg.areas, 0.5)
    inner = set()
    exposed = set()
    for f in np.nonzero(fault)[0]:
        a, b = int(nb[f, 0]), int(nb[f, 1])
        if b == -1:
            exposed.add(ijk[a])
        else:
            inner.add(tuple(sorted((ijk[a], ijk[b]))))
    for j in (0, 1):
        assert {((0, j, 0), (1, j, 0)), ((0, j, 1), (1, j, 0)),
                ((0, j, 1), (1, j, 1))} <= inner
        assert {(0, j, 0), (1, j, 1)} <= exposed
    assert len(inner) == 6
    assert len(exposed) == 4
    # No unit-area inner i-face is left: every inner i-face is a fault face.
    inner_i = (grid.faces.direction == 0) & (nb[:, 1] != -1)
    np.testing.assert_allclose(fg.areas[inner_i], 0.5)


def test_fault_simple_tip_triangle_is_exposed_boundary(fixtures):
    """fault_simple moves one corner of cell (1,0,0) down by 0.5 on one
    pillar. The gap it opens on cell (0,0,0)'s +i side is a triangle
    (legs 1 × 0.5, area 0.25), returned as a boundary face. Closing it
    makes the volume exact: 8 - 0.5/4 = 7.875."""
    grid = build_topology(from_grdecl(read_grdecl(fixtures / "fault_simple.grdecl")))
    closure, fg, cg = _closure_and_geometry(grid)
    np.testing.assert_allclose(closure, 0.0, atol=1e-12)
    assert cg.volumes.sum() == pytest.approx(7.875, rel=1e-12)
    tri = np.array([len(set(q.tolist())) == 3 for q in grid.faces.nodes])
    assert tri.sum() == 1
    f = int(np.nonzero(tri)[0][0])
    assert grid.faces.neighbours[f, 1] == -1
    assert fg.areas[f] == pytest.approx(0.25)


def _tilted_throw_deck(tmp_path, throw=1.5, shift=0.5):
    """2x1x3 deck on parallel pillars tilted in x (x = x0 + shift*z),
    cells 4 × 1 × 1. The i=1 column is slid down its pillars by
    ``throw``. Every cell is a sheared box of volume 4."""
    nx, ny, nz = 2, 1, 3
    coord = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord += [4.0 * i, float(j), 0.0, 4.0 * i + 6.0 * shift, float(j), 6.0]
    zcorn = []
    for ks in range(2 * nz):
        z = float(ks // 2 + ks % 2)
        for _ in range(2 * ny):
            zcorn += [z, z, z + throw, z + throw]
    text = (
        f"SPECGRID\n {nx} {ny} {nz} 1 F /\n"
        "COORD\n " + " ".join(map(str, coord)) + " /\n"
        "ZCORN\n " + " ".join(map(str, zcorn)) + " /\n"
        f"ACTNUM\n {nx * ny * nz}*1 /\n"
    )
    p = tmp_path / "tilted_throw.grdecl"
    p.write_text(text)
    return p


def test_tilted_pillar_throw_closed_exact_volumes(tmp_path):
    """Throw 1.5 on tilted pillars. A = [0,1],[1,2],[2,3]; B =
    [1.5,2.5],[2.5,3.5],[3.5,4.5]. Inner pairs: A1-B0, A2-B0, A2-B1;
    A0 and B2 face no cell at all. Volumes must stay 4 (shear-invariant)
    — this also checks face areas of the sheared j-faces."""
    grid = build_topology(from_grdecl(read_grdecl(_tilted_throw_deck(tmp_path))))
    closure, _, cg = _closure_and_geometry(grid)
    np.testing.assert_allclose(closure, 0.0, atol=1e-10)
    np.testing.assert_allclose(cg.volumes, 4.0, rtol=1e-12)
    ijk = [tuple(map(int, r)) for r in grid.original_ijk]
    nb = grid.faces.neighbours
    inner_i = (grid.faces.direction == 0) & (nb[:, 1] != -1)
    pairs = {tuple(sorted((ijk[a], ijk[b]))) for a, b in nb[inner_i]}
    assert pairs == {((0, 0, 1), (1, 0, 0)), ((0, 0, 2), (1, 0, 0)),
                     ((0, 0, 2), (1, 0, 1))}


def test_unfaulted_cell_faces_keep_six_columns(fixtures):
    """No fault → no padding: cell_faces stays (N, 6) and fully filled."""
    grid = build_topology(from_grdecl(read_grdecl(fixtures / "simple_3x3x2.grdecl")))
    assert grid.cell_faces.shape == (18, 6)
    assert (grid.cell_faces >= 0).all()
