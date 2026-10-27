"""Fault stacks beyond the all-active, quad-only case.

Inactive cells beside a fault, pinch-outs beside a fault, and fault
polygons with edge-intersection corners (a layer boundary crossing the
inside of a fault face). Grids are built on vertical pillars so contact
areas, side areas and volumes have closed forms.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured import (
    CornerPointGrid,
    build_mesh_graph_3d,
    build_topology,
    cell_geometry,
    face_edges,
    face_geometry,
    from_grdecl,
    projected_topology,
    read_grdecl,
)
from cpg2unstructured.unstructured import Faces

DX = DY = 10.0


def _cake(nx: int, ny: int, dz: list[float]) -> np.ndarray:
    """ZCORN cube ``z[k-slot, j-slot, i-slot]`` of a flat layer cake."""
    h = np.concatenate([[0.0], np.cumsum(dz)])
    s = np.arange(2 * len(dz))
    z = h[s // 2 + s % 2]
    return np.broadcast_to(z[:, None, None], (2 * len(dz), 2 * ny, 2 * nx)).copy()


def _grid(z: np.ndarray, act: np.ndarray | None = None) -> CornerPointGrid:
    nz, ny, nx = (n // 2 for n in z.shape)
    lo, hi = float(z.min()) - 1.0, float(z.max()) + 1.0
    coord = np.array([[i * DX, j * DY, lo, i * DX, j * DY, hi]
                      for j in range(ny + 1) for i in range(nx + 1)])
    if act is None:
        act = np.ones((nx, ny, nz), dtype=bool)
    return CornerPointGrid(
        cart_dims=(nx, ny, nz), coord=coord, zcorn=z.ravel().copy(),
        actnum=act.transpose(2, 1, 0).ravel().astype(np.int8),
    )


def _overlap(a_top, a_bot, b_top, b_bot) -> float:
    """Exact ∫₀¹ max(0, min(a_bot, b_bot) - max(a_top, b_top)) dt for
    depths linear in t (each argument = (value at t=0, value at t=1))."""
    lines = [np.asarray(v, dtype=float) for v in (a_top, a_bot, b_top, b_bot)]
    ts = {0.0, 1.0}
    for p, q in ((lines[0], lines[2]), (lines[1], lines[3])):
        den = (p[1] - p[0]) - (q[1] - q[0])
        if den != 0.0 and 0.0 < (q[0] - p[0]) / den < 1.0:
            ts.add((q[0] - p[0]) / den)

    def h(t: float) -> float:
        at, ab, bt, bb = (v[0] + t * (v[1] - v[0]) for v in lines)
        return min(ab, bb) - max(at, bt)

    total = 0.0
    knots = sorted(ts)
    for t0, t1 in zip(knots, knots[1:]):
        h0, h1 = h(t0), h(t1)
        if h0 >= 0.0 and h1 >= 0.0:
            total += 0.5 * (h0 + h1) * (t1 - t0)
        elif h0 > 0.0 or h1 > 0.0:
            top = max(h0, h1)
            total += 0.5 * top * (t1 - t0) * top / (top - min(h0, h1))
    return total


def _side(z, i, j, k, side):
    """(top, bottom) depth pairs along the cell's lateral side
    ``side`` in ``"i-", "i+", "j-", "j+"``, ordered by the other axis."""
    zt, zb = z[2 * k, 2 * j:2 * j + 2, 2 * i:2 * i + 2], z[2 * k + 1, 2 * j:2 * j + 2, 2 * i:2 * i + 2]
    pick = {"i-": (slice(None), 0), "i+": (slice(None), 1),
            "j-": (0, slice(None)), "j+": (1, slice(None))}[side]
    return zt[pick], zb[pick]


def _process(g: CornerPointGrid, pinch_tol: float = 0.0):
    grid = build_topology(from_grdecl(g, pinch_tol=pinch_tol))
    fg = face_geometry(grid.nodes, grid.faces, grid.cell_corners)
    cg = cell_geometry(grid.nodes, grid.faces, grid.cell_faces, fg)
    return grid, fg, cg


def _contacts(grid, fg) -> dict:
    """``{((i,j,k), (i,j,k)): area}`` summed over inner i/j faces."""
    out: dict = {}
    ijk = [tuple(map(int, r)) for r in grid.original_ijk]
    for f in np.flatnonzero(grid.faces.direction <= 1):
        a, b = grid.faces.neighbours[f]
        if b >= 0:
            key = tuple(sorted((ijk[a], ijk[b])))
            out[key] = out.get(key, 0.0) + float(fg.areas[f])
    return out


def _assert_exact(g: CornerPointGrid, pinch_tol: float = 0.0):
    """Contacts, lateral side areas and volumes equal their closed forms;
    every face has positive area and each cell is closed."""
    grid, fg, cg = _process(g, pinch_tol)
    nx, ny, nz = g.cart_dims
    z = g.zcorn.reshape(2 * nz, 2 * ny, 2 * nx)
    ijk = [tuple(map(int, r)) for r in grid.original_ijk]
    kept = set(ijk)
    nb = grid.faces.neighbours
    assert (fg.areas > 0.0).all()
    assert (nb[:, 0] >= 0).all() and (nb < grid.n_cells).all()

    expected = {}
    for (i, j, k) in kept:
        for di, dj, s_a, s_b in ((1, 0, "i+", "i-"), (0, 1, "j+", "j-")):
            for kb in range(nz):
                other = (i + di, j + dj, kb)
                if other not in kept:
                    continue
                width = DY if di else DX
                area = width * _overlap(*_side(z, i, j, k, s_a), *_side(z, *other, s_b))
                if area > 1e-9:
                    expected[((i, j, k), other)] = area
    found = _contacts(grid, fg)
    assert found.keys() == expected.keys()
    for key, area in expected.items():
        assert found[key] == pytest.approx(area, rel=1e-9)

    for c, (i, j, k) in enumerate(ijk):
        sides = dict.fromkeys(("i-", "i+", "j-", "j+"), 0.0)
        closure = np.zeros(3)
        for f in grid.cell_faces[c]:
            if f < 0:
                continue
            sign = 1.0 if nb[f, 0] == c else -1.0
            closure += sign * fg.areas[f] * fg.normals[f]
            d = grid.faces.direction[f]
            if d == 2:
                continue
            x = fg.centroids[f, d]
            lo = i * DX if d == 0 else j * DY
            sides[("i" if d == 0 else "j") + ("-" if abs(x - lo) < DX / 2 else "+")] += fg.areas[f]
        for name, total in sides.items():
            top, bot = _side(z, i, j, k, name)
            width = DY if name[0] == "i" else DX
            assert total == pytest.approx(width * float(np.mean(bot - top)), rel=1e-9)
        np.testing.assert_allclose(closure, 0.0, atol=1e-9)
        th = z[2 * k + 1, 2 * j:2 * j + 2, 2 * i:2 * i + 2] - z[2 * k, 2 * j:2 * j + 2, 2 * i:2 * i + 2]
        assert cg.volumes[c] == pytest.approx(DX * DY * float(th.mean()), rel=1e-9)
    return grid, fg


# ── inactive cells beside a fault ──────────────────────────────────


def test_inactive_neighbour_across_fault_is_boundary():
    """East column thrown 5 down; east k=1 inactive. Each west cell meets
    it over 5 m: those pieces are boundary faces of the west cells."""
    z = _cake(2, 1, [10.0, 10.0, 10.0])
    z[:, :, 2:] += 5.0
    act = np.ones((2, 1, 3), dtype=bool)
    act[1, 0, 1] = False
    grid, fg = _assert_exact(_grid(z, act))
    assert grid.n_cells == 5
    assert (1, 0, 1) not in {tuple(r) for r in grid.original_ijk}
    assert _contacts(grid, fg) == pytest.approx({
        ((0, 0, 0), (1, 0, 0)): 50.0,
        ((0, 0, 1), (1, 0, 0)): 50.0,
        ((0, 0, 2), (1, 0, 2)): 50.0,
    })
    on_fault = (grid.faces.direction == 0) & np.isclose(fg.centroids[:, 0], DX)
    boundary = on_fault & (grid.faces.neighbours[:, 1] == -1)
    # W0 exposed above E0, W1 and W2 facing the inactive cell, E2 exposed below.
    assert sorted(fg.areas[boundary]) == pytest.approx([50.0] * 4)
    assert grid.faces.nodes.shape[1] == 4


def test_inactive_cells_on_both_sides_of_fault():
    """Inactive cells on both sides: their mutual overlap gives no face,
    and every active cell's fault side is still fully covered."""
    z = _cake(3, 2, [10.0, 10.0, 10.0, 10.0])
    z[:, :, 2:] += 7.0
    z[:, 2:, 4:] += 6.0              # second fault, along j=1, east block only
    act = np.ones((3, 2, 4), dtype=bool)
    act[0, 0, 2] = act[1, 0, 1] = act[1, 1, 3] = act[2, 1, 0] = False
    grid, _ = _assert_exact(_grid(z, act))
    assert grid.n_cells == 24 - 4


def test_contact_hidden_between_inactive_cells():
    """West active only at k=1, east only at k=0, east thrown 15 down: no
    active same-k pair is faulted, yet the two cells touch over 5 m."""
    z = _cake(2, 1, [10.0, 10.0])
    z[:, :, 2:] += 15.0
    act = np.zeros((2, 1, 2), dtype=bool)
    act[0, 0, 1] = act[1, 0, 0] = True
    grid, fg = _assert_exact(_grid(z, act))
    assert _contacts(grid, fg) == pytest.approx({((0, 0, 1), (1, 0, 0)): 50.0})


# ── pinch-outs beside a fault ──────────────────────────────────────


@pytest.mark.parametrize("pinch_tol", [0.0, -1.0])
def test_pinchout_beside_fault(pinch_tol):
    """East block: layer 1 has zero thickness, block thrown 15 down.
    ``pinch_tol=0`` pinches it out; ``-1`` keeps it active, and
    build_topology drops it as zero-volume. No zero-area face either way."""
    z = _cake(2, 2, [10.0, 10.0, 10.0])
    z[3:5, :, 2:] = z[2, :, 2:]
    z[:, :, 2:] += 15.0
    grid, _ = _assert_exact(_grid(z), pinch_tol)
    assert grid.n_cells == 10


@pytest.mark.parametrize("pinch_tol", [0.0, -1.0])
@pytest.mark.parametrize(("i_slots", "n_cells"), [(slice(2, 3), 6), (slice(2, 4), 5)])
def test_partial_pinchout_beside_fault(pinch_tol, i_slots, n_cells):
    """Layer 1 of the east block thins to zero at its fault-side j=0 corner,
    so its fault side is a triangle; the cells beside it meet it on
    triangles and pentagons with edge-intersection corners. If the whole
    j=0 edge collapses the wedge has a 2-node face and is dropped as
    zero-volume (as away from faults); its neighbours get boundary faces."""
    z = _cake(2, 1, [10.0, 10.0, 10.0])
    z[3:5, 0, i_slots] = z[2, 0, i_slots]
    z[:, :, 2:] += 7.0
    grid, _ = _assert_exact(_grid(z), pinch_tol)
    assert grid.n_cells == n_cells
    assert grid.faces.nodes.shape[1] > 4
    assert (grid.faces.nodes >= grid.n_corner_nodes).any()


def test_zero_thickness_active_cell_without_fault(fixtures):
    """An active zero-thickness cell (pinch detection off) used to merge
    the k-faces above and below it into one face with three cells."""
    cube = from_grdecl(read_grdecl(fixtures / "pinched_1x1x3.grdecl"), pinch_tol=-1.0)
    grid = build_topology(cube)
    assert grid.n_cells == 2
    k_inner = (grid.faces.direction == 2) & (grid.faces.neighbours[:, 1] >= 0)
    assert grid.faces.neighbours[k_inner].tolist() == [[0, 1]]


# ── fault polygons with edge-intersection nodes ────────────────────


def _tapered():
    """2x1x3 cake; east column thrown 3 m at pillar row j=0 and 25 m at
    j=1, so its layer boundaries cross the west ones inside the face."""
    z = _cake(2, 1, [10.0, 10.0, 10.0])
    z[:, 0, 2:] += 3.0
    z[:, 1, 2:] += 25.0
    return z


def test_tapered_throw_intersection_nodes_exact():
    grid, fg = _assert_exact(_grid(_tapered()))
    n_corner = grid.n_corner_nodes
    assert n_corner is not None and grid.nodes.shape[0] > n_corner
    assert grid.faces.nodes.shape[1] > 4
    assert grid.cell_corners.max() < n_corner
    # Intersection nodes: on the fault plane, at a west layer boundary,
    # and unique. Used by the fault (i-)faces and by the k-faces whose
    # fault-side edge they lie on, whose areas they leave unchanged.
    hanging = grid.nodes[n_corner:]
    np.testing.assert_allclose(hanging[:, 0], DX)
    np.testing.assert_allclose(hanging[:, 2] / 10.0, np.round(hanging[:, 2] / 10.0), atol=1e-12)
    assert np.unique(hanging, axis=0).shape[0] == hanging.shape[0]
    uses = np.isin(grid.faces.nodes, np.arange(n_corner, grid.nodes.shape[0])).any(axis=1)
    assert set(grid.faces.direction[uses].tolist()) == {0, 2}
    k_uses = uses & (grid.faces.direction == 2)
    east = fg.centroids[k_uses, 0] > DX
    np.testing.assert_allclose(fg.areas[k_uses], np.where(east, DX * np.hypot(DY, 22.0), DX * DY))
    used = np.unique(grid.faces.nodes[grid.faces.nodes >= n_corner])
    assert used.tolist() == list(range(n_corner, grid.nodes.shape[0]))
    distinct = [np.unique(r).size for r in grid.faces.nodes]
    assert max(distinct) == grid.faces.nodes.shape[1] >= 5


def test_tapered_throw_with_inactive_cells():
    act = np.ones((2, 1, 3), dtype=bool)
    act[0, 0, 1] = act[1, 0, 2] = False
    _assert_exact(_grid(_tapered(), act))


def test_tapered_j_fault_numbering_follows_i_faults():
    """Intersection nodes of j-faults come after those of i-faults."""
    z = _cake(2, 2, [10.0, 10.0, 10.0])
    z[:, 0:2, 2:4] += np.array([3.0, 14.0])[None, :, None]     # tapered along j
    z[:, 2:4, 0:2] += np.array([4.0, 17.0])[None, None, :]     # tapered along i
    z[:, 2:4, 2:4] += 25.0
    grid, _ = _assert_exact(_grid(z))
    n_corner = grid.n_corner_nodes
    hang = grid.faces.nodes >= n_corner
    on_i = grid.faces.nodes[hang & (grid.faces.direction[:, None] == 0)]
    on_j = grid.faces.nodes[hang & (grid.faces.direction[:, None] == 1)]
    assert on_i.size and on_j.size
    assert on_i.min() == n_corner and on_j.min() > on_i.max()


# ── tilted pillars ─────────────────────────────────────────────────

_HEX_FACES = np.array([[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4],
                       [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]])
_HEX_SIGNS = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0])


def _hex_volumes(x: np.ndarray) -> np.ndarray:
    """Exact trilinear volumes of ``(n, 8, 3)`` cell corners (i fastest):
    the flux of a bilinear face is its vertex mean · ½ d1 × d2."""
    p = x[:, _HEX_FACES]                                   # (n, 6, 4, 3)
    va = 0.5 * np.cross(p[:, :, 2] - p[:, :, 0], p[:, :, 3] - p[:, :, 1])
    return (_HEX_SIGNS * (p.mean(axis=2) * va).sum(axis=-1)).sum(axis=1) / 3.0


def _cell_sums(grid, fg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cell: Σ sign·n·A, Σ A and the divergence-theorem volume."""
    n = grid.n_cells
    nb = grid.faces.neighbours
    va = fg.normals * fg.areas[:, None]
    flux = (fg.centroids * va).sum(axis=1) / 3.0
    closure, total, volume = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
    for side, sign in ((0, 1.0), (1, -1.0)):
        m = nb[:, side] >= 0
        np.add.at(closure, nb[m, side], sign * va[m])
        np.add.at(total, nb[m, side], fg.areas[m])
        np.add.at(volume, nb[m, side], sign * flux[m])
    return closure, total, volume


def _tilted(slope: float, twist: float = 0.0) -> CornerPointGrid:
    """:func:`_tapered` with the j=1 pillar row leaning by ``slope`` in y
    (and ``twist`` in x) while the j=0 row stays vertical, so the two
    pillars of the fault face lean differently. Without twist they lie
    in the plane x=DX and every face is planar."""
    g = _grid(_tapered())
    coord = g.coord.reshape(2, 3, 6).copy()               # [j, i, xyz xyz]
    for col, d in ((1, slope), (0, twist)):
        coord[1, :, col] += d * coord[1, :, 2]
        coord[1, :, col + 3] += d * coord[1, :, 5]
    return CornerPointGrid(cart_dims=g.cart_dims, coord=coord.reshape(-1, 6),
                           zcorn=g.zcorn, actnum=g.actnum)


def _shoelace(p: np.ndarray) -> float:
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _clip(poly: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman: convex counter-clockwise polygons in 2D."""
    out = list(poly)
    for a, b in zip(clipper, np.roll(clipper, -1, axis=0)):
        def side(p, a=a, b=b):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        inp, out = out, []
        for p, q in zip(inp, inp[1:] + inp[:1]):
            sp, sq = side(p), side(q)
            if sp >= 0.0:
                out.append(p)
            if sp * sq < 0.0:
                out.append(p + (q - p) * sp / (sp - sq))
    return np.array(out).reshape(-1, 2)


def test_tilted_pillars_crossing_on_both_edges():
    """The fault face's pillars lean differently and the throw carries
    layer boundaries across it. Each intersection node is the true
    crossing of the two cells' edges (a z-parametrised crossing would lie
    off the east edge), and lies in the k-faces of both cells, so every
    cell closes and has its exact (prism) volume."""
    slope = 0.3
    g = _tilted(slope)
    grid, fg, _ = _process(g)
    z = g.zcorn.reshape(6, 2, 4)

    def section(i, k):          # (y, z) section of column i, layer k
        top, bot = z[2 * k, :, 2 * i], z[2 * k + 1, :, 2 * i]
        return np.array([[0.0, top[0]], [DY + slope * top[1], top[1]],
                         [DY + slope * bot[1], bot[1]], [0.0, bot[0]]])

    closure, total, volume = _cell_sums(grid, fg)
    assert np.abs(closure).max() < 1e-12 * total.max()
    ijk = [tuple(map(int, r)) for r in grid.original_ijk]
    for c, (i, _, k) in enumerate(ijk):
        assert volume[c] == pytest.approx(DX * _shoelace(section(i, k)), rel=1e-12)
    np.testing.assert_allclose(volume, _hex_volumes(grid.nodes[grid.cell_corners]), rtol=1e-12)

    expected = {}
    for ka in range(3):
        for kb in range(3):
            area = _shoelace(_clip(section(0, ka), section(1, kb)))
            if area > 1e-9:
                expected[((0, 0, ka), (1, 0, kb))] = area
    found = _contacts(grid, fg)
    assert found.keys() == expected.keys()
    for key, area in expected.items():
        assert found[key] == pytest.approx(area, rel=1e-12)

    n_corner = grid.n_corner_nodes
    hanging = grid.nodes[n_corner:]
    assert len(hanging) > 0
    np.testing.assert_allclose(hanging[:, 0], DX)
    for y, depth in hanging[:, 1:]:
        # on a west layer boundary (flat) and on an east one (3 -> 25 m throw)
        assert depth / 10.0 == pytest.approx(round(depth / 10.0), abs=1e-12)
        on_east = [abs(y - u * (DY + slope * (h + 25.0)))
                   for h in (0.0, 10.0, 20.0, 30.0)
                   if 0.0 <= (u := (depth - h - 3.0) / 22.0) <= 1.0]
        assert min(on_east) < 1e-9


def test_twisted_fault_face_cells_close():
    """Pillars of the fault face are skew: the two cells' edges miss each
    other, and the shared node sits midway between them. The cells still
    close to round-off and keep (nearly) their trilinear volumes."""
    grid, fg, cg = _process(_tilted(0.3, twist=0.05))
    closure, total, volume = _cell_sums(grid, fg)
    assert grid.nodes.shape[0] > grid.n_corner_nodes
    assert np.abs(closure).max() < 1e-12 * total.max()
    np.testing.assert_allclose(cg.volumes, volume, rtol=1e-12)
    hexv = _hex_volumes(grid.nodes[grid.cell_corners])
    assert (volume > 0.0).all()
    np.testing.assert_allclose(volume, hexv, rtol=1e-2)


_NORNE = Path(os.environ.get(
    "CPG_NORNE_GRDECL",
    "/mnt/c/projects/opm-machine/opm-tests/norne/include/GRID/IRAP_1005.GRDECL",
))


@pytest.mark.skipif(not _NORNE.is_file(), reason="Norne deck not found (set CPG_NORNE_GRDECL)")
def test_norne_cells_close():
    """Real tilted, twisted pillars with large throws: every cell closes.
    Volumes keep the sign of the cell's trilinear volume (the deck has 64
    cells, in 5 columns, whose corner pillars cross in plan view) and add
    up to the trilinear total."""
    grid = build_topology(from_grdecl(read_grdecl(_NORNE)))
    fg = face_geometry(grid.nodes, grid.faces, grid.cell_corners)
    closure, total, volume = _cell_sums(grid, fg)
    assert (np.linalg.norm(closure, axis=1) / total).max() < 1e-9
    hexv = _hex_volumes(grid.nodes[grid.cell_corners])
    hexv *= np.sign(hexv.sum())
    np.testing.assert_array_equal(volume < 0.0, hexv < 0.0)
    assert (hexv < 0.0).sum() == 64
    assert volume.sum() == pytest.approx(hexv.sum(), rel=1e-4)


def test_face_geometry_padded_polygon():
    """A convex hexagon in the plane x=1, given as a sorted, padded row."""
    yz = np.array([[0, 0], [2, 0], [3, 1], [2, 3], [0, 3], [-1, 1]], dtype=float)
    nodes = np.column_stack([np.ones(6), yz])
    shoelace = 0.5 * abs(np.dot(yz[:, 0], np.roll(yz[:, 1], -1))
                         - np.dot(yz[:, 1], np.roll(yz[:, 0], -1)))
    tri = 0.5 * 2.0 * 3.0                                   # nodes 0, 1, 4
    faces = Faces(
        nodes=np.array([[0, 1, 2, 3, 4, 5, 5, 5], [0, 1, 1, 1, 4, 4, 4, 4]]),
        neighbours=np.array([[0, -1], [0, -1]]),
        direction=np.zeros(2, dtype=np.int8),
    )
    cell = np.zeros((1, 8), dtype=np.int64)                 # centroid at node 0
    fg = face_geometry(nodes, faces, cell)
    np.testing.assert_allclose(fg.areas, [shoelace, tri])
    np.testing.assert_allclose(np.abs(fg.normals[:, 0]), 1.0)
    # Area-weighted centroid of the hexagon (fan from vertex 0).
    c = np.zeros(2)
    for a, b in zip(yz[1:-1], yz[2:]):
        w = 0.5 * abs((a[0] - yz[0, 0]) * (b[1] - yz[0, 1]) - (a[1] - yz[0, 1]) * (b[0] - yz[0, 0]))
        c += w * (yz[0] + a + b) / 3.0
    np.testing.assert_allclose(fg.centroids[0], [1.0, *(c / shoelace)])
    np.testing.assert_allclose(fg.centroids[1], [1.0, 2.0 / 3.0, 1.0])


def test_planview_and_mesh3d_with_intersection_nodes():
    grid = build_topology(from_grdecl(_grid(_tapered())))
    plain = build_topology(from_grdecl(_grid(_cake(2, 1, [10.0, 10.0, 10.0]))))
    pt, pt_plain = projected_topology(grid), projected_topology(plain)
    np.testing.assert_array_equal(pt.nodes2d, pt_plain.nodes2d)
    np.testing.assert_array_equal(pt.edges, pt_plain.edges)
    assert (pt.node_of_grid_node[grid.n_corner_nodes:] == -1).all()

    with pytest.raises(ValueError, match="edge-intersection"):
        face_edges(grid)
    g3 = build_mesh_graph_3d(grid)
    near = g3.nearest_node(grid.nodes[grid.n_corner_nodes:])
    assert (near < grid.n_corner_nodes).all()


def test_fixtures_keep_four_columns_and_no_extra_nodes(fixtures):
    for name in ("simple_faulted_2x2x2", "simple_faulted_2x2x2_j",
                 "simple_faulted_3x3x2", "simple_faulted_3x3x2_jfault",
                 "fault_simple", "pinchout_2x2x2", "simple_3x3x3"):
        cube = from_grdecl(read_grdecl(fixtures / f"{name}.grdecl"))
        grid = build_topology(cube)
        assert grid.faces.nodes.shape[1] == 4, name
        assert grid.n_corner_nodes == grid.nodes.shape[0] == cube.nodes.shape[0], name
