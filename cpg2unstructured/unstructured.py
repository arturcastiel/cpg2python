# Portions of this module are derived from MRST's processGRDECL.m
# (corner-point grid processing), (C) SINTEF Digital, GNU GPL v3.
# See the README "Attribution & provenance" section and the NOTICE file.
"""Unstructured-mesh face finders + topology builder for matching-Z CPG.

 covers i-faces + j-faces (regular, matching-Z).
 adds k-faces, cell-face inversion, active-cell rebuild,
and the topology-only orchestrator `build_topology`.
 promotes the topology-only `_TopoMesh` to the public
`UnstructuredGrid` (read-only properties dict + `__getitem__` accessor
+ lazy IndexMap cache) and adds the `attach_props` filter for
Eclipse cell-properties → active-cell order. `_TopoMesh` survives as
a backward-compat type alias so existing `cpg2unstructured.graph`
imports keep working.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cpg2unstructured._errors import NonMatchingZError
from cpg2unstructured.cpg import CornerCube

if TYPE_CHECKING:
    from cpg2unstructured.grdecl_reader import Grdecl
    from cpg2unstructured.index_map import IndexMap

__all__ = [
    "Faces",
    "find_i_faces", "find_j_faces", "find_k_faces", "find_ij_faces",
    "build_topology",
    "UnstructuredGrid",
    "attach_props",
]


# ── fault-face glue (faulted-grid support) ──────────────────
# Imports for the fault-face glue function (used internally).
# The actual integration into find_*_faces happens below.
from cpg2unstructured._fault_connections import find_connections
from cpg2unstructured._fault_geometry import compute_face_geometry

# Local-corner ordering — matches an earlier `_LOCAL_CORNERS`.
_LOCAL_CORNERS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)

# Direction → (low-cell slot, high-cell slot) in cell_faces[i-, i+, j-, j+, k-, k+]
#.
_SLOT_LOW  = (1, 3, 5)   # low side gets the "+" slot
_SLOT_HIGH = (0, 2, 4)   # high side gets the "-" slot


# Local face → 4 cell-corner indices (per `_LOCAL_CORNERS` order).
# Matches an earlier `_toy._LOCAL_FACES` slot pattern; used by `_real_cell_mask`
# for the per-face uniqueness check.
_LOCAL_FACES = np.array([
    [0, 2, 6, 4],  # i-
    [1, 3, 7, 5],  # i+
    [0, 1, 5, 4],  # j-
    [2, 3, 7, 6],  # j+
    [0, 1, 3, 2],  # k-
    [4, 5, 7, 6],  # k+
], dtype=np.int64)


@dataclass(frozen=True)
class Faces:
    """Per-face topology produced by the face finders.

    Attributes
    ----------
    nodes
        (Nfaces, M) int64 — sorted node ids per face (sort axis 1). A
        face is the SET of its distinct ids: one with fewer than M
        corners repeats ids, which carry no meaning (a triangular fault
        face repeats one id, ADR-104). ``M = 4`` unless a fault polygon
        has more corners (edge-intersection nodes,
        :func:`_fault_stack_faces`), so grids without such polygons keep
        ``(Nfaces, 4)``. The sorted row does not give the polygon's
        cyclic order; geometry recovers it
        (:func:`cpg2unstructured.geometry.face_geometry`).
    neighbours
        (Nfaces, 2) int64 — adjacent cell ids. Inner faces: smaller id
        in column 0. Boundary faces: -1 in column 1.
    direction
        (Nfaces,) int8 — 0 = i-face, 1 = j-face, 2 = k-face.
    """
    nodes: np.ndarray
    neighbours: np.ndarray
    direction: np.ndarray


# ── shared helpers ──────────────────────────────────────────────────────────


def _flat_active_id(actnum_3d: np.ndarray) -> np.ndarray:
    """Return a flat array of active-cell ids in Eclipse F-order.

    Active cells get 0..N_active-1; inactive cells get -1.
    """
    nx, ny, nz = actnum_3d.shape
    flat_actnum = actnum_3d.ravel(order="F")
    flat_id = np.full(nx * ny * nz, -1, dtype=np.int64)
    flat_id[flat_actnum] = np.arange(int(flat_actnum.sum()))
    return flat_id


def _move_minus_one_to_col1(nb: np.ndarray) -> np.ndarray:
    """Sort each row ascending, then swap rows whose col 0 is -1.

    Inner faces (both cells active) keep the smaller id in col 0;
    boundary faces (one side -1) land -1 in col 1.
    """
    s = np.sort(nb, axis=1)
    flip = s[:, 0] == -1
    if flip.any():
        s[flip] = s[flip][:, ::-1]
    return s


def _expand_actnum_axis_i(plane: np.ndarray) -> np.ndarray:
    """Expand a (NY, NZ) cell-resolution mask to (2*NY, 2*NZ) corner-slot."""
    return np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)


# ── i-faces ─────────────────────────────────────────────────────────────────


def _faulted_pillar_mask_i(P: np.ndarray, actnum_3d: np.ndarray) -> np.ndarray:
    """Per-cell-pair faulted mask along the i direction.

    Returns
    -------
    (NX-1, NY, NZ) bool — True at (i, j, k) means the active i-face
    between cells (i, j, k) and (i+1, j, k) is NON-MATCHING (faulted).
    False if matching-Z OR at least one of the two cells is inactive
    (in which case the face isn't an active-active i-face).

    Compares only at corner-slot positions where BOTH adjacent cells
    are active. an earlier ``-1`` sentinel for inactive cells would
    otherwise produce spurious mismatches.

    Replaces the earlier ``_verify_matching_z_i`` raise-on-fault
    helper. The faulted-grid support widens ``find_*_faces`` to dispatch
    matching vs faulted; this predicate is the dispatch primitive.
    """
    nx, ny, nz = actnum_3d.shape
    mask = np.zeros((nx - 1, ny, nz), dtype=bool)
    for i in range(1, nx):
        right_of_left = P[2 * i - 1]
        left_of_right = P[2 * i]
        slot_diff = right_of_left != left_of_right          # (2*NY, 2*NZ)
        # Reduce 4 corner-slots per (j, k) cell pair → one bool
        cell_diff = slot_diff.reshape(ny, 2, nz, 2).any(axis=(1, 3))
        active_pair = actnum_3d[i - 1] & actnum_3d[i]
        mask[i - 1] = cell_diff & active_pair
    return mask


def _i_face_quads(P: np.ndarray) -> np.ndarray:
    """(NX+1, NY, NZ, 4) of node ids per i-plane face.

    Reads BOTH side columns from P; selects elementwise via
    `np.where(vL == -1, vR, vL)` so inactive-cell -1 sentinels never
    leak into the face quad. Boundary planes degenerate both columns
    to the only available side. See .
    """
    twonx, twony, twonz = P.shape
    nx, ny, nz = twonx // 2, twony // 2, twonz // 2
    ip_idx = np.arange(nx + 1)
    left_col  = np.where(ip_idx == 0,  0,             2 * ip_idx - 1)
    right_col = np.where(ip_idx == nx, 2 * nx - 1,    (2 * ip_idx).clip(max=2 * nx - 1))

    out = np.empty((nx + 1, ny, nz, 4), dtype=np.int64)
    for offset, (dj, dk) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        j_slots = 2 * np.arange(ny) + dj
        k_slots = 2 * np.arange(nz) + dk
        vL = P[left_col[:, None, None],  j_slots[None, :, None], k_slots[None, None, :]]
        vR = P[right_col[:, None, None], j_slots[None, :, None], k_slots[None, None, :]]
        out[..., offset] = np.where(vL == -1, vR, vL)
    return out


def _stack_side_columns(
    P: np.ndarray, direction: int, i: int, j: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Corner node ids of both sides of one i- or j-stack.

    Returns ``(col_a, col_b)``, each ``(2, 2*NZ)``: row = face pillar
    (pillar 1, pillar 2), column = z-slot. The A side is the cell column
    at ``(i, j)``; the B side is ``(i+1, j)`` for i-stacks and ``(i, j+1)``
    for j-stacks.
    """
    if direction == 0:
        return P[2 * i + 1, 2 * j:2 * j + 2, :], P[2 * i + 2, 2 * j:2 * j + 2, :]
    return P[2 * i:2 * i + 2, 2 * j + 1, :], P[2 * i:2 * i + 2, 2 * j + 2, :]


def _pillar_point(a: np.ndarray, b: np.ndarray, z: float) -> np.ndarray:
    """XYZ at depth ``z`` on the line through corner points ``a`` and ``b``."""
    dz = b[2] - a[2]
    if dz == 0.0:
        return np.array([a[0], a[1], z])
    t = (z - a[2]) / dz
    return np.array([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), z])


def _virtual_pillar_points(
    nodes: np.ndarray, col_a: np.ndarray, col_b: np.ndarray,
    z_lo: float, z_hi: float,
) -> np.ndarray:
    """``(4, 3)`` XYZ of a stack's virtual nodes: top p1, top p2, base p1,
    base p2, on each pillar line at ``z_lo`` / ``z_hi``.

    The line is taken through the shallowest and deepest corner of any
    active cell of the stack on that pillar. Virtual nodes never reach
    the output (see :func:`_fault_stack_faces`), so only their z matters.
    """
    out = np.empty((4, 3), dtype=np.float64)
    for p in (0, 1):
        ids = np.concatenate([col_a[p], col_b[p]])
        ids = ids[ids >= 0]
        z = nodes[ids, 2]
        top, bot = nodes[ids[np.argmin(z)]], nodes[ids[np.argmax(z)]]
        out[p] = _pillar_point(top, bot, z_lo)
        out[2 + p] = _pillar_point(top, bot, z_hi)
    return out


def _padded_side(
    rows: np.ndarray, pts: np.ndarray, v_ids: np.ndarray,
    z_lo: float, z_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Corner rows and z rows of one stack side, padded MRST-style.

    ``rows`` is ``(2, R)`` local node ids (row = face pillar) of the
    side's active-cell boundaries, top to bottom:
    ``[top k0, base k0, top k1, base k1, ...]``; ``v_ids`` are the 4
    virtual local ids (top p1, top p2, base p1, base p2).

    The side becomes ``R + 1`` stack cells. Cell 0 is the virtual cell
    above the stack and cell ``R`` the one below; odd cell ``2m+1`` is
    the ``m``-th active cell; even cell ``2m`` in between is the *void*
    from the base of active cell ``m-1`` to the top of active cell ``m``
    — the room of inactive cells, zero-thick where the two touch.

    Returns ``corners`` ``(R+1, 4)`` in :func:`compute_face_geometry`
    column order ``[smaller-z p1, smaller-z p2, larger-z p1, larger-z p2]``
    and ``z`` ``(R+2, 2)`` cell boundaries per pillar for
    :func:`find_connections`.
    """
    upper = np.concatenate([v_ids[[0, 1], None], rows], axis=1)
    lower = np.concatenate([rows, v_ids[[2, 3], None]], axis=1)
    corners = np.stack(
        [upper[0], upper[1], lower[0], lower[1]], axis=1,
    ).astype(np.int64)
    z = np.vstack([[z_lo, z_lo], pts[rows, 2].T, [z_hi, z_hi]])
    return corners, z


@dataclass(frozen=True)
class _StackSides:
    """Both padded sides of one i- or j-stack (see :func:`_stack_sides`)."""
    real_ids: np.ndarray    # (n_real,) global ids of local nodes 0..n_real-1
    pts: np.ndarray         # (n_real + 4, 3) local point table
    pa: np.ndarray          # (R_a + 1, 4) A-side stack-cell corners
    za: np.ndarray          # (R_a + 2, 2)
    a_cells: np.ndarray     # (n_active_a,) original F-order ids, top to bottom
    pb: np.ndarray
    zb: np.ndarray
    b_cells: np.ndarray


def _stack_sides(
    cube: CornerCube, direction: int, i: int, j: int,
    z_lo: float, z_hi: float,
) -> _StackSides:
    """Padded sides of the stack at ``(i, j)``, built from ACTIVE cells only.

    Local node ids ``0..n_real-1`` are the active corners of the stack;
    ``n_real..n_real+3`` are the virtual nodes.
    """
    nx, ny, _ = cube.cart_dims
    col_a, col_b = _stack_side_columns(cube.P, direction, i, j)
    if direction == 0:
        act_a, act_b, b_step = cube.actnum_3d[i, j], cube.actnum_3d[i + 1, j], 1
    else:
        act_a, act_b, b_step = cube.actnum_3d[i, j], cube.actnum_3d[i, j + 1], nx
    ks_a, ks_b = np.nonzero(act_a)[0], np.nonzero(act_b)[0]
    rows_a = col_a[:, np.stack([2 * ks_a, 2 * ks_a + 1], axis=1).ravel()]
    rows_b = col_b[:, np.stack([2 * ks_b, 2 * ks_b + 1], axis=1).ravel()]

    real_ids = np.unique(np.concatenate([rows_a.ravel(), rows_b.ravel()]))
    n_real = real_ids.size
    pts = np.empty((n_real + 4, 3), dtype=np.float64)
    pts[:n_real] = cube.nodes[real_ids]
    pts[n_real:] = _virtual_pillar_points(cube.nodes, rows_a, rows_b, z_lo, z_hi)
    v_ids = n_real + np.arange(4, dtype=np.int64)

    pa, za = _padded_side(np.searchsorted(real_ids, rows_a), pts, v_ids, z_lo, z_hi)
    pb, zb = _padded_side(np.searchsorted(real_ids, rows_b), pts, v_ids, z_lo, z_hi)
    a_cells = i + j * nx + ks_a * (nx * ny)
    b_cells = i + j * nx + ks_b * (nx * ny) + b_step
    return _StackSides(real_ids, pts, pa, za, a_cells, pb, zb, b_cells)


def _side_cell(cells: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Original cell id of stack cells ``c`` (odd = active), else ``-1``."""
    out = np.full(c.shape, -1, dtype=np.int64)
    real = c % 2 == 1
    out[real] = cells[(c[real] - 1) // 2]
    return out


def _z_padding(nodes: np.ndarray) -> tuple[float, float]:
    """Depths of the virtual layers: well above / below every node."""
    z_all = nodes[:, 2]
    z_pad = max(1.0, float(z_all.max() - z_all.min()))
    return float(z_all.min()) - z_pad, float(z_all.max()) + z_pad


def _faulted_stacks(cube: CornerCube, direction: int) -> np.ndarray:
    """``(NX-1, NY)`` (i) or ``(NX, NY-1)`` (j) bool — stacks that need the
    fault sweep of :func:`_fault_stack_faces`.

    A stack is faulted when an active cell pair across it has
    non-matching corners (:func:`_faulted_pillar_mask_i` /
    ``_j``). It is also faulted, although every active same-k pair
    matches, when an active cell faces an inactive one and still
    overlaps an active cell of another k across the stack: e.g. A
    active at k=0 only and B active at k=1 only, thrown one layer
    against each other. The plain per-k faces would miss that contact.
    Such a pair can only be separated by inactive cells on both sides
    (a matching active pair between them would split them), so only
    those stacks are swept to check.
    """
    act = cube.actnum_3d
    if direction == 0:
        mask = _faulted_pillar_mask_i(cube.P, act)
        a, b = act[:-1], act[1:]
    else:
        mask = _faulted_pillar_mask_j(cube.P, act)
        a, b = act[:, :-1], act[:, 1:]
    faulted = mask.any(axis=2)

    # Per stack: is there an A-only and a B-only cell with no matching
    # active pair between them? ``seg`` numbers the runs between pairs.
    seg = np.cumsum(a & b, axis=2)
    nz = act.shape[2]
    has_a = np.zeros(faulted.shape + (nz + 1,), dtype=bool)
    has_b = np.zeros_like(has_a)
    ia, ja, ka = np.nonzero(a & ~b)
    has_a[ia, ja, seg[ia, ja, ka]] = True
    ib, jb, kb = np.nonzero(b & ~a)
    has_b[ib, jb, seg[ib, jb, kb]] = True
    candidate = (has_a & has_b).any(axis=2) & ~faulted
    if candidate.any():
        z_lo, z_hi = _z_padding(cube.nodes)
        for i_int, j_int in zip(*np.nonzero(candidate)):
            s = _stack_sides(cube, direction, int(i_int), int(j_int), z_lo, z_hi)
            C = find_connections(s.za, s.zb)
            ca, cb = _side_cell(s.a_cells, C[:, 0]), _side_cell(s.b_cells, C[:, 1])
            layer = cube.cart_dims[0] * cube.cart_dims[1]
            if ((ca >= 0) & (cb >= 0) & (ca // layer != cb // layer)).any():
                faulted[i_int, j_int] = True
    return faulted


# Relative area below which a fault polygon counts as degenerate:
# area <= _ZERO_AREA_REL * (bounding-box diagonal)^2.
_ZERO_AREA_REL = 1e-12


def _cyclic_polygons(
    corners: np.ndarray, num_nodes: np.ndarray, pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fault polygons as padded rows, and which ones have positive area.

    Returns ``(rows, ok)``: ``rows`` ``(n, M)`` node ids in the cyclic
    order of :func:`compute_face_geometry`, each padded to
    ``M = max(4, corners per polygon)`` by repeating its last corner (a
    triangle is ``[a, b, c, c]``, ADR-104); ``ok`` ``(n,)`` is False for
    polygons with fewer than 3 distinct corners or zero area, e.g. a
    contact that closes to a line where a pinched-out cell touches the
    fault. Repeated corners add nothing to the fan area.
    """
    num = num_nodes.astype(np.int64).copy()
    starts = np.cumsum(num) - num
    # A 5+ corner loop may list its first corner again at the end.
    big = num > 4
    if big.any():
        wrap = np.zeros_like(big)
        wrap[big] = corners[starts[big] + num[big] - 1] == corners[starts[big]]
        num[wrap] -= 1
    m = max(4, int(num.max()) if num.size else 4)
    last = np.maximum(num - 1, 0)
    pick = starts[:, None] + np.minimum(np.arange(m), last[:, None])
    pick = np.minimum(pick, max(corners.size - 1, 0))
    rows = corners[pick] if corners.size else np.zeros((num.size, m), np.int64)

    srt = np.sort(rows, axis=1)
    distinct = 1 + (srt[:, 1:] != srt[:, :-1]).sum(axis=1)
    v = pts[rows]                                          # (n, M, 3)
    e = v[:, 1:] - v[:, :1]
    area = 0.5 * np.linalg.norm(
        np.cross(e[:, :-1], e[:, 1:]).sum(axis=1), axis=1,
    )
    diag2 = ((v.max(axis=1) - v.min(axis=1)) ** 2).sum(axis=1)
    ok = (num >= 3) & (distinct >= 3) & (area > _ZERO_AREA_REL * diag2)
    return rows, ok


def _pad_rows(rows: np.ndarray, m: int) -> np.ndarray:
    """Pad ``(n, k)`` face rows to ``m`` columns by repeating the last."""
    if rows.shape[1] >= m:
        return rows
    extra = np.repeat(rows[:, -1:], m - rows.shape[1], axis=1)
    return np.concatenate([rows, extra], axis=1)


@dataclass(frozen=True)
class _StackFaces:
    """Output of :func:`_fault_stack_faces`."""
    stack_faulted: np.ndarray   # (NX-1, NY) or (NX, NY-1) bool
    polys: np.ndarray           # (n, M) cyclic node ids, last corner repeated
    low_orig: np.ndarray        # (n,) A-side original cell id or -1
    high_orig: np.ndarray       # (n,) B-side original cell id or -1
    new_points: np.ndarray      # (n_new, 3) intersection nodes, ids n_base + k


def _fault_stack_faces(
    cube: CornerCube, direction: int, n_base: int | None = None,
) -> _StackFaces:
    """Fault-aware faces of every faulted i-stack (``direction=0``) or
    j-stack (``direction=1``).

    A stack is the pair of cell columns on either side of one pillar
    pair; :func:`_faulted_stacks` says which are faulted. Their faces
    are rebuilt with the MRST sweep (:func:`find_connections` +
    :func:`compute_face_geometry`), so each face joins the cells that
    really overlap, whatever their k.

    Each side is built as in MRST's ``processGRDECL``
    (:func:`_padded_side`): only ACTIVE cells exist; the room between
    two active cells (inactive cells, or nothing) is a void cell; a
    virtual cell pads the side above its top and below its base. An
    overlap of an active cell with a void or virtual cell on the other
    side is the part of its fault side that faces no active cell — an
    *exposed piece* — and is returned as a boundary face. Overlaps with
    no active cell on either side are dropped, so inactive cells never
    appear. The padding lies outside the z range of every node, so a
    face is always bounded by real corners and no virtual node reaches
    the output.

    Faces of zero area (fewer than 3 distinct corners, or collinear
    ones, e.g. where a pinched-out layer touches the fault) are dropped.

    Fault polygons have 3 to 8 corners. When the top or base edge of a
    cell crosses an edge of a cell on the other side between the two
    pillars, the polygon gets a corner at the crossing — an *edge
    intersection node*, not a pillar corner. Those nodes are numbered
    ``n_base, n_base + 1, ...`` (default ``n_base = len(cube.nodes)``)
    in stack order, then in the order :func:`compute_face_geometry`
    creates them; a crossing shared by neighbouring polygons is one
    node, and only nodes used by a returned face are kept. They lie on
    cell top/base edges but are not corners of the k-faces there
    (hanging nodes, as in MRST).

    Returns
    -------
    _StackFaces
        ``polys`` is ``(n, M)``, ``M = max(4, corners per polygon)``,
        node ids in :func:`compute_face_geometry` order (clockwise,
        normal from the A side to the B side), padded by repeating the
        last corner (a triangle — e.g. at a fault tip — is
        ``[a, b, c, c]``, ADR-104). Not sorted. ``low_orig`` /
        ``high_orig`` are original F-order cell ids on the A side
        (column ``i`` / row ``j``) and on the B side (``i+1`` /
        ``j+1``), ``-1`` on the side an exposed piece faces.
        ``new_points`` holds the intersection nodes.
    """
    nodes = cube.nodes
    if n_base is None:
        n_base = nodes.shape[0]
    stack_faulted = _faulted_stacks(cube, direction)

    empty_ids = np.zeros(0, dtype=np.int64)
    empty = _StackFaces(
        stack_faulted, np.zeros((0, 4), dtype=np.int64),
        empty_ids, empty_ids.copy(), np.zeros((0, 3), dtype=np.float64),
    )
    if not stack_faulted.any():
        return empty

    axis = "ij"[direction]
    z_lo, z_hi = _z_padding(nodes)

    poly_chunks: list[np.ndarray] = []
    low_chunks: list[np.ndarray] = []
    high_chunks: list[np.ndarray] = []
    new_chunks: list[np.ndarray] = []
    n_new = 0

    for i_int, j_int in zip(*np.nonzero(stack_faulted)):
        i, j = int(i_int), int(j_int)
        s = _stack_sides(cube, direction, i, j, z_lo, z_hi)

        C = find_connections(s.za, s.zb)
        low = _side_cell(s.a_cells, C[:, 0])
        high = _side_cell(s.b_cells, C[:, 1])
        keep = (low >= 0) | (high >= 0)
        C, low, high = C[keep], low[keep], high[keep]
        if C.shape[0] == 0:
            continue

        pts, num_nodes, corners = compute_face_geometry(
            s.pa[C[:, 0]], s.pb[C[:, 1]], s.pts,
        )
        rows, ok = _cyclic_polygons(corners, num_nodes, pts)
        rows, low, high = rows[ok], low[ok], high[ok]
        if rows.shape[0] == 0:
            continue

        n_real = s.real_ids.size
        if ((rows >= n_real) & (rows < n_real + 4)).any():
            where = (f"i={i}-{i + 1}, j={j}" if direction == 0
                     else f"i={i}, j={j}-{j + 1}")
            raise RuntimeError(
                f"virtual padding node in a fault face of {axis}-stack "
                f"({where}) — bug"
            )
        to_global = np.full(pts.shape[0], -1, dtype=np.int64)
        to_global[:n_real] = s.real_ids
        used = np.unique(rows[rows >= n_real + 4])
        to_global[used] = n_base + n_new + np.arange(used.size)
        n_new += used.size

        poly_chunks.append(to_global[rows])
        low_chunks.append(low)
        high_chunks.append(high)
        new_chunks.append(pts[used])

    if not poly_chunks:
        return empty
    m = max(c.shape[1] for c in poly_chunks)
    return _StackFaces(
        stack_faulted,
        np.concatenate([_pad_rows(c, m) for c in poly_chunks]).astype(np.int64),
        np.concatenate(low_chunks).astype(np.int64),
        np.concatenate(high_chunks).astype(np.int64),
        np.concatenate(new_chunks, axis=0).astype(np.float64),
    )


def _faulted_faces_record(
    cube: CornerCube, direction: int,
) -> tuple[Faces, np.ndarray, np.ndarray]:
    """:func:`_fault_stack_faces` as a :class:`Faces` record in active ids.

    Returns ``(faces, node_table, stack_faulted)``. Edge-intersection
    nodes are numbered as in :func:`build_topology` — after
    ``cube.nodes``, i-stack nodes first, then j-stack nodes — and
    ``node_table`` is ``cube.nodes`` followed by every intersection node
    up to the last one these faces use.
    """
    n_cube = cube.nodes.shape[0]
    res = _fault_stack_faces(cube, direction, n_cube)
    polys = res.polys
    table = cube.nodes
    if res.new_points.shape[0]:
        prior = np.zeros((0, 3), dtype=np.float64)
        if direction == 1:
            prior = _fault_stack_faces(cube, 0, n_cube).new_points
            polys = np.where(polys >= n_cube, polys + prior.shape[0], polys)
        table = np.concatenate([cube.nodes, prior, res.new_points], axis=0)
    if polys.shape[0] == 0:
        return _empty_faces(direction=direction), table, res.stack_faulted
    flat_id = _flat_active_id(cube.actnum_3d)
    low, high = res.low_orig, res.high_orig
    lo = np.where(low >= 0, flat_id[np.maximum(low, 0)], -1)
    hi = np.where(high >= 0, flat_id[np.maximum(high, 0)], -1)
    nbs = _move_minus_one_to_col1(np.stack([lo, hi], axis=1))
    return Faces(
        nodes=np.sort(polys, axis=1),
        neighbours=nbs,
        direction=np.full(polys.shape[0], direction, dtype=np.int8),
    ), table, res.stack_faulted


def _find_faulted_i_faces(cube: CornerCube) -> tuple[Faces, np.ndarray]:
    """Build i-direction fault faces from non-matching-Z pillar pairs.

    Walks every faulted (i, j) pillar-pair stack and emits one face per
    overlap found by :func:`find_connections`: an inner face for each
    pair of active cells that really touch across the fault (any k on
    either side), and a boundary face for each exposed piece of an
    active cell's fault side that faces no active cell (inactive cells
    and gaps included). See :func:`_fault_stack_faces` for the method.

    Returns
    -------
    faces : :class:`Faces`
        Fault i-faces. ``faces.nodes`` is ``(n_faulted_faces, M)`` of
        sorted node ids, ``M = 4`` unless a polygon has more corners
        (see :class:`Faces` for the padding).
        ``faces.neighbours`` is ``(n_faulted_faces, 2)`` in active ids;
        ``-1`` in column 1 marks an exposed boundary piece.
        ``faces.direction`` is all-0.
    new_points : (n_points, 3) float64
        Node table — ``cube.nodes`` followed by the edge-intersection
        nodes of the fault polygons (none for polygons of pillar
        corners only).
    """
    faces, table, _ = _faulted_faces_record(cube, direction=0)
    return faces, table


def _empty_faces(direction: int = 0) -> Faces:
    """Return an empty Faces record for a given direction."""
    return Faces(
        nodes=np.zeros((0, 4), dtype=np.int64),
        neighbours=np.zeros((0, 2), dtype=np.int64),
        direction=np.zeros(0, dtype=np.int8),
    )


def find_i_faces(cube: CornerCube) -> Faces:
    """All i-direction faces (inner + boundary) as a ``Faces`` record.

    Dispatches matching-Z stacks to the v0.1.0 fast path and faulted
    stacks to :func:`_find_faulted_i_faces`.
    When no faulted stacks are present, the output is byte-identical
    to v0.1.0 —  invariance.

    A fault polygon with an edge-intersection corner references a node
    id ``>= len(cube.nodes)`` (numbered as in :func:`_faulted_faces_record`);
    :func:`build_topology` appends the nodes its faces use, in the same
    order, to :attr:`UnstructuredGrid.nodes`.
    """
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    fault_faces, _, stack_faulted = _faulted_faces_record(cube, direction=0)

    quads_raw = _i_face_quads(P)
    quads = np.sort(quads_raw, axis=-1)
    quads_flat = quads.reshape(-1, 4)

    flat_id = _flat_active_id(actnum_3d)
    cell_id = np.full((nx + 2, ny, nz), -1, dtype=np.int64)
    cell_id[1:-1] = flat_id.reshape(nx, ny, nz, order="F")
    left  = cell_id[0:nx + 1]
    right = cell_id[1:nx + 2]
    keep = ((left != -1) | (right != -1)).ravel()

    if not stack_faulted.any():
        # FAST PATH — byte-identical to v0.1.0 for matching-Z decks.
        faces_kept = quads_flat[keep]
        nb_raw = np.stack([left.ravel(), right.ravel()], axis=1)[keep]
        nb = _move_minus_one_to_col1(nb_raw)
        direction = np.zeros(faces_kept.shape[0], dtype=np.int8)
        return Faces(nodes=faces_kept, neighbours=nb, direction=direction)

    # DISPATCH PATH — drop matching faces in faulted (i, j) stacks and
    # concatenate with _find_faulted_i_faces output.
    face_drop = np.zeros((nx + 1, ny, nz), dtype=bool)
    if stack_faulted.shape[0] > 0:
        face_drop[1:nx, :, :] = stack_faulted[:, :, None]

    keep_no_fault = keep & ~face_drop.ravel()
    faces_kept = quads_flat[keep_no_fault]
    nb_raw = np.stack(
        [left.ravel(), right.ravel()], axis=1,
    )[keep_no_fault]
    nb = _move_minus_one_to_col1(nb_raw)
    direction = np.zeros(faces_kept.shape[0], dtype=np.int8)

    m = fault_faces.nodes.shape[1]
    return Faces(
        nodes=np.concatenate([_pad_rows(faces_kept, m), fault_faces.nodes], axis=0),
        neighbours=np.concatenate([nb, fault_faces.neighbours], axis=0),
        direction=np.concatenate([direction, fault_faces.direction], axis=0),
    )


# ── j-faces (symmetric, i ↔ j) ──────────────────────────────────────────────


def _faulted_pillar_mask_j(P: np.ndarray, actnum_3d: np.ndarray) -> np.ndarray:
    """Per-cell-pair faulted mask along the j direction.

    Returns
    -------
    (NX, NY-1, NZ) bool — True at (i, j, k) means the active j-face
    between cells (i, j, k) and (i, j+1, k) is non-matching.

    See :func:`_faulted_pillar_mask_i` for context.
    """
    nx, ny, nz = actnum_3d.shape
    mask = np.zeros((nx, ny - 1, nz), dtype=bool)
    for j in range(1, ny):
        front_of_low = P[:, 2 * j - 1]
        back_of_high = P[:, 2 * j]
        slot_diff = front_of_low != back_of_high            # (2*NX, 2*NZ)
        cell_diff = slot_diff.reshape(nx, 2, nz, 2).any(axis=(1, 3))
        active_pair = actnum_3d[:, j - 1] & actnum_3d[:, j]
        mask[:, j - 1] = cell_diff & active_pair
    return mask


def _j_face_quads(P: np.ndarray) -> np.ndarray:
    """(NX, NY+1, NZ, 4) of node ids per j-plane face."""
    twonx, twony, twonz = P.shape
    nx, ny, nz = twonx // 2, twony // 2, twonz // 2
    jp_idx = np.arange(ny + 1)
    left_col  = np.where(jp_idx == 0,  0,             2 * jp_idx - 1)
    right_col = np.where(jp_idx == ny, 2 * ny - 1,    (2 * jp_idx).clip(max=2 * ny - 1))

    out = np.empty((nx, ny + 1, nz, 4), dtype=np.int64)
    for offset, (di, dk) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        i_slots = 2 * np.arange(nx) + di
        k_slots = 2 * np.arange(nz) + dk
        vL = P[i_slots[:, None, None], left_col[None, :, None],  k_slots[None, None, :]]
        vR = P[i_slots[:, None, None], right_col[None, :, None], k_slots[None, None, :]]
        out[..., offset] = np.where(vL == -1, vR, vL)
    return out


def _find_faulted_j_faces(cube: CornerCube) -> tuple[Faces, np.ndarray]:
    """Build j-direction fault faces from non-matching-Z pillar pairs.

    Symmetric to :func:`_find_faulted_i_faces` along the j-axis: inner
    faces between overlapping cells across the fault plus boundary faces
    for exposed pieces. Same method (:func:`_fault_stack_faces`).

    Returns
    -------
    faces : :class:`Faces`
        Fault j-faces with ``direction = 1``.
    new_points : (n_points, 3) float64
        Node table — ``cube.nodes``, then the i-stack and j-stack
        edge-intersection nodes when these faces use any (numbered as in
        :func:`build_topology`).
    """
    faces, table, _ = _faulted_faces_record(cube, direction=1)
    return faces, table


def find_j_faces(cube: CornerCube) -> Faces:
    """All j-direction faces (inner + boundary) as a ``Faces`` record.

    Dispatches matching-Z stacks to the v0.1.0 fast path and faulted
    stacks to :func:`_find_faulted_j_faces`.
    When no faulted stacks are present, the output is byte-identical
    to v0.1.0 —  invariance. Edge-intersection node ids: see
    :func:`find_i_faces`.
    """
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    fault_faces, _, stack_faulted = _faulted_faces_record(cube, direction=1)

    quads_raw = _j_face_quads(P)
    quads = np.sort(quads_raw, axis=-1)
    quads_flat = quads.reshape(-1, 4)

    flat_id = _flat_active_id(actnum_3d)
    cell_id = np.full((nx, ny + 2, nz), -1, dtype=np.int64)
    cell_id[:, 1:-1] = flat_id.reshape(nx, ny, nz, order="F")
    low  = cell_id[:, 0:ny + 1]
    high = cell_id[:, 1:ny + 2]
    keep = ((low != -1) | (high != -1)).ravel()

    if not stack_faulted.any():
        # FAST PATH — byte-identical to v0.1.0 for matching-Z decks.
        faces_kept = quads_flat[keep]
        nb_raw = np.stack([low.ravel(), high.ravel()], axis=1)[keep]
        nb = _move_minus_one_to_col1(nb_raw)
        direction = np.ones(faces_kept.shape[0], dtype=np.int8)
        return Faces(nodes=faces_kept, neighbours=nb, direction=direction)

    # DISPATCH PATH — drop matching faces in faulted (i, j) j-stacks
    # and concatenate with _find_faulted_j_faces output.
    face_drop = np.zeros((nx, ny + 1, nz), dtype=bool)
    if stack_faulted.shape[1] > 0:
        face_drop[:, 1:ny, :] = stack_faulted[:, :, None]

    keep_no_fault = keep & ~face_drop.ravel()
    faces_kept = quads_flat[keep_no_fault]
    nb_raw = np.stack(
        [low.ravel(), high.ravel()], axis=1,
    )[keep_no_fault]
    nb = _move_minus_one_to_col1(nb_raw)
    direction = np.ones(faces_kept.shape[0], dtype=np.int8)

    m = fault_faces.nodes.shape[1]
    return Faces(
        nodes=np.concatenate([_pad_rows(faces_kept, m), fault_faces.nodes], axis=0),
        neighbours=np.concatenate([nb, fault_faces.neighbours], axis=0),
        direction=np.concatenate([direction, fault_faces.direction], axis=0),
    )


# ── combined helper ─────────────────────────────────────────────────────────


def find_ij_faces(cube: CornerCube) -> Faces:
    """Concatenate i-faces + j-faces. Convenience for downstream callers."""
    fi = find_i_faces(cube)
    fj = find_j_faces(cube)
    m = max(fi.nodes.shape[1], fj.nodes.shape[1])
    return Faces(
        nodes=np.concatenate([_pad_rows(fi.nodes, m), _pad_rows(fj.nodes, m)], axis=0),
        neighbours=np.concatenate([fi.neighbours, fj.neighbours], axis=0),
        direction=np.concatenate([fi.direction, fj.direction], axis=0),
    )


# ── k-faces ──────────────────────────────────────────────────────────


def _faulted_pillar_mask_k(P: np.ndarray, actnum_3d: np.ndarray) -> np.ndarray:
    """Per-cell-pair faulted mask along the k direction.

    Returns
    -------
    (NX, NY, NZ-1) bool — True at (i, j, k) means the active k-face
    between cells (i, j, k) and (i, j, k+1) is non-matching.

    Mirrors :func:`_faulted_pillar_mask_i` along the k axis.
    Cartesian + pinchout decks produce all-False (matching-Z);
    stair-stepped k-direction faults flip bits to True.
    """
    nx, ny, nz = actnum_3d.shape
    mask = np.zeros((nx, ny, nz - 1), dtype=bool)
    for k in range(1, nz):
        bot_of_top = P[:, :, 2 * k - 1]
        top_of_bot = P[:, :, 2 * k]
        slot_diff = bot_of_top != top_of_bot                # (2*NX, 2*NY)
        cell_diff = slot_diff.reshape(nx, 2, ny, 2).any(axis=(1, 3))
        active_pair = actnum_3d[:, :, k - 1] & actnum_3d[:, :, k]
        mask[:, :, k - 1] = cell_diff & active_pair
    return mask


def _k_face_quads(P: np.ndarray) -> np.ndarray:
    """(NX, NY, NZ+1, 4) of node ids per k-plane face.

    Same active-side picker as `_i_face_quads` / `_j_face_quads` (
    via ): reads both side columns and selects elementwise via
    `np.where(vL == -1, vR, vL)`.
    """
    twonx, twony, twonz = P.shape
    nx, ny, nz = twonx // 2, twony // 2, twonz // 2
    kp_idx = np.arange(nz + 1)
    left_col  = np.where(kp_idx == 0,  0,             2 * kp_idx - 1)
    right_col = np.where(kp_idx == nz, 2 * nz - 1,    (2 * kp_idx).clip(max=2 * nz - 1))

    out = np.empty((nx, ny, nz + 1, 4), dtype=np.int64)
    for offset, (di, dj) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        i_slots = 2 * np.arange(nx) + di
        j_slots = 2 * np.arange(ny) + dj
        vL = P[i_slots[:, None, None], j_slots[None, :, None], left_col[None, None, :]]
        vR = P[i_slots[:, None, None], j_slots[None, :, None], right_col[None, None, :]]
        out[..., offset] = np.where(vL == -1, vR, vL)
    return out


def find_k_faces(cube: CornerCube) -> Faces:
    """All k-direction faces (inner + boundary) as a `Faces` record."""
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    faulted = _faulted_pillar_mask_k(P, actnum_3d)
    if faulted.any():
        i, j, k = map(int, np.argwhere(faulted)[0])
        raise NonMatchingZError(
            f"non-matching-Z across k={k + 1} at (i={i}, j={j}) — v2 only"
        )

    quads_raw = _k_face_quads(P)
    quads = np.sort(quads_raw, axis=-1)
    quads_flat = quads.reshape(-1, 4)

    flat_id = _flat_active_id(actnum_3d)
    cell_id = np.full((nx, ny, nz + 2), -1, dtype=np.int64)
    cell_id[:, :, 1:-1] = flat_id.reshape(nx, ny, nz, order="F")
    low  = cell_id[:, :, 0:nz + 1]
    high = cell_id[:, :, 1:nz + 2]

    keep = ((low != -1) | (high != -1)).ravel()
    faces_kept = quads_flat[keep]
    nb_raw = np.stack([low.ravel(), high.ravel()], axis=1)[keep]
    nb = _move_minus_one_to_col1(nb_raw)
    direction = np.full(faces_kept.shape[0], 2, dtype=np.int8)

    return Faces(nodes=faces_kept, neighbours=nb, direction=direction)


# ── topology builder ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class UnstructuredGrid:
    """Public unstructured mesh — nodes + faces + cells + properties.

    Constructed by :func:`build_topology`. Optionally enriched
    with cell properties via :func:`attach_props`, which
    returns a NEW grid with the populated ``props`` dict.

    ``nodes`` carries the unique XYZ from ``CornerCube.nodes``; node ids
    referenced by ``faces.nodes`` and ``cell_corners`` index into this
    array. When a fault polygon has a corner where a cell's top or base
    edge crosses a cell edge across the fault, that edge-intersection
    node is appended after them: ids ``>= n_corner_nodes`` are such
    nodes, used only by fault faces (never by ``cell_corners``; they
    hang on the k-faces, as in MRST). ``n_corner_nodes`` is ``None``
    when every node is a corner node. ``original_ijk[c]`` is the
    Eclipse-natural ``(i, j, k)`` of the cell that used to be at new
    id ``c``.

    ``props`` is keyed by Eclipse keyword names verbatim (uppercase,
    e.g. ``"PORO"``). Use the ``grid["PORO"]`` accessor instead of
    ``grid.props["PORO"]`` for the helpful "available keys" error on
    misses.

    ``cell_faces`` is ``(Ncells, 6 + E)``. Columns 0-5 are the slots
    ``[i-, i+, j-, j+, k-, k+]``, always filled. A cell side on an i/j
    fault can touch several faces (one per overlapping cell across the
    fault, plus exposed boundary pieces): the slot holds the first, the
    others go in columns 6+, padded with ``-1``. ``E`` is the largest
    number of extra faces any cell has, so a grid without faults keeps
    the plain ``(Ncells, 6)`` shape. To walk every face of a cell, use
    ``f for f in grid.cell_faces[c] if f >= 0``.
    """
    nodes: np.ndarray              # (Nnodes, 3) float64
    faces: Faces                   # merged i+j+k
    cell_faces: np.ndarray         # (Ncells_active, 6 + E) int64, -1 padded
    cell_corners: np.ndarray       # (Ncells_active, 8) int64 — node ids
    original_ijk: np.ndarray       # (Ncells_active, 3) int64
    cart_dims: tuple[int, int, int]
    # (Mnnc, 2) int64 — non-neighbour connections: cell-id pairs that are
    # connected without sharing a geometric face. Populated for cells
    # bridged across a pinched-out layer; empty (0, 2) otherwise.
    nnc: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    # (Ncells,) int64 — connected-component label per cell over the face
    # + NNC adjacency. 0-based, assigned in order of first cell id, so a
    # single-component grid is all zeros. Use ``split()`` to materialise
    # the components as independent grids.
    component: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    props: dict[str, np.ndarray] = field(default_factory=dict)
    # Number of leading ``nodes`` that are cell corners; the rest are
    # fault edge-intersection nodes. ``None``: all nodes are corners.
    n_corner_nodes: int | None = None
    _index_map: IndexMap | None = field(default=None, repr=False, compare=False)

    def __getitem__(self, key: str) -> np.ndarray:
        try:
            return self.props[key]
        except KeyError as exc:
            avail = sorted(self.props)
            raise KeyError(
                f"property {key!r} not attached. Available: {avail}"
            ) from exc

    def __contains__(self, key: object) -> bool:
        return key in self.props

    @property
    def n_cells(self) -> int:
        return int(self.cell_faces.shape[0])

    @property
    def index_map(self) -> IndexMap:
        """Lazy-cached :class:`IndexMap` built from cart_dims + original_ijk.

        Frozen-dataclass cache: stored via ``object.__setattr__`` on
        first access — same pattern ``functools.cached_property``
        uses on frozen dataclasses.
        """
        from cpg2unstructured.index_map import IndexMap
        if self._index_map is None:
            object.__setattr__(
                self, "_index_map",
                IndexMap(self.cart_dims, self.original_ijk),
            )
        return self._index_map  # type: ignore[return-value]

    def split(self) -> list[UnstructuredGrid]:
        """Materialise each connected component as an independent grid.

        Returns one :class:`UnstructuredGrid` per ``component`` label, in
        label order. A single-component grid returns ``[self]`` unchanged.
        Node ids are kept global (the shared ``nodes`` array is reused), so
        only cell ids, face ids, ``nnc`` and ``props`` are re-indexed.
        """
        n_comp = int(self.component.max()) + 1 if self.component.size else 0
        if n_comp <= 1:
            return [self]

        nb = self.faces.neighbours
        n_faces = self.faces.nodes.shape[0]
        out: list[UnstructuredGrid] = []
        for cid in range(n_comp):
            cell_mask = self.component == cid
            cell_remap = np.full(self.n_cells, -1, dtype=np.int64)
            cell_remap[cell_mask] = np.arange(int(cell_mask.sum()))

            # A face belongs to the component if either neighbour cell does.
            in0 = cell_mask[nb[:, 0]]                 # col 0 is always a real id
            in1 = np.zeros(nb.shape[0], dtype=bool)
            v1 = nb[:, 1] != -1
            in1[v1] = cell_mask[nb[v1, 1]]
            face_mask = in0 | in1
            face_remap = np.full(n_faces, -1, dtype=np.int64)
            face_remap[face_mask] = np.arange(int(face_mask.sum()))

            sub_nb = nb[face_mask].copy()
            for col in (0, 1):
                m = sub_nb[:, col] != -1
                sub_nb[m, col] = cell_remap[sub_nb[m, col]]
            sub_nb = _move_minus_one_to_col1(sub_nb)

            sub_faces = Faces(
                nodes=self.faces.nodes[face_mask],
                neighbours=sub_nb,
                direction=self.faces.direction[face_mask],
            )
            if self.nnc.size:
                nm = cell_mask[self.nnc[:, 0]] & cell_mask[self.nnc[:, 1]]
                sub_nnc = cell_remap[self.nnc[nm]]
            else:
                sub_nnc = np.empty((0, 2), dtype=np.int64)

            n_sub = int(cell_mask.sum())
            out.append(UnstructuredGrid(
                nodes=self.nodes,
                faces=sub_faces,
                cell_faces=_remap_cell_faces(
                    self.cell_faces[cell_mask], face_remap,
                ),
                cell_corners=self.cell_corners[cell_mask],
                original_ijk=self.original_ijk[cell_mask],
                cart_dims=self.cart_dims,
                nnc=sub_nnc,
                component=np.zeros(n_sub, dtype=np.int64),
                props={k: v[cell_mask] for k, v in self.props.items()},
                n_corner_nodes=self.n_corner_nodes,
            ))
        return out


# Backward-compat type alias: existing code (e.g. cpg2unstructured.graph)
# imports `_TopoMesh` for type hints. The class was renamed in 
# without changing field shape — the alias keeps those imports working.
_TopoMesh = UnstructuredGrid


def _remap_cell_faces(cell_faces: np.ndarray, face_remap: np.ndarray) -> np.ndarray:
    """Remap face ids, keep ``-1`` padding, and drop padding columns
    that no cell uses any more (never the 6 slot columns)."""
    out = np.where(cell_faces >= 0, face_remap[np.maximum(cell_faces, 0)], -1)
    used = (out[:, 6:] != -1).any(axis=0)
    width = 6 + (int(np.nonzero(used)[0][-1]) + 1 if used.any() else 0)
    return out[:, :width]


def attach_props(grid: UnstructuredGrid, grdecl: Grdecl) -> UnstructuredGrid:
    """Filter each property in ``grdecl.props`` to active cells in the
    grid's order. Returns a NEW :class:`UnstructuredGrid` (frozen) with
    the populated ``props`` dict — the input grid is not mutated.

    Active-cell filtering uses :attr:`IndexMap.linear_to_cell`:
    ``new_props[k] = grdecl.props[k][index_map.linear_to_cell].copy()``.

    Raises ``ValueError`` if ``grdecl.cart_dims`` differs from
    ``grid.cart_dims``. Property dtypes are preserved (an earlier whitelist
    classifies float vs int32 at parse time).
    """
    if grid.cart_dims != grdecl.cart_dims:
        raise ValueError(
            f"cart_dims mismatch: grid={grid.cart_dims} vs "
            f"grdecl={grdecl.cart_dims}"
        )
    idx = grid.index_map.linear_to_cell      # (n_active,)
    new_props = {k: v[idx].copy() for k, v in grdecl.props.items()}
    return dataclasses.replace(grid, props=new_props, _index_map=None)


def _read_cell_corners_from_P(P: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Return (n_cells, 8) int64 of node ids per cell in Eclipse F-order.

    Cell `c = i + j*nx + k*nx*ny` (NX-fastest); corner ordering per
    `_LOCAL_CORNERS`.
    """
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    i_flat = i_idx.ravel(order="F")
    j_flat = j_idx.ravel(order="F")
    k_flat = k_idx.ravel(order="F")
    n_cells = nx * ny * nz
    out = np.empty((n_cells, 8), dtype=np.int64)
    for o, (di, dj, dk) in enumerate(_LOCAL_CORNERS):
        out[:, o] = P[2 * i_flat + di, 2 * j_flat + dj, 2 * k_flat + dk]
    return out


def _real_cell_mask(cell_corners: np.ndarray) -> np.ndarray:
    """Return (n_cells,) bool — True iff every face has ≥ 3 unique node ids.

    Per spec rationale: a cell is "zero-volume" when at least
    one of its 6 faces collapses to fewer than 3 unique nodes (i.e.,
    the face degenerates from a quad to a line or point). Wedge cells
    with one coincident corner-pair (7 unique corners, two faces with
    3 unique nodes) survive — their topology is still valid.
    Cells with all 8 corners equal to -1 (inactive in an earlier
    sentinel-fill) have every face at 1 unique → caught here.
    """
    n_cells = cell_corners.shape[0]
    is_real = np.ones(n_cells, dtype=bool)
    for face_idx_set in _LOCAL_FACES:
        face_corners = cell_corners[:, face_idx_set]      # (n_cells, 4)
        s = np.sort(face_corners, axis=1)
        # number-of-dup-pairs across (0-1), (1-2), (2-3) gives 4 - unique.
        n_dup = (
            (s[:, 0] == s[:, 1]).astype(np.int8)
            + (s[:, 1] == s[:, 2]).astype(np.int8)
            + (s[:, 2] == s[:, 3]).astype(np.int8)
        )
        face_unique = 4 - n_dup
        is_real &= face_unique >= 3
    return is_real


def _active_lookup(idx_arr: np.ndarray, actnum_flat: np.ndarray) -> np.ndarray:
    """Bool array: True iff idx_arr[k] != -1 AND actnum_flat[idx_arr[k]]."""
    result = np.zeros_like(idx_arr, dtype=bool)
    mask = idx_arr != -1
    if mask.any():
        result[mask] = actnum_flat[idx_arr[mask]]
    return result


def _pinch_nncs(
    keep3: np.ndarray, pin3: np.ndarray, o2n3: np.ndarray,
    nx: int, ny: int, nz: int,
) -> np.ndarray:
    """Cross-layer NNC pairs (new cell ids) for pinch-bridged cells.

    For each column (i, j), consecutive surviving cells separated *only*
    by pinched-out cells (no genuine ACTNUM=0 gap) are connected by an
    NNC — the pinch collapsed the layer between them to zero thickness,
    so they are physically adjacent. Returns sorted unique ``(M, 2)``
    int64 pairs (smaller new id in column 0).
    """
    pairs: set[tuple[int, int]] = set()
    for i in range(nx):
        for j in range(ny):
            ks = np.where(keep3[i, j])[0]
            for a, b in zip(ks[:-1], ks[1:]):
                # gap between two survivors, filled entirely by pinches
                if b > a + 1 and pin3[i, j, a + 1:b].all():
                    ca, cb = int(o2n3[i, j, a]), int(o2n3[i, j, b])
                    pairs.add((min(ca, cb), max(ca, cb)))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(pairs), dtype=np.int64)


def _connected_components(
    n_cells: int, neighbours: np.ndarray, nnc: np.ndarray,
) -> np.ndarray:
    """0-based component label per cell over the face + NNC adjacency.

    Two cells are connected if they share an inner face or an NNC. Labels
    are assigned in order of first cell id, so cell 0 is always in
    component 0 and a fully-connected grid is all zeros.
    """
    parent = np.arange(n_cells, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = root, int(parent[x])
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if neighbours.size:
        for a, b in neighbours[neighbours[:, 1] != -1]:
            union(int(a), int(b))
    for a, b in nnc:
        union(int(a), int(b))

    out = np.empty(n_cells, dtype=np.int64)
    label_of: dict[int, int] = {}
    nxt = 0
    for i in range(n_cells):
        r = find(i)
        if r not in label_of:
            label_of[r] = nxt
            nxt += 1
        out[i] = label_of[r]
    return out


def build_topology(cube: CornerCube) -> UnstructuredGrid:
    """Assemble i+j+k faces, invert to cell_faces, drop inactive +
    zero-volume cells, and rebuild contiguous cell ids.

    Faulted i/j pillar-pair stacks are fault-aware: their raw same-k
    faces are replaced by the MRST sweep output of
    :func:`_fault_stack_faces` — one face per pair of cells that really
    overlap across the fault, plus a boundary face for each exposed
    piece of a fault side. A cell side on a fault can therefore carry
    several faces; see :class:`UnstructuredGrid` for how
    ``cell_faces`` stores them.

    Returns the topology-only :class:`UnstructuredGrid`.
    """
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    n_old = nx * ny * nz

    # i/j faults are handled below via _fault_stack_faces. k-direction
    # faults still raise: the i/j pillar-pair sweep does not apply to
    # k-faces, which span 4 pillars, not 2.
    if _faulted_pillar_mask_k(P, actnum_3d).any():
        i, j, k = map(int, np.argwhere(
            _faulted_pillar_mask_k(P, actnum_3d),
        )[0])
        raise NonMatchingZError(
            f"non-matching-Z across k={k + 1} at (i={i}, j={j}) — "
            "k-direction fault dispatch not yet implemented (v2 only)"
        )

    # Cell index lookup with boundary padding — stores original F-order
    # cell index (0..n_old-1) or -1 for boundary positions.
    orig_idx_3d = np.arange(n_old, dtype=np.int64).reshape(nx, ny, nz, order="F")
    actnum_flat = actnum_3d.ravel(order="F")

    pad_i = np.full((nx + 2, ny, nz), -1, dtype=np.int64)
    pad_i[1:-1] = orig_idx_3d
    pad_j = np.full((nx, ny + 2, nz), -1, dtype=np.int64)
    pad_j[:, 1:-1] = orig_idx_3d
    pad_k = np.full((nx, ny, nz + 2), -1, dtype=np.int64)
    pad_k[:, :, 1:-1] = orig_idx_3d

    # Per-axis raw face slot tables: low/high original cell index + keep mask.
    i_quads = _i_face_quads(P).reshape(-1, 4)
    i_low_orig  = pad_i[0:nx + 1].ravel()
    i_high_orig = pad_i[1:nx + 2].ravel()
    i_keep = _active_lookup(i_low_orig, actnum_flat) | _active_lookup(i_high_orig, actnum_flat)

    j_quads = _j_face_quads(P).reshape(-1, 4)
    j_low_orig  = pad_j[:, 0:ny + 1].ravel()
    j_high_orig = pad_j[:, 1:ny + 2].ravel()
    j_keep = _active_lookup(j_low_orig, actnum_flat) | _active_lookup(j_high_orig, actnum_flat)

    k_quads = _k_face_quads(P).reshape(-1, 4)
    k_low_orig  = pad_k[:, :, 0:nz + 1].ravel()
    k_high_orig = pad_k[:, :, 1:nz + 2].ravel()
    k_keep = _active_lookup(k_low_orig, actnum_flat) | _active_lookup(k_high_orig, actnum_flat)

    # Faulted stacks: drop their raw same-k i/j faces (interior planes
    # only) and take the fault-aware faces instead. Edge-intersection
    # nodes of fault polygons are numbered after cube.nodes, i first.
    n_cube = cube.nodes.shape[0]
    i_res = _fault_stack_faces(cube, 0, n_cube)
    j_res = _fault_stack_faces(cube, 1, n_cube + i_res.new_points.shape[0])
    i_faulted, i_fq, i_flo, i_fhi = (
        i_res.stack_faulted, i_res.polys, i_res.low_orig, i_res.high_orig)
    j_faulted, j_fq, j_flo, j_fhi = (
        j_res.stack_faulted, j_res.polys, j_res.low_orig, j_res.high_orig)
    if i_faulted.any():
        i_drop = np.zeros((nx + 1, ny, nz), dtype=bool)
        i_drop[1:nx] = i_faulted[:, :, None]
        i_keep &= ~i_drop.ravel()
    if j_faulted.any():
        j_drop = np.zeros((nx, ny + 1, nz), dtype=bool)
        j_drop[:, 1:ny] = j_faulted[:, :, None]
        j_keep &= ~j_drop.ravel()

    def _filter_neighbours(low_orig, high_orig, keep):
        """Replace inactive-cell positions with -1, then apply keep mask."""
        low_active  = _active_lookup(low_orig,  actnum_flat)
        high_active = _active_lookup(high_orig, actnum_flat)
        lo = np.where(low_active,  low_orig,  -1)
        hi = np.where(high_active, high_orig, -1)
        return lo[keep], hi[keep]

    def _all(n):
        return np.ones(n, dtype=bool)

    i_l, i_h = _filter_neighbours(
        np.concatenate([i_low_orig, i_flo]),
        np.concatenate([i_high_orig, i_fhi]),
        np.concatenate([i_keep, _all(i_flo.size)]),
    )
    j_l, j_h = _filter_neighbours(
        np.concatenate([j_low_orig, j_flo]),
        np.concatenate([j_high_orig, j_fhi]),
        np.concatenate([j_keep, _all(j_flo.size)]),
    )
    k_l, k_h = _filter_neighbours(k_low_orig, k_high_orig, k_keep)

    # Faces are rows of M sorted node ids, M = 4 unless a fault polygon
    # has more corners; shorter faces repeat ids (see Faces).
    n_cols = max(4, i_fq.shape[1], j_fq.shape[1])

    def _rows(raw: np.ndarray, fault: np.ndarray) -> np.ndarray:
        return np.concatenate([
            _pad_rows(np.sort(raw, axis=-1), n_cols),
            np.sort(_pad_rows(fault, n_cols), axis=-1),
        ])

    i_q = _rows(i_quads[i_keep], i_fq)
    j_q = _rows(j_quads[j_keep], j_fq)
    k_q = _pad_rows(np.sort(k_quads[k_keep], axis=-1), n_cols)

    all_quads = np.concatenate([i_q, j_q, k_q], axis=0)
    all_low   = np.concatenate([i_l, j_l, k_l])
    all_high  = np.concatenate([i_h, j_h, k_h])
    n_i, n_j, n_k = i_q.shape[0], j_q.shape[0], k_q.shape[0]
    all_dirs = np.concatenate([
        np.zeros(n_i, dtype=np.int8),
        np.ones (n_j, dtype=np.int8),
        np.full (n_k, 2, dtype=np.int8),
    ])

    # Global dedup.
    uniq, first_idx, inverse = np.unique(
        all_quads, axis=0, return_index=True, return_inverse=True
    )
    n_uniq = uniq.shape[0]

    # Cell corners + real-cell mask (zero-volume detection).
    cell_corners = _read_cell_corners_from_P(P, nx, ny, nz)
    is_real = _real_cell_mask(cell_corners)
    keep_cells = actnum_flat & is_real
    n_new = int(keep_cells.sum())
    old_to_new = np.full(n_old, -1, dtype=np.int64)
    old_to_new[keep_cells] = np.arange(n_new)

    # Dropped (zero-volume) cells are no face's neighbour. Do this before
    # merging: the k-faces above and below a zero-thickness active cell
    # have the same nodes, so the merge would see 3 cells on one face.
    for side in (all_low, all_high):
        m_side = side != -1
        side[m_side] = np.where(keep_cells[side[m_side]], side[m_side], -1)

    # Merge neighbours per unique face. For matching-Z grids each face
    # appears in exactly one raw row, so this loop is defensive — but
    # cheap relative to the np.unique call above.
    merged_nb = np.full((n_uniq, 2), -1, dtype=np.int64)
    for raw_id in range(all_quads.shape[0]):
        u = int(inverse[raw_id])
        for v in (int(all_low[raw_id]), int(all_high[raw_id])):
            if v == -1:
                continue
            if merged_nb[u, 0] == v or merged_nb[u, 1] == v:
                continue
            if merged_nb[u, 0] == -1:
                merged_nb[u, 0] = v
            elif merged_nb[u, 1] == -1:
                merged_nb[u, 1] = v
            else:
                raise RuntimeError(
                    f"face {u} has more than 2 neighbours: "
                    f"{merged_nb[u]} + {v}"
                )

    # First-occurrence direction.
    merged_dir = all_dirs[first_idx]

    # Remap neighbours with new cell ids; cells that were dropped become -1.
    nb_remapped = merged_nb.copy()
    for col in (0, 1):
        m = nb_remapped[:, col] != -1
        nb_remapped[m, col] = old_to_new[nb_remapped[m, col]]

    # Drop bridge faces (both sides dropped or boundary).
    face_keep = (nb_remapped[:, 0] != -1) | (nb_remapped[:, 1] != -1)
    nb_final = _move_minus_one_to_col1(nb_remapped[face_keep])

    n_kept_faces = int(face_keep.sum())
    face_old_to_new = np.full(n_uniq, -1, dtype=np.int64)
    face_old_to_new[face_keep] = np.arange(n_kept_faces)

    # Edge-intersection nodes: keep those a kept face uses, in order.
    face_nodes = uniq[face_keep]
    new_points = np.concatenate([i_res.new_points, j_res.new_points], axis=0)
    nodes_out = cube.nodes
    if new_points.shape[0]:
        used = np.zeros(new_points.shape[0], dtype=bool)
        used[face_nodes[face_nodes >= n_cube] - n_cube] = True
        node_map = np.arange(n_cube + new_points.shape[0], dtype=np.int64)
        node_map[n_cube:][used] = n_cube + np.arange(int(used.sum()))
        face_nodes = node_map[face_nodes]        # monotone: rows stay sorted
        nodes_out = np.concatenate([cube.nodes, new_points[used]], axis=0)

    final_faces = Faces(
        nodes=face_nodes,
        neighbours=nb_final,
        direction=merged_dir[face_keep],
    )

    # Cell-faces inversion. Walk raw faces; assign each to the right
    # slot of each adjacent cell. A fault side can carry several faces:
    # the first fills the slot, the rest go to padding columns 6+.
    cell_faces_old = np.full((n_old, 6), -1, dtype=np.int64)
    extra_faces: dict[int, list[int]] = {}

    def _assign(cell: int, slot: int, face_id: int) -> None:
        current = int(cell_faces_old[cell, slot])
        if current == -1:
            cell_faces_old[cell, slot] = face_id
        elif current != face_id:
            more = extra_faces.setdefault(cell, [])
            if face_id not in more:
                more.append(face_id)

    for raw_id in range(all_quads.shape[0]):
        u = int(inverse[raw_id])
        new_face_id = int(face_old_to_new[u])
        if new_face_id == -1:
            continue
        d = int(all_dirs[raw_id])
        low = int(all_low[raw_id])
        high = int(all_high[raw_id])
        if low != -1 and keep_cells[low]:
            _assign(low, _SLOT_LOW[d], new_face_id)
        if high != -1 and keep_cells[high]:
            _assign(high, _SLOT_HIGH[d], new_face_id)

    if (cell_faces_old[keep_cells] == -1).any():
        raise RuntimeError("active+real cell has an unfilled face slot — bug")

    n_extra = max((len(v) for v in extra_faces.values()), default=0)
    if n_extra:
        padded = np.full((n_old, 6 + n_extra), -1, dtype=np.int64)
        padded[:, :6] = cell_faces_old
        for cell, more in extra_faces.items():
            padded[cell, 6:6 + len(more)] = more
        cell_faces_old = padded

    cell_faces_new = cell_faces_old[keep_cells]
    cell_corners_new = cell_corners[keep_cells]

    # Eclipse-natural original_ijk from F-order kept indices.
    kept_old_idx = np.where(keep_cells)[0]
    i_idx = kept_old_idx % nx
    j_idx = (kept_old_idx // nx) % ny
    k_idx = kept_old_idx // (nx * ny)
    original_ijk = np.stack([i_idx, j_idx, k_idx], axis=1).astype(np.int64)

    #  — NNCs bridging the cells above/below each pinched-out layer.
    pinched_3d = cube.pinched_3d
    if pinched_3d is not None and pinched_3d.any():
        keep3 = keep_cells.reshape(nx, ny, nz, order="F")
        o2n3 = old_to_new.reshape(nx, ny, nz, order="F")
        nnc = _pinch_nncs(keep3, pinched_3d, o2n3, nx, ny, nz)
    else:
        nnc = np.empty((0, 2), dtype=np.int64)

    #  — connected-component labels over the face + NNC adjacency.
    component = _connected_components(n_new, nb_final, nnc)

    return UnstructuredGrid(
        nodes=nodes_out,
        faces=final_faces,
        cell_faces=cell_faces_new,
        cell_corners=cell_corners_new,
        original_ijk=original_ijk,
        cart_dims=cube.cart_dims,
        nnc=nnc,
        component=component,
        n_corner_nodes=n_cube,
    )
