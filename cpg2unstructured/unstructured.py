"""Unstructured-mesh face finders + topology builder for matching-Z CPG.

PR-6 covers i-faces + j-faces (regular, matching-Z).
PR-7 adds k-faces, cell-face inversion, active-cell rebuild,
and the topology-only orchestrator `build_topology`.
PR-12 promotes the topology-only `_TopoMesh` to the public
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


# ── phase 2-faults · PR-5 ───────────────────────────────────────────
# Imports for the fault-face glue function (used internally).
# The actual integration into find_*_faces lands in PR-6/PR-7/PR-8.
from cpg2unstructured._fault_connections import find_connections
from cpg2unstructured._fault_geometry import compute_face_geometry


# Local-corner ordering — matches PR-5's `_LOCAL_CORNERS` (see D-PR7-6).
_LOCAL_CORNERS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)

# Direction → (low-cell slot, high-cell slot) in cell_faces[i-, i+, j-, j+, k-, k+]
# (see D-PR7-7).
_SLOT_LOW  = (1, 3, 5)   # low side gets the "+" slot
_SLOT_HIGH = (0, 2, 4)   # high side gets the "-" slot


# Local face → 4 cell-corner indices (per `_LOCAL_CORNERS` order).
# Matches PR-2's `_toy._LOCAL_FACES` slot pattern; used by `_real_cell_mask`
# for the per-face uniqueness check (D-PR7-8).
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
        (Nfaces, 4) int64 — sorted node ids per face (sort axis 1).
    neighbours
        (Nfaces, 2) int64 — adjacent cell ids. Inner faces: smaller id
        in column 0. Boundary faces: -1 in column 1.
    direction
        (Nfaces,) int8 — 0 = i-face, 1 = j-face, 2 = k-face (PR-7).
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
    are active. PR-5's ``-1`` sentinel for inactive cells would
    otherwise produce spurious mismatches (D-PR6-3).

    Replaces phase 1-design's ``_verify_matching_z_i`` raise-on-fault
    helper. Phase 2-faults (D-2F-1) widens ``find_*_faces`` to dispatch
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
    to the only available side. See D-PR6-4.
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


def _find_faulted_i_faces(cube: CornerCube) -> tuple[Faces, np.ndarray]:
    """Build i-direction fault faces from non-matching-Z pillar pairs.

    Walks every faulted (i, j) pillar-pair stack and emits a face per
    connection produced by :func:`find_connections`.

    Restrictions (phase 2-faults PR-5 scope — will be relaxed in PR-9):

    1. **All-active stacks only.** Every cell in the stack on both
       A-side (column i) and B-side (column i+1) must be active.
       Inactive cells in the middle of a fault stack raise
       :exc:`NonMatchingZError`.
    2. **4-corner fault polygons only.** If
       :func:`compute_face_geometry` emits a polygon with anything
       other than 4 corners (i.e. the face has envelope crossings
       or a diamond case from slanted cells), raise
       :exc:`NonMatchingZError`. The most common faulted-deck case
       — rigid blocks with vertical Z offsets and flat cell
       tops/bots, e.g. the ``simple_faulted_2x2x2`` fixture —
       always produces 4-corner faults.

    Returns
    -------
    faces : :class:`Faces`
        Fault i-faces. ``faces.nodes`` is ``(n_faulted_faces, 4)``
        of sorted node IDs (same convention as matching-Z output).
        ``faces.neighbours`` is ``(n_faulted_faces, 2)``, ``-1`` in
        column 1 means a boundary fault face (rare; happens at the
        grid edge with mixed active/boundary cells on a faulted
        pillar pair).  ``faces.direction`` is all-0 (i-direction).
    new_points : (n_points, 3) float64
        Updated node table — copy of ``cube.nodes`` since under PR-5
        restrictions no intersection points are appended. PR-9 will
        relax this and append intersection nodes here.

    Notes
    -----
    No dZ artificial-offset trick (per-stack loop instead). The trick
    is only needed if all faulted stacks are processed in one
    ``find_connections`` call; we process them individually for
    clarity. Performance impact: negligible on typical decks (few
    faulted stacks per direction).
    """
    P = cube.P
    nodes = cube.nodes
    actnum_3d = cube.actnum_3d
    nx, ny, nz = cube.cart_dims

    faulted_mask = _faulted_pillar_mask_i(P, actnum_3d)
    if not faulted_mask.any():
        return _empty_faces(direction=0), nodes

    # Aggregate to per-(i, j) stack: True if any cell in the column
    # has a faulted i-face on this pillar pair.
    stack_faulted = faulted_mask.any(axis=2)         # (NX-1, NY)

    # Global F-order cell index → active-cell ID (PR-7 style).
    actnum_flat = actnum_3d.ravel(order="F")
    flat_active_id = np.full(nx * ny * nz, -1, dtype=np.int64)
    flat_active_id[actnum_flat] = np.arange(int(actnum_flat.sum()))

    # Per-row z-slot for the (NZ+1)-row za/zb arrays.
    # Row 0      → top of cell 0          (z-slot 0)
    # Row k      → top of cell k = bot of cell k-1  (z-slot 2*k)
    # Row NZ     → bot of cell NZ-1       (z-slot 2*NZ - 1)
    if nz > 0:
        z_slots_for_rows = np.concatenate(
            [2 * np.arange(nz), [2 * nz - 1]],
        )
    else:
        z_slots_for_rows = np.zeros(0, dtype=np.int64)

    bot_slots = 2 * np.arange(nz) + 1                # bot z-slot per cell
    top_slots = 2 * np.arange(nz)                    # top z-slot per cell

    quads_chunks: list[np.ndarray] = []
    nbs_chunks: list[np.ndarray] = []

    for (i_int, j_int) in zip(*np.where(stack_faulted)):
        i = int(i_int)
        j = int(j_int)

        # Active-cell IDs along the A-stack (column i) and B-stack
        # (column i+1).
        a_global = i + j * nx + np.arange(nz) * (nx * ny)
        b_global = (i + 1) + j * nx + np.arange(nz) * (nx * ny)
        if not (actnum_flat[a_global].all() and actnum_flat[b_global].all()):
            raise NonMatchingZError(
                f"inactive cells in faulted i-stack (i={i}-{i+1}, "
                f"j={j}) — phase 2-faults v0.2.0 supports only "
                "all-active fault stacks (see PR-9 — v2 only)"
            )
        a_active_ids = flat_active_id[a_global]
        b_active_ids = flat_active_id[b_global]

        # Build za / zb : shape (NZ+1, 2)
        # A-side: corners on the +i side of A-cells = P[2i+1, ...]
        # B-side: corners on the -i side of B-cells = P[2i+2, ...]
        # Pillar 1 of the face = j-pillar j  (j-slot 2*j)
        # Pillar 2 of the face = j-pillar j+1 (j-slot 2*j+1)
        za_p1_ids = P[2 * i + 1, 2 * j,     z_slots_for_rows]
        za_p2_ids = P[2 * i + 1, 2 * j + 1, z_slots_for_rows]
        zb_p1_ids = P[2 * i + 2, 2 * j,     z_slots_for_rows]
        zb_p2_ids = P[2 * i + 2, 2 * j + 1, z_slots_for_rows]
        za = np.stack(
            [nodes[za_p1_ids, 2], nodes[za_p2_ids, 2]], axis=1,
        )
        zb = np.stack(
            [nodes[zb_p1_ids, 2], nodes[zb_p2_ids, 2]], axis=1,
        )

        # Build pa / pb : shape (NZ, 4) with
        # column order [bot_p1, bot_p2, top_p1, top_p2].
        pa_stack = np.stack(
            [
                P[2 * i + 1, 2 * j,     bot_slots],
                P[2 * i + 1, 2 * j + 1, bot_slots],
                P[2 * i + 1, 2 * j,     top_slots],
                P[2 * i + 1, 2 * j + 1, top_slots],
            ],
            axis=1,
        )
        pb_stack = np.stack(
            [
                P[2 * i + 2, 2 * j,     bot_slots],
                P[2 * i + 2, 2 * j + 1, bot_slots],
                P[2 * i + 2, 2 * j,     top_slots],
                P[2 * i + 2, 2 * j + 1, top_slots],
            ],
            axis=1,
        )

        # Sweep-line: which (A_cell, B_cell) k-pairs overlap?
        C = find_connections(za, zb)
        if C.shape[0] == 0:
            continue

        # Build per-connection pa/pb rows.
        pa_conn = pa_stack[C[:, 0]]                   # (n_conn, 4)
        pb_conn = pb_stack[C[:, 1]]

        # Geometry — build the fault face polygons.
        new_nodes, num_nodes, corners = compute_face_geometry(
            pa_conn, pb_conn, nodes,
        )

        # PR-5 restriction #2: only 4-corner polygons supported.
        if not (num_nodes == 4).all():
            offender = int(np.argmax(num_nodes != 4))
            raise NonMatchingZError(
                f"fault polygon at connection {offender} of stack "
                f"i={i}-{i+1}, j={j} has {int(num_nodes[offender])} "
                "corners; phase 2-faults v0.2.0 supports only "
                "4-corner fault polygons (see PR-9 — v2 only)"
            )
        # PR-5 restriction (corollary): no new intersection points
        # may be appended (a 4-corner polygon uses only pillar
        # corners → no entries in compute_face_geometry's intersection
        # list).
        if new_nodes.shape[0] != nodes.shape[0]:
            raise NonMatchingZError(
                f"unexpected intersection nodes from fault stack "
                f"i={i}-{i+1}, j={j} — fault polygons are 4-corner "
                "but new_nodes grew; see PR-9 for variable-corner "
                "support (v2 only)"
            )

        quads = corners.reshape(C.shape[0], 4)
        quads_sorted = np.sort(quads, axis=1)         # Faces convention
        nbs = np.stack([a_active_ids[C[:, 0]],
                        b_active_ids[C[:, 1]]], axis=1)
        nbs = _move_minus_one_to_col1(nbs)
        quads_chunks.append(quads_sorted)
        nbs_chunks.append(nbs)

    if not quads_chunks:
        return _empty_faces(direction=0), nodes

    all_quads = np.concatenate(quads_chunks, axis=0)
    all_nbs = np.concatenate(nbs_chunks, axis=0)
    direction = np.zeros(all_quads.shape[0], dtype=np.int8)
    return Faces(nodes=all_quads, neighbours=all_nbs,
                 direction=direction), nodes


def _empty_faces(direction: int = 0) -> Faces:
    """Return an empty Faces record for a given direction (PR-5 helper)."""
    return Faces(
        nodes=np.zeros((0, 4), dtype=np.int64),
        neighbours=np.zeros((0, 2), dtype=np.int64),
        direction=np.zeros(0, dtype=np.int8),
    )


def find_i_faces(cube: CornerCube) -> Faces:
    """All i-direction faces (inner + boundary) as a ``Faces`` record.

    Dispatches matching-Z stacks to the v0.1.0 fast path and faulted
    stacks to :func:`_find_faulted_i_faces` (phase 2-faults D-2F-1).
    When no faulted stacks are present, the output is byte-identical
    to v0.1.0 — Tier A invariance.
    """
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    faulted = _faulted_pillar_mask_i(P, actnum_3d)

    quads_raw = _i_face_quads(P)
    quads = np.sort(quads_raw, axis=-1)
    quads_flat = quads.reshape(-1, 4)

    flat_id = _flat_active_id(actnum_3d)
    cell_id = np.full((nx + 2, ny, nz), -1, dtype=np.int64)
    cell_id[1:-1] = flat_id.reshape(nx, ny, nz, order="F")
    left  = cell_id[0:nx + 1]
    right = cell_id[1:nx + 2]
    keep = ((left != -1) | (right != -1)).ravel()

    if not faulted.any():
        # FAST PATH — byte-identical to v0.1.0 for matching-Z decks.
        faces_kept = quads_flat[keep]
        nb_raw = np.stack([left.ravel(), right.ravel()], axis=1)[keep]
        nb = _move_minus_one_to_col1(nb_raw)
        direction = np.zeros(faces_kept.shape[0], dtype=np.int8)
        return Faces(nodes=faces_kept, neighbours=nb, direction=direction)

    # DISPATCH PATH — drop matching faces in faulted (i, j) stacks and
    # concatenate with _find_faulted_i_faces output.
    stack_faulted = faulted.any(axis=2)              # (nx-1, ny)
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

    fault_faces, _ = _find_faulted_i_faces(cube)

    return Faces(
        nodes=np.concatenate([faces_kept, fault_faces.nodes], axis=0),
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

    Symmetric to :func:`_find_faulted_i_faces` along the j-axis.
    Walks every faulted (i, j) j-pillar-pair stack and emits a face
    per connection produced by :func:`find_connections`.

    Restrictions (same as PR-5/PR-6 — lifted in PR-9):
        - All-active stacks only.
        - 4-corner fault polygons only.

    Returns
    -------
    faces : :class:`Faces`
        Fault j-faces with ``direction = 1``.
    new_points : (n_points, 3) float64
        Updated node table — under the 4-corner restriction, this is
        a copy of ``cube.nodes`` (no intersection nodes appended).
    """
    P = cube.P
    nodes = cube.nodes
    actnum_3d = cube.actnum_3d
    nx, ny, nz = cube.cart_dims

    faulted = _faulted_pillar_mask_j(P, actnum_3d)
    if not faulted.any():
        return _empty_faces(direction=1), nodes

    # Per-(i, j) j-stack mask: any k flagged.
    stack_faulted = faulted.any(axis=2)              # (NX, NY-1)

    actnum_flat = actnum_3d.ravel(order="F")
    flat_active_id = np.full(nx * ny * nz, -1, dtype=np.int64)
    flat_active_id[actnum_flat] = np.arange(int(actnum_flat.sum()))

    if nz > 0:
        z_slots_for_rows = np.concatenate(
            [2 * np.arange(nz), [2 * nz - 1]],
        )
    else:
        z_slots_for_rows = np.zeros(0, dtype=np.int64)

    bot_slots = 2 * np.arange(nz) + 1
    top_slots = 2 * np.arange(nz)

    quads_chunks: list[np.ndarray] = []
    nbs_chunks: list[np.ndarray] = []

    for (i_int, j_int) in zip(*np.where(stack_faulted)):
        i = int(i_int)
        j = int(j_int)

        a_global = i + j * nx + np.arange(nz) * (nx * ny)
        b_global = i + (j + 1) * nx + np.arange(nz) * (nx * ny)
        if not (actnum_flat[a_global].all() and actnum_flat[b_global].all()):
            raise NonMatchingZError(
                f"inactive cells in faulted j-stack (i={i}, j={j}-{j+1}) "
                "— phase 2-faults v0.2.0 supports only all-active fault "
                "stacks (see PR-9 — v2 only)"
            )
        a_active_ids = flat_active_id[a_global]
        b_active_ids = flat_active_id[b_global]

        # Build za / zb along the j-pillar pair (pillars at i and i+1
        # on the j+1 boundary).
        # A-side corners are on the +j side of A-cells: j-slot 2*j+1.
        # B-side corners are on the -j side of B-cells: j-slot 2*j+2.
        # Pillar 1 of the face = i-pillar i  (i-slot 2*i).
        # Pillar 2 of the face = i-pillar i+1 (i-slot 2*i+1).
        za_p1_ids = P[2 * i,     2 * j + 1, z_slots_for_rows]
        za_p2_ids = P[2 * i + 1, 2 * j + 1, z_slots_for_rows]
        zb_p1_ids = P[2 * i,     2 * j + 2, z_slots_for_rows]
        zb_p2_ids = P[2 * i + 1, 2 * j + 2, z_slots_for_rows]
        za = np.stack(
            [nodes[za_p1_ids, 2], nodes[za_p2_ids, 2]], axis=1,
        )
        zb = np.stack(
            [nodes[zb_p1_ids, 2], nodes[zb_p2_ids, 2]], axis=1,
        )

        pa_stack = np.stack(
            [
                P[2 * i,     2 * j + 1, bot_slots],
                P[2 * i + 1, 2 * j + 1, bot_slots],
                P[2 * i,     2 * j + 1, top_slots],
                P[2 * i + 1, 2 * j + 1, top_slots],
            ],
            axis=1,
        )
        pb_stack = np.stack(
            [
                P[2 * i,     2 * j + 2, bot_slots],
                P[2 * i + 1, 2 * j + 2, bot_slots],
                P[2 * i,     2 * j + 2, top_slots],
                P[2 * i + 1, 2 * j + 2, top_slots],
            ],
            axis=1,
        )

        C = find_connections(za, zb)
        if C.shape[0] == 0:
            continue

        pa_conn = pa_stack[C[:, 0]]
        pb_conn = pb_stack[C[:, 1]]

        new_nodes, num_nodes, corners = compute_face_geometry(
            pa_conn, pb_conn, nodes,
        )

        if not (num_nodes == 4).all():
            offender = int(np.argmax(num_nodes != 4))
            raise NonMatchingZError(
                f"fault polygon at connection {offender} of j-stack "
                f"i={i}, j={j}-{j+1} has {int(num_nodes[offender])} "
                "corners; phase 2-faults v0.2.0 supports only "
                "4-corner fault polygons (see PR-9 — v2 only)"
            )
        if new_nodes.shape[0] != nodes.shape[0]:
            raise NonMatchingZError(
                f"unexpected intersection nodes from j-stack "
                f"i={i}, j={j}-{j+1} — see PR-9 for variable-corner "
                "support (v2 only)"
            )

        quads = corners.reshape(C.shape[0], 4)
        quads_sorted = np.sort(quads, axis=1)
        nbs = np.stack(
            [a_active_ids[C[:, 0]], b_active_ids[C[:, 1]]], axis=1,
        )
        nbs = _move_minus_one_to_col1(nbs)
        quads_chunks.append(quads_sorted)
        nbs_chunks.append(nbs)

    if not quads_chunks:
        return _empty_faces(direction=1), nodes

    all_quads = np.concatenate(quads_chunks, axis=0)
    all_nbs = np.concatenate(nbs_chunks, axis=0)
    direction = np.ones(all_quads.shape[0], dtype=np.int8)
    return Faces(nodes=all_quads, neighbours=all_nbs,
                 direction=direction), nodes


def find_j_faces(cube: CornerCube) -> Faces:
    """All j-direction faces (inner + boundary) as a ``Faces`` record.

    Dispatches matching-Z stacks to the v0.1.0 fast path and faulted
    stacks to :func:`_find_faulted_j_faces` (phase 2-faults D-2F-1).
    When no faulted stacks are present, the output is byte-identical
    to v0.1.0 — Tier A invariance.
    """
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    faulted = _faulted_pillar_mask_j(P, actnum_3d)

    quads_raw = _j_face_quads(P)
    quads = np.sort(quads_raw, axis=-1)
    quads_flat = quads.reshape(-1, 4)

    flat_id = _flat_active_id(actnum_3d)
    cell_id = np.full((nx, ny + 2, nz), -1, dtype=np.int64)
    cell_id[:, 1:-1] = flat_id.reshape(nx, ny, nz, order="F")
    low  = cell_id[:, 0:ny + 1]
    high = cell_id[:, 1:ny + 2]
    keep = ((low != -1) | (high != -1)).ravel()

    if not faulted.any():
        # FAST PATH — byte-identical to v0.1.0 for matching-Z decks.
        faces_kept = quads_flat[keep]
        nb_raw = np.stack([low.ravel(), high.ravel()], axis=1)[keep]
        nb = _move_minus_one_to_col1(nb_raw)
        direction = np.ones(faces_kept.shape[0], dtype=np.int8)
        return Faces(nodes=faces_kept, neighbours=nb, direction=direction)

    # DISPATCH PATH — drop matching faces in faulted (i, j) j-stacks
    # and concatenate with _find_faulted_j_faces output.
    stack_faulted = faulted.any(axis=2)              # (NX, NY-1)
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

    fault_faces, _ = _find_faulted_j_faces(cube)

    return Faces(
        nodes=np.concatenate([faces_kept, fault_faces.nodes], axis=0),
        neighbours=np.concatenate([nb, fault_faces.neighbours], axis=0),
        direction=np.concatenate([direction, fault_faces.direction], axis=0),
    )


# ── combined helper ─────────────────────────────────────────────────────────


def find_ij_faces(cube: CornerCube) -> Faces:
    """Concatenate i-faces + j-faces. Convenience for downstream callers."""
    fi = find_i_faces(cube)
    fj = find_j_faces(cube)
    return Faces(
        nodes=np.concatenate([fi.nodes, fj.nodes], axis=0),
        neighbours=np.concatenate([fi.neighbours, fj.neighbours], axis=0),
        direction=np.concatenate([fi.direction, fj.direction], axis=0),
    )


# ── k-faces (PR-7) ──────────────────────────────────────────────────────────


def _faulted_pillar_mask_k(P: np.ndarray, actnum_3d: np.ndarray) -> np.ndarray:
    """Per-cell-pair faulted mask along the k direction.

    Returns
    -------
    (NX, NY, NZ-1) bool — True at (i, j, k) means the active k-face
    between cells (i, j, k) and (i, j, k+1) is non-matching.

    Mirrors :func:`_faulted_pillar_mask_i` along the k axis (D-PR7-5).
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

    Same active-side picker as `_i_face_quads` / `_j_face_quads` (D-PR7-4
    via D-PR6-4): reads both side columns and selects elementwise via
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

    Constructed by :func:`build_topology` (PR-7). Optionally enriched
    with cell properties via :func:`attach_props` (PR-12), which
    returns a NEW grid with the populated ``props`` dict.

    ``nodes`` carries the unique XYZ from ``CornerCube.nodes``; node ids
    referenced by ``faces.nodes`` and ``cell_corners`` index into this
    array. ``original_ijk[c]`` is the Eclipse-natural ``(i, j, k)`` of
    the cell that used to be at new id ``c``.

    ``props`` is keyed by Eclipse keyword names verbatim (uppercase,
    e.g. ``"PORO"``). Use the ``grid["PORO"]`` accessor instead of
    ``grid.props["PORO"]`` for the helpful "available keys" error on
    misses.
    """
    nodes: np.ndarray              # (Nnodes, 3) float64
    faces: Faces                   # merged i+j+k
    cell_faces: np.ndarray         # (Ncells_active, 6) int64
    cell_corners: np.ndarray       # (Ncells_active, 8) int64 — node ids
    original_ijk: np.ndarray       # (Ncells_active, 3) int64
    cart_dims: tuple[int, int, int]
    # (Mnnc, 2) int64 — non-neighbour connections: cell-id pairs that are
    # connected without sharing a geometric face. Populated for cells
    # bridged across a pinched-out layer (PR-12); empty (0, 2) otherwise.
    nnc: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))
    # (Ncells,) int64 — connected-component label per cell over the face
    # + NNC adjacency. 0-based, assigned in order of first cell id, so a
    # single-component grid is all zeros. Use ``split()`` to materialise
    # the components as independent grids (PR-14).
    component: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    props: dict[str, np.ndarray] = field(default_factory=dict)
    _index_map: "IndexMap | None" = field(default=None, repr=False, compare=False)

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
    def index_map(self) -> "IndexMap":
        """Lazy-cached :class:`IndexMap` built from cart_dims + original_ijk.

        Frozen-dataclass cache: stored via ``object.__setattr__`` on
        first access (D-PR12-1) — same pattern ``functools.cached_property``
        uses on frozen dataclasses.
        """
        from cpg2unstructured.index_map import IndexMap
        if self._index_map is None:
            object.__setattr__(
                self, "_index_map",
                IndexMap(self.cart_dims, self.original_ijk),
            )
        return self._index_map  # type: ignore[return-value]

    def split(self) -> "list[UnstructuredGrid]":
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
                cell_faces=face_remap[self.cell_faces[cell_mask]],
                cell_corners=self.cell_corners[cell_mask],
                original_ijk=self.original_ijk[cell_mask],
                cart_dims=self.cart_dims,
                nnc=sub_nnc,
                component=np.zeros(n_sub, dtype=np.int64),
                props={k: v[cell_mask] for k, v in self.props.items()},
            ))
        return out


# Backward-compat type alias: existing code (e.g. cpg2unstructured.graph)
# imports `_TopoMesh` for type hints. The class was renamed in PR-12
# without changing field shape — the alias keeps those imports working.
_TopoMesh = UnstructuredGrid


def attach_props(grid: UnstructuredGrid, grdecl: "Grdecl") -> UnstructuredGrid:
    """Filter each property in ``grdecl.props`` to active cells in the
    grid's order. Returns a NEW :class:`UnstructuredGrid` (frozen) with
    the populated ``props`` dict — the input grid is not mutated.

    Active-cell filtering uses :attr:`IndexMap.linear_to_cell` (PR-9):
    ``new_props[k] = grdecl.props[k][index_map.linear_to_cell].copy()``.

    Raises ``ValueError`` if ``grdecl.cart_dims`` differs from
    ``grid.cart_dims``. Property dtypes are preserved (PR-4's whitelist
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
    `_LOCAL_CORNERS` (D-PR7-6).
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

    Per spec rationale (D-PR7-8): a cell is "zero-volume" when at least
    one of its 6 faces collapses to fewer than 3 unique nodes (i.e.,
    the face degenerates from a quad to a line or point). Wedge cells
    with one coincident corner-pair (7 unique corners, two faces with
    3 unique nodes) survive — their topology is still valid.
    Cells with all 8 corners equal to -1 (inactive in PR-5's
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

    Returns the topology-only `_TopoMesh`. Geometry (centroids, areas,
    volumes) lands in PR-8.
    """
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    n_old = nx * ny * nz

    # Phase 2-faults · PR-8: the top-level i/j fault guard is REMOVED
    # — find_i_faces (PR-6) and find_j_faces (PR-7) now dispatch
    # matching vs faulted inline. k-direction faults still raise
    # downstream in find_k_faces (k-fault dispatch deferred to a
    # follow-up; the i/j-pair sweep-line algorithm does not directly
    # apply to k-faces, which have 4 pillars not 2).
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

    def _filter_neighbours(low_orig, high_orig, keep):
        """Replace inactive-cell positions with -1, then apply keep mask."""
        low_active  = _active_lookup(low_orig,  actnum_flat)
        high_active = _active_lookup(high_orig, actnum_flat)
        l = np.where(low_active,  low_orig,  -1)
        h = np.where(high_active, high_orig, -1)
        return l[keep], h[keep]

    i_l, i_h = _filter_neighbours(i_low_orig, i_high_orig, i_keep)
    j_l, j_h = _filter_neighbours(j_low_orig, j_high_orig, j_keep)
    k_l, k_h = _filter_neighbours(k_low_orig, k_high_orig, k_keep)

    i_q = np.sort(i_quads[i_keep], axis=-1)
    j_q = np.sort(j_quads[j_keep], axis=-1)
    k_q = np.sort(k_quads[k_keep], axis=-1)

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

    # First-occurrence direction (D-PR7-2).
    merged_dir = all_dirs[first_idx]

    # Cell corners + real-cell mask (zero-volume detection).
    cell_corners = _read_cell_corners_from_P(P, nx, ny, nz)
    is_real = _real_cell_mask(cell_corners)
    keep_cells = actnum_flat & is_real
    n_new = int(keep_cells.sum())
    old_to_new = np.full(n_old, -1, dtype=np.int64)
    old_to_new[keep_cells] = np.arange(n_new)

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

    final_faces = Faces(
        nodes=uniq[face_keep],
        neighbours=nb_final,
        direction=merged_dir[face_keep],
    )

    # Cell-faces inversion (D-PR7-7). Walk raw faces; assign each to the
    # right slot of each adjacent cell.
    cell_faces_old = np.full((n_old, 6), -1, dtype=np.int64)
    for raw_id in range(all_quads.shape[0]):
        u = int(inverse[raw_id])
        new_face_id = int(face_old_to_new[u])
        if new_face_id == -1:
            continue
        d = int(all_dirs[raw_id])
        low = int(all_low[raw_id])
        high = int(all_high[raw_id])
        if low != -1 and keep_cells[low]:
            cell_faces_old[low, _SLOT_LOW[d]] = new_face_id
        if high != -1 and keep_cells[high]:
            cell_faces_old[high, _SLOT_HIGH[d]] = new_face_id

    cell_faces_new = cell_faces_old[keep_cells]
    cell_corners_new = cell_corners[keep_cells]

    if (cell_faces_new == -1).any():
        raise RuntimeError("active+real cell has an unfilled face slot — bug")

    # Eclipse-natural original_ijk (D-PR7-3) from F-order kept indices.
    kept_old_idx = np.where(keep_cells)[0]
    i_idx = kept_old_idx % nx
    j_idx = (kept_old_idx // nx) % ny
    k_idx = kept_old_idx // (nx * ny)
    original_ijk = np.stack([i_idx, j_idx, k_idx], axis=1).astype(np.int64)

    # PR-12 — NNCs bridging the cells above/below each pinched-out layer.
    pinched_3d = cube.pinched_3d
    if pinched_3d is not None and pinched_3d.any():
        keep3 = keep_cells.reshape(nx, ny, nz, order="F")
        o2n3 = old_to_new.reshape(nx, ny, nz, order="F")
        nnc = _pinch_nncs(keep3, pinched_3d, o2n3, nx, ny, nz)
    else:
        nnc = np.empty((0, 2), dtype=np.int64)

    # PR-13 — connected-component labels over the face + NNC adjacency.
    component = _connected_components(n_new, nb_final, nnc)

    return UnstructuredGrid(
        nodes=cube.nodes,
        faces=final_faces,
        cell_faces=cell_faces_new,
        cell_corners=cell_corners_new,
        original_ijk=original_ijk,
        cart_dims=cube.cart_dims,
        nnc=nnc,
        component=component,
    )
