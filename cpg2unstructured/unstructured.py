"""Unstructured-mesh face finders + topology builder for matching-Z CPG.

PR-6 covers i-faces + j-faces (regular, matching-Z).
PR-7 adds k-faces, cell-face inversion, active-cell rebuild,
and the topology-only orchestrator `build_topology`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpg2unstructured._errors import NonMatchingZError
from cpg2unstructured.cpg import CornerCube

__all__ = [
    "Faces",
    "find_i_faces", "find_j_faces", "find_k_faces", "find_ij_faces",
    "build_topology",
]


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


def _verify_matching_z_i(P: np.ndarray, actnum_3d: np.ndarray) -> None:
    """Raise NonMatchingZError if an active-active i-plane disagrees.

    Compares only at corner-slot positions where BOTH adjacent cells
    are active. PR-5's `-1` sentinel for inactive cells would
    otherwise produce spurious mismatches (D-PR6-3).
    """
    nx = actnum_3d.shape[0]
    for i in range(1, nx):
        right_of_left = P[2 * i - 1]
        left_of_right = P[2 * i]
        mask = _expand_actnum_axis_i(actnum_3d[i - 1]) & _expand_actnum_axis_i(actnum_3d[i])
        diff = (right_of_left != left_of_right) & mask
        if diff.any():
            jk = np.argwhere(diff)[0]
            raise NonMatchingZError(
                f"non-matching-Z across i={i} at (j={jk[0] // 2}, "
                f"k={jk[1] // 2}) — v2 only"
            )


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


def find_i_faces(cube: CornerCube) -> Faces:
    """All i-direction faces (inner + boundary) as a `Faces` record."""
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    _verify_matching_z_i(P, actnum_3d)

    quads_raw = _i_face_quads(P)
    quads = np.sort(quads_raw, axis=-1)
    quads_flat = quads.reshape(-1, 4)

    flat_id = _flat_active_id(actnum_3d)
    cell_id = np.full((nx + 2, ny, nz), -1, dtype=np.int64)
    cell_id[1:-1] = flat_id.reshape(nx, ny, nz, order="F")
    left  = cell_id[0:nx + 1]
    right = cell_id[1:nx + 2]

    keep = ((left != -1) | (right != -1)).ravel()
    faces_kept = quads_flat[keep]
    nb_raw = np.stack([left.ravel(), right.ravel()], axis=1)[keep]
    nb = _move_minus_one_to_col1(nb_raw)
    direction = np.zeros(faces_kept.shape[0], dtype=np.int8)

    return Faces(nodes=faces_kept, neighbours=nb, direction=direction)


# ── j-faces (symmetric, i ↔ j) ──────────────────────────────────────────────


def _verify_matching_z_j(P: np.ndarray, actnum_3d: np.ndarray) -> None:
    """Raise NonMatchingZError if an active-active j-plane disagrees."""
    ny = actnum_3d.shape[1]
    for j in range(1, ny):
        front_of_low = P[:, 2 * j - 1]
        back_of_high = P[:, 2 * j]
        mask_l = _expand_actnum_axis_i(actnum_3d[:, j - 1])
        mask_r = _expand_actnum_axis_i(actnum_3d[:, j])
        mask = mask_l & mask_r
        diff = (front_of_low != back_of_high) & mask
        if diff.any():
            ik = np.argwhere(diff)[0]
            raise NonMatchingZError(
                f"non-matching-Z across j={j} at (i={ik[0] // 2}, "
                f"k={ik[1] // 2}) — v2 only"
            )


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


def find_j_faces(cube: CornerCube) -> Faces:
    """All j-direction faces (inner + boundary) as a `Faces` record."""
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    _verify_matching_z_j(P, actnum_3d)

    quads_raw = _j_face_quads(P)
    quads = np.sort(quads_raw, axis=-1)
    quads_flat = quads.reshape(-1, 4)

    flat_id = _flat_active_id(actnum_3d)
    cell_id = np.full((nx, ny + 2, nz), -1, dtype=np.int64)
    cell_id[:, 1:-1] = flat_id.reshape(nx, ny, nz, order="F")
    low  = cell_id[:, 0:ny + 1]
    high = cell_id[:, 1:ny + 2]

    keep = ((low != -1) | (high != -1)).ravel()
    faces_kept = quads_flat[keep]
    nb_raw = np.stack([low.ravel(), high.ravel()], axis=1)[keep]
    nb = _move_minus_one_to_col1(nb_raw)
    direction = np.ones(faces_kept.shape[0], dtype=np.int8)

    return Faces(nodes=faces_kept, neighbours=nb, direction=direction)


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


def _verify_matching_z_k(P: np.ndarray, actnum_3d: np.ndarray) -> None:
    """Raise NonMatchingZError if an active-active k-plane disagrees.

    Mirrors `_verify_matching_z_i` along the k axis (D-PR7-5).
    Cartesian + pinchout decks pass; stair-stepped non-matching-Z grids
    raise.
    """
    nz = actnum_3d.shape[2]
    for k in range(1, nz):
        bot_of_top = P[:, :, 2 * k - 1]
        top_of_bot = P[:, :, 2 * k]
        mask_l = np.repeat(np.repeat(actnum_3d[:, :, k - 1], 2, axis=0), 2, axis=1)
        mask_r = np.repeat(np.repeat(actnum_3d[:, :, k],     2, axis=0), 2, axis=1)
        mask = mask_l & mask_r
        diff = (bot_of_top != top_of_bot) & mask
        if diff.any():
            ij = np.argwhere(diff)[0]
            raise NonMatchingZError(
                f"non-matching-Z across k={k} at (i={ij[0] // 2}, "
                f"j={ij[1] // 2}) — v2 only"
            )


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
    _verify_matching_z_k(P, actnum_3d)

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
class _TopoMesh:
    """Topology-only mesh — geometry lands in PR-8.

    `nodes` carries the unique XYZ from `CornerCube.nodes`; node ids
    referenced by `faces.nodes` and `cell_corners` index into this array.
    `original_ijk[c]` is the Eclipse-natural `(i, j, k)` of the cell that
    used to be at new id `c`.
    """
    nodes: np.ndarray              # (Nnodes, 3) float64
    faces: Faces                   # merged i+j+k
    cell_faces: np.ndarray         # (Ncells_active, 6) int64
    cell_corners: np.ndarray       # (Ncells_active, 8) int64 — node ids
    original_ijk: np.ndarray       # (Ncells_active, 3) int64
    cart_dims: tuple[int, int, int]


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


def build_topology(cube: CornerCube) -> _TopoMesh:
    """Assemble i+j+k faces, invert to cell_faces, drop inactive +
    zero-volume cells, and rebuild contiguous cell ids.

    Returns the topology-only `_TopoMesh`. Geometry (centroids, areas,
    volumes) lands in PR-8.
    """
    nx, ny, nz = cube.cart_dims
    P = cube.P
    actnum_3d = cube.actnum_3d
    n_old = nx * ny * nz

    _verify_matching_z_i(P, actnum_3d)
    _verify_matching_z_j(P, actnum_3d)
    _verify_matching_z_k(P, actnum_3d)

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

    return _TopoMesh(
        nodes=cube.nodes,
        faces=final_faces,
        cell_faces=cell_faces_new,
        cell_corners=cell_corners_new,
        original_ijk=original_ijk,
        cart_dims=cube.cart_dims,
    )
