"""Unstructured-mesh face finders for matching-Z CPG.

PR-6 covers i-faces + j-faces (regular, matching-Z). PR-7 adds
k-faces and cell-face inversion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpg2unstructured._errors import NonMatchingZError
from cpg2unstructured.cpg import CornerCube

__all__ = ["Faces", "find_i_faces", "find_j_faces", "find_ij_faces"]


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
