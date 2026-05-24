"""CPG geometry assembly: Grdecl → corner cube → unique nodes + P map.

PR-5 scope: `from_grdecl` returns a `CornerCube` carrying
- `nodes`     : (Nnodes, 3) float64 unique XYZ
- `P`         : (2*NX, 2*NY, 2*NZ) int64 indexing into `nodes`
                (or -1 for slots owned by inactive cells)
- `cart_dims` : same as Grdecl.cart_dims
- `actnum_3d` : (NX, NY, NZ) bool convenience reshape of Grdecl.actnum
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from cpg2unstructured.grdecl_reader import Grdecl

__all__ = ["from_grdecl", "CornerCube"]


_QUANT = 1_000_000_000  # 1e9 — must match cpg2unstructured._toy._QUANT (D-PR5-2)


@dataclass(frozen=True)
class CornerCube:
    """Output of corner-cube assembly + node dedup."""
    nodes: np.ndarray
    P: np.ndarray
    cart_dims: tuple[int, int, int]
    actnum_3d: np.ndarray
    # (NX,NY,NZ) bool — cells pinched out (collapsed) and dropped. Carried
    # so build_topology can bridge the surviving neighbours with NNCs
    # (PR-12). None when assembled without pinch tracking.
    pinched_3d: np.ndarray | None = None


def _coord_pillars(g: Grdecl) -> np.ndarray:
    """Reshape g.coord → (NX+1, NY+1, 6) with coord_pp[i, j] = pillar(i, j).

    Eclipse stores COORD NX-fastest in flat (i is the fast index). The
    native reshape of the (NX+1)*(NY+1) pillar block is therefore
    (NY+1, NX+1, 6); transposing the leading two axes recovers the
    intuitive coord_pp[i, j] = pillar(i, j) indexing. See D-PR5-3.
    """
    nx, ny, _ = g.cart_dims
    return g.coord.reshape(ny + 1, nx + 1, 6).transpose(1, 0, 2).copy()


def _zcorn_3d(g: Grdecl) -> np.ndarray:
    """Reshape g.zcorn → (2*NX, 2*NY, 2*NZ) corner-cube layout."""
    nx, ny, nz = g.cart_dims
    return g.zcorn.reshape(2 * nz, 2 * ny, 2 * nx).transpose(2, 1, 0).copy()


def _actnum_3d(g: Grdecl) -> np.ndarray:
    """Reshape g.actnum → (NX, NY, NZ) bool."""
    nx, ny, nz = g.cart_dims
    return g.actnum.reshape(nz, ny, nx).transpose(2, 1, 0).astype(bool).copy()


def _pinched_cell_mask(g: Grdecl, tol: float = 0.0) -> np.ndarray:
    """(NX, NY, NZ) bool — True where a cell is pinched out.

    A cell is pinched (collapsed) when its vertical thickness is ≤ ``tol``
    on *all four* pillars, i.e. the thickest corner is still ≤ tol. Such
    zero-height cells carry no volume and are dropped from the active set
    (MRST `processGRDECL` removes pinched cells; PR-12 then bridges the
    cells above and below with an NNC). With the default ``tol=0.0`` only
    exactly-collapsed cells are caught, so ordinary decks are untouched.
    """
    nx, ny, nz = g.cart_dims
    zc = _zcorn_3d(g)                       # (2*NX, 2*NY, 2*NZ)
    top = zc[:, :, 0::2]                    # k-slots 2k  → (2*NX, 2*NY, NZ)
    bot = zc[:, :, 1::2]                    # k-slots 2k+1
    thick = bot - top                       # per corner-slot thickness
    # collapse the 2×2 (i, j) corner block per cell → max corner thickness
    thick_cell = thick.reshape(nx, 2, ny, 2, nz).max(axis=(1, 3))
    return thick_cell <= tol


def _assemble_corner_xyz(g: Grdecl) -> np.ndarray:
    """Build (2*NX, 2*NY, 2*NZ, 3) of XYZ for every corner-cube slot.

    Linearly interpolates each COORD pillar between its top and bottom
    at the ZCORN-given depth. Cartesian (vertical-pillar) decks reduce
    to constant (x, y).
    """
    nx, ny, nz = g.cart_dims
    coord_pp = _coord_pillars(g)
    zcorn = _zcorn_3d(g)

    pillar_i = np.empty(2 * nx, dtype=np.int64)
    pillar_i[0::2] = np.arange(nx)
    pillar_i[1::2] = np.arange(nx) + 1
    pillar_j = np.empty(2 * ny, dtype=np.int64)
    pillar_j[0::2] = np.arange(ny)
    pillar_j[1::2] = np.arange(ny) + 1
    II, JJ = np.meshgrid(pillar_i, pillar_j, indexing="ij")

    top = coord_pp[II, JJ, 0:3]
    bot = coord_pp[II, JJ, 3:6]
    dz_p = bot[..., 2] - top[..., 2]
    safe = np.where(dz_p == 0.0, 1.0, dz_p)

    out = np.empty((2 * nx, 2 * ny, 2 * nz, 3), dtype=np.float64)
    for k_corner in range(2 * nz):
        z = zcorn[..., k_corner]
        t = (z - top[..., 2]) / safe
        out[..., k_corner, 0] = top[..., 0] + t * (bot[..., 0] - top[..., 0])
        out[..., k_corner, 1] = top[..., 1] + t * (bot[..., 1] - top[..., 1])
        out[..., k_corner, 2] = z
    return out


def _mask_inactive_corners(xyz: np.ndarray, actnum_3d: np.ndarray) -> np.ndarray:
    """NaN-fill every corner-cube slot whose owning cell has ACTNUM=0.

    In corner-cube space each slot is owned by exactly one cell —
    slot (2i+di, 2j+dj, 2k+dk) → cell (i, j, k). After NaN-marking,
    the dedup pass groups all NaN slots into a single sentinel cluster
    via an int64 sentinel quantisation; the cluster is excised in
    `from_grdecl` and its P-entries become -1.
    """
    nx, ny, nz = actnum_3d.shape
    keep = np.zeros((2 * nx, 2 * ny, 2 * nz), dtype=bool)
    for di, dj, dk in itertools.product((0, 1), repeat=3):
        keep[di::2, dj::2, dk::2] = actnum_3d
    out = xyz.copy()
    out[~keep] = np.nan
    return out


def from_grdecl(g: Grdecl, pinch_tol: float = 0.0) -> CornerCube:
    """Build the unique-node corner cube from a parsed `Grdecl`.

    Cells whose vertical thickness is ≤ ``pinch_tol`` on all pillars are
    pinched out (collapsed) and dropped from the active set — see
    `_pinched_cell_mask`. ``pinch_tol=0.0`` (default) drops only
    exactly-collapsed cells, leaving ordinary decks unchanged.
    """
    nx, ny, nz = g.cart_dims
    actnum_3d = _actnum_3d(g)
    # Pinched-out (zero-thickness) cells that were otherwise active. A
    # genuine ACTNUM=0 hole is NOT a pinch (so it is never NNC-bridged).
    pinched_3d = _pinched_cell_mask(g, tol=pinch_tol) & actnum_3d
    # Drop them, treating them as inactive (PR-11). The pinch mask is
    # carried on CornerCube so build_topology can bridge the surviving
    # cells above/below each pinch with an NNC (PR-12).
    actnum_3d = actnum_3d & ~pinched_3d

    xyz = _assemble_corner_xyz(g)
    masked = _mask_inactive_corners(xyz, actnum_3d)

    flat = masked.reshape(-1, 3)
    valid = ~np.isnan(flat).any(axis=1)

    quant = np.empty_like(flat, dtype=np.int64)
    if valid.any():
        quant[valid] = np.round(flat[valid] * _QUANT).astype(np.int64)
    if (~valid).any():
        quant[~valid] = np.iinfo(np.int64).min   # sentinel — won't collide with real coords

    _, first_idx, inverse = np.unique(
        quant, axis=0, return_index=True, return_inverse=True
    )

    P_flat = inverse.astype(np.int64).copy()
    keep_node = np.ones(len(first_idx), dtype=bool)
    if (~valid).any():
        sentinel_id = int(inverse[~valid][0])
        keep_node[sentinel_id] = False
        new_ids = np.full(len(first_idx), -1, dtype=np.int64)
        new_ids[keep_node] = np.arange(int(keep_node.sum()))
        P_flat = new_ids[P_flat]

    nodes = flat[first_idx[keep_node]]
    P = P_flat.reshape(2 * nx, 2 * ny, 2 * nz)

    return CornerCube(
        nodes=nodes,
        P=P,
        cart_dims=(nx, ny, nz),
        actnum_3d=actnum_3d,
        pinched_3d=pinched_3d,
    )
