"""Toy 2×2×2 cartesian CPG → unstructured-mesh pipeline.

Internal smoke fixture used by tests and (later) by PR-3..PR-7 as a
regression target. NOT public API — leading underscore.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["build_toy_cpg", "build_toy_mesh", "ToyCPG", "ToyMesh"]


@dataclass(frozen=True)
class ToyCPG:
    """Synthetic Eclipse-style corner-point arrays for an NX×NY×NZ box."""

    nx: int
    ny: int
    nz: int
    coord: np.ndarray   # (NX+1, NY+1, 6)
    zcorn: np.ndarray   # (2*NX, 2*NY, 2*NZ)
    actnum: np.ndarray  # (NX, NY, NZ) of bool


@dataclass(frozen=True)
class ToyMesh:
    """Unstructured mesh derived from a ToyCPG."""

    nodes: np.ndarray         # (Nnodes, 3) float64 — unique XYZ
    cell_nodes: np.ndarray    # (Ncells, 8) int64 — Eclipse-ordered corners per cell
    faces: np.ndarray         # (Nfaces, 4) int64 — sorted node-ids per face
    cell_faces: np.ndarray    # (Ncells, 6) int64 — i-, i+, j-, j+, k-, k+ face ids
    face_cells: np.ndarray    # (Nfaces, 2) int64 — adjacent cell ids; -1 if boundary
    neighbours: np.ndarray    # (Npairs, 2) int64 — sorted (a, b) cell-pairs across inner faces


def build_toy_cpg(nx: int = 2, ny: int = 2, nz: int = 2,
                  dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> ToyCPG:
    """Build the synthetic CPG arrays for a regular cartesian box.

    All pillars are perfectly vertical, all ZCORN values are shared across
    neighbouring cells (matching-Z, no faults). ACTNUM all-true.
    """
    px = np.arange(nx + 1, dtype=np.float64) * dx
    py = np.arange(ny + 1, dtype=np.float64) * dy
    XX, YY = np.meshgrid(px, py, indexing="ij")
    z_top = 0.0
    z_bot = nz * dz
    coord = np.stack(
        [XX, YY, np.full_like(XX, z_top),
         XX, YY, np.full_like(XX, z_bot)],
        axis=-1,
    )

    zcorn = np.empty((2 * nx, 2 * ny, 2 * nz), dtype=np.float64)
    for k in range(nz):
        z_lo = k * dz
        z_hi = (k + 1) * dz
        zcorn[:, :, 2 * k]     = z_lo
        zcorn[:, :, 2 * k + 1] = z_hi

    actnum = np.ones((nx, ny, nz), dtype=bool)
    return ToyCPG(nx=nx, ny=ny, nz=nz, coord=coord, zcorn=zcorn, actnum=actnum)


_LOCAL_CORNERS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)

# i-, i+, j-, j+, k-, k+ in local-corner indices
_LOCAL_FACES = np.array([
    [0, 2, 6, 4],
    [1, 3, 7, 5],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 1, 3, 2],
    [4, 5, 7, 6],
], dtype=np.int64)

_QUANT = 1e9


def _cell_corner_xyz(cpg: ToyCPG) -> np.ndarray:
    """Expand (COORD, ZCORN) → (Ncells, 8, 3) float64.

    Cell ordering: i-fastest, then j, then k (Python-natural).
    Corner ordering (Eclipse): see _LOCAL_CORNERS above.
    """
    nx, ny, nz = cpg.nx, cpg.ny, cpg.nz
    out = np.empty((nx * ny * nz, 8, 3), dtype=np.float64)
    cid = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for c, (di, dj, dk) in enumerate(_LOCAL_CORNERS):
                    x, y = cpg.coord[i + di, j + dj, 0:2]   # cartesian: top == bot
                    z = cpg.zcorn[2 * i + di, 2 * j + dj, 2 * k + dk]
                    out[cid, c] = (x, y, z)
                cid += 1
    return out


def build_toy_mesh(cpg: ToyCPG) -> ToyMesh:
    """Build the full unstructured mesh from a ToyCPG (cartesian, matching-Z)."""
    corners = _cell_corner_xyz(cpg)
    flat = corners.reshape(-1, 3)

    quant = np.round(flat * _QUANT).astype(np.int64)
    _, inv = np.unique(quant, axis=0, return_inverse=True)
    cell_nodes = inv.reshape(-1, 8).astype(np.int64)

    n_unique = int(cell_nodes.max()) + 1
    first_idx = np.zeros(n_unique, dtype=np.int64)
    seen = np.zeros(n_unique, dtype=bool)
    for slot, nid in enumerate(inv):
        if not seen[nid]:
            first_idx[nid] = slot
            seen[nid] = True
    nodes = flat[first_idx]

    raw = cell_nodes[:, _LOCAL_FACES]                 # (Ncells, 6, 4)
    sorted_face = np.sort(raw.reshape(-1, 4), axis=1)
    faces, inv_f = np.unique(sorted_face, axis=0, return_inverse=True)
    cell_faces = inv_f.reshape(-1, 6).astype(np.int64)

    nfaces = faces.shape[0]
    face_cells = np.full((nfaces, 2), -1, dtype=np.int64)
    for cid in range(cell_faces.shape[0]):
        for fid in cell_faces[cid]:
            slot = 0 if face_cells[fid, 0] == -1 else 1
            face_cells[fid, slot] = cid

    inner = face_cells[(face_cells[:, 0] != -1) & (face_cells[:, 1] != -1)]
    pairs = np.sort(inner, axis=1)
    neighbours = np.unique(pairs, axis=0)

    return ToyMesh(
        nodes=nodes,
        cell_nodes=cell_nodes,
        faces=faces,
        cell_faces=cell_faces,
        face_cells=face_cells,
        neighbours=neighbours,
    )
