"""ZCORN-only grid edits: corner addressing, shifts along pillars, validity.

A corner-point grid stores one depth per cell corner (ZCORN); the corner's ``(x, y)``
follows from where its pillar (COORD) crosses that depth. Changing ZCORN alone therefore
moves every corner *along its own pillar*: a vertical shift on a vertical pillar, a
shift along the dip on a tilted one. COORD never changes.

**Corner cube.** :func:`zcorn_cube` returns ZCORN as a ``(2*NX, 2*NY, 2*NZ)`` array.
Slot ``(2*i + u, 2*j + v, 2*k + w)`` is the corner of cell ``(i, j, k)`` on pillar
``(i + u, j + v)``, at the top (``w = 0``) or base (``w = 1``) of the cell — the same
layout as :class:`cpg2unstructured.cpg.CornerCube`. So the four slots ``(2i+u, 2j+v)``
of one column are that column's corners, and the up to four column corners that sit on
pillar ``(p, q)`` are slots ``(2p-1|2p, 2q-1|2q)``.

**Welds.** Two cells share a lateral face when their corners on both pillars of that
face are equal. :func:`unwelded_faces` lists the lateral faces where they are not, which
is how a fault (or an edit that tore the grid) shows up in ZCORN. This module knows
nothing about FAULTS: deciding which faces *may* open is the caller's business.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cpg2unstructured.corner_point import CornerPointGrid

__all__ = [
    "zcorn_cube",
    "with_zcorn",
    "apply_zcorn_displacement",
    "unwelded_faces",
    "ZcornValidity",
    "zcorn_validity",
]


def zcorn_cube(grid: CornerPointGrid) -> np.ndarray:
    """ZCORN as a ``(2*NX, 2*NY, 2*NZ)`` float64 copy, indexed ``[i-slot, j-slot, k-slot]``."""
    nx, ny, nz = grid.cart_dims
    cube = np.asarray(grid.zcorn, dtype=np.float64).reshape(2 * nz, 2 * ny, 2 * nx)
    return cube.transpose(2, 1, 0).copy()


def with_zcorn(grid: CornerPointGrid, cube: np.ndarray) -> CornerPointGrid:
    """A copy of ``grid`` whose ZCORN is ``cube`` (the :func:`zcorn_cube` layout).

    The new ZCORN is float64 whatever the input dtype, so that shifts are not rounded
    back to float32. COORD, ACTNUM and every property are shared with ``grid``.
    """
    nx, ny, nz = grid.cart_dims
    cube = np.asarray(cube, dtype=np.float64)
    if cube.shape != (2 * nx, 2 * ny, 2 * nz):
        raise ValueError(f"cube has shape {cube.shape}; expected {(2 * nx, 2 * ny, 2 * nz)}")
    return replace(grid, zcorn=np.ascontiguousarray(cube.transpose(2, 1, 0)).ravel())


def apply_zcorn_displacement(grid: CornerPointGrid, dz: object) -> CornerPointGrid:
    """Add a depth shift to ZCORN, moving each corner along its pillar.

    ``dz`` (positive = deeper) is one of:

    - a scalar: every corner moves by the same amount;
    - ``(2*NX, 2*NY)``: one shift per column corner, the same down the column, so every
      cell keeps its thickness (a rigid slide of each column corner);
    - ``(2*NX, 2*NY, NZ+1)``: one shift per column corner and *horizon*, where horizon
      ``h`` is the top of layer ``h`` and horizon ``NZ`` the base of the last layer.
      The base of layer ``k`` and the top of layer ``k+1`` both take horizon ``k+1``, so
      layers that touched still touch; unequal values thicken or thin a layer;
    - ``(2*NX, 2*NY, 2*NZ)``: one shift per ZCORN slot (the :func:`zcorn_cube` layout).

    The shift is not checked; pass the result to :func:`zcorn_validity`.
    """
    nx, ny, nz = grid.cart_dims
    cube = zcorn_cube(grid)
    shift = np.asarray(dz, dtype=np.float64)
    if shift.ndim == 0 or shift.shape == cube.shape:
        cube += shift
    elif shift.shape == (2 * nx, 2 * ny):
        cube += shift[:, :, None]
    elif shift.shape == (2 * nx, 2 * ny, nz + 1):
        slot_horizon = (np.arange(2 * nz) + 1) // 2
        cube += shift[:, :, slot_horizon]
    else:
        raise ValueError(
            f"dz has shape {shift.shape}; expected a scalar, {(2 * nx, 2 * ny)}, "
            f"{(2 * nx, 2 * ny, nz + 1)} or {cube.shape}"
        )
    return with_zcorn(grid, cube)


def unwelded_faces(
    grid: CornerPointGrid, tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lateral faces whose two cells no longer share their corners.

    Returns ``(open_i, open_j)``: ``open_i[i, j]`` (shape ``(NX-1, NY)``) is True when
    the face between columns ``(i, j)`` and ``(i+1, j)`` differs in depth by more than
    ``tol`` at any corner of any layer; ``open_j[i, j]`` (shape ``(NX, NY-1)``) likewise
    between ``(i, j)`` and ``(i, j+1)``. ACTNUM is ignored: this compares geometry.
    """
    nx, ny, _ = grid.cart_dims
    cube = zcorn_cube(grid)
    gap_i = np.abs(cube[2:-1:2] - cube[1:-1:2]) > tol         # (NX-1, 2*NY, 2*NZ)
    gap_j = np.abs(cube[:, 2:-1:2] - cube[:, 1:-1:2]) > tol   # (2*NX, NY-1, 2*NZ)
    open_i = gap_i.any(axis=2).reshape(nx - 1, ny, 2).any(axis=2)
    open_j = gap_j.any(axis=2).reshape(nx, 2, ny - 1).any(axis=1)
    return open_i, open_j


@dataclass(frozen=True)
class ZcornValidity:
    """Outcome of :func:`zcorn_validity`.

    Attributes
    ----------
    ok
        No inverted cell and no overlapping layers.
    inverted
        ``(NX, NY, NZ)`` bool — a corner of the cell has its base above its top by more
        than ``tol`` (negative thickness).
    overlapping
        ``(NX, NY, NZ-1)`` bool — on some corner of the column, the base of layer ``k``
        lies below the top of layer ``k+1`` by more than ``tol`` (the layers overlap).
        A gap (base above the next top) is allowed.
    min_thickness
        Smallest corner thickness found, in depth units.
    """

    ok: bool
    inverted: np.ndarray
    overlapping: np.ndarray
    n_inverted: int
    n_overlapping: int
    min_thickness: float

    def summary(self) -> str:
        if self.ok:
            return f"valid ZCORN (min corner thickness {self.min_thickness:.6g})"
        bits = []
        if self.n_inverted:
            bits.append(f"{self.n_inverted} inverted cell(s)")
        if self.n_overlapping:
            bits.append(f"{self.n_overlapping} overlapping layer pair(s)")
        return "INVALID ZCORN: " + ", ".join(bits)


def zcorn_validity(grid: CornerPointGrid, tol: float = 0.0) -> ZcornValidity:
    """Check that every corner keeps a non-negative thickness and layers do not overlap.

    These are the two ways a ZCORN edit can break cells along a pillar. Lateral
    consistency (which faces opened) is :func:`unwelded_faces`; plan-view folds of the
    pillar lattice are :func:`cpg2unstructured.validity.check_validity`.
    """
    nx, ny, nz = grid.cart_dims
    cube = zcorn_cube(grid)
    thick = cube[:, :, 1::2] - cube[:, :, 0::2]                    # (2NX, 2NY, NZ)
    inverted = (thick < -tol).reshape(nx, 2, ny, 2, nz).any(axis=(1, 3))
    lap = cube[:, :, 1:-1:2] - cube[:, :, 2::2]                    # base k - top k+1
    overlapping = (lap > tol).reshape(nx, 2, ny, 2, nz - 1).any(axis=(1, 3))
    n_inv, n_lap = int(inverted.sum()), int(overlapping.sum())
    return ZcornValidity(
        ok=n_inv == 0 and n_lap == 0,
        inverted=inverted,
        overlapping=overlapping,
        n_inverted=n_inv,
        n_overlapping=n_lap,
        min_thickness=float(thick.min()) if thick.size else 0.0,
    )
