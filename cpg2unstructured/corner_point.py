"""Format-agnostic corner-point grid (read from GRDECL text or EGRID binary).

This module provides ``CornerPointGrid``, the single dataclass returned by
both ``read_grdecl`` and ``read_egrid``. The legacy name ``Grdecl`` is
preserved as a ``TypeAlias`` for back-compatibility and exposed from the
same import surface.

MAPAXES is a first-class field, NOT routed through ``props`` (which is the
PORO / PERMX / SATNUM cell-property bucket).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np

__all__ = ["CornerPointGrid", "Grdecl"]


@dataclass(frozen=True)
class CornerPointGrid:
    """Parsed corner-point grid (format-agnostic).

    Fields
    ------
    cart_dims
        ``(NX, NY, NZ)`` integer dimensions.
    coord
        ``(n_pillars, 6)`` array of pillar endpoints ``(xt, yt, zt, xb, yb, zb)``.
        Native dtype is whatever the reader produced (float32 for EGRID by
        default, float64 for GRDECL text decks).
    zcorn
        ``(8 * NX * NY * NZ,)`` flat array of corner Z values.
    actnum
        ``(NX * NY * NZ,)`` int8 active-cell mask.
    mapaxes
        ``(6,)`` raw MAPAXES record, or ``None`` if absent.
        Stored as float64 regardless of ``coord``'s dtype.
    mapaxes_applied
        ``True`` if MAPAXES rotation has already been applied to ``coord``.
        ``False`` means ``coord`` is in the raw / local grid frame.
    props
        Cell-property bucket: PORO / PERMX / PERMY / PERMZ / NTG / MULT* /
        PRESSURE / SWAT / SGAS / SOIL / SATNUM / PVTNUM / EQLNUM / FIPNUM.
        MAPAXES is NOT stored here.
    unrecognized
        Names of GRDECL keywords seen but not consumed (text-deck reader only).
    source
        Which reader produced this grid: ``"grdecl"`` or ``"egrid"``.
    """

    cart_dims: tuple[int, int, int]
    coord: np.ndarray
    zcorn: np.ndarray
    actnum: np.ndarray
    mapaxes: np.ndarray | None = None
    mapaxes_applied: bool = False
    props: dict[str, np.ndarray] = field(default_factory=dict)
    unrecognized: list[str] = field(default_factory=list)
    source: Literal["grdecl", "egrid"] = "grdecl"

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.cart_dims
        return nx * ny * nz

    @property
    def n_pillars(self) -> int:
        nx, ny, _ = self.cart_dims
        return (nx + 1) * (ny + 1)


# Back-compat alias. ``Grdecl`` was the previous name when only the text
# GRDECL reader existed; it now refers to the same dataclass. Existing code
# importing ``from cpg2unstructured import Grdecl`` continues to work
# unchanged. Do NOT remove without a major-version bump.
Grdecl: TypeAlias = CornerPointGrid
