"""The pillar lattice, read straight from COORD.

A corner-point grid's pillars form a regular ``(NX+1) x (NY+1)`` lattice by
construction, independent of how the cells above them are shaped or which are
active. That makes the lattice the natural domain for anything that moves grid
geometry laterally: neighbours are ``(i±1, j)`` and ``(i, j±1)``, so a Laplacian
is the ordinary 5-point stencil and no adjacency structure has to be built.

**This module deliberately does not go through** :mod:`cpg2unstructured.planview`.
``projected_topology`` collapses 3D nodes by coordinate equality, which silently
stops collapsing as soon as pillars are slanted — on a 100 m cell, 5 m of tilt is
enough for every node in every layer to become its own "projected" node, with no
error raised. Real decks are overwhelmingly slanted (Norne: 7 vertical pillars out
of 5,311), so reading ``coord`` directly is both simpler and correct.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpg2unstructured.corner_point import CornerPointGrid

__all__ = [
    "PillarLattice",
    "pillar_lattice",
    "active_columns",
    "boundary_mask",
    "local_spacing",
    "apply_displacement",
]


@dataclass(frozen=True)
class PillarLattice:
    """Pillar endpoints on the ``(NX+1, NY+1)`` lattice.

    Attributes
    ----------
    top, bot
        ``(NX+1, NY+1, 2)`` float64 — the ``(x, y)`` of each pillar's top and
        bottom endpoint. ``z`` is carried separately in ``z_top`` / ``z_bot``
        because lateral motion never changes it.
    z_top, z_bot
        ``(NX+1, NY+1)`` float64.
    cart_dims
        ``(NX, NY, NZ)`` of the grid this came from.
    """

    top: np.ndarray
    bot: np.ndarray
    z_top: np.ndarray
    z_bot: np.ndarray
    cart_dims: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int]:
        nx, ny, _ = self.cart_dims
        return nx + 1, ny + 1

    @property
    def n_pillars(self) -> int:
        nx, ny = self.shape
        return nx * ny

    def is_vertical(self, atol: float = 1e-6) -> np.ndarray:
        """``(NX+1, NY+1)`` bool — pillars whose top and bottom ``(x, y)`` coincide.

        An ABSOLUTE tolerance, deliberately: ``np.isclose`` would apply its default
        ``rtol=1e-5`` on top, and field coordinates are UTM-sized. On Norne
        (~450,000 m easting) that makes the effective tolerance ~4.5 m, which counts
        a pillar leaning by metres as vertical — it reported 469 of 5,311 vertical
        where the true count is 7.
        """
        return np.all(np.abs(self.top - self.bot) <= atol, axis=-1)


def _coord_pillars(grid: CornerPointGrid) -> np.ndarray:
    """``coord`` reshaped to ``(NX+1, NY+1, 6)``.

    Eclipse stores COORD with the I index fastest, so the natural reshape is
    ``(NY+1, NX+1, 6)``; transposing the leading axes gives ``[i, j]`` indexing.
    Mirrors ``cpg._coord_pillars`` — kept local so this module does not depend on
    the corner-cube assembly path.
    """
    nx, ny, _ = grid.cart_dims
    coord = np.asarray(grid.coord, dtype=np.float64).reshape(ny + 1, nx + 1, 6)
    return coord.transpose(1, 0, 2).copy()


def pillar_lattice(grid: CornerPointGrid) -> PillarLattice:
    """Read the pillar lattice out of ``grid.coord``."""
    pp = _coord_pillars(grid)
    return PillarLattice(
        top=pp[..., 0:2].copy(),
        bot=pp[..., 3:5].copy(),
        z_top=pp[..., 2].copy(),
        z_bot=pp[..., 5].copy(),
        cart_dims=tuple(int(v) for v in grid.cart_dims),  # type: ignore[arg-type]
    )


def active_columns(grid: CornerPointGrid) -> np.ndarray:
    """``(NX, NY)`` bool — columns holding at least one active cell.

    Public because validity needs it: a check that counts folds in columns holding
    no cells is checking absent geometry. See ``check_validity(active=...)``.
    """
    nx, ny, nz = grid.cart_dims
    actnum = np.asarray(grid.actnum).reshape(nz, ny, nx).transpose(2, 1, 0)
    return actnum.astype(bool).any(axis=2)


_active_columns = active_columns   # pre-existing private name, kept working


def boundary_mask(grid: CornerPointGrid, scope: str = "active") -> np.ndarray:
    """``(NX+1, NY+1)`` bool — pillars on the boundary of the domain.

    Parameters
    ----------
    scope
        ``"active"`` (default) — the rim of the ACTIVE domain: a pillar touching at
        least one active column and at least one inactive-or-absent one. This is the
        reservoir outline, and on a real deck it is nothing like the bounding box
        (Norne: 2,264 of 5,152 columns active, so the outline is deeply irregular).

        ``"lattice"`` — the bounding-box edge, ``i in {0, NX}`` or ``j in {0, NY}``.
    """
    nx, ny, _ = grid.cart_dims
    if scope == "lattice":
        mask = np.zeros((nx + 1, ny + 1), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        return mask
    if scope != "active":
        raise ValueError(f"unknown scope: {scope!r}; expected 'active' or 'lattice'")

    active = _active_columns(grid)
    # Each pillar (i, j) touches up to four columns: (i-1..i, j-1..j). Pad with
    # False so lattice-edge pillars see "absent" outside the grid and therefore
    # count as boundary whenever they touch anything active.
    padded = np.zeros((nx + 2, ny + 2), dtype=bool)
    padded[1:-1, 1:-1] = active
    touch = np.stack(
        [padded[:-1, :-1], padded[1:, :-1], padded[:-1, 1:], padded[1:, 1:]], axis=-1
    )
    return touch.any(axis=-1) & ~touch.all(axis=-1)


def local_spacing(lattice: PillarLattice, *, use: str = "top") -> np.ndarray:
    """``(NX+1, NY+1)`` float64 — mean distance to existing lattice neighbours.

    The natural length scale at each pillar. Used to make displacements
    dimensionless, and to express tolerances as a fraction of cell size rather than
    in model units, so one default transfers between decks.
    """
    xy = lattice.top if use == "top" else lattice.bot
    total = np.zeros(xy.shape[:2], dtype=np.float64)
    count = np.zeros(xy.shape[:2], dtype=np.float64)
    for sl_a, sl_b in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        d = np.linalg.norm(xy[sl_b] - xy[sl_a], axis=-1)
        total[sl_a] += d
        total[sl_b] += d
        count[sl_a] += 1
        count[sl_b] += 1
    return total / np.maximum(count, 1.0)


def apply_displacement(
    grid: CornerPointGrid,
    d_top: np.ndarray,
    d_bot: np.ndarray | None = None,
) -> CornerPointGrid:
    """Return a copy of ``grid`` with pillar ``(x, y)`` displaced.

    ``d_top`` is ``(NX+1, NY+1, 2)``. ``d_bot`` defaults to ``d_top`` — that is
    stage-1 rigid translation, which preserves whatever slant a pillar already has.
    Passing a different ``d_bot`` tilts the pillar (stage 2). ``ZCORN`` is untouched
    in both cases, so layering is preserved exactly.
    """
    nx, ny, _ = grid.cart_dims
    if d_bot is None:
        d_bot = d_top
    for name, d in (("d_top", d_top), ("d_bot", d_bot)):
        if d.shape != (nx + 1, ny + 1, 2):
            raise ValueError(f"{name} must be {(nx + 1, ny + 1, 2)}; got {d.shape}")

    pp = _coord_pillars(grid)
    pp[..., 0:2] += d_top
    pp[..., 3:5] += d_bot
    coord = pp.transpose(1, 0, 2).reshape(-1, 6)

    return CornerPointGrid(
        cart_dims=grid.cart_dims,
        coord=coord,
        zcorn=grid.zcorn,
        actnum=grid.actnum,
        mapaxes=grid.mapaxes,
        mapaxes_applied=grid.mapaxes_applied,
        props=dict(grid.props),
        unrecognized=list(grid.unrecognized),
        source=grid.source,
    )
