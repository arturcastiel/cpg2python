"""Binary EGRID reader.

Reads an Eclipse / OPM EGRID (or .FEGRID) corner-point grid into the
format-agnostic :class:`~cpg2unstructured.corner_point.CornerPointGrid`,
so every downstream consumer (``from_grdecl``, ``CornerCube``,
``build_mesh_graph_3d``, projection workflows) is unchanged.

Design notes
------------
- ``resfo`` is a HARD dependency (numpy + typing_extensions only).
- The reader streams ``resfo.lazy_read`` and **never reads the array of
  unwanted keywords** (LGR sections, NNC blocks, FILEHEAD, GRIDUNIT, ...).
  This makes the runtime O(WANT bytes) rather than O(file bytes).
- First occurrence wins for any wanted keyword: that's the root grid.
  LGR sub-grids are not exposed by this reader (deferred phase).
- COORD / ZCORN keep their native dtype (default ``np.float32``) — caller
  widens explicitly if downstream needs ``float64``.
- MAPAXES is applied to COORD x/y by default so coordinates land in the
  field's map (UTM) frame. Pass ``apply_mapaxes=False`` to keep raw
  pillar coords — a :class:`RuntimeWarning` is emitted in that case if
  MAPAXES is actually present, so silent-bad-output is impossible.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import resfo

from cpg2unstructured._errors import (
    GrdeclConsistencyError,
    GrdeclSyntaxError,
)
from cpg2unstructured.corner_point import CornerPointGrid

__all__ = ["read_egrid"]


_REQUIRED = frozenset({"GRIDHEAD", "COORD", "ZCORN"})
_OPTIONAL = frozenset({"ACTNUM", "MAPAXES"})
_WANT = _REQUIRED | _OPTIONAL


def read_egrid(
    path: str | Path,
    *,
    apply_mapaxes: bool = True,
    dtype: np.dtype | type = np.float32,
) -> CornerPointGrid:
    """Parse a binary EGRID (or .FEGRID) file into a ``CornerPointGrid``.

    Parameters
    ----------
    path
        Path to the ``.EGRID`` (binary) or ``.FEGRID`` (formatted text)
        file. ``resfo`` auto-detects the format.
    apply_mapaxes
        When MAPAXES is present in the file, rotate COORD pillar (x, y)
        into the map frame defined by MAPAXES. Default ``True`` — this
        matches what downstream consumers (fault projection, viz) expect
        when fed real field data in UTM. Pass ``False`` to keep raw
        pillar coords; a :class:`RuntimeWarning` is emitted in that case
        to make the choice explicit.
    dtype
        Element dtype for COORD and ZCORN. Default ``np.float32`` matches
        the on-disk EGRID representation; pass ``np.float64`` to widen.
        ACTNUM is always ``int8``; MAPAXES is always ``float64`` (it's
        small and used for transforms).

    Returns
    -------
    CornerPointGrid
        ``source="egrid"``. ``mapaxes`` is the raw 6-number record (or
        ``None``); ``mapaxes_applied`` reflects whether the rotation was
        actually applied to ``coord``.

    Raises
    ------
    GrdeclSyntaxError
        Required keyword (GRIDHEAD / COORD / ZCORN) missing.
    GrdeclConsistencyError
        Array sizes inconsistent with GRIDHEAD dimensions, or grid type
        is not corner-point (``GRIDHEAD[0] != 1``).
    """
    p = str(path)
    kws: dict[str, np.ndarray] = {}

    # Stream entries; only materialise arrays whose keyword we want.
    # On ENDGRID we stop — anything after that belongs to LGR / NNC.
    for entry in resfo.lazy_read(p):
        name = entry.read_keyword().strip()
        if name == "ENDGRID":
            break
        if name in _WANT and name not in kws:
            kws[name] = np.asarray(entry.read_array())
            if _WANT.issubset(kws):
                break  # all wanted keywords in hand

    if "GRIDHEAD" not in kws:
        raise GrdeclSyntaxError(f"{p}: EGRID missing GRIDHEAD keyword")
    head = kws["GRIDHEAD"].astype(np.int64, copy=False)
    if head.size < 4 or int(head[0]) != 1:
        raise GrdeclConsistencyError(
            f"{p}: only corner-point grids supported (GRIDHEAD[0]==1); "
            f"got grid_type={int(head[0]) if head.size else 'EMPTY'}"
        )
    nx, ny, nz = int(head[1]), int(head[2]), int(head[3])

    if "COORD" not in kws or "ZCORN" not in kws:
        raise GrdeclSyntaxError(f"{p}: EGRID missing COORD or ZCORN")

    coord = kws["COORD"].astype(dtype, copy=False).reshape(-1, 6)
    n_pillars = (nx + 1) * (ny + 1)
    if coord.shape[0] != n_pillars:
        raise GrdeclConsistencyError(
            f"{p}: COORD has {coord.shape[0]} pillars, expected {n_pillars}"
        )
    zcorn = kws["ZCORN"].astype(dtype, copy=False).ravel()
    if zcorn.size != 8 * nx * ny * nz:
        raise GrdeclConsistencyError(
            f"{p}: ZCORN size {zcorn.size}, expected {8 * nx * ny * nz}"
        )

    if "ACTNUM" in kws:
        actnum = kws["ACTNUM"].astype(np.int8, copy=False).ravel()
        if actnum.size != nx * ny * nz:
            raise GrdeclConsistencyError(
                f"{p}: ACTNUM size {actnum.size}, expected {nx * ny * nz}"
            )
    else:
        actnum = np.ones(nx * ny * nz, dtype=np.int8)

    mapaxes: np.ndarray | None = None
    mapaxes_applied = False
    if "MAPAXES" in kws:
        mapaxes = kws["MAPAXES"].astype(np.float64, copy=False).ravel()
        if mapaxes.size != 6:
            raise GrdeclConsistencyError(
                f"{p}: MAPAXES has {mapaxes.size} entries, expected 6"
            )
        if apply_mapaxes:
            coord = _apply_mapaxes(coord, mapaxes, dtype)
            mapaxes_applied = True
        else:
            warnings.warn(
                f"{p}: MAPAXES present but apply_mapaxes=False; "
                "pillar coordinates are in raw (unrotated) grid frame.",
                RuntimeWarning,
                stacklevel=2,
            )

    return CornerPointGrid(
        cart_dims=(nx, ny, nz),
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        mapaxes=mapaxes,
        mapaxes_applied=mapaxes_applied,
        props={},
        unrecognized=[],
        source="egrid",
    )


def _apply_mapaxes(
    coord: np.ndarray,
    mapaxes: np.ndarray,
    dtype: np.dtype | type,
) -> np.ndarray:
    """Apply Eclipse MAPAXES rotation to the (x, y) of each pillar endpoint.

    MAPAXES is a 6-number record:
        ``(y_end_x, y_end_y, origin_x, origin_y, x_end_x, x_end_y)``
    interpreted as: the +Y axis of the LOCAL grid frame points from
    ``origin`` to ``(y_end_x, y_end_y)`` in MAP coords, and the +X axis
    points from ``origin`` to ``(x_end_x, x_end_y)``. The vectors are
    unit-normalised here.

    The transform takes LOCAL (x, y) to MAP (x, y):

        map_xy = origin + local_x * x_hat + local_y * y_hat

    In matrix form ``R = [x_hat | y_hat]`` (columns), and for a batch of
    row-vectors ``local_xy`` (shape ``(N, 2)``):

        map_xy = local_xy @ R.T + origin

    COORD layout per pillar: ``(xt, yt, zt, xb, yb, zb)`` — slices
    ``[0:2]`` and ``[3:5]`` are the top and bottom xy.
    """
    yax = np.array(
        [mapaxes[0] - mapaxes[2], mapaxes[1] - mapaxes[3]],
        dtype=np.float64,
    )
    xax = np.array(
        [mapaxes[4] - mapaxes[2], mapaxes[5] - mapaxes[3]],
        dtype=np.float64,
    )
    yn = np.linalg.norm(yax)
    xn = np.linalg.norm(xax)
    if yn == 0 or xn == 0:
        raise GrdeclConsistencyError(
            f"MAPAXES defines a degenerate axis (|x|={xn}, |y|={yn})"
        )
    yax /= yn
    xax /= xn
    origin = np.array([mapaxes[2], mapaxes[3]], dtype=np.float64)
    R = np.column_stack([xax, yax])  # (2, 2)

    out = coord.astype(np.float64, copy=True)
    for sl in (slice(0, 2), slice(3, 5)):  # top xy, bottom xy
        xy = out[:, sl]
        out[:, sl] = xy @ R.T + origin
    return out.astype(dtype, copy=False)
