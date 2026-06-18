"""Regenerate the small EGRID fixtures used by ``tests/test_egrid_reader.py``.

Run once when adding / changing fixtures; commit the produced ``.EGRID``
files alongside this script. Tests must NOT call this at collection
time — fixtures are committed so tests run without a network/install
dance.

Usage
-----
    python tests/data/_make_fixtures.py

What it writes
--------------
- ``cart_2x2x2.EGRID``            — 2x2x2 Cartesian grid, no MAPAXES, ACTNUM=1.
- ``cart_2x2x2_rotated.EGRID``    — same grid + MAPAXES describing a 45 deg
                                    rotation of the LOCAL frame relative to MAP.
- ``cart_2x2x2_no_actnum.EGRID``  — same grid without an ACTNUM array
                                    (reader must default to all-active).
- ``streaming_padded.EGRID``      — same grid with a large dummy LGR-like
                                    block appended AFTER ENDGRID so the
                                    streaming short-circuit can be checked.

All numbers below are LOCAL grid coordinates (origin 0,0,0; cell size 1;
nz = 2 layers).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import resfo


def _cartesian_2x2x2() -> dict[str, np.ndarray]:
    """Build the keyword arrays for a 2x2x2 unit-cell Cartesian grid.

    Returns a dict {keyword (8-char padded): ndarray} ready for
    ``resfo.write``.
    """
    nx = ny = nz = 2
    n_pillars = (nx + 1) * (ny + 1)

    # COORD: 9 pillars; each is (xt, yt, zt, xb, yb, zb).
    # Top z = 0, bottom z = nz = 2.
    coord = np.zeros((n_pillars, 6), dtype=np.float32)
    idx = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord[idx] = (i, j, 0.0, i, j, float(nz))
            idx += 1
    coord_flat = coord.ravel()

    # ZCORN: 8 corners per cell, per-layer linear in z.
    # Convention: top-of-layer-k z = k; bottom-of-layer-k z = k+1.
    # Stored in i-j-k order, with the 8 corners of each cell tightly packed.
    zcorn = np.empty(8 * nx * ny * nz, dtype=np.float32)
    pos = 0
    # ZCORN ordering per Eclipse: for each k, two slabs (top corners of
    # each cell, then bottom corners), each slab in i-fastest j-stride.
    # Per cell: top-SW, top-SE, top-NW, top-NE, bot-SW, bot-SE, bot-NW, bot-NE.
    for k in range(nz):
        ztop = float(k)
        zbot = float(k + 1)
        for j in range(ny):
            for i in range(nx):
                zcorn[pos : pos + 4] = ztop
                zcorn[pos + 4 : pos + 8] = zbot
                pos += 8

    actnum = np.ones(nx * ny * nz, dtype=np.int32)

    # GRIDHEAD: [grid_type=1 (corner-point), nx, ny, nz, num_grids=1, ...]
    # Eclipse writes 100 ints; only the first 6-ish are inspected by us.
    gridhead = np.zeros(100, dtype=np.int32)
    gridhead[0] = 1   # corner-point
    gridhead[1] = nx
    gridhead[2] = ny
    gridhead[3] = nz
    gridhead[4] = 1   # only one grid (root)

    # FILEHEAD: 100 ints; version etc. Zeros are fine for parser-tolerance
    # tests of resfo.
    filehead = np.zeros(100, dtype=np.int32)
    filehead[1] = 2024  # version stamp (arbitrary; reader doesn't check)

    # ENDGRID: signals end of root grid block (zero-length array).
    endgrid = np.zeros(0, dtype=np.int32)

    return {
        "FILEHEAD": filehead,
        "GRIDHEAD": gridhead,
        "COORD   ": coord_flat,
        "ZCORN   ": zcorn,
        "ACTNUM  ": actnum,
        "ENDGRID ": endgrid,
    }


def _mapaxes_45deg() -> np.ndarray:
    """A MAPAXES record that rotates the LOCAL frame 45 deg CCW.

    Convention: (y_end_x, y_end_y, origin_x, origin_y, x_end_x, x_end_y).
    Origin at (10, 20) in MAP coords (so the LOCAL origin sits there).
    Local +X axis points to MAP (10+cos45, 20+sin45).
    Local +Y axis points to MAP (10-sin45, 20+cos45).
    """
    c = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)
    origin = (10.0, 20.0)
    x_end = (origin[0] + c, origin[1] + s)
    y_end = (origin[0] - s, origin[1] + c)
    return np.array(
        [y_end[0], y_end[1], origin[0], origin[1], x_end[0], x_end[1]],
        dtype=np.float32,
    )


def _ordered(entries: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Pad every keyword to 8 chars; resfo requires it."""
    return [(kw.ljust(8)[:8], arr) for kw, arr in entries]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _cartesian_2x2x2()

    # Standard fixture (no MAPAXES, ACTNUM present).
    standard = _ordered([
        ("FILEHEAD", base["FILEHEAD"]),
        ("GRIDHEAD", base["GRIDHEAD"]),
        ("COORD",    base["COORD   "]),
        ("ZCORN",    base["ZCORN   "]),
        ("ACTNUM",   base["ACTNUM  "]),
        ("ENDGRID",  base["ENDGRID "]),
    ])
    resfo.write(out_dir / "cart_2x2x2.EGRID", standard)

    # Rotated fixture (MAPAXES present, 45 deg, origin at (10, 20)).
    rotated = _ordered([
        ("FILEHEAD", base["FILEHEAD"]),
        ("MAPAXES",  _mapaxes_45deg()),
        ("GRIDHEAD", base["GRIDHEAD"]),
        ("COORD",    base["COORD   "]),
        ("ZCORN",    base["ZCORN   "]),
        ("ACTNUM",   base["ACTNUM  "]),
        ("ENDGRID",  base["ENDGRID "]),
    ])
    resfo.write(out_dir / "cart_2x2x2_rotated.EGRID", rotated)

    # No-ACTNUM fixture.
    no_actnum = _ordered([
        ("FILEHEAD", base["FILEHEAD"]),
        ("GRIDHEAD", base["GRIDHEAD"]),
        ("COORD",    base["COORD   "]),
        ("ZCORN",    base["ZCORN   "]),
        ("ENDGRID",  base["ENDGRID "]),
    ])
    resfo.write(out_dir / "cart_2x2x2_no_actnum.EGRID", no_actnum)

    # Padded fixture: root grid + a moderate dummy block AFTER ENDGRID so the
    # streaming short-circuit can be observed. The reader must stop at
    # ENDGRID and never touch the dummy block. Keep the dummy small enough
    # to commit (~40 KB).
    big_dummy = np.arange(10_000, dtype=np.float32)
    padded = _ordered([
        ("FILEHEAD", base["FILEHEAD"]),
        ("GRIDHEAD", base["GRIDHEAD"]),
        ("COORD",    base["COORD   "]),
        ("ZCORN",    base["ZCORN   "]),
        ("ACTNUM",   base["ACTNUM  "]),
        ("ENDGRID",  base["ENDGRID "]),
        ("LGRSGONE", big_dummy),  # not a real keyword, just bulk
        ("BIGDUMMY", big_dummy),
    ])
    resfo.write(out_dir / "streaming_padded.EGRID", padded)


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
    print("Fixtures regenerated.")
