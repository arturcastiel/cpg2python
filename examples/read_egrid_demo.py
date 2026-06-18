"""Minimal demo: read a binary EGRID and summarise the parsed grid.

Run:
    python examples/read_egrid_demo.py tests/data/cart_2x2x2.EGRID
    python examples/read_egrid_demo.py tests/data/cart_2x2x2_rotated.EGRID
    python examples/read_egrid_demo.py tests/data/cart_2x2x2_rotated.EGRID --no-mapaxes

No projection / no plotting — that's covered by ``fault-projections``'
``examples/quad_mesh_fault_egrid.py``.
"""
from __future__ import annotations

import argparse

import numpy as np

from cpg2unstructured import from_grdecl, read_egrid


def main() -> None:
    p = argparse.ArgumentParser(description="Demo: read an EGRID file.")
    p.add_argument("path", help="path to a .EGRID (binary) or .FEGRID (formatted) file")
    p.add_argument(
        "--no-mapaxes",
        action="store_true",
        help="skip MAPAXES rotation; keep raw LOCAL pillar coords (warns if MAPAXES present)",
    )
    p.add_argument(
        "--float64",
        action="store_true",
        help="widen COORD/ZCORN to float64 (default: native float32)",
    )
    args = p.parse_args()

    dtype = np.float64 if args.float64 else np.float32
    grid = read_egrid(args.path, apply_mapaxes=not args.no_mapaxes, dtype=dtype)
    nx, ny, nz = grid.cart_dims
    print(f"file       : {args.path}")
    print(f"dims       : {nx} x {ny} x {nz}   ({grid.n_cells} cells, {grid.n_pillars} pillars)")
    print(f"coord      : {grid.coord.shape}   dtype={grid.coord.dtype}")
    print(f"zcorn      : {grid.zcorn.shape}   dtype={grid.zcorn.dtype}")
    print(f"actnum     : {int(grid.actnum.sum())} / {grid.actnum.size} active")
    if grid.mapaxes is None:
        print("mapaxes    : (none)")
    else:
        print(f"mapaxes    : {grid.mapaxes.tolist()}")
        print(f"             applied: {grid.mapaxes_applied}")
    print(f"source     : {grid.source}")

    # Build the unique-node corner cube so the user sees the downstream
    # hand-off is exactly the same as for read_grdecl.
    cube = from_grdecl(grid)
    print(
        f"cornercube : n_nodes={cube.nodes.shape[0]}   "
        f"actnum_3d shape={cube.actnum_3d.shape}"
    )


if __name__ == "__main__":
    main()
