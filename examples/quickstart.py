"""End-to-end demo: parse → mesh → properties → graph → shortest path.

Runs in well under a second on a normal laptop. Self-contained — only
depends on the in-tree ``quickstart.grdecl`` fixture next to this script.

Usage::

    python examples/quickstart.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cpg2unstructured import (
    attach_props,
    build_topology,
    cell_adjacency,
    cell_geometry,
    face_geometry,
    from_grdecl,
    read_grdecl,
    shortest_path,
)

HERE = Path(__file__).parent


def main() -> None:
    deck = HERE / "quickstart.grdecl"
    g = read_grdecl(deck)
    print(f"deck       : {deck.name}")
    print(f"cart_dims  : {g.cart_dims}")
    print(f"n_cells    : {g.n_cells}   "
          f"(active={int(g.actnum.sum())})")
    print(f"props      : {sorted(g.props)}")

    grid = attach_props(build_topology(from_grdecl(g)), g)
    print()
    print(f"grid       : {type(grid).__name__}")
    print(f"  n_cells    : {grid.n_cells}   "
          f"({grid.n_cells} active rows)")
    print(f"  n_faces    : {grid.faces.nodes.shape[0]}")
    print(f"  n_nodes    : {grid.nodes.shape[0]}")
    print(f"  index_map  : shape {grid.index_map.linear_to_cell.shape}")

    fg = face_geometry(grid.nodes, grid.faces, grid.cell_corners)
    cg = cell_geometry(grid.nodes, grid.faces, grid.cell_faces, fg)
    print()
    print(f"geometry   : total volume = {cg.volumes.sum():.3f}")

    if "PORO" in grid:
        poro = grid["PORO"]
        print(f"  PORO       : mean={poro.mean():.3f}  "
              f"min={poro.min():.3f}  max={poro.max():.3f}")

    adj = cell_adjacency(grid, weights="distance",
                         cell_centroids=cg.centroids)
    src, dst = 0, grid.n_cells - 1
    dist, path = shortest_path(adj, src, dst)
    print()
    print(f"shortest   : cell {src} → cell {dst}")
    print(f"  distance   : {dist:.3f}")
    print(f"  hops       : {len(path) - 1}   "
          f"(path length {len(path)})")

    # If scipy is present, show the fast path agrees on the cost.
    try:
        from cpg2unstructured import (
            cell_adjacency_csgraph,
            shortest_path_fast,
        )
        csr = cell_adjacency_csgraph(grid, weights="distance",
                                     cell_centroids=cg.centroids)
        d_fast, _ = shortest_path_fast(csr, src, dst)
        print(f"  fast path  : {d_fast:.3f}  "
              f"(scipy CSR shape={csr.shape}, nnz={csr.nnz})")
        np.testing.assert_allclose(d_fast, dist, atol=1e-12)
    except ImportError:
        print("  fast path  : skipped — install scipy via "
              "'pip install -e .[fast]'")


if __name__ == "__main__":
    main()
