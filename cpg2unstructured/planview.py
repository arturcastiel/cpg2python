"""Projected (2.5D) plan-view topology.

Collapses a 3D :class:`UnstructuredGrid` onto a chosen coordinate plane
("xy", "xz", or "yz") and returns the in-plane topology: unique projected
nodes, the lateral faces as node-pair edges (which double as graph edges),
each edge's adjacent cell structured indices, and a per-edge axis label.

This is pure topology (numpy only) — no graph object and no spatial index.
Those are cheap instantiations a consumer builds from this structure
(``edges`` + ``edge_len`` → a weighted graph; ``nodes2d`` → a KD-tree).

Two ways to decide which 3D nodes are "the same" in plan view:

- **coordinate collapse** (default) — nodes sharing in-plane coordinates. Correct
  only for vertically-aligned pillars. A tilted pillar's corners have different
  ``(x, y)`` at every layer, so nothing collapses and the result is silently NOT a
  plan view: measured on a 6x6x4 grid of 100 m cells, 5 m of tilt already produced
  245 projected nodes where 49 were expected, with no error raised.
- **pillar-identity collapse** — pass ``pillar_of_node`` (see
  :func:`cpg2unstructured.cpg.pillar_of_node`) and nodes are grouped by the pillar
  they belong to, whatever their coordinates. Correct for tilted and faulted
  pillars, and the right choice for any real deck.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpg2unstructured.unstructured import UnstructuredGrid

__all__ = ["ProjectedTopology", "projected_topology", "PLANE_AXES"]

# plane -> the two coordinate / ijk axes kept in-plane (the third is collapsed).
# x<->i (0), y<->j (1), z<->k (2). The kept coord axes equal the kept ijk axes.
PLANE_AXES: dict[str, tuple[int, int]] = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


@dataclass(frozen=True)
class ProjectedTopology:
    """In-plane topology of a grid projected onto a coordinate plane.

    Attributes
    ----------
    plane
        "xy", "xz", or "yz".
    nodes2d
        (Np, 2) float64 — unique in-plane node coordinates (the two kept
        axes), after collapsing 3D nodes along the dropped axis.
    edges
        (Ne, 2) int64 — projected-node-id pairs (sorted ascending per row).
        Each edge is one lateral face viewed in-plane; it is both a graph
        edge and a fault-face candidate. Duplicate faces stacked along the
        collapsed axis are merged into a single edge.
    edge_len
        (Ne,) float64 — in-plane Euclidean length of each edge (graph weight).
    edge_uv
        (Ne, 2, 2) int64 — the two in-plane structured indices (u, v) of the
        cells adjacent to each edge; a boundary side is (-1, -1). For
        plane="xy" these are (I, J).
    edge_dir
        (Ne,) int8 — 0 if the edge's face is normal to the first in-plane
        axis, 1 if normal to the second. For plane="xy": 0 = 'X' face
        (cells differ in I), 1 = 'Y' face (cells differ in J).
    node_of_grid_node
        (Nnodes,) int64 — maps each 3D ``grid.nodes`` id to its projected
        node id in ``nodes2d``.
    """

    plane: str
    nodes2d: np.ndarray
    edges: np.ndarray
    edge_len: np.ndarray
    edge_uv: np.ndarray
    edge_dir: np.ndarray
    node_of_grid_node: np.ndarray


def projected_topology(
    grid: UnstructuredGrid,
    plane: str = "xy",
    *,
    decimals: int | None = None,
    pillar_of_node: np.ndarray | None = None,
) -> ProjectedTopology:
    """Project ``grid`` onto ``plane`` and return its in-plane topology.

    Parameters
    ----------
    grid
        Unstructured grid from :func:`build_topology`.
    plane
        "xy" (default, plan view), "xz", or "yz".
    decimals
        If given, in-plane coordinates are rounded to this many decimals
        before deduplicating nodes — useful when vertically-aligned pillars
        carry tiny floating-point drift. ``None`` (default) deduplicates on
        exact coordinates.

    Returns
    -------
    ProjectedTopology
    """
    try:
        u_ax, v_ax = PLANE_AXES[plane]
    except KeyError:
        raise ValueError(
            f"unknown plane: {plane!r}; expected one of {tuple(PLANE_AXES)}"
        ) from None

    # 1. Projected nodes — collapse the 3D nodes of each column into one.
    coords2d = grid.nodes[:, (u_ax, v_ax)]
    if pillar_of_node is not None:
        # IDENTITY collapse: nodes sharing a pillar are one projected node,
        # whatever their coordinates. Correct for tilted pillars, where the
        # coordinate collapse below silently stops collapsing anything.
        pon = np.asarray(pillar_of_node, dtype=np.int64).ravel()
        if pon.shape[0] != grid.nodes.shape[0]:
            raise ValueError(
                f"pillar_of_node must have one entry per node "
                f"({grid.nodes.shape[0]}); got {pon.shape[0]}"
            )
        # -1 = "no pillar owns this node": an unvisited corner-cube slot, or a
        # fault intersection node appended by build_topology, which lies on no
        # pillar by construction. np.unique would weld every such node into ONE
        # projected node — a grid-wide false weld. Give each its own key so it
        # keeps a distinct identity.
        free = pon < 0
        if free.any():
            pon = pon.copy()
            pon[free] = (
                int(pon.max(initial=-1)) + 1 + np.arange(int(free.sum()))
            )
        uniq, inverse = np.unique(pon, return_inverse=True)
        node_of_grid_node = inverse.astype(np.int64, copy=False).ravel()
        # representative position: the mean of the column's nodes
        nodes2d = np.zeros((uniq.shape[0], 2), dtype=np.float64)
        counts = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(nodes2d, node_of_grid_node, coords2d)
        np.add.at(counts, node_of_grid_node, 1.0)
        nodes2d /= np.maximum(counts, 1.0)[:, None]
    else:
        key = coords2d if decimals is None else np.round(coords2d, decimals)
        # Indexed rather than tuple-unpacked: mypy 2.3.0 crashes with an INTERNAL
        # ERROR resolving the np.unique overload for `a, b = np.unique(..., axis=0,
        # return_inverse=True)` against the numpy stubs CI installs. Local runs pass
        # on the same mypy with an older numpy, so pinning either would only hide it.
        collapsed = np.unique(key, axis=0, return_inverse=True)
        nodes2d = np.asarray(collapsed[0], dtype=np.float64)
        node_of_grid_node = np.asarray(collapsed[1]).astype(np.int64, copy=False).ravel()

    # 2. Select the lateral faces that lie in-plane (normal to a kept axis).
    fdir = grid.faces.direction
    sel = (fdir == u_ax) | (fdir == v_ax)
    fnodes = grid.faces.nodes[sel]
    fnb = grid.faces.neighbours[sel]
    fsel_dir = fdir[sel]

    # 3. Each face's 4 corners collapse to exactly 2 projected nodes.
    pv = node_of_grid_node[fnodes]
    pv_sorted = np.sort(pv, axis=1)
    a = pv_sorted[:, 0]
    b = pv_sorted[:, -1]
    nondegenerate = a != b
    a, b = a[nondegenerate], b[nondegenerate]
    fnb = fnb[nondegenerate]
    fsel_dir = fsel_dir[nondegenerate]
    pair = np.stack([a, b], axis=1)

    # 4. Merge faces stacked along the collapsed axis into unique edges.
    edges, idx_first = np.unique(pair, axis=0, return_index=True)
    edge_len = np.linalg.norm(
        nodes2d[edges[:, 0]] - nodes2d[edges[:, 1]], axis=1
    )
    edge_dir = (fsel_dir[idx_first] == v_ax).astype(np.int8)

    # 5. In-plane structured indices of each edge's two adjacent cells.
    nb_rep = fnb[idx_first]
    edge_uv = np.full((nb_rep.shape[0], 2, 2), -1, dtype=np.int64)
    valid = nb_rep >= 0
    edge_uv[valid] = grid.original_ijk[nb_rep[valid]][:, (u_ax, v_ax)]

    return ProjectedTopology(
        plane=plane,
        nodes2d=nodes2d,
        edges=edges.astype(np.int64, copy=False),
        edge_len=edge_len.astype(np.float64, copy=False),
        edge_uv=edge_uv,
        edge_dir=edge_dir,
        node_of_grid_node=node_of_grid_node,
    )
