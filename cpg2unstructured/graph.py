"""Cell-cell adjacency + shortest-path (pure numpy default).

PR-11 will add an optional scipy CSR + dijkstra path under the [fast]
extra; this module is the always-available reference.
"""
from __future__ import annotations

import heapq
from typing import Literal

import numpy as np

from cpg2unstructured.unstructured import _TopoMesh

__all__ = ["cell_adjacency", "shortest_path"]

WeightMode = Literal["unit", "distance", "none"]
_ALLOWED_WEIGHTS = ("unit", "distance", "none")


def cell_adjacency(
    mesh: _TopoMesh,
    weights: WeightMode = "unit",
    cell_centroids: np.ndarray | None = None,
) -> list[list[tuple[int, float]]]:
    """Build the cell-cell adjacency list.

    Parameters
    ----------
    mesh
        ``_TopoMesh`` from ``build_topology``.
    weights
        - ``"unit"``     : weight = 1.0 on every edge.
        - ``"distance"`` : weight = Euclidean ‖centroid_i − centroid_j‖₂.
                           Requires ``cell_centroids`` (PR-8 output).
        - ``"none"``     : same numerical behaviour as ``"unit"`` but
                           documented as "no weighting" so user code
                           reads more clearly.
    cell_centroids
        Required when ``weights="distance"``. Shape ``(n_active, 3)``.

    Returns
    -------
    A list of length ``n_active``; entry ``c`` is a list of
    ``(neighbour_id, weight)`` tuples for cell ``c``. Adjacency is
    symmetric (every inner face produces two directed entries).
    """
    if weights not in _ALLOWED_WEIGHTS:
        raise ValueError(
            f"unknown weights mode: {weights!r}; "
            f"expected one of {_ALLOWED_WEIGHTS}"
        )
    if weights == "distance" and cell_centroids is None:
        raise ValueError("weights='distance' requires cell_centroids")

    nb = mesh.faces.neighbours
    inner_mask = (nb[:, 0] != -1) & (nb[:, 1] != -1)
    inner = nb[inner_mask]
    n = mesh.cell_faces.shape[0]
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]

    for a, b in inner:
        ia, ib = int(a), int(b)
        if weights == "distance":
            w = float(np.linalg.norm(cell_centroids[ia] - cell_centroids[ib]))
        else:
            w = 1.0
        adj[ia].append((ib, w))
        adj[ib].append((ia, w))

    return adj


def shortest_path(
    adj: list[list[tuple[int, float]]],
    src: int,
    dst: int,
) -> tuple[float, list[int]]:
    """Dijkstra's algorithm (heapq + ndarray frontier).

    Returns ``(total_weight, path_as_list_of_cell_ids)`` where the path
    includes both ``src`` and ``dst``. Raises ``ValueError`` if ``dst``
    is unreachable from ``src``.
    """
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    dist[src] = 0.0
    heap: list[tuple[float, int]] = [(0.0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if not np.isfinite(dist[dst]):
        raise ValueError(f"no path from {src} to {dst}")

    path = [dst]
    while path[-1] != src:
        path.append(int(prev[path[-1]]))
    path.reverse()
    return float(dist[dst]), path
