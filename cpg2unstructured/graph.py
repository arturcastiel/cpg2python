"""Cell-cell adjacency + shortest-path.

 — pure-numpy reference path: ``cell_adjacency`` + ``shortest_path``.
 — optional scipy.sparse fast path under the ``[fast]`` extra:
        ``cell_adjacency_csgraph`` + ``shortest_path_fast``. Functions
        exist unconditionally; calls raise ``ImportError`` (with a
        message naming the ``[fast]`` extra) if scipy is not installed.
"""
from __future__ import annotations

import heapq
from typing import Literal

import numpy as np

from cpg2unstructured.unstructured import _TopoMesh

__all__ = [
    "cell_adjacency",
    "shortest_path",
    "cell_adjacency_csgraph",
    "shortest_path_fast",
]

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
                           Requires ``cell_centroids``.
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


# ──────────────────────────────────────────────────────────────────────
#  — optional scipy.sparse fast path
# ──────────────────────────────────────────────────────────────────────

try:
    import scipy.sparse as _sp
    import scipy.sparse.csgraph as _csgraph
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _require_scipy() -> None:
    if not _HAS_SCIPY:
        raise ImportError(
            "scipy is required for the [fast] code path. "
            "Install with: pip install cpg2unstructured[fast]"
        )


def cell_adjacency_csgraph(
    mesh: _TopoMesh,
    weights: WeightMode = "unit",
    cell_centroids: np.ndarray | None = None,
):
    """Build the cell-cell adjacency as a ``scipy.sparse.csr_array``.

    Scipy-backed counterpart to :func:`cell_adjacency`. Returns a
    symmetric CSR with shape ``(n_active, n_active)``. ``nnz`` equals
    the slow-path directed-edge count (2 × n_inner_faces).
    """
    _require_scipy()
    if weights not in _ALLOWED_WEIGHTS:
        raise ValueError(
            f"unknown weights mode: {weights!r}; "
            f"expected one of {_ALLOWED_WEIGHTS}"
        )
    if weights == "distance" and cell_centroids is None:
        raise ValueError("weights='distance' requires cell_centroids")

    nb = mesh.faces.neighbours
    mask = (nb[:, 0] != -1) & (nb[:, 1] != -1)
    a = nb[mask, 0].astype(np.int64, copy=False)
    b = nb[mask, 1].astype(np.int64, copy=False)
    if weights == "distance":
        w = np.linalg.norm(cell_centroids[a] - cell_centroids[b], axis=1)
    else:
        w = np.ones(a.shape[0], dtype=np.float64)

    n = mesh.cell_faces.shape[0]
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.concatenate([w, w])
    return _sp.csr_array((data, (rows, cols)), shape=(n, n))


def shortest_path_fast(csr, src: int, dst: int | None = None):
    """Scipy-backed Dijkstra. Drop-in for :func:`shortest_path`.

    - ``dst is None``: returns ``(distances, predecessors)`` arrays
      from scipy's ``csgraph.dijkstra`` (full-source mode).
    - ``dst`` is an ``int``: returns ``(dist, path)`` matching
      :func:`shortest_path`'s signature, where ``path`` is a Python
      ``list[int]`` including both endpoints. ``src == dst`` returns
      ``(0.0, [src])``. Raises ``ValueError`` if ``dst`` is
      unreachable.
    """
    _require_scipy()
    src = int(src)
    distances, predecessors = _csgraph.dijkstra(
        csr, directed=False, indices=src,
        return_predecessors=True,
    )
    if dst is None:
        return distances, predecessors

    dst = int(dst)
    if dst == src:
        return 0.0, [src]
    if not np.isfinite(distances[dst]):
        raise ValueError(f"no path from {src} to {dst}")
    path = [dst]
    while path[-1] != src:
        path.append(int(predecessors[path[-1]]))
    path.reverse()
    return float(distances[dst]), path
