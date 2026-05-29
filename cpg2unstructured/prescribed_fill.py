"""Prescribed-boundary fill: a cell-face surface with a GIVEN boundary loop.

Given a closed grid edge-loop ``gamma`` (a 1-cycle), find a 2-chain ``Phi`` (a set
of grid cell-faces) whose boundary is exactly ``gamma``: ``d2 Phi == gamma``. This
is the discrete prescribed-boundary minimal-surface problem and the core primitive
for projecting a quad fault mesh watertight (each mesh patch is filled to its
shared-skeleton boundary; the patches then tile with no seam).

Two routes, both exact (``d2 Phi == gamma``):

- :func:`fill_prescribed_boundary` — solve ``B @ phi = gamma`` over GF(2), where
  ``B`` is the face->edge incidence (the ``d2`` operator). A solution exists iff
  ``gamma`` bounds in the active-cell complex; if not (e.g. it encircles a
  pinch-out / inactive region) the system is inconsistent and we raise
  :class:`BoundaryDoesNotBound`. The bare solution is *some* surface with the right
  boundary — not necessarily plane-hugging.

- :func:`fill_with_seed` — COLLAR route: given a near-correct seed surface
  ``seed`` (typically the planar straddle from :func:`planar_minimal_surface`),
  solve GF(2) only for the thin collar ``C`` reconciling the seed's rim to
  ``gamma`` (``d2 C = rim(seed) ^ gamma``) and return ``seed ^ C``. The result has
  ``d2 == gamma`` exactly AND hugs the seed's plane (near-minimal). On a planar
  patch whose seed rim already equals ``gamma`` the collar is empty and the seed is
  returned unchanged.

The work is restricted to a local SLAB of faces (cells whose centroid is within a
bbox of ``gamma`` expanded by ``margin``) so the GF(2) elimination stays small;
the fill of a fault patch lives near its plane, so a modest margin suffices.
"""
from __future__ import annotations

import numpy as np

from cpg2unstructured.mesh3d import cell_centroids, face_edges, mesh_edges
from cpg2unstructured.minimal_surface import boundary_edges
from cpg2unstructured.unstructured import UnstructuredGrid

__all__ = [
    "BoundaryDoesNotBound",
    "gf2_solve_boundary",
    "fill_prescribed_boundary",
    "fill_with_seed",
]


class BoundaryDoesNotBound(ValueError):
    """Raised when no cell-face surface has the requested boundary.

    The GF(2) system ``d2 Phi = gamma`` is inconsistent: ``gamma`` is not the
    boundary of any face set within the considered (active-cell) region. Typically
    means ``gamma`` encircles an inactive region / pinch-out, or the local slab is
    too small (raise ``margin``).
    """


def gf2_solve_boundary(face_edge_lists, gamma_rows):
    """Solve ``B @ x = b`` over GF(2); return a particular ``x`` (bool) or ``None``.

    ``face_edge_lists[j]`` = the row ids (edge indices) incident to unknown face
    ``j`` (a column of ``B``). ``gamma_rows`` = the row ids set in ``b``. Returns a
    boolean indicator over the faces whose columns sum (XOR) to ``b``, or ``None``
    if ``b`` is not in the column space (inconsistent).

    Method: Gaussian elimination building a GF(2) basis of the columns, tracking
    each basis vector's PROVENANCE (which original columns produced it) as a
    parallel bitmask. Rows and provenance are bit-packed into Python big-ints for
    fast wide XOR. Reducing ``b`` through the pivots accumulates a provenance whose
    faces have boundary exactly ``b``.
    """
    n = len(face_edge_lists)
    pivots: list[tuple[int, int, int]] = []   # (pivot_row, col_mask, prov_mask)
    for j in range(n):
        m = 0
        for r in face_edge_lists[j]:
            m |= 1 << int(r)
        prov = 1 << j
        for pr, pm, pp in pivots:
            if (m >> pr) & 1:
                m ^= pm
                prov ^= pp
        if m:
            pr = (m & -m).bit_length() - 1     # lowest set bit = pivot row
            pivots.append((pr, m, prov))

    b = 0
    for r in gamma_rows:
        b |= 1 << int(r)
    prov = 0
    for pr, pm, pp in pivots:
        if (b >> pr) & 1:
            b ^= pm
            prov ^= pp
    if b != 0:
        return None                            # inconsistent -> does not bound

    x = np.zeros(n, dtype=bool)
    j = 0
    while prov:
        if prov & 1:
            x[j] = True
        prov >>= 1
        j += 1
    return x


def _slab_faces(grid: UnstructuredGrid, gamma_nodes: np.ndarray, margin: float) -> np.ndarray:
    """Face ids whose adjacent active cell centroid is within gamma's bbox+margin."""
    P = grid.nodes[gamma_nodes]
    lo = P.min(axis=0) - margin
    hi = P.max(axis=0) + margin
    cc = cell_centroids(grid)
    in_box = np.all((cc >= lo) & (cc <= hi), axis=1)
    nb = grid.faces.neighbours
    a, b = nb[:, 0], nb[:, 1]
    sel = np.zeros(nb.shape[0], dtype=bool)
    sel |= (a >= 0) & in_box[np.clip(a, 0, None)]
    sel |= (b >= 0) & in_box[np.clip(b, 0, None)]
    return np.nonzero(sel)[0]


def fill_prescribed_boundary(
    grid: UnstructuredGrid,
    gamma_edge_ids: np.ndarray,
    *,
    margin: float = 2.0,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Face ids of a surface with boundary exactly ``gamma`` (bare GF(2) solve).

    ``gamma_edge_ids`` indexes into the unique mesh edges (:func:`mesh_edges`).
    Restricts to a local slab (see module docstring) and solves ``d2 Phi = gamma``
    over GF(2). Raises :class:`BoundaryDoesNotBound` if ``gamma`` does not bound in
    the slab. The returned surface is exact but not necessarily plane-hugging — use
    :func:`fill_with_seed` for a near-minimal, plane-hugging fill.
    """
    if edges is None:
        edges = mesh_edges(grid)
    FE = face_edges(grid, edges)
    g_edges = np.asarray(gamma_edge_ids, dtype=np.int64)
    if g_edges.size == 0:
        return np.empty(0, dtype=np.int64)

    gamma_nodes = np.unique(edges[g_edges].ravel())
    slab = _slab_faces(grid, gamma_nodes, margin)

    universe = np.unique(FE[slab].ravel())
    erow = {int(e): i for i, e in enumerate(universe)}
    if not set(int(e) for e in g_edges).issubset(erow):
        raise BoundaryDoesNotBound(
            "gamma leaves the local slab; increase margin"
        )
    cols = [[erow[int(e)] for e in FE[f]] for f in slab]
    rhs = [erow[int(e)] for e in g_edges]
    x = gf2_solve_boundary(cols, rhs)
    if x is None:
        raise BoundaryDoesNotBound(
            "gamma does not bound a cell-face surface here "
            "(pinch-out / inactive region, or slab too small — raise margin)"
        )
    return slab[x]


def fill_with_seed(
    grid: UnstructuredGrid,
    gamma_edge_ids: np.ndarray,
    seed_face_ids: np.ndarray,
    *,
    margin: float = 3.0,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Collar fill: ``seed ^ collar`` with boundary exactly ``gamma``.

    The collar ``C`` is solved (GF(2)) to satisfy ``d2 C = rim(seed) ^ gamma``,
    where ``rim(seed)`` is the seed's ``d2`` boundary. Then ``Phi = seed ^ C`` has
    ``d2 Phi = rim(seed) ^ (rim(seed) ^ gamma) = gamma`` exactly, while staying on
    the seed's surface (plane-hugging, near-minimal). If the seed rim already
    equals ``gamma`` the collar is empty and ``seed`` is returned.

    Raises :class:`BoundaryDoesNotBound` if the collar correction does not bound.
    """
    if edges is None:
        edges = mesh_edges(grid)
    seed = np.asarray(seed_face_ids, dtype=np.int64)
    r = set(int(e) for e in boundary_edges(grid, seed))
    g = set(int(e) for e in np.asarray(gamma_edge_ids, dtype=np.int64))
    rhs = np.fromiter(r ^ g, dtype=np.int64)
    if rhs.size == 0:
        return np.unique(seed)
    collar = fill_prescribed_boundary(grid, rhs, margin=margin, edges=edges)
    phi = np.setxor1d(seed, collar, assume_unique=False)
    return phi
