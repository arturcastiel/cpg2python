# Derived from MRST's processGRDECL.m findConnections (corner-point grid
# processing), (C) SINTEF Digital, GNU GPL v3. See README + NOTICE.
"""Fault-stack connection sweeping.

Pure 1D segment-overlap algorithm. No geometry, no node coordinates.
Given the z-values along a faulted pillar pair (one for each side of
the fault), pair up cells whose vertical extents overlap.

Direct port of MRST's ``findConnections`` / ``doIntersect`` /
``overlap`` from ``processGRDECL.m``. See the README
"Attribution & provenance" section for the algorithmic provenance.

NOT public API — ``cpg2unstructured.unstructured`` consumes these
internally by the topology builder.
"""
from __future__ import annotations

import numpy as np

__all__ = ["overlap", "do_intersect", "find_connections"]


def overlap(xa1: float, xa2: float, xb1: float, xb2: float) -> bool:
    """Do intervals ``[xa1, xa2]`` and ``[xb1, xb2]`` overlap with
    positive measure?

    Strict (``<``, not ``<=``): touching but zero-width does NOT
    overlap. Port of MRST's ``overlap`` helper
    (``processGRDECL.m`` lines 1285-1292).
    """
    return max(xa1, xb1) < min(xa2, xb2)


def do_intersect(za_top: np.ndarray, za_bot: np.ndarray,
                 zb_top: np.ndarray, zb_bot: np.ndarray) -> bool:
    """Do A-face and B-face overlap with positive measure?

    Parameters
    ----------
    za_top, za_bot
        Length-2 z values at top + bot of the A-face. Index 0 = pillar 1,
        index 1 = pillar 2.
    zb_top, zb_bot
        Same for the B-face.

    Returns
    -------
    True if the two faces share positive area, False otherwise.
    Pinched cases (``za_top == za_bot`` or ``zb_top == zb_bot`` along
    both pillars) are explicitly rejected — MRST's pinch guards.

    Port of MRST's ``doIntersect`` (``processGRDECL.m`` lines 1251-1281).

    Three positive-overlap cases:

    1. Overlap on pillar 1: ``[za_top[0], za_bot[0]]`` and
       ``[zb_top[0], zb_bot[0]]`` share a z-range.
    2. Overlap on pillar 2 (symmetric).
    3. "Le fix speciale" — one face's top edge crosses the other's
       top edge (or same for bottom). Catches the diamond case where
       two faces have no single-pillar overlap but DO overlap because
       their lines cross.
    """
    za_top = np.asarray(za_top)
    za_bot = np.asarray(za_bot)
    zb_top = np.asarray(zb_top)
    zb_bot = np.asarray(zb_bot)

    # Pinch guards — MRST lines 1279-1280
    if (za_top == za_bot).all():
        return False
    if (zb_top == zb_bot).all():
        return False

    # Case 1: overlap on pillar 1
    if overlap(float(za_top[0]), float(za_bot[0]),
               float(zb_top[0]), float(zb_bot[0])):
        return True
    # Case 2: overlap on pillar 2
    if overlap(float(za_top[1]), float(za_bot[1]),
               float(zb_top[1]), float(zb_bot[1])):
        return True
    # Case 3a "le fix speciale": A-top crosses B-top
    if (za_top[0] - zb_top[0]) * (za_top[1] - zb_top[1]) < 0:
        return True
    # Case 3b "le fix speciale": A-bot crosses B-bot
    if (za_bot[0] - zb_bot[0]) * (za_bot[1] - zb_bot[1]) < 0:
        return True
    return False


def find_connections(za: np.ndarray, zb: np.ndarray) -> np.ndarray:
    """Pair up overlapping A-cells and B-cells along a faulted pillar pair.

    Parameters
    ----------
    za : (n_a + 1, 2) float64
        Z-coordinates along the A-side cell stack. Row ``i`` is the
        TOP of A-cell ``i`` and the BOT of A-cell ``i-1`` (shared
        corner row). Columns = (pillar 1, pillar 2).
    zb : (n_b + 1, 2) float64
        Same for the B-side stack.

    Returns
    -------
    C : (n_connections, 2) int64
        Each row ``(ia, ib)`` is a connection between A-cell ``ia``
        and B-cell ``ib``. Empty ``(0, 2)`` array if no overlaps.

    Notes
    -----
    Port of MRST's ``findConnections`` (``processGRDECL.m``
    lines 1194-1247). Walks A-cells outer, B-cells inner, with
    per-pillar ``j1`` / ``j2`` checkpoints to avoid re-scanning the
    full B-list on each new A-cell. Worst case O((n_a + n_b))
    when stacks are roughly aligned in z.
    """
    za = np.asarray(za, dtype=np.float64)
    zb = np.asarray(zb, dtype=np.float64)
    n_a = za.shape[0] - 1
    n_b = zb.shape[0] - 1
    if n_a <= 0 or n_b <= 0:
        return np.empty((0, 2), dtype=np.int64)

    out: list[tuple[int, int]] = []
    j1 = 0
    j2 = 0
    for i in range(n_a):
        j = min(j1, j2)
        # Walk B-cells while at least one pillar has zb[j] above za[i+1]
        # (i.e. the top of B-cell j hasn't yet passed below the bot of
        # A-cell i on at least one pillar). Bound: j < n_b so zb[j+1]
        # stays in range.
        while j < n_b and (zb[j, 0] < za[i + 1, 0]
                           or zb[j, 1] < za[i + 1, 1]):
            if do_intersect(za[i], za[i + 1], zb[j], zb[j + 1]):
                out.append((i, j))
            # Per-pillar checkpoint update — the next i iteration
            # starts at min(j1, j2) to avoid re-scanning.
            if zb[j, 0] < za[i + 1, 0]:
                j1 = j
            if zb[j, 1] < za[i + 1, 1]:
                j2 = j
            j += 1

    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)
