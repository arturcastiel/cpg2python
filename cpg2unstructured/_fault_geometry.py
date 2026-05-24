# Derived from MRST's processGRDECL.m computeFaceGeometry / intersection
# (corner-point grid processing), (C) SINTEF Digital, GNU GPL v3.
# See README + NOTICE.
"""Fault-face geometry helpers (PR-3+ of phase 2-faults).

Pure geometric primitives:

- :func:`line_intersection` (PR-3) — XYZ intersection of two
  line-segment families, z-parametrised.
- :func:`compute_face_geometry` (PR-4) — full 8-position J-table
  polygon assembler for fault faces, including the 4 diamond cases.

NOT public API — ``cpg2unstructured.unstructured`` consumes these
internally when integration lands in PR-5+ of phase 2-faults.
"""
from __future__ import annotations

import numpy as np

__all__ = ["line_intersection", "compute_face_geometry"]

# Sentinel value used in the J-table to mark "this position is empty".
# Real node IDs are non-negative int64; we use -1 as the NaN-equivalent.
_EMPTY = np.int64(-1)

# Threshold for "lines have parallel z-trajectories" — denominator
# below this is treated as numerically singular. Real CPG pillars
# deviate <10% from vertical so Δz_A − Δz_B is typically of order
# the cell thickness (≥1e-6 in any non-degenerate deck).
_DENOM_EPS = 1e-12


def line_intersection(la_start: np.ndarray,
                      la_end: np.ndarray,
                      lb_start: np.ndarray,
                      lb_end: np.ndarray) -> np.ndarray:
    """XYZ intersection of two line-segment families, z-parametrised.

    Each line is defined by its start + end XYZ on the same pillar
    pair (so both lines go from pillar-1-XYZ to pillar-2-XYZ; their
    (x, y) trajectory is shared in the vertical-pillar limit; only
    the z values at the endpoints differ).

    Parameterising by ``t`` along line A (pillar 1 → pillar 2):

    .. math::
        z_A(t) = z_A^{p1} + t \\, (z_A^{p2} - z_A^{p1}) \\\\
        z_B(t) = z_B^{p1} + t \\, (z_B^{p2} - z_B^{p1}) \\\\
        z_A(t^*) = z_B(t^*) \\;\\Rightarrow\\;
            t^* = \\frac{z_B^{p1} - z_A^{p1}}{\\Delta z_A - \\Delta z_B}

    The intersection XYZ is then computed by interpolating along
    line A at ``t = t*``. This matches MRST's ``intersection``
    helper (``processGRDECL.m`` lines 1164-1190).

    Parameters
    ----------
    la_start, la_end : array-like, shape ``(..., 3)``
        Start + end XYZ of A-line(s).
    lb_start, lb_end : array-like, shape ``(..., 3)``
        Start + end XYZ of B-line(s).

    Returns
    -------
    pts : ndarray, shape ``(..., 3)``
        Intersection XYZ per line pair. The z column is the exact
        intersection z (not an interpolation — avoids round-off
        drift on the parametric reconstruction).

    Raises
    ------
    ValueError
        If any line pair has ``|Δz_A - Δz_B| < _DENOM_EPS`` —
        the two z-trajectories are parallel and no unique
        intersection exists by this parametrisation. (R-3 from
        the phase 2-faults study.)
    """
    la_start = np.asarray(la_start, dtype=np.float64)
    la_end   = np.asarray(la_end,   dtype=np.float64)
    lb_start = np.asarray(lb_start, dtype=np.float64)
    lb_end   = np.asarray(lb_end,   dtype=np.float64)

    za_p1 = la_start[..., 2]
    za_p2 = la_end[..., 2]
    zb_p1 = lb_start[..., 2]
    zb_p2 = lb_end[..., 2]

    delta_za = za_p2 - za_p1
    delta_zb = zb_p2 - zb_p1
    denom = delta_za - delta_zb

    # R-3 guard — parallel z-trajectories are numerically singular.
    # Use abs().min() over any leading axes so we surface the WORST
    # case rather than only the last.
    if np.abs(denom).min() < _DENOM_EPS:
        raise ValueError(
            "line_intersection: a line pair has parallel z-trajectories "
            "(|Δz_A − Δz_B| below numerical threshold); no unique "
            "intersection by z-parametrisation."
        )

    t = (zb_p1 - za_p1) / denom               # scalar or (n,)
    delta_xyz = la_end - la_start
    out = la_start + t[..., None] * delta_xyz
    # Exact z (avoid drift on the parametric form)
    out[..., 2] = za_p1 + t * delta_za
    return out


def compute_face_geometry(
    pa: np.ndarray,
    pb: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 3D polygons of fault faces from corner-id arrays.

    For each connection (row), constructs a clockwise polygon whose
    normal points from cell-A to cell-B. The polygon has between 4
    and 8 corners. New intersection nodes are appended to ``points``
    and assigned fresh IDs.

    Parameters
    ----------
    pa : (n, 4) int64
        Node IDs of A-face corners per connection. Column layout:

        - ``pa[:, 0]`` — A-face bottom corner on pillar 1
        - ``pa[:, 1]`` — A-face bottom corner on pillar 2
        - ``pa[:, 2]`` — A-face top corner on pillar 1
        - ``pa[:, 3]`` — A-face top corner on pillar 2

        ("Bottom" = larger z, "top" = smaller z under Eclipse Z-down
        convention; but the algorithm is sign-agnostic.)
    pb : (n, 4) int64
        Same for B-face corners.
    points : (n_points, 3) float64
        Global node coordinate table.

    Returns
    -------
    new_points : (n_points + n_new, 3) float64
        ``points`` with intersection nodes appended (does NOT modify
        the input array in place — uses ``np.concatenate``).
    num_nodes : (n,) int64
        Corner count per output face, between 4 and 8.
    corners : (sum(num_nodes),) int64
        Flat list of node IDs in clockwise order, concatenated
        across all faces. Reconstruct per-face slices with
        ``cumsum(num_nodes)``.

    Notes
    -----
    Direct port of MRST's ``computeFaceGeometry`` (``processGRDECL.m``
    lines 853-1160). See the README "Attribution & provenance" section.

    The 8-position J-table layout (clockwise, normal A→B):

    .. code-block::

       position 0  bot p1 (pillar 1, bottom row)
       position 1  p2  =  A12 × B12  (bot-edge crossing)
       position 2  bot p2 (pillar 2, bottom row)
       position 3  p4  =  A12 × B34 or A34 × B12 (right diamond)
       position 4  top p2 (pillar 2, top row)
       position 5  p6  =  A34 × B34  (top-edge crossing)
       position 6  top p1 (pillar 1, top row)
       position 7  p8  =  A12 × B34 or A34 × B12 (left diamond)

    Empty positions (no intersection / no pillar point) are marked
    ``_EMPTY`` (= -1) and stripped during compaction.
    """
    pa = np.asarray(pa, dtype=np.int64)
    pb = np.asarray(pb, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    n = pa.shape[0]
    if n == 0:
        return (
            points,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    z = points[:, 2]
    az = z[pa]                                  # (n, 4)
    bz = z[pb]                                  # (n, 4)

    # ── Step 1: pillar points (positions 0, 2, 4, 6 in J) ───────────
    # For each of the 4 pillar corners, pick the INNER one. At bot
    # corners (cols 0, 1) the inner corner is the one with LARGER z;
    # at top corners (cols 2, 3) it's the one with SMALLER z.
    pick_pb = np.concatenate(
        [
            az[:, 0:2] < bz[:, 0:2],   # bot: pick pb if pb has larger z
            az[:, 2:4] > bz[:, 2:4],   # top: pick pb if pb has smaller z
        ],
        axis=1,
    )                                            # (n, 4)
    I = pa.copy()
    I[pick_pb] = pb[pick_pb]

    # Pinch dedup on the pillar points: if bot and top on the SAME
    # pillar collapsed to the same node (zero-height pinch on that
    # pillar), drop the bot corner.
    I[I[:, 0] == I[:, 2], 0] = _EMPTY            # pillar 1 pinch
    I[I[:, 1] == I[:, 3], 1] = _EMPTY            # pillar 2 pinch

    # Place pillar points into J in clockwise order.
    # MATLAB: J(:, [1,3,5,7]) = I(:, [1,2,4,3])
    # Python: J(:, [0,2,4,6]) = I(:, [0,1,3,2])
    J = np.full((n, 8), _EMPTY, dtype=np.int64)
    J[:, 0] = I[:, 0]                            # bot p1
    J[:, 2] = I[:, 1]                            # bot p2
    J[:, 4] = I[:, 3]                            # top p2 (swap with top p1)
    J[:, 6] = I[:, 2]                            # top p1

    # ── Step 2: intersection candidates (4 types per row) ───────────
    # PP: A-line endpoint pairs (4 stacks of (n, 2)).
    # QQ: B-line endpoint pairs (4 stacks of (n, 2)).
    # Stack order:
    #   type 0  A12 × B12  (bot × bot)         → p2
    #   type 1  A34 × B34  (top × top)         → p6
    #   type 2  A12 × B34  (bot × top diamond) → p4 or p8
    #   type 3  A34 × B12  (top × bot diamond) → p4 or p8
    PP = np.concatenate([pa[:, 0:2], pa[:, 2:4], pa[:, 0:2], pa[:, 2:4]],
                        axis=0)                 # (4n, 2)
    QQ = np.concatenate([pb[:, 0:2], pb[:, 2:4], pb[:, 2:4], pb[:, 0:2]],
                        axis=0)                 # (4n, 2)

    # Filter: only line pairs whose z-coords on the two pillars
    # straddle each other (product of differences < 0 → they CROSS).
    cross = (z[PP[:, 0]] - z[QQ[:, 0]]) * (z[PP[:, 1]] - z[QQ[:, 1]]) < 0

    # ── Step 3: compute the surviving intersections ─────────────────
    f = np.full((n, 4), _EMPTY, dtype=np.int64)  # (n, 4) per-row, per-type
    if cross.any():
        la_start = points[PP[cross, 0]]
        la_end   = points[PP[cross, 1]]
        lb_start = points[QQ[cross, 0]]
        lb_end   = points[QQ[cross, 1]]
        Q = line_intersection(la_start, la_end, lb_start, lb_end)

        # Dedup (rows-of-XYZ may repeat across the 4 types when faces
        # share an intersection point).
        Q_unique, inverse = np.unique(Q, axis=0, return_inverse=True)

        # Assign fresh node IDs starting from the current point count.
        new_ids_per_crossing = points.shape[0] + np.asarray(inverse,
                                                            dtype=np.int64)
        points = np.concatenate([points, Q_unique], axis=0)

        # Slot the new IDs into f at (row, type) positions where cross
        # was True. PP/QQ were stacked type-major: type k spans rows
        # k*n .. (k+1)*n - 1, with row index within the type giving
        # the connection index.
        true_indices = np.where(cross)[0]
        type_indices = true_indices // n
        row_indices = true_indices % n
        f[row_indices, type_indices] = new_ids_per_crossing

    # ── Step 4: place straight intersections (p2, p6) ───────────────
    J[:, 1] = f[:, 0]                            # p2 = A12 × B12
    J[:, 5] = f[:, 1]                            # p6 = A34 × B34

    # ── Step 5: diamond cases (p4, p8) ──────────────────────────────
    intersect_botA_topB = f[:, 2] != _EMPTY      # A12 × B34 exists
    intersect_topA_botB = f[:, 3] != _EMPTY      # A34 × B12 exists

    # Case 1 / Case 2 predicate: A12 above B34 on pillar 1
    intersect_left_botA = az[:, 0] > bz[:, 2]

    # Case 1: A12 × B34 + left → vertex@7, NaN positions 0 + 6
    case1 = intersect_botA_topB & intersect_left_botA
    J[case1, 0] = _EMPTY
    J[case1, 6] = _EMPTY
    J[case1, 7] = f[case1, 2]

    # Case 2: A12 × B34 + NOT left → vertex@3, NaN positions 2 + 4
    case2 = intersect_botA_topB & ~intersect_left_botA
    J[case2, 2] = _EMPTY
    J[case2, 4] = _EMPTY
    J[case2, 3] = f[case2, 2]

    # Case 3 / Case 4 predicate: B12 above A34 on pillar 1
    intersect_left_topA = bz[:, 0] > az[:, 2]

    # Case 4: A34 × B12 + left → vertex@7, NaN positions 0 + 6
    case4 = intersect_topA_botB & intersect_left_topA
    J[case4, 0] = _EMPTY
    J[case4, 6] = _EMPTY
    J[case4, 7] = f[case4, 3]

    # Case 3: A34 × B12 + NOT left → vertex@3, NaN positions 2 + 4
    case3 = intersect_topA_botB & ~intersect_left_topA
    J[case3, 2] = _EMPTY
    J[case3, 4] = _EMPTY
    J[case3, 3] = f[case3, 3]

    # ── Step 6: compact J → per-face flat corner list ───────────────
    # For each row of J, strip _EMPTY sentinels and collapse consecutive
    # duplicates that survived (pinch artefacts).
    num_nodes = np.zeros(n, dtype=np.int64)
    corners_chunks: list[np.ndarray] = []
    for i, row in enumerate(J):
        valid = row[row != _EMPTY]
        if valid.size > 0:
            keep = np.concatenate(
                [np.array([True]), valid[1:] != valid[:-1]],
            )
            valid = valid[keep]
        num_nodes[i] = valid.size
        corners_chunks.append(valid)

    corners = (
        np.concatenate(corners_chunks)
        if corners_chunks
        else np.zeros(0, dtype=np.int64)
    )
    return points, num_nodes, corners
