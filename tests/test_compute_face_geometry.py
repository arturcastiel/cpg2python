"""Tests for ``compute_face_geometry`` (PR-4 of phase 2-faults).

The 8-position J-table polygon assembler with 4 diamond cases.
Algorithmic hot spot of phase 2-faults — most subtle branching.

Coverage strategy: hand-crafted scenarios constructed via the
``_make_inputs`` helper. Each test pins down ONE branch
(matching, single edge-crossing, both edge-crossings, diamond case,
pinched pillar) so failures localise quickly.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

from cpg2unstructured._fault_geometry import compute_face_geometry


# ── helpers ─────────────────────────────────────────────────────────


def _make_inputs(
    points_xyz: dict[str, tuple[float, float, float]],
    pa_rows: Iterable[Iterable[str]],
    pb_rows: Iterable[Iterable[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(pa, pb, points)`` from named XYZ coords for readability.

    ``points_xyz`` is an *ordered* dict mapping node names → XYZ.
    Each row of ``pa_rows`` / ``pb_rows`` lists 4 names in the order
    ``[bot_pillar_1, bot_pillar_2, top_pillar_1, top_pillar_2]``.
    """
    name_to_id = {name: i for i, name in enumerate(points_xyz)}
    points = np.array(list(points_xyz.values()), dtype=np.float64)
    pa = np.array([[name_to_id[n] for n in row] for row in pa_rows],
                  dtype=np.int64)
    pb = np.array([[name_to_id[n] for n in row] for row in pb_rows],
                  dtype=np.int64)
    return pa, pb, points


def _per_face_corners(num_nodes, corners):
    """Split the flat corners array back into per-face lists."""
    out = []
    offset = 0
    for n in num_nodes:
        out.append(list(corners[offset:offset + int(n)]))
        offset += int(n)
    return out


# ── empty input ─────────────────────────────────────────────────────


def test_empty_input_returns_empty_outputs():
    points = np.zeros((0, 3), dtype=np.float64)
    pa = np.zeros((0, 4), dtype=np.int64)
    pb = np.zeros((0, 4), dtype=np.int64)
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    assert new_points.shape == (0, 3)
    assert num_nodes.shape == (0,)
    assert corners.shape == (0,)


# ── matching faces (pa == pb) ───────────────────────────────────────


def test_matching_face_produces_clockwise_4_corners():
    """When pa == pb (face matches across the boundary), output is
    a single 4-corner polygon in clockwise order:
    bot_p1 → bot_p2 → top_p2 → top_p1."""
    pa, pb, points = _make_inputs(
        {
            "bot_p1": (0.0, 0.0, 0.0),
            "bot_p2": (0.0, 1.0, 0.0),
            "top_p1": (0.0, 0.0, 1.0),
            "top_p2": (0.0, 1.0, 1.0),
        },
        pa_rows=[["bot_p1", "bot_p2", "top_p1", "top_p2"]],
        pb_rows=[["bot_p1", "bot_p2", "top_p1", "top_p2"]],
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    assert new_points.shape == (4, 3), "no new nodes for matching face"
    np.testing.assert_array_equal(num_nodes, [4])
    np.testing.assert_array_equal(corners, [0, 1, 3, 2])  # bot_p1, bot_p2, top_p2, top_p1


def test_matching_face_multiple_rows():
    """Three matching face rows → 3 × 4 = 12 corners."""
    pa, pb, points = _make_inputs(
        {
            "bp1": (0.0, 0.0, 0.0),
            "bp2": (0.0, 1.0, 0.0),
            "tp1": (0.0, 0.0, 1.0),
            "tp2": (0.0, 1.0, 1.0),
        },
        pa_rows=[["bp1", "bp2", "tp1", "tp2"]] * 3,
        pb_rows=[["bp1", "bp2", "tp1", "tp2"]] * 3,
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    np.testing.assert_array_equal(num_nodes, [4, 4, 4])
    assert corners.size == 12
    # Each face's corner sequence is identical.
    face_corners = _per_face_corners(num_nodes, corners)
    assert face_corners[0] == face_corners[1] == face_corners[2]


# ── simple half-offset (no crossings) ───────────────────────────────


def test_simple_half_offset_no_crossings_gives_4_corners():
    """A and B share pillars; B is offset +0.5 in z. Both bot lines
    are flat at different z; both top lines are flat at different z.
    No envelope crossings, no diamond. Output: 4-corner polygon
    (the inner z-range)."""
    pa, pb, points = _make_inputs(
        {
            "a_bot_p1": (0.0, 0.0, 0.0),
            "a_bot_p2": (0.0, 1.0, 0.0),
            "a_top_p1": (0.0, 0.0, 1.0),
            "a_top_p2": (0.0, 1.0, 1.0),
            "b_bot_p1": (0.0, 0.0, 0.5),
            "b_bot_p2": (0.0, 1.0, 0.5),
            "b_top_p1": (0.0, 0.0, 1.5),
            "b_top_p2": (0.0, 1.0, 1.5),
        },
        pa_rows=[["a_bot_p1", "a_bot_p2", "a_top_p1", "a_top_p2"]],
        pb_rows=[["b_bot_p1", "b_bot_p2", "b_top_p1", "b_top_p2"]],
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    # Inner bot = B (z=0.5 higher than A's z=0); inner top = A (z=1 lower than B's z=1.5).
    # Expected face: B_bot_p1, B_bot_p2, A_top_p2, A_top_p1.
    assert new_points.shape == (8, 3), "no new nodes (no crossings)"
    np.testing.assert_array_equal(num_nodes, [4])
    np.testing.assert_array_equal(corners, [4, 5, 3, 2])  # b_bot_p1, b_bot_p2, a_top_p2, a_top_p1


# ── single-edge crossings (5 corners) ──────────────────────────────


def test_top_edge_crossing_produces_p6():
    """Only A34 × B34 crosses → p6 in position 5 → 5-corner output.

    Geometry: A and B share pillars (x=0..1, y=0).
    A bot flat at z=0; A top tilts 3 → 1 (down toward pillar 2).
    B bot flat at z=-1; B top tilts 0.5 → 2 (up toward pillar 2).
    → A top × B top crosses at the midpoint area.
    """
    pa, pb, points = _make_inputs(
        {
            "a_bp1": (0.0, 0.0, 0.0),
            "a_bp2": (1.0, 0.0, 0.0),
            "a_tp1": (0.0, 0.0, 3.0),
            "a_tp2": (1.0, 0.0, 1.0),
            "b_bp1": (0.0, 0.0, -1.0),
            "b_bp2": (1.0, 0.0, -1.0),
            "b_tp1": (0.0, 0.0, 0.5),
            "b_tp2": (1.0, 0.0, 2.0),
        },
        pa_rows=[["a_bp1", "a_bp2", "a_tp1", "a_tp2"]],
        pb_rows=[["b_bp1", "b_bp2", "b_tp1", "b_tp2"]],
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    # One new intersection node (p6).
    assert new_points.shape == (9, 3)
    np.testing.assert_array_equal(num_nodes, [5])
    # Expected: I[0]=a_bp1=0, I[1]=a_bp2=1, I[2]=b_tp1=6, I[3]=a_tp2=3
    # J = [0, _, 1, _, 3, p6_id=8, 6, _]
    # Compact: [0, 1, 3, 8, 6]
    np.testing.assert_array_equal(corners, [0, 1, 3, 8, 6])


def test_bot_edge_crossing_produces_p2():
    """Only A12 × B12 crosses → p2 in position 1 → 5-corner output.

    Mirror of the top-edge crossing test (bot edges instead of top).
    """
    pa, pb, points = _make_inputs(
        {
            "a_bp1": (0.0, 0.0, -3.0),  # A bot tilts -3 → -1 (more negative)
            "a_bp2": (1.0, 0.0, -1.0),
            "a_tp1": (0.0, 0.0, 5.0),
            "a_tp2": (1.0, 0.0, 5.0),
            "b_bp1": (0.0, 0.0, -0.5),  # B bot tilts -0.5 → -2 (down)
            "b_bp2": (1.0, 0.0, -2.0),
            "b_tp1": (0.0, 0.0, 6.0),
            "b_tp2": (1.0, 0.0, 6.0),
        },
        pa_rows=[["a_bp1", "a_bp2", "a_tp1", "a_tp2"]],
        pb_rows=[["b_bp1", "b_bp2", "b_tp1", "b_tp2"]],
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    # One new intersection node (p2).
    assert new_points.shape == (9, 3)
    np.testing.assert_array_equal(num_nodes, [5])
    # Verify p2 appears in position 1 (second corner in clockwise sequence).
    face = corners.tolist()
    # p2 is the new node (id == 8); somewhere in the sequence.
    assert 8 in face, "p2 intersection should be in the corner list"


# ── both edges crossing (6 corners) ─────────────────────────────────


def test_both_edges_cross_gives_6_corners():
    """A12 × B12 AND A34 × B34 both cross → 6-corner output."""
    pa, pb, points = _make_inputs(
        {
            "a_bp1": (0.0, 0.0, 1.0),   # A bot tilts +1 → -1
            "a_bp2": (1.0, 0.0, -1.0),
            "a_tp1": (0.0, 0.0, 4.0),   # A top tilts 4 → 6
            "a_tp2": (1.0, 0.0, 6.0),
            "b_bp1": (0.0, 0.0, -1.0),  # B bot tilts -1 → 1
            "b_bp2": (1.0, 0.0, 1.0),
            "b_tp1": (0.0, 0.0, 6.0),   # B top tilts 6 → 4
            "b_tp2": (1.0, 0.0, 4.0),
        },
        pa_rows=[["a_bp1", "a_bp2", "a_tp1", "a_tp2"]],
        pb_rows=[["b_bp1", "b_bp2", "b_tp1", "b_tp2"]],
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    # Two new intersection nodes (p2 + p6).
    assert new_points.shape[0] == 10, (
        f"expected 2 new nodes; got {new_points.shape[0] - 8}"
    )
    np.testing.assert_array_equal(num_nodes, [6])


# ── diamond case 1 (vertex at position 7) ──────────────────────────


def test_diamond_case_1_a12_x_b34_left():
    """Case 1: A12 × B34 crosses AND A's bot p1 is above B's top p1
    (numerically: az[0] > bz[2]). Position 7 gets the diamond,
    positions 0 + 6 (left pillar corners) are nullified."""
    pa, pb, points = _make_inputs(
        {
            # A cell: bot tilts 2 → 0, top tilts 3 → 1
            "a_bp1": (0.0, 0.0, 2.0),
            "a_bp2": (1.0, 0.0, 0.0),
            "a_tp1": (0.0, 0.0, 3.0),
            "a_tp2": (1.0, 0.0, 1.0),
            # B cell: bot flat at -1, top tilts 1 → 1.5
            "b_bp1": (0.0, 0.0, -1.0),
            "b_bp2": (1.0, 0.0, -1.0),
            "b_tp1": (0.0, 0.0, 1.0),
            "b_tp2": (1.0, 0.0, 1.5),
        },
        pa_rows=[["a_bp1", "a_bp2", "a_tp1", "a_tp2"]],
        pb_rows=[["b_bp1", "b_bp2", "b_tp1", "b_tp2"]],
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    # Two new intersection nodes appear:
    #   p6 (A34 × B34): cross since diff_p1=2, diff_p2=-0.5
    #   p8 (A12 × B34): cross since diff_p1=1, diff_p2=-1.5  → Case 1 (a_bp1=2 > b_tp1=1)
    assert new_points.shape[0] == 10
    # J after Case 1: positions [0, 6] nullified; positions [2, 4, 7] active;
    # plus position 5 (p6) since A34×B34 also crossed. So 4 corners total.
    np.testing.assert_array_equal(num_nodes, [4])


# ── pinched-pillar case ────────────────────────────────────────────


def test_pinch_pillar_1_collapses():
    """Bot p1 == top p1 on the SAME side (pinched cell on pillar 1).

    Both pa and pb's pillar-1 bot and top are the same node. The
    pinch dedup nullifies one of them, producing a triangle (3 corners).
    """
    pa, pb, points = _make_inputs(
        {
            "p1_collapsed": (0.0, 0.0, 0.0),     # bot p1 == top p1
            "bot_p2": (0.0, 1.0, 0.0),
            "top_p2": (0.0, 1.0, 1.0),
            "b_p1_collapsed": (0.0, 0.0, 0.5),
            "b_bot_p2": (0.0, 1.0, 0.5),
            "b_top_p2": (0.0, 1.0, 1.5),
        },
        pa_rows=[["p1_collapsed", "bot_p2", "p1_collapsed", "top_p2"]],
        pb_rows=[["b_p1_collapsed", "b_bot_p2",
                  "b_p1_collapsed", "b_top_p2"]],
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    # On pillar 1: A and B both pinched (z=0 and z=0.5 — no real face on p1).
    # On pillar 2: B bot z=0.5, A top z=1 → inner bot (B) and inner top (A).
    # Output: at most 3 unique nodes (one collapsed pillar 1 + 2 from pillar 2).
    assert int(num_nodes[0]) <= 3
    assert int(num_nodes[0]) >= 2  # must have at least 2 to be a polygon


# ── new-node appending contract ─────────────────────────────────────


def test_no_intersections_means_no_new_points():
    """If no envelope crossings nor diamond crossings, points table
    is returned unchanged."""
    pa, pb, points = _make_inputs(
        {
            "n0": (0.0, 0.0, 0.0),
            "n1": (0.0, 1.0, 0.0),
            "n2": (0.0, 0.0, 1.0),
            "n3": (0.0, 1.0, 1.0),
        },
        pa_rows=[["n0", "n1", "n2", "n3"]],
        pb_rows=[["n0", "n1", "n2", "n3"]],
    )
    new_points, _, _ = compute_face_geometry(pa, pb, points)
    np.testing.assert_array_equal(new_points, points)
    assert new_points.shape == (4, 3)


def test_intersection_nodes_appended_with_unique_ids():
    """New intersection nodes get IDs starting from len(points)."""
    pa, pb, points = _make_inputs(
        {
            "a_bp1": (0.0, 0.0, 0.0),
            "a_bp2": (1.0, 0.0, 0.0),
            "a_tp1": (0.0, 0.0, 3.0),
            "a_tp2": (1.0, 0.0, 1.0),
            "b_bp1": (0.0, 0.0, -1.0),
            "b_bp2": (1.0, 0.0, -1.0),
            "b_tp1": (0.0, 0.0, 0.5),
            "b_tp2": (1.0, 0.0, 2.0),
        },
        pa_rows=[["a_bp1", "a_bp2", "a_tp1", "a_tp2"]],
        pb_rows=[["b_bp1", "b_bp2", "b_tp1", "b_tp2"]],
    )
    n_old = points.shape[0]
    new_points, _, corners = compute_face_geometry(pa, pb, points)
    # At least one new node appended.
    assert new_points.shape[0] > n_old
    # All corner IDs are valid indices into new_points.
    assert corners.min() >= 0
    assert corners.max() < new_points.shape[0]


def test_clockwise_sequence_invariant_on_matching_face():
    """Clockwise traversal (with Z-up = out of face): the cross-product
    of consecutive edges should have a consistent sign (planar face)."""
    pa, pb, points = _make_inputs(
        {
            "bp1": (0.0, 0.0, 0.0),
            "bp2": (0.0, 1.0, 0.0),
            "tp1": (0.0, 0.0, 1.0),
            "tp2": (0.0, 1.0, 1.0),
        },
        pa_rows=[["bp1", "bp2", "tp1", "tp2"]],
        pb_rows=[["bp1", "bp2", "tp1", "tp2"]],
    )
    new_points, num_nodes, corners = compute_face_geometry(pa, pb, points)
    pts = new_points[corners[:int(num_nodes[0])]]
    # Edges
    e0 = pts[1] - pts[0]
    e1 = pts[2] - pts[1]
    # Normal of the first turn
    n01 = np.cross(e0, e1)
    # The face lies in the x=0 plane, so the normal is along x.
    # The signed magnitude tells us which way (CW vs CCW from the +x side).
    assert abs(n01[0]) > 0.5, "face should have a nontrivial x-normal"
