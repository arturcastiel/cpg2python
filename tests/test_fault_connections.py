"""Tests for the 1D fault-stack connection sweeper (PR-2 of phase 2-faults).

Pure unit tests with hand-crafted z-list inputs — no GRDECL parsing,
no geometry. Covers all three branches of ``do_intersect`` (per-pillar
overlap and ``le fix speciale`` diamond crossing), pinch guards, and
a hand-worked overlap example.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured._fault_connections import (
    do_intersect,
    find_connections,
    overlap,
)


# ── overlap predicate ───────────────────────────────────────────────


def test_overlap_basic():
    assert overlap(0.0, 1.0, 0.5, 1.5) is True


def test_overlap_disjoint():
    assert overlap(0.0, 1.0, 2.0, 3.0) is False


def test_overlap_touching_is_false():
    """Strict overlap: zero-width touching at a single z is NOT overlap."""
    assert overlap(0.0, 1.0, 1.0, 2.0) is False


def test_overlap_containment():
    assert overlap(0.0, 5.0, 2.0, 3.0) is True


def test_overlap_reversed_args_symmetric():
    assert overlap(2.0, 3.0, 0.0, 5.0) is True


# ── do_intersect predicate ──────────────────────────────────────────


def test_do_intersect_simple_overlap_both_pillars():
    za_top, za_bot = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    zb_top, zb_bot = np.array([0.5, 0.5]), np.array([1.5, 1.5])
    assert do_intersect(za_top, za_bot, zb_top, zb_bot) is True


def test_do_intersect_pinched_a_returns_false():
    """If A-face has zero thickness on both pillars, no overlap counts."""
    za_top = np.array([0.5, 0.5])
    za_bot = np.array([0.5, 0.5])
    zb_top = np.array([0.0, 0.0])
    zb_bot = np.array([1.0, 1.0])
    assert do_intersect(za_top, za_bot, zb_top, zb_bot) is False


def test_do_intersect_pinched_b_returns_false():
    za_top = np.array([0.0, 0.0])
    za_bot = np.array([1.0, 1.0])
    zb_top = np.array([0.5, 0.5])
    zb_bot = np.array([0.5, 0.5])
    assert do_intersect(za_top, za_bot, zb_top, zb_bot) is False


def test_do_intersect_disjoint_above():
    """B entirely above A → no overlap."""
    za_top, za_bot = np.array([2.0, 2.0]), np.array([3.0, 3.0])
    zb_top, zb_bot = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    assert do_intersect(za_top, za_bot, zb_top, zb_bot) is False


def test_do_intersect_diamond_top_crosses():
    """Diamond case — A's top crosses B's top, but no single-pillar overlap.

    A's top is sloped from z=0 on pillar 1 to z=2 on pillar 2.
    B's top is sloped from z=2 on pillar 1 to z=0 on pillar 2.
    They cross between the pillars → "le fix speciale" triggers.

    Note: this test is also expected to overlap on one of the pillars
    via the bottom edges, but the key signal is the diamond.
    """
    za_top = np.array([0.0, 2.0])
    za_bot = np.array([3.0, 3.0])  # A-face goes from sloped top to flat bot
    zb_top = np.array([2.0, 0.0])  # B-top crosses A-top
    zb_bot = np.array([3.0, 3.0])
    assert do_intersect(za_top, za_bot, zb_top, zb_bot) is True


def test_do_intersect_overlap_only_pillar_1():
    """Overlap on pillar 1 only — pillar 2 is disjoint."""
    za_top = np.array([0.0, 5.0])
    za_bot = np.array([1.0, 6.0])
    zb_top = np.array([0.5, 0.0])  # pillar-2 of B is in different range
    zb_bot = np.array([1.5, 0.5])
    assert do_intersect(za_top, za_bot, zb_top, zb_bot) is True


# ── find_connections — worked example from study §6b ────────────────


def test_find_connections_worked_example_half_offset_2x2():
    """The §6b prediction: 2 A-cells, 2 B-cells offset by half →
    3 connections (0,0), (1,0), (1,1)."""
    za = np.array([[0.0, 0.0],
                   [1.0, 1.0],
                   [2.0, 2.0]])
    zb = np.array([[0.5, 0.5],
                   [1.5, 1.5],
                   [2.5, 2.5]])
    C = find_connections(za, zb)
    assert C.shape == (3, 2)
    assert C.dtype == np.int64
    np.testing.assert_array_equal(C, np.array([[0, 0], [1, 0], [1, 1]]))


# ── find_connections — basic cases ──────────────────────────────────


def test_find_connections_matching_stacks():
    """za == zb → each A-cell connects to exactly its corresponding B-cell."""
    z = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    C = find_connections(z, z)
    np.testing.assert_array_equal(C, np.array([[0, 0], [1, 1]]))


def test_find_connections_empty_when_fully_disjoint():
    """B entirely below A's lowest cell → no connections."""
    za = np.array([[0.0, 0.0], [1.0, 1.0]])      # A-cell at z ∈ [0, 1]
    zb = np.array([[10.0, 10.0], [11.0, 11.0]])  # B-cell at z ∈ [10, 11]
    C = find_connections(za, zb)
    assert C.shape == (0, 2)
    assert C.dtype == np.int64


def test_find_connections_single_cell_each_side():
    """Single A-cell, single B-cell, simple overlap → one connection."""
    za = np.array([[0.0, 0.0], [1.0, 1.0]])
    zb = np.array([[0.5, 0.5], [1.5, 1.5]])
    C = find_connections(za, zb)
    np.testing.assert_array_equal(C, np.array([[0, 0]]))


def test_find_connections_asymmetric_stack_sizes():
    """A has 3 cells, B has 2 — connections distribute correctly."""
    za = np.array([[0.0, 0.0],
                   [1.0, 1.0],
                   [2.0, 2.0],
                   [3.0, 3.0]])               # 3 A-cells
    zb = np.array([[0.5, 0.5],
                   [2.0, 2.0],
                   [3.5, 3.5]])               # 2 B-cells, larger spans
    C = find_connections(za, zb)
    # A-cell 0 [0, 1] overlaps B-cell 0 [0.5, 2.0] → (0, 0)
    # A-cell 1 [1, 2] overlaps B-cell 0 [0.5, 2.0] → (1, 0)
    # A-cell 2 [2, 3] overlaps B-cell 1 [2.0, 3.5] → (2, 1)
    np.testing.assert_array_equal(C, np.array([[0, 0], [1, 0], [2, 1]]))


def test_find_connections_empty_stack():
    """No A-cells (za has length 1) → no connections regardless of B."""
    za = np.array([[0.0, 0.0]])
    zb = np.array([[0.5, 0.5], [1.5, 1.5]])
    C = find_connections(za, zb)
    assert C.shape == (0, 2)


def test_find_connections_pinched_cells_skipped():
    """Pinched A-cell (zero thickness on both pillars) shouldn't connect."""
    za = np.array([[0.0, 0.0],
                   [0.0, 0.0],   # pinched A-cell at z=0
                   [1.0, 1.0]])
    zb = np.array([[0.0, 0.0],
                   [1.0, 1.0]])
    C = find_connections(za, zb)
    # The pinched A-cell 0 has za[0] == za[1] both → do_intersect False.
    # A-cell 1 spans [0, 1] → overlaps B-cell 0 [0, 1].
    np.testing.assert_array_equal(C, np.array([[1, 0]]))


def test_find_connections_overlap_chain_3a_2b():
    """3 A-cells, 2 B-cells where A's middle cell straddles both B-cells.

    A: [0,1], [1,2], [2,3]
    B: [0.5, 2.5], [2.5, 3.5]
    """
    za = np.array([[0.0, 0.0],
                   [1.0, 1.0],
                   [2.0, 2.0],
                   [3.0, 3.0]])
    zb = np.array([[0.5, 0.5],
                   [2.5, 2.5],
                   [3.5, 3.5]])
    C = find_connections(za, zb)
    # A0 [0,1]   ∩ B0 [0.5, 2.5] → (0, 0)
    # A1 [1,2]   ∩ B0 [0.5, 2.5] → (1, 0)
    # A2 [2,3]   ∩ B0 [0.5, 2.5] → (2, 0)
    # A2 [2,3]   ∩ B1 [2.5, 3.5] → (2, 1)
    np.testing.assert_array_equal(
        C, np.array([[0, 0], [1, 0], [2, 0], [2, 1]]),
    )


def test_find_connections_returns_int64():
    """Type contract: output dtype is int64 even on the empty case."""
    za = np.array([[0.0, 0.0], [1.0, 1.0]])
    zb = np.array([[2.0, 2.0], [3.0, 3.0]])
    C = find_connections(za, zb)
    assert C.dtype == np.int64
